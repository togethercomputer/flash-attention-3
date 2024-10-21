from __future__ import annotations

import torch

import rotary_emb

import flash_attn_2_cuda as flash_attn_cuda

from math import sqrt
from typing import List, Tuple

from loguru import logger

from torch import nn
from torch.nn.functional import layer_norm

from text_generation_server.core.envs import DISABLE_FA3, KV_SPLITS
from text_generation_server.core.kernel import BaseKernel
from text_generation_server.core.layers.base import (
    T,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)
from text_generation_server.core.layers.config import QuantStrategy
from text_generation_server.core.weights import Weights
from text_generation_server.distributed import RANK
from text_generation_server.guard import import_guard


AttentionLogSumExp = Tuple[torch.Tensor, torch.Tensor]


FLOAT8: str = 'fp8'
HOPPER: bool = torch.cuda.get_device_capability() >= (9, 0)


with import_guard('flashattn_hopper_cuda', HOPPER) as HAS_FA3:

    import flashattn_hopper_cuda as flash_attn_hopper_cuda  # noqa


def resolve_prefixes(prefix: str, name: str, packed: bool) -> List[str] | str:

    if packed:

        return f"{prefix}.{name}"

    else:

        return list(map(f"{prefix}.{name}".format, 'qkv'))


def resolve_splits(query_heads: int, key_value_heads: int) -> int | List[int]:

    if query_heads != key_value_heads:

        kv_groups, not_divisible = divmod(query_heads, key_value_heads)

        if not_divisible:

            raise ValueError(
                f"total query attention heads of {query_heads} is not compatible with "
                f"total key-value heads of {key_value_heads}"
            )

        return [kv_groups, 1, 1]

    else:

        return 3


def load_qkv(
    prefix: str,
    name: str,
    config: T,
    weights: Weights,
    packed: bool,
    bias: bool,
    groups: int | None,
) -> TensorParallelColumnLinear:
    """
    Loads the weights for the attention query, key, and value components, which could either be
    equi-sized for MHA or with fewer heads for KV for GQA as in llama-70b models.

    """
    prefixes = resolve_prefixes(prefix, name, packed)
    kwargs = dict(config=config, weights=weights, bias=bias, dim=0)

    if packed:

        splits = resolve_splits(config.num_attention_heads, config.num_key_value_heads)

        return TensorParallelColumnLinear.load_packed(
            prefix=prefixes, splits=splits, groups=groups, **kwargs
        )

    elif groups is not None:

        raise ValueError(
            'received non-null value for groups when loading non-packed qkv weights'
        )

    else:

        return TensorParallelColumnLinear.load_multi(prefixes=prefixes, **kwargs)


def chunked_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

    x1, x2 = x.chunk(2, dim=-1)

    rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)

    return x


def interleaved_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)

    return x


class FlashPrefill(BaseKernel):

    @classmethod
    def initialize(cls, name: str, target: str, hopper: bool = False) -> BaseKernel:

        cls.forward = cls.forward_hopper if hopper else cls.forward_legacy

        return cls(name, target, cls.schematize())

    def meta(
        self,
        # The QKV components.
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        # The cumulative Q and K sequence lengths.
        cu_seqlen_query: torch.Tensor, cu_seqlen_key: torch.Tensor,
        # The maximum Q and K length upper bounds.
        max_seqlen_q: int, max_seqlen_k: int,
        # The softmax scale in attention.
        softmax_scale: float,
        # Causal self-attention.
        causal: bool,
        # Non-sliding window attention.
        left_window: int, right_window: int,
        # Attention logit soft-capping.
        softcap: float,
    ) -> AttentionLogSumExp:

        n_queries, n_heads, _ = query.shape

        softmax_lse = query.new_empty((n_heads, n_queries))

        return torch.empty_like(query), softmax_lse

    def forward_legacy(
        self,
        # The QKV components.
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        # The cumulative Q and K sequence lengths.
        cu_seqlen_query: torch.Tensor, cu_seqlen_key: torch.Tensor,
        # The maximum Q and K length upper bounds.
        max_seqlen_q: int, max_seqlen_k: int,
        # The softmax scale in attention.
        softmax_scale: float,
        # Causal self-attention.
        causal: bool,
        # Sliding window attention bounds.
        left_window: int, right_window: int,
        # Attention logit soft-capping.
        softcap: float,
    ) -> AttentionLogSumExp:

        attn_out, *_, softmax_lse, _, _ = flash_attn_cuda.varlen_fwd(
            # The QKV components and output tensor.
            query, key, value, None,
            # The Q and K cumulative sequence lengths.
            cu_seqlen_query, cu_seqlen_key,
            # The length bounds for K, prefix lengths, paged-kv block table, and alibi slopes.
            None, None, None, None,
            # The maximum sequence lengths.
            max_seqlen_q, max_seqlen_k,
            # The dropout probability.
            0.0,
            # The softmax scale in attention.
            softmax_scale,
            # Non-zero tensors and causal self-attention.
            False, causal,
            # Sliding window attention widths.
            left_window, right_window,
            # Attention logit soft-cap value.
            softcap,
            # Return softmax attention scores.
            False,
            # Optional RNG for dropout.
            None,
        )

        return attn_out, softmax_lse

    def forward_hopper(
        self,
        # The QKV components.
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        # The cumulative Q and K sequence lengths.
        cu_seqlen_query: torch.Tensor, cu_seqlen_key: torch.Tensor,
        # The maximum Q and K length upper bounds.
        max_seqlen_q: int, max_seqlen_k: int,
        # The softmax scale in attention.
        softmax_scale: float,
        # Causal self-attention.
        causal: bool,
        # Sliding window attention bounds.
        left_window: int, right_window: int,
        # Attention logit soft-capping.
        softcap: float,
    ) -> AttentionLogSumExp:
        #
        # EA: Using only first and second-to-last return value
        #
        attn_out, *_, softmax_lse, _ = flash_attn_hopper_cuda.fwd_varlen(
            # The QKV components and output tensor.
            query, key, value, None,
            # The Q and K cumulative sequence lengths.
            cu_seqlen_query, cu_seqlen_key,
            # The maximum sequence lengths.
            max_seqlen_q, max_seqlen_k,
            # The softmax scale in attention.
            softmax_scale,
            # Causal self-attention.
            causal,
        )

        return attn_out, softmax_lse


class FlashDecode(BaseKernel):

    def meta(
        self,
        # The QKV components.
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        # The cached KV components.
        kcache: torch.Tensor, vcache: torch.Tensor,
        # The tracking sequence lengths.
        cache_seqlens: torch.Tensor,
        # The allocated sequence indices for the cache.
        cache_indices: torch.Tensor,
        # The cached rotary embedding supports.
        cos: torch.Tensor, sin: torch.Tensor,
        # The softmax scale in attention.
        softmax_scale: float,
        # Non-sliding window attention.
        left_window: int, right_window: int,
        # Causal self-attention.
        causal: bool,
        # Non-interleaved rotary embedding.
        interleaved: bool,
        # Attention logits soft-capping.
        softcap: float,
    ) -> AttentionLogSumExp:

        batch_size, timesteps, n_heads, _ = query.shape

        softmax_lse = query.new_empty((batch_size, n_heads, timesteps))

        return torch.empty_like(query), softmax_lse

    def forward(
        self,
        # The QKV components.
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        # The cached KV components.
        kcache: torch.Tensor, vcache: torch.Tensor,
        # The tracking sequence lengths.
        cache_seqlens: torch.Tensor,
        # 
        # EA: "sequence indices"
        # 
        # The allocated sequence indices for the cache.
        cache_indices: torch.Tensor,
        # The cached rotary embedding supports.
        cos: torch.Tensor, sin: torch.Tensor,
        # The softmax scale in attention.
        softmax_scale: float,
        # Non-sliding window attention.
        left_window: int, right_window: int,
        # Causal self-attention.
        causal: bool,
        # Non-interleaved rotary embedding.
        interleaved: bool,
        # Attention logits soft-capping.
        softcap: float,
    ) -> AttentionLogSumExp:

        return flash_attn_cuda.fwd_kvcache(
            # The Q component.
            query,
            # The cached KV components.
            kcache, vcache,
            # The KV components.
            key, value,
            # The tracking sequence lengths.
            cache_seqlens,
            # The cached rotary embedding supports.
            cos, sin,
            # 
            # EA: "sequence indices"
            # 
            # The allocated sequence indices for the cache.
            cache_indices,
            # The prefix lengths, block table, alibi slopes, and output tensor.
            None, None, None, None,
            # The softmax scale in attention.
            softmax_scale,
            # Causal self-attention.
            causal,
            # 
            # EA: "non-sliding"?
            # 
            # Non-sliding window attention.
            left_window, right_window,
            # Attention logits soft-capping.
            softcap,
            # Non-interleaved rotary embedding.
            interleaved,
            # Heuristic based splitting.
            KV_SPLITS,
        )


class AttentionLayerNorm(nn.Module):

    def __init__(
        # Required.
        self, prefix: str, config: T, weights: Weights, name: str,
        # Optional(s).
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

        self.weight = nn.Parameter(weights.get_tensor(f'{prefix}.{name}.weight'))
        self.bias = nn.Parameter(weights.get_tensor(f'{prefix}.{name}.bias')) if bias else None

        self.shape = tuple(self.weight.shape)
        self.variance_epsilon = getattr(config, 'layer_norm_epsilon', eps)

    def forward(self, attention_states: torch.Tensor) -> torch.Tensor:

        return layer_norm(
            input=attention_states,
            normalized_shape=self.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )


class FlashSelfAttention(nn.Module):

    """
    Flash Self-Attention with Tensor Parallelism across the attention heads.

    Also supports both MHA e.g. 7B and 13B variants of LLaMA 2 models and GQA e.g. 70B
    variant of LLaMA 2.

    """

    def __init__(
        self,
        # Required.
        prefix: str,
        config: T,
        weights: Weights,
        # Naming scheme(s).
        qkv_name: str = '{}_proj',
        out_name: str = 'o_proj',
        # Optional(s).
        packed: bool = False,
        causal: bool = True,
        interleaved: bool = False,
        qk_norm: bool = False,
        qkv_bias: bool = False,
        out_bias: bool = False,
        windowed: bool = True,
        groups: int | None = None,
    ) -> None:

        super().__init__()

        # Logging specifics.
        _, layer_id, _ = prefix.rsplit('.', 2)

        self.layer_id = int(layer_id)

        # Attention heads and dimensions.
        self.num_heads = config.num_attention_heads

        self.hidden_size, self.head_size = self.resolve_sizes(
            config.hidden_size, getattr(config, 'head_size', None)
        )

        self.softmax_scale = 1 / sqrt(getattr(config, 'attention_scale', self.head_size))

        self.world_size = weights.world_size
        self.compiled = getattr(config, 'enable_inductor', False)

        self.num_heads = self.validate_num_heads()
        self.num_key_value_heads = config.num_key_value_heads // self.world_size

        # Attention QKV-O weights.
        self.qkv_proj = load_qkv(prefix, qkv_name, config, weights, packed, qkv_bias, groups)
        self.validate_qkv(config.quantize)

        self.o_proj = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.{out_name}", weights=weights, bias=out_bias
        )

        # Rotary positional embedding and flag for flash-decode kernel.
        self.apply_rotary = interleaved_rotary if interleaved else chunked_rotary
        self.rotary_dim = getattr(config, 'rotary_dim', self.head_size)

        self.interleaved = interleaved

        # Sliding window configuration.
        self.left_window = self.right_window = -1

        if windowed:

            self.left_window = config.left_window
            self.right_window = config.right_window

            windowed = not (self.left_window == self.right_window == -1)

        # Decode sequence length, primarily applicable for speculative decoding.
        self.decode_seqlen = config.decode_seqlen

        # Flag indicating causal self-attention.
        self.causal = causal

        # Logit soft-capping via tanh.
        self.softcap = getattr(config, 'attention_softcap', 0.0)

        # Hopper-specific attention acceleration flag.
        hopper = HAS_FA3 and not DISABLE_FA3 and not windowed and not self.softcap

        # Inductor compatibility.
        self.prefill_kernel = FlashPrefill.initialize('flash_prefill', weights.device.type, hopper)
        self.prefill_forward = self.prefill_kernel.operator(self.compiled)

        self.decode_kernel = FlashDecode.initialize('flash_decode', weights.device.type)
        self.decode_forward = self.decode_kernel.operator(self.compiled)

        # Query and key layer norm.
        self.query_norm = self.key_norm = lambda x: x

        if qk_norm:

            self.query_norm = AttentionLayerNorm(prefix, config, weights, 'q_norm')
            self.key_norm = AttentionLayerNorm(prefix, config, weights, 'k_norm')

    def register_cache(self, kcache: torch.Tensor, vcache: torch.Tensor) -> None:

        self.register_buffer('kcache', kcache)
        self.register_buffer('vcache', vcache)

    def clear_cache(self) -> None:

        self.register_buffer('kcache', None)
        self.register_buffer('vcache', None)

    def resolve_sizes(self, hidden_size: int, head_size: int | None) -> Tuple[int, int]:
        """
        Resolves both the `hidden_size` and the `head_size` with the latter taking priority when
        provided for computing the former.

        Note(s):

            • In the absence of a `head_size`, divisibility validation is performed on the
              `hidden_size`.

        """
        if (head_size_ := head_size) is None:

            head_size_, not_divisible = divmod(hidden_size, self.num_heads)

            if not_divisible:

                raise ValueError(
                    "`hidden_size` must be divisible by `num_heads` (got `num_heads`: "
                    f"{self.num_heads} and `hidden_size`: {hidden_size}"
                )

        return hidden_size, head_size_

    def validate_num_heads(self) -> int:
        """
        Ensures that `num_heads` is a multiplicative factor of the `world_size`.

        Returns: The validated `num_heads`.

        """
        num_heads, not_divisible = divmod(self.num_heads, self.world_size)

        if not_divisible:

            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {self.world_size}"
            )

        return num_heads

    def validate_qkv(self, quant_strategy: QuantStrategy | None) -> None:

        if quant_strategy is None or quant_strategy.is_fp8:

            weight = self.qkv_proj.linear.weight
            total_heads = self.num_heads + (self.num_key_value_heads * 2)

            out_channels = total_heads * self.head_size
            in_channels = self.hidden_size

            expectation = f"{list(weight.shape)} != [{out_channels}, {in_channels}]"

            assert weight.shape == (out_channels, in_channels), expectation

        elif RANK == self.layer_id == 0:

            logger.warning(
                f'validation of {quant_strategy.method}-quantized qkv tensors is not supported'
            )

    def prefill(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        cache_indices: torch.Tensor,
        max_seqlen: torch.Tensor,
        cu_seqlen_prefill: torch.Tensor,
    ) -> torch.Tensor:

        query = query.view(-1, self.num_heads, self.head_size)

        key, value = (
            kv.view(-1, 2, self.num_key_value_heads, self.head_size)
            .unbind(1)
        )

        query = self.query_norm(query)
        key = self.key_norm(key)
        # Here's the rotary being applied outside the kernel
        self.apply_rotary(query[..., :self.rotary_dim], cos, sin)
        self.apply_rotary(key[..., :self.rotary_dim], cos, sin)

        kcache = getattr(self, 'kcache', None)
        vcache = getattr(self, 'vcache', None)

        if kcache is None or vcache is None:

            ValueError('cache buffers for self-attention has not been registered')
        #
        # EA: Update kv cache in place manually
        #
        # Update the KV cache in-place.
        kcache[cache_indices[sequence_ids], position_ids] = key
        vcache[cache_indices[sequence_ids], position_ids] = value

        max_seqlen_q = max_seqlen_k = int(max_seqlen.item())

        attn_out, _ = self.prefill_forward(
            # The QKV components.
            query, key, value,
            # The Q & K cumulative sequence lengths.
            cu_seqlen_prefill, cu_seqlen_prefill,
            # The maximum sequence lengths.
            max_seqlen_q, max_seqlen_k,
            # The softmax scale in attention.
            self.softmax_scale,
            # Causal self-attention.
            self.causal,
            # Sliding window attention bounds.
            self.left_window, self.right_window,
            # Attention logit soft-capping.
            self.softcap,
        )

        return self.o_proj(attn_out.view(-1, self.num_heads * self.head_size))

    def decode(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:

        query = query.view(-1, self.decode_seqlen, self.num_heads, self.head_size)

        key, value = (
            kv.view(-1, self.decode_seqlen, 2, self.num_key_value_heads, self.head_size)
            .unbind(2)
        )

        query = self.query_norm(query)
        key = self.key_norm(key)

        kcache = getattr(self, 'kcache', None)
        vcache = getattr(self, 'vcache', None)

        if kcache is None or vcache is None:

            ValueError('cache buffers for self-attention has not been registered')

        attn_out, _ = self.decode_forward(
            # The QKV components.
            query, key, value,
            # The cached KV components.
            kcache, vcache,
            # The tracking sequence lengths.
            cache_seqlens,
            # The allocated sequence indices for the cache.
            cache_indices,
            # The cached rotary embedding supports.
            cos, sin,
            # The softmax scale in attention.
            self.softmax_scale,
            # Non-sliding window attention.
            self.left_window, self.right_window,
            # Causal self-attention.
            self.causal,
            # Non-interleaved rotary embedding.
            self.interleaved,
            # Attention logit soft-capping.
            self.softcap,
        )

        return self.o_proj(attn_out.view(-1, self.num_heads * self.head_size))

    def merge_partials(
        self,
        prefix_out: torch.Tensor,
        suffix_out: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expectation on shapes:

            • Prefix Out -> [ L, H, Z ].

            • Prefix Lse -> [ H, L ].

            • Suffix Out -> [ B, K + 1, H, Z ].

            • Suffix Lse -> [ B, H, K + 1 ].

        where L -> B x (K + 1).

        """
        # Prefix Out -> [ B, K + 1, H, Z ].
        prefix_out = prefix_out.view(-1, self.decode_seqlen, self.num_heads, self.head_size)

        # Prefix / Suffix LSE -> [ B, K + 1, H ].
        prefix_lse = prefix_lse.T.view(-1, self.decode_seqlen, self.num_heads)
        suffix_lse = suffix_lse.transpose(1, 2)

        # Total LSE -> [ B, K + 1, H ].
        total_lse = prefix_lse.logaddexp(suffix_lse)

        # Prefix / Suffix Scale -> [ B, K + 1, H, 1 ].
        prefix_scale = prefix_lse.sub_(total_lse).exp_().unsqueeze_(-1)
        suffix_scale = suffix_lse.sub_(total_lse).exp_().unsqueeze_(-1)

        # Rescaled Prefix / Suffix.
        rescaled_prefix = prefix_out.mul_(prefix_scale)
        rescaled_suffix = suffix_out.mul_(suffix_scale)

        return rescaled_prefix.add_(rescaled_suffix)

    def split_decode(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        prefix_seqlens: torch.Tensor,
        prefix_sequence_ids: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        cu_seqlen_query: torch.Tensor,
        cu_seqlen_key: torch.Tensor,
    ) -> torch.Tensor:

        # First Stage: Prefix-Attention.
        query = self.query_norm(query.view(-1, self.num_heads, self.head_size))

        kcache = getattr(self, 'kcache', None)
        vcache = getattr(self, 'vcache', None)

        if kcache is None or vcache is None:

            ValueError('cache buffers for self-attention has not been registered')

        shared_keys = kcache[cache_indices[prefix_sequence_ids], prefix_position_ids]
        shared_values = vcache[cache_indices[prefix_sequence_ids], prefix_position_ids]

        max_seqlen_q = self.query.size(0)
        max_seqlen_k = self.kcache.size(1)

        # Prefix Out -> [ L = (B x (K + 1)), H, Z ] | Prefix LSE -> [ H, L ].
        prefix_out, prefix_lse = self.prefill_forward(
            # The QKV components.
            query, shared_keys, shared_values,
            # The cumulative Q and K sequence lengths.
            cu_seqlen_query * self.decode_seqlen, cu_seqlen_key,
            # The maximum Q and K length upper bounds.
            max_seqlen_q, max_seqlen_k,
            # The softmax scale and non-causal attention.
            self.softmax_scale, False,
            # Non-sliding window attention.
            self.left_window, self.right_window,
        )

        # Second Stage: Suffix-Attention.
        query = query.view(-1, self.decode_seqlen, self.num_heads, self.head_size)

        key, value = (
            kv.view(-1, self.decode_seqlen, 2, self.num_key_value_heads, self.head_size)
            .unbind(2)
        )

        key = self.key_norm(key)

        # Suffix Out -> [ B, K + 1, H, Z ] | Suffix LSE -> [ B, H, K + 1 ].
        suffix_out, suffix_lse = self.decode_forward(
            # The QKV components.
            query, key, value,
            # The cached KV components.
            kcache, vcache,
            # The tracking sequence lengths.
            cache_seqlens,
            # The allocated sequence indices for the cache.
            cache_indices,
            # The prefix lengths for constraining the KV subset.
            prefix_seqlens,
            # The cached rotary embedding supports.
            cos, sin,
            # The softmax scale in attention.
            self.softmax_scale,
            # Non-sliding window attention.
            self.left_window, self.right_window,
            # Causal self-attention.
            self.causal,
            # Non-interleaved rotary embedding.
            self.interleaved,
        )

        attn_out = self.merge_partials(prefix_out, suffix_out, prefix_lse, suffix_lse)

        return self.o_proj(attn_out.view(-1, self.num_heads * self.head_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_indices: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        cu_seqlen_prefill: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        prefix_seqlens: torch.Tensor | None = None,
        prefix_sequence_ids: torch.Tensor | None = None,
        prefix_position_ids: torch.Tensor | None = None,
        cu_seqlen_query: torch.Tensor | None = None,
        cu_seqlen_key: torch.Tensor | None = None,
    ) -> torch.Tensor:

        qkv = self.qkv_proj(hidden_states)

        q_dims = self.head_size * self.num_heads
        kv_dims = self.head_size * self.num_key_value_heads * 2

        query, kv = qkv.split((q_dims, kv_dims), dim=1)

        if cu_seqlen_prefill is not None:

            return self.prefill(
                # The QKV components.
                query, kv,
                # The cached rotary embedding supports.
                cos[position_ids, None, ...], sin[position_ids, None, ...],
                # The KV cache indexing specifics.
                position_ids, sequence_ids, cache_indices,
                # The sequence length specifics for flash attention.
                max_seqlen, cu_seqlen_prefill,
            )

        elif prefix_seqlens is not None:

            return self.split_decode(
                # The QKV components.
                query, kv,
                # The cached rotary embedding supports.
                cos, sin,
                # Allocation indices for cache.
                cache_indices,
                # The incremental update specifics.
                cache_seqlens,
                # The repeated shared-prefix lengths.
                prefix_seqlens,
                # The shared-prefix KV cache indexing specifics.
                prefix_sequence_ids, prefix_position_ids,
                # The shared-prefix sequence bounds.
                cu_seqlen_query, cu_seqlen_key,
            )

        elif cache_seqlens is not None:

            return self.decode(
                # The QKV components.
                query, kv,
                # The cached rotary embedding supports.
                cos, sin,
                # Allocation indices for cache.
                cache_indices,
                # The incremental update specifics.
                cache_seqlens,
            )

        else:

            raise ValueError('one of `cu_seqlen_prefill` or `cache_seqlens` must be provided')
