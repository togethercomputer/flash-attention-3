import math

import pytest
import torch
from flash_attn_interface import flash_attn_with_kvcache
from flashattn_hopper_cuda import fwd_kvcache
from test_flash_attn import attention_ref, attention_ref_new
from einops import rearrange, repeat

from flash_attn.layers.rotary import apply_rotary_emb

seed = 123
torch.manual_seed(seed)

@pytest.mark.parametrize("descale", [False, True])
# @pytest.mark.parametrize("page_size", [None])
@pytest.mark.parametrize("page_size", [None, 4])
def test_fp8(
    page_size,
    descale,
):
    dtype                        = torch.float8_e4m3fn
    dtype_ref                    = torch.bfloat16
    dtype_descale                = torch.float32
    device                       = 'cuda'
           
    batch_size                   = 2
    batch_size_c                 = batch_size
    num_heads_q                  = 2
    num_heads_k                  = num_heads_q
    seqlen_q                     = 2
    seqlen_new                   = seqlen_q
    kv_capacity                  = 8
    kv_used_constant             = 3
    head_size                    = 32
    
    cache_seqlens                = torch.full((batch_size_c,), kv_used_constant, dtype = torch.int32, device=device)

    q                            = torch.randn(batch_size  , seqlen_q  , num_heads_q, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    q_ref                        = q      .clone()
    q                            = q      .to(dtype)
            
    k                            = torch.randn(batch_size  , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    v                            = torch.randn(batch_size  , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    k_ref                        = k      .clone()
    v_ref                        = v      .clone()
    k                            = k      .to(dtype)
    v                            = v      .to(dtype)

    if page_size is None:        
        k_cache                  = torch.randn(batch_size_c, kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        v_cache                  = torch.randn(batch_size_c, kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        k_cache_ref              = k_cache.clone()
        v_cache_ref              = v_cache.clone()
        k_cache                  = k_cache.to(dtype)
        v_cache                  = v_cache.to(dtype)
        
        page_table = None
    else:
        # Why times 3 here? I guess just to simulate unoccupied logical pages
        num_pages                = math.ceil(kv_capacity / page_size) * batch_size_c * 3
        page_table_flat          = torch.randperm(num_pages, dtype = torch.int32, device = device)
        page_table               = rearrange(page_table_flat, '(b blocks_per_elem) -> b blocks_per_elem', b = batch_size_c)
        
        k_cache_paged            = torch.randn((num_pages, page_size, num_heads_k, head_size), dtype = dtype_ref, device = device)
        v_cache_paged            = torch.randn((num_pages, page_size, num_heads_k, head_size), dtype = dtype_ref, device = device)
        k_cache_paged_ref        = k_cache_paged.clone()
        v_cache_paged_ref        = v_cache_paged.clone()
        k_cache_paged            = k_cache_paged.to(dtype)
        v_cache_paged            = v_cache_paged.to(dtype)
        
        k_cache_fat              = k_cache_paged_ref[page_table.flatten()]
        v_cache_fat              = v_cache_paged_ref[page_table.flatten()]
        k_cache                  = rearrange(k_cache_fat, '(b p_local) E ... -> b (p_local E) ...', b = batch_size_c)[:, :kv_capacity]
        v_cache                  = rearrange(v_cache_fat, '(b p_local) E ... -> b (p_local E) ...', b = batch_size_c)[:, :kv_capacity]
        
        k_cache_ref              = k_cache.clone()
        v_cache_ref              = v_cache.clone()
        k_cache                  = k_cache.to(dtype)
        v_cache                  = v_cache.to(dtype)

    if descale is False:
        q_descale                = None
        k_descale                = None
        v_descale                = None
    else:
        q_descale                = torch.rand((batch_size_c,), dtype = dtype_descale, device = device)
        k_descale                = torch.rand((batch_size_c,), dtype = dtype_descale, device = device)
        v_descale                = torch.rand((batch_size_c,), dtype = dtype_descale, device = device)


    q                            = q
    k_cache                      = k_cache
    v_cache                      = v_cache
    k                            = k
    v                            = v
    # out
    cache_seqlens                = cache_seqlens
    cos                          = None
    sin                          = None
    cache_batch_idx              = None
    leftpad_k                    = None
    page_table                   = page_table
    cu_seqlens_q                 = None
    max_seqlen_q                 = None
    softmax_scale                = q.shape[-1] ** (-0.5)
    causal                       = True
    q_descale                    = q_descale
    k_descale                    = k_descale
    v_descale                    = v_descale
    window_left, window_right    = (-1, -1)
    sink_token_length            = 0
    softcap                      = 0.0
    rotary_interleave            = False
    num_splits                   = 1
    pack_gqa                     = None
    
    id_ten                       = torch.arange(kv_capacity, device=device)
    id_ten                       = rearrange(id_ten       , "s -> 1 s")
    cache_seqlens_expanded       = rearrange(cache_seqlens, 'b -> b 1')
    slots_used_before_append     = id_ten < cache_seqlens_expanded
    new_slots                    = torch.logical_and(cache_seqlens_expanded <= id_ten, 
                                                     id_ten < cache_seqlens_expanded + seqlen_new)
    
    # manually append {k,v}new to {k,v}_cache_ref
    k_cache_ref_og               = k_cache_ref.clone()
    v_cache_ref_og               = v_cache_ref.clone()
    
    k_cache_ref[new_slots]       = rearrange(k_ref, 'b s ... -> (b s) ...')
    v_cache_ref[new_slots]       = rearrange(v_ref, 'b s ... -> (b s) ...')
    out, *rest = flash_attn_with_kvcache(
        q,
        k_cache if page_size is None else k_cache_paged,
        v_cache if page_size is None else v_cache_paged,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=(window_left, window_right),
        rotary_interleaved=rotary_interleave,
        num_splits=num_splits,
        return_softmax_lse=True
    )
    
    (
        out_ref,
        *rest,
    )                            = attention_ref_new(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        key_padding_mask = torch.logical_or(slots_used_before_append, new_slots),
        q_descale        = q_descale,
        k_descale        = k_descale,
        v_descale        = v_descale,
    )

    if page_size is not None:
        # unpage the now-larger kvcache
        k_cache_fat = k_cache_paged.to(dtype_ref)[page_table.flatten()]
        k_cache_fat = rearrange(k_cache_fat, '(B p_local) E ... -> B (p_local E) ...', B = batch_size_c)
        v_cache_fat = v_cache_paged.to(dtype_ref)[page_table.flatten()]
        v_cache_fat = rearrange(v_cache_fat, '(B p_local) E ... -> B (p_local E) ...', B = batch_size_c)
        k_cache     = k_cache_fat[:, :kv_capacity].to(dtype)
        v_cache     = v_cache_fat[:, :kv_capacity].to(dtype)


    k_cache = k_cache.to(dtype_ref)
    # Don't get why we need to do this again, since what we're appending is
    # downcast from dtype_ref already
    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
    assert torch.equal(k_cache, k_cache_ref)
    assert ((out.to(dtype_ref) - out_ref).abs().mean().item() < .1)

if __name__=='__main__':
    dtype                        = torch.float8_e4m3fn
    dtype_ref                    = torch.bfloat16
    dtype_descale                = torch.float32
    device                       = 'cuda'
           
    batch_size                   = 2
    batch_size_cache             = batch_size
    num_heads_q                  = 2
    num_heads_k                  = num_heads_q
    seqlen_q                     = 2
    seqlen_new                   = seqlen_q
    kv_capacity                  = 8
    kv_used_constant             = 3
    head_size                    = 32
    cache_batch_idx              = None
    # cache_seqlens is size batch_size, not batch_size_cache, we expect it to be
    # "pre-sliced"
    cache_seqlens                = torch.full((batch_size,), kv_used_constant, dtype = torch.int32, device=device)

    q                            = torch.randn(batch_size        , seqlen_q  , num_heads_q, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    q_ref                        = q      .clone()
    q                            = q      .to(dtype)
           
    k                            = torch.randn(batch_size        , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    v                            = torch.randn(batch_size        , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    k_ref                        = k      .clone()
    v_ref                        = v      .clone()
    k                            = k      .to(dtype)
    v                            = v      .to(dtype)
    
    k_cache                      = torch.randn(batch_size_cache  , kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    v_cache                      = torch.randn(batch_size_cache  , kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    k_cache_ref                  = (
                                        k_cache if cache_batch_idx is None else k_cache[cache_batch_idx.to(dtype=torch.long)]
                                    ).clone()
    v_cache_ref                  = (
                                        v_cache if cache_batch_idx is None else v_cache[cache_batch_idx.to(dtype=torch.long)]
                                    ).clone()
    k_cache                      = k_cache.to(dtype)
    v_cache                      = v_cache.to(dtype)
    
    q                            = q
    k_cache                      = k_cache
    v_cache                      = v_cache
    k                            = k
    v                            = v
    # out
    cache_seqlens                = cache_seqlens
    cos                          = None
    sin                          = None
    # cache_batch_idx    this changes
    leftpad_k                    = None
    page_table                   = None
    cu_seqlens_q                 = None
    max_seqlen_q                 = None
    softmax_scale                = q.shape[-1] ** (-0.5)
    causal                       = True
    # q_descale         these change
    # k_descale
    # v_descale
    window_left, window_right    = (-1, -1)
    sink_token_length            = 0
    softcap                      = 0.0
    rotary_interleave            = False
    num_splits                   = 1
    pack_gqa                     = None
    
    id_ten                       = torch.arange(kv_capacity, device=device)
    id_ten                       = rearrange(id_ten       , "s -> 1 s")
    cache_seqlens_expanded       = rearrange(cache_seqlens, 'b -> b 1')
    slots_used_before_append     = id_ten < cache_seqlens_expanded
    new_slots                    = torch.logical_and(cache_seqlens_expanded <= id_ten, 
                                                     id_ten < cache_seqlens_expanded + seqlen_new)

    print()
    print("No descale, no cache batch index")
    print()

    q_descale                    = None
    k_descale                    = None
    v_descale                    = None

    cache_batch_idx              = None

    # manually append knew_ro, vnew to {k,v}_cache_ref
    k_cache_ref_og               = k_cache_ref.clone()
    v_cache_ref_og               = v_cache_ref.clone()
    
    k_cache_ref[new_slots]       = rearrange(k_ref,    'b s ... -> (b s) ...')
    v_cache_ref[new_slots]       = rearrange(v_ref   , 'b s ... -> (b s) ...')
        
    out, *rest = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=(window_left, window_right),
        rotary_interleaved=rotary_interleave,
        num_splits=num_splits,
        return_softmax_lse=True
    )

    (
        out_ref,
        *rest,
    )                            = attention_ref_new(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        key_padding_mask = torch.logical_or(slots_used_before_append, new_slots),
        q_descale        = q_descale,
        k_descale        = k_descale,
        v_descale        = v_descale,
    )

    # Don't get why we need to do this again, since what we're appending is
    # downcast from dtype_ref already
    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
    print(f"k_cache max diff: {(k_cache.to(dtype_ref) - k_cache_ref).abs().max().item()}")
    print(f"v_cache max diff: {(v_cache.to(dtype_ref) - v_cache_ref).abs().max().item()}")
    print(f"Output max diff:  {(out.to(dtype_ref) - out_ref).abs().max() .item()}")
    print(f"Output mean diff: {(out.to(dtype_ref) - out_ref).abs().mean().item()}")

    print()
    print("With descale, no cache batch index")
    print()

    # undo append
    k_cache                      = k_cache_ref_og.to(dtype)
    v_cache                      = v_cache_ref_og.to(dtype)
    
    q_descale                    = torch.rand((batch_size      ,), dtype = dtype_descale, device = device)
    k_descale                    = torch.rand((batch_size_cache,), dtype = dtype_descale, device = device)
    v_descale                    = torch.rand((batch_size_cache,), dtype = dtype_descale, device = device)

    out, *rest = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=(window_left, window_right),
        rotary_interleaved=rotary_interleave,
        num_splits=num_splits,
        return_softmax_lse=True
    )

    (
        out_ref,
        *rest,
    )                            = attention_ref_new(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        key_padding_mask = torch.logical_or(slots_used_before_append, new_slots),
        q_descale        = q_descale,
        k_descale        = k_descale,
        v_descale        = v_descale,
    )

    # Don't get why we need to do this again, since what we're appending is
    # downcast from dtype_ref already
    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
    print(f"k_cache max diff: {(k_cache.to(dtype_ref) - k_cache_ref).abs().max().item()}")
    print(f"v_cache max diff: {(v_cache.to(dtype_ref) - v_cache_ref).abs().max().item()}")
    print(f"Output max diff:  {(out.to(dtype_ref) - out_ref).abs().max() .item()}")
    print(f"Output mean diff: {(out.to(dtype_ref) - out_ref).abs().mean().item()}")

    print()
    print("No descale, with cache batch index")
    print()
    
    batch_size_cache              = batch_size * 2
    cache_batch_idx               = torch.randperm(batch_size_cache, dtype = torch.int32, device = device)[:batch_size]
    print(f"{cache_batch_idx = }")

    # cache_seqlens is size batch_size, not batch_size_cache, we expect it to be
    # "pre-sliced"
    cache_seqlens                = torch.full((batch_size,), kv_used_constant, dtype = torch.int32, device=device)

    id_ten                       = torch.arange(kv_capacity, device=device)
    id_ten                       = rearrange(id_ten       , "s -> 1 s")
    cache_seqlens_expanded       = rearrange(cache_seqlens, 'b -> b 1')
    slots_used_before_append     = id_ten < cache_seqlens_expanded
    new_slots                    = torch.logical_and(cache_seqlens_expanded <= id_ten, 
                                                     id_ten < cache_seqlens_expanded + seqlen_new)

    q_descale                     = None
    k_descale                     = None
    v_descale                     = None

    q                            = torch.randn(batch_size        , seqlen_q  , num_heads_q, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    q_ref                        = q      .clone()
    q                            = q      .to(dtype)
           
    k                            = torch.randn(batch_size        , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    v                            = torch.randn(batch_size        , seqlen_new, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    k_ref                        = k      .clone()
    v_ref                        = v      .clone()
    k                            = k      .to(dtype)
    v                            = v      .to(dtype)
    
    k_cache                      = torch.randn(batch_size_cache  , kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    v_cache                      = torch.randn(batch_size_cache  , kv_capacity, num_heads_k, head_size, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
    k_cache_og                   = k_cache.clone()
    v_cache_og                   = v_cache.clone()
    # clone again cause we'll be manually appending to k_cache_ref but not k_cache_og
    k_cache_ref                  = k_cache_og.clone()[cache_batch_idx.to(dtype=torch.long)]
    v_cache_ref                  = v_cache_og.clone()[cache_batch_idx.to(dtype=torch.long)]
    k_cache                      = k_cache.to(dtype)
    v_cache                      = v_cache.to(dtype)
    
    k_cache_ref[new_slots]       = rearrange(k_ref,    'b s ... -> (b s) ...')
    v_cache_ref[new_slots]       = rearrange(v_ref   , 'b s ... -> (b s) ...')

    out, *rest = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=(window_left, window_right),
        rotary_interleaved=rotary_interleave,
        num_splits=num_splits,
        return_softmax_lse=True
    )

    (
        out_ref,
        *rest,
    )                            = attention_ref_new(
        q_ref,
        k_cache_ref,
        v_cache_ref,
        key_padding_mask = torch.logical_or(slots_used_before_append, new_slots),
        q_descale        = q_descale,
        k_descale        = k_descale,
        v_descale        = v_descale,
    )

    # Don't get why we need to do this again, since what we're appending is
    # downcast from dtype_ref already
    k_cache_ref    = k_cache_ref.to(dtype).to(dtype_ref)
    v_cache_ref    = v_cache_ref.to(dtype).to(dtype_ref)
    k_cache_select = k_cache.to(dtype_ref)[cache_batch_idx.to(dtype=torch.long)]
    v_cache_select = v_cache.to(dtype_ref)[cache_batch_idx.to(dtype=torch.long)]
    print(f"k_cache max diff: {(k_cache_select - k_cache_ref).abs().max().item()}")
    print(f"v_cache max diff: {(v_cache_select - v_cache_ref).abs().max().item()}")
    print(f"Output max diff:  {(out.to(dtype_ref) - out_ref).abs().max() .item()}")
    print(f"Output mean diff: {(out.to(dtype_ref) - out_ref).abs().mean().item()}")
