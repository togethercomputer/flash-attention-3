import torch
from flashattn_hopper_cuda import fwd_kvcache

seed = 123
torch.manual_seed(seed)

dtype = torch.bfloat16
device = 'cuda'

batch_size        = 2
batch_size_c      = batch_size
num_heads_q       = 2
num_heads_k       = 1
seqlen_q          = 2
seqlen_knew       = seqlen_q
kv_capacity       = 7
head_size         = 2
max_seqlen_before = kv_capacity - seqlen_knew

full_tensor_k         = torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype, device = device)
full_tensor_v         = torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype, device = device)
seqlens_k_            = torch.randint(1, max_seqlen_before + 1, size=(batch_size_c,), dtype = torch.int32, device = device)
identity_tensor       = torch.arange(kv_capacity, device = device)[None, :]
mask                  = identity_tensor < seqlens_k_[:, None]
mask                  = mask.unsqueeze(-1).unsqueeze(-1)

q                     = torch.randn((batch_size  , seqlen_q   , num_heads_q, head_size), dtype = dtype, device = device)
kcache                = torch.where(mask, full_tensor_k, torch.zeros_like(full_tensor_k))
vcache                = torch.where(mask, full_tensor_v, torch.zeros_like(full_tensor_v))
k_                    = torch.randn((batch_size_c, seqlen_knew, num_heads_k, head_size), dtype = dtype, device = device)
v_                    = torch.randn((batch_size_c, seqlen_knew, num_heads_k, head_size), dtype = dtype, device = device)
seqlens_k_            = seqlens_k_
rotary_cos_           = None
rotary_sin_           = None
cache_batch_idx_      = None
leftpad_k_            = None
block_table_          = None
alibi_slopes_         = None
out_                  = None
softmax_scale         = q.shape[-1] ** (-0.5)
descale_q_            = None
descale_k_            = None
descale_v_            = None
is_causal             = True
window_size_left      = -1
window_size_right     = -1
softcap               = 0.0
is_rotary_interleaved = False
num_splits            = 4
max_seqlen_k_hint     = kcache.shape[1]
use_gqa_packing       = False

# y = fwd_kvcache(q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_, alibi_slopes_, out_, softmax_scale, descale_q_, descale_k_, descale_v_, is_causal, window_size_left, window_size_right, softcap, is_rotary_interleaved, num_splits, max_seqlen_k_hint, use_gqa_packing)