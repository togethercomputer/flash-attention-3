import torch
from flashattn_hopper_cuda import fwd_kvcache

seed = 123
torch.manual_seed(seed)

dtype = torch.float8_e4m3fn
dtype_ref = torch.bfloat16
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

full_tensor_k         = torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype_ref, device = device)
full_tensor_v         = torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype_ref, device = device)
seqlens_k_            = torch.randint(1, max_seqlen_before + 1, size=(batch_size_c,), dtype = torch.int32, device = device)
identity_tensor       = torch.arange(kv_capacity, device = device)[None, :]
mask                  = identity_tensor < seqlens_k_[:, None]
mask                  = mask.unsqueeze(-1).unsqueeze(-1)

q                     = torch.randn((batch_size, seqlen_q   , num_heads_q, head_size), dtype = dtype_ref, device = device).to(dtype)
kcache                = torch.where(mask, full_tensor_k, torch.zeros_like(full_tensor_k)).to(dtype)
vcache                = torch.where(mask, full_tensor_v, torch.zeros_like(full_tensor_v)).to(dtype)
k_                    = torch.randn((batch_size, seqlen_knew, num_heads_k, head_size), dtype = dtype_ref, device = device).to(dtype)
v_                    = torch.randn((batch_size, seqlen_knew, num_heads_k, head_size), dtype = dtype_ref, device = device).to(dtype)
out                   = None
seqlens_k_            = seqlens_k_
rotary_cos_           = None
rotary_sin_           = None
cache_batch_idx_      = None
leftpad_k_            = None
page_table_           = None
cu_seqlens_q_         = None
max_seqlen_q_         = None
softmax_scale         = q.shape[-1] ** (-0.5)
is_causal             = True
descale_q_            = torch.tensor([3.0], device=device)
descale_k_            = torch.tensor([4.0], device=device)
descale_v_            = torch.tensor([5.0], device=device)
window_size_left      = -1
window_size_right     = -1
sink_token_length     = 0
softcap               = 0.0
is_rotary_interleaved = False
num_splits            = 4
pack_gqa_             = False

y = fwd_kvcache(
    q, 
    kcache, 
    vcache, 
    k_, 
    v_, 
    out, 
    seqlens_k_, 
    rotary_cos_, 
    rotary_sin_, 
    cache_batch_idx_, 
    leftpad_k_, 
    page_table_, 
    cu_seqlens_q_,
    max_seqlen_q_,
    softmax_scale,
    is_causal, 
    descale_q_, 
    descale_k_, 
    descale_v_, 
    window_size_left, 
    window_size_right, 
    sink_token_length,
    softcap, 
    is_rotary_interleaved, 
    num_splits, 
    pack_gqa_)

'''
Proximate goal for scalar -> vector modification: running this prints

batch element 0
descale_q: 3.0
descale_k: 4.0
descale_v: 5.0
batch element 1
descale_q: 6.0
descale_k: 7.0
descale_v: 8.0
'''
