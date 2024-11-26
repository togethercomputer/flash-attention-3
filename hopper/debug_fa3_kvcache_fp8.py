import torch
from flashattn_hopper_cuda import fwd_kvcache
from test_flash_attn import attention_ref, attention_ref_new

seed = 123
torch.manual_seed(seed)

dtype         = torch.float8_e4m3fn
dtype_ref     = torch.bfloat16
dtype_descale = torch.float32
device        = 'cuda'

batch_size        = 2
batch_size_c      = batch_size
num_heads_q       = 2
num_heads_k       = num_heads_q
seqlen_q          = 1
seqlen_knew       = seqlen_q
kv_capacity       = 4
kv_used_constant  = 2
head_size         = 2
# max_seqlen_before = 

full_tensor_kcache_ref= torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
full_tensor_vcache_ref= torch.randn((batch_size_c, kv_capacity, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
full_tensor_kcache    = full_tensor_kcache_ref.to(dtype)
full_tensor_vcache    = full_tensor_vcache_ref.to(dtype)

k_ref                 = torch.randn((batch_size,   seqlen_knew, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
v_ref                 = torch.randn((batch_size,   seqlen_knew, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
k_                    = k_ref.to(dtype)
v_                    = v_ref.to(dtype)
# k_                    = None
# v_                    = None

seqused_k_            = torch.full((batch_size,), kv_used_constant                     , dtype = torch.int32  , device = device)
# seqused_k_            = torch.randint(1, max_seqlen_before + 1, size=(batch_size_c,),    dtype = torch.int32, device = device)
identity_tensor       = torch.arange(kv_capacity,                                                               device = device)[None, :]
mask                  = identity_tensor < seqused_k_[:, None]
mask                  = mask.unsqueeze(-1).unsqueeze(-1)

q_ref                 = torch.randn((batch_size, seqlen_q    , num_heads_q, head_size) , dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
kcache_ref            = torch.where(mask, full_tensor_kcache_ref, torch.zeros_like(full_tensor_kcache_ref))
vcache_ref            = torch.where(mask, full_tensor_vcache_ref, torch.zeros_like(full_tensor_vcache_ref))
kcache_before         = kcache_ref.to(dtype)
vcache_before         = vcache_ref.to(dtype)

q                     = q_ref.to(dtype)
kcache                = kcache_ref.to(dtype)
vcache                = vcache_ref.to(dtype)
k_                    = k_
v_                    = v_
out                   = None
seqused_k_            = seqused_k_
rotary_cos_           = None
rotary_sin_           = None
cache_batch_idx_      = None
leftpad_k_            = None
page_table_           = None
cu_seqlens_q_         = None
max_seqlen_q_         = None
softmax_scale         = q.shape[-1] ** (-0.5)
is_causal             = False
q_descale_            = torch.rand(batch_size                                           , dtype = dtype_descale, device = device) * 2
k_descale_            = torch.rand(batch_size                                           , dtype = dtype_descale, device = device) * 2
v_descale_            = torch.rand(batch_size                                           , dtype = dtype_descale, device = device) * 2
q_descale_trivial     = torch.ones(batch_size                                           , dtype = dtype_descale, device = device)
k_descale_trivial     = torch.ones(batch_size                                           , dtype = dtype_descale, device = device)
v_descale_trivial     = torch.ones(batch_size                                           , dtype = dtype_descale, device = device)
q_descale_0           = q_descale_[0].unsqueeze(0)
k_descale_0           = k_descale_[0].unsqueeze(0)
v_descale_0           = v_descale_[0].unsqueeze(0)
q_descale_0_trivial   = q_descale_trivial[0].unsqueeze(0)
k_descale_0_trivial   = k_descale_trivial[0].unsqueeze(0)
v_descale_0_trivial   = v_descale_trivial[0].unsqueeze(0)
q_descale_0_repeated  = torch.full((batch_size,), q_descale_[0]                                                , device = device)
k_descale_0_repeated  = torch.full((batch_size,), k_descale_[0]                                                , device = device)
v_descale_0_repeated  = torch.full((batch_size,), v_descale_[0]                                                , device = device)
window_size_left      = -1
window_size_right     = -1
sink_token_length     = 0
softcap               = 0.0
is_rotary_interleaved = False
num_splits            = 1
pack_gqa_             = None

print(f"{k_=}\n")
print(f"kcache before fwd_kvcache\n{kcache}")
(
    mha_fwd_kvcache_output_no_descale                ,
    softmax_lse                                      ,
    out_accum                                        ,
    softmax_lse_accum                                ,
)                                                      = fwd_kvcache(
    q                                                ,
    kcache                                           ,
    vcache                                           ,
    k_                                               ,
    v_                                               ,
    out                                              ,
    seqused_k_                                       ,
    rotary_cos_                                      ,
    rotary_sin_                                      ,
    cache_batch_idx_                                 ,
    leftpad_k_                                       ,
    page_table_                                      ,
    cu_seqlens_q_                                    ,
    max_seqlen_q_                                    ,
    softmax_scale                                    ,
    is_causal                                        ,
    None                                             ,
    None                                             ,
    None                                             ,
    window_size_left                                 ,
    window_size_right                                ,
    sink_token_length                                ,
    softcap                                          , 
    is_rotary_interleaved                            , 
    num_splits                                       , 
    pack_gqa_                                        ,
)
print(f"kcache after fwd_kvcache\n{kcache}")

# (
#     mha_fwd_kvcache_output_descale,
#     softmax_lse                                      ,
#     out_accum                                        ,
#     softmax_lse_accum                                ,
# )                                                      = fwd_kvcache(
#     q                                                ,
#     kcache                                           ,
#     vcache                                           ,
#     k_                                               ,
#     v_                                               ,
#     out                                              ,
#     seqused_k_                                       ,
#     rotary_cos_                                      ,
#     rotary_sin_                                      ,
#     cache_batch_idx_                                 ,
#     leftpad_k_                                       ,
#     page_table_                                      ,
#     cu_seqlens_q_                                    ,
#     max_seqlen_q_                                    ,
#     softmax_scale                                    ,
#     is_causal                                        ,
#     q_descale_                                       ,
#     k_descale_                                       ,
#     v_descale_                                       ,
#     window_size_left                                 ,
#     window_size_right                                ,
#     sink_token_length                                ,
#     softcap                                          , 
#     is_rotary_interleaved                            , 
#     num_splits                                       , 
#     pack_gqa_                                        ,
# )
           
(           
    ref_no_descale                                    ,
    _                                                 ,
)                                                       = attention_ref_new(
    q_ref                                             ,
    kcache_ref                                        ,
    vcache_ref                                        ,
    q_descale = None                                  ,
    k_descale = None                                  ,
    v_descale = None                                  ,
)

(
    ref_descale                                       ,
    _                                                 ,
)                                                       = attention_ref_new(
    q_ref                                             ,
    kcache_ref                                        ,
    vcache_ref                                        ,
    q_descale = q_descale_                            ,
    k_descale = k_descale_                            ,
    v_descale = v_descale_                            ,
)

width = 35

print()

for x in (
    # 'q_descale_'                                      ,
    # 'k_descale_'                                      ,
    # 'v_descale_'                                      ,
):
    print(f"{x:<{width}}{eval(x)}\n")

for x in (
    # 'mask'                                ,
    # 'identity_tensor'                     ,
    # 'q'                                   ,
    'kcache'                              ,
    # 'vcache'                              ,
    'k_'                                  ,
    # 'v_'                                  ,
    # 'ref_no_descale'                      ,
    # 'ref_single_descale_v_only'           ,
    # 'mha_fwd_kvcache_output_no_descale'   ,
):
    print(f"{x:<{width-2}}~ {eval(x).size()}")

print()
for x in (
    # 'mask'                                             ,
    # 'q'                                                ,
    'seqused_k_'                                       ,
    'k_'                                               ,
    # 'v_'                                               ,
    'kcache_before'                                    ,
    'kcache'                                           ,
    # 'vcache_before'                                    ,
    # 'vcache'                                           ,
    'ref_no_descale'                                   ,
    'mha_fwd_kvcache_output_no_descale'                ,
    # 'ref_no_descale * v_descale_[:, None, None, None]' ,
    # 'ref_descale'                                      ,
    # 'mha_fwd_kvcache_output_descale'                   ,
):
    print(f"{x}\n\n{eval(x)}\n")

width = 80
for x in (
    '(mha_fwd_kvcache_output_no_descale - ref_no_descale).abs().max().item()'       ,
    # '(mha_fwd_kvcache_output_descale - ref_descale).abs().max().item()'             ,
):
    print(f"{x:<{width}}{eval(x)}")
    print()