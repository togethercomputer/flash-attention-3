import torch
from flashattn_hopper_cuda import fwd
from test_flash_attn import attention_ref, attention_ref_new
# from flash_attn_interface import flash_attn_func

seed = 123
torch.manual_seed(seed)

dtype         = torch.float8_e4m3fn
dtype_ref     = torch.bfloat16
dtype_descale = torch.float32
device        = 'cuda'

batch_size        = 2
num_heads_q       = 1
num_heads_k       = num_heads_q
seqlen_q          = 2
seqlen_k          = seqlen_q
head_size         = 2

q_ref                 = torch.randn((batch_size, seqlen_q, num_heads_q, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
k_ref                 = torch.randn((batch_size, seqlen_k, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
v_ref                 = torch.randn((batch_size, seqlen_k, num_heads_k, head_size), dtype = dtype_ref    , device = device).to(dtype).to(dtype_ref)
q                     = q_ref.to(dtype)
k                     = k_ref.to(dtype)
v                     = v_ref.to(dtype)
out_                  = None
softmax_scale         = q.shape[-1] ** (-0.5)
is_causal             = False

q_descale_trivial     = torch.ones(batch_size                                     , dtype = dtype_descale, device = device)
k_descale_trivial     = torch.ones(batch_size                                     , dtype = dtype_descale, device = device)
v_descale_trivial     = torch.ones(batch_size                                     , dtype = dtype_descale, device = device)

q_descale_trivial_0   = q_descale_trivial[0]
k_descale_trivial_0   = k_descale_trivial[0]
v_descale_trivial_0   = v_descale_trivial[0]

q_descale_            = torch.rand(batch_size,                                      dtype = dtype_descale, device = device) * 2
k_descale_            = torch.rand(batch_size,                                      dtype = dtype_descale, device = device) * 2
v_descale_            = torch.rand(batch_size                                     , dtype = dtype_descale, device = device) * 2

q_descale_0           = q_descale_[0].unsqueeze(0)
k_descale_0           = k_descale_[0].unsqueeze(0)
v_descale_0           = v_descale_[0].unsqueeze(0)

window_size_left      = -1
window_size_right     = -1
window_size           = (window_size_left, window_size_right)
sink_token_length     = 0   # Don't understand this one, 0 is the default in eg interface/flash_attn_func
softcap               = 0.0
num_splits            = 1
pack_gqa_             = False

(
    mha_fwd_output_no_descale     ,
    q_padded                      ,
    k_padded                      ,
    v_padded                      ,
    out_padded                    ,
    softmax_lse                   ,
) = fwd( 
    q                             ,
    k                             ,
    v                             ,
    out_                          ,
    softmax_scale                 ,
    is_causal                     ,
    None                          ,
    None                          ,
    None                          ,
    window_size_left              ,
    window_size_right             ,
    sink_token_length             ,
    softcap                       ,
    num_splits                    ,
    pack_gqa_                     ,
) 
 
( 
    mha_fwd_output_with_descale   ,
    q_padded                      ,
    k_padded                      ,
    v_padded                      ,
    out_padded                    ,
    softmax_lse                   ,
) = fwd( 
    q                             ,
    k                             ,
    v                             ,
    out_                          ,
    softmax_scale                 ,
    is_causal                     ,
    q_descale_                    ,
    k_descale_                    ,
    v_descale_                    ,
    window_size_left              ,
    window_size_right             ,
    sink_token_length             ,
    softcap                       ,
    num_splits                    ,
    pack_gqa_                     ,
)

# out, lse = flash_attn_func(
#     q,
#     k,
#     v,
#     causal=is_causal,
#     q_descale=q_descale_, k_descale=k_descale_, v_descale=v_descale_,
#     window_size=window_size,
#     sink_token_length=sink_token_length,
#     softcap=softcap,
# )
ref_no_descale  , attn_ref       = attention_ref   (
    q_ref,
    k_ref,
    v_ref,
    None,
    None,
    causal=is_causal,
    q_descale=None      , k_descale=None      , v_descale=None,
    window_size=window_size,
    sink_token_length=sink_token_length,
    softcap=softcap
)
ref_with_descale, attn_ref_new = attention_ref_new (
    q_ref,
    k_ref,
    v_ref,
    None,
    None,
    causal=is_causal,
    q_descale=q_descale_, k_descale=k_descale_ , v_descale=v_descale_,
    window_size=window_size,
    sink_token_length=sink_token_length,
    softcap=softcap
)

# print(f"{q_descale_ = }")
# print(f"{k_descale_ = }")
print(  "v_descale_")
print(f"{v_descale_    }")
print()
print(  "mha_fwd_output_no_descale")
print(f"{mha_fwd_output_no_descale}")
print()
print(  "ref_no_descale")
print(f"{ref_no_descale}")
print()
print(  "mha_fwd_output_with_descale")
print(f"{mha_fwd_output_with_descale}")
print()
print(  "ref_with_descale")
print(f"{ref_with_descale}")
print() 
print(f"{(mha_fwd_output_no_descale - ref_no_descale).abs().max().item() = }")
print()
print(f"{(mha_fwd_output_with_descale - ref_with_descale).abs().max().item() = }")