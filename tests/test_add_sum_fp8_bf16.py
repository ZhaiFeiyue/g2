import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn

import argparse
from tabulate import tabulate
 
WARMUP = 10
REPEAT = 500

parser = argparse.ArgumentParser()
parser.add_argument('--batch','-B',type=int, default=32)
parser.add_argument('--numhead','-M',type=int, default=40)
parser.add_argument('--headdim','-H',type=int, default=128)
parser.add_argument('--seqlength','-T',type=int, default=2048)

args = parser.parse_args()

B = args.batch
M = args.numhead
H = args.headdim
T = args.seqlength
print(f"[{B} x {M} x 1 x {H}] x [{B} x {M} x {H} x {T}]")
fp8type =torch.float8_e4m3fn
#fp8type =torch.float8_e5m2

def test_custom_matmul_f8_function():
    #e4m3_type = torch.float8_e4m3fn
    #e5m2_type = torch.float8_e5m2
    #typedef _Float8_143 minifloat;
    #typedef _Float8_152 minihalf;
    
    torch.manual_seed(1017)
    
    q = torch.rand((B, M, 1, H)).to(fp8type)
    q = q.to("hpu")

    k = torch.rand((B, M, H, T)).to(fp8type)
    k = k.to("hpu")
    
    print("q.dtype: ", q.dtype)
    print("k.dtype: ", k.dtype)
    
    if H == 128:
        scale = 1.0
    else:
        scale = 1/16.0

    sc = torch.tensor([scale])
    sc = sc.to(torch.bfloat16).to("hpu")
    c_hpu_scale = torch.ops.custom_op.custom_matmul_fp8bf16scale(q, k, sc, sc)
    htcore.mark_step()
    htcore.hpu.synchronize()
    print("c_hpu_scale.dtype: ", c_hpu_scale.dtype)
    
    ### fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv=None, Tensor? B_scale_inv=None, Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor
    fp8_hpu = torch.ops.hpu.fp8_gemm_v2(q, False, k, False, None, torch.bfloat16, scale, scale, None, False)
    print("fp8_hpu.dtype:     ", fp8_hpu.dtype)
    
    htcore.hpu.synchronize()
    refer_bf16 = torch.matmul(q.to(torch.bfloat16).to("cpu"), k.to(torch.bfloat16).to("cpu"))*scale*scale
    
    print("q.shape:         ", q.shape)
    print("k.shape:         ", k.shape)
    print("c_hpu_scale.shape:     ", c_hpu_scale.shape)
    print("fp8_hpu.shape:         ", fp8_hpu.shape)

    print("(c_hpu_scale - refer_bf16).abs.max = ", (c_hpu_scale - refer_bf16).abs().max())
    print("(c_hpu_scale - fp8_hpu).abs.max    = ", (fp8_hpu - c_hpu_scale).abs().max())
    
    #torch.set_printoptions(profile="full")
    #print("refer_bf16: ", refer_bf16[0,0,0])
    print("fp8_hpu:       ", fp8_hpu[0,0,0])
    print("c_hpu_scale:   ", c_hpu_scale[0,0,0])
    #torch.set_printoptions(profile="default") # reset
    htcore.mark_step()
    htcore.hpu.synchronize()

test_custom_matmul_f8_function()

