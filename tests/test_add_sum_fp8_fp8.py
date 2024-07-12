import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn

import argparse
from tabulate import tabulate

''' 
f32_type = torch.float32
bf16_type = torch.bfloat16
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2

# collect finfo for each type
table = []
for dtype in [f32_type, bf16_type, e4m3_type, e5m2_type]:
    numbits = 32 if dtype == f32_type else 16 if dtype == bf16_type else 8
    info = torch.finfo(dtype)
    table.append([info.dtype, numbits, info.max, 
                  info.min, info.smallest_normal, info.eps])

headers = ['data type', 'bits', 'max', 'min', 'smallest normal', 'eps']
print(tabulate(table, headers=headers))

x = torch.randn(2, 2, dtype=f32_type)
x_bf16 = x.to(bf16_type)
x_e4m3 = x.to(e4m3_type)
x_e5m2 = x.to(e5m2_type)
print(tabulate([['float32', *x.cpu().flatten().tolist()],
                ['bfloat16', *x_bf16.cpu().flatten().tolist()],
                ['float8_e4m3fn', *x_e4m3.cpu().flatten().tolist()],
                ['float8_e5m2', *x_e5m2.cpu().flatten().tolist()]],
               headers=['data type', 'x[0]', 'x[1]', 'x[2]', 'x[3]']))
'''
 
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

    c_hpu = torch.ops.custom_op.custom_matmul_fp8fp8(q, k)
    print("c_hpu.dtype: ", c_hpu.dtype)
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    refer = torch.matmul(q, k)*scale*scale
    print("refer.dtype: ", refer.dtype)
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    ### fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv=None, Tensor? B_scale_inv=None, Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor
    fp8_hpu = torch.ops.hpu.fp8_gemm_v2(q, False, k, False, None, torch.bfloat16, scale, scale, None, False)
    print("fp8_hpu.dtype: ", fp8_hpu.dtype)
    
    print("q.shape:       ", q.shape)
    print("k.shape:       ", k.shape)
    print("c_hpu.shape:   ", c_hpu.shape)
    print("refer.shape:   ", refer.shape)
    print("fp8_hpu.shape: ", fp8_hpu.shape)
    print("(c_hpu - refer).abs.max = ", (c_hpu - refer.to(c_hpu.dtype)).abs().max())
    print("(fp8_hpu - refer).abs.max = ", (fp8_hpu - refer.to(fp8_hpu.dtype)).abs().max())
    print("refer: ", refer[0,0,0])
    torch.set_printoptions(profile="full")
    print("c_hpu: ", c_hpu[0,0,0])
    torch.set_printoptions(profile="default")
    print("fp8_hpu(bf16): ", fp8_hpu[0,0,0])
    htcore.mark_step()
    htcore.hpu.synchronize()

class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
         return torch.ops.custom_op.custom_matmul_fp8fp8(q, k)

def run_custom_matmul_f8_function():   
    q = (torch.rand((B, M, 1, H))).to(fp8type)
    q = q.to("hpu")
    
    k = (torch.rand((B, M, H, T))).to(fp8type)
    k = k.to("hpu")
    
    tpcMatmul = TpcModule()
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = tpcMatmul(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    return out1
       
def profile_custom_matmul_f8_function(): 
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.HPU)
    s = torch.profiler.schedule(wait=0, warmup=2, active=2, repeat=1)
    r = torch.profiler.tensorboard_trace_handler('./profile/')
    with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
        for i in range(4):
            out = run_custom_matmul_f8_function()
            htcore.mark_step()
            htcore.hpu.synchronize()
            prof.step()

               
test_custom_matmul_f8_function()
#profile_custom_matmul_f8_function()

