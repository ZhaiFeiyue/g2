import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn

import argparse

WARMUP = 10
REPEAT = 500

parser = argparse.ArgumentParser()
parser.add_argument('--batch','-B',type=int, default=32)
parser.add_argument('--numhead','-M',type=int, default=40)
parser.add_argument('--headdim','-H',type=int, default=128)
parser.add_argument('--seqlength','-T',type=int, default=2176)

args = parser.parse_args()

B = args.batch
M = args.numhead
H = args.headdim
T = args.seqlength
print(f"[{B} x {M} x 1 x {H}] x [{B} x {M} x {H} x {T}]")


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
         c_hpu = torch.ops.custom_op.custom_matmul_bf16(q, k)
         return c_hpu


def test_custom_matmul_f32_function():
    q = torch.rand((2, 2, 1, 128)).to(torch.float32)
    q = q.to("hpu")

    k = torch.rand((2, 2, 128, 1024)).to(torch.float32)
    k = k.to("hpu")

    c_hpu = torch.ops.custom_op.custom_matmul_fp32(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    refer = torch.matmul(q, k)
    print((refer - c_hpu).max().abs())
    htcore.mark_step()
    htcore.hpu.synchronize()


def test_custom_matmul_f16_function():
    q = torch.rand((B, M, 1, H)).to(torch.bfloat16)
    q = q.to("hpu")

    k = torch.rand((B, M, H, T)).to(torch.bfloat16)
    k = k.to("hpu")

    c_hpu = torch.ops.custom_op.custom_matmul_bf16(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    refer = torch.matmul(q, k)
    print(q.shape)
    print(k.shape)
    print(refer.shape)
    print((refer - c_hpu).abs().max())
    htcore.mark_step()
    htcore.hpu.synchronize()

class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
         return torch.ops.custom_op.custom_matmul_bf16(q, k)

def run_custom_matmul_f16_function():   
    q = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    q = q.to("hpu")
    
    k = (torch.rand((B, M, H, T))).to(torch.bfloat16)
    k = k.to("hpu")
    
    tpcMatmul = TpcModule()
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = tpcMatmul(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    return out1
       
def profile_custom_matmul_f16_function(): 
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.HPU)
    s = torch.profiler.schedule(wait=0, warmup=2, active=2, repeat=1)
    r = torch.profiler.tensorboard_trace_handler('./profile/')
    with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
        for i in range(4):
            out = run_custom_matmul_f16_function()
            htcore.mark_step()
            htcore.hpu.synchronize()
            prof.step()

test_custom_matmul_f16_function()
#profile_custom_matmul_f16_function()

'''
q = torch.rand((2, 2, 1, 128)).to(torch.float32)
q = q.to("hpu")

k = torch.rand((2, 2, 128, 1024)).to(torch.float32)
k = k.to("hpu")
m = Module()
#m = wrap_in_hpu_graph(m)


o = m(q, k)

print(o.dtype)
print(o)
print(o.shape)
'''
