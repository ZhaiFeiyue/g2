import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn
import torch.nn.functional as F
import argparse

WARMUP = 10
REPEAT = 500

parser = argparse.ArgumentParser()
parser.add_argument('--batch','-B',type=int, default=32)
parser.add_argument('--numhead','-M',type=int, default=40)
parser.add_argument('--headdim','-H',type=int, default=128)
parser.add_argument('--seqlength','-T',type=int, default=2176)
parser.add_argument('--index','-i',type=int, default=256)

args = parser.parse_args()

B = args.batch
M = args.numhead
H = args.headdim
T = args.seqlength
index = args.index
Tb = (index + 255) & (-256)
print(f"TPC: [{B} x {M} x 1 x {T}] x [{B} x {M} x {T} x {H}]  index = {index}")
print(f"MME: [{B} x {M} x 1 x {Tb}] x [{B} x {M} x {Tb} x {H}]")

class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, v, idx):
         c_hpu = torch.ops.custom_op.custom_matmul_idx_bf16(s, v, idx)
         return c_hpu
    #def forward(self, s, v):
    #     c_hpu = torch.ops.custom_op.custom_matmul_bf16(s, v)
    #     return c_hpu
        
class MmeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
         return torch.matmul(x, y)
         
def test_custom_matmul_idx_f16_function():    
    s = (torch.rand((B, M, 1, T))).to(torch.bfloat16)
    s = s.to("hpu")

    v = (torch.rand((B, M, T, H))).to(torch.bfloat16)
    v = v.to("hpu")
    
    idx = torch.tensor([index])
    idx = idx.to("hpu")
    
    tpcMatmul = TpcModule().eval()
    tpcMatmul = wrap_in_hpu_graph(tpcMatmul)
    
    for _ in range(WARMUP):
        tpcResult = tpcMatmul(s, v, idx)
    htcore.mark_step()
    htcore.hpu.synchronize()

    start =ht.hpu.Event(enable_timing=True)
    end = ht.hpu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        tpcResult = tpcMatmul(s, v, idx)
    end.record()
    end.synchronize()
    
    print(s.shape)
    s = s[...,0:index]
    print(s.shape)
    s = F.pad(s, (0, T-index))
    print(s.shape)
    print(v.shape)
    v = v[...,0:index,:]
    print(v.shape)
    v = F.pad(v, (0, 0, 0, T-index))
    print(v.shape)
    
    htcore.mark_step()
    htcore.hpu.synchronize()

    refer = torch.matmul(s, v)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    total_time = start.elapsed_time(end)
    time = total_time / REPEAT
    print(f"custom_matmul_idx_bf16: {B} x {M} x {H} x {T} {total_time:.4f} ms")
    print(f"custom_matmul_idx_bf16: per {REPEAT}: {time:.4f} ms")

    torch.set_printoptions(precision=20)
    diff_MME_TPC = (refer - tpcResult).abs()
    maxindex = diff_MME_TPC.argmax()
    print(f'******* diff_MME_TPC: max_diff = {diff_MME_TPC.max()} [ref={refer.view(-1)[maxindex]}, tpc={tpcResult.view(-1)[maxindex]}]')
    print(f'******* diff_MME_TPC: diff.abs.sum = {diff_MME_TPC.sum()}')
    print(f'******* diff_MME_TPC: tpc.sum = {tpcResult.sum()}')
 

def run_custom_matmul_idx_f16_function():   
    s = (torch.rand((B, M, 1, T))).to(torch.bfloat16)
    s = s.to("hpu")
    
    v = (torch.rand((B, M, T, H))).to(torch.bfloat16)
    v = v.to("hpu")
    
    idx = torch.tensor([index])
    idx = idx.to("hpu")
    
    tpcMatmul = TpcModule()
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = tpcMatmul(s, v, idx)
        #out1 = tpcMatmul(s, v)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    return out1
       
def profile_custom_matmul_idx_f16_function(): 
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.HPU)
    s = torch.profiler.schedule(wait=0, warmup=2, active=2, repeat=1)
    r = torch.profiler.tensorboard_trace_handler('./profile/')
    with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
        for i in range(4):
            out = run_custom_matmul_idx_f16_function()
            htcore.mark_step()
            htcore.hpu.synchronize()
            prof.step()


def run_mme_matmul_f16_function():
    s = (torch.rand((B, M, 1, Tb))).to(torch.bfloat16)
    s = s.to("hpu")
    
    v = (torch.rand((B, M, Tb, H))).to(torch.bfloat16)
    v = v.to("hpu")

    mmeMatmul = MmeModule()
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = mmeMatmul(s, v)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    return out1
    
def profile_mme_matmul_f16_function(): 
    activities = []
    activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.HPU)
    s = torch.profiler.schedule(wait=0, warmup=2, active=2, repeat=1)
    r = torch.profiler.tensorboard_trace_handler('./profile/')
    with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
        for i in range(4):
            out = run_mme_matmul_f16_function()
            htcore.mark_step()
            htcore.hpu.synchronize()
            prof.step()
            
test_custom_matmul_idx_f16_function()
#profile_custom_matmul_idx_f16_function()
#profile_mme_matmul_f16_function()

