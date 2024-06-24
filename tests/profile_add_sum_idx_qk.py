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
parser.add_argument('--batch',     '-B',type=int, default=32)
parser.add_argument('--numhead',   '-M',type=int, default=40)
parser.add_argument('--headdim',   '-H',type=int, default=128)
parser.add_argument('--seqlength', '-T',type=int, default=2176)
parser.add_argument('--index',     '-i',type=int, default=256)

args = parser.parse_args()

B = args.batch
M = args.numhead
H = args.headdim
T = args.seqlength
index = args.index
Tb = (T + 255) & (-256)
print(f"TPC: [{B} x {M} x 1 x {H}] x [{B} x {M} x {H} x {T}]  index = {index}")
print(f"MME: [{B} x {M} x 1 x {H}] x [{B} x {M} x {H} x {Tb}]")

    
class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, idx):
         return torch.ops.custom_op.custom_matmul_idx_bf16(q, k, idx)

        
class MmeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
         return torch.matmul(x, y)
         
def test_custom_matmul_idx_f16_function():
    q = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    q = q.to("hpu")

    k = (torch.rand((B, M, H, T))).to(torch.bfloat16)
    k = k.to("hpu")
    
    idx = torch.tensor([index])
    idx = idx.to("hpu")

    tpcMatmul = TpcModule().eval()
    tpcMatmul = wrap_in_hpu_graph(tpcMatmul)
    
    for _ in range(WARMUP):
        tpcResult = tpcMatmul(q, k, idx)
    htcore.mark_step()
    htcore.hpu.synchronize()

    start =ht.hpu.Event(enable_timing=True)
    end = ht.hpu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        tpcResult = tpcMatmul(q, k, idx)
    end.record()
    end.synchronize()
    
    print(q.shape)
    print(k.shape)
    #k = k[...,0:index]
    #print(k.shape)
    #k = F.pad(k, (0, T-index))
    #print(k.shape)
    
    htcore.mark_step()
    htcore.hpu.synchronize()

    refer = torch.matmul(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    
    total_time = start.elapsed_time(end)
    time = total_time / REPEAT
    print(f"custom_matmul_idx_bf16: {B} x {M} x {H} x {T} {total_time:.4f} ms")
    print(f"custom_matmul_idx_bf16: per {REPEAT}: {time:.4f} ms")

    torch.set_printoptions(precision=20)
    print(refer.shape)
    refer = refer[...,0:index]
    print(refer.shape)
    refer = F.pad(refer, (0, T-index))
    print(refer.shape)
    print(tpcResult.shape)
    tpcResult = tpcResult[...,0:index]
    print(tpcResult.shape)
    tpcResult = F.pad(tpcResult, (0, T-index))
    print(tpcResult.shape)
    diff_MME_TPC = (refer - tpcResult).abs()
    maxindex = diff_MME_TPC.argmax()
    print(f'******* diff_MME_TPC: max_diff = {diff_MME_TPC.max()} [ref={refer.view(-1)[maxindex]}, tpc={tpcResult.view(-1)[maxindex]}]')
    print(f'******* diff_MME_TPC: diff.abs.sum = {diff_MME_TPC.sum()}')
    print(f'******* diff_MME_TPC: tpc.sum = {tpcResult.sum()}')
    #torch.set_printoptions(profile="full")
    #print(refer[0,0,:])
    #print(tpcResult[0,0,:])
    #torch.set_printoptions(profile="default")

   
def run_custom_matmul_idx_f16_function():   
    q = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    q = q.to("hpu")
    #q2 = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    #q2 = q2.to("hpu")
    
    k = (torch.rand((B, M, H, T))).to(torch.bfloat16)
    k = k.to("hpu")
    
    idx = torch.tensor([index])
    idx = idx.to("hpu")
    
    tpcMatmul = TpcModule()
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = tpcMatmul(q, k, idx)
        #out2 = tpcMatmul(q2, k, idx)
        #out1 = out1 + out2
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
    q = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    q = q.to("hpu")
    
    k = (torch.rand((B, M, H, Tb))).to(torch.bfloat16)
    k = k.to("hpu")

    mmeMatmul = MmeModule()
    
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = mmeMatmul(q, k)
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

