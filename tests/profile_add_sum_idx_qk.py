import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn
import torch.nn.functional as F

WARMUP = 10
REPEAT = 500
#WARMUP = 10
#REPEAT = 10

B = 32
M = 40
H = 128
T = 2176
index = 1920
    
class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, idx):
         c_hpu = torch.ops.custom_op.custom_matmul_idx_bf16(q, k, idx)
         return c_hpu
    #def forward(self, q, k):
    #     c_hpu = torch.ops.custom_op.custom_matmul_bf16(q, k)
    #     return c_hpu
        
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
    k = k[...,0:index]
    print(k.shape)
    k = F.pad(k, (0, T-index))
    print(k.shape)
    
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
    
    k = (torch.rand((B, M, H, T))).to(torch.bfloat16)
    k = k.to("hpu")
    
    idx = torch.tensor([index])
    idx = idx.to("hpu")
    
    tpcMatmul = TpcModule()
    htcore.mark_step()
    htcore.hpu.synchronize()
    for _ in range(REPEAT):
        out1 = tpcMatmul(q, k, idx)
        #out1 = tpcMatmul(q, k)
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

test_custom_matmul_idx_f16_function()
#profile_custom_matmul_idx_f16_function()

