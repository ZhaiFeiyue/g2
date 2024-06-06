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

B = 32
M = 40
H = 2176
T = 128
index = 1
    
class TpcModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, v, idx):
         c_hpu = torch.ops.custom_op.custom_matmul_idx_bf16(s, v, idx)
         return c_hpu
    #def forward(self, s, v):
    #     c_hpu = torch.ops.custom_op.custom_matmul_bf16(s, v)
    #     return c_hpu
        
def test_custom_matmul_idx_f16_function():    
    s = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    s = s.to("hpu")

    v = (torch.rand((B, M, H, T))).to(torch.bfloat16)
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
    s = F.pad(s, (0, H-index))
    print(s.shape)
    print(v.shape)
    v = v[...,0:index,:]
    print(v.shape)
    v = F.pad(v, (0, 0, 0, H-index))
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
    print(f'******* diff_MME_TPC = {diff_MME_TPC.max()} {refer.view(-1)[maxindex]} {tpcResult.view(-1)[maxindex]}')

   
test_custom_matmul_idx_f16_function()

 
    
def run_custom_matmul_idx_f16_function():   
    s = (torch.rand((B, M, 1, H))).to(torch.bfloat16)
    s = s.to("hpu")
    
    v = (torch.rand((B, M, H, T))).to(torch.bfloat16)
    v = v.to("hpu")
    
    idx = torch.tensor([2048])
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

#profile_custom_matmul_idx_f16_function()

