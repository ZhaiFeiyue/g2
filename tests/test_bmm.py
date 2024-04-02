import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
#import g2attn


def test_custom_matmul_bf16_function():
    b = 64
    m = 32
    h = 128
    t = 1024

    q = torch.rand((b, m, 1, h)).to(torch.bfloat16)
    q = q.to("hpu")

    k = torch.rand((b, m, t, h)).to(torch.bfloat16)
    k = k.to("hpu")

    c_hpu = q + k
    htcore.mark_step()
    htcore.hpu.synchronize()
    o = c_hpu.sum(-1)
    htcore.mark_step()
    htcore.hpu.synchronize()


activities = []
activities.append(torch.profiler.ProfilerActivity.CPU)
activities.append(torch.profiler.ProfilerActivity.HPU)
s = torch.profiler.schedule(wait=0, warmup=2, active=10, repeat=1)
r = torch.profiler.tensorboard_trace_handler('./profile/')
with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
    for i in range(10):
        test_custom_matmul_bf16_function()
        htcore.mark_step()
        htcore.hpu.synchronize()
        prof.step()
