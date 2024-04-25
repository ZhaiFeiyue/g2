import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
         c_hpu = self.fc(x)
         return c_hpu


def test_custom_matmul_f32_function():
    x = torch.rand((1, 4096)).to(torch.bfloat16)
    x = x.to("hpu")

    m = Module()
    m = m.to(torch.bfloat16)
    m.to('hpu')
    w = m.fc.weight.transpose(-2, -1)
    htcore.mark_step()
    htcore.hpu.synchronize()

    c_hpu = torch.ops.custom_op.custom_gemm_bf16(x, w)
    htcore.mark_step()
    htcore.hpu.synchronize()
    refer = m(x)
    print((refer - c_hpu).max().abs())
    htcore.mark_step()
    htcore.hpu.synchronize()



test_custom_matmul_f32_function()

activities = []
activities.append(torch.profiler.ProfilerActivity.CPU)
activities.append(torch.profiler.ProfilerActivity.HPU)
s = torch.profiler.schedule(wait=0, warmup=2, active=5, repeat=1)
r = torch.profiler.tensorboard_trace_handler('./gemm_profile/')
with torch.profiler.profile(activities=activities,schedule=s,on_trace_ready=r, record_shapes=True,with_stack=True) as prof:
    for i in range(10):
        test_custom_matmul_f32_function()
        htcore.mark_step()
        htcore.hpu.synchronize()
        prof.step()

    prof.stop()
