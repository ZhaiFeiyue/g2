import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn


def test_custom_matmul_bf16_function():
    q = torch.rand((2, 2, 1, 128)).to(torch.bfloat16)
    q = q.to("hpu")

    k = torch.rand((2, 2, 128, 1024)).to(torch.bfloat16)
    k = k.to("hpu")

    c_hpu = torch.ops.custom_op.custom_matmul_bf16(q, k)
    htcore.mark_step()
    htcore.hpu.synchronize()
    refer = torch.matmul(q, k)
    print((refer - c_hpu).max().abs())
    htcore.mark_step()
    htcore.hpu.synchronize()

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

test_custom_matmul_bf16_function()
test_custom_matmul_f32_function()

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