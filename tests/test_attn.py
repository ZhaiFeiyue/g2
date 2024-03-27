import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn


B = 64
M = 32
QT = 1
H = 128
KVT = 256


class MMEAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k ,v):
        q = q.reshape([B, QT, M, H])
        k = k.reshape([B, KVT, M, H])
        v = v.reshape([B, KVT, M, H])
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 3, 1])
        v = v.permute([0, 2, 1, 3])
        
        score = torch.matmul(q, k)
        score = self.softmax(score)
        out = torch.matmul(score, v)
        return out


class TPCAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k ,v):
        q = q.reshape([B, QT, M, H])
        k = k.reshape([B, KVT, M, H])
        v = v.reshape([B, KVT, M, H])
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 3, 1])
        v = v.permute([0, 2, 1, 3])
        
        score = torch.ops.custom_op.custom_matmul_bf16(q, k)
        score = self.softmax(score)
        out = torch.ops.custom_op.custom_matmul_bf16(score, v)
        return out


def test_custom_matmul_bf16_function():
    q = torch.rand((B, QT, M * H)).to(torch.bfloat16)
    q = q.to("hpu")

    k = torch.rand((B, KVT, M * H)).to(torch.bfloat16)
    k = k.to("hpu")

    v = torch.rand((B, KVT, M * H)).to(torch.bfloat16)
    v = v.to("hpu")

    mmeattn = MMEAttn()
    tpcattn = TPCAttn()
    htcore.mark_step()
    htcore.hpu.synchronize()
    out1 = mmeattn(q, k, v)
    htcore.mark_step()
    htcore.hpu.synchronize()
    out2 = tpcattn(q, k , v)
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
