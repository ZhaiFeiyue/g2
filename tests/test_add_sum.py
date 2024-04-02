import torch
import habana_frameworks.torch.core
import habana_frameworks.torch.core as htcore
import g2attn
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import torch.nn as nn


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


q = torch.rand((2, 2, 1, 128)).to(torch.float32)
q = q.to("hpu")

k = torch.rand((2, 2, 128, 1024)).to(torch.float32)
k = k.to("hpu")
m = Module()
#m = wrap_in_hpu_graph(m)


o = m(q, k)

print(o.dtype)

