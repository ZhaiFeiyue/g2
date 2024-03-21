import torch

class QKBmm(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k):
        # q.shape = [B, M, QT, H]
        # kv.shape = [B, M, KVT, H]
        B, M, QT, H = q.shape
        _, _, KVT,_ = k.shape
        if QT != 1:
            q = q.reshape([B, M, QT, 1, H])
            k = k.reshape([B, M, 1, KVT, H])

        score = q * k
        score = torch.sum(score, dim=-1, keepdim=True)
        score = score.reshape([B, M, QT, KVT])
        return score


def qk_bmm_test():
    qk_bmm = QKBmm()
    b = 2
    m = 16
    qt = 1
    kvt = 256
    h = 128
    dtype = torch.float32
    Q = torch.randn([b, m, qt, h]).to(dtype)
    K = torch.randn([b, m, kvt, h]).to(dtype)

    score = qk_bmm(Q, K)

    score_gt = torch.matmul(Q, K.permute(0, 1, 3, 2))

    diff = score - score_gt

    print(diff.abs().max())


class SVBmm(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, s, v):
        # s.shape = [B, M, QT, KVT]
        # kv.shape = [B, M, KVT, H]
        B, M, QT, KVT = s.shape
        _, _, _,H = v.shape

        s = s.reshape([B, M, QT, KVT, 1])
        v = v.reshape([B, M, 1, KVT, H])

        out = s * v
        out = torch.sum(out, dim=-2, keepdim=True)
        out = out.reshape([B, M, QT, H])
        return out

def sv_bmm_test():
    qk_bmm = SVBmm()
    b = 2
    m = 16
    qt = 1
    kvt = 256
    h = 128
    # dtype = torch.bfloat16
    dtype = torch.float32

    S = torch.randn([b, m, qt, kvt]).to(dtype)
    V = torch.randn([b, m, kvt, h]).to(dtype)

    score = qk_bmm(S, V)
    score_gt = torch.matmul(S, V)
    diff = score - score_gt
    print(diff.abs().max())


class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = QKBmm()
        self.sv = SVBmm()
    
    def forward(self, q, k, v):
        # s.shape = [B, M, QT, KVT]
        # kv.shape = [B, M, KVT, H]
        score = self.qk(q, k)
        score = torch.softmax(score, dim=-1)
        out = self.sv(score, v)
        return out

def sdpa_test():
    sdpa = SDPA()
    b = 2
    m = 16
    qt = 1
    kvt = 256
    h = 128
    # dtype = torch.bfloat16
    dtype = torch.float32

    Q = torch.randn([b, m, qt, h]).to(dtype)
    K = torch.randn([b, m, kvt, h]).to(dtype)
    V = torch.randn([b, m, kvt, h]).to(dtype)
    out = sdpa(Q, K, V)
    
    S = torch.matmul(Q, K.permute(0, 1, 3, 2))
    S = torch.softmax(S, dim=-1)
    out_gt = torch.matmul(S, V)
    diff = out - out_gt
    print(diff.abs().max())


if __name__ == '__main__':
    qk_bmm_test()
    sv_bmm_test()
    sdpa_test()
