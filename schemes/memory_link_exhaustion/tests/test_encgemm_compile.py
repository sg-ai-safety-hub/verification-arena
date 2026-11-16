"""Custom decode op inside torch.compile + CUDA graph capture (like vLLM FULL graphs)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
import torch
import vllm.model_executor.determinism.batch_invariant as bi
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.inference import patches as P
bi.enable_batch_invariant_mode()
cl = P.build_cuda_linear()
Kd = 3584
ws, wes, bs = [], [], []
for i, N in enumerate((4608, 3584, 37888)):
    key = 0x1000 + 77 * i
    w = (torch.randn(N, Kd, device="cuda") * 0.02).to(torch.bfloat16)
    we = torch.empty_like(w); K.xor16_w(w, we, key, Kd)
    P.W_KEYS[we.data_ptr()] = key
    ws.append(w); wes.append(we)

def f(x, enc):
    outs = []
    for w, we in zip(ws, wes):
        y = cl(x, we, None, 0, True) if enc else bi.matmul_persistent(x, w.t())
        outs.append(torch.nn.functional.silu(y.float()).to(torch.bfloat16).sum(-1))
    return torch.stack(outs)

fc = torch.compile(f, fullgraph=False)
bad = 0
for M in (1, 7, 64, 256):
    x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
    for _ in range(2):
        fc(x, True)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fc(x, True)
    for it in range(5):
        x.copy_(torch.randn(M, Kd, device="cuda").to(torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        ref = fc(x, False)
        e = fc(x, True)
        ok_g, ok_e = torch.equal(y, ref), torch.equal(e, ref)
        if not (ok_g and ok_e):
            bad += 1
            print("MISMATCH M", M, "it", it, "graph", ok_g, "eager", ok_e, flush=True)
print("compile+graph test done, bad =", bad)
sys.exit(1 if bad else 0)
