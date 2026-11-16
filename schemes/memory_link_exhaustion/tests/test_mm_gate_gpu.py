"""VARENA_MM_GATE: the gated copy of vLLM's batch-invariant persistent matmul
1. gives bit-identical results to the stock kernel (gate down), for decode- and prefill-sized M;
2. waits while the gate is raised: a matmul launched with the gate up does not finish until the gate drops;
3. timing: gated (gate down) vs stock, to show the per-tile check is free.

  VARENA=1 VARENA_MM_GATE=1 VLLM_BATCH_INVARIANT=1 VARENA_RESERVE_SMS=2 python schemes/memory_link_exhaustion/tests/test_mm_gate_gpu.py
"""
import os
import sys
import time

os.environ.setdefault("VARENA", "1")
os.environ.setdefault("VARENA_MM_GATE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

import vllm.model_executor.determinism.batch_invariant as bi
stock = bi.matmul_kernel_persistent
from schemes.memory_link_exhaustion.inference import patches as P
from schemes.memory_link_exhaustion.encodings import actenc2 as AE

P._install_mm_gate(bi)
gated = bi.matmul_kernel_persistent
ok = True
torch.manual_seed(0)
for M, N, K in ((16, 7168, 5120), (300, 55296, 5120), (2048, 5120, 27648)):
    a = torch.randn(M, K, device="cuda").bfloat16()
    b = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    bi.matmul_kernel_persistent = stock
    y0 = bi.matmul_persistent(a, b.t())
    bi.matmul_kernel_persistent = gated
    AE.gate_dev()[0] = 0
    y1 = bi.matmul_persistent(a, b.t())
    torch.cuda.synchronize()
    e = torch.equal(y0.view(torch.int16), y1.view(torch.int16))
    print(f"M={M} N={N} K={K}: gated == stock: {e}", flush=True)
    ok &= e
# gate up: the matmul must wait
a = torch.randn(2048, 5120, device="cuda").bfloat16()
b = (torch.randn(55296, 5120, device="cuda") * 0.02).bfloat16()
s = torch.cuda.Stream()
AE.gate_dev()[0] = 1
torch.cuda.synchronize()
ev = torch.cuda.Event()
with torch.cuda.stream(s):
    y = bi.matmul_persistent(a, b.t())
    ev.record()
time.sleep(0.3)
waited = not ev.query()
# drop the gate with a host-to-device copy (copy engine: the matmul's waiting CTAs hold every SM here)
with torch.cuda.stream(torch.cuda.Stream()):
    AE.gate_dev()[0:1].copy_(torch.zeros(1, dtype=torch.int32).pin_memory(), non_blocking=True)
t0 = time.time()
ev.synchronize()
print(f"gate up: matmul still waiting after 0.3 s: {waited}; finished {1e3 * (time.time() - t0):.1f} ms after the drop", flush=True)
ok &= waited


def t(fn, n=20):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for M in (16, 2048):
    a = torch.randn(M, 5120, device="cuda").bfloat16()
    bi.matmul_kernel_persistent = stock
    ts = t(lambda: bi.matmul_persistent(a, b.t()))
    bi.matmul_kernel_persistent = gated
    tg = t(lambda: bi.matmul_persistent(a, b.t()))
    print(f"M={M} gate_up-sized: stock {ts:.3f} ms, gated {tg:.3f} ms", flush=True)
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
