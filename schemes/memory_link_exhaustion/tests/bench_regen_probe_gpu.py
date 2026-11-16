"""Regenerate-on-probe cost, butterfly W vs sloth W, kernel time only. The
regenerate-on-probe cheater was removed from the cheater module; its chase kernel lives on, benchmark only, in
gpu/cuda/regen_probe_torch.cu (build.load_regen).

  1. regenerate on probe: the cheater's chase kernel (chase_cheat) at one answer's shape (K 4224, P 1200, 8 probes),
     with a fraction f of the arena's pages being zero-weight W pages regenerated per probe: butterfly W through its
     dependency cone (USE 8, bfly_w) or sloth W through one square root per word (USE 32), against all resident;
  2. restore before a challenge: re-encode X GiB of W from zeros in bulk (bfly_xor_to / sloth64.encode_to), the time a
     rewrite cheater pays per challenge to borrow X GiB between challenges.

  [REGEN_FRACS=0.001,0.01,...] python -m schemes.memory_link_exhaustion.tests.bench_regen_probe_gpu [arena GiB, default 8]
"""
import os
import statistics as st
import sys

os.environ.setdefault("VARENA_FILLER", "indep")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

from schemes.memory_link_exhaustion.encodings import ref, sloth64
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.gpu.cuda import build as CB

ext = CB.load()
rg = CB.load_regen()
CK_W = 3                 # page kind: zero-weight W page, regenerated per probe (regen_probe_torch.cu)
gib = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
unit = ref.CH_GRAN_PAGES * ref.CH_GRAN_PER_BLOCK
npg = int(gib * 2**30) // 65536 // unit * unit
base = torch.empty(npg * 16384, dtype=torch.int32, device="cuda")
K.fill(base.view(torch.uint8), 0, 777)
pm = torch.arange(npg, dtype=torch.int64, device="cuda") * 16384
gp = ref.ch_granules(0, 0, npg)
tb = len(gp) // ref.CH_GRAN_PER_BLOCK
gpt = torch.tensor(gp, dtype=torch.int64, device="cuda")
nonce = torch.tensor([12345], dtype=torch.int32).pin_memory()
Kc, P = 4224, 1200
nb = min(11, tb)
out = torch.empty(Kc, dtype=torch.int64, device="cuda")
NCOLS, KEY = 5120, 0x13579BDF
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)


def chase(kind, a0, a1, aord, use, bw, n=15):
    ts = []
    for i in range(n + 3):
        nonce[0] = 2000 + i
        e0.record()
        rg.chase_cheat(gpt, tb, nb, nonce.data_ptr(), Kc, P, out, base, kind, a0, a1, 777, ref.INDEP_HF // 8, aord, bw, use)
        e1.record()
        e1.synchronize()
        if i >= 3:
            ts.append(e0.elapsed_time(e1))
    return st.median(ts)


kind0 = torch.zeros(npg, dtype=torch.int8, device="cuda")
z = torch.zeros(npg, dtype=torch.int64, device="cuda")
zo = torch.zeros(npg, dtype=torch.int32, device="cuda")
b0 = chase(kind0, pm, z, zo, 2, 0)
print(f"arena {npg * 65536 / 2**30:.1f} GiB, {Kc * P * 8 / 1e6:.1f} M probes per answer; all resident: {b0:.4f} ms")
g = torch.Generator(device="cuda").manual_seed(7)
FRACS = [float(x) for x in os.environ.get("REGEN_FRACS", "0.001,0.003,0.01,0.03,0.1,1.0").split(",")]
for f in FRACS:
    sel = torch.rand(npg, device="cuda", generator=g) < f
    kind = kind0.clone()
    kind[sel] = CK_W
    pidx = torch.arange(npg, dtype=torch.int64, device="cuda")
    # a W page p: element offset within a [*, NCOLS] matrix (one matrix per 1024 pages), key, ordinal
    a0 = torch.where(sel, (pidx % 1024) * 32768, pm)
    a1 = torch.where(sel, torch.full_like(pidx, (NCOLS << 32) | KEY), z)
    aord = (pidx // 1024).to(torch.int32)
    tb_ = chase(kind, a0, a1, aord, 8, 1)          # butterfly W regenerated through its cone
    ts_ = chase(kind, a0, a1, aord, 32, 0)         # sloth W regenerated: one square root per word
    print(f"regenerate {f:6.1%} of pages: butterfly {tb_:9.3f} ms (+{tb_ - b0:9.3f}), sloth {ts_:9.3f} ms (+{ts_ - b0:9.3f})")


def bulk(fn, n=10):
    ts = []
    for i in range(n + 2):
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        if i >= 2:
            ts.append(e0.elapsed_time(e1))
    return st.median(ts)


for gb in (0.25, 0.5, 1.0, 2.0):
    rows = int(gb * 2**30) // (NCOLS * 2) // 128 * 128
    w = torch.zeros(rows, NCOLS, dtype=torch.bfloat16, device="cuda")
    tbf = bulk(lambda: ext.bfly_xor_to(w, w, KEY, 3))
    tsl = bulk(lambda: sloth64.encode_to(w, w, 0x13579BDF00000003))
    gbs = rows * NCOLS * 2 / 1e9
    print(f"restore {gb:4} GiB from zeros: butterfly {tbf:8.3f} ms ({gbs / tbf * 1e3:7.0f} GB/s), "
          f"sloth {tsl:8.3f} ms ({gbs / tsl * 1e3:7.0f} GB/s)")
    del w
    torch.cuda.empty_cache()
