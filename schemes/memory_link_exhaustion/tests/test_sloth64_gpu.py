"""Asymmetric word encoding (schemes.memory_link_exhaustion.encodings.sloth64) on the GPU: bit-exact round trip, equality with the Python-int
reference (the verifier's), and encode / decode rates (GB/s) on 1 GiB.

  python schemes/memory_link_exhaustion/tests/test_sloth64_gpu.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import torch

from schemes.memory_link_exhaustion.encodings import sloth64 as S

ok = True
# Python reference self-consistency (bijection incl. edge values)
rng = random.Random(0)
vals = [0, 1, 2, S.P - 1, S.P - 2, S.P, S.M64, 1 << 63] + [rng.getrandbits(64) for _ in range(2000)]
seed = 0x1234_5678_9ABC_DEF0
for j, x in enumerate(vals):
    y = S.encode_word(x, seed, j)
    if S.decode_word(y, seed, j) != x:
        ok = False
        print("ref round trip FAIL", j, hex(x))
        break
print("python reference round trip:", ok, flush=True)

dev = "cuda"
n = len(vals)
x = torch.tensor([v - (1 << 64) if v >= 1 << 63 else v for v in vals], dtype=torch.int64, device=dev)
y = torch.empty_like(x)
S.encode_to(x, y, seed)
x2 = torch.empty_like(x)
S.decode_to(y, x2, seed)
torch.cuda.synchronize()
yh = [v & S.M64 for v in y.cpu().tolist()]
ref = [S.encode_word(v, seed, j) for j, v in enumerate(vals)]
eq_enc = yh == ref
eq_rt = torch.equal(x, x2)
print(f"GPU encode == reference: {eq_enc}; GPU round trip: {eq_rt}", flush=True)
if not eq_enc:
    bad = [j for j in range(n) if yh[j] != ref[j]][:5]
    print("first mismatches", [(j, hex(vals[j]), hex(yh[j]), hex(ref[j])) for j in bad])
ok &= eq_enc and eq_rt

# zeros (the cheater's plaintext): encoded zeros differ per address and decode back
z = torch.zeros(1 << 16, dtype=torch.int64, device=dev)
ez = torch.empty_like(z)
S.encode_to(z, ez, seed, i0=12345)
dz = torch.empty_like(z)
S.decode_to(ez, dz, seed, i0=12345)
torch.cuda.synchronize()
print(f"zeros: decode(encode(0)) == 0: {bool((dz == 0).all())}, distinct encodings: {ez.unique().numel()} of {z.numel()}")
ok &= bool((dz == 0).all())


def rate(fn, nbytes, reps):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return nbytes * reps / (time.perf_counter() - t) / 1e9


big = torch.randint(-(1 << 62), 1 << 62, (1 << 27,), dtype=torch.int64, device=dev)     # 1 GiB
out = torch.empty_like(big)
nb = big.numel() * 8
for blk in (256, 1024):
    print(f"decode (block {blk}): {rate(lambda: S.decode_to(big, out, seed, block=blk), nb, 10):.0f} GB/s", flush=True)
for blk in (128, 256):
    print(f"encode (block {blk}): {rate(lambda: S.encode_to(big, out, seed, block=blk), nb, 2):.1f} GB/s", flush=True)
cp = rate(lambda: out.copy_(big), nb, 10)
print(f"plain copy: {cp:.0f} GB/s")
# verifier reference on a bf16 matrix range == GPU encode (the prover's load path), via ref.w_encode / w_xor
import numpy as np
os.environ["VARENA_WENC"] = "sloth"
from schemes.memory_link_exhaustion.encodings import ref
ref.WENC = "sloth"
W = (torch.randn(256, 512, device=dev) * 0.02).bfloat16()
We = torch.empty_like(W)
ref.w_encode(W, We, 0x1234567, 77)
Wd = torch.empty_like(W)
ref.w_xor(We, Wd, 0x1234567, 77)
raw = W.view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
encg = We.view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
j0 = 512 * 37 + 64
encr = S.encode_range(raw[j0:j0 + 256], ref.sloth_seed(0x1234567, 77), j0)
okm = torch.equal(W.view(torch.int16), Wd.view(torch.int16)) and np.array_equal(encr, encg[j0:j0 + 256])
print(f"matrix: decode(encode(W)) == W and verifier encode_range == GPU encode: {okm}")
ok &= okm
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
