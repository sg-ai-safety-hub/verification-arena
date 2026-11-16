"""Butterfly W (VARENA_WENC=bfly, L=11) on the GPU == the verifier's CPU reference (encodings/bfly_ref.py):

  * encode: ref.w_encode on the GPU equals bfly_ref.encode_range on the CPU, bit for bit, for the whole matrix and for an
    unaligned element range (what memcheck reconstructs for a challenged page);
  * decode: ref.w_xor restores the plaintext exactly, for the whole matrix and for 128-row slices (bfly_xor_rows, used by
    the sliced decode-ahead).

  python -m schemes.memory_link_exhaustion.tests.test_wbfly_gpu
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["VARENA_WENC"] = "bfly"
import numpy as np
import torch

from schemes.memory_link_exhaustion.encodings import bfly_ref, ref

ref.WENC = "bfly"
ok = True
key, ordn = 0x2468ACE1, 7
for (N, K) in ((256, 384), (640, 5120)):
    W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    We = torch.empty_like(W)
    ref.w_encode(W, We, key, ordn)
    raw = W.view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
    enc_gpu = We.view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
    enc_cpu = bfly_ref.encode_range(raw, key, ordn, 0, K)
    e1 = bool(np.array_equal(enc_gpu, enc_cpu))
    j0, n = 3 * K + 77, 2 * K + 5
    e2 = bool(np.array_equal(enc_gpu[j0:j0 + n], bfly_ref.encode_range(raw[j0:j0 + n], key, ordn, j0, K)))
    Wd = torch.empty_like(W)
    ref.w_xor(We, Wd, key, ordn)
    d1 = bool(torch.equal(Wd, W))
    d2 = True
    for r0 in range(0, N, 128):
        part = torch.empty(128, K, dtype=W.dtype, device="cuda")
        ref.w_xor(We[r0:r0 + 128].contiguous(), part, key, ordn, row0=r0)
        d2 &= bool(torch.equal(part, W[r0:r0 + 128]))
    print(f"[{N}x{K}] encode==CPU {e1}, range==CPU {e2}, decode exact {d1}, 128-row slices exact {d2}")
    ok &= e1 and e2 and d1 and d2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
