"""Butterfly sources of the batched chain replay (CPU, 16 Nov): the per-word W keystream (chainreplay.Sources._wks64)
equals bfly_ref.encode_range of zeros, and the per-unit KV keystream the verifier hands the replay
(Verifier._kvbf_words) equals kvbfly.encode_page_u16 of a zero page.

  python -m schemes.memory_link_exhaustion.tests.test_replay_bfly_cpu
"""
import os
import sys
import types

os.environ.setdefault("VARENA_WENC", "bfly")
os.environ.setdefault("VARENA_KVENC", "bfly")
os.environ.setdefault("VARENA_FILLER", "sloth")
os.environ.setdefault("VARENA_W_BFLY_L", "11")
os.environ.setdefault("VARENA_KV_HEADS", "8")              # Qwen2.5-32B
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np

from schemes.memory_link_exhaustion.encodings import bfly_ref, kvbfly, ref
from schemes.memory_link_exhaustion.verifier import chainreplay as CR
from schemes.memory_link_exhaustion.verifier import verifier as V

PAGE = 65536
ok = True
# W: one [384, 640] matrix (3 x 5 super-chunks) at ordinal-bearing offset
N, C, KEY = 384, 640, 0x13579BDF
off = 3 * 2**20
ent = types.SimpleNamespace(offset=off, nbytes=N * C * 2, key=KEY, shape=(N, C), sources=["x"], name="w")
src = CR.Sources.__new__(CR.Sources)
src.ents = [ent]
zero = np.zeros(N * C, np.uint16)
want = bfly_ref.encode_range(zero, KEY, ref.w_ordinal(off), 0, C).view(np.uint64)
rng = np.random.default_rng(1)
wi = np.unique(rng.integers(0, N * C // 4, 4000))
got = src._wks64(0, wi)
r = bool((got == want[wi]).all())
print("W per-word keystream == encode_range(0):", r)
ok &= r
# KV: units of a zero page, 8 kv heads
v = V.Verifier.__new__(V.Verifier)
v.info = {"kv_shape": [None, kvbfly.NHEADS]}
v.master = 0xABCDEF
for layer, block in ((0, 0), (5, 123), (63, 10239)):
    key = V.kv_key(v.master, layer)
    page = kvbfly.encode_page_u16(np.zeros(PAGE // 2, np.uint16), key, block).view(np.uint32)
    for wo in rng.integers(0, PAGE // 4 // 32, 64) * 32:
        g = v._kvbf_words(layer, block, int(wo))
        r = bool((g == page[wo:wo + 32]).all())
        if not r:
            print("KV mismatch", layer, block, wo)
        ok &= r
print("KV per-unit keystream == encode_page_u16(0):", ok)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
