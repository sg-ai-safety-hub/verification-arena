"""Sloth filler (VARENA_FILLER=sloth) on the GPU == the verifier's CPU reference:
1. kernels.fill over a contiguous range of global pages; 2. kernels.fill_pages over a page list via a page map;
3. memcheck.indep_units / filler_pages; 4. fill rate (the square root, ~30 GB/s expected) and readback = squaring.

  VARENA_FILLER=sloth python schemes/memory_link_exhaustion/tests/test_sloth_filler_gpu.py
"""
import os
import sys
import time

os.environ["VARENA_FILLER"] = "sloth"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import torch

from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.encodings import sloth64 as S
from schemes.memory_link_exhaustion.verifier.memcheck import filler_pages, indep_units

PAGE = ref.PAGE
fkey = 0x5EED1234
ok = True
buf = torch.empty(8 * PAGE, dtype=torch.uint8, device="cuda")
K.fill(buf, 100, fkey)
torch.cuda.synchronize()
w = buf.view(torch.int32).cpu().numpy().view(np.uint32).reshape(8, -1)
e = all(np.array_equal(w[i], S.filler_words(fkey, 100 + i, 0, ref.PAGE_WORDS)) for i in (0, 3, 7))
e2 = np.array_equal(filler_pages(fkey, [103])[0], w[3])
u = indep_units(fkey, [105, 106], [0, 15])
e3 = np.array_equal(u[0], w[5][:1024]) and np.array_equal(u[1], w[6][15 * 1024:])
print(f"fill range == reference: {e}; filler_pages: {e2}; indep_units: {e3}", flush=True)
ok &= e and e2 and e3
# page list through a page map (pages scattered in a buffer)
pages = torch.tensor([7, 2, 40], dtype=torch.int64, device="cuda")
pm = torch.zeros(64, dtype=torch.int64, device="cuda")
pm[7], pm[2], pm[40] = 2 * ref.PAGE_WORDS, 0, 5 * ref.PAGE_WORDS
b2 = torch.zeros(6 * PAGE, dtype=torch.uint8, device="cuda")
K.fill_pages(b2.view(torch.int32), pm, pages, fkey)
torch.cuda.synchronize()
w2 = b2.view(torch.int32).cpu().numpy().view(np.uint32).reshape(6, -1)
e = np.array_equal(w2[2], S.filler_words(fkey, 7, 0, ref.PAGE_WORDS)) and \
    np.array_equal(w2[0], S.filler_words(fkey, 2, 0, ref.PAGE_WORDS)) and \
    np.array_equal(w2[5], S.filler_words(fkey, 40, 0, ref.PAGE_WORDS)) and not w2[1].any()
print(f"fill_pages via page map == reference (other pages untouched): {e}", flush=True)
ok &= e
# rates
big = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
K.fill(big[:PAGE * 16], 0, fkey)
torch.cuda.synchronize()
t = time.perf_counter()
K.fill(big, 0, fkey)
torch.cuda.synchronize()
dt = time.perf_counter() - t
dec = torch.empty_like(big)
S.decode_to(big, dec, S.filler_seed(fkey))
torch.cuda.synchronize()
t = time.perf_counter()
S.decode_to(big, dec, S.filler_seed(fkey))
torch.cuda.synchronize()
dt2 = time.perf_counter() - t
z = not dec.view(torch.int64).any().item()
print(f"fill 4 GiB: {dt * 1e3:.1f} ms = {4 / dt:.1f} GiB/s; decode (squaring) {4 / dt2:.0f} GiB/s; decodes to zero: {z}",
      flush=True)
ok &= z
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
