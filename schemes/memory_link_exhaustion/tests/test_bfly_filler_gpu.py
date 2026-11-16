"""Butterfly filler (VARENA_FILLER=bfly) on the GPU == the verifier's CPU reference (16 Nov):
1. kernels.fill over a contiguous range of global pages (incl. pages above 2^28, where the chunk id's high word is
   used); 2. kernels.fill_pages over a page list via a page map; 3. memcheck.indep_units / filler_pages and the
   batched replay's filler rows; 4. fill rate (restore-from-zeros speed of the filler).

  python -m schemes.memory_link_exhaustion.tests.test_bfly_filler_gpu
"""
import os
import sys
import time

os.environ["VARENA_FILLER"] = "bfly"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import torch

from schemes.memory_link_exhaustion.encodings import kvbfly as KB
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.verifier import chainreplay as CR
from schemes.memory_link_exhaustion.verifier.memcheck import filler_pages, indep_units

PAGE = ref.PAGE
fkey = 0x5EED1234
ok = True
for first in (100, (1 << 28) + 5):
    buf = torch.empty(8 * PAGE, dtype=torch.uint8, device="cuda")
    K.fill(buf, first, fkey)
    torch.cuda.synchronize()
    w = buf.view(torch.int32).cpu().numpy().view(np.uint32).reshape(8, -1)
    e = all(np.array_equal(w[i], KB.filler_page_words(fkey, first + i)[0]) for i in (0, 3, 7))
    e2 = np.array_equal(filler_pages(fkey, [first + 3])[0], w[3])
    u = indep_units(fkey, [first + 5, first + 6], [0, 15])
    e3 = np.array_equal(u[0], w[5][:1024]) and np.array_equal(u[1], w[6][15 * 1024:])
    print(f"first page {first}: fill range == reference: {e}; filler_pages: {e2}; indep_units: {e3}", flush=True)
    ok &= e and e2 and e3
# distinct pages / chunks differ (no repeats from a truncated chunk id)
a = KB.filler_page_words(fkey, [3, 3 + (1 << 16), 3 + (1 << 28)])
d = not (np.array_equal(a[0], a[1]) or np.array_equal(a[0], a[2]) or np.array_equal(a[0][:1024], a[0][1024:2048]))
print(f"pages p, p + 2^16, p + 2^28 and two chunks of one page all differ: {d}")
ok &= d
# batched replay filler rows
src = CR.Sources.__new__(CR.Sources)
src.fkey, src._fcache = fkey, {}
pg = np.array([first + 3, first + 5, first + 3]); wo = np.array([0, 1024 * 7 + 96, 16384 - 32])
r = src._fill32(pg, wo)
e = all(np.array_equal(r[i], KB.filler_page_words(fkey, int(pg[i]))[0][wo[i]:wo[i] + 32]) for i in range(3))
print(f"batched replay filler rows == reference: {e}")
ok &= e
# page list through a page map
pages = torch.tensor([7, 2, 40], dtype=torch.int64, device="cuda")
pm = torch.zeros(64, dtype=torch.int64, device="cuda")
pm[7], pm[2], pm[40] = 2 * ref.PAGE_WORDS, 0, 5 * ref.PAGE_WORDS
b2 = torch.zeros(6 * PAGE, dtype=torch.uint8, device="cuda")
K.fill_pages(b2.view(torch.int32), pm, pages, fkey)
torch.cuda.synchronize()
w2 = b2.view(torch.int32).cpu().numpy().view(np.uint32).reshape(6, -1)
ref_ = KB.filler_page_words(fkey, [7, 2, 40])
e = np.array_equal(w2[2], ref_[0]) and np.array_equal(w2[0], ref_[1]) and np.array_equal(w2[5], ref_[2]) and \
    not w2[1].any()
print(f"fill_pages via page map == reference (other pages untouched): {e}", flush=True)
ok &= e
big = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
K.fill(big[:PAGE * 16], 0, fkey)
torch.cuda.synchronize()
t = time.perf_counter()
K.fill(big, 0, fkey)
torch.cuda.synchronize()
dt = time.perf_counter() - t
print(f"fill 4 GiB: {dt * 1e3:.1f} ms = {4 / dt:.1f} GiB/s (restore-from-zeros rate of the filler)", flush=True)
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
