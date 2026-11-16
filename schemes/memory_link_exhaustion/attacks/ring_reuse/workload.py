"""ring_reuse_idle: a regression cheater that satisfies noise-pool bytes already resident on the GPU by an on-device
(D2D) copy instead of an H2D link copy, freeing that link with no spare memory and still passing every rule (the ring
content is bit-identical -- the same pool bytes). It is a measurement/idle cheater: it does the reuse and reports the
fraction of noise bytes served from the device and the link freed, but runs no theft workload.

The noise pool is `n_pages` 4 KiB pages. Each window is `ppw` contiguous pages at a scheduled base page. The device
noise ring holds the last `n_ring` windows (`n_ring * ppw` pages). A window's pages that are already resident -- in
the ring, or in an optional STATIC pinned cache of `cache_pages` pool pages (the cached variant) -- are served D2D.

Expected, by schedule (`ResidencyReuse` computes it exactly from the offset stream):
  * old schedule (per-window hash over the pool, WITH REPLACEMENT): ring residency ~= ring_bytes / pool_bytes of each
    window is already present (~3% at 2 GiB ring / 64 GiB pool), ~1.7 GB/s freed at 56 GB/s, no cache needed.
  * keyed-permutation schedule (no slot reuse within the ring): ring residency -> 0; only the pinned cache hits, a
    static c-GiB pin harvesting each pinned page's once-per-cycle appearance = c / pool_bytes every cycle, no churn
    (~0.875 GB/s per GiB). Still a bad trade vs spending that memory on the shard under the 0.70 GiB cut (the split
    (0.70-c)(0.45+0.875c) peaks at c ~= 0.09 GiB, ~2%), but not zero.

The accounting here (ResidencyReuse) is pure-Python and CPU-tested. The GPU path is a one-method seam in the verifier
streamer: `copy_window(rs, base)` D2D-copies the resident pages and H2D-copies the rest instead of one H2D of the
whole window; `reuse_copy` below is that body, guarded for torch/cuda. Blue wires it into `Streamer._copy_window`."""
from __future__ import annotations

import hashlib


class ResidencyReuse:
    """Exact page-level reuse accounting for a stream of window base pages. Residency = the pages of the last
    `n_ring` windows (ring) plus a fixed pinned-cache page set. Call `window(base)` per issued window, in order."""

    def __init__(self, n_pages: int, ppw: int, n_ring: int, cache_pages: int = 0, cache_seed: int = 0,
                 page_bytes: int = 4096):
        assert ppw >= 1 and n_ring >= 1 and 0 <= cache_pages <= n_pages
        self.n_pages, self.ppw, self.n_ring, self.pb = n_pages, ppw, n_ring, page_bytes
        self.ring: list[int] = []                 # base page of each resident ring window, oldest first
        self._res: dict[int, int] = {}            # resident pool page -> count of ring windows covering it
        self.cache = self._pin(cache_pages, cache_seed)   # static pinned pool pages (the cached variant)
        self.reused_pages = self.total_pages = 0

    def _pin(self, c: int, seed: int) -> set[int]:
        """A fixed pseudo-random set of c distinct pool pages (a cheater pins the same pages for the whole run)."""
        s: set[int] = set()
        k = 0
        while len(s) < c:
            h = hashlib.sha256(f"{seed}:{k}".encode()).digest()
            s.add(int.from_bytes(h[:8], "big") % self.n_pages)
            k += 1
        return s

    def _add(self, base: int):
        for p in range(base, base + self.ppw):
            self._res[p] = self._res.get(p, 0) + 1
        self.ring.append(base)
        if len(self.ring) > self.n_ring:
            old = self.ring.pop(0)
            for p in range(old, old + self.ppw):
                c = self._res[p] - 1
                if c:
                    self._res[p] = c
                else:
                    del self._res[p]

    def window(self, base: int) -> int:
        """Reuse window i (base page). Returns pages served D2D (resident in ring or pinned cache) before it lands."""
        r = 0
        for p in range(base, base + self.ppw):
            if p in self._res or p in self.cache:
                r += 1
        self.reused_pages += r
        self.total_pages += self.ppw
        self._add(base)                            # the window now occupies its ring slot (evicting the oldest)
        return r

    @property
    def fraction(self) -> float:
        return self.reused_pages / self.total_pages if self.total_pages else 0.0

    def freed_gbs(self, window_rate_hz: float) -> float:
        """Link freed = reused bytes per second, given the honest window rate (windows/s)."""
        per_window_s = window_rate_hz / max(1, self.total_pages // self.ppw)
        return self.reused_pages * self.pb * per_window_s / 1e9

    def describe(self) -> dict:
        return {"reused_pages": self.reused_pages, "total_pages": self.total_pages,
                "reused_fraction": round(self.fraction, 4), "ring_pages": self.n_ring * self.ppw,
                "cache_pages": len(self.cache), "pool_pages": self.n_pages}


def reuse_copy(ring_slot, pool_host, base_page, acct: ResidencyReuse, resident_dev: dict, page_i32: int, cuda_copy):
    """GPU seam for Streamer._copy_window: serve window `base_page` into `ring_slot`, D2D-copying any resident pages
    and H2D-copying the rest, and record the written pages as resident. `resident_dev` maps pool page -> device
    (buffer, offset) for pages currently on the device (ring tail + pinned cache); `cuda_copy(dst, src)` does one copy.
    `page_i32` = page_bytes // 4 (int32 element stride). Returns pages served D2D. Pure accounting lives in `acct`;
    this mirrors it on real tensors so the ring content is bit-identical and the window passes the digest check."""
    served = 0
    for k in range(acct.ppw):
        p = base_page + k
        dst = ring_slot[k * page_i32:(k + 1) * page_i32]
        src_dev = resident_dev.get(p)
        if src_dev is not None:
            cuda_copy(dst, src_dev)                # on-device: no link
            served += 1
        else:
            cuda_copy(dst, pool_host[p * page_i32:(p + 1) * page_i32])   # H2D: the link
    acct.window(base_page)
    return served
