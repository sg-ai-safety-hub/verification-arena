"""Verifier-side expected memory: what any arena page must contain.

W pages: filler overlaid with encode(declared weights).  F pages: filler.
K pages: depend on the KV state, so they are checked later by replay.
"""

from __future__ import annotations

import numpy as np
import torch

from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.core.arena import PAGE, Layout

try:
    import numba as nb

    @nb.njit(cache=True, inline="always")
    def fm(x):
        x = np.uint32(x * np.uint32(0x85EBCA6B))
        x = np.uint32(x ^ (x >> np.uint32(13)))
        x = np.uint32(x * np.uint32(0xC2B2AE35))
        x = np.uint32(x ^ (x >> np.uint32(16)))
        return x

    @nb.njit(cache=True, parallel=True)
    def _filler_nb(fkey, pages, rounds, out):
        for i in nb.prange(pages.size):
            p = np.uint64(pages[i])
            lo = np.uint32(p & np.uint64(0xFFFFFFFF))
            hi = np.uint32(p >> np.uint64(32))
            s = fm(np.uint32(fm(np.uint32(fkey) ^ lo) ^ hi ^ np.uint32(0x9E3779B9)))
            for j in range(16384):
                for _ in range(rounds):
                    s = fm(np.uint32(s + np.uint32(j)))
                out[i, j] = s

    @nb.njit(parallel=True, cache=True)
    def _indep_nb(fkey, k1, hf8, chunks, out):
        """out[r, 0:32] = INDEP chunk chunks[r] (see ref.indep_words)."""
        M0, M1, W0, W1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57), np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
        msk = np.uint64(0xFFFFFFFF)
        for r in nb.prange(chunks.size):
            q = np.uint64(chunks[r])
            qlo, qhi = q & msk, q >> np.uint64(32)
            for j in range(8):
                s0 = s1 = s2 = s3 = np.uint64(0)
                for i in range(hf8):
                    c0, c1, c2, c3 = qlo, qhi, np.uint64(8 * i + j), np.uint64(0)
                    a, b = np.uint64(fkey), np.uint64(k1)
                    for _ in range(8):
                        p0 = c0 * M0
                        p1 = c2 * M1
                        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ a) & msk, p1 & msk, ((p0 >> np.uint64(32)) ^ c3 ^ b) & msk, p0 & msk
                        a = (a + W0) & msk
                        b = (b + W1) & msk
                    s0 += c0
                    s1 += c1
                    s2 += c2
                    s3 += c3
                out[r, 4 * j] = np.uint32(s0 & msk)
                out[r, 4 * j + 1] = np.uint32(s1 & msk)
                out[r, 4 * j + 2] = np.uint32(s2 & msk)
                out[r, 4 * j + 3] = np.uint32(s3 & msk)

    def indep_units(fkey, pages, subs):
        """INDEP filler of 4 KiB units (page, sub). Returns uint32 [n, 1024]."""
        if ref.FILLER == "sloth":
            from schemes.memory_link_exhaustion.encodings import sloth64
            return np.stack([sloth64.filler_words(fkey, int(p), int(sb) * 1024, 1024) for p, sb in zip(pages, subs)])
        if ref.FILLER == "bfly":
            from schemes.memory_link_exhaustion.encodings import kvbfly
            return kvbfly.filler_units(fkey, pages, subs)
        pages = np.asarray(pages, dtype=np.uint64)
        subs = np.asarray(subs, dtype=np.uint64)
        ch = (pages[:, None] * np.uint64(512) + subs[:, None] * np.uint64(32) +
              np.arange(32, dtype=np.uint64)[None, :]).reshape(-1)
        out = np.empty((ch.size, 32), dtype=np.uint32)
        _indep_nb(np.uint64(fkey), np.uint64(ref.INDEP_K1), ref.INDEP_HF // 8, ch, out)
        return out.reshape(pages.size, 1024)

    def filler_pages(fkey, pages, rounds=ref.FILLER_ROUNDS):
        if ref.FILLER == "sloth":
            from schemes.memory_link_exhaustion.encodings import sloth64
            pages = np.atleast_1d(np.asarray(pages, dtype=np.uint64))
            return np.stack([sloth64.filler_words(fkey, int(p), 0, ref.PAGE_WORDS) for p in pages])
        if ref.FILLER == "bfly":
            from schemes.memory_link_exhaustion.encodings import kvbfly
            return kvbfly.filler_page_words(fkey, pages)
        if ref.FILLER == "indep":
            pages = np.atleast_1d(np.asarray(pages, dtype=np.uint64))
            subs = np.tile(np.arange(16, dtype=np.uint64), pages.size)
            return indep_units(fkey, np.repeat(pages, 16), subs).reshape(pages.size, ref.PAGE_WORDS)
        pages = np.atleast_1d(np.asarray(pages, dtype=np.uint64))
        out = np.empty((pages.size, ref.PAGE_WORDS), dtype=np.uint32)
        _filler_nb(np.uint32(fkey), pages, rounds, out)
        return out
except Exception:          # numba missing: slow NumPy path
    filler_pages = ref.filler_pages

    def indep_units(fkey, pages, subs):
        return np.stack([ref.indep_words(fkey, int(p), int(s) * 1024, 1024) for p, s in zip(pages, subs)])


class WeightSource:
    """Declared weights as flat uint16 arrays (loaded lazily from safetensors)."""

    def __init__(self, path):
        from schemes.memory_link_exhaustion.core.model import SafeTensors
        self.st = SafeTensors(path)
        self.cache = {}

    def flat(self, name):
        if name not in self.cache:
            t = self.st.get(name).to(torch.bfloat16).contiguous()
            self.cache[name] = t.view(torch.int16).numpy().view(np.uint16).ravel()
        return self.cache[name]

    def range(self, sources, j0, n):
        """Elements [j0, j0+n) of the dim-0 concatenation of `sources`."""
        out = []
        base = 0
        for s in sources:
            a = self.flat(s)
            lo, hi = max(j0, base), min(j0 + n, base + a.size)
            if lo < hi:
                out.append(a[lo - base: hi - base])
            base += a.size
            if base >= j0 + n:
                break
        return np.concatenate(out) if len(out) > 1 else out[0]


class Expect:
    def __init__(self, layout: Layout, weights: WeightSource, fkey: int):
        self.lay, self.w, self.fkey = layout, weights, fkey
        self.wents = [e for e in layout.entries if e.sources is not None]

    def kind(self, p):
        b = p * PAGE
        if b < self.lay.w_end:
            return "W"
        if b < self.lay.kv_end:
            return "K"
        return "F"

    def pages(self, pages):
        """Expected uint32 words for W/F pages. [n, PAGE_WORDS]."""
        pages = np.asarray(pages, dtype=np.int64)
        out = filler_pages(self.fkey, pages)
        for r, p in enumerate(pages):
            a, b = int(p) * PAGE, int(p + 1) * PAGE
            if a >= self.lay.w_end:
                continue
            u16 = out[r].view(np.uint16)
            for e in self.wents:
                lo, hi = max(a, e.offset), min(b, e.offset + e.nbytes)
                if lo >= hi:
                    continue
                j0 = (lo - e.offset) // 2
                n = (hi - lo) // 2
                raw = self.w.range(e.sources, j0, n)
                if e.key < 0:
                    enc = raw
                elif ref.WENC == "sloth":
                    from schemes.memory_link_exhaustion.encodings import sloth64
                    enc = sloth64.encode_range(raw, ref.sloth_seed(e.key, ref.w_ordinal(e.offset)), j0)
                elif ref.WENC == "bfly":
                    from schemes.memory_link_exhaustion.encodings import bfly_ref
                    enc = bfly_ref.encode_range(raw, e.key, ref.w_ordinal(e.offset), j0, e.shape[1])
                else:
                    enc = ref.encode_w(raw, e.key, j0, e.shape[1])
                u16[(lo - a) // 2:(hi - a) // 2] = enc
        return out

    def digests(self, pages, nonce):
        exp = self.pages(pages)
        return [ref.page_digest(exp[i], nonce) for i in range(len(pages))]
