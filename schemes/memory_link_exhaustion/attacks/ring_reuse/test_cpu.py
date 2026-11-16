"""ring_reuse_idle accounting: on the old (with-replacement) schedule the ring already holds ~ring/pool of each
window (~3% at the real sizes); on a keyed-permutation schedule ring reuse is 0 and only a static pinned cache hits
(~cache/pool), with no churn.

    python -m pytest -q schemes/memory_link_exhaustion/attacks/ring_reuse/test_cpu.py
"""
import hashlib

from schemes.memory_link_exhaustion.attacks.ring_reuse.workload import ResidencyReuse


def _old_offsets(n_starts, n, seed=0):
    """Per-window uniform base page WITH REPLACEMENT (the old hash schedule)."""
    for i in range(n):
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        yield int.from_bytes(h[:8], "big") % n_starts


def _perm_naive(n_tiles, ppw, n, seed=1):
    """Disjoint windows, a fresh shuffle each cycle with NO lag: a tile in the previous cycle's ring tail can reappear
    early in the next cycle, so the cycle boundary leaks. This is the old permutation without blue's lag guard."""
    import random
    cyc = 0
    while n > 0:
        order = list(range(n_tiles))
        random.Random(seed + cyc).shuffle(order)
        for t in order:
            yield t * ppw
            n -= 1
            if n <= 0:
                return
        cyc += 1


def _perm_lag(n_tiles, ppw, n, n_ring, seed=1):
    """The FIXED schedule: a keyed permutation per cycle with a lag guard so no tile recurs within n_ring windows,
    including across the cycle boundary (the first n_ring tiles of each cycle avoid the previous cycle's ring tail)."""
    import random
    assert n_tiles >= 2 * n_ring
    cyc, tail = 0, []
    while n > 0:
        order = list(range(n_tiles))
        random.Random(seed + cyc).shuffle(order)
        tset = set(tail)
        early, late = order[:n_ring], order[n_ring:]
        for idx in range(len(early)):
            if early[idx] in tset:
                for j in range(len(late)):
                    if late[j] not in tset:
                        early[idx], late[j] = late[j], early[idx]
                        break
        order = early + late
        for t in order:
            yield t * ppw
            tail = (tail + [t])[-n_ring:]
            n -= 1
            if n <= 0:
                return
        cyc += 1


def test_old_schedule_reuse_matches_ring_over_pool():
    n_pages, ppw, n_ring = 4000, 20, 32            # pool 4000 pages; ring holds 32 windows = 640 pages = 16%
    n_starts = n_pages - ppw + 1
    acct = ResidencyReuse(n_pages, ppw, n_ring)
    for off in _old_offsets(n_starts, 20000):
        acct.window(off)
    expected = (n_ring * ppw) / n_pages            # ~0.16
    assert abs(acct.fraction - expected) < 0.02, (acct.fraction, expected)
    assert acct.reused_pages > 0


def test_permutation_schedule_ring_reuse_is_zero():
    n_tiles, ppw, n_ring = 200, 20, 32             # 200 disjoint windows; ring < cycle, so no page repeats in-ring
    acct = ResidencyReuse(n_tiles * ppw, ppw, n_ring)
    for base in _perm_lag(n_tiles, ppw, 4000, n_ring):
        acct.window(base)
    assert acct.reused_pages == 0                   # lag guard: nothing resident recurs within the ring
    # and the naive per-cycle shuffle (no lag) DOES leak at the cycle boundary -- why lag is needed
    leak = ResidencyReuse(n_tiles * ppw, ppw, n_ring)
    for base in _perm_naive(n_tiles, ppw, 4000):
        leak.window(base)
    assert leak.reused_pages > 0


def test_permutation_with_static_pinned_cache_hits_cache_over_pool():
    n_tiles, ppw, n_ring = 200, 20, 32
    n_pages = n_tiles * ppw
    cache_pages = n_pages // 10                      # pin ~10% of the pool
    acct = ResidencyReuse(n_pages, ppw, n_ring, cache_pages=cache_pages, cache_seed=7)
    for base in _perm_lag(n_tiles, ppw, 20000, n_ring):
        acct.window(base)
    # the only hits are pinned-cache pages; expected fraction ~= cache / pool, steady, no churn
    assert abs(acct.fraction - cache_pages / n_pages) < 0.02, (acct.fraction, cache_pages / n_pages)


def test_freed_gbs_scales_with_reuse():
    acct = ResidencyReuse(4000, 20, 32)
    for off in _old_offsets(4000 - 19, 20000):
        acct.window(off)
    g = acct.freed_gbs(window_rate_hz=1750.0)       # 1750 windows/s * 32 MiB ~= 56 GB/s link
    assert g > 0
    d = acct.describe()
    assert d["reused_fraction"] == round(acct.fraction, 4) and d["ring_pages"] == 32 * 20
