"""Chase-bound KV claims (unified challenge), CPU only: the pages a mixed chase reads all lie in the probed range that
ref.ch_range_pages derives from the nonce, the reads are uniform over it (untouched fraction ~ exp(-lam)), and the
reference geometry gives lam ~ 56 (so the range is the touched set). Run: python schemes/memory_link_exhaustion/tests/test_kvclaim_cpu.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
from schemes.memory_link_exhaustion.encodings import ref

G = ref.CH_GRAN_PAGES


def touched(nonce, K, P, tb, nb, gm, content):
    seen = set()

    def words(page, wo):
        seen.add(page)
        return content(page, wo)
    for c in range(K):
        ref.ch_chain(nonce, c, P, tb, nb, gm, words)
    return seen


def main():
    # small mixed arena: W, KV, filler granules plus 3 pool granules, interleaved like the real table
    nW, nK, nF = 40 * G, 30 * G, 70 * G
    gm = ref.ch_granules_mix(nW, nK, nF, 3)
    tb = len(gm) // ref.CH_GRAN_PER_BLOCK
    nb = 1
    rnd = np.random.default_rng(1)
    zero = np.zeros(32, np.uint32)
    fails = 0
    for nonce in (12345, 777, 2 ** 31 + 5):
        kv = set(ref.ch_range_pages(nonce, gm, tb, nb, nW, nW + nK))
        rng_all = set(ref.ch_range_pages(nonce, gm, tb, nb, 0, ref.CH_PA0 * 2))
        pages = ref.ch_range_pages(nonce, gm, tb, nb, 0, ref.CH_PA0)          # arena pages in the range
        K, P = 64, 128                                                       # 65,536 probes over 65,536 page slots
        lam = ref.ch_coverage(K, P, nb)
        tab = {}
        seen = touched(nonce, K, P, tb, nb, gm,
                       lambda p, wo: tab.setdefault((p, wo), rnd.integers(0, 2 ** 32, 32, dtype=np.uint32))
                       if p % 3 else zero)
        outside = [p for p in seen if p not in rng_all]
        kv_seen = {p for p in seen if nW <= p < nW + nK}
        frac = 1 - len(kv_seen) / max(1, len(kv))
        ok = not outside and kv_seen <= kv and abs(frac - math.exp(-lam)) < 0.05
        fails += not ok
        print(f"nonce {nonce}: range {len(pages)} arena pages, {len(kv)} KV; lam {lam:.2f}; reads outside the range "
              f"{len(outside)}; KV untouched {frac:.3f} (exp(-lam) {math.exp(-lam):.3f}) {'OK' if ok else 'FAIL'}")
    lam_ref = ref.ch_coverage(132 * 32, 1200, 11)
    p_miss = math.exp(-lam_ref)
    print(f"reference K=4224 P=1200 nb=11: lam {lam_ref:.1f}, P(page in range untouched) {p_miss:.1e}; "
          f"1 h at 4/s x 2048 claimed pages: {4 * 3600 * 2048 * p_miss:.1e} expected untouched claims")
    fails += not (lam_ref > 50)
    # the granule helper and the page helper agree
    g = ref.ch_range_granules(12345, gm, tb, nb, nW, nW + nK)
    fails += ref.ch_range_pages(12345, gm, tb, nb, nW, nW + nK) != [x + i for x in g for i in range(G)]
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
