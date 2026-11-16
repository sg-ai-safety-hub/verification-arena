"""Settled-KV rule for the mixed chase, CPU model (encodings/ref.py ch_masked / ch_mask_words).

A chain over W, KV, filler and the ring is computed over a simulated memory in which the KV region is written by
in-flight steps. With the per-block table (lo, hi) applied:
  1. a write anywhere inside the in-flight band leaves every chain value unchanged (the timed pass and a replay agree);
  2. a write below lo (settled KV) or at/above hi (the previous owner's bytes) changes chains;
  3. the verifier's replay from sources (current owner's KV below lo, previous owner's above hi, mask in the band,
     pool bytes for the ring at the returned head) equals the device-side chain;
  4. declaring a wider band than the true one changes the chain values, so the table that came with the answer is the
     one the chain was computed under.
"""
import numpy as np

from schemes.memory_link_exhaustion.encodings import ref

G, GPB, TW = ref.CH_GRAN_PAGES, ref.CH_GRAN_PER_BLOCK, ref.CH_KV_TOK_WORDS


def h32(tag, page, wo, n=32):
    x = (np.arange(wo, wo + n, dtype=np.uint64) + np.uint64(page) * np.uint64(16384) + np.uint64(tag) * np.uint64(1 << 50))
    x = x * np.uint64(0x9E3779B97F4A7C15)
    return ((x >> np.uint64(27)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def main():
    nW, nK, nF, nb_kv = 2 * G, 4 * G, 2 * G, 2 * G            # 2 layers x 2,048 blocks, one page per (layer, block)
    nrg = ref.ring_chase_granules(64, 24)
    gp = ref.ch_granules_mix(nW, nK, nF, 0, nrg)
    tb = len(gp) // GPB
    nb = min(2, tb)
    rng = np.random.default_rng(5)
    lo = rng.integers(0, 17, nb_kv)
    band = {b: (int(lo[b]), int(min(16, lo[b] + rng.integers(0, 4)))) for b in range(nb_kv)}
    owner = {b: 100 + b for b in range(nb_kv)}               # current owner id per block
    prev = {b: 900 + b for b in range(nb_kv)}                # previous owner id per block
    head = 5000
    assert ref.ch_kv_token(0) == 0 and ref.ch_kv_token(1023) == 0 and ref.ch_kv_token(1024) == 1 and ref.ch_kv_token(16383) == 15
    assert not ref.ch_masked(None, 0) and ref.ch_masked((2, 4), 2 * TW) and not ref.ch_masked((2, 4), 4 * TW)

    def source(page, wo, nonce, tab=band, scribble=None):
        """What the device reads / the verifier reconstructs for one chunk."""
        if page >= ref.CH_RA0:
            q, w_in = ref.ring_chase_slot(page, wo)
            return h32(7, (head - q) % 2048, w_in % 16384)
        if nW <= page < nW + nK:
            layer, b = divmod(page - nW, nb_kv)
            t = ref.ch_kv_token(wo)
            if ref.ch_masked(tab[b], wo):
                return ref.ch_mask_words(nonce, page - nW, wo)
            if scribble is not None and scribble(b, t):
                return h32(55, page, wo)                      # bytes someone wrote over this position
            return h32(owner[b] if t < tab[b][0] else prev[b], page, wo)
        return h32(1, page, wo)

    def chains(nonce, **kw):
        return [ref.ch_chain(nonce, c, 60, tb, nb, gp, lambda p, w: source(p, w, nonce, **kw)) for c in range(6)]

    for nonce in (3, 777):
        base = chains(nonce)
        # 1. writes inside the band: nothing changes
        assert chains(nonce, scribble=lambda b, t: band[b][0] <= t < band[b][1]) == base
        # 2. writes to settled KV, or to the previous owner's bytes, change chains
        assert chains(nonce, scribble=lambda b, t: t < band[b][0]) != base
        assert chains(nonce, scribble=lambda b, t: t >= band[b][1]) != base
        # 4. a wider declared band is a different computation
        wide = {b: (max(0, band[b][0] - 2), band[b][1]) for b in band}
        assert chains(nonce, tab=wide) != base
    # the mask depends on the nonce and on the chunk, and only on them
    a = ref.ch_mask_words(3, 10, 64)
    assert (a == ref.ch_mask_words(3, 10, 64 + 31)).all() and (a != ref.ch_mask_words(4, 10, 64)).any()
    assert (a != ref.ch_mask_words(3, 10, 96)).any() and (a != ref.ch_mask_words(3, 11, 64)).any() and a.dtype == np.uint32
    # share of KV probes that land in the band (reported by honest runs; the cap rule bounds it)
    masked = total = 0

    def count(p, w):
        nonlocal masked, total
        if nW <= p < nW + nK:
            total += 1
            masked += ref.ch_masked(band[(p - nW) % nb_kv], w)
        return source(p, w, 9)
    ref.ch_chain(9, 0, 200, tb, nb, gp, count)
    assert total > 100 and 0 < masked < total
    print(f"test_settled_kv_cpu: PASS (masked share of KV probes in this model: {masked / total:.3f})")


if __name__ == "__main__":
    main()
