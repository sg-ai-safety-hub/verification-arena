"""Noise-ring granules in the mixed chase (--ring-chase-guard), CPU model.

The device kernel maps a ring probe as: entry CH_RING_TAG | j, chunk offset `off` in the 64 MiB granule ->
logical slot q = 2 j + (off >> 18), physical slot (head - q) mod n_ring, byte (off & (2^18 - 1)) * 128 in that slot.
The verifier recomputes the same probe as the pool bytes of window (head - q) (ref.ring_chase_slot). This test
drives ref.ch_chain through both mappings over a simulated ring and pool and requires the same chain state, checks
that no probed slot is inside the guard, and that a stale ring (one window not copied) changes the chain.
"""
import numpy as np

from schemes.memory_link_exhaustion.encodings import ref

G, GPB = ref.CH_GRAN_PAGES, ref.CH_GRAN_PER_BLOCK
SLOT_W = (1 << 25) // 4                    # uint32 words per 32 MiB window


def main():
    n_ring, guard, n_slots_pool = 64, 24, 256
    nrg = ref.ring_chase_granules(n_ring, guard)
    assert nrg == 20 and ref.ring_chase_granules(n_ring, 0) == 0
    nW, nK, nF = 4 * G, 2 * G, 6 * G
    gp = ref.ch_granules_mix(nW, nK, nF, 0, nrg)
    assert sum(1 for x in set(gp) if x >= ref.CH_RA0) == nrg
    assert ref.ch_granules_mix(nW, nK, nF, 0, 0) == ref.ch_granules_mix(nW, nK, nF)       # switch off: table unchanged
    tb = len(gp) // GPB
    # window i comes from pool slot slot_of(i); a window's words are a hash of (pool slot, word index)
    slot_of = lambda i: (i * 7 + 3) % n_slots_pool

    def pool_words(slot, w, n=32):
        x = (np.arange(w, w + n, dtype=np.uint64) + np.uint64(slot) * np.uint64(SLOT_W)) * np.uint64(0x9E3779B97F4A7C15)
        return ((x >> np.uint64(29)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    def arena_words(page, wo):
        x = (np.arange(wo, wo + 32, dtype=np.uint64) + np.uint64(page) * np.uint64(16384)) * np.uint64(0xD1B54A32D192ED03)
        return ((x >> np.uint64(31)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    head = 1000
    ring_win = {s: max(i for i in range(head - n_ring + 1, head + 1) if i % n_ring == s) for s in range(n_ring)}
    probed = set()

    def verifier_words(page, wo):                       # the verifier's view: window (head - q) from the pool
        if page >= ref.CH_RA0:
            q, w_in = ref.ring_chase_slot(page, wo)
            return pool_words(slot_of(head - q), w_in)
        return arena_words(page, wo)

    def device_words(ring_state):
        def f(page, wo):                                # the kernel's view: physical slot, byte offset
            if page >= ref.CH_RA0:
                j, pg = divmod(page - ref.CH_RA0, G)
                off = pg * 512 + wo // 32               # 128-byte chunk index inside the granule
                q = 2 * j + (off >> 18)
                ps = (head - q) % n_ring
                probed.add(q)
                byte = (off & ((1 << 18) - 1)) * 128
                return pool_words(slot_of(ring_state[ps]), byte // 4)
            return arena_words(page, wo)
        return f

    nb = min(2, tb)
    for nonce in (1, 99, 4242):
        for chain in (0, 5):
            a = ref.ch_chain(nonce, chain, 40, tb, nb, gp, verifier_words)
            b = ref.ch_chain(nonce, chain, 40, tb, nb, gp, device_words(ring_win))
            assert a == b, (nonce, chain)
    assert probed and max(probed) <= n_ring - guard - 1 and min(probed) >= 0, sorted(probed)[-3:]
    # a ring that missed windows (every slot one lap stale) cannot give the same chains
    stale = {s: w - n_ring for s, w in ring_win.items()}
    diff = sum(ref.ch_chain(n, 0, 40, tb, nb, gp, verifier_words) != ref.ch_chain(n, 0, 40, tb, nb, gp, device_words(stale))
               for n in range(20))
    assert diff >= 15, diff
    q, w_in = ref.ring_chase_slot(ref.CH_RA0 + 3 * G + 513, 64)
    assert (q, w_in) == (7, 16384 + 64)
    print("test_ring_chase_cpu: PASS")


if __name__ == "__main__":
    main()
