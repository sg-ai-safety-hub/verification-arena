"""Chase round function (ref.CH_MIX / chase.cu combine).

With the older affine round (CH_MIX = 0: t = t * MULC + w) a 128-byte chunk enters a chain only through
H(chunk) = sum over its 32 words of 31^popcount(l) * word[l] mod 2^64: two chunks with the same H give the same
chain for every nonce, chain and round, so a prover could keep 8 bytes per chunk. This test shows that for CH_MIX = 0
(the hole) and requires that it is NOT so for CH_MIX = 1 (t = rotl(t ^ w, 23) * MULC): there, changing a chunk while
keeping its weighted sum changes the chain.
"""
import numpy as np

from schemes.memory_link_exhaustion.encodings import ref

G, GPB = ref.CH_GRAN_PAGES, ref.CH_GRAN_PER_BLOCK


def words(page, wo):
    x = (np.arange(wo, wo + 32, dtype=np.uint64) + np.uint64(page) * np.uint64(16384)) * np.uint64(0x9E3779B97F4A7C15)
    return ((x >> np.uint64(29)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def same_sum(page, wo):
    """A different chunk with the same weighted sum: lanes 1 and 2 have equal weight (popcount 1), so +d / -d."""
    w = words(page, wo).astype(np.int64)
    d = 12345
    if w[1] + d > 0xFFFFFFFF or w[2] - d < 0:
        d = -d
        if w[1] + d < 0 or w[2] - d > 0xFFFFFFFF:
            return w.astype(np.uint32)
    w[1] += d
    w[2] -= d
    return w.astype(np.uint32)


def run(mix):
    old = ref.CH_MIX
    ref.CH_MIX = mix
    try:
        gp = ref.ch_granules_mix(4 * G, 2 * G, 2 * G)
        tb = len(gp) // GPB
        a = [ref.ch_chain(n, c, 50, tb, min(2, tb), gp, words) for n in (1, 77) for c in (0, 3)]
        b = [ref.ch_chain(n, c, 50, tb, min(2, tb), gp, same_sum) for n in (1, 77) for c in (0, 3)]
        return a, b
    finally:
        ref.CH_MIX = old


def main():
    a0, b0 = run(0)
    assert a0 == b0, "affine round: chunks with equal weighted sums must give equal chains (the documented hole)"
    a1, b1 = run(1)
    assert all(x != y for x, y in zip(a1, b1)), "CH_MIX = 1: a chunk with the same weighted sum must change every chain"
    assert a1 != a0
    print("test_chase_mix_cpu: PASS (affine round reproduces chains from per-chunk sums; rotate-xor round does not)")


if __name__ == "__main__":
    main()
