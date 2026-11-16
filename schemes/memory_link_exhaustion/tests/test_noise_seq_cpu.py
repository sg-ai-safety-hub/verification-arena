"""--noise-perm 2: the sequential (public) noise schedule. Every cycle walks the pool's slots in order, the revealed
values still decode to those slots on the prover side, the range and whole-table builders agree, and no slot recurs
within a ring-sized horizon (ring reuse 0, by the same accountant the keyed permutation is tested with)."""
import numpy as np

from schemes.memory_link_exhaustion.attacks.ring_reuse.workload import ResidencyReuse  # noqa: F401  (import check)
from schemes.memory_link_exhaustion.encodings import ref


def main():
    S, pps, H = 2048, 8192, 147
    ns = S * pps
    cyc = ref.perm_cycles(ref.SEQ_KEY, 3, S, H)
    assert (cyc == np.arange(S)).all()
    assert (ref.perm_cycle(ref.SEQ_KEY, 2, S, H) == np.arange(S)).all()
    keys = ref.perm_keys(ref.SEQ_KEY, 3 * S, S, pps, ns, H)
    assert (keys == ref.perm_keys_range(ref.SEQ_KEY, 0, 3, S, pps, ns, H)).all()
    assert (keys[S:2 * S] == ref.perm_keys_range(ref.SEQ_KEY, 1, 1, S, pps, ns, H)).all()
    offs = np.array([ref.sched_offset(int(k), ns) for k in keys[:S + 300]])
    assert (offs == (np.arange(S + 300) % S) * pps).all(), "revealed values must decode to slots 0, 1, 2, ..."
    slots = offs // pps
    last = {}
    dmin = 10 ** 9
    for i, sl in enumerate(slots):
        if sl in last:
            dmin = min(dmin, i - last[sl])
        last[sl] = i
    assert dmin == S >= H, dmin
    k2 = ref.perm_keys(b"0123456789abcdef", S, S, pps, ns, H)       # the keyed schedule is unchanged
    o2 = np.array([ref.sched_offset(int(k), ns) for k in k2]) // pps
    assert sorted(o2.tolist()) == list(range(S)) and (o2 != np.arange(S)).any()
    print("test_noise_seq_cpu: PASS")


if __name__ == "__main__":
    main()
