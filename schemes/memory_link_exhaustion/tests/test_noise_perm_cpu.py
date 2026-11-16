"""Keyed permutation noise schedule (--noise-perm, ref.perm_keys), CPU only (GPU packages stubbed; numpy needed):
  1. each cycle is a permutation of the pool's whole-window slots;
  2. no slot recurs within the horizon (ring + lag + lead windows), across cycle boundaries too;
  3. the prover's unchanged sched_offset(s_i) gives slot * pages_per_slot;
  4. Revealer -> checker (d0_all, off_all), the Python reveal loop and the native revealer (mock build, if g++ is
     present) all publish the same values; with the flag off they still publish the old fmix keys;
  5. a revealed value does not determine the next: the fmix-inversion attack that recovers the old schedule's 32-bit
     key from one reveal predicts every later offset of the old schedule, and only chance (1/slots) of the new one.
Run: python schemes/memory_link_exhaustion/tests/test_noise_perm_cpu.py"""
import os
import sys
import threading
import types
import importlib.abc
import importlib.machinery

ROOTS = ("torch", "triton", "vllm", "transformers", "safetensors", "numba")


class _Any(types.ModuleType):
    __path__ = []

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        v = _Any(f"{self.__name__}.{k}")
        setattr(self, k, v)
        return v

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f=None, *aa, **kk: f

    def __getitem__(self, k):
        return _Any("item")

    def __or__(self, o):
        return self

    def __mro_entries__(self, bases):
        return (object,)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ROOTS:
            try:
                __import__("importlib.util").util.find_spec  # noqa
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _Any(spec.name)

    def exec_module(self, m):
        pass


if os.environ.get("VARENA_REAL_TORCH") != "1":
    sys.meta_path.insert(0, _Finder())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import subprocess
import tempfile
import time

import numpy as np

from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.link import pcie

MIB = 1 << 20
FAIL = []


def check(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c:
        FAIL.append(msg)


def slots_of(s, ns, pps):
    off = np.array([ref.sched_offset(int(x), ns) for x in s], dtype=np.int64)
    return off // pps, off % pps


def min_repeat_distance(seq):
    last, best = {}, 1 << 62
    for t, v in enumerate(seq):
        if v in last:
            best = min(best, t - last[v])
        last[v] = t
    return best


def main():
    key = os.urandom(16)
    # 1-3 at the reference size: 64 GiB pool, 32 MiB windows -> 2,048 slots, 8,192 pages per slot
    pool_b, W = 64 << 30, 32 * MIB
    S, pps = pool_b // W, W // pcie.ALIGN
    ns = pcie.window_slots(pool_b, W)
    H = 64 + 3 + 16 + 64                           # ring 2 GiB + lag + lead + margin
    cyc = ref.perm_cycles(key, 6, S, H)
    check(all((np.sort(r) == np.arange(S)).all() for r in cyc), f"each of 6 cycles is a permutation of {S} slots")
    check(len({tuple(r[:16]) for r in cyc}) == 6, "a fresh permutation per cycle")
    s = ref.perm_keys(key, 6 * S, S, pps, ns, H)
    sl, rem = slots_of(s, ns, pps)
    check((rem == 0).all() and (sl == cyc.reshape(-1)).all(), "sched_offset(s_i) = slot * pages_per_slot, in order")
    d = min_repeat_distance(sl.tolist())
    check(d >= H, f"no slot recurs within {H} windows across 6 cycles (min distance {d})")
    # stress the boundary guard: few slots, horizon = half of them, many cycles
    for S2, H2 in ((64, 32), (16, 8), (2048, 1024)):
        c2 = ref.perm_cycles(os.urandom(16), 50 if S2 < 2048 else 4, S2, H2)
        ok = all((np.sort(r) == np.arange(S2)).all() for r in c2)
        d2 = min_repeat_distance(c2.reshape(-1).tolist())
        check(ok and d2 >= H2, f"slots {S2}, horizon {H2}: permutations, min repeat distance {d2} >= {H2}")
    # the default (flag-off) schedule for comparison: independent offsets reuse ring-resident bytes
    vk = 0x1234567
    old = [ref.sched_offset(ref.sched_key(vk, i), ns) for i in range(20000)]
    ov = sum(1 for i in range(64, 20000) if any(abs(old[i] - old[j]) < pps for j in range(i - 64, i)))
    print(f"info default schedule: {ov / (20000 - 64):.2%} of windows overlap one of the previous 64 (2 GiB ring)")

    # 4: Revealer -> checker, Python reveal loop, native revealer
    tmp = tempfile.mkdtemp()
    ck = types.SimpleNamespace(pool=types.SimpleNamespace(size=pool_b // 4), W=W, sched={"slack": 1.0})
    r = pcie.Revealer(os.path.join(tmp, "rev"), 1 << 16, 777, 1e-3, 4e-3, ck, perm={"key": key, "horizon": H})
    tab = r.ktab
    check(r.ktab_n == len(tab) and r.ktab_n % S == 0, f"table of {r.ktab_n} windows ({r.ktab_n // S} cycles)")
    check((tab[:6 * S] == s).all(), "Revealer table = ref.perm_keys")
    check(all(ck.d0_all[i] == int(tab[i]) and ck.off_all[i] == ref.sched_offset(int(tab[i]), ns)
              for i in list(range(3000)) + [r.ktab_n - 1]), "checker d0_all / off_all = table")
    E = r.perm_args["E"]
    far = [r.ktab_n, r.ktab_n + 5, 5 * E + 3, 40 * E + 17]          # beyond the table: the schedule does not wrap
    exp_far = [int(ref.perm_keys_range(key, (i // S), 1, S, pps, ns, r.perm_args["horizon"])[i % S]) for i in far]
    check([ck.d0_all[i] for i in far] == exp_far and int(tab[0]) != ck.d0_all[r.ktab_n],
          f"schedule continues past the {r.ktab_n}-window table (epochs of {E}), no wrap")
    ck0 = types.SimpleNamespace(pool=types.SimpleNamespace(size=pool_b // 4), W=W, sched={"slack": 1.0})
    pcie.Revealer(os.path.join(tmp, "rev0"), 1 << 16, 777, 1e-3, 4e-3, ck0)
    check(all(ck0.d0_all[i] == ref.sched_key(777, i) for i in range(3000)), "flag off: checker keeps the fmix keys")
    # Python reveal loop (VARENA_NATIVE_REVEALER=0) in a thread, timed schedule, 0.1 ms per window
    for tag, perm in (("perm", r.perm_args), ("off", None)):
        p = os.path.join(tmp, "py_" + tag)
        _mm, _ = pcie.create_pool(p, pcie.rev_words(1 << 16) * 8, None)
        np.frombuffer(_mm, dtype=np.int64)[:2 << 16] = -1           # as Revealer.__init__: nothing revealed yet
        os.environ["VARENA_NATIVE_REVEALER"] = "0"
        th = threading.Thread(target=pcie._reveal_loop, args=(p, 1 << 16, 777, 1e-4, 0.0, time.perf_counter() + 5,
                                                               1.0, 0.0, None, None, perm), daemon=True)
        th.start()
        time.sleep(5.6)
        a = np.memmap(p, dtype=np.int64, mode="r")[:1 << 16]
        got = {int(x) >> 32: int(x) & 0xFFFFFFFF for x in a if int(x) >= 0}
        exp = (lambda i: int(tab[i])) if perm else (lambda i: ref.sched_key(777, i))
        bad = [(i, v, exp(i)) for i, v in got.items() if v != exp(i)][:3]
        check(len(got) > 1000 and not bad,
              f"Python reveal loop ({tag}): {len(got)} reveals match {bad}")
    gpp = subprocess.run(["which", "g++"], capture_output=True).returncode == 0
    if gpp:
        b = subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "..", "tools", "sq_mock",
                                                  "run_revealer.sh"), tmp], capture_output=True, text=True)
        exe = b.stdout.strip().splitlines()[-1] if b.returncode == 0 else None
        check(exe is not None, "native revealer builds against the mock runtime" + ("" if exe else b.stderr[-400:]))
        if exe:
            tf = os.path.join(tmp, "ktab.bin")
            tab[:4 * S].astype("<u4").tofile(tf)
            for tag, arg, exp in (("perm", tf, lambda i: int(tab[i % (4 * S)])),
                                  ("off", "-", lambda i: ref.sched_key(777, i))):
                out = subprocess.run([exe, str(5 * S), arg, "777"], capture_output=True, text=True, timeout=120).stdout
                rows = [tuple(map(int, ln.split())) for ln in out.split("\n") if ln.strip()]
                check(len(rows) == 5 * S and all(v == exp(i) for i, v in rows),
                      f"native revealer ({tag}): {len(rows)} reveals match (table wraps at {4 * S})")
    else:
        print("skip native revealer (no g++)")

    # 4b: indefinite schedule, small geometry (64 slots, horizon 16, 256-window epochs, 4-epoch ring): exact page-level ring reuse
    # over 8 epochs and the Python reveal loop with a fake prover echoing reports, so the epoch refill thread runs
    from schemes.memory_link_exhaustion.attacks.ring_reuse.workload import ResidencyReuse
    old_n = ref.PERM_TAB_N
    ref.PERM_TAB_N = 1024
    try:
        ck2 = types.SimpleNamespace(pool=types.SimpleNamespace(size=(64 * W) // 4), W=W, sched={"slack": 1.0})
        r2 = pcie.Revealer(os.path.join(tmp, "rev2"), 1 << 12, 5, 1e-3, 4e-3, ck2, perm={"key": key, "horizon": 999})
        E2, S2, H2 = r2.perm_args["E"], r2.perm_args["S"], r2.perm_args["horizon"]
        ns2 = r2.perm_args["ns"]
        sl2 = [ref.sched_offset(ck2.d0_all[i], ns2) // pps for i in range(8 * E2)]
        acc2 = ResidencyReuse(S2 * 8, 8, H2)
        for o in sl2:
            acc2.window(o * 8)
        d2 = min_repeat_distance(sl2)
        check(E2 == 256 and H2 == 16 and acc2.reused_pages == 0 and d2 >= H2,
              f"indefinite schedule over 8 epochs: ring reuse {acc2.reused_pages} pages, min repeat {d2} >= {H2}")
        p2 = os.path.join(tmp, "py_epochs")
        _mm2, _ = pcie.create_pool(p2, pcie.rev_words(1 << 12) * 8, None)
        a2 = np.frombuffer(_mm2, dtype=np.int64)
        a2[:2 << 12] = -1
        os.environ["VARENA_NATIVE_REVEALER"] = "0"
        import threading as _th
        _th.Thread(target=pcie._reveal_loop, args=(p2, 1 << 12, 5, 2e-3, 0.0, time.perf_counter() + 4, 1.0, 0.0,
                                                   None, None, r2.perm_args), daemon=True).start()
        N2, got2, j = 1 << 12, {}, 0
        t_end = time.perf_counter() + 9
        while time.perf_counter() < t_end:            # fake prover: report every revealed window in order
            x = int(a2[j % N2])
            if x >= 0 and (x >> 32) == j:
                got2[j] = x & 0xFFFFFFFF
                a2[N2 + j % N2] = (j << 32) | (x & 0xFFFFFFFF)
                j += 1
            else:
                time.sleep(0.0005)
        bad2 = [i for i, v in got2.items() if v != ck2.d0_all[i]][:3]
        check(j > 3 * E2 and not bad2, f"Python reveal loop across {j // E2} epochs: {j} reveals match {bad2}")
    finally:
        ref.PERM_TAB_N = old_n

    # 5: does a revealed value determine the next one?
    def attack(s0, i0):            # old schedule: s_i = fmix(vkey ^ fmix(i ^ C)) -> invert one reveal
        return ref.fmix_inv_int(s0) ^ ref.fmix_int((i0 & 0xFFFFFFFF) ^ 0x2545F491)

    vk2 = attack(ref.sched_key(vk, 100), 100)
    hit_old = np.mean([ref.sched_offset(ref.sched_key(vk2, i), ns) == old[i] for i in range(101, 2101)])
    check(hit_old == 1.0, f"(the hole) default schedule: one reveal predicts the next 2,000 offsets ({hit_old:.0%})")
    big = ref.perm_keys(key, 10 * S, S, pps, ns, H)
    bsl = slots_of(big, ns, pps)[0]
    hits = sum(ref.sched_offset(ref.sched_key(attack(int(big[i]), i), i + 1), ns) // pps == bsl[i + 1]
               for i in range(len(big) - 1))
    nxt = sum(bsl[i + 1] == (bsl[i] + 1) % S for i in range(len(bsl) - 1))
    lim = 5 * len(big) / S + 10
    check(hits < lim and nxt < lim, f"perm schedule: inversion attack predicts {hits}, slot+1 predicts {nxt} of "
                                    f"{len(big) - 1} next slots (chance ~{len(big) / S:.0f})")
    # each reveal carries independent padding: two reveals of one cycle give no common 32-bit key under inversion
    ks = {attack(int(big[i]), i) for i in range(200)}
    check(len(ks) == 200, "perm schedule: no common key across reveals")
    # reveals are near-uniform 32-bit values (padding r spans the 2^32 / ns preimages of each offset)
    hi = np.bincount(big >> 28, minlength=16) / len(big)
    check(hi.max() < 1.3 / 16 and hi.min() > 0.7 / 16, "perm schedule: revealed values spread over 32 bits")
    # 6: exact page-level reuse accounting (ResidencyReuse, attacks/ring_reuse): pages of a new
    # window already resident in the ring. Scaled geometry: 2,048 slots, 8 pages per window, 64-window ring.
    from schemes.memory_link_exhaustion.attacks.ring_reuse.workload import ResidencyReuse
    S3, pp3, nr3 = 2048, 8, 64
    ns3 = S3 * pp3 - pp3 + 1
    H3 = nr3 + 3 + 16 + 64
    for tag, offs, lo, hi in (
            ("flag on (perm + guard)", [ref.sched_offset(int(x), ns3) for x in ref.perm_keys(os.urandom(16), 12 * S3, S3, pp3, ns3, H3)], 0, 0),
            ("flag off (default)", [ref.sched_offset(ref.sched_key(4242, i), ns3) for i in range(12 * S3)], 0.02, 0.045)):
        acc = ResidencyReuse(S3 * pp3, pp3, nr3)
        for o in offs:
            acc.window(o)
        f = acc.fraction
        check(lo <= f <= hi, f"ring reuse {tag}: {f:.4%} over 12 cycles (expect {'exactly 0' if hi == 0 else '~ring/pool = 3.1%'})")
    # the cross-boundary guard is load-bearing: the same per-cycle permutations without it reuse ring pages
    leaks = 0
    for t in range(10):
        cy = ref.perm_cycles(os.urandom(16), 12, S3, 0)
        acc = ResidencyReuse(S3 * pp3, pp3, nr3)
        for o in (cy.reshape(-1) * pp3).tolist():
            acc.window(o)
        leaks += acc.reused_pages > 0
    check(leaks >= 8, f"without the boundary guard (horizon 0) ring pages are reused in {leaks}/10 runs of 12 cycles")
    print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
