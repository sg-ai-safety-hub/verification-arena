"""PCIe link exhaustion with verifier-supplied random noise (inbound only in the final configuration).

The verifier fills a host noise pool (shared memory) with random bytes and reveals the schedule one window at a time
(Revealer): window i is a 32 MiB slice at a pool slot given by a keyed permutation (--noise-perm 1), so no slot recurs
while it could still be in the ring. The prover (Streamer, the native C++ thread in gpu/native.py) copies each
revealed window into a small HBM ring and reports completion; in saturating mode (VARENA_SAT_LEAD) the next window is
revealed as soon as fewer than LEAD are outstanding, so the copies run back to back at the link's rate.

The verifier (Checker) stamps completions on its own clock, measures the rate, and content-checks ring units sampled
by the challenges against its own copy of the noise.
"""

from __future__ import annotations

import mmap
import os
import threading
import time

import numpy as np
import torch

from schemes.memory_link_exhaustion.encodings import ref

ALIGN = 4096


def window_slots(pool_bytes, window_bytes):
    return (pool_bytes - window_bytes) // ALIGN + 1


def _aes_fill_part(path, lo, hi, key, cpu):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    try:
        os.sched_setaffinity(0, {cpu})
    except Exception:
        pass
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, os.path.getsize(path))
    os.close(fd)
    mv = memoryview(mm)
    CH = 1 << 20
    zero = bytes(CH)
    enc = Cipher(algorithms.AES(key), modes.CTR(((1 << 127) | (lo // 16)).to_bytes(16, "big"))).encryptor()
    for o in range(lo, hi, CH):
        n = min(CH, hi - o)
        if o + n + 15 <= len(mv):
            enc.update_into(zero[:n], mv[o:o + n + 15])     # (cryptography wants 15 B of slack)
        else:                                               # end of the file: no slack there
            mv[o:o + n] = enc.update(zero[:n])


def create_pool_aes(path: str, nbytes: int, key: bytes, nproc: int = 24, cpus=None):
    """A large random pool filled in parallel with AES-CTR keystream (seconds for tens of GiB)."""
    import multiprocessing as mp
    with open(path, "wb") as f:
        f.truncate(nbytes)
    cl = sorted(cpus) if cpus else list(range(nproc))
    step = -(-nbytes // nproc)
    step = (step + (1 << 20) - 1) // (1 << 20) * (1 << 20)
    ps = [mp.get_context("fork").Process(target=_aes_fill_part,
                                         args=(path, k * step, min(nbytes, (k + 1) * step), key, cl[k % len(cl)]))
          for k in range(nproc) if k * step < nbytes]
    [q.start() for q in ps]
    [q.join() for q in ps]
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, nbytes)
    os.close(fd)
    return mm, np.frombuffer(mm, dtype=np.uint32)


def create_pool(path: str, nbytes: int, seed: int | None):
    """Create (and, if seed is given, fill with random noise) a shared host buffer."""
    with open(path, "wb") as f:
        f.truncate(nbytes)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, nbytes)
    os.close(fd)
    arr = np.frombuffer(mm, dtype=np.uint32)
    if seed is not None:
        rng = np.random.default_rng(seed)
        step = 1 << 26
        for j in range(0, arr.size, step):
            arr[j:j + step] = rng.integers(0, 1 << 32, size=min(step, arr.size - j),
                                           dtype=np.uint32)
    return mm, arr


def attach(path: str):
    nbytes = os.path.getsize(path)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, nbytes)
    os.close(fd)
    return mm


def pin(mm) -> torch.Tensor:
    """Register an mmap'ed host buffer with CUDA and wrap it as an int32 tensor."""
    t = torch.frombuffer(mm, dtype=torch.int32)
    err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * 4, 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")
    if not t.is_pinned():
        raise RuntimeError("registered host buffer is not seen as pinned by torch")
    return t


class Streamer:
    """Prover side: the scheduled noise stream. The copies run in a C++ thread (gpu/native.py sstream_start, no GIL);
    a Python thread only does the bookkeeping. Call start() / stop()."""

    def __init__(self, pool_path, key: int, window_bytes=32 << 20, lag=3, device="cuda", on_window=None,
                 ring_bytes=0, sched=None):
        assert sched, "the noise stream is verifier-scheduled (sched = dict(rev_path, N))"
        self.key, self.W, self.lag = key, window_bytes, lag
        # sched: dict(rev_path, N). Window i starts only once the verifier has revealed it
        # (rev[i % N] = i << 32 | s_i); s_i picks the pool slice
        self.sched = sched
        self.rev_mm = attach(sched["rev_path"])
        # long-term pin (cudaHostRegister) of the shared schedule pages: unpinned tmpfs pages can be
        # migrated by memory compaction, and a write to a page under migration blocks (reveal /
        # report stalls of ~0.3 s were seen on a host with heavy compaction)
        try:
            self.rev_pin = pin(self.rev_mm)
        except Exception:
            self.rev_pin = None
        self.rev = np.frombuffer(self.rev_mm, dtype=np.int64)
        self.rev_n = int(sched["N"])
        self.Ww = window_bytes // 4
        self.pool_mm = attach(pool_path)
        self.pool = pin(self.pool_mm)
        self.n_slots = window_slots(self.pool.numel() * 4, window_bytes)
        # noise ring: window i lands in slot i % n_ring (a big ring is part of the arena)
        self.n_ring = max(lag, ring_bytes // window_bytes)
        self.ring = torch.empty(self.n_ring, self.Ww, dtype=torch.int32, device=device)
        self.s_h2d = torch.cuda.Stream(priority=-1)      # high-priority noise stream
        self.on_window = on_window        # f(i, slot, s, 0, t)
        self.running = False
        self.n_done = 0
        self.times = []                   # completion times, for rate EMA
        self.thread = None
        self.pause = 0.0                  # cheater hook: seconds to idle per window
        self.last = (-1, 0)               # (index, s) of the newest completed window
        self.last_arr = torch.full((1,), -1, dtype=torch.int64).pin_memory()
        self.last_np = self.last_arr.numpy()
        self.S = window_bytes // (64 << 10)
        self.loc_cap = 1 << 14
        self.loc = np.zeros((self.loc_cap, 4), dtype=np.int64)
        self.loc_head = np.zeros(1, dtype=np.int64)
        self.run_flag = np.zeros(1, dtype=np.int32)
        self.pause_ns = np.zeros(1, dtype=np.int64)
        self.hb = np.zeros(1, dtype=np.int64)

    def hbm_bytes(self):
        return self.ring.numel() * 4

    def start(self):
        from schemes.memory_link_exhaustion.gpu import native
        self.running = True
        self.run_flag[0] = 1
        self.pause_ns[0] = int(self.pause * 1e9)
        self._nm = native.mod()
        self._nm.sstream_start(
            self.rev.ctypes.data, self.rev_n, self.pool.data_ptr(), self.n_slots, ALIGN, self.W,
            self.ring.data_ptr(), self.n_ring, self.lag, self.s_h2d.cuda_stream,
            self.loc.ctypes.data, self.loc_cap, self.loc_head.ctypes.data, self.last_np.ctypes.data,
            self.run_flag.ctypes.data, self.pause_ns.ctypes.data, self.ring.device.index or 0,
            self.hb.ctypes.data)
        self.thread = threading.Thread(target=self._drain, daemon=True, name="pcie-drain")
        self.thread.start()

    def _drain(self, final=False):
        """Python bookkeeping for the native stream: completion times, last window, on_window."""
        tail = getattr(self, "_tail", 0)
        # VARENA_STALL_DUMP=1 (noise-dip diagnosis): a C watchdog (faulthandler, no GIL needed) dumps
        # every thread's Python stack whenever one iteration of this loop is held up > 100 ms (stall of the GIL,
        # a GC pass, a driver call ...); re-armed each iteration, so a dump marks exactly the stalled moment
        _sd = (not final) and os.environ.get("VARENA_STALL_DUMP", "0") == "1"
        if _sd:
            import faulthandler as _fh
            _sdf = open(os.environ.get("VARENA_STALL_DUMP_FILE", "/tmp/varena_stall_dump.log"), "a", buffering=1)
            _sd_rep = os.environ.get("VARENA_STALL_DUMP_REPEAT", "0") == "1" or os.path.exists("/root/diag_repeat")
        while True:
            if _sd:
                _fh.cancel_dump_traceback_later()
                _now = time.perf_counter()
                _gap = _now - getattr(self, "_sd_last", _now)
                if _gap > 0.1:      # the stall that the dump above (if any) captured has just ended
                    print(f"--- drain stall ended {time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d} "
                          f"after {_gap * 1e3:.0f} ms (windows done {self.n_done})", file=_sdf)
                self._sd_last = _now
                # VARENA_STALL_DUMP_REPEAT=1 (or /root/diag_repeat present): dump every 100 ms for the whole stall,
                # so the dumps show what the stalled thread does over time, not just at 100 ms
                _fh.dump_traceback_later(0.1, repeat=_sd_rep, file=_sdf)
            h = int(self.loc_head[0])
            if h - tail > self.loc_cap:
                tail = h - self.loc_cap
            while tail < h:
                i, off, s, t_ns = (int(x) for x in self.loc[tail % self.loc_cap])
                tail += 1
                t = t_ns * 1e-9
                self.last = (i, s)
                self.n_done += 1
                self.times.append(t)
                if self.on_window:
                    self.on_window(i, off, s, 0, t)
            if len(self.times) > 8192:
                del self.times[:4096]
            self._tail = tail
            if final or not self.running:
                if _sd:
                    _fh.cancel_dump_traceback_later()
                return
            self.pause_ns[0] = int(self.pause * 1e9)
            hb = int(self.hb[0])
            if hb and time.perf_counter() - hb * 1e-9 > 0.15 and not getattr(self, "_dumped", False):
                # a copy enqueue has been blocked > 150 ms: who holds the driver? (all Python stacks)
                import faulthandler, sys
                self._dumped = True
                print(f"[sstream] enqueue blocked {time.perf_counter() - hb * 1e-9:.3f} s; Python stacks:",
                      file=sys.stderr, flush=True)
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            if not hb:
                self._dumped = False
            time.sleep(0.002)

    def stop(self):
        self.running = False
        if getattr(self, "_nm", None) is not None:
            self.run_flag[0] = 0
            self._nm.sstream_join()
            if self.thread:
                self.thread.join()
            self._drain(final=True)

    def rate(self, horizon=1.0):
        """Recent window rate in GB/s (prover's own view, used by pacing)."""
        ts = self.times
        if len(ts) < 3:
            return 0.0
        t1 = ts[-1]
        k = len(ts) - 1
        while k > 0 and t1 - ts[k - 1] < horizon:
            k -= 1
        n = len(ts) - 1 - k
        dt = t1 - ts[k]
        return n * self.W / dt / 1e9 if dt > 0 else 0.0


FZ_N = 3 * 16384       # freeze log: (start us, end us, credit us) per Revealer self-freeze, from t0
DL_N = 1 << 23         # per-window arrival log (us from t0): 84 min of windows at 56 GB/s (2^21 held only 21 min,
                       # and a 30-minute session was then scored at 39.7 GB/s: 14 Nov)
LINK_FRAC = 50.0 / 55.5   # schedule rate / measured one-way link max: the fastest catch-up
HIST = 1002            # lateness histogram: bin 0 = on time (<= 0 ms), bin k = (k-1, k] ms, last = more
STATS = 16             # n, late, max_us, sum_us, seq, reveal_late, reveal_max, handshake, late_adj, max_adj_us,
                       # freezes, frozen_us, freeze events logged, freeze credit us


def rev_words(N):
    """Shared schedule file (int64): [0, N) reveals, [N, 2N) prover completions,
    [2N, 2N + STATS) counters, the lateness histogram, then N spare words (once the fresh-noise marks; the native
    revealer's layout is unchanged)."""
    return 3 * N + STATS + HIST


def _reveal_loop(path, N, vkey, delta, lead, t0, slack, grace, core=None, link_frac=None, perm=None):
    """Separate process (no GIL shared with the busy verifier): publish window i's secret at
    t0 + i*delta - lead, and timestamp the prover's completion reports (rep[j % N] = j << 32 | s_j)
    as they appear: lateness = arrival - (t0 + j*delta), counted here, GIL-free on both sides."""
    if core is not None:
        try:
            os.sched_setaffinity(0, {core})
        except Exception:
            pass
    try:
        os.nice(-10)
    except Exception:
        pass
    import gc
    gc.disable()
    ppid = os.getppid()       # exit with the verifier (spawned children outlive a kill -9 otherwise)
    n_chk = 0
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, rev_words(N) * 8)
    os.close(fd)
    a = np.frombuffer(mm, dtype=np.int64)
    rev, rep, st = a[:N], a[N:2 * N], a[2 * N:2 * N + STATS]
    hist = a[2 * N + STATS:2 * N + STATS + HIST]
    ktab, ktab_n = None, 0
    if perm:                # keyed permutation schedule (--noise-perm): rebuilt here from the secret, never in a file
        E, S_ = int(perm["E"]), int(perm["S"])
        NE = int(perm.get("n_epochs", 4))
        ktab = np.ascontiguousarray(np.concatenate(
            [ref.perm_keys_range(perm["key"], e * (E // S_), E // S_, S_, perm["pps"], perm["ns"], perm["horizon"])
             for e in range(NE)]), dtype=np.uint32)
        ktab_n = NE * E

        def _refill():
            # the schedule never wraps: the table is a ring of NE epochs of E windows; once the reports enter epoch e,
            # epochs up to e + NE - 2 are generated into the slots epochs <= e - 2 used, so the revealer may run up to
            # an epoch ahead of the reports without ever reading a slot being rewritten
            filled = NE - 1
            poll = max(0.001, min(0.5, E * float(delta) / 8))     # an epoch lasts ~E * delta s (~40 min at 55 GB/s)
            while True:
                time.sleep(poll)
                need = int(st[4]) // E + NE - 2
                while filled < need:
                    filled += 1
                    h = filled % NE
                    ktab[h * E:(h + 1) * E] = ref.perm_keys_range(perm["key"], filled * (E // S_), E // S_, S_,
                                                                  perm["pps"], perm["ns"], perm["horizon"])
        threading.Thread(target=_refill, daemon=True, name="perm-refill").start()
    nat = None
    if os.environ.get("VARENA_NATIVE_REVEALER", "1") == "1":
        try:
            from schemes.memory_link_exhaustion.gpu import native
            nat = native.mod()          # compile/load before the handshake
        except Exception:
            nat = None
    if t0 is None:          # handshake: ready (imports done), then the parent publishes the clock
        st[7] = 1
        while int(st[7]) == 1:
            time.sleep(0.001)
        t0 = float(st[7:8].view(np.float64)[0])
    if nat is not None:     # GIL-free C++ loop, same layout and semantics as below
        dfd = os.open(path + ".dl", os.O_RDWR)
        dmm = mmap.mmap(dfd, DL_N * 8)
        os.close(dfd)
        dla = np.frombuffer(dmm, dtype=np.int64)
        due_ptr = 0                 # constant-rate schedule (the native loop also takes a per-window due table)
        fza = None
        if os.path.exists(path + ".fz"):
            ffd = os.open(path + ".fz", os.O_RDWR)
            fmm = mmap.mmap(ffd, FZ_N * 8)
            os.close(ffd)
            fza = np.frombuffer(fmm, dtype=np.int64)
        nat.revealer(a.ctypes.data, N, int(vkey) & 0xFFFFFFFF, float(delta), float(lead), float(t0),
                     float(slack), float(grace), STATS, HIST, float(delta) * float(link_frac or LINK_FRAC),
                     dla.ctypes.data, DL_N, due_ptr, ktab.ctypes.data if ktab is not None else 0, int(ktab_n),
                     fza.ctypes.data if fza is not None else 0, FZ_N if fza is not None else 0)
        return
    i = j = 0
    n = late = mx = sm = 0

    def poll():
        nonlocal j, n, late, mx, sm
        while int(rep[j % N]) >> 32 == j:
            due = t0 + j * delta
            if due - t0 >= grace:
                lat = time.perf_counter() - due
                us = int(lat * 1e6)
                n += 1
                late += lat > slack
                mx = max(mx, us)
                sm += us
                hist[0 if us <= 0 else min(HIST - 1, 1 + (us - 1) // 1000)] += 1
                st[0], st[1], st[2], st[3] = n, late, mx, sm
            j += 1
            st[4] = j            # reports seen in order (incl. grace)

    while True:
        poll()
        n_chk += 1
        if n_chk % 2048 == 0 and os.getppid() != ppid:
            return
        t = t0 + i * delta - lead
        dt = t - time.perf_counter()
        if dt > 0.0003:
            time.sleep(min(dt - 0.0002, 0.0005))
            continue
        while time.perf_counter() < t:
            pass
        rev[i % N] = (i << 32) | (int(ktab[i % ktab_n]) if ktab is not None else ref.sched_key(vkey, i))
        rd = int((time.perf_counter() - t) * 1e6)       # verifier-side reveal delay
        if rd > 5000:
            st[5] += 1
        if rd > st[6]:
            st[6] = rd
        i += 1


class _SchedDict(dict):
    """Window i -> value, computed from the verifier's key on first use (secret / pool slot)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def __missing__(self, i):
        if i < 0:
            raise KeyError(i)
        v = self.f(i)
        self[i] = v
        return v

    def get(self, i, default=None):
        try:
            return self[i]
        except KeyError:
            return default


class Revealer:
    """Verifier side of the scheduled noise: window i is due at t0 + i*delta; its secret s_i is
    published lead seconds before (rev[i % N] = i << 32 | s_i in a shared-memory page the prover
    polls), by a separate process so the busy verifier's GIL can't delay it. Until then the prover
    can't know which pool slice window i is, nor its echo key."""

    def __init__(self, path, N, vkey, delta, lead, checker, core=None, link_frac=None, perm=None):
        """perm: None, or dict(key=16+ secret bytes, horizon=windows) for the keyed permutation schedule
        (ref.perm_keys): window starts walk a fresh keyed permutation of the pool's whole-window slots per cycle."""
        self.core = core
        self.link_frac = link_frac
        self.mm, _ = create_pool(path, rev_words(N) * 8, None)
        self.dl_mm, _ = create_pool(path + ".dl", DL_N * 8, None)
        self.dl = np.frombuffer(self.dl_mm, dtype=np.int64)
        self.dl[:] = -1
        self.fz_mm, _ = create_pool(path + ".fz", FZ_N * 8, None)
        self.fz = np.frombuffer(self.fz_mm, dtype=np.int64)
        self.fz[:] = 0
        self.all = np.frombuffer(self.mm, dtype=np.int64)
        self.all[:2 * N] = -1
        self.all[2 * N:] = 0
        self.rev = self.all[:N]
        checker.sched_stats = self.stats
        checker.freeze_events = self.freeze_events
        self.path, self.N, self.vkey, self.delta, self.lead, self.ck = path, N, vkey, delta, lead, checker
        ns = window_slots(checker.pool.size * 4, checker.W)
        self.ktab_n = 0
        if perm:
            pool_b, W = checker.pool.size * 4, checker.W
            assert pool_b % W == 0 and W % ALIGN == 0, (pool_b, W)
            S = pool_b // W
            horizon = int(min(perm["horizon"], S // 4))
            NE = 4                                              # epochs in the revealer's table ring
            E = max(S, (ref.PERM_TAB_N // NE) // S * S)       # windows per epoch; the schedule continues past it
            key, pps = bytes(perm["key"]), W // ALIGN
            eps = {}

            def epoch(e):
                if e not in eps:
                    if len(eps) > 3:
                        eps.pop(min(eps))
                    eps[e] = ref.perm_keys_range(key, e * (E // S), E // S, S, pps, ns, horizon)
                return eps[e]

            def val(i):
                return int(epoch(i // E)[i % E])
            self.ktab = np.concatenate([epoch(e) for e in range(NE)])   # verifier memory only (rebuilt by the revealer)
            self.ktab_n = NE * E
            self.perm_args = {"key": key, "n": NE * E, "E": E, "n_epochs": NE, "S": S, "pps": pps, "ns": ns,
                              "horizon": horizon}
            self.perm_val = val
            checker.d0_all = _SchedDict(val)
            checker.off_all = _SchedDict(lambda i: ref.sched_offset(val(i), ns))
            checker.d0 = checker.d0_all
            checker.perm = {"slots": S, "horizon": horizon, "tab_n": NE * E, "epoch": E}
        else:
            checker.d0_all = _SchedDict(lambda i: ref.sched_key(vkey, i))
            checker.d0 = checker.d0_all
        if self.ktab_n:
            pass                          # off_all set above
        else:
            checker.off_all = _SchedDict(lambda i: ref.sched_offset(ref.sched_key(vkey, i), ns))
        self.proc = None

    def start(self):
        import multiprocessing as mp
        self.t0 = None           # set after the spawned Revealer is ready (it imports for seconds)
        # spawn, not fork: a child forked from the big verifier/vLLM-frontend process stalled for up to
        # ~330 ms (copy-on-write faults on its inherited heap), delaying reveals; a fresh interpreter
        # on its own core doesn't
        self.proc = mp.get_context("spawn").Process(
            target=_reveal_loop, args=(self.path, self.N, self.vkey, self.delta, self.lead, self.t0,
                                       self.ck.sched["slack"], self.ck.sched.get("grace", 0.0),
                                       self.core, self.link_frac,
                                       getattr(self, "perm_args", None)),
            daemon=True)
        self.proc.start()
        st = self.all[2 * self.N:2 * self.N + STATS]
        tw = time.perf_counter()
        while int(st[7]) != 1 and time.perf_counter() - tw < 60:
            time.sleep(0.005)
        self.t0 = time.perf_counter() + 0.02
        self.ck.sched_t0 = self.t0
        st[7:8].view(np.float64)[0] = self.t0

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()

    def freeze_events(self, absolute=False):
        """The Revealer's own freezes so far, as an (n, 3) float array of (start s, end s, credit s); times from t0
        (absolute=True: on the verifier's clock). Verifier-side evidence: written by the Revealer loop only."""
        k = int(self.all[2 * self.N + 12])
        ev = np.asarray(self.fz[:3 * k], dtype=np.float64).reshape(-1, 3) / 1e6
        if absolute and len(ev) and self.t0 is not None:
            ev = ev + np.array([self.t0, self.t0, 0.0])
        return ev

    def freeze_credit(self, t_rel):
        """Link time (s) the Revealer's own freezes cost before t_rel seconds after t0 (verifier/floor.py)."""
        from schemes.memory_link_exhaustion.verifier import floor as _fl
        return _fl.credit_before(self.freeze_events(), float(t_rel))

    def stats(self):
        """Revealer-side lateness of the prover's completion reports (None if it sent none)."""
        N = self.N
        st = self.all[2 * N:2 * N + STATS]
        hist = self.all[2 * N + STATS:2 * N + STATS + HIST].copy()
        n = int(st[0])
        if n == 0:
            return None
        c = np.cumsum(hist)

        def q(f):
            k = int(np.searchsorted(c, f * n))
            return 0.0 if k == 0 else float(k)            # ms (bin upper edge)
        return {"n": n, "seq": int(st[4]), "late": int(st[1]), "reveal_late_5ms": int(st[5]),
                "late_reveal_adjusted": int(st[8]), "lateness_adj_ms_max": int(st[9]) / 1e3,
                "verifier_freezes": int(st[10]), "verifier_frozen_s": int(st[11]) / 1e6,
                "verifier_freeze_events": int(st[12]), "verifier_freeze_credit_s": int(st[13]) / 1e6,
                "reveal_delay_ms_max": int(st[6]) / 1e3, "lateness_ms_max": int(st[2]) / 1e3,
                "lateness_ms_mean": int(st[3]) / n / 1e3, "lateness_ms_p50": q(0.5),
                "lateness_ms_p99": q(0.99), "lateness_ms_p999": q(0.999)}


class Checker:
    """Verifier side. Feed it the prover's window reports."""

    def __init__(self, pool_arr: np.ndarray, key: int, window_bytes: int, lag: int, seed=0):
        self.d0_all = {}               # window -> revealed value s_i
        self.off_all = {}              # window -> pool offset
        self.pool, self.key = pool_arr, key
        self.W, self.Ww, self.lag = window_bytes, window_bytes // 4, lag
        self.n_slots = window_slots(pool_arr.size * 4, window_bytes)
        self.next_i = 0
        self.rng = np.random.default_rng(seed)
        self.bad_offsets = 0
        self.gaps = 0
        self.first = None              # (verifier time, i)
        self.last = None
        self.rate_series = []          # (verifier time, cumulative windows)
        self.sched = None              # dict(delta, slack), set by the verifier before the stream starts
        self.sched_t0 = None
        self.late = 0
        self.lateness = []

    def feed(self, windows, t_recv):
        for i, off, d0, d1 in windows:
            if i != self.next_i:
                self.gaps += 1
            self.next_i = i + 1
            if off != self.off_all.get(i) or d0 != self.d0_all.get(i):
                self.bad_offsets += 1
            due = self.sched_t0 + i * self.sched["delta"]
            lat = t_recv - due
            if due - self.sched_t0 < self.sched.get("grace", 0.0):
                continue                      # start-up: ring filling
            self.lateness.append(lat)
            if len(self.lateness) > 200000:
                del self.lateness[:100000]
            if lat > self.sched["slack"]:
                self.late += 1
        if windows:
            last_i = windows[-1][0]
            if self.first is None:
                self.first = (t_recv, last_i)
            self.last = (t_recv, last_i)
            self.rate_series.append((t_recv, last_i))

    def rate(self, t0=None, t1=None):
        """GB/s on the verifier's clock between t0 and t1 (default: whole run)."""
        s = self.rate_series
        if t0 is not None:
            s = [x for x in s if t0 <= x[0] <= (t1 or 1e30)]
        if len(s) < 2:
            return 0.0
        (ta, ia), (tb, ib) = s[0], s[-1]
        return (ib - ia) * self.W / (tb - ta) / 1e9 if tb > ta else 0.0

    def summary(self):
        lt = np.sort(np.asarray(self.lateness)) if self.lateness else np.zeros(1)
        sch = {"sched": True, "late": self.late, "lateness_ms_p50": round(float(lt[len(lt) // 2]) * 1e3, 3),
               "lateness_ms_p99": round(float(lt[int(0.99 * (len(lt) - 1))]) * 1e3, 3)} if self.sched else {}
        rs = self.sched_stats() if (self.sched and getattr(self, "sched_stats", None)) else None
        if rs:
            # authoritative: arrival times stamped by the verifier's Revealer process (no GIL on
            # either side); the report-socket lateness above is kept for reference
            sch = {"sched": True, "late": rs["late"], "sched_n": rs["n"], "sched_seq": rs["seq"],
                   "late_reveal_adjusted": rs["late_reveal_adjusted"],
                   "lateness_adj_ms_max": rs["lateness_adj_ms_max"],
                   "reveal_late_frac": round(rs["reveal_late_5ms"] / max(1, rs["seq"]), 5),
                   "verifier_freezes": rs["verifier_freezes"], "verifier_frozen_s": rs["verifier_frozen_s"],
                   "verifier_freeze_events": rs.get("verifier_freeze_events", 0),
                   "verifier_freeze_credit_s": rs.get("verifier_freeze_credit_s", 0.0),
                   "freeze_events_abs": [[round(float(x), 6) for x in e] for e in
                                         (self.freeze_events(absolute=True) if getattr(self, "freeze_events", None) else [])][:2000],
                   "reveal_late_5ms": rs["reveal_late_5ms"], "reveal_delay_ms_max": rs["reveal_delay_ms_max"],
                   "lateness_ms_p50": rs["lateness_ms_p50"], "lateness_ms_p99": rs["lateness_ms_p99"],
                   "lateness_ms_p999": rs["lateness_ms_p999"], "lateness_ms_max": rs["lateness_ms_max"],
                   "late_socket": self.late,
                   "lateness_socket_ms_p99": round(float(lt[int(0.99 * (len(lt) - 1))]) * 1e3, 3)}
        return {**sch, "windows": self.next_i, "gaps": self.gaps, "bad_offsets": self.bad_offsets,
                "rate_gbs": round(self.rate(), 3), "rate_var": self.rate_var()}

    def rate_var(self):
        """Achieved noise GB/s over consecutive 1 s / 10 s / 60 s intervals (min, mean, max, sd): how small a
        stolen share a rate rule could see."""
        s = self.rate_series
        if len(s) < 4:
            return None
        t = np.array([x[0] for x in s]); w = np.array([x[1] for x in s], dtype=np.float64)
        out = {}
        for L in (1.0, 10.0, 60.0, 300.0):
            edges = np.arange(t[0] + 5.0, t[-1] - 2.0, L)       # skip start-up
            if len(edges) < 3:
                continue
            cw = np.interp(edges, t, w)
            g = np.diff(cw) * self.W / L / 1e9
            out[f"{int(L)}s"] = {"n": len(g), "min": round(float(g.min()), 3), "mean": round(float(g.mean()), 3),
                                 "max": round(float(g.max()), 3), "sd": round(float(g.std()), 4),
                                 "worst_dev": round(float(np.abs(g - g.mean()).max()), 4)}
        s1 = np.diff(np.interp(np.arange(t[0] + 5.0, t[-1] - 2.0, 1.0), t, w)) * self.W / 1e9
        out["series_1s"] = [round(float(x), 3) for x in s1][:4000]
        # the Revealer's freeze credit that falls in each of those seconds (for the windowed floor)
        fe = getattr(self, "freeze_events", None)
        ev = fe(absolute=True) if fe is not None else []
        if len(ev):
            from schemes.memory_link_exhaustion.verifier import floor as _fl
            out["freeze_credit_1s"] = [round(float(x), 4) for x in
                                       _fl.credit_bins(ev, float(t[0]) + 5.0, len(s1))][:4000]
        # sliding-window minima (1 s steps) for the windowed floor rule; the last 10 s (shutdown) are left out
        sm = {}
        body = s1[:-10] if len(s1) > 20 else s1
        for L in (10, 60):
            if len(body) >= L:
                c = np.concatenate([[0.0], np.cumsum(body)])
                sm[f"{L}s"] = round(float(((c[L:] - c[:-L]) / L).min()), 3)
        out["slide_min"] = sm
        return out
