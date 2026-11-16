"""Verifier for inference on vLLM (memory + link exhaustion).

  python -m schemes.memory_link_exhaustion.verifier.verifier --model DIR --out run.json [--calib calib.json]
         [--baseline-tokens base.json] [--write-calib calib.json] [--prover cheater --cheater-spec JSON]
  (the final configuration's flags: scripts/reproduce/reference.env and lib.sh)

It owns the secrets (weight / KV keys, filler key, noise keys), creates the noise pool, and is the client: it starts
vLLM with VARENA=1 (the EngineCore process is the prover, see prover.py) or the cheater prover, sends the prompts and
measures throughput on its own clock. While generation runs it

  * memory: sends the challenge at random times (one mixed probe chase over W, K, F and the noise ring) and times
    every answer against the calibration; afterwards replays one chain of every answer from its own sources (W and
    filler from the keys, the ring from its own noise, KV from a reference recompute and the block table it derived
    itself);
  * PCIe: streams noise on a public schedule and measures the rate from its own completion stamps; the
    end-of-session and head rules (verifier/floor.py) apply when armed.

Thresholds come from a calibration file written by an honest run on the same host (--write-calib): challenge timing,
allowance (measured outside bytes + margin), pad.
"""

from __future__ import annotations

import os

# The verifier's own CPU thread pools (OpenMP via torch/numba): bounded and passive. The box has
# 256 cores but an ~82-CPU cgroup quota; 250 spinning idle OpenMP threads in this process ate the
# quota and throttled everything, vLLM's EngineCore included. Restored before the LLM is created,
# so the engine sees the same environment as the stock baseline (bench.py).
# (NUMBA_NUM_THREADS is not restored: numba refuses a different value once its pool has started)
_ENV0 = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")}
os.environ.update({"OMP_NUM_THREADS": "16", "OMP_WAIT_POLICY": "PASSIVE", "NUMBA_NUM_THREADS": "16"})
import torch as _torch  # noqa: E402  (load libgomp under the bounded settings)
_torch.set_num_threads(16)

import argparse
import json
import random
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from schemes.memory_link_exhaustion.encodings import kvbfly, ref
from schemes.memory_link_exhaustion.core.arena import Entry, Layout
from schemes.memory_link_exhaustion.verifier.memcheck import Expect, WeightSource
from schemes.memory_link_exhaustion.core.proto import Conn
from baselines.inference import common
from schemes.memory_link_exhaustion.verifier.pad import pad_calib, resolve_pad

PAGE = ref.PAGE


def _same_socket(cpus, root="/sys/devices/system/cpu"):
    """Every CPU on the physical package of cpus[0], from the kernel's topology. Empty if it cannot be read. (The
    earlier arithmetic guess added the OTHER socket on two-socket hosts without sub-NUMA clustering, which put the
    noise pool and the Revealer across sockets whenever the GPU was not on socket 0.)"""
    def pkg(c):
        with open(f"{root}/cpu{c}/topology/physical_package_id") as f:
            return f.read().strip()
    try:
        p0 = pkg(cpus[0])
        return {c for c in range(os.cpu_count() or 0) if os.path.exists(f"{root}/cpu{c}/topology/physical_package_id")
                and pkg(c) == p0}
    except Exception:
        return set()


def _gpu_cpus():
    """CPU cores local to GPU 0 (nvidia-smi topo 'CPU Affinity'), minus the first 4 of them."""
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.strip().startswith("GPU0") and "X" in line:
                f = line.split()
                for tok in f:
                    if "-" in tok and tok.replace("-", "").replace(",", "").isdigit():
                        cpus = []
                        for part in tok.split(","):
                            lo, hi = part.split("-")
                            cpus += list(range(int(lo), int(hi) + 1))
                        # plus the other NUMA nodes of the same socket (sub-NUMA clustering halves a
                        # node's memory channels; the generators' first touch spreads the ring's pages)
                        return set(cpus[4:]) | (_same_socket(cpus) - set(cpus))
    except Exception:
        pass
    return None


def log(*a):
    print(time.strftime("%H:%M:%S"), "[verifier]", *a, file=sys.stderr, flush=True)


try:
    import numba as nb

    @nb.njit(cache=True, parallel=True, nogil=True)
    def _digest_nb(w, nonce, out):
        n = w.size
        nchunk = 64
        step = (n + nchunk - 1) // nchunk
        a0 = np.zeros(nchunk, dtype=np.uint64)
        a1 = np.zeros(nchunk, dtype=np.uint64)
        for c in nb.prange(nchunk):
            s0 = np.uint32(0)
            s1 = np.uint32(0)
            for j in range(c * step, min(n, (c + 1) * step)):
                ju = np.uint32(j)
                x = np.uint32(w[j] ^ np.uint32(ju * np.uint32(0x9E3779B9)) ^ np.uint32(nonce))
                x = np.uint32(x * np.uint32(0x85EBCA6B))
                x = np.uint32(x ^ (x >> np.uint32(13)))
                x = np.uint32(x * np.uint32(0xC2B2AE35))
                x = np.uint32(x ^ (x >> np.uint32(16)))
                s0 = np.uint32(s0 + x)
                s1 = np.uint32(s1 + np.uint32(x * np.uint32(2 * ju + 1)))
            a0[c] = s0
            a1[c] = s1
        out[0] = np.uint32(a0.sum() & np.uint64(0xFFFFFFFF))
        out[1] = np.uint32(a1.sum() & np.uint64(0xFFFFFFFF))

    def fast_digest(words, nonce):
        out = np.zeros(2, dtype=np.uint32)
        _digest_nb(np.ascontiguousarray(words, dtype=np.uint32), np.uint32(nonce), out)
        return int(out[0]), int(out[1])
except Exception:
    def fast_digest(words, nonce):
        return ref.page_digest(words, nonce)


class ZeroWeights:
    """Declared weights of the all-zero model (threat model: the cheater need not serve)."""

    def range(self, sources, j0, n):
        return np.zeros(n, dtype=np.uint16)


class _SecretRng:
    """The subset of numpy's Generator the verifier uses for secrets, backed by a random.Random (SystemRandom in
    normal runs): integers(lo, hi[, size, dtype]) and choice(n, p)."""

    def __init__(self, r):
        self.r = r

    def integers(self, lo, hi=None, size=None, dtype=np.int64):
        if hi is None:
            lo, hi = 0, lo
        if size is None:
            return lo + self.r.randrange(hi - lo)
        return np.array([lo + self.r.randrange(hi - lo) for _ in range(int(np.prod(size)))], dtype=dtype).reshape(size)

    def choice(self, n, p=None):
        return self.r.choices(range(n), weights=p)[0]


# Total device memory (cudaMemGetInfo total) per declared GPU model; the capacity check uses this, not the
# prover's report. H200 NVL measured on rented boxes.
GPU_CAPACITY = {"H200_NVL": 150111977472}


class Verifier:
    def __init__(self, a):
        self.a = a
        # Secrets (keys, challenge kinds, nonces, arrival times, post-run samples) come from the OS CSPRNG,
        # independent of --seed (which only fixes the prompt order). --secret-seed (debug only) makes them
        # reproducible with a non-cryptographic generator; the seed is logged only in the verifier's own output.
        self.secret_seed = a.secret_seed if a.secret_seed else None
        self.srng = random.Random(a.secret_seed) if a.secret_seed else random.SystemRandom()
        self.rng = _SecretRng(self.srng)
        self.post_seeds = [self.srng.getrandbits(128) for _ in range(3)]   # checker sample, check_memory, reference
        self.master = 1 + self.srng.randrange((1 << 62) - 1)
        self.fkey = 1 + self.srng.randrange((1 << 32) - 1)
        self.pcie_key = 1 + self.srng.randrange((1 << 32) - 1)
        self.ready = threading.Event()
        self.bye = threading.Event()
        self.conn = None
        self.responses = {}
        self.bd = {}
        self.claims = {}
        self.chal_sent = {}
        self.chal_head = {}
        self.cid_lock = threading.Lock()
        self.next_cid = 0
        self.error = None
        self.stats = None
        self.corrupted = []

    # ---------------------------------------------------------------- setup
    def start(self):
        a = self.a
        os.makedirs(a.shm, exist_ok=True)
        self.revealer_core = None
        if a.pcie and a.noise_sched:
            # reserve one GPU-local core for the Revealer: everything this process spawns from here
            # on (vLLM's engine, the prover/cheater) inherits an affinity without it (it was stalled
            # by up to ~300 ms by other threads on its core)
            gc_ = _gpu_cpus()
            if gc_:
                self.revealer_core = (sorted(c for c in gc_ if c >= 32) or sorted(gc_))[0]
                try:
                    allc = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, allc - {self.revealer_core})
                except Exception:
                    pass
        tag = f"{os.getpid()}"
        self.pool_path = os.path.join(a.shm, f"varena_noise_{tag}")
        if a.pcie:
            from schemes.memory_link_exhaustion.link.pcie import Checker, create_pool
            t0 = time.time()
            # a fixed random pool, read window by window on the schedule
            if a.pool_mib >= 8192:
                # large pool (a cheater caching x GiB of it in HBM saves only x / pool of the link):
                # filled in parallel with AES-CTR keystream on the GPU's socket
                from schemes.memory_link_exhaustion.link.pcie import create_pool_aes
                self.pool_mm, self.pool = create_pool_aes(self.pool_path, a.pool_mib << 20,
                                                          bytes(self.rng.integers(0, 256, 16, dtype=np.uint8)),
                                                          24, _gpu_cpus())
            else:
                self.pool_mm, self.pool = create_pool(self.pool_path, a.pool_mib << 20, int(self.rng.integers(1 << 62)))
            log(f"noise pool {a.pool_mib} MiB ready in {time.time() - t0:.1f}s")
            assert a.noise_sched, "the noise stream is verifier-scheduled (--noise-sched 1)"
            self.checker = Checker(self.pool, self.pcie_key, a.window_mib << 20, a.lag, seed=self.post_seeds[0])
            self.revealer = None
            if a.noise_sched:
                from schemes.memory_link_exhaustion.link.pcie import Revealer
                delta = (a.window_mib << 20) / (a.noise_gbs * 1e9)
                self.checker.sched = {"delta": delta, "slack": a.sched_slack_ms / 1e3,
                                      "grace": a.sched_grace_s}
                self.rev_path = os.path.join(a.shm, f"varena_rev_{tag}")
                self.revealer = Revealer(self.rev_path, 1 << 16, int(self.rng.integers(1, 1 << 32)),
                                         delta, a.sched_lead * delta, self.checker,
                                         core=self.revealer_core,
                                         link_frac=min(1.0, a.noise_gbs / a.link_gbs),
                                         perm=self._perm_spec(a))
        self.sock_path = f"/tmp/varena_v{tag}.sock"
        self.native_path = f"/tmp/varena_n{tag}.sock"
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.sock_path)
        self.srv.listen(1)
        threading.Thread(target=self._serve, daemon=True).start()
        self.cfg = {"master": self.master, "fkey": self.fkey, "pcie": a.pcie,
                    "pool_path": self.pool_path,
                    "pcie_key": self.pcie_key, "window_bytes": a.window_mib << 20, "lag": a.lag,
                    "allowance_bytes": int(a.allowance_gib * (1 << 30)), "pad_mib": int(a.pad_mib),
                    "capacity_bytes": int(a.gpu_capacity_bytes or GPU_CAPACITY[a.gpu_model]),
                    **({"prereserve_mib": a.prereserve_mib, "pad_nontorch_mib": a.pad_nontorch_mib}
                       if getattr(a, "prereserve_mib", None) is not None else {}),
                    "switch_interval": a.switch_interval,
                    "noise_sched": {"rev_path": self.rev_path, "N": 1 << 16} if (a.pcie and a.noise_sched) else None,
                    "ring_mib": a.ring_mib if a.pcie else 0,
                    "native_sock": self.native_path if a.native else None,
                    "plain_w": bool(a.plain_w),
                    "chase_K": a.chase_k, "chase_blocks": a.chase_blocks, "ring_tag": int(self._ring_tagged()),
                    "mchase_P": a.mchase_p if (a.mchase_rate > 0 and a.native) else 0,
                    "ring_chase_guard": a.ring_chase_guard if (a.pcie and a.ring_mib > 0) else 0,
                    "kv_tab_test": int(getattr(a, "kv_tab_test", 0) or 0),
                    "kv_table": int(getattr(a, "kv_table", 0) or 0),
                    **(json.loads(a.debug) if a.debug else {})}


    def _perm_spec(self, a):
        """--noise-perm: the schedule secret (128 bits from the OS CSPRNG) and the no-repeat horizon in windows."""
        if not getattr(a, "noise_perm", 0):
            return None
        W = a.window_mib << 20
        n_ring = (a.ring_mib << 20) // W if a.ring_mib else 64
        horizon = n_ring + a.lag + a.sched_lead + 64
        if int(a.noise_perm) == 2:
            log(f"noise schedule: sequential walk of {(a.pool_mib << 20) // W} slots/cycle (public, no secret order)")
            return {"key": ref.SEQ_KEY, "horizon": horizon}
        log(f"noise schedule: keyed permutation of {(a.pool_mib << 20) // W} slots/cycle, no repeat within "
            f"{horizon} windows")
        return {"key": bytes(self.rng.integers(0, 256, 16, dtype=np.uint8)), "horizon": horizon}
    def _serve(self):
        s, _ = self.srv.accept()
        self.conn = Conn(s)
        m = self.conn.recv()
        assert m["op"] == "hello", m
        self.conn.send({"op": "config", "cfg": self.cfg})
        while True:
            m = self.conn.recv()
            t = time.perf_counter()
            if m is None:
                self.bye.set()
                return
            op = m["op"]
            if op == "ready":
                self.info = m["info"]
                if getattr(self, "revealer", None) is not None:
                    if self.a.prover != "cheater":
                        self._rev_deferred = True  # honest serving: started after the untimed vLLM warm-up (run())
                    else:
                        self.revealer.start()      # cheaters (and the idle control): the schedule's clock starts here
                self.ready.set()
            elif op == "response":
                m["t_recv"] = t
                m["head"] = self.checker.next_i - 1 if self.a.pcie else -1
                self.responses[m["id"]] = m
            elif op == "chase_debug":
                self._dbg_resp = m
            elif op == "claims":
                self.claims[m["id"]] = m
            elif op == "pcie":
                self.checker.feed(m["w"], t)
            elif op == "outputs":
                self.outputs[m["id"]] = m["tokens"]
            elif op == "corrupted":
                self.corrupted.append(m["page"])
            elif op == "wl_progress":            # one message per completed call (calls >= 1); 0 = counter reset
                self.__dict__.setdefault("wl_events", []).append((time.perf_counter(), int(m.get("calls", 0))))
            elif op == "bye":
                self.stats = m["stats"]
                self.bye.set()

    # ---------------------------------------------------------------- run
    def _send_challenge(self, kind):
        a = self.a
        with self.cid_lock:
            cid = self.next_cid
            self.next_cid += 1
        nonce = int(self.rng.integers(1, 1 << 32))
        self.chal_head[cid] = self.checker.next_i - 1 if a.pcie else -1
        self.chal_sent[cid] = (time.perf_counter(), nonce, kind)
        if not (a.native and kind in ("mchase", "warmmchase")):
            raise ValueError(f"no such challenge kind: {kind}")     # (the mixed chase is the only challenge)
        self._native_chase(cid, kind, nonce)
        return cid

    def _chase_debug_request(self):
        """--chase-debug: generation is over, so nothing is being written. One more mixed chase, then ask the prover
        to walk one of its chains on the CPU from device memory and send every probe (replay_chains compares)."""
        try:
            time.sleep(1.0)
            cid = self._send_challenge("mchase")
            t0 = time.time()
            while cid not in self.responses and time.time() - t0 < 30:
                time.sleep(0.001)
            m = self.responses[cid]
            # the nonce the kernel actually ran under (returned by the challenge server), not a recomputation: the
            # noise workers have stopped, so the checker may not have window `win` yet
            nonce = int(m["neff"])
            chain = int(self.rng.integers(0, len(m["states"])))
            with self.cid_lock:
                dbg_id = self.next_cid
                self.next_cid += 1
            self._dbg_resp = None
            self.conn.send({"op": "chase_debug", "id": dbg_id, "nonce": int(nonce), "chain": chain})
            t0 = time.time()
            while self._dbg_resp is None and time.time() - t0 < 180:
                time.sleep(0.01)
            # not a timed challenge: take it out of the books the timing rules and content checks read
            self.responses.pop(cid, None)
            self.chal_sent.pop(cid, None)
            self._chase_dbg = {"cid": cid, "chain": chain, "nonce": int(nonce), "m": m, "walk": self._dbg_resp or {}}
            log(f"chase debug: chase {cid} chain {chain}, prover walk "
                f"{'received' if self._dbg_resp is not None else 'MISSING'}")
        except Exception as e:
            log("chase debug request failed:", repr(e))

    def _native_connect(self):
        t0 = time.time()
        while not os.path.exists(self.native_path) and time.time() - t0 < 60:
            time.sleep(0.05)
        self.nsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.nsock.connect(self.native_path)
        from schemes.memory_link_exhaustion.gpu import native
        self.native = native

    def _native_chase(self, cid, kind, nonce):
        a = self.a
        t_sent = self.chal_sent[cid][0]
        gi = 0                                          # the prover's only challenge graph: the mixed chase
        ns, rid, rk, win, neff, serve_ns, payload, srv = self.native.mod().rtt(
            self.nsock.fileno(), cid, gi, nonce)
        t_wall = time.perf_counter()  # verifier clock once the answer is in hand (head rule: after the prover's head read)
        tr, tl_, td = srv
        st_ = np.frombuffer(payload, dtype=np.uint64).copy()
        kvt, tags = None, None
        ntag = a.chase_k * ref.CH_TAG_SLOTS // 2 if getattr(a, "kv_table", 0) == 3 else 0
        if ntag and kind in ("mchase", "warmmchase") and st_.size >= a.chase_k + 2 + ntag:
            # tagged in-flight reads: per chain [number of in-band reads, their tags], after everything else
            tags = st_[-ntag:].view(np.uint32).reshape(a.chase_k, ref.CH_TAG_SLOTS).copy()
            st_ = st_[:-ntag]
        if kind in ("mchase", "warmmchase") and getattr(a, "kv_table", 0) and st_.size >= a.chase_k + 2:
            # settled-KV rule: the kernel appends the block table's (version, digest) it ran under
            kvt = (int(st_[-2]), int(st_[-1]))
            st_ = st_[:-2]
        gpu_ms = None
        if kind in ("mchase", "warmmchase") and \
                st_.size == a.chase_k + 2:
            # gated prover: GPU start / end time of the challenge graph appended (tail study)
            gpu_ms = (int(st_[-1]) - int(st_[-2])) / 1e6
            st_ = st_[:-2]
        self.responses[cid] = {"id": cid, "kind": kind, "win": int(win),
                               "states": st_, "gpu_ms": gpu_ms, "kvt": kvt, "tags": tags, "neff": int(neff) & 0xFFFFFFFF,
                               "t_recv": t_sent + ns / 1e9, "t_recv_wall": t_wall,
                               "head": self.checker.next_i - 1 if a.pcie else -1,
                               "kernel_ms": serve_ns / 1e6, "server_ms": (td - tl_) / 1e6,
                               "srv_ms": ((tl_ - tr) / 1e6, (td - tl_) / 1e6)}

    def _chunk_words(self, exp):
        """chunk_words(page, word_off) for the chase reference: W pages via Expect, F via the filler."""
        info = self.info
        nW, nK = info["nW"], info["nK"]
        cache = self._chase_cache

        def f(page, wo):
            if ref.FILLER in ("indep", "sloth", "bfly") and page >= nW + nK:
                key = (page, wo >> 10)
                if key not in cache:
                    from schemes.memory_link_exhaustion.verifier.memcheck import indep_units
                    cache[key] = indep_units(self.fkey, [page], [wo >> 10])[0]
                u = cache[key]
                return u[(wo & 1023): (wo & 1023) + 32]
            if page not in cache:
                if len(cache) > 20000:
                    cache.clear()
                cache[page] = exp.pages([page])[0]
            return cache[page][wo: wo + 32]
        return f

    def _ring_granules(self):
        """Noise-ring granules in the mixed chase table (--ring-chase-guard), from the verifier's own ring size
        (--ring-mib over its window size), not from what the prover reports."""
        a = self.a
        g = int(getattr(a, "ring_chase_guard", 0) or 0)
        if not (g and getattr(a, "pcie", False) and a.ring_mib > 0):
            return 0
        W = int(getattr(getattr(self, "checker", None), "W", 0) or (int(a.window_mib) << 20))
        return ref.ring_chase_granules((int(a.ring_mib) << 20) // W, g, tagged=self._ring_tagged())

    def _ring_tagged(self):
        """--ring-tag with tagged in-flight reads (--kv-table 3): the chase reads every ring slot and tags the zone."""
        a = self.a
        return bool(getattr(a, "ring_tag", 0)) and int(getattr(a, "kv_table", 0) or 0) == 3

    def _check_mchase(self, cid, m, nonce, exp, mem, rng, bad_pages, kv_real=None, dbg=None):
        """Mixed chase (one kind over W, KV, filler and the activation pool). Content of every region is known for
        the zero-weight model; with real weights live KV is not, so the chain is a timing check only."""
        a, info = self.a, self.info
        st = mem.setdefault("mchase", {"challenges": 0, "chains_checked": 0, "chains_bad": 0})
        st["challenges"] += 1
        if not a.zero_weights and kv_real is None:
            # live KV content is unknown here with real weights: timing only (KV covered by claims). With the
            # settled-KV table the chain is replayed after the reference pass (replay_chains): queue it.
            if getattr(a, "kv_table", 0) and m.get("kvt") is not None:
                self.__dict__.setdefault("_replay_q", []).append((cid, nonce))
            return
        nW, nK, nb = info["nW"], info["nK"], info["nb"]
        nF = info.get("nF_chase", info["R0"] - nW - nK)
        ap_ = info.get("act_pool") or {}
        gl = ap_.get("granules") or []
        gp = ref.ch_granules_mix(nW, nK, nF, len(gl), self._ring_granules())
        tb = len(gp) // ref.CH_GRAN_PER_BLOCK
        nbk = min(a.chase_blocks, tb)
        wf = self._chunk_words(exp)
        ck, head_w = getattr(self, "checker", None), m.get("win")     # (not `head`: words() has a local of that name)
        ktab = getattr(self, "_kv_tab_test", None)
        if ktab is None and int(getattr(a, "kv_tab_test", 0) or 0):
            t_ = ref.kv_tab_test(a.kv_tab_test, nb)
            ktab = self._kv_tab_test = [(int(x) & 0xff, int(x) >> 8) for x in t_]
        st.setdefault("kv_probes", 0); st.setdefault("kv_masked", 0)
        pseed = ref_fmix32((ap_.get("epoch", 0) ^ ap_.get("master", 0)) & 0xFFFFFFFF) if gl else 0
        cache = {}

        def words(page, wo):
            if page >= ref.CH_RA0:
                # noise ring, head-relative: logical slot q holds window (head - q); its words are the pool's at
                # that window's offset. An unknown window (head not yet revealed to the checker) reads as zeros and
                # fails the chain.
                q, w_in = ref.ring_chase_slot(page, wo)

                def win(i):
                    try:                  # off_all computes a window's pool offset on first use (KeyError: none yet)
                        w0 = ck.off_all[i] * 4096 // 4 + w_in
                    except (KeyError, TypeError, AttributeError):
                        return np.zeros(32, np.uint32)
                    return np.asarray(ck.pool[w0:w0 + 32], dtype=np.uint32)
                old = win(int(head_w) - q)
                if rz_n and q >= rz_n - rz_zone:
                    # tagged ring read: the slot holds its old window or the incoming one; the tag the kernel returned
                    # for this read says which (reads are tagged in probe order)
                    k_ = rtag[0]
                    rtag[0] += 1
                    want_t = int(ctags[1 + k_]) if (ctags is not None and k_ < ref.CH_TAG_SLOTS - 1) else None
                    cid_ = ref.ring_tag_id(q, w_in)
                    if want_t is not None and ref.ch_tag_id(nonce, cid_, old) == want_t:
                        st["ring_old"] = st.get("ring_old", 0) + 1
                        return old
                    new = win(int(head_w) - q + rz_n)
                    if want_t is not None and ref.ch_tag_id(nonce, cid_, new) == want_t:
                        st["ring_new"] = st.get("ring_new", 0) + 1
                        return new
                    st["ring_neither"] = st.get("ring_neither", 0) + 1
                return old
            if page >= ref.CH_PA0:
                j, pg = divmod(page - ref.CH_PA0, ref.CH_GRAN_PAGES)
                addr = gl[j] + pg * 65536 + wo * 4
                from schemes.memory_link_exhaustion.encodings import actenc2 as _AEc
                if _AEc.AE_S32:
                    from schemes.memory_link_exhaustion.encodings import sloth32 as _S32
                    return _S32.pool_unit_zero(pseed, addr)
                c = (addr >> 12) & 0xFFFFFFFF
                if ("p", c) not in cache:
                    cache[("p", c)] = kvbfly.chunk_ks(pseed, c, 8, 2)
                u0 = (addr & 4095) >> 4
                return cache[("p", c)][u0:u0 + 8].reshape(-1)
            if nW <= page < nW + nK:
                layer, block = divmod(page - nW, nb)
                st["kv_probes"] += 1
                if kv_real is not None:               # real weights: the block table and the recomputed KV store
                    w_ = kv_real(page, layer, block, wo)
                    if w_ is None:
                        st["kv_masked"] += 1
                        return ref.ch_mask_words(nonce, page - nW, wo)
                    if isinstance(w_, str):               # never allocated: the page still holds the arena filler
                        return wf(page, wo)
                    return w_
                if ktab is not None and ref.ch_masked(ktab[block], wo):
                    st["kv_masked"] += 1                      # settled-KV rule: in-flight band, not read from memory
                    return ref.ch_mask_words(nonce, page - nW, wo)
                if kvbfly.KVENC == "sloth":
                    from schemes.memory_link_exhaustion.encodings import sloth64
                    return sloth64.encode_range(np.zeros(64, np.uint16), sloth64.kv_seed(kv_key(self.master, layer)),
                                                block * (PAGE // 2) + 2 * wo).view(np.uint32)
                return self._kvbf_words(layer, block, wo)
            return wf(page, wo)
        c = int(rng.integers(0, len(m["states"])))
        # tagged ring reads (--ring-tag with --kv-table 3): this chain's [count, tags] and the zone
        rz_n = (int(a.ring_mib) << 20) // int(ck.W) if (self._ring_tagged() and ck is not None) else 0
        rz_zone = int(getattr(a, "ring_chase_guard", 0) or 0)
        ctags = m["tags"][c] if (rz_n and m.get("tags") is not None) else None
        rtag = [0]

        def mask64(page, wo):                  # wide probes: a piece in its block's in-flight band, as the kernel
            if ktab is None or not (nW <= page < nW + nK):
                return None
            if not ref.ch_masked(ktab[(page - nW) % nb], wo):
                return None
            st["kv_masked"] += 1
            return ref.ch_mask_words64(nonce, page - nW, wo, 16 * ref.CH_WIDE)
        if dbg is not None:
            # debug transcript (--chase-debug): the prover's own CPU walk of one chain, probe by probe, against
            # what the verifier expects at each probe; no chain is recomputed here
            import zlib
            reg = lambda p: "ring" if p >= ref.CH_RA0 else "pool" if p >= ref.CH_PA0 else \
                "W" if p < nW else "K" if p < nW + nK else "F"
            diff = {}
            for n_, (page, wo, crc) in enumerate(dbg["probes"]):
                dbg["klast"][0] = None
                e = zlib.crc32(np.ascontiguousarray(np.asarray(words(page, wo), dtype=np.uint32)).tobytes())
                r_ = dbg["klast"][0] or reg(page)
                dbg["by_region"][r_] = dbg["by_region"].get(r_, 0) + 1
                if e != crc:
                    diff[r_] = diff.get(r_, 0) + 1
                    if len(dbg["first"]) < 6:
                        ent = {"probe": n_, "round": n_ // 8, "page": int(page), "wo": int(wo), "region": r_}
                        if nW <= page < nW + nK:
                            ent.update({"layer": int((page - nW) // nb), "block": int((page - nW) % nb),
                                        "token_in_block": int(wo // 1024)})
                        dbg["first"].append(ent)
            dbg["diff_by_region"] = diff
            return
        want = ref.ch_chain(nonce, c, a.mchase_p, tb, nbk, gp, words, mask64 if ref.CH_WIDE > 1 else None)
        st["chains_checked"] += 1
        if rz_n and (ctags is None or int(ctags[0]) != rtag[0]):
            st["tag_count_bad"] = st.get("tag_count_bad", 0) + 1       # the answer's tag count differs from the walk's
            st["chains_bad"] += 1
            bad_pages.append((cid, -1, "mchase_tags"))
        elif int(m["states"][c]) != want:
            st["chains_bad"] += 1
            bad_pages.append((cid, -1, "mchase"))

    def _challenger(self, stop, kinds):
        """One challenge at a time (the prover serves them serially, so overlapping
        ones would measure queueing): Poisson arrivals at the summed rate, kind drawn
        by rate, the next one only after the previous answer."""
        a = self.a
        total = sum(r for _, r in kinds)
        names = [k for k, _ in kinds]
        probs = [r / total for _, r in kinds]
        stop.wait(a.chal_delay)
        self._host0 = host_snapshot()           # host state over the challenged part of the session (report only)
        while not stop.is_set():
            stop.wait(self.srng.expovariate(total))
            if stop.is_set():
                self._host1 = host_snapshot()
                break
            kind = names[int(self.rng.choice(len(names), p=probs))]
            cid = self._send_challenge(kind)
            t0 = time.time()
            while cid not in self.responses and not stop.is_set() and time.time() - t0 < 30:
                time.sleep(0.0005)

    def spot(self, pages, kind="spot"):
        """Digest specific pages now (selftest). Returns the response id."""
        with self.cid_lock:
            cid = self.next_cid
            self.next_cid += 1
        nonce = int(self.rng.integers(1, 1 << 32))
        self.chal_sent[cid] = (time.perf_counter(), nonce, kind)
        self.conn.send({"op": "spot", "id": cid, "nonce": nonce, "pages": [int(p) for p in pages]})
        return cid

    def _ext_generate(self, prompts):
        """Inference through the socket (external prover): batches of 256 prompts."""
        a = self.a
        self.outputs = {}
        ids = list(range(0, len(prompts), 256))
        if getattr(a, "until_calls", 0) > 0:
            # measurement harness: keep the verified session running, one batch at a time, until the cheater's hidden
            # job has completed that many whole calls (so every timed call lies inside the session); --num-prompts is
            # then the cap. Without it the session is num_prompts long and a call still in progress at the end
            # finishes after the checks have stopped.
            sent, t_s0 = 0, time.perf_counter()
            done = lambda: sum(1 for t, c in getattr(self, "wl_events", []) if t >= t_s0 and c >= 1)   # calls ended in the session
            for k, i in enumerate(ids):
                if done() >= a.until_calls or self.bye.is_set():
                    break
                self.conn.send({"op": "generate", "id": k, "prompts": prompts[i:i + 256], "n_new": a.output_len})
                sent += 1
                while len(self.outputs) < sent and not self.bye.is_set():
                    time.sleep(0.005)
            ids = ids[:sent]
            log(f"until-calls {a.until_calls}: session ended after {sent} batches ({time.perf_counter() - t_s0:.0f} s), "
                f"hidden calls completed inside it: {done()}")
        else:
            for k, i in enumerate(ids):
                self.conn.send({"op": "generate", "id": k, "prompts": prompts[i:i + 256],
                                "n_new": a.output_len})
        while len(self.outputs) < len(ids) and not self.bye.is_set():
            time.sleep(0.005)
        toks = []
        for k in range(len(ids)):
            toks += self.outputs.get(k, [])
        return toks

    def run(self):
        a = self.a
        self.start()
        env = {"VARENA": "1", "VARENA_SOCK": self.sock_path}
        for k, v in _ENV0.items():        # engine: same env as the baseline (then BASE_ENV on top)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        common.apply_env(env)
        if a.prover == "cheater":
            return self._run_external()
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        t0 = time.time()
        if getattr(a, "kv_table_derive", 0):
            os.environ["VARENA_SCHED_TRACE"] = a.out + ".schedtrace"       # the engine records its scheduler's call order
        if os.environ.get("VARENA_SCHED_TRACE"):
            # scheduler replay (verifier/schedreplay.py): the engine settings this verifier asked for
            json.dump(common.llm_kwargs(a), open(os.environ["VARENA_SCHED_TRACE"] + ".llm.json", "w"), default=str)
        llm = LLM(**common.llm_kwargs(a))
        self.t_startup = time.time() - t0
        self.ready.wait(600)
        info = self.info
        log("prover ready:", json.dumps({k: v for k, v in info.items() if k != "w_entries"}))
        ps = [TokensPrompt(prompt_token_ids=p) for p in common.prompts(a)]
        sp = common.sampling(a)
        self.sched = common.sched_info(llm)
        log("scheduler:", json.dumps(self.sched))
        if a.warmup_prompts > 0:
            # untimed warm-up at the real shapes (large prefill batches up to the token cap, decode at many batch
            # sizes), so Triton / CUDA-graph first-use compiles happen before timing and challenges start
            from vllm import SamplingParams as _SP
            t_w = time.time()
            # at most the request's own output length: a warm-up that generates past it writes KV for tokens the
            # verifier never receives, and the chain replay cannot recompute those positions (ShareGPT has outputs
            # shorter than 32; greedy decoding makes the warm-up a prefix of the real request)
            nw_ = common.out_lens(a)[:a.warmup_prompts]
            llm.generate(ps[:a.warmup_prompts],
                         [_SP(temperature=0.0, max_tokens=max(1, min(32, int(n_))), ignore_eos=True) for n_ in nw_],
                         use_tqdm=False)
            log(f"warm-up: {a.warmup_prompts} prompts in {time.time() - t_w:.1f}s")
        self.t_warm_end = time.time()
        if getattr(self, "_rev_deferred", False):
            # honest serving: the noise (and its clock) starts only now. The warm-up is untimed and unchallenged, so noise
            # there proves nothing, and its first-use Triton compiles / module loads stall every other GPU call, noise
            # copies included, for ~100-500 ms each. Measured on Box W, 16 Nov: honest end-of-session 56.59-56.62 GB/s
            # with the late start against 56.38-56.42 from prover ready, 0 stalls in 7 runs. (A late start froze honest
            # serving once on 12 Nov, b24 box A, before the routed single queue.)
            self.revealer.start()
            self._rev_deferred = False
            log("noise started after warm-up")
        # untimed warm-up challenges (first-use costs); their pages are still checked
        self._warm_challenges()
        stop = threading.Event()
        threads = []
        if self._kinds():
            threads.append(threading.Thread(target=self._challenger, args=(stop, self._kinds()),
                                            daemon=True))
        for th in threads:
            th.start()
        if a.pcie:
            self.pcie_t0 = time.perf_counter()
        t0 = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        t1 = time.perf_counter()
        self.gen_window = (t0, t1)
        stop.set()
        for th in threads:
            th.join(timeout=30)
        if getattr(a, "chase_debug", 0) and a.native and a.mchase_rate > 0:
            self._chase_debug_request()          # the challenger thread has stopped: the native socket is ours
        self.tokens = [list(o.outputs[0].token_ids) for o in outs]
        n = sum(len(t) for t in self.tokens)
        self.tok_s = n / (t1 - t0)
        log(f"generated {n} tokens in {t1 - t0:.1f}s: {self.tok_s:.1f} tok/s")
        if a.selftest:
            self.selftest()
            # claims are answered from the scheduler loop: run a little work to drive it
            llm.generate(ps[:4], [SamplingParams(temperature=0.0, max_tokens=max(1, min(8, int(n_))), ignore_eos=True)
                                  for n_ in common.out_lens(a)[:4]], use_tqdm=False)
            time.sleep(0.5)
        time.sleep(1.0)
        self.conn.send({"op": "stop"})
        self.bye.wait(300)
        time.sleep(0.5)
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception as e:
            log("shutdown:", e)
        del llm

    def _kinds(self):
        a = self.a
        ks = []
        if a.mchase_rate > 0 and a.mchase_p > 0 and a.native:
            ks.append(("mchase", a.mchase_rate))
        return ks

    def _warm_challenges(self):
        if self.a.native:
            self._native_connect()
        # ring challenges need the ring filled once (slots start uninitialised)
        if self.a.pcie and self.info.get("nR"):
            t0 = time.time()
            while self.checker.next_i < self.info["n_ring"] + self.a.lag + 4 and time.time() - t0 < 120:
                time.sleep(0.05)
        for kind, _ in self._kinds():
            for _ in range(3):
                cid = self._send_challenge("warm" + kind)
                t0 = time.time()
                while cid not in self.responses and time.time() - t0 < 30:
                    time.sleep(0.01)

    def _run_external(self):
        """The prover is schemes.memory_link_exhaustion.attacks.zero_weight_cheater.cheater (no vLLM): prompts go over the socket."""
        a = self.a
        spec = json.loads(a.cheater_spec or "{}")
        calib = json.load(open(a.calib)) if a.calib else {}
        if calib.get("tok_s"):
            spec["rate_tok_s"] = calib["tok_s"] * spec.get("rate_mult", 1.03)
        if calib.get("pcie_gbs"):
            spec["pcie_gbs"] = calib["pcie_gbs"] * spec.get("rate_mult", 1.03)
        env = dict(os.environ, VARENA_SOCK=self.sock_path, VARENA_CHEAT=json.dumps(spec),
                   VARENA_MODEL=a.model, VARENA_NUM_BLOCKS=str(a.num_blocks or 4096))
        self.plog = open(a.out + ".prover.log", "w")
        t0 = time.time()
        self.proc = subprocess.Popen([sys.executable, "-m", "schemes.memory_link_exhaustion.attacks.zero_weight_cheater.cheater"], env=env,
                                     stdout=self.plog, stderr=subprocess.STDOUT)
        t_dead = time.time() + 1200
        while not self.ready.wait(2.0):                  # do not sit out the 20 min if the cheater has already died
            if self.proc.poll() is not None or time.time() > t_dead:
                break
        if not self.ready.is_set():
            self.plog.flush()
            raise RuntimeError(f"the cheater prover stopped before it was ready (exit code {self.proc.poll()}); "
                               f"see {a.out}.prover.log")
        self.t_startup = time.time() - t0
        log("cheater ready:", json.dumps(self.info.get("cheater")))
        # an untrusted prover's scheduler config is self-reported (the verifier cannot read it): token_cap then
        # checks the declared value, as it would in a deployment (; before this every cheater run
        # failed token_cap only because the harness had no in-process engine to read)
        if self.info.get("sched"):
            self.sched = self.info["sched"]
        prompts = common.prompts(a)
        self._warm_challenges()
        stop = threading.Event()
        threads = [threading.Thread(target=self._challenger, args=(stop, self._kinds()),
                                    daemon=True)]
        for th in threads:
            th.start()
        t0 = time.perf_counter()
        self.tokens = self._ext_generate(prompts)
        t1 = time.perf_counter()
        self.gen_window = (t0, t1)
        stop.set()
        for th in threads:
            th.join(timeout=30)
        if getattr(a, "chase_debug", 0) and a.native and a.mchase_rate > 0:
            self._chase_debug_request()          # the challenger thread has stopped: the native socket is ours
        n = sum(len(t) for t in self.tokens)
        self.tok_s = n / (t1 - t0)
        log(f"generated {n} tokens in {t1 - t0:.1f}s: {self.tok_s:.1f} tok/s")
        time.sleep(1.0)
        self.conn.send({"op": "stop"})
        self.bye.wait(600)
        self.proc.kill()

    def selftest(self):
        """Corrupt one byte in a W, an F and a live K page; spot-check them."""
        info = self.info
        nW, nK = info["nW"], info["nK"]
        rng = np.random.default_rng(7)
        w = int(rng.integers(0, nW))
        f = int(rng.integers(nW + nK, info["npages"] - info["n_evicted"]))
        k = nW + info["nb"] - 1          # layer 0, last block: free (filler) in practice
        self.selftest_pages = {"W": w, "F": f, "K": k}
        for p in (w, f, k):
            self.conn.send({"op": "corrupt", "page": p})
        t0 = time.time()
        while len(self.corrupted) < 3 and time.time() - t0 < 30:
            time.sleep(0.05)
        self.selftest_ids = {x: self.spot([p]) for x, p in self.selftest_pages.items()}
        time.sleep(1.0)

    # ---------------------------------------------------------------- checks
    def layout(self):
        info = self.info
        ents = [Entry(n, o, tuple(s), src, k) for n, o, s, src, k in info["w_entries"]]
        lay = Layout(ents, info["w_bytes"], info["w_bytes"], (info["nW"] + info["nK"]) * PAGE)
        return lay

    def check_memory(self):
        a = self.a
        info = self.info
        exp = Expect(self.layout(), ZeroWeights() if a.zero_weights else WeightSource(a.model),
                     self.fkey)
        nW, nK, npages = info["nW"], info["nK"], info["npages"]
        rng = np.random.default_rng(self.post_seeds[1])
        mem = {"w_checked": 0, "w_bad": 0, "f_checked": 0,
               "f_bad": 0, "kfree_checked": 0, "kfree_bad": 0, "k_pages": 0, "k_states": {}}
        kq = []
        lat_k = {}
        bad_pages = []
        self._page_cache = {}
        self._unit_cache = {}
        self._chase_cache = {}
        self._dl_arr = None
        self._dl_all = None
        _tk, _prev, _t_all = {}, None, time.perf_counter()     # post-run check time per kind
        # content checks on a fixed random sample of challenges per kind (timing uses all of them)
        chk = None
        if a.check_max > 0:
            by = {}
            for c in self.responses:
                by.setdefault(self.chal_sent[c][2], []).append(c)
            chk = set()
            for k, cs in by.items():
                cs = sorted(cs)
                # mixed chases: a larger secret random sample across the whole run (each verified chain costs
                # ~4 s of verifier CPU: the slow sloth direction for every probed word)
                cap = max(a.check_max, a.mchase_verify) if k == "mchase" else a.check_max
                chk |= set(cs if len(cs) <= cap else rng.choice(cs, cap, replace=False).tolist())
            mem["content_sample_per_kind"] = a.check_max
        lat_detail = mem.setdefault("lat_detail", {})
        fz_abs = None
        try:
            rv_f = getattr(self, "revealer", None)
            if rv_f is not None and hasattr(rv_f, "freeze_events"):
                fz_abs = np.asarray(rv_f.freeze_events(absolute=True), dtype=np.float64).reshape(-1, 3)
        except Exception:
            fz_abs = None
        t_first = min((v[0] for v in self.chal_sent.values()), default=0.0)
        for cid, m in sorted(self.responses.items()):
            t_sent, nonce, kind = self.chal_sent[cid]
            if kind == "mchase" and "t_recv" in m:
                # per-challenge (seconds since the first challenge, latency ms): time to first failed challenge
                mem.setdefault("lat_t", {}).setdefault(kind, []).append(
                    (round(t_sent - t_first, 3), round((m["t_recv"] - t_sent) * 1e3, 3)))
            if kind == "mchase" and "srv_ms" in m:
                # tail study: (send time, latency, server queue: receive -> launch, launch -> done, GPU kernel time)
                lat_detail.setdefault(kind, []).append(
                    (round(t_sent, 4), round((m["t_recv"] - t_sent) * 1e3, 3), round(m["srv_ms"][0], 3),
                     round(m["srv_ms"][1], 3), round(m.get("kernel_ms", 0.0), 3),
                     round(m["gpu_ms"], 3) if m.get("gpu_ms") is not None else None))
            _now = time.perf_counter()
            if _prev is not None:
                _tk[_prev[0]] = _tk.get(_prev[0], 0.0) + _now - _prev[1]
            _prev = (kind, _now)
            if m.get("win") is not None and m["win"] >= 0 and a.pcie:
                # (b): the chain is keyed by the newest noise digest the prover had
                w = m["win"]
                d0w = self.checker.d0_all.get(w)
                hs = self.chal_head.get(cid, -1)
                fresh = d0w is not None and w >= hs - 2 and w <= m["head"] + 64
                mem["win_checked"] = mem.get("win_checked", 0) + 1
                if not fresh:
                    mem["win_bad"] = mem.get("win_bad", 0) + 1
                nonce = ref.chain_nonce(nonce, d0w or 0)
                mem.setdefault("win_lag", []).append(max(0, hs - w) if hs >= 0 else None)
            sampled = chk is None or cid in chk
            if kind in ("mchase", "warmmchase"):
                if sampled:
                    self._check_mchase(cid, m, nonce, exp, mem, rng, bad_pages)
                elif kind == "mchase" and getattr(a, "replay_all", 0) and getattr(a, "kv_table", 0) \
                        and not a.zero_weights and m.get("kvt") is not None:
                    self.__dict__.setdefault("_replay_q", []).append((cid, nonce))    # --replay-all: one chain of every answer
                if kind == "mchase":
                    # an answer whose round trip overlaps one of the verifier's own freezes (the Revealer's log: the
                    # whole host stopped, e.g. for a PCIe correctable error) is not timed: the time lost is not the
                    # prover's. At most one answer can be in flight per freeze (challenges are serial).
                    if fz_abs is not None and len(fz_abs) and bool(((fz_abs[:, 0] <= m["t_recv"]) & (fz_abs[:, 1] >= t_sent)).any()):
                        mem["mchase_excused_freeze"] = mem.get("mchase_excused_freeze", 0) + 1
                    else:
                        lat_k.setdefault("mchase", []).append((m["t_recv"] - t_sent) * 1e3)
                continue
            pages = np.array(m["pages"], dtype=np.int64)
            d0 = np.array(m["d0"], dtype=np.int64) & 0xFFFFFFFF
            d1 = np.array(m["d1"], dtype=np.int64) & 0xFFFFFFFF
            claims = {c[0]: c for c in self.claims.get(cid, {}).get("claims", [])}
            flat = pages
            R0 = info.get("R0", npages)
            w_idx = [j for j, p in enumerate(flat) if p < nW]
            f_idx = [j for j, p in enumerate(flat) if nW + nK <= p < R0]
            k_idx = [j for j, p in enumerate(flat) if nW <= p < nW + nK]
            kf_idx, kz_idx = [], []
            for j in k_idx:
                c = claims.get(int(flat[j]))
                st = c[1] if c else "no_claim"
                mem["k_pages"] += 1
                mem["k_states"][st] = mem["k_states"].get(st, 0) + 1
                if st == "filler":
                    kf_idx.append(j)
                elif st == "settled" and a.zero_weights:
                    kz_idx.append(j)
                elif st == "settled":
                    pg = int(flat[j])
                    layer, block = divmod(pg - nW, info["nb"])
                    kq.append({"page": pg, "nonce": nonce, "d0": int(d0[j]), "d1": int(d1[j]),
                               "layer": layer, "block": block, "fp": tuple(c[2]),
                               "block_index": c[3], "cid": cid})
            if len(kz_idx) > a.kzero_sample:
                kz_idx = rng.choice(kz_idx, a.kzero_sample, replace=False).tolist()
            for j in kz_idx:
                # zero model: KV is all zeros, so the page is the bare keystream
                pg = int(flat[j])
                layer, block = divmod(pg - nW, info["nb"])
                words = kv_encode(np.zeros(PAGE // 2, dtype=np.uint16), kv_key(self.master, layer), block)
                mem["kzero_checked"] = mem.get("kzero_checked", 0) + 1
                if fast_digest(words.view(np.uint32), nonce) != (int(d0[j]), int(d1[j])):
                    mem["kzero_bad"] = mem.get("kzero_bad", 0) + 1
                    bad_pages.append((cid, pg, "kzero"))
            for idx, key in ((w_idx, "w"), (f_idx, "f"), (kf_idx, "kfree")):
                if not len(idx):
                    continue
                dg = exp.digests(flat[idx], nonce)
                for j, (e0, e1) in zip(idx, dg):
                    mem[key + "_checked"] += 1
                    if (e0, e1) != (int(d0[j]), int(d1[j])):
                        mem[key + "_bad"] += 1
                        bad_pages.append((cid, int(flat[j]), key))
        if _prev is not None:
            _tk[_prev[0]] = _tk.get(_prev[0], 0.0) + time.perf_counter() - _prev[1]
        mem["check_s"] = {k: round(v, 1) for k, v in _tk.items()}
        mem["check_s"]["total"] = round(time.perf_counter() - _t_all, 1)
        log("post-run check time per kind (s):", json.dumps(mem["check_s"]))
        mem["bad_pages"] = bad_pages[:20]
        for kind, ls in lat_k.items():
            ls2 = sorted(ls)
            mem[f"{kind}_lat_ms"] = {"p50": round(ls2[len(ls2) // 2], 3),
                                     "p99": round(ls2[min(len(ls2) - 1, int(0.99 * len(ls2)))], 3),
                                     "max": round(ls2[-1], 3), "n": len(ls2),
                                     "all": [round(x, 2) for x in ls]}
        self.lat_k = lat_k
        if getattr(self, "bd", None):
            # per challenge: kind, rtt, srv_pre, launch_to_start, kernel, post, transport (ms),
            # step median / p99 (us)
            mem["breakdown"] = [[b[0]] + [round(x, 4) for x in b[1:]] for _, b in sorted(self.bd.items())]
        self.kq, self.mem = kq, mem
        if self.a.selftest:
            st = {}
            for k, cid in self.selftest_ids.items():
                st[k + "_caught"] = any(c == cid for c, _, _ in bad_pages)
            mem["selftest"] = st

    @staticmethod
    def _fp64(prompt):
        f = common.prompt_fp(prompt)
        return (int(f[0]) << 32) | int(f[1])

    def _table_fps(self):
        """Fingerprints named by the block tables of the queued answers (writers in tight mode, owners otherwise);
        None if the tables cannot be read (then every request is stored)."""
        try:
            from schemes.memory_link_exhaustion.inference import kvtable as KT
            a = self.a
            bodies = prover_tables(a, self.info)
            cache = self.__dict__.setdefault("_kvt_parsed", {})
            fps = set()
            for cid, _ in getattr(self, "_replay_q", []):
                ver, dig = self.responses[cid]["kvt"]
                if (ver, dig) not in cache:
                    b_ = bodies.get(ver)
                    cache[(ver, dig)] = KT.parse(b_) if (b_ is not None and KT.body_digest(b_) == dig) else None
                    t = cache[(ver, dig)]
                    if t is None:
                        continue
                    if t.tight:
                        fps.update(int(f) for f, _ in t.writers[1:])
                        fps.update(int(f) for f in np.unique(t.fp[t.kind == KT.LIVE]))
                    else:
                        fps.update(int(f) for f in np.unique(t.fp[t.kind != KT.FILLER]))
            return fps
        except Exception as e:
            log("table fingerprints: not read:", type(e).__name__, e)
            return None

    def _kvbf_words(self, layer, block, wo):
        """Butterfly KV keystream of the 128-byte unit at word offset wo of a KV page (uint32 [32]): the encoding of
        an all-zero unit; a unit's encoding is its plaintext XOR these words."""
        cache = self.__dict__.setdefault("_kvbf_cache", {})
        if len(cache) > 65536:
            cache.clear()
        hk = self.info["kv_shape"][1]
        T = kvbfly.tsc()
        tok, head, kvh, dg0 = wo // (hk * 128), (wo >> 7) % hk, (wo >> 6) & 1, (wo >> 2) & 15
        sc = kvbfly.chunk_s(block, head, kvh, tok // T)
        k = (layer, sc)
        if k not in cache:
            cache[k] = kvbfly.chunk_ks(kv_key(self.master, layer), sc)
        u0 = (tok % T) * 16 + dg0
        return cache[k][u0:u0 + 8].reshape(-1)

    def replay_chains(self):
        """Settled-KV rule, real weights: replay one chain of each queued mixed chase from sources only. W, filler,
        ring and pool bytes as in the zero-weight check; a KV chunk from the block table that answer ran under
        (masked -> the fixed stand-in; filler -> the zero-plaintext K filler; live / stale -> the owner's KV from the
        reference store, encoded for that page). Every replayed chain must equal the value returned in time."""
        a, info = self.a, self.info
        q = getattr(self, "_replay_q", [])
        out = {"queued": len(q), "replayed": 0, "bad": 0, "no_table": 0, "no_store": 0}
        if not q:
            return out
        from schemes.memory_link_exhaustion.inference import kvtable as KT
        from schemes.memory_link_exhaustion.encodings import sloth64
        assert kvbfly.KVENC in ("sloth", "bfly"), "chain replay: sloth or butterfly KV encoding only"
        bodies = prover_tables(a, self.info)
        prompts = common.prompts(a)
        # fingerprint -> request. A workload can hold the same prompt more than once with different output lengths
        # (ShareGPT does); greedy decoding makes the shorter one a prefix of the longer, so the longest stands for all
        fpi, dup = {}, 0
        tot = lambda i: len(prompts[i]) + (len(self.tokens[i]) if i < len(self.tokens) else 0)
        for i, p in enumerate(prompts):
            f = common.prompt_fp(p)
            k = (int(f[0]) << 32) | int(f[1])
            if k in fpi:
                dup += 1
                if tot(i) <= tot(fpi[k]):
                    continue
            fpi[k] = i
        out["duplicate_prompts"] = dup
        sdir = a.out + ".kvstore"
        L, nb, nW = info["L"], info["nb"], info["nW"]
        exp = Expect(self.layout(), WeightSource(a.model), self.fkey)
        rng = np.random.default_rng(self.post_seeds[1] + 7)
        mem, bad = {}, []
        files, tabs, why = {}, {}, {}
        klast = [None]
        dq = getattr(self, "_chase_dbg", None)
        if dq is not None:
            q = list(q) + [(dq["cid"], dq["nonce"])]
        from schemes.memory_link_exhaustion.verifier import chainreplay as CR
        # batched replay where its sources apply (sloth64 or butterfly W, KV and filler; no
        # activation pool in the chase); the first
        # --replay-scalar answers still go through the scalar per-chain check, an independent implementation
        use_batch = bool(int(getattr(a, "replay_batch", 512)) > 0 and CR.supported() and ref.WENC in ("sloth", "bfly")
                         and ref.FILLER in ("sloth", "bfly") and kvbfly.KVENC in ("sloth", "bfly") and not ((info.get("act_pool") or {}).get("granules")))
        bjobs, n_scalar = [], 0
        # --kv-table 3 (tagged in-flight reads): only the batched replay knows the old / new candidates
        tagm = int(getattr(a, "kv_table", 0)) == 3
        assert use_batch or not tagm, "tagged in-flight reads need the batched replay"
        for cid, nonce in q:
            m = dq["m"] if (dq is not None and cid == dq["cid"]) else self.responses[cid]
            if m.get("neff") is not None and int(m["neff"]) != (int(nonce) & 0xFFFFFFFF):
                out["nonce_differs"] = out.get("nonce_differs", 0) + 1     # verifier's chained nonce vs the kernel's
            ver, dig = m["kvt"]
            if ver not in tabs:
                b_ = bodies.get(ver)
                pc = getattr(self, "_kvt_parsed", {})
                tabs[ver] = pc[(ver, dig)] if (ver, dig) in pc else \
                    (KT.parse(b_) if (b_ is not None and KT.body_digest(b_) == dig) else None)
            t = tabs[ver]
            if t is None:
                out["no_table"] += 1
                continue
            miss = [0]

            def kv_real(page, layer, block, wo, t=t, miss=miss):
                src = t.source(block, ref.ch_kv_token(wo))      # None: in the band; "filler"; or (fp, block index)
                k = int(t.kind[block])
                klast[0] = "K-" + KT.KIND_NAMES.get(k, "?") + \
                    ("-masked" if src is None else "-filler" if src == "filler" else "")
                if src is None:
                    return None                                        # in the band: not read
                if src == "filler":
                    return "filler"                  # (free K pages are checked as arena filler, exp.digests, too)
                off = block * (PAGE // 2) + 2 * wo
                i = fpi.get(int(src[0]))
                if i is None:
                    miss[0] += 1
                    why["fp_unknown"] = why.get("fp_unknown", 0) + 1
                    return np.zeros(32, np.uint32)
                if i not in files:
                    pth = os.path.join(sdir, f"req{i}.bin")
                    files[i] = np.memmap(pth, dtype=np.uint16, mode="r") if os.path.exists(pth) else None
                f = files[i]
                o16 = ((int(src[1]) * L + layer) * PAGE + wo * 4) // 2
                if f is None or o16 + 64 > f.size:
                    miss[0] += 1
                    why["no_file" if f is None else "past_end"] = why.get("no_file" if f is None else "past_end", 0) + 1
                    if len(why.setdefault("examples", [])) < 4:
                        why["examples"].append({"req": i, "idx": int(src[1]), "settled": int(t.settled[block]),
                                                "kind": k, "file_blocks": None if f is None else int(f.size // (L * PAGE // 2)),
                                                "tokens": len(prompts[i]) + len(self.tokens[i]) if i < len(self.tokens) else None})
                    return np.zeros(32, np.uint32)
                if kvbfly.KVENC == "bfly":
                    return np.ascontiguousarray(f[o16:o16 + 64]).view(np.uint32) ^ self._kvbf_words(layer, block, wo)
                return sloth64.encode_range(np.ascontiguousarray(f[o16:o16 + 64]),
                                            sloth64.kv_seed(kv_key(self.master, layer)), off).view(np.uint32)
            if dq is not None and cid == dq["cid"]:
                d = dq.get("walk") or {}
                dbg = {"probes": d.get("probes") or [], "klast": klast, "by_region": {}, "first": []}
                res = {"prover_error": d.get("error"), "probes": len(dbg["probes"]), "chain": dq["chain"],
                       "kernel_state_equals_prover_walk": (str(int(m["states"][dq["chain"]])) == d.get("state")) if d.get("state") else None,
                       "table_version_kernel": ver, "table_version_walk": d.get("kvt_version"),
                       "kernel_state": str(int(m["states"][dq["chain"]])), "walk_state": d.get("state"),
                       "kernel_state_equals_walk_under_device_table":
                           (str(int(m["states"][dq["chain"]])) == d.get("state_dev_table")) if d.get("state_dev_table") else None,
                       **{k: d.get(k) for k in ("dev_table_equals_host", "dev_table_diff", "dev_table_diff_examples",
                                                 "dev_info", "host_info", "mix", "direct_full", "direct_plain",
                                                 "walk_plain", "direct_full_neighbours", "direct_error")},
                       "verifier": [int(ref.CH_MIX), int(ref.CH_WIDE), int(a.mchase_p), int(a.chase_k), int(a.chase_blocks)],
                       "neff": m.get("neff"), "nonce_used": dq["nonce"], "n_states": int(len(m["states"]))}
                if dbg["probes"]:
                    self._check_mchase(cid, m, nonce, exp, {}, rng, [], kv_real=kv_real, dbg=dbg)
                    res.update({"probes_by_region": dbg["by_region"], "mismatch_by_region": dbg.get("diff_by_region"),
                                "first_mismatches": dbg["first"]})
                out["debug_transcript"] = res
                continue
            if use_batch and (tagm or n_scalar >= int(getattr(a, "replay_scalar", 2))):
                bjobs.append((cid, nonce, m, t))                      # replayed together below
                continue
            n_scalar += 1
            n0 = len(bad)
            self._check_mchase(cid, m, nonce, exp, mem, rng, bad, kv_real=kv_real)
            out["replayed"] += 1
            out["no_store"] += int(miss[0] > 0)
            out["bad"] += int(len(bad) > n0)
        st = mem.get("mchase") or {}
        out.update({"kv_probes": st.get("kv_probes", 0), "kv_masked": st.get("kv_masked", 0), "scalar": n_scalar})
        if bjobs:
            # every other queued answer: one chain each, advanced together (verifier/chainreplay.py; integer only)
            import time as _time
            t0 = _time.time()
            nK = info["nK"]
            nF = info.get("nF_chase", info["R0"] - nW - nK)
            gp = ref.ch_granules_mix(nW, nK, nF, 0, self._ring_granules())
            tb = len(gp) // ref.CH_GRAN_PER_BLOCK
            nbk = min(a.chase_blocks, tb)
            ck = getattr(self, "checker", None)
            cur = []

            def store(i):
                if i not in files:
                    pth = os.path.join(sdir, f"req{i}.bin")
                    files[i] = np.memmap(pth, dtype=np.uint16, mode="r") if os.path.exists(pth) else None
                return files[i]

            n_ring_ = (int(a.ring_mib) << 20) // int(ck.W) if (ck is not None and a.ring_mib > 0) else 0

            def ring(jx, page, wo, new=False):
                # noise ring, head-relative (as in _check_mchase): logical slot q holds window (head - q); with
                # tagged ring reads a slot in the zone may already hold the incoming window (head - q + ring slots)
                assert page >= ref.CH_RA0, "batched replay: no activation-pool source"
                q_, w_in = ref.ring_chase_slot(page, wo)
                try:
                    off = ck.off_all[int(cur[jx][2].get("win")) - q_ + (n_ring_ if new else 0)]
                except (KeyError, TypeError, AttributeError):
                    return np.zeros(32, np.uint32)
                w0 = off * 4096 // 4 + w_in
                return np.asarray(ck.pool[w0:w0 + 32], dtype=np.uint32)
            src = CR.Sources(nW=nW, nK=nK, nb=nb, L=L, layout=exp.lay, weights=exp.w, fkey=self.fkey,
                             kv_seeds=[sloth64.kv_seed(kv_key(self.master, l)) for l in range(L)],
                             store=store, fpi=fpi, scalar=ring,
                             scalar_new=(lambda jx, page, wo: ring(jx, page, wo, new=True)) if self._ring_tagged() else None,
                             ring_zone=(n_ring_, int(a.ring_chase_guard)) if self._ring_tagged() else None,
                             kv_ks=self._kvbf_words if kvbfly.KVENC == "bfly" else None)
            B = int(getattr(a, "replay_batch", 512))
            for k in range(0, len(bjobs), B):
                cur = bjobs[k:k + B]
                cs = [int(rng.integers(0, len(m["states"]))) for _, _, m, _ in cur]
                tg = None
                if tagm:
                    # the tags the kernel returned for the chain sampled from each answer
                    tg = [None if m.get("tags") is None else m["tags"][cs[x]] for x, (_, _, m, _) in enumerate(cur)]
                    if any(t_ is None for t_ in tg):
                        out["no_tags"] = out.get("no_tags", 0) + sum(1 for t_ in tg if t_ is None)
                        tg = [np.zeros(ref.CH_TAG_SLOTS, np.uint32) if t_ is None else t_ for t_ in tg]
                src.bind([n_ for _, n_, _, _ in cur], [t for _, _, _, t in cur], tags=tg)
                got = CR.walk([n_ for _, n_, _, _ in cur], cs, a.mchase_p, tb, nbk, gp, src)
                for x, (cid, _, m, _) in enumerate(cur):
                    out["replayed"] += 1
                    out["no_store"] += int(src.miss[x] > 0)
                    if tagm and (src.neither[x] > 0 or src.tag_over[x]):
                        # a read in a band matched neither the old nor the new content (caught mid-write), or the
                        # chain had more in-band reads than tag slots: this chain cannot be replayed
                        out["unreplayable"] = out.get("unreplayable", 0) + 1
                        k_ = "unreplayable_torn" if src.neither[x] > 0 else "unreplayable_tag_slots"
                        out[k_] = out.get(k_, 0) + 1
                        continue
                    if tagm and int(src.nband[x]) != int(tg[x][0]):
                        out["tag_count_bad"] = out.get("tag_count_bad", 0) + 1
                        out["bad"] += 1
                        bad.append((cid, -1, "mchase"))
                        continue
                    if int(m["states"][cs[x]]) != int(got[x]):
                        out["bad"] += 1
                        bad.append((cid, -1, "mchase"))
            out["kv_probes"] += src.stats["kv"]
            out["kv_masked"] += src.stats["kv_masked"]
            for k, v in src.why.items():
                why[k] = (why.get(k, []) + v)[:4] if k == "examples" else why.get(k, 0) + v
            out.update({"batched": len(bjobs), "batch_s": round(_time.time() - t0, 1),
                        "probes_by_region": {k: src.stats[k] for k in ("w", "filler", "kv", "kv_masked", "kv_filler", "kv_live", "kv_prev_owner", "other",
                                                   "band_new", "band_old", "band_neither", "band_both",
                                                   "ring_new", "ring_old", "ring_neither")}})
        out.update({"bad_cids": [b[0] for b in bad][:10], "store_misses": why})
        try:
            # diagnostic, independent of any chain: the prover's own settled page claims (whole pages, digested at
            # challenge time) recomputed from the store. Splits "the store does not reproduce the prover's KV" from
            # "the table or the in-flight band is off". By block position in the request: prompt blocks / output blocks.
            d = {"claims": 0, "prompt_ok": 0, "prompt_bad": 0, "output_ok": 0, "output_bad": 0, "no_store": 0}
            bs_ = info["kv_shape"][2]
            for c in (getattr(self, "_kq_saved", []) or [])[:300]:
                i = fpi.get((int(c["fp"][0]) << 32) | int(c["fp"][1]))
                d["claims"] += 1
                pth = os.path.join(sdir, f"req{i}.bin") if i is not None else None
                if pth is None or not os.path.exists(pth):
                    d["no_store"] += 1
                    continue
                if i not in files:
                    files[i] = np.memmap(pth, dtype=np.uint16, mode="r")
                f = files[i]
                o16 = (int(c["block_index"]) * L + int(c["layer"])) * (PAGE // 2)
                if f is None or o16 + PAGE // 2 > f.size:
                    d["no_store"] += 1
                    continue
                w = kv_encode(np.ascontiguousarray(f[o16:o16 + PAGE // 2]), kv_key(self.master, int(c["layer"])), int(c["block"]))
                ok = fast_digest(w.view(np.uint32), c["nonce"]) == (c["d0"], c["d1"])
                side = "prompt" if (int(c["block_index"]) + 1) * bs_ <= len(prompts[i]) else "output"
                d[f"{side}_{'ok' if ok else 'bad'}"] += 1
            out["claim_diag"] = d
        except Exception as e:
            out["claim_diag"] = {"error": f"{type(e).__name__}: {e}"}
        return out

    def reference(self):
        """Recompute claimed KV pages and a sample of outputs with a reference engine."""
        a = self.a
        rng = np.random.default_rng(self.post_seeds[2])
        kq = self.kq
        if len(kq) > a.max_claims:
            kq = [kq[i] for i in sorted(rng.choice(len(kq), a.max_claims, replace=False))]
        prompts = common.prompts(a)
        fp = {tuple(common.prompt_fp(p)): i for i, p in enumerate(prompts)}
        bs = self.info["kv_shape"][2]
        for q in kq:
            i = fp.get(q["fp"])
            need = (q["block_index"] + 1) * bs
            q["tokens"] = (prompts[i] + self.tokens[i])[:need] if i is not None else []
            q["fp"] = list(q["fp"])
        n_claims = len(kq)
        kq = [q for q in kq if q["tokens"]]
        # a settled claim whose prompt fingerprint matches no prompt cannot be recomputed: count it (it used to be
        # dropped silently, so forged claims passed unverified)
        k_unmatched = n_claims - len(kq)
        # output replay is off by default (protocol: no output checks; outputs may be all zeros); the reference
        # recomputes claimed KV from the prover's own claimed tokens
        if a.output_replay:
            n_out = max(1, int(a.replay_frac * len(self.tokens)))
            out_ids = sorted(rng.choice(len(self.tokens), n_out, replace=False).tolist())
        else:
            out_ids = []
        store = None
        if getattr(a, "kv_table", 0) and getattr(self, "_replay_q", None):
            # settled-KV chain replay: the reference also recomputes every request once and keeps its KV
            # (only the requests some answered table names as a writer or owner: a prover that served nothing, the
            # non-serving cheaters, names none and needs no store)
            named = self._table_fps()
            store = {"dir": a.out + ".kvstore", "block_tokens": bs,
                     # at most max_model_len - 1 tokens: the prefill asks for one more token, and a request that
                     # filled the context never fed its last sampled token back, so that position holds no KV
                     "requests": [{"i": i, "tokens": (prompts[i] + self.tokens[i])[:max(1, int(a.max_model_len) - 1)]}
                                  for i in range(len(prompts)) if i < len(self.tokens)
                                  and (named is None or self._fp64(prompts[i]) in named)]}
            if not store["requests"]:
                store = None
        if not kq and not out_ids and not store:
            return {"kv_checked": 0, "kv_ok": 0, "kv_bad": k_unmatched, "kv_missing": 0, "kv_unmatched": k_unmatched,
                    "kv_claims_sampled": n_claims, "outputs_checked": 0, "outputs_ok": 0, "ref_s": 0.0}
        job = {"model": a.model, "claims": kq, "outputs": out_ids,
               "args": {k: getattr(a, k) for k in ("num_prompts", "input_len", "output_len",
                                                     "max_num_seqs", "max_model_len",
                                                     "gpu_mem_util", "num_blocks", "seed",
                                                     "dataset", "dataset_path", "sched_defaults")},
               "mutant_every": a.mutant_every, "selftest_k": bool(a.selftest)}
        if store:
            job["kv_store"] = store
        jp = a.out + ".refjob.json"
        rp = a.out + ".ref.json"
        json.dump(job, open(jp, "w"))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "schemes.memory_link_exhaustion.verifier.reference", jp, rp],
                           stdout=open(a.out + ".ref.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode != 0 or not os.path.exists(rp):
            return {"error": f"reference failed rc={r.returncode}"}
        res = json.load(open(rp))
        # compare
        enc = res["pages"]
        k_ok = k_bad = k_missing = 0
        bad = []
        for q, e in zip(kq, enc):
            if e is None:
                k_missing += 1
                continue
            words = np.frombuffer(bytes.fromhex(e), dtype=np.uint16).copy()
            l, b = q["layer"], q["block"]
            key = kv_key(self.master, l)
            words = kv_encode(words, key, b)
            got = fast_digest(words.view(np.uint32), q["nonce"])
            if got == (q["d0"], q["d1"]):
                k_ok += 1
            else:
                k_bad += 1
                bad.append({k: q[k] for k in ("page", "layer", "block", "block_index", "cid")})
        o_ok = sum(1 for i, t in zip(out_ids, res["outputs"]) if t == self.tokens[i])
        rep = {"kv_checked": k_ok + k_bad, "kv_ok": k_ok, "kv_bad": k_bad + k_unmatched, "kv_missing": k_missing,
               "kv_unmatched": k_unmatched, "kv_claims_sampled": n_claims,
               "kv_bad_examples": bad[:10], "outputs_checked": len(out_ids), "outputs_ok": o_ok,
               "mutants": res.get("mutants"), "ref_s": round(time.time() - t0, 1)}
        return rep


def kv_encode(words_u16, key, block):
    """Encode one KV page (uint16 [PAGE // 2], physical order) of `block` with the layer key."""
    if kvbfly.KVENC == "bfly":
        return kvbfly.encode_page_u16(words_u16, key, block)
    if kvbfly.KVENC == "sloth":
        from schemes.memory_link_exhaustion.encodings import sloth64
        return sloth64.encode_range(np.ascontiguousarray(words_u16), sloth64.kv_seed(key), block * (PAGE // 2))
    return ref.encode16(words_u16, key, block * (PAGE // 2))


def ref_fmix32(x: int) -> int:
    """32-bit fmix (schemes.memory_link_exhaustion.encodings.kvbfly _kvbf_fmix)."""
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def host_snapshot():
    """Counters of the host as this container sees it (two small file reads): CPU-quota throttling of the cgroup and
    pressure-stall totals. A throttled cgroup has every thread stopped for the rest of a 100 ms period, verifier and
    prover alike; the deltas over a session say whether its link stalls coincide with that. Report only."""
    out = {"t": time.time()}
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            if k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec"):
                out[k] = int(v)
    except Exception:
        pass
    for res in ("cpu", "memory", "io"):
        try:
            first = open(f"/proc/pressure/{res}").readline()
            out[f"psi_{res}_some_usec"] = int(first.rsplit("total=", 1)[1])
        except Exception:
            pass
    return out


def prover_tables(a, info):
    """Block tables for the checks and the chain replay. A non-serving cheater produces no scheduler trace, so there
    is nothing to derive; rather than trust a table the cheater declares, the verifier BUILDS the only legitimate
    table for a prover that never writes KV after sealing -- every block filler, nothing in flight, version 0 -- and
    the cheater's answers must match its digest. A serving prover's tables come from the derivation of its trace."""
    from schemes.memory_link_exhaustion.inference import kvtable as KT
    if getattr(a, "prover", None) == "cheater":
        t = KT.KVTable(int(info["nb"]), int(info.get("kv_block_tokens", 16)), tight=int(getattr(a, "kv_table", 0) or 0) >= 2)
        return {0: t.body()}
    pth = kvtab_path(a)
    return KT.read_log(pth) if os.path.exists(pth) else {}


def kvtab_path(a):
    """The block tables the checks and the chain replay read: the verifier's own derivation (--kv-table-derive, made
    by verifier/schedreplay.py from the scheduler's call order) when it exists, else the prover's log."""
    return a.out + ".kvtab.derived"      # the derived tables; a cheater's table is built by prover_tables(), not read


def derive_tables(a, log=print, tokens=None):
    """--kv-table-derive: run the scheduler replay on the trace the engine wrote and keep the derived tables. From
    then on every table the verifier uses is its own; the prover's table log is not read."""
    import glob
    tr = sorted(glob.glob(a.out + ".schedtrace.[0-9]*"), key=os.path.getsize)
    if not tr:
        return {"error": "no scheduler trace"}
    lag = int(a.settle_lag) if a.settle_lag >= 0 else int(os.environ.get("VARENA_SETTLE_LAG", "0"))
    # what this verifier submitted and received: the trace's requests and tokens are checked against it
    json.dump({"prompts": [list(map(int, p)) for p in common.prompts(a)], "out_lens": [int(n) for n in common.out_lens(a)],
               "warmup": int(a.warmup_prompts), "selftest": bool(a.selftest),
               "tokens": [list(map(int, t)) for t in (tokens or [])]}, open(a.out + ".schedtrace.expect.json", "w"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "schemes.memory_link_exhaustion.verifier.schedreplay", tr[-1],
                        "--expect", a.out + ".schedtrace.expect.json",
                        "--llm-kwargs", a.out + ".schedtrace.llm.json", "--lag", str(lag),
                        "--tight", str(int(getattr(a, "kv_table", 0) >= 2)), "--dump", a.out + ".kvtab.derived"],
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "VARENA_SCHED_TRACE": ""},
                       capture_output=True, text=True, timeout=3600)
    out = {"rc": r.returncode, "s": round(time.time() - t0, 1), "trace_bytes": os.path.getsize(tr[-1])}
    try:
        out.update(json.loads(r.stdout.strip().splitlines()[-1]))
    except Exception:
        out["error"] = (r.stderr or r.stdout)[-600:]
        if os.path.exists(a.out + ".kvtab.derived"):
            os.remove(a.out + ".kvtab.derived")           # never fall back silently to a partial derivation
    log("kv_table_derived:", json.dumps({k: v for k, v in out.items() if k != "trace_digest_mismatch"}, default=str)[:900])
    return out


def kv_key(master, layer):
    from schemes.memory_link_exhaustion.inference.prover import kv_key as k
    return k(master, layer)




def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--ring-mib", type=int, default=1024, help="noise ring in the arena (0: off)")
    ap.add_argument("--deadline-mult-p50", type=float, default=2.0)
    ap.add_argument("--chal-delay", type=float, default=2.0, help="no challenges in the first s")
    ap.add_argument("--until-calls", type=int, default=0,
                    help="cheater runs: end the session once the hidden job has completed this many whole calls "
                         "(--num-prompts is then the cap); 0 = a fixed session of --num-prompts")
    ap.add_argument("--native", type=int, default=1, help="fan/ring challenges over the C++ path")
    ap.add_argument("--plain-w", type=int, default=0,
                    help="weights stored in plaintext in W (still declared and checked)")
    ap.add_argument("--deadline-mult-max", type=float, default=1.25)
    ap.add_argument("--pcie", type=int, default=1)
    ap.add_argument("--window-mib", type=int, default=32)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--pool-mib", type=int, default=2048)
    ap.add_argument("--chase-k", type=int, default=132 * 32, help="chase chains (warps)")
    ap.add_argument("--chase-blocks", type=int, default=11, help="4 GiB blocks probed per chase")
    ap.add_argument("--hard-deadline-ms", type=float, default=0.0,
                    help="single-challenge rule: every chase-type answer within this many ms (0: off)")
    ap.add_argument("--mchase-rate", type=float, default=0.0,
                    help="mixed chase (one kind over W, KV, filler and the activation pool) per second")
    ap.add_argument("--mchase-p", type=int, default=1200, help="mixed chase rounds")
    ap.add_argument("--kv-linger-max", type=int, default=-1, help="tight block tables: the largest version gap at "
                    "which two committed tables may show the same position of the same owner in the band "
                    "(-1: settle lag + 2; an honest position is in the band for lag + 2 versions, largest gap lag + 1)")
    ap.add_argument("--settle-lag", type=int, default=-1, help="tight block tables: the settle lag the verifier "
                    "expects (schedule() calls a settled count is held back); -1: VARENA_SETTLE_LAG of its own "
                    "environment, default 0: a token is settled once its step's outputs are processed; 389 of 389 and 391 of 391 chains at full size, 14 Nov)")
    ap.add_argument("--kv-promise-frac-max", type=float, default=0.002, help="tight block tables: share of band "
                    "positions that may not be read and owner-written lag + 2 versions later (honest: 0 on ShareGPT, "
                    "0.013-0.028%% with heavy preemption)")
    ap.add_argument("--kv-table-derive", type=int, default=0, help="1 = the verifier derives the block tables itself: "
                    "the engine records the order of its scheduler's calls, verifier/schedreplay.py replays them on "
                    "the engine's Scheduler and builds every table; the chain replay and the table checks then use "
                    "those tables, and the digest in each timed answer must equal the derived table's")
    ap.add_argument("--kv-mask-mean-max-mib", type=float, default=0.0, help="tight block tables: session mean of the "
                    "masked KV over the committed tables may not exceed this (0 = reported only)")
    ap.add_argument("--kv-pace-max", type=int, default=1, help="tight block tables: the most a request's computed "
                    "edge may be beyond one token per schedule() call (kvtable.history; honest runs measure 0)")
    ap.add_argument("--ring-tag", type=int, default=1, help="1 (default; needs --kv-table 3): the chase reads every ring "
                    "slot; a read in the --ring-chase-guard oldest slots, which incoming windows overwrite while the "
                    "chase runs, returns a tag and the replay takes the old or the incoming window by it. 0 (or a "
                    "kernel-validation table mode): those slots are not read")
    ap.add_argument("--verifier-frozen-max-s", type=float, default=3.0, help="verifier health: the run is flagged when "
                    "the verifier's own process was frozen for more than this many seconds, or 2%% of the session if "
                    "that is longer (a note on the run, not a prover check; the frozen time is credited either way)")
    ap.add_argument("--replay-unreplayable-max", type=int, default=1, help="--kv-table 3 (tagged in-flight "
                    "reads): how many replayed chains of a session may have an in-band read matching neither the old "
                    "nor the new content (a position caught mid-write) or more in-band reads than tag slots. Honest "
                    "runs measure 0; a torn read is expected about once in several thousand sessions "
                    "(tests/test_chase_tag_race_gpu.py)")
    ap.add_argument("--replay-all", type=int, default=0, help="settled-KV table: 1 = replay one chain of every "
                    "answered mixed chase after the session (batched, verifier/chainreplay.py), not only the "
                    "--mchase-verify sample")
    ap.add_argument("--replay-batch", type=int, default=512, help="chains advanced together in the batched replay "
                    "(0 = scalar per-chain replay only)")
    ap.add_argument("--replay-scalar", type=int, default=2, help="answers still replayed by the scalar per-chain "
                    "check when the batched replay is used (a second implementation of the same walk)")
    ap.add_argument("--mchase-verify", type=int, default=20, help="mixed chases content-checked: a secret random sample across the whole run, chosen after it (1 chain "
                         "each; zero-weight runs only: live KV is unknown with real weights)")
    ap.add_argument("--hard-mult", type=float, default=0.0,
                    help="tail rule: fail if more than --tail-allow mixed-chase / ring answers are slower than "
                         "hard_mult x T (0: off)")
    ap.add_argument("--mean-ref-ms", type=float, default=0.0,
                    help="mean rule: reference (ms) = the slowest honest floor run's mean answer time without its "
                         "slowest answer (floors.env MEANREF); 0 = off")
    ap.add_argument("--mean-tol", type=float, default=0.05,
                    help="mean rule: the session's mean answer time without its slowest answer may exceed the "
                         "reference by this fraction")
    ap.add_argument("--tail-allow", type=int, default=1,
                    help="tail rule: how many answers of a session may be slower than hard_mult x T. Default 1 "
                         "(15 Nov): with 0 a do-nothing prover fails a few percent of sessions on one stray "
                         "answer (docs/RESULTS.md); a prover that is late often still fails")
    ap.add_argument("--warmup-prompts", type=int, default=256,
                    help="untimed warm-up generation before timing / challenges (0 = none)")
    ap.add_argument("--noise-sched", type=int, default=0,
                    help="verifier-clocked noise: window i due at t0 + i*W/rate, its secret revealed "
                         "lag windows ahead; no per-window digest, echo = window ^ keyed arena stretch")
    ap.add_argument("--noise-gbs", type=float, default=38.0, help="scheduled noise rate")
    ap.add_argument("--check-max", type=int, default=150,
                    help="content-check at most this many challenges per kind (random sample; 0 = all); "
                         "timing rules always use every challenge")
    ap.add_argument("--link-gbs", type=float, default=55.5,
                    help="measured one-way H2D link maximum (catch-up bound for the schedule deadlines)")
    ap.add_argument("--ring-chase-guard", type=int, default=int(os.environ.get("VARENA_RING_CHASE_GUARD", "0") or 0),
                    help="> 0: the mixed chase also probes the noise ring, head-relative: logical slot q is the window q "
                         "behind the head the prover returns, for q < ring slots - guard (the guard covers windows whose "
                         "copy can start while the chase runs; 24 of 64 is ample at 0.6 ms per window). 0: off")
    ap.add_argument("--chase-debug", type=int, default=int(os.environ.get("VARENA_CHASE_DEBUG", "0") or 0),
                    help="debug: after generation, one more mixed chase and the prover's CPU walk of one of its chains, "
                         "compared probe by probe with the verifier's sources (needs --kv-table 1 and no ring granules)")
    ap.add_argument("--kv-table", type=int, default=int(os.environ.get("VARENA_KV_TABLE", "0") or 0),
                    help="1: settled-KV rule in the honest prover: it keeps a block table from its scheduler, the mixed "
                         "chase does not read in-flight positions, and each chase answer carries the table's version "
                         "and digest. 2: tight table (the band is only what steps in flight may write; per-position "
                         "writers). 3: tight table and tagged in-flight reads: the chase reads the band too and returns "
                         "a tag per such read, so no KV position is left unread")
    ap.add_argument("--kv-tab-test", type=int, default=int(os.environ.get("VARENA_KV_TAB_TEST", "0") or 0),
                    help="kernel validation: > 0 = seed of a synthetic KV block table; the zero-weight cheater runs the "
                         "settled-KV chase kernel with it and the verifier replays chains under the same table")
    ap.add_argument("--head-floor-gbs", type=float, default=float(os.environ.get("VARENA_HEAD_FLOOR_GBS", "0") or 0),
                    help="head rule as a counted check: no head jump or step back between consecutive mixed chases, and the "
                         "link rate counted from the chase heads at least this (GB/s; 0 = report only)")
    ap.add_argument("--end-floor-gbs", type=float, default=float(os.environ.get("VARENA_END_FLOOR_GBS", "0") or 0),
                    help="end-of-session link rule: the average noise rate from t0 to the last ring challenge must be at "
                         "least this (GB/s, verifier clock; 0 = report only)")
    ap.add_argument("--end-min-s", type=float, default=float(os.environ.get("VARENA_END_MIN_S", "0") or 0),
                    help="end-of-session rule: sessions shorter than this fail it (0 = any length)")
    ap.add_argument("--noise-perm", type=int, default=2, choices=[1, 2],
                    help="2 (default): public sequential walk of the pool's whole-window slots; 1: a keyed permutation "
                         "of the slots per cycle (128-bit secret, no slot repeats within ring + lag + lead windows "
                         "across cycles). Both run through the same schedule table (ref.perm_keys)")
    ap.add_argument("--sched-lead", type=int, default=16,
                    help="window i's secret is revealed this many windows before it is due")
    ap.add_argument("--sched-grace-s", type=float, default=5.0)
    ap.add_argument("--sched-slack-ms", type=float, default=30.0,
                    help="a window reported later than due + slack counts as late")
    ap.add_argument("--shm", default="/dev/shm")
    ap.add_argument("--allowance-gib", type=float, default=4.0)
    ap.add_argument("--gpu-model", default="H200_NVL", help="declared GPU model: capacity = GPU_CAPACITY[model]")
    ap.add_argument("--gpu-capacity-bytes", type=int, default=0, help="override the declared model's capacity")
    ap.add_argument("--secret-seed", type=int, default=0,
                    help="debug only: reproducible (non-cryptographic) secrets; 0 = OS CSPRNG")
    ap.add_argument("--pad-mib", default="1536",
                    help="free HBM the honest prover leaves outside the arena at seal (MiB), or 'auto': the pad its "
                         "calibration measured (pad_mib_cal in --pad-calib, else in --calib)")
    ap.add_argument("--pad-calib", default=None,
                    help="calibration whose measured pad sizes this run's arena (--pad-mib auto); default --calib")
    ap.add_argument("--pad-safety-frac", type=float, default=0.25,
                    help="calibration: pad_mib_cal = measured pad use x (1 + this) + --pad-safety-mib")
    ap.add_argument("--pad-safety-mib", type=int, default=64)
    ap.add_argument("--allowance-margin-gib", type=float, default=1.0,
                    help="calibration: allowance = honest outside-arena bytes + this margin")
    ap.add_argument("--switch-interval", type=float, default=0.0005)
    ap.add_argument("--replay-frac", type=float, default=0.02)
    ap.add_argument("--max-claims", type=int, default=300)
    ap.add_argument("--mutant-every", type=int, default=20)
    ap.add_argument("--max-excused", type=float, default=0.1,
                    help="max fraction of challenged K pages excused (unsettled/transition/stale_partial); honest 3.4%%")
    ap.add_argument("--debug", default=None, help="JSON of prover debug flags (ablations)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prover", default="vllm", choices=["vllm", "cheater"])
    ap.add_argument("--cheater-spec", default=None)
    ap.add_argument("--zero-weights", action="store_true",
                    help="declared model is all zeros (W = encode(0), KV = encode(0), outputs 0)")
    ap.add_argument("--kzero-sample", type=int, default=64)
    ap.add_argument("--cross-key", type=int, default=0,
                    help="noise windows fold in an arena stretch; chains keyed by the noise")
    ap.add_argument("--baseline-tokens", default=None)
    ap.add_argument("--save-tokens", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--write-calib", default=None)
    ap.add_argument("--tol-tput", type=float, default=0.05)
    ap.add_argument("--no-reference", action="store_true", help="no KV recompute (live KV then unverified)")
    ap.add_argument("--output-replay", action="store_true",
                    help="also replay a sample of outputs against the reference (off: the protocol has no output checks)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.pcie = bool(a.pcie)
    sys.setswitchinterval(0.0005)     # timestamp responses promptly (GIL)
    try:
        import numba
        numba.set_num_threads(min(16, numba.config.NUMBA_NUM_THREADS))
    except Exception:
        pass
    calib = json.load(open(a.calib)) if a.calib else None
    if calib and "allowance_gib" in calib:
        a.allowance_gib = calib["allowance_gib"]
    resolve_pad(a, calib)
    v = Verifier(a)
    v.run()
    kvd = None
    if getattr(a, "kv_table_derive", 0) and getattr(a, "kv_table", 0) and not a.zero_weights and a.prover != "cheater":
        try:
            kvd = derive_tables(a, log, v.tokens)
        except Exception as e:
            kvd = {"error": f"{type(e).__name__}: {e}"}
        os.environ.pop("VARENA_SCHED_TRACE", None)       # (the reference engine started later must not trace)
    info = v.info
    rep = {"tag": os.path.basename(a.out), "args": vars(a), "cheater": info.get("cheater"),
           "info": {k: x for k, x in info.items() if k != "w_entries"},
           "tok_s": v.tok_s, "startup_s": v.t_startup, "prover": v.stats,
           "t_warm_end": getattr(v, "t_warm_end", None),
           "sched": getattr(v, "sched", None)}
    if a.save_tokens:
        json.dump(v.tokens, open(a.save_tokens, "w"))
    _t = time.perf_counter()
    v.check_memory()
    log(f"check_memory {time.perf_counter() - _t:.1f} s")
    rep["memory"] = v.mem
    checks_extra = {}
    if a.pcie:
        s = v.checker.summary()
        g0, g1 = v.gen_window
        s["rate_gbs"] = round(v.checker.rate(g0 + 2, g1 - 1), 3)
        rep["pcie"] = s
    if a.zero_weights:
        bad = sum(1 for t in v.tokens if any(x != 0 for x in t))
        rep["reference"] = {"kv_checked": v.mem.get("kzero_checked", 0),
                            "kv_ok": v.mem.get("kzero_checked", 0) - v.mem.get("kzero_bad", 0),
                            "kv_bad": v.mem.get("kzero_bad", 0), "outputs_checked": len(v.tokens),
                            "outputs_ok": len(v.tokens) - bad, "zero_model": True}
    elif not a.no_reference:
        rep["reference"] = v.reference()
    if getattr(a, "kv_table", 0) and getattr(v, "_replay_q", None) and not a.zero_weights:
        if a.no_reference:
            # the honest reproduction runs skip the KV claim recompute (--no-reference); the chain replay still
            # needs the reference engine once, for the per-request KV store only
            v._kq_saved = list(getattr(v, "kq", []) or [])
            v.kq = []
            rep["reference_store"] = v.reference()
        try:
            rep["mchase_replay"] = v.replay_chains()
            rr = rep["mchase_replay"]
            # (tagged in-flight reads: a chain with a band read that matched neither content is not replayable; at
            # most --replay-unreplayable-max chains of the session may be, and every answer must carry its tags)
            rr["ok"] = bool(rr["replayed"] > 0 and rr["bad"] == 0 and rr["no_table"] == 0 and rr["no_store"] == 0
                            and rr.get("no_tags", 0) == 0
                            and rr.get("unreplayable", 0) <= a.replay_unreplayable_max)
        except Exception as e:
            import traceback
            rep["mchase_replay"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:],
                                    "ok": False}
        log("mchase_replay:", json.dumps({k: x for k, x in rep["mchase_replay"].items() if k != "trace"}))
        import shutil
        shutil.rmtree(a.out + ".kvstore", ignore_errors=True)          # tens of GB: not kept
    if a.baseline_tokens and os.path.exists(a.baseline_tokens):
        base = json.load(open(a.baseline_tokens))
        rep["vs_baseline"] = {"identical": sum(1 for x, y in zip(base, v.tokens) if x == y),
                              "of": len(base)}
    # ---- verdicts
    checks = dict(checks_extra)
    if kvd is not None:
        # the verifier's own tables: the replay ran, the scheduler replay scheduled what the engine did, and (below,
        # kv_table) the digest inside every timed answer equals the derived table's
        rep["kv_table_derived"] = kvd
        checks["kv_table_derived"] = bool(kvd.get("rc") == 0 and "error" not in kvd and kvd.get("sched_mismatch") is None
                                          and not kvd.get("errors") and kvd.get("requests_unknown", 1) == 0
                                          and kvd.get("requests_missing", 1) == 0 and kvd.get("tokens_bad", 1) == 0)
    if "mchase_replay" in rep:           # settled-KV rule: every replayed chain equals the value returned in time
        checks["mchase_replay"] = bool(rep["mchase_replay"].get("ok"))
    allowance = int(a.allowance_gib * (1 << 30))
    cap = a.gpu_capacity_bytes or GPU_CAPACITY[a.gpu_model]
    rep["capacity"] = {"declared_model": a.gpu_model, "capacity_bytes": cap,
                       "prover_reported_bytes": info.get("capacity"),        # information only
                       "declared_bytes": info["declared_bytes"], "allowance_bytes": allowance}
    checks["capacity"] = info["declared_bytes"] >= cap - allowance
    from schemes.memory_link_exhaustion.verifier import act_pool as _ap
    ok_ap, ap_decl, ap_allowed = _ap.check(info, calib)
    rep["act_pool"] = {"declared_bytes": ap_decl, "allowed_bytes": ap_allowed}
    checks["no_act_pool"] = ok_ap
    if not ok_ap:
        log(f"no_act_pool FAIL: the prover declares an activation pool of {ap_decl} B ({ap_decl / 2**30:.3f} GiB); "
            f"the calibration allows {ap_allowed} B")
    # which content checks this run had (timing-only regions are not content-verified)
    kv_live = "zero_model_keystream" if a.zero_weights else ("reference_recompute" if not a.no_reference else "none")
    rep["content_checks"] = {"w_pages": True, "filler_pages": True, "kv_live": kv_live,
                             "mchase_chains": bool(a.zero_weights and a.mchase_verify > 0),
                             "outputs": "replay" if (a.output_replay and not a.zero_weights and not a.no_reference)
                             else "none"}
    rep["secrets"] = {"source": "debug_seed" if v.secret_seed else "os_csprng", "secret_seed": v.secret_seed,
                      "post_seeds": [str(x) for x in v.post_seeds]}
    if getattr(a, "prover", None) == "cheater" and kv_live == "none":
        # a real-weights cheater run must recompute (a sample of) claimed live KV, else KV is unverified
        checks["kv_live_content"] = False
    m = v.mem
    if a.pcie and a.cross_key:
        checks["chain_window"] = m.get("win_bad", 0) == 0 and m.get("win_checked", 0) > 0
        wl = [x for x in m.pop("win_lag", []) if x is not None]
        m["win_ahead_max"] = max(wl) if wl else None
    checks["w_pages"] = m["w_bad"] == 0
    # content of the chase-type challenges (zero-weight model: every probed word is predictable)
    for key, nb, nc in (("mchase", "chains_bad", "chains_checked"),):
        st = m.get(key)
        if st and st.get(nc):
            checks[f"{key}_content"] = st[nb] == 0
    checks["f_pages"] = m["f_bad"] == 0 and m["kfree_bad"] == 0
    if m["k_pages"]:
        exc = sum(n for s, n in m["k_states"].items() if s not in ("settled", "filler"))
        m["k_excused_frac"] = exc / m["k_pages"]
        checks["k_excused"] = m["k_excused_frac"] <= a.max_excused
    r = rep.get("reference")
    if r and "error" not in r:
        checks["kv_pages"] = r["kv_bad"] == 0
        if a.output_replay:
            checks["outputs"] = r["outputs_ok"] == r["outputs_checked"]
    if a.pcie:
        p = rep["pcie"]
        checks["pcie_chain"] = p["bad_offsets"] == 0 and p["gaps"] == 0
        if p.get("sched"):
            # (sched_n: windows the Revealer process timestamped, in order)
            n_ok = p.get("sched_n", p["windows"])
            if "verifier_frozen_s" in p:
                # verifier health (voids the run, not a prover check): the Revealer was frozen < 2% of the run
                # (the larger of --verifier-frozen-max-s and 2% of the session)
                checks["verifier_healthy"] = p.get("verifier_frozen_s", 0) <= max(
                    float(a.verifier_frozen_max_s), 0.02 * max(1.0, n_ok * a.window_mib * (1 << 20) / (a.noise_gbs * 1e9)))
    for kind, ls in getattr(v, "lat_k", {}).items():
        # mean rule: drop the slowest answer, the mean of the rest must stay within --mean-tol of the
        # slowest honest floor run's same statistic (--mean-ref-ms, from floors.env MEANREF; 0 = off). The tail rule
        # bounds each answer; this bounds the total extra time a cheater spends over the session. A session with
        # fewer than two answers fails the rule (nothing to average).
        if len(ls) >= 2:
            mean1 = float(np.mean(sorted(ls)[:-1]))
            m.setdefault(f"{kind}_timing", {})["mean1_ms"] = round(mean1, 4)
            if kind == "mchase" and a.mean_ref_ms > 0:
                checks[f"{kind}_mean"] = mean1 <= a.mean_ref_ms * (1 + a.mean_tol)
                m[f"{kind}_timing"].update({"mean_ref_ms": a.mean_ref_ms, "mean_ratio": round(mean1 / a.mean_ref_ms, 4)})
        elif kind == "mchase" and a.mean_ref_ms > 0:
            checks[f"{kind}_mean"] = False
        # timing rules against the calibration (T = honest p99): the tail rule and the mean rule above are the only
        # counted ones. The 95%-within-T, median and share rules were removed: none fired in any of
        # the 24 cheater / idle / RL runs re-scored from Boxes R to W. A steady per-answer delay is bounded by the
        # mean rule (5% of the slowest honest floor run's mean, plus the head start a non-serving prover has: ~0.8 ms
        # on Box W), single slow answers by the tail rule; under_frac, median_ms and median_ratio are still reported.
        c = calib or {}
        T = c.get(f"lat_{kind}_ms_p99")
        if T and ls:
            under = sum(1 for x in ls if x <= T) / len(ls)
            if a.hard_deadline_ms > 0:
                # rare-challenge regime: every challenge must be decisive on its own (no statistics), so each
                # answer must land within a fixed deadline (honest max over a long halted run + margin)
                checks[f"{kind}_hard"] = max(ls) <= a.hard_deadline_ms
            n_over = None
            if a.hard_mult > 0:
                # tail rule: at most --tail-allow answers slower than hard_mult x T (per kind). The count is reported
                # either way (over_tail), so a single late answer can be told from a prover that is late often.
                n_over = sum(1 for x in ls if x > a.hard_mult * T)
                checks[f"{kind}_tail"] = n_over <= a.tail_allow
            m[f"{kind}_timing"] = {**m.get(f"{kind}_timing", {}), "max_ms": round(max(ls), 3), "over_tail": n_over,
                                   "top_ms": [round(x, 3) for x in sorted(ls)[-5:]],
                                   "T_ms": round(T, 3), "under_frac": round(under, 4), "n": len(ls),
                                   "median_ms": round(float(np.median(ls)), 3),
                                   "median_ratio": round(float(np.median(ls)) /
                                                         c.get(f"lat_{kind}_ms_p50", 1), 3)}
            rep[f"deadline_{kind}_ms"] = T
    if calib:
        # throughput vs calibration: reported only (no check depends on how fast the prover computes)
        rep["throughput"] = {"tok_s": round(v.tok_s, 1), "calib_tok_s": round(calib["tok_s"], 1),
                             "ratio": round(v.tok_s / calib["tok_s"], 4),
                             "ok": v.tok_s >= calib["tok_s"] * (1 - a.tol_tput)}
        rep["calib"] = calib
    if a.selftest:
        rep["selftest"] = {**m.get("selftest", {}),
                           "K_mutants_identical": (r or {}).get("mutants")}
        checks = {k: x for k, x in checks.items() if k not in ("w_pages", "f_pages")}
    if getattr(a, "kv_table", 0):
        # settled-KV rule, first stage: what the chase answers say about the block table they ran under
        kv_ = [(float(v.chal_sent[c][0]), v.responses[c]["kvt"]) for c in v.responses
               if v.responses[c].get("kind") == "mchase" and v.responses[c].get("kvt") is not None and c in v.chal_sent]
        kv_.sort()
        vers = [x[1][0] for x in kv_]
        rep["kv_table"] = {"answers": len(kv_), "versions_first_last": [vers[0], vers[-1]] if vers else None,
                           "distinct_versions": len(set(vers)), "version_steps_back": sum(1 for i in range(1, len(vers)) if vers[i] < vers[i - 1]),
                           "distinct_digests": len(set(x[1][1] for x in kv_))}
        try:
            # every answer's table: present in the prover's log, digest equal to the one in the timed answer,
            # structurally sound; and how much of KV it leaves unread
            from schemes.memory_link_exhaustion.inference import kvtable as _KT
            bodies = prover_tables(a, info)
            miss = dmis = sbad = 0
            masked, seen_v, committed = [], {}, []
            mult0 = {}
            for p_ in common.prompts(a):            # a prompt the workload holds m times may be live m times
                mult0[Verifier._fp64(p_)] = mult0.get(Verifier._fp64(p_), 0) + 1
            for _t, (ver, dig) in kv_:
                if ver in seen_v:
                    masked.append(seen_v[ver])
                    continue
                b_ = bodies.get(ver)
                if b_ is None:
                    miss += 1
                    continue
                if _KT.body_digest(b_) != dig:
                    dmis += 1
                    continue
                t_ = _KT.parse(b_)
                if _KT.check(t_, mult0):
                    sbad += 1
                seen_v[ver] = t_.masked_tokens()
                masked.append(seen_v[ver])
                committed.append((ver, t_))
            tok_mib = info["L"] * 4096 / 2**20
            rep["kv_table"].update({"tables_in_log": len(bodies), "missing": miss, "digest_mismatch": dmis,
                                    "structure_bad": sbad,
                                    "masked_mib_mean": round(sum(masked) / max(1, len(masked)) * tok_mib, 1),
                                    "masked_mib_max": round(max(masked or [0]) * tok_mib, 1),
                                    "masked_share_mean": round(sum(masked) / max(1, len(masked)) / (info["nb"] * 16), 5)})
            # tight tables: the band can hold at most the tokens of the steps in flight. The engine schedules at most
            # --max-batched-tokens per step and the band spans the settle lag plus the step being scheduled, so
            # (lag + 2) x that many token positions bounds every table the prover switched in, not only the answered
            # ones. A table over the bound hides memory from the chase: the run fails.
            band_ok = True
            if bodies and any(_KT.parse(b_).tight for b_ in list(bodies.values())[:1]):
                lag_ = int(os.environ.get("VARENA_SETTLE_LAG", "0"))
                # (lag + 2): the settled count used is the one `lag` calls back, which itself excludes the steps that
                # were in flight then (up to two with asynchronous scheduling); measured maximum 4 x the step budget
                bound = (lag_ + 2) * int(a.max_batched_tokens or 1024)
                mx, over, unk = 0, 0, 0
                for ver in sorted(bodies):
                    t_ = _KT.parse(bodies[ver])
                    mt = t_.masked_tokens()
                    unk = max(unk, int((t_.kind == _KT.UNKNOWN).sum()))
                    mx = max(mx, mt)
                    over += int(mt > bound)            # (blocks of an unnamed owner are fully masked: they count)
                band_ok = over == 0
                rep["kv_table"].update({"band_bound_tokens": bound, "band_max_tokens_all_tables": mx,
                                        "band_max_mib_all_tables": round(mx * tok_mib, 1), "tables_over_bound": over,
                                        "unknown_owner_blocks_max": unk})
            # the committed tables taken together (tight): no band may stay on the same positions of the same owner
            # across many schedule() calls, the version numbers may not advance slower than requests decode (one
            # token a call), and every position's named writer must be a request that is long enough to have written
            # it. Honest runs (kt2, sgkt1 logs, every table): a position is in the band for exactly 4 consecutive
            # versions (largest gap 3), and no request is ever ahead of one token per call (excess 0).
            hist_ok = True
            committed.sort(key=lambda x: x[0])
            if committed and committed[0][1].tight:
                # the committed tables taken together (kvtable.session_verdict): owners are the verifier's own
                # requests, no unnamed owners, every band position is read and written by its named owner lag + 2
                # versions later, no request ahead of one token per version. Settle lag and step budget are the
                # verifier's own (--settle-lag, --max-batched-tokens), not the prover's.
                prompts_ = common.prompts(a)
                tokens_of, mult_ = {}, {}
                for i_, p_ in enumerate(prompts_):
                    k_ = Verifier._fp64(p_)
                    mult_[k_] = mult_.get(k_, 0) + 1
                    tokens_of[k_] = max(tokens_of.get(k_, 0), len(p_) + (len(v.tokens[i_]) if i_ < len(v.tokens) else 0))
                lag_v = int(a.settle_lag) if a.settle_lag >= 0 else int(os.environ.get("VARENA_SETTLE_LAG", "0"))
                hi_ = _KT.session_verdict(committed, tokens_of, mult_, lag=lag_v,
                                          step_budget=int(a.max_batched_tokens or 1024),
                                          linger_max=int(a.kv_linger_max), pace_max=int(a.kv_pace_max),
                                          promise_frac_max=float(a.kv_promise_frac_max),
                                          mask_mean_max_tokens=float(a.kv_mask_mean_max_mib) / tok_mib)
                hi_["masked_mib_mean_committed"] = round(hi_["masked_tokens_mean"] * tok_mib, 1)
                hist_ok = bool(hi_["ok"])
                rep["kv_table"]["history"] = hi_
            checks["kv_table"] = bool(kv_) and miss == 0 and dmis == 0 and sbad == 0 and band_ok and hist_ok and \
                rep["kv_table"]["version_steps_back"] == 0
        except Exception as e:
            rep["kv_table"]["error"] = f"{type(e).__name__}: {e}"
            checks["kv_table"] = False
        log("kv_table:", json.dumps(rep["kv_table"]))
    try:
        # raw verifier-side evidence for the CPU replay (scripts/reproduce/replay.py): every window's arrival stamp,
        # the ring challenge times and the Revealer's freeze log. Scoring rules can then be rerun without a GPU.
        rv_ = getattr(v, "revealer", None)
        if rv_ is not None and a.out:
            nrep = int(rv_.all[2 * rv_.N + 4])
            ring_t = sorted((float(t) - rv_.t0, int(c)) for c, (t, _n, k_) in v.chal_sent.items() if k_ == "ring")
            # every mixed chase's and ring sample's (t_sent, t_recv, head) on the Revealer's clock: the head rule's evidence
            # (t_recv_wall: stamped after the round trip returned; t_sent + rtt alone misses any delay between the
            # send stamp and the actual send, which read as a 5-window "jump" in the first GPU run)
            heads = sorted((float(t) - rv_.t0, float(v.responses[c].get("t_recv_wall", v.responses[c]["t_recv"])) - rv_.t0,
                            int(v.responses[c].get("win", -1)))
                           for c, (t, _n, k_) in v.chal_sent.items()
                           if k_ in ("mchase", "ring") and c in v.responses and "t_recv" in v.responses[c])
            np.savez_compressed(a.out + ".replay.npz", arr_us=np.asarray(rv_.dl[:max(0, min(nrep, len(rv_.dl)))]),
                                heads=np.asarray(heads, dtype=np.float64).reshape(-1, 3),
                                ring_t=np.array([x[0] for x in ring_t]), ring_cid=np.array([x[1] for x in ring_t]),
                                freeze=np.asarray(rv_.freeze_events() if hasattr(rv_, "freeze_events") else np.zeros((0, 3))),
                                W=int(v.checker.W), t0_abs=float(rv_.t0), slack=float(v.checker.sched["slack"]),
                                floor_gbs=float(os.environ.get("VARENA_SAT_FLOOR_GBS", "0") or 0))
            rep["replay"] = {"file": os.path.basename(a.out) + ".replay.npz", "t0_abs": float(rv_.t0), "windows": nrep,
                             "ring_challenges": len(ring_t)}
            if len(heads) >= 2:
                # head rule: the head the prover returns with each mixed chase cannot advance faster than the link
                # (--noise-gbs, the nominal maximum) between consecutive chases, nor go back; and the link rate counted
                # from those heads (no prover report involved). --head-floor-gbs > 0 makes both a counted check.
                from schemes.memory_link_exhaustion.verifier import floor as _flh
                hr = _flh.head_rule(heads, int(v.checker.W), float(a.noise_gbs),
                                    rv_.freeze_events() if hasattr(rv_, "freeze_events") else None)
                hr = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in hr.items()}
                hr["floor_gbs"] = float(a.head_floor_gbs)
                rep["head_rule"] = hr
                if a.head_floor_gbs > 0:
                    hr["ok"] = bool(hr["jumps"] == 0 and hr["backwards"] == 0 and (hr["gbs"] or 0) >= a.head_floor_gbs)
                    checks["head_rate"] = hr["ok"]
            # the last timed answer of the session (mixed chase or ring sample), on the Revealer's clock
            ans_t = sorted(float(t) - rv_.t0 for c, (t, _n, k_) in v.chal_sent.items()
                           if k_ in ("mchase", "ring") and c in v.responses)
            if ans_t:
                # end-of-session rule: one number for the link. Bytes that had arrived by the last timed challenge
                # (verifier-side stamps, counted from t0) over the elapsed time, with the Revealer's own frozen time
                # taken off the clock when the freeze credit is on. --end-floor-gbs > 0 makes it a counted check; a
                # session shorter than --end-min-s fails it (one stall moves a short average too far).
                arr_ = np.asarray(rv_.dl[:max(0, min(nrep, len(rv_.dl)))], dtype=np.float64)
                el_ = ans_t[-1] - float(v.checker.sched["slack"])
                t_cr = ans_t[-1]
                log_full = bool(nrep > len(rv_.dl) and len(arr_))
                if log_full:
                    # a session longer than the arrival log: the rate is taken over the logged part only (windows
                    # after it were never stamped, and counting the whole elapsed time would score them as missing)
                    t_last = float(arr_[-1]) * 1e-6
                    if t_last < el_:
                        el_, t_cr = t_last, t_last + float(v.checker.sched["slack"])
                gb_ = int(np.searchsorted(arr_, el_ * 1e6, side="right")) * int(v.checker.W) / 1e9
                cr_ = float(rv_.freeze_credit(t_cr)) if hasattr(rv_, "freeze_credit") else 0.0
                er = {"elapsed_s": round(el_, 2), "gb": round(gb_, 2), "gbs_no_credit": round(gb_ / max(el_, 1e-6), 4),
                      "arrival_log_full": log_full,
                      "gbs": round(gb_ / max(el_ - cr_, 1e-6), 4), "credit_s": round(cr_, 4),
                      "floor_gbs": float(a.end_floor_gbs), "min_s": float(a.end_min_s)}
                rep["end_rule"] = er
                if a.end_floor_gbs > 0:
                    er["ok"] = bool(el_ >= a.end_min_s and er["gbs"] >= a.end_floor_gbs)
                    checks["pcie_end_rate"] = er["ok"]
    except Exception as e:                                   # a report file must never cost a run
        rep["replay"] = {"error": f"{type(e).__name__}: {e}"}
    # link rules: the end-of-session rate (--end-floor-gbs) and the head rule (--head-floor-gbs), against floors a
    # host's own honest runs set (floors.sh). Without floors they are reported but decide nothing; the report says so.
    rep["link_rules_counted"] = bool(a.pcie and "head_rate" in checks and "pcie_end_rate" in checks)
    h0, h1 = getattr(v, "_host0", None), getattr(v, "_host1", None) or host_snapshot()
    if h0:
        rep["host"] = {k: (round(h1[k] - h0[k], 3) if k == "t" else h1[k] - h0[k]) for k in h0 if k in h1}
        log("host over the session:", json.dumps(rep["host"]))
    rep["checks"] = checks
    rep["pass"] = all(checks.values())
    if a.write_calib:
        extra = {}
        for kind, lk in getattr(v, "lat_k", {}).items():
            lk = sorted(lk)
            extra[f"lat_{kind}_ms_p50"] = lk[len(lk) // 2]
            extra[f"lat_{kind}_ms_max"] = lk[-1]
            extra[f"lat_{kind}_ms_p99"] = lk[min(len(lk) - 1, int(0.99 * len(lk)))]
            extra[f"lat_{kind}_ms_p75"] = lk[min(len(lk) - 1, int(0.75 * len(lk)))]
            extra[f"lat_{kind}_ms_p90"] = lk[min(len(lk) - 1, int(0.90 * len(lk)))]
            extra[f"deadline_{kind}_ms"] = max(a.deadline_mult_p50 * lk[len(lk) // 2],
                                               a.deadline_mult_max * lk[-1])
        cal = {**extra, "tok_s": v.tok_s, "pcie_gbs": (rep.get("pcie") or {}).get("rate_gbs"),
               "outside_gib": info["outside_bytes"] / 2**30,
               "allowance_gib": round(info["outside_bytes"] / 2**30 + a.allowance_margin_gib, 3),
               "allowance_margin_gib": a.allowance_margin_gib,
               "act_pool_bytes": int(((info.get("act_pool") or {}).get("bytes")) or 0),
               **pad_calib(v.stats, a)}
        json.dump(cal, open(a.write_calib, "w"), indent=1)
        rep["wrote_calib"] = cal
    json.dump(rep, open(a.out, "w"), indent=1)
    log("RESULT", json.dumps({"tok_s": v.tok_s, "pass": rep["pass"], "checks": checks}))
    if getattr(v, "revealer", None) is not None:
        v.revealer.stop()
    for f in (v.pool_path, getattr(v, "rev_path", ""), getattr(v, "rev_path", "") + ".dl"):
        if os.path.exists(f):
            os.unlink(f)
    os._exit(0)


if __name__ == "__main__":
    main()
