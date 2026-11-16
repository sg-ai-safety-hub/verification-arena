"""Prover side, inside vLLM's EngineCore process (scheduler + GPU worker).

Enabled by VARENA=1 (see plugin.py). The verifier owns the secrets and is the client; this module connects to its
socket (VARENA_SOCK) when the worker loads the model.

Arena (64 KiB pages, one global page index space, sealed once after warm-up):
  W  [0, nW)          one buffer holding every big weight matrix, sloth64-encoded in place (the embedding stays
                      plain); gaps are filler
  K  [nW, nW+L*nb)    page nW + l*nb + b = vLLM KV block b of layer l (one layer's block is 64 KiB), encoded on
                      store; with stale KV (VARENA_STALE_KV=1) a freed block keeps its last owner's KV and is claimed
                      under that owner instead of being refilled
  F  after            filler covering the rest of the declared memory
  ring                the last noise windows (the PCIe streamer's HBM ring)
Sized from the verifier's pad (the honest pad probe's measured peak, verifier/pad.py).

Threads in this process (besides vLLM's): socket reader, native challenge server (gpu/native.py, own high-priority
stream), the native noise streamer + flusher (link/pcie.py), stall watchdog. Scheduler hooks (same thread as the
worker): KV block events, settled-token snapshots and the chase-bound KV claims.
"""

from __future__ import annotations

import json
import collections
import os
import queue
import socket
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

HEAVY_TOK = int(os.environ.get("VARENA_HEAVY_TOK", "1024"))
MEMSNAP = (os.environ.get("VARENA_MEMSNAP") or "").replace("{pid}", str(os.getpid())) or None

import faulthandler as _fh, signal as _sig
try:                                   # `kill -USR1 <EngineCore>` dumps every thread's Python stack to stderr
    _fh.register(_sig.SIGUSR1, all_threads=True)
except Exception:
    pass
import torch

PROF_KV = os.environ.get("VARENA_KV_PROF") == "1"
REFILL_ORDERED = os.environ.get("VARENA_REFILL_ORDERED", "1") == "1"
STALE_KV = os.environ.get("VARENA_STALE_KV", "0") == "1"

if os.environ.get("VARENA_GC_LOG", "0") == "1" or os.path.exists("/root/diag_gc"):
    # (noise-dip diagnosis): log every CPython GC pass that takes > 20 ms in this process; a long
    # gen-2 collection holds the GIL and would delay the Python-side noise completion reports
    import gc as _gc
    _gc_t0 = [0.0]

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_t0[0] = time.perf_counter()
        else:
            dt = time.perf_counter() - _gc_t0[0]
            if dt > 0.02 or info.get("generation") == 2:
                print(f"{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d} [varena-prover] gc pid {os.getpid()} gen "
                      f"{info.get('generation')} {dt * 1e3:.0f} ms collected {info.get('collected')}",
                      file=sys.stderr, flush=True)
    _gc.callbacks.append(_gc_cb)
PAUSE_VLLM = os.environ.get("VARENA_PAUSE_VLLM") == "1"

from schemes.memory_link_exhaustion.gpu import kernels as K
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.core.proto import Conn

PAGE = ref.PAGE
STATE = None          # the ArenaState of this process (worker side)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[varena-prover]", *a, file=sys.stderr, flush=True)


def raw_bytes(t: torch.Tensor, nbytes: int) -> torch.Tensor:
    """uint8 view of nbytes of t's memory from its first element (any strides)."""
    flat = torch.as_strided(t, (nbytes // t.element_size(),), (1,))
    return flat.view(torch.uint8)


def kv_key(master: int, layer: int) -> int:
    return ref.tensor_key(master, (1 << 44) + layer * PAGE)


def weight_entries(cfg):
    """(name, shape, hf sources) of the encoded weights, in W order."""
    from schemes.memory_link_exhaustion.core.model import big_tensors
    return big_tensors(cfg)


PLAIN = -1     # key of a matrix stored in plaintext (no keystream)


def w_layout(cfg, master, plain_embed=True, plain_all=False):
    """[(name, offset, shape, sources, key)] and total W bytes (page aligned).
    The embedding stays plaintext: on Hopper, replacing vLLM's F.embedding with a
    decoding gather changed torch.compile's fusions downstream (outputs not bit-exact);
    it is still in W and checked against the declared weights."""
    out, off = [], 0
    for name, shape, src in weight_entries(cfg):
        n = int(np.prod(shape)) * 2
        key = PLAIN if (plain_all or (plain_embed and name == "embed")) else ref.tensor_key(master, off)
        out.append((name, off, list(shape), src, key))
        off = (off + n + PAGE - 1) // PAGE * PAGE
    return out, off


def _module_for(model, name):
    m = model.model
    if name == "embed":
        return m.embed_tokens
    if name == "lm_head":
        return model.lm_head
    l, what = name[1:].split(".")
    layer = m.layers[int(l)]
    return {"qkv": layer.self_attn.qkv_proj, "o": layer.self_attn.o_proj,
            "gate_up": layer.mlp.gate_up_proj, "down": layer.mlp.down_proj}[what]


class BlockLog:
    """Per KV block: recent (time, kind) events, kind in alloc/free/refilled."""

    def __init__(self, nb):
        self.ev = [deque(maxlen=6) for _ in range(nb)]

    def add(self, blocks, kind, t=None):
        t = time.perf_counter() if t is None else t
        for b in blocks:
            self.ev[b].append((t, kind))

    def last(self, b):
        return self.ev[b][-1][1] if self.ev[b] else None

    def classify(self, b, t0, t1):
        """State of block b over [t0, t1]: filler | live | dirty | transition | unknown."""
        evs = list(self.ev[b])
        if any(t0 <= t <= t1 for t, _ in evs):
            return "transition"
        before = [k for t, k in evs if t < t0]
        if not before:
            if evs and len(evs) == self.ev[b].maxlen:
                return "unknown"      # history too short
            return "filler"           # never touched since seal
        last = before[-1]
        return {"refilled": "filler", "alloc": "live", "free": "dirty", "pending": "transition", "stale": "stale"}[last]


class ArenaState:
    def __init__(self):
        self.sealed = False
        self.conn = None
        self.cfg = None
        self.lock = threading.Lock()
        self.chal_q = queue.Queue()
        self.claim_q = deque()             # challenges whose K pages need classifying
        self.refills = deque()             # (pool, blocks, done_event, pinned ids)
        self.pools = []
        self.snaps = deque(maxlen=256)     # (t, {rid: (settled, req, blocks)})
        self.stale_owner = {}              # stale-KV route: block -> (prompt fp, block index, settled) | None
        self.step_times = deque(maxlen=4096)   # (t, scheduled tokens) per schedule()
        self.stats = {"refill_launches": 0, "refill_blocks": 0, "challenges": 0,
                      "claims": {}, "held_max": 0, "chal_ms": []}
        self.held = 0
        self.streamer = None
        self.corrupt_log = []

    # ------------------------------------------------------------------ connect
    def connect(self, model_path):
        path = os.environ["VARENA_SOCK"]
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(100):
            try:
                s.connect(path)
                break
            except OSError:
                time.sleep(0.1)
        self.conn = Conn(s)
        self.conn.send({"op": "hello", "pid": os.getpid()})
        m = self.conn.recv()
        assert m and m["op"] == "config", m
        self.cfg = m["cfg"]
        self.master, self.fkey = int(self.cfg["master"]), int(self.cfg["fkey"])
        from schemes.memory_link_exhaustion.core.model import Cfg
        self.mcfg = Cfg.from_hf(model_path)
        self.model_path = model_path
        if os.environ.get("VARENA_MEMSNAP"):
            # leftover census: record allocation stacks in this (engine) process from here on; recording from the
            # plugin's register() initialised CUDA in the parent before it forked the engine
            torch.cuda.memory._record_memory_history(max_entries=400000, stacks="python")
        sys.setswitchinterval(float(self.cfg.get("switch_interval", 0.0005)))
        log("connected; config received")

    # ------------------------------------------------------------------ weights (W)
    @torch.no_grad()
    def encode_weights(self, model):
        ents, w_bytes = w_layout(self.mcfg, self.master, plain_all=bool(self.cfg.get("plain_w")))
        self.w_entries, self.w_bytes = ents, w_bytes
        self.nW = w_bytes // PAGE
        buf = torch.empty(w_bytes, dtype=torch.uint8, device="cuda")
        K.fill(buf, 0, self.fkey)
        tied = {}
        for name, off, shape, src, key in ents:
            mod = _module_for(model, name)
            p = mod.weight
            assert list(p.shape) == shape and p.dtype == torch.bfloat16, (name, p.shape, shape)
            n = int(np.prod(shape))
            dst = buf[off: off + 2 * n].view(torch.int16).view(shape)
            dst.copy_(p.data.view(torch.int16))
            p.data = dst.view(torch.bfloat16)
            only = self.cfg.get("debug_wenc_only")
            if key != PLAIN and not self.cfg.get("debug_no_wenc") and \
                    (not only or any(o in name for o in only)):
                from schemes.memory_link_exhaustion.inference import patches as _P
                if ref.WENC in ("bfly", "sloth"):
                    wv = dst.view(torch.bfloat16).view(shape)
                    ref.w_encode(wv, wv, key, ref.w_ordinal(off))
                    _P.W_KEYS[int(dst.data_ptr())] = (int(key), ref.w_ordinal(off))
                else:
                    K.xor16_w(dst, dst, key, shape[1])
                    _P.W_KEYS[int(dst.data_ptr())] = int(key)
                mod._varena_key = key
            tied[name] = mod
        if self.mcfg.tied and not self.cfg.get("debug_no_wenc") and \
                getattr(model.model.embed_tokens, "_varena_key", None) is not None:
            model.lm_head._varena_key = model.model.embed_tokens._varena_key
            assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        self.wbuf = buf
        self.vllm_model = model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"encoded {len(ents)} weight matrices into W ({w_bytes / 2**30:.2f} GiB, {self.nW} pages)")

    # ------------------------------------------------------------------ KV (K)
    def tag_kv(self, vllm_config):
        """Find the per-layer KV caches, check the 64 KiB block layout, set keys."""
        ctx = vllm_config.compilation_config.static_forward_context
        from vllm.model_executor.models.utils import extract_layer_index
        layers = []
        for name, mod in ctx.items():
            kv = getattr(mod, "kv_cache", None)
            if kv is None:
                continue
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
                continue
            layers.append((extract_layer_index(name), name, mod, kv))
        layers.sort(key=lambda x: x[0])
        assert [l for l, *_ in layers] == list(range(len(layers))), [l for l, *_ in layers]
        self.kv = []
        nb = None
        for l, name, mod, kv in layers:
            assert kv.dtype == torch.bfloat16, kv.dtype
            blk = kv[0].numel() * 2
            assert blk == PAGE, f"{name}: one KV block is {blk} bytes, need {PAGE} (set block_size)"
            assert kv.stride(0) * 2 == PAGE and kv.stride(3) == 1, (name, kv.shape, kv.stride())
            nb = nb or kv.shape[0]
            assert kv.shape[0] == nb
            if not self.cfg.get("debug_no_kvenc"):
                mod._varena_kv_key = kv_key(self.master, l)
            self.kv.append(kv)
        self.L, self.nb = len(self.kv), nb
        self.kv_shape = list(self.kv[0].shape)
        self.kv_stride = list(self.kv[0].stride())
        self.block_log = BlockLog(nb)
        log(f"KV: {self.L} layers x {nb} blocks of 64 KiB, shape {self.kv_shape} "
            f"stride {self.kv_stride}")

    # ------------------------------------------------------------------ seal
    @torch.no_grad()
    def seal(self, worker):
        cfg = self.cfg
        torch.cuda.synchronize()
        nW, L, nb = self.nW, self.L, self.nb
        for l, kv in enumerate(self.kv):
            K.fill(raw_bytes(kv, nb * PAGE), nW + l * nb, self.fkey)
        self.nK = L * nb
        # PCIe noise stream (allocates its HBM rings before F is sized)
        if cfg.get("pcie"):
            from schemes.memory_link_exhaustion.link.pcie import Streamer
            self.streamer = Streamer(cfg["pool_path"], cfg["pcie_key"],
                                     window_bytes=cfg["window_bytes"], lag=cfg["lag"],
                                     on_window=self._on_window,
                                     ring_bytes=int(cfg.get("ring_mib", 0)) << 20,
                                     sched=cfg.get("noise_sched"))
        self.chal_stream = torch.cuda.Stream(priority=-1)
        # high priority: refill CTAs get SMs at vLLM's CTA boundaries instead of waiting behind
        # its whole step (freed KV blocks are held until refilled; at full KV load the holding
        # starved vLLM: 5-7k of 8192 blocks held at peak)
        self.fill_stream = torch.cuda.Stream(priority=-1)
        # filler region F: the rest of free HBM (minus pad), in 1 GiB chunks
        npages_est = nW + self.nK + (torch.cuda.mem_get_info()[0] // PAGE)
        self.page_map = torch.empty(npages_est + 16, dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        free, cap = torch.cuda.mem_get_info()
        pad = int(cfg.get("pad_mib", 512)) << 20
        f_bytes = max(0, free - pad)
        self.f_chunks = []
        first = nW + self.nK
        left = (f_bytes + PAGE - 1) // PAGE * PAGE
        while left > 0:
            # small chunks fit into the allocator's cached (fragmented) free blocks
            n = min(left, int(cfg.get("f_chunk_mib", 64)) << 20)
            t = torch.empty(n, dtype=torch.uint8, device="cuda")
            K.fill(t, first, self.fkey)
            self.f_chunks.append((first, t))
            first += n // PAGE
            left -= n
        self.lend_first = first      # no lending
        # region R: the noise ring (declared; content = the noise window in each slot)
        self.R0 = first
        self.nR = (self.streamer.ring.numel() * 4 // PAGE) if (self.streamer and cfg.get("ring_mib")) else 0
        self.npages = self.R0 + self.nR
        # page map: int32-word offset of each page from the W buffer
        base = self.wbuf.data_ptr()
        pm = np.empty(self.npages, dtype=np.int64)
        pm[:nW] = np.arange(nW, dtype=np.int64) * (PAGE // 4)
        for l, kv in enumerate(self.kv):
            o = (kv.data_ptr() - base) // 4
            pm[nW + l * nb: nW + (l + 1) * nb] = o + np.arange(nb, dtype=np.int64) * (PAGE // 4)
        for f0, t in self.f_chunks:
            n = t.numel() // PAGE
            pm[f0: f0 + n] = (t.data_ptr() - base) // 4 + np.arange(n, dtype=np.int64) * (PAGE // 4)
        if self.nR:
            r = self.streamer.ring
            pm[self.R0:] = (r.data_ptr() - base) // 4 + np.arange(self.nR, dtype=np.int64) * (PAGE // 4)
        for kv in self.kv:
            assert (kv.data_ptr() - base) % 4 == 0
        self.page_map = torch.from_numpy(pm).to("cuda")
        self.base_i32 = self.wbuf.view(torch.int32)
        # warm the challenge kernels
        with torch.cuda.stream(self.chal_stream):
            K.digest_pages(self.base_i32, self.page_map,
                           torch.zeros(1, dtype=torch.int64, device="cuda"), 12345)
            # compile the refill kernel on a page that is filler already (last F page)
            K.fill_pages(self.base_i32, self.page_map,
                         torch.full((1,), self.R0 - 1, dtype=torch.int64, device="cuda"),
                         self.fkey)
        torch.cuda.synchronize()
        free_after, cap = torch.cuda.mem_get_info()
        # pad study: device state right after the arena is sized; everything the honest run needs on top of this
        # (torch's per-step growth plus lazily loaded driver memory) comes out of the pad
        self._padF = {"free": free_after, "cap": cap, "reserved": torch.cuda.memory_reserved(),
                      "allocated": torch.cuda.memory_allocated(), "pad": pad}
        from schemes.memory_link_exhaustion.encodings import actenc2 as _AE
        act_ranges = _AE.pool_ranges()
        act_bytes = sum(n for _, n in act_ranges)
        declared = self.npages * PAGE + act_bytes
        self.info = {"act_pool": {"granules": _AE.pool_granules(), "bytes": act_bytes, "master": _AE.MASTER,
                                  "epoch": 1} if act_bytes else None,"npages": self.npages, "nW": nW, "nK": self.nK, "L": L, "nb": nb,
                     "nF": self.R0 - nW - self.nK, "n_evicted": 0,
                     "nF_chase": self.lend_first - nW - self.nK,
                     "lend_pages": 0,
                     "R0": self.R0, "nR": self.nR,
                     "n_ring": self.streamer.n_ring if self.streamer else 0,
                     "kv_shape": self.kv_shape, "kv_stride": self.kv_stride,
                     "declared_bytes": declared, "capacity": cap, "free_after": free_after,
                     "outside_bytes": cap - declared, "w_bytes": self.w_bytes,
                     "torch_reserved": torch.cuda.memory_reserved(),
                     "pcie_hbm_bytes": self.streamer.hbm_bytes() if self.streamer else 0,
                     "gpu": torch.cuda.get_device_name(0), "w_entries": self.w_entries}
        # Load every challenge/refill kernel's GPU module now: Triton loads a kernel's module lazily on
        # its first launch (cuModuleLoad takes the CUDA driver lock), and the first KV-chain challenge or
        # refill at generation start blocked the native noise stream's copy enqueue for ~0.9 s.
        try:
            with torch.cuda.stream(self.chal_stream):
                # every integer specialisation a random u32 nonce can hit: Triton compiles a separate kernel for
                # 1, for multiples of 16, and for values >= 2^31 (a wider int type). Warming only small nonces left
                # the first KV-claim with a nonce >= 2^31 to compile mid-serving (seen once: 0.12 s
                # compile + 0.3 s load on the challenge thread, the serving thread's launch blocked 175 ms)
                for nz in (1, 16, 12345, (1 << 31) + 1, (1 << 31) + 16, (1 << 32) - 1):
                    K.digest_pages(self.base_i32, self.page_map, torch.zeros(1, dtype=torch.int64, device="cuda"), nz)
                    K.digest_pages(self.base_i32, self.page_map, torch.zeros(2048, dtype=torch.int64, device="cuda"), nz)
            with torch.cuda.stream(self.fill_stream):
                pg = torch.tensor([self.nW], dtype=torch.int64).pin_memory()    # a free KV page: already filler
                K.fill_pages(self.base_i32, self.page_map,
                             pg, self.fkey)
            torch.cuda.synchronize()
        except Exception as e:
            log("kernel warm-up failed:", e)
        # Pre-grow torch's caching allocator: vLLM's first big batch used to cudaMalloc ~1.7 GiB of new
        # segments mid-run, and the driver call blocked the native noise stream's copy enqueue for
        # 0.5-1 s (late windows). Allocate-and-free once now; later growth is served from the cache.
        # The verifier's calibration may supply the size (cfg prereserve_mib: torch's measured growth after seal); it is
        # then clamped to the pad minus the measured driver-side (non-torch) growth. (Unclamped, an allocation larger
        # than the free pad fails and torch releases its cache on the way: 1024 MiB into 0.98 GiB.)
        mib = int(cfg["prereserve_mib"]) if cfg.get("prereserve_mib") is not None else \
            int(os.environ.get("VARENA_PRERESERVE_MIB", "2048"))
        if mib > 0 and cfg.get("prereserve_mib") is not None:      # calibrated: keep the driver-side growth free
            nt_keep = int(cfg.get("pad_nontorch_mib", 64)) << 20
            mib = max(0, min(mib, (torch.cuda.mem_get_info()[0] - nt_keep) >> 20))
        if mib > 0:
            try:
                _t = torch.empty(mib << 20, dtype=torch.uint8, device="cuda")
                del _t
            except Exception as e:
                log("pre-reserve failed:", e)
        self.stats["prereserve_mib"] = mib
        self._mem_track_start()
        if cfg.get("native_sock"):
            self._start_native()
        self.sealed = True
        self.t_seal = time.perf_counter()
        self.conn.send({"op": "ready", "info": self.info})
        threading.Thread(target=self._reader, daemon=True, name="varena-reader").start()
        threading.Thread(target=self._challenger, daemon=True, name="varena-chal").start()
        if self.streamer:
            threading.Thread(target=self._flusher, daemon=True, name="varena-flush").start()
            self.streamer.start()
        # stall watchdog thread (independent of the scheduler hook, which does not run during a stall): if no
        # schedule() call for 8 s, dump every thread's stack once per stall
        self._compute_stream = torch.cuda.current_stream()
        streams = {"compute": self._compute_stream, "fill": self.fill_stream, "chal": self.chal_stream}
        if self.streamer is not None:
            for nm in ("s_h2d", "s_k", "s_d2h"):
                if hasattr(self.streamer, nm):
                    streams["noise_" + nm] = getattr(self.streamer, nm)

        try:
            import ctypes
            _crt = ctypes.CDLL("libcudart.so.12")
            fl = {}
            for nm, st_ in streams.items():
                f_, p_ = ctypes.c_uint(0), ctypes.c_int(0)
                _crt.cudaStreamGetFlags(ctypes.c_void_p(st_.cuda_stream), ctypes.byref(f_))
                _crt.cudaStreamGetPriority(ctypes.c_void_p(st_.cuda_stream), ctypes.byref(p_))
                fl[nm] = f"0x{st_.cuda_stream:x} {'NonBlocking' if f_.value & 1 else 'BLOCKING'} prio {p_.value}"
            log("stream flags:", json.dumps(fl))
        except Exception as e:
            log("stream flags failed:", e)

        def _stall_wd():
            import faulthandler
            torch.cuda.set_device(0)
            wd_stream = torch.cuda.Stream()
            pos_h = torch.zeros(8, dtype=torch.int64).pin_memory()
            dumped_at = None
            while True:
                time.sleep(1.0)
                last = getattr(self, "_last_sched_t", None)
                if last is None:
                    continue
                if time.perf_counter() - last > 8.0 and dumped_at != last:
                    dumped_at = last
                    print(f"[stall-watchdog] no schedule() for {time.perf_counter() - last:.1f}s; stacks:",
                          file=sys.stderr, flush=True)
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    # which of our streams still have outstanding work (non-blocking queries)
                    q = {}
                    for nm, st_ in streams.items():
                        try:
                            q[nm] = "idle" if st_.query() else "BUSY"
                        except Exception as e:
                            q[nm] = f"err {e}"
                    try:
                        from schemes.memory_link_exhaustion.encodings import actenc2 as _AEs
                        for nm, st_ in _AEs.VLLM_STREAMS.items():
                            if st_ is not None:
                                q["vllm_" + nm] = ("idle" if st_.query() else "BUSY") + f"(0x{st_.cuda_stream:x})"
                        q["compute_handle"] = f"0x{streams['compute'].cuda_stream:x}"
                    except Exception as e:
                        q["vllm"] = f"err {e}"
                    print(f"[stall-watchdog] streams: {q}; refills in flight {len(self.refills_ordered)} held "
                          f"{len(self.refills)} deferred {len(self.deferred)} busy_flag {int(self.busy[0]) if getattr(self, 'busy', None) is not None else None}",
                          file=sys.stderr, flush=True)
                    # device state, 3 reads 1 s apart on our own stream: position marker, gate (a blocking copy
                    # here means the device is not servicing other streams)
                    from schemes.memory_link_exhaustion.encodings import actenc2 as _AEw
                    # small H2D on our own stream, timed: a starved host-to-device path shows here
                    try:
                        hsrc = torch.ones(1024, dtype=torch.int64).pin_memory()
                        hdst = torch.empty(1024, dtype=torch.int64, device="cuda")
                        t0 = time.perf_counter()
                        with torch.cuda.stream(wd_stream):
                            hdst.copy_(hsrc, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(wd_stream)
                        while not ev.query() and time.perf_counter() - t0 < 5.0:
                            time.sleep(0.001)
                        print(f"[stall-watchdog] h2d 8 KiB on own stream: "
                              f"{'%.1f ms' % ((time.perf_counter() - t0) * 1e3) if ev.query() else 'BLOCKED >5s'}",
                              file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[stall-watchdog] h2d probe failed: {e}", file=sys.stderr, flush=True)
                    # tiny kernel on fresh high- and low-priority streams, timed: if these run while the compute
                    # stream is frozen, SMs are free and the compute stream is held by a queue dependency
                    for pr_ in (-5, 0):
                        try:
                            ks_ = torch.cuda.Stream(priority=pr_)
                            kt_ = torch.zeros(16, device="cuda")
                            t0 = time.perf_counter()
                            with torch.cuda.stream(ks_):
                                kt_.add_(1.0)
                            ev = torch.cuda.Event()
                            ev.record(ks_)
                            while not ev.query() and time.perf_counter() - t0 < 5.0:
                                time.sleep(0.001)
                            print(f"[stall-watchdog] tiny kernel, new stream prio {pr_}: "
                                  f"{'%.1f ms' % ((time.perf_counter() - t0) * 1e3) if ev.query() else 'BLOCKED >5s'}",
                                  file=sys.stderr, flush=True)
                        except Exception as e:
                            print(f"[stall-watchdog] kernel probe failed: {e}", file=sys.stderr, flush=True)
                    try:
                        if _AEw.VLLM_EVENTS:
                            e0, e1, t_ = _AEw.VLLM_EVENTS
                            print(f"[stall-watchdog] last AsyncOutput ({time.perf_counter() - t_:.2f}s ago): main-stream point "
                                  f"{'done' if e0.query() else 'PENDING'}, output D2H {'done' if e1.query() else 'PENDING'}",
                                  file=sys.stderr, flush=True)
                        dsrc = torch.ones(512, dtype=torch.int32, device="cuda")
                        for pr_ in (0, -1):
                            ddst = torch.empty(512, dtype=torch.int32).pin_memory()
                            ds_ = torch.cuda.Stream(priority=pr_)
                            t0 = time.perf_counter()
                            with torch.cuda.stream(ds_):
                                ddst.copy_(dsrc, non_blocking=True)
                            ev = torch.cuda.Event()
                            ev.record(ds_)
                            while not ev.query() and time.perf_counter() - t0 < 5.0:
                                time.sleep(0.001)
                            print(f"[stall-watchdog] d2h 2 KiB on fresh stream prio {pr_}: "
                                  f"{'%.1f ms' % ((time.perf_counter() - t0) * 1e3) if ev.query() else 'BLOCKED >5s'}",
                                  file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[stall-watchdog] async-output probe failed: {e}", file=sys.stderr, flush=True)
                    print(f"[stall-watchdog] tail ops seen by dispatch mode: {_AEw.OPS_SEEN[0]}; tags {_AEw.TAG_IDS}",
                          file=sys.stderr, flush=True)
                    if "vshim" in os.environ.get("LD_PRELOAD", ""):
                        try:
                            import ctypes
                            _sh = ctypes.CDLL([p_ for p_ in os.environ["LD_PRELOAD"].split(":") if "vshim" in p_][0])
                            _buf = ctypes.create_string_buffer(1 << 18)
                            for nm, sh_ in (("vllm main (legacy 0x0)", 0),
                                            ("vllm output copy", _AEw.VLLM_STREAMS["copy"].cuda_stream
                                             if _AEw.VLLM_STREAMS.get("copy") is not None else -2),
                                            ("all streams", -1)):
                                k_ = _sh.vshim_dump(ctypes.c_long(sh_), 400 if sh_ == 0 else 30 if sh_ != -1 else 40, _buf, len(_buf))
                                lines_ = _buf.raw[:k_].decode(errors="replace").splitlines()
                                if sh_ == 0:
                                    # boundary: the oldest PENDING op on stream 0 and the 12 entries before it
                                    pend = [i_ for i_, l_ in enumerate(lines_) if "PENDING" in l_]
                                    if pend:
                                        o_ = pend[-1]
                                        lines_ = lines_[max(0, o_ - 6):o_ + 13] + [f"  ({len(pend)} pending of {len(lines_)} listed)"]
                                    else:
                                        lines_ = lines_[:30] + ["  (none pending)"]
                                print(f"[stall-watchdog] cuda calls, {nm}, newest first:\n" + "\n".join(lines_),
                                      file=sys.stderr, flush=True)
                        except Exception as e:
                            print(f"[stall-watchdog] vshim dump failed: {e}", file=sys.stderr, flush=True)
                    print(f"[stall-watchdog] CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'default 8')}",
                          file=sys.stderr, flush=True)
                    for k in range(3):
                        t0 = time.perf_counter()
                        try:
                            with torch.cuda.stream(wd_stream):
                                pos_h[:4].copy_(_AEw.pos("cuda")[:4], non_blocking=True)
                                pos_h[4:5].copy_(_AEw.gate_dev()[:1].to(torch.int64), non_blocking=True)
                            ev = torch.cuda.Event()
                            ev.record(wd_stream)
                            ok = False
                            while time.perf_counter() - t0 < 3.0:
                                if ev.query():
                                    ok = True
                                    break
                                time.sleep(0.01)
                            print(f"[stall-watchdog] read {k}: {'ok' if ok else 'D2H BLOCKED >3s'} "
                                  f"tag {next((t for t, i in _AEw.TAG_IDS.items() if i == int(pos_h[3])), int(pos_h[3]))} pos {pos_h[:3].tolist()} op_in_fwd {int(pos_h[0] - pos_h[1])} site {(_AEw.MARK_SITES[int(pos_h[0] - pos_h[1])] if 0 <= int(pos_h[0] - pos_h[1]) < len(_AEw.MARK_SITES) else '?')} nsites {len(_AEw.MARK_SITES)} gate {int(pos_h[4])} "
                                  f"compute_stream {'idle' if streams['compute'].query() else 'BUSY'}",
                                  file=sys.stderr, flush=True)
                        except Exception as e:
                            print(f"[stall-watchdog] read {k} failed: {e}", file=sys.stderr, flush=True)
                        time.sleep(1.0)
        threading.Thread(target=_stall_wd, daemon=True).start()
        pw = os.environ.get("VARENA_PROF_WINDOW")
        if pw:
            # slack attribution: a CUPTI kernel trace of a window of the run (every stream of this process,
            # challenge kernels included), for schemes/memory_link_exhaustion/scripts/attrib_chase.py
            def _prof():
                t0_, dur_ = (float(x) for x in pw.split(","))
                time.sleep(t0_)
                try:
                    from torch.profiler import ProfilerActivity, profile
                    prof = profile(activities=[ProfilerActivity.CUDA])
                    prof.start()
                    time.sleep(dur_)
                    prof.stop()
                    out_ = os.environ.get("VARENA_PROF_OUT", "/tmp/varena_prof.json")
                    prof.export_chrome_trace(out_)
                    log("profile written", out_)
                except Exception as e:
                    log("profile failed:", e)
            threading.Thread(target=_prof, daemon=True, name="varena-prof").start()
        log("sealed:", json.dumps({k: v for k, v in self.info.items() if k != "w_entries"}))

    # ------------------------------------------------------------------ scheduler hooks
    def on_alloc(self, blocks):
        if self.sealed:
            self.block_log.add([b.block_id for b in blocks], "alloc")
            if getattr(self, "kvt", None) is not None:
                self._kvt_fresh.update(b.block_id for b in blocks)

    # ------------------------------------------------------------------ settled-KV block table
    def _kvt_init(self):
        """cfg kv_table: the block table (inference/kvtable.py) on the host, its band on the device for the chase
        kernel, (version, digest) next to it, and the flag that keeps a table switch and a chase apart."""
        from schemes.memory_link_exhaustion.inference import kvtable as KT
        # cfg kv_table 2 = tight mode: the band is only what steps in flight may write (per-position writers)
        # (3 = tight table + tagged in-flight reads: the chase reads the band and returns a tag per such read)
        self.kvt = KT.KVTable(self.nb, self.kv_shape[2], tight=int(self.cfg.get("kv_table", 1)) >= 2)
        self._kvt_fresh, self._kvt_freed = set(), {}
        n4 = -(-self.nb // 4) * 4
        self._kvt_host = torch.zeros(n4, dtype=torch.int16).pin_memory()        # band staging, padded to 8 bytes
        self._kvt_host[:] = torch.from_numpy(np.full(n4, 16 | (16 << 8), dtype=np.uint16).view(np.int16).copy())
        self._kvt_info_host = torch.zeros(2, dtype=torch.int64).pin_memory()    # version, digest
        self._kvt_dev = self._kvt_host.cuda()
        self._kvt_info_dev = self._kvt_info_host.cuda()
        self.tabupd = torch.zeros(1, dtype=torch.int32).pin_memory()
        self._kvt_stream = torch.cuda.Stream()
        self._kvt_stats = {"updates": 0, "apply_ms_max": 0.0, "wait_ms_max": 0.0, "masked_tok_max": 0, "masked_tok_sum": 0}
        self._kvt_publish()

    def _kvt_publish(self):
        """Stage the current table and switch the device copy: raise TABUPD, wait for any chase in flight (BUSY), run
        the apply kernel, drop TABUPD. The challenge server backs off while TABUPD is up (gpu/native.py serve)."""
        from schemes.memory_link_exhaustion.gpu.cuda import build as CB
        kt = self.kvt
        band = kt.band()
        self._kvt_host[:self.nb] = torch.from_numpy(band.view(np.int16).copy())
        dg = kt.digest()
        self._kvt_info_host[0] = int(kt.version)
        self._kvt_info_host[1] = dg - (1 << 64) if dg >= (1 << 63) else dg
        t0 = time.perf_counter()
        self.tabupd[0] = 1
        busy = getattr(self, "busy", None)
        while busy is not None and int(busy[0]) != 0 and time.perf_counter() - t0 < 0.5:
            pass
        t1 = time.perf_counter()
        with torch.cuda.stream(self._kvt_stream):
            CB.load().kv_tab_apply(self._kvt_host.data_ptr(), self._kvt_dev, self._kvt_info_host.data_ptr(), self._kvt_info_dev)
        self._kvt_stream.synchronize()
        self.tabupd[0] = 0
        t2 = time.perf_counter()
        st = self._kvt_stats
        mt = kt.masked_tokens()
        st["updates"] += 1
        st["wait_ms_max"] = max(st["wait_ms_max"], (t1 - t0) * 1e3)
        st["apply_ms_max"] = max(st["apply_ms_max"], (t2 - t1) * 1e3)
        st["masked_tok_max"] = max(st["masked_tok_max"], mt)
        st["masked_tok_sum"] += mt
        if st["updates"] % 200 == 1:      # a line every 200 schedule() calls: in-flight share, cost of the switch
            tok_mib = self.L * 4096 / 2**20           # one token position in every layer
            log(f"KVTAB updates {st['updates']} version {kt.version} masked now {mt * tok_mib:.0f} MiB "
                f"mean {st['masked_tok_sum'] / st['updates'] * tok_mib:.0f} MiB max {st['masked_tok_max'] * tok_mib:.0f} MiB | "
                f"switch: wait for chase max {st['wait_ms_max']:.3f} ms, apply max {st['apply_ms_max']:.3f} ms | kinds "
                f"{np.bincount(kt.kind, minlength=5).tolist()}")

    def on_scheduled(self, sched):
        """After schedule(): rebuild the table from the scheduler's state (settled counts `lag` calls back, blocks
        allocated or freed since the last call) and switch it in before the step this call scheduled can write."""
        if getattr(self, "kvt", None) is None or not self.sealed:
            return
        from baselines.inference.common import prompt_fp
        lag = int(os.environ.get("VARENA_SETTLE_LAG", "0"))
        blocks = sched._varena_blocks
        hist = self.__dict__.setdefault("_kvt_hist", deque(maxlen=lag + 1))
        now = {r.request_id: r.num_computed_tokens - r.num_in_flight_tokens for r in sched.running}
        hist.append(now)
        old = hist[0] if len(hist) > lag else {}
        live = {}
        for r in sched.running:
            bl = blocks.get(r.request_id)
            if bl is None:
                continue
            st = max(0, min(now[r.request_id], old.get(r.request_id, 0)))
            fpc = self.__dict__.setdefault("_kvt_fp", {})
            fp = fpc.get(r.request_id)
            if fp is None:
                f = prompt_fp(r.prompt_token_ids)
                fp = fpc[r.request_id] = (int(f[0]) << 32) | int(f[1])
            for i, x in enumerate(bl):
                live[x.block_id] = (fp, i, st)
        if self.kvt.tight:
            # band = [settled `lag` calls ago, computed now): num_computed_tokens already counts the step this call
            # scheduled. A freed block's owner wrote everything it had computed (frees wait for in-flight steps).
            comp = {r.request_id: int(r.num_computed_tokens) for r in sched.running}
            rid_of = {}
            for r in sched.running:
                for x in blocks.get(r.request_id) or ():
                    rid_of[x.block_id] = r.request_id
            # only the blocks at or after each request's previous settled edge can change (earlier ones are full
            # and already its own): keeps the per-call work proportional to the tokens in flight, not to all of KV
            prev = self.__dict__.setdefault("_kvt_prev", {})
            bs_ = self.kvt.bs
            live_t = {}
            for b, (fp, i, st) in live.items():
                rid = rid_of[b]
                if i >= prev.get(rid, 0) // bs_:
                    live_t[b] = (fp, i, st, max(st, comp[rid]))
            st_of = {rid_of[b]: st for b, (fp, i, st) in live.items()}
            self._kvt_prev = st_of
            freed_t = {b: ((int(o[0][0]) << 32 | int(o[0][1]), o[1], o[3] if len(o) > 3 else o[2]) if o is not None else None)
                       for b, o in self._kvt_freed.items()}
            self._kvt_fresh, self._kvt_freed = set(), {}
            self.kvt.update_tight(live_t, freed_t)
            self._kvt_publish()
            return
        fresh = [b for b in self._kvt_fresh if b not in live]
        freed = {b: ((int(o[0][0]) << 32 | int(o[0][1]), o[1], o[2]) if o is not None else None)
                 for b, o in self._kvt_freed.items() if b not in live}
        self._kvt_fresh, self._kvt_freed = set(), {}
        self.kvt.update(live, fresh, freed)
        self._kvt_publish()

    def on_free(self, pool, blocks):
        t0 = time.perf_counter()
        try:
            return self._on_free(pool, blocks)
        finally:
            dt = time.perf_counter() - t0
            self._onfree_t += dt
            self._onfree_n += 1
            self._onfree_max = max(self._onfree_max, dt)

    _onfree_t = 0.0
    _onfree_n = 0
    _onfree_max = 0.0

    def _stale_owners(self, ids):
        """Record the owner of each freed block from the newest snapshots (settled count `lag` steps back)."""
        from baselines.inference.common import prompt_fp
        want = set(ids)
        snaps = list(self.snaps)
        if not snaps:
            return
        lag = int(os.environ.get("VARENA_SETTLE_LAG", "0"))
        snap = snaps[-1][1]
        old = snaps[-1 - lag][1] if len(snaps) > lag else {}
        own = self.stale_owner
        for rid, (settled, req, bl) in snap.items():
            hit = [i for i, x in enumerate(bl) if x.block_id in want]
            if not hit:
                continue
            o = old.get(rid) if lag else (settled,)
            st_old = min(settled, o[0]) if o is not None else 0
            fp = prompt_fp(req.prompt_token_ids)
            for i in hit:
                own[bl[i].block_id] = (fp, i, st_old)
                want.discard(bl[i].block_id)
        for b in want:                      # owner not in the snapshot (e.g. aborted early): partial
            own[b] = None
        if getattr(self, "kvt", None) is not None:
            # (fp, block index, lagged settled count, tokens computed by the owner when it was freed)
            comp_at_free = {}
            for rid, (settled, req, bl) in snap.items():
                for x in bl:
                    comp_at_free[x.block_id] = int(req.num_computed_tokens)
            for b in ids:
                o = own.get(b)
                self._kvt_freed[b] = (o[0], o[1], o[2], comp_at_free.get(b, o[2])) if o is not None else None

    def _on_free(self, pool, blocks):
        """Blocks that just hit ref 0: keep them out of the free queue until refilled.
        Frees are deferred by vLLM (defer_block_free) until no in-flight step uses them,
        so the refill can start at once on its own stream."""
        ids = [b.block_id for b in blocks]
        self.block_log.add(ids, "free")
        if STALE_KV or self.cfg.get("stale_kv"):
            # stale-KV route: no refill. A freed block keeps its last owner's encoded KV; we remember the owner
            # (prompt fingerprint, block index, KV-settled token count) and claim the page under it, so the
            # verifier's KV recompute covers it like a live page (the partial last block stays excused).
            _t0 = time.perf_counter()
            self._stale_owners(ids)
            _dt = time.perf_counter() - _t0
            if _dt > 0.02 or len(ids) >= 256:     # bulk frees (wave ends) vs the noise dips
                print(f"{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d} [varena-prover] stale-free "
                      f"{len(ids)} blocks ({len(ids) * self.L * 64 // 1024} MiB) owners {_dt * 1e3:.1f} ms",
                      file=sys.stderr, flush=True)
            self.block_log.add(ids, "stale")
            pool.free_block_queue.append_n(blocks)
            self.stats["stale_freed"] = self.stats.get("stale_freed", 0) + len(blocks)
            return
        if self.cfg.get("debug_no_refill"):        # ablation: freed blocks go straight back
            pool.free_block_queue.append_n(blocks)
            self.stats["refill_skipped"] = self.stats.get("refill_skipped", 0) + len(blocks)
            return
        pages = (np.asarray(ids, dtype=np.int64)[None, :] +
                 (self.nW + np.arange(self.L, dtype=np.int64) * self.nb)[:, None]).ravel()
        host = torch.from_numpy(pages).pin_memory()
        with torch.cuda.stream(self.fill_stream):
            # the fill kernel reads the page list from pinned memory (zero-copy): an H2D copy of
            # it would queue on the copy engine behind the PCIe noise windows
            dev = host
            if not self.cfg.get("debug_refill_noop"):   # ablation: same holding/events, no fill kernel
                K.fill_pages(self.base_i32, self.page_map, dev, self.fkey)
            ev = torch.cuda.Event()
            ev.record()
        if REFILL_ORDERED:
            # Return the blocks at once and order the refill before vLLM's next step instead: the
            # compute stream waits for the fill (vLLM defers frees until no in-flight step uses the
            # blocks, so the fill can't race a reader, and a new owner's KV writes come after it).
            # Holding blocks until the fill was polled (~1 step) starved vLLM at full KV load:
            # 4000 ShareGPT requests ran at 4.4-4.6k tok/s vs 6.0k without refill.
            from schemes.memory_link_exhaustion.encodings import actenc2 as _AEr
            if _AEr.MARKERS and os.environ.get("VARENA_TAIL_MARKS", "1") == "1":
                _AEr.mark(torch.device("cuda", torch.cuda.current_device()), tag="refill:pre_wait")
            torch.cuda.current_stream().wait_event(ev)
            if _AEr.MARKERS and os.environ.get("VARENA_TAIL_MARKS", "1") == "1":
                _AEr.mark(torch.device("cuda", torch.cuda.current_device()), tag="refill:post_wait")
            # "pending" until the fill kernel has completed (a probe in between reads the old KV:
            # logging "refilled" at launch made such pages count as bad filler, the kfree false positive)
            self.block_log.add(ids, "pending")
            pool.free_block_queue.append_n(blocks)
            self.refills_ordered.append((ev, host, time.perf_counter(), ids))
            self.stats["refill_launches"] += 1
            self.stats["refill_blocks"] += len(blocks)
            return
        self.refills.append((pool, blocks, ev, host, dev, time.perf_counter()))
        self.held += len(blocks)
        self.stats["held_max"] = max(self.stats["held_max"], self.held)
        self.stats["refill_launches"] += 1
        self.stats["refill_blocks"] += len(blocks)

    deferred = []
    _refill_lat = []
    refills_ordered = collections.deque()

    def poll_refills(self):
        if self.deferred and not (getattr(self, 'busy', None) is not None and int(self.busy[0])):
            d, self.deferred = self.deferred, []
            for pool, blocks in d:
                self._on_free(pool, blocks)
        while self.refills_ordered and self.refills_ordered[0][0].query():
            _ev, _h, t_free, ids = self.refills_ordered.popleft()      # keeps the pinned page list alive
            self._refill_lat.append(time.perf_counter() - t_free)
            # filler from now on, unless vLLM already reallocated the block (then it is live)
            self.block_log.add([b for b in ids if self.block_log.last(b) == "pending"], "refilled")
        while self.refills and self.refills[0][2].query():
            pool, blocks, _ev, _h, _d, t_free = self.refills.popleft()
            self._refill_lat.append(time.perf_counter() - t_free)
            if len(self._refill_lat) > 20000:
                del self._refill_lat[:10000]
            self.block_log.add([b.block_id for b in blocks], "refilled")
            pool.free_block_queue.append_n(blocks)
            self.held -= len(blocks)

    _sched_t = 0.0
    _peak_hist = []
    _step_log = []

    def on_schedule(self, sched):
        _t0 = time.perf_counter()
        try:
            return self._on_schedule(sched)
        finally:
            self._sched_t += time.perf_counter() - _t0

    def _on_schedule(self, sched):
        self._last_sched_t = time.perf_counter()
        self.poll_refills()
        # the honest prover yields the GPU during a challenge: no new step is scheduled until the chase has been
        # answered (in-flight steps still run)
        if PAUSE_VLLM and getattr(self, 'busy', None) is not None and getattr(self, "mchase_kind", -1) >= 0:
            t0 = time.perf_counter()
            while int(self.busy[0]) == self.mchase_kind + 1 \
                    and time.perf_counter() - t0 < 0.2:
                time.sleep(2e-5)
        if not self.sealed:
            return
        t = time.perf_counter()
        blocks = sched._varena_blocks
        snap = {}
        for req in sched.running:
            bl = blocks.get(req.request_id)
            if bl is not None:
                snap[req.request_id] = (req.num_computed_tokens - req.num_in_flight_tokens, req, bl)
        self.snaps.append((t, snap))
        self.step_times.append((t, len(sched.running)))
        # per-step log for the GPU-time model: tokens computed since the previous schedule() call
        prev = getattr(self, "_step_prev", {})
        cur = {r.request_id: r.num_computed_tokens for r in sched.running}
        dtok = sum(max(0, n - prev.get(k, 0)) for k, n in cur.items())
        self._step_prev = cur
        # hidden-rl v2 gate (cheat experiment): vLLM is in a prefill-heavy phase (the last step computed
        # many tokens); the waiting queue is no signal under a long declared queue
        self.vllm_heavy = dtok >= HEAVY_TOK
        if len(self._step_log) < 400000:
            # allocator evidence (stall hunt): cumulative cudaMalloc retries / OOMs, reserved and peak-allocated bytes
            try:
                ms = torch.cuda.memory_stats()
                extra = (ms.get("num_alloc_retries", 0), ms.get("num_ooms", 0),
                         ms.get("reserved_bytes.all.current", 0) / 2**30, ms.get("allocated_bytes.all.peak", 0) / 2**30)
            except Exception:
                extra = (0, 0, 0.0, 0.0)
            self._step_log.append((t, dtok, len(cur)) + extra)
        # hang watchdog: running requests whose computed-token counts stop advancing for 20 s
        prog = tuple(sorted((r.request_id, r.num_computed_tokens) for r in sched.running))
        if prog != getattr(self, "_wd_prog", None):
            self._wd_prog, self._wd_t, self._wd_dumped = prog, t, False
        elif prog and t - self._wd_t > 20 and not self._wd_dumped:
            self._wd_dumped = True
            import faulthandler
            st = [(r.request_id, r.num_computed_tokens, r.num_tokens, getattr(r, "max_tokens", None),
                   str(getattr(r, "status", "")), getattr(r, "num_output_tokens", None)) for r in sched.running]
            log(f"[hang-watchdog] {len(st)} running requests made no progress for {t - self._wd_t:.0f} s: "
                f"{st[:8]}; waiting {len(sched.waiting)}; busy {int(self.busy[0]) if getattr(self, 'busy', None) is not None else None}; "
                f"deferred {len(self.deferred)} refills {len(self.refills)} ordered {len(self.refills_ordered)}")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        if PROF_KV and t - getattr(self, "_kv_log_t", 0.0) > 5.0:
            self._kv_log_t = t
            try:
                nfree = sched.kv_cache_manager.block_pool.get_num_free_blocks()
            except Exception:
                nfree = -1
            lat = sorted(self._refill_lat[-4000:]) or [0.0]
            ms = torch.cuda.memory_stats()
            log(f"[kv-prof] running {len(sched.running)} waiting {len(sched.waiting)} free {nfree} held {self.held} "
                f"refill lat ms p50 {lat[len(lat) // 2] * 1e3:.2f} p99 {lat[int(0.99 * (len(lat) - 1))] * 1e3:.2f} "
                f"max {lat[-1] * 1e3:.2f} | alloc retries {ms.get('num_alloc_retries', -1)} "
                f"device allocs {ms.get('num_device_alloc', -1)} frees {ms.get('num_device_free', -1)} "
                f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
                f"peak5s {torch.cuda.max_memory_allocated() / 2**30:.2f} free {torch.cuda.mem_get_info()[0] / 2**30:.2f} | on_free {self._onfree_n} calls "
                f"{self._onfree_t:.2f} s total, max {self._onfree_max * 1e3:.1f} ms | sched-hook "
                f"{self._sched_t:.2f} s")
            self._peak_hist.append(torch.cuda.max_memory_allocated())
            self._peak_res = max(getattr(self, "_peak_res", 0), torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            if MEMSNAP and len(sched.running) >= 100:
                self._memsnap(t)
        while self.claim_q:
            m = self.claim_q.popleft()
            try:
                self.conn.send(self._claims(m))
            except Exception:
                log("claim failed", traceback.format_exc())

    def _memsnap(self, t):
        """VARENA_MEMSNAP=path: two torch memory snapshots during generation (30 s apart): segments by pool
        (CUDA-graph private pools vs the regular pool), address stability of graph-pool segments, the
        non-torch remainder (driver used - torch reserved) and the CUDA stack-size limit."""
        snaps = getattr(self, "_snaps_mem", [])
        if len(snaps) >= 2 or (snaps and t - snaps[-1]["t"] < 30):
            return
        import ctypes
        import subprocess
        segs = torch.cuda.memory_snapshot()
        pools = {}
        for sg in segs:
            pid = str(tuple(sg.get("segment_pool_id", (0, 0))))
            d = pools.setdefault(pid, {"segments": 0, "bytes": 0, "allocated": 0, "addrs": []})
            d["segments"] += 1
            d["bytes"] += int(sg["total_size"])
            d["allocated"] += int(sg.get("allocated_size", 0))
            d["addrs"].append([int(sg["address"]), int(sg["total_size"])])
        free, total = torch.cuda.mem_get_info()
        try:
            smi = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()[0]) << 20
        except Exception:
            smi = -1
        stack = ctypes.c_size_t(0)
        try:
            ctypes.CDLL("libcudart.so").cudaDeviceGetLimit(ctypes.byref(stack), 0)   # cudaLimitStackSize
        except Exception:
            pass
        snaps.append({"t": t, "running": None, "pools": pools, "reserved": torch.cuda.memory_reserved(),
                      "allocated": torch.cuda.memory_allocated(), "driver_used": total - free, "smi_used": smi,
                      "non_torch": (total - free) - torch.cuda.memory_reserved(), "stack_limit_bytes": stack.value,
                      "module_loading": os.environ.get("CUDA_MODULE_LOADING")})
        self._snaps_mem = snaps
        if len(snaps) == 2:
            a, b = snaps
            stab = {}
            for pid, d in a["pools"].items():
                if pid == "(0, 0)":
                    continue
                stab[pid] = sorted(map(tuple, d["addrs"])) == sorted(map(tuple, b["pools"].get(pid, {}).get("addrs", [])))
            out = {"snapshots": [{k: (v if k != "pools" else {p: {kk: vv for kk, vv in d.items() if kk != "addrs"}
                                                             for p, d in v.items()}) for k, v in sn.items()}
                                 for sn in snaps],
                   "graph_pool_addresses_stable": stab,
                   "graph_pool_ranges": {p: d["addrs"] for p, d in a["pools"].items() if p != "(0, 0)"}}
            json.dump(out, open(MEMSNAP, "w"))
            log(f"[memsnap] wrote {MEMSNAP}: {len(a['pools'])} pools, non-torch {a['non_torch'] / 2**30:.2f} GiB")

    def _claims(self, m):
        def _prompt_fp(req, _c={}):
            k = req.request_id
            if k not in _c:
                if len(_c) > 100000:
                    _c.clear()
                from baselines.inference.common import prompt_fp
                _c[k] = prompt_fp(req.prompt_token_ids)
            return _c[k]
        """Classify challenged K pages. Live settled pages are claimed as
        (prompt fingerprint, block index): the verifier, being the client, knows
        every request's prompt and output tokens, so it rebuilds the prefix."""
        t0, t1 = m["t0"], m["t1"]
        snap_t, snap, old = None, None, None
        snaps = list(self.snaps)
        lag = int(os.environ.get("VARENA_SETTLE_LAG", "0"))
        for j in range(len(snaps) - 1, -1, -1):
            if snaps[j][0] < t0:
                snap_t, snap = snaps[j]
                old = snaps[j - lag][1] if j >= lag else {}
                break
        where = {}
        if snap is not None:
            for rid, (settled, req, bl) in snap.items():
                # settled count from `lag` steps earlier: with async scheduling, the tokens of the
                # step(s) still in flight may count as computed before their KV is written
                o = old.get(rid) if lag else (settled,)
                st_old = min(settled, o[0]) if o is not None else 0
                for i, x in enumerate(bl):
                    where[x.block_id] = (req, i, st_old)
        out = []
        bs = self.kv_shape[2]
        cnt = self.stats["claims"]
        for p in m["kpages"]:
            l, b = divmod(p - self.nW, self.nb)
            # events since the snapshot used count as a transition: a block freed and reallocated
            # after the snapshot (e.g. inside schedule() during the burst pause) would otherwise be
            # claimed under its old request's fingerprint (honest KV-recompute false positive)
            st = self.block_log.classify(b, t0 if snap_t is None else min(t0, snap_t), t1)
            c = [p, st]
            if st == "stale":
                o = self.stale_owner.get(b)
                if o is not None and o[2] >= (o[1] + 1) * bs:
                    c = [p, "settled", o[0], o[1]]
                else:
                    c[1] = "stale_partial"
            elif st == "live":
                w = where.get(b)
                if w is None:
                    c[1] = "unresolved"
                else:
                    req, i, settled = w
                    need = (i + 1) * bs
                    if settled >= need:
                        c = [p, "settled", _prompt_fp(req), i]
                    else:
                        c[1] = "unsettled"
            cnt[c[1]] = cnt.get(c[1], 0) + 1
            out.append(c)
        return {"op": "claims", "id": m["id"], "claims": out, "snap_age_ms":
                None if snap_t is None else (t0 - snap_t) * 1e3}

    # ------------------------------------------------------------------ threads
    def _on_window(self, i, off, d0, d1, t):
        with self.lock:
            self.win_buf.append([i, off, d0, d1])

    win_buf = []

    def _flusher(self):
        while True:
            time.sleep(0.005)
            with self.lock:
                buf, self.win_buf = self.win_buf, []
            if buf:
                self.conn.send({"op": "pcie", "w": buf})

    def _reader(self):
        while True:
            m = self.conn.recv()
            if m is None:
                return
            op = m["op"]
            if op in ("spot", "chase_debug"):
                m["t_recv"] = time.perf_counter()
                self.chal_q.put(m)
            elif op == "corrupt":
                self.chal_q.put(m)
            elif op == "stop":
                self.chal_q.put(m)

    def _challenger(self):
        torch.cuda.set_device(0)
        while True:
            m = self.chal_q.get()
            op = m["op"]
            if op == "stop":
                self._bye()
                continue
            if op == "corrupt":
                self._corrupt(m)
                continue
            if op == "chase_debug":
                self._chase_debug(m)
                continue
            # a spot check: digests of the requested pages (selftest)
            t0 = time.perf_counter()
            with torch.cuda.stream(self.chal_stream):
                pages = torch.tensor(m["pages"], dtype=torch.int64, device="cuda")
                d0, d1 = K.digest_pages(self.base_i32, self.page_map, pages, m["nonce"])
                pages, d0, d1 = pages.cpu(), d0.cpu(), d1.cpu()
            t1 = time.perf_counter()
            self.stats["challenges"] += 1
            self.stats["chal_ms"].append((t1 - t0) * 1e3)
            pl = pages.flatten().tolist()
            self.conn.send({"op": "response", "id": m["id"], "kind": "spot", "pages": pl, "win": -1,
                            "d0": d0.flatten().tolist(), "d1": d1.flatten().tolist(),
                            "queue_ms": (t0 - m["t_recv"]) * 1e3, "kernel_ms": (t1 - t0) * 1e3})
            kp = sorted({p for p in pl if self.nW <= p < self.nW + self.nK})
            if kp:
                tc = t0
                self.claim_q.append({"id": m["id"], "t0": tc, "t1": t1, "kpages": kp})

    def _start_native(self):
        from schemes.memory_link_exhaustion.gpu import native
        cfg = self.cfg
        from schemes.memory_link_exhaustion.encodings import actenc2 as AE
        gating = (bool(AE.POOL_BUFS) or os.environ.get("VARENA_MM_GATE") == "1") and AE.GATE_ON
        if gating:
            AE.gate_set(0)                      # compile before any capture
            AE.gate_set(0, torch.zeros(1, dtype=torch.int64).pin_memory())
            if AE.DRAIN:
                _t = torch.zeros(1, dtype=torch.int64).pin_memory()
                with torch.inference_mode():
                    AE.gate_dev()[1].fill_(1)
                    AE.drain_wait(_t, 0.01)             # compile before any capture
                    AE.gate_dev()[1].fill_(0)

        def _g(cls):
            """yield on challenge: the challenge graph raises the device gate first and drops it last, so the
            compute stream's position markers (actenc2.mark) wait while it runs"""
            if not gating:
                return cls

            if cls is native.ChaseGraph:
                class GatedChase(cls):
                    # two extra int64 slots at the end of the response: GPU start / end time of the graph
                    def __init__(self_, launch, K, stream, extra=0):
                        def l2(o, n):
                            if AE.DRAIN:
                                AE.gate_set(1)
                                AE.drain_wait(o[K:K + 1])      # chase start = after the drain
                            else:
                                AE.gate_set(1, o[K:K + 1])
                            launch(o, n)
                            AE.gate_set(0, o[K + 1:K + 2])
                        super().__init__(l2, K + 2, stream, extra=extra)   # extra words follow the two time slots
                return GatedChase
            return cls
        ChaseG = _g(native.ChaseGraph)
        gs = []            # the challenge graphs, by index: the mixed chase is the only one
        mchase_idx = -1
        if cfg.get("mchase_P"):
            # mixed chase: one kind over W, KV, filler and the activation pool (union granule table)
            from schemes.memory_link_exhaustion.gpu.cuda import build as CB
            ext = CB.load()
            nF_ = self.lend_first - self.nW - self.nK
            glm = AE.pool_granules() if AE.POOL_BUFS else []
            # --ring-chase-guard: noise-ring granules in the chase, head-relative (ref.ring_chase_granules)
            rtag = bool(cfg.get("ring_tag")) and int(cfg.get("kv_table", 0) or 0) == 3
            nrg = ref.ring_chase_granules(self.streamer.n_ring, int(cfg.get("ring_chase_guard", 0)), tagged=rtag) \
                if self.streamer else 0
            if nrg:
                assert self.streamer.Ww * 4 == 1 << 25, "ring chase: 32 MiB noise windows only"
            gm = ref.ch_granules_mix(self.nW, self.nK, nF_, len(glm), nrg)
            pm_h = self.page_map.cpu().numpy()
            for g in sorted(set(x for x in gm if x < ref.CH_PA0)):
                if pm_h[g + ref.CH_GRAN_PAGES - 1] - pm_h[g] != (ref.CH_GRAN_PAGES - 1) * (PAGE // 4):
                    raise RuntimeError(f"mixed chase: granule at page {g} is not contiguous")
            gam = torch.tensor([(x - ref.CH_RA0) // ref.CH_GRAN_PAGES - ref.CH_RING_TAG if x >= ref.CH_RA0 else
                                glm[(x - ref.CH_PA0) // ref.CH_GRAN_PAGES] if x >= ref.CH_PA0 else
                                self.base_i32.data_ptr() + int(pm_h[x]) * 4 for x in gm],
                               dtype=torch.int64, device="cuda")
            tbm = len(gm) // ref.CH_GRAN_PER_BLOCK
            nbm = min(int(cfg.get("chase_blocks", 11)), tbm)
            Km, Pm = int(cfg.get("chase_K", 132 * 32)), int(cfg["mchase_P"])
            self._mchase_dbg = (gm, tbm, nbm, Pm)
            self._mchase_ga = gam
            if cfg.get("kv_table"):
                # settled-KV rule: the chase looks up the block table and does not read in-flight positions; it
                # appends the table's (version, digest) after the K chain states
                self._kvt_init()
                gkv = torch.tensor([x - self.nW if self.nW <= x < self.nW + self.nK else -1 for x in gm],
                                   dtype=torch.int32, device="cuda")
                rb_ = self.streamer.ring.data_ptr() if nrg else 0
                nr_ = self.streamer.n_ring if nrg else 1
                kd_, ki_, nbk_ = self._kvt_dev, self._kvt_info_dev.data_ptr(), self.nb
                # keep the tensors the captured graph points at alive: the graph holds raw device pointers, and the
                # launch closure (the only other reference to gkv) is dropped after capture. Without this the
                # granule -> KV page map was freed and reused, the kernel masked at garbage positions, and no
                # chain of an honest run replayed (found with --chase-debug, 14 Nov)
                self._mchase_gkv = gkv
                self._mchase_dbg2 = (gam, gkv, Km)
                log(f"mixed chase: settled-KV block table over {self.nb} blocks, ring granules {nrg}")
                # cfg kv_table 3: tagged in-flight reads (info_off bit 8); the tags (ref.CH_TAG_SLOTS uint32 per chain)
                # follow the (version, digest) pair in the answer
                tagm = int(cfg.get("kv_table", 1)) == 3
                # ... and with cfg ring_tag every ring slot is read, the guard's slots tagged (info_off bits 16..23)
                ioff_ = (2 if gating else 0) | (0x100 if tagm else 0) | \
                    ((int(cfg.get("ring_chase_guard", 0)) << 16) if (rtag and tagm and nrg) else 0)
                gs.append(ChaseG(lambda o, nn: ext.chase_honest_full(gam, tbm, nbm, nn.data_ptr(), Km, Pm, o, rb_, nr_,
                                                                     gkv, kd_, nbk_, ki_, ioff_),
                                 Km, self.chal_stream, extra=2 + (Km * ref.CH_TAG_SLOTS // 2 if tagm else 0)))
                if tagm:
                    log(f"mixed chase: tagged in-flight reads ({ref.CH_TAG_SLOTS - 1} tags per chain)")
            else:
                # the ring is read only through the tagged settled-KV path above (kv_table 3); there is no
                # untagged ring chase any more
                assert not nrg, "ring chase needs --kv-table 3 (tagged ring reads); the untagged path was removed"
                gs.append(ChaseG(lambda o, nn: ext.chase_honest(gam, tbm, nbm, nn.data_ptr(), Km, Pm, o),
                                 Km, self.chal_stream))
            mchase_idx = len(gs) - 1
        self.mchase_kind = mchase_idx
        self.native_graphs = gs
        last = self.streamer.last_arr if self.streamer else \
            torch.full((1,), -1, dtype=torch.int64).pin_memory()
        self._native_last = last
        self.busy = None
        if PAUSE_VLLM:
            # busy flag while a challenge is in flight (read by the vLLM pause hook)
            self.busy = torch.zeros(1, dtype=torch.int32).pin_memory()
        native.start_server(cfg["native_sock"], gs, last, busy=self.busy, tabupd=getattr(self, "tabupd", None))
        log("native challenge server on", cfg["native_sock"])

    def _chase_debug(self, m):
        """Debug only (verifier --chase-debug): walk one chain of the mixed chase on the CPU straight from device
        memory, with the block table as it is now, and return the final state and every probe (page, word offset,
        crc32 of the 32 words used). For a quiet moment (no step in flight) and without ring granules."""
        import zlib
        try:
            gm, tbm, nbm, Pm = self._mchase_dbg
            pm = self.page_map.cpu().numpy()
            nonce, c = int(m["nonce"]), int(m["chain"])
            kt = getattr(self, "kvt", None)
            probes = []
            from schemes.memory_link_exhaustion.gpu.cuda import build as CB
            ext = CB.load()
            base_ptr = self.base_i32.data_ptr()
            buf = torch.zeros(32, dtype=torch.int32, device="cuda")

            def words(page, wo):
                if page >= ref.CH_PA0:
                    raise RuntimeError("chase_debug: pool / ring granules are not supported")
                if kt is not None and self.nW <= page < self.nW + self.nK and \
                        int(kt.lo[(page - self.nW) % self.nb]) <= ref.ch_kv_token(wo):
                    w = ref.ch_mask_words(nonce, page - self.nW, wo)
                else:
                    # the chase's own addressing: page map offset from the arena base pointer (K and filler pages
                    # are separate allocations, outside the base tensor's bounds, so no tensor slicing)
                    ext.peek(base_ptr + (int(pm[page]) + wo) * 4, buf)
                    w = buf.cpu().numpy().view(np.uint32).copy()
                if w.size != 32:
                    raise RuntimeError(f"chase_debug: page {page} word {wo}: read {w.size} words (page map {int(pm[page])})")
                probes.append([int(page), int(wo), zlib.crc32(np.ascontiguousarray(w).tobytes())])
                return w
            st = ref.ch_chain(nonce, c, Pm, tbm, nbm, gm, words)
            extra = {}
            if kt is not None:
                # is the table on the device the one on the host? (the kernel reads the device copy)
                n4 = self._kvt_dev.numel()
                dbuf = torch.zeros(n4 // 2, dtype=torch.int32, device="cuda")
                ext.peek(self._kvt_dev.data_ptr(), dbuf)
                dev = dbuf.cpu().numpy().view(np.uint16)[:self.nb]
                host = kt.band()
                ne = np.nonzero(dev != host)[0]
                extra = {"dev_table_equals_host": bool(ne.size == 0), "dev_table_diff": int(ne.size),
                         "dev_table_diff_examples": [[int(b), int(dev[b]), int(host[b])] for b in ne[:6]],
                         "dev_info": [int(x) for x in self._kvt_info_dev.cpu().numpy().view(np.uint64)],
                         "host_info": [int(kt.version), int(kt.digest())]}
                # the same walk under the device table
                lo_dev = (dev & 0xff).astype(np.int64)
                hi_dev = (dev >> 8).astype(np.int64)

                def words_dev(page, wo):
                    if self.nW <= page < self.nW + self.nK:
                        b = (page - self.nW) % self.nb
                        tk = ref.ch_kv_token(wo)
                        if lo_dev[b] <= tk < hi_dev[b]:
                            return ref.ch_mask_words(nonce, page - self.nW, wo)
                    ext.peek(base_ptr + (int(pm[page]) + wo) * 4, buf)
                    return buf.cpu().numpy().view(np.uint32).copy()
                extra["state_dev_table"] = str(ref.ch_chain(nonce, c, Pm, tbm, nbm, gm, words_dev))
                extra["mix"] = [int(ref.CH_MIX), int(ref.CH_WIDE), os.environ.get("VARENA_CHASE_MIX"), int(Pm), int(tbm), int(nbm), len(gm)]
                try:
                    # the kernel called directly (no graph, no challenge server), with and without the table, against
                    # the reference walk with and without the mask
                    gam_, gkv_, Km_ = self._mchase_dbg2
                    nn2 = torch.tensor([nonce - (1 << 32) if nonce >= (1 << 31) else nonce, 0], dtype=torch.int32).pin_memory()
                    o2 = torch.zeros(Km_ + 4, dtype=torch.int64).pin_memory()
                    o3 = torch.zeros(Km_ + 4, dtype=torch.int64).pin_memory()
                    with torch.cuda.stream(self.chal_stream):
                        ext.chase_honest_full(gam_, tbm, nbm, nn2.data_ptr(), Km_, Pm, o2, 0, 1, gkv_, self._kvt_dev,
                                              self.nb, self._kvt_info_dev.data_ptr(), 2)
                        ext.chase_honest(gam_, tbm, nbm, nn2.data_ptr(), Km_, Pm, o3)
                    self.chal_stream.synchronize()

                    def words_plain(page, wo):
                        ext.peek(base_ptr + (int(pm[page]) + wo) * 4, buf)
                        return buf.cpu().numpy().view(np.uint32).copy()
                    u64 = lambda x: str(int(x) & 0xFFFFFFFFFFFFFFFF)
                    extra["direct_full"] = u64(o2[c])
                    extra["direct_plain"] = u64(o3[c])
                    extra["walk_plain"] = str(ref.ch_chain(nonce, c, Pm, tbm, nbm, gm, words_plain))
                    extra["direct_full_neighbours"] = [u64(o2[max(0, c - 1)]), u64(o2[min(Km_ - 1, c + 1)])]
                except Exception as e:
                    extra["direct_error"] = f"{type(e).__name__}: {e}"
            self.conn.send({"op": "chase_debug", "id": m["id"], "state": str(st), "probes": probes,
                            "kvt_version": int(kt.version) if kt is not None else -1, **extra})
        except Exception as e:
            self.conn.send({"op": "chase_debug", "id": m["id"], "error": f"{type(e).__name__}: {e}"})

    def _corrupt(self, m):
        """Selftest: flip one byte of page p (W, K or F)."""
        p = int(m["page"])
        woff = int(self.page_map[p])
        with torch.cuda.stream(self.chal_stream):
            K.xor_word(self.base_i32, woff + int(m.get("word", 1000)), 0x5A)
        self.chal_stream.synchronize()
        self.corrupt_log.append(p)
        self.conn.send({"op": "corrupted", "page": p})

    def _bye(self):
        st = dict(self.stats)
        ms = sorted(st.pop("chal_ms"))
        st["chal_ms_p50"] = ms[len(ms) // 2] if ms else None
        st["chal_ms_max"] = ms[-1] if ms else None
        st["pcie_rate_self"] = self.streamer.rate(5.0) if self.streamer else None
        try:
            # outside-arena memory by owner (for the allowance study)
            free, cap = torch.cuda.mem_get_info()
            used = cap - free
            arena = self.npages * PAGE
            res = torch.cuda.memory_reserved()
            ms = torch.cuda.memory_stats()
            st["outside_breakdown_gib"] = {
                "device_used": used / 2**30, "arena": arena / 2**30,
                "torch_reserved": res / 2**30,
                "torch_reserved_outside_arena": (res - arena) / 2**30,
                "non_torch (context, cuBLAS/NCCL workspaces, graph pools outside torch)": (used - res) / 2**30,
                "torch_alloc_now_outside_arena": (torch.cuda.memory_allocated() - arena) / 2**30,
                "torch_peak_alloc_per_5s_outside_arena_max": (max(self._peak_hist) - arena) / 2**30 if self._peak_hist else None,
                "torch_peak_alloc_per_5s_outside_arena_p50": (sorted(self._peak_hist)[len(self._peak_hist) // 2] - arena) / 2**30
                if self._peak_hist else None,
                "allocator_inactive_split": ms.get("inactive_split_bytes.all.current", 0) / 2**30,
                "segments": ms.get("segment.all.current"),
            }
            snap = os.environ.get("VARENA_MEMSNAP")
            if snap:
                torch.cuda.memory._dump_snapshot(snap)
                log("memory snapshot written", snap)
        except Exception as e:
            st["outside_breakdown_gib"] = {"error": repr(e)}
        try:
            import numpy as _np
            path = os.environ.get("VARENA_STEPLOG")
            if path and self._step_log:
                _np.save(path, _np.asarray(self._step_log, dtype=_np.float64))
                st["steplog"] = path
        except Exception as e:
            st["steplog_error"] = repr(e)
        if getattr(self, "hidden", None) is not None:
            self.hidden.running = False
            time.sleep(0.5)
            st["hidden_rl"] = self.hidden.summary()
        st["max_reserved"] = torch.cuda.max_memory_reserved()
        try:
            st["pad_study"] = self._pad_study()
        except Exception as e:
            st["pad_study"] = {"error": repr(e)}
        self.conn.send({"op": "bye", "stats": st})

    # ------------------------------------------------------------------ pad study (arena sized from the measured peak)
    def _mem_track_start(self):
        """Start of serving: reset torch's peak counters (model load and warm-up peaks are not serving needs) and
        optionally sample the driver's free memory (VARENA_MEMWATCH_S seconds; 0 = off, the bound below suffices)."""
        torch.cuda.reset_peak_memory_stats()
        self._peak_res = torch.cuda.memory_reserved()
        free, cap = torch.cuda.mem_get_info()
        self._min_free = free
        per = float(os.environ.get("VARENA_MEMWATCH_S", "0") or 0)
        if per > 0:
            def _watch():
                while True:
                    time.sleep(per)
                    try:
                        self._min_free = min(self._min_free, torch.cuda.mem_get_info()[0])
                    except Exception:
                        return
            threading.Thread(target=_watch, daemon=True, name="varena-memwatch").start()

    def _pad_study(self):
        """How much of the pad the honest run used (verifier/pad.py); the calibration sizes the next arena from it."""
        from schemes.memory_link_exhaustion.verifier.pad import pad_study
        free, cap = torch.cuda.mem_get_info()
        res = torch.cuda.memory_reserved()
        peak = max(getattr(self, "_peak_res", 0), torch.cuda.max_memory_reserved())
        sampled = float(os.environ.get("VARENA_MEMWATCH_S", "0") or 0) > 0
        out = pad_study(self._padF, {"free": free, "cap": cap, "reserved": res}, peak,
                        self._min_free if sampled else None, self.npages * PAGE)
        out["prereserve_mib"] = self.stats.get("prereserve_mib")
        try:
            from schemes.memory_link_exhaustion.encodings import actenc2 as _AEp
            out["varena_scratch_mib"] = round(_AEp.scratch_bytes() / 2**20, 1)
        except Exception:
            pass
        return out


def get():
    global STATE
    if STATE is None:
        STATE = ArenaState()
    return STATE
