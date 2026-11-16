"""vLLM general plugin (entry point `vllm.general_plugins`, loaded in every process).

  VARENA_TRITON_GEMM=1   batch-invariant mode also routes aten::mm & co. through
                         vLLM's Triton persistent matmul on Hopper/Blackwell
                         (baseline inference and inference both set it)
  VARENA=1               inference: encoded weights/KV, filler, PCIe noise,
                         challenges; talks to the verifier at VARENA_SOCK
"""

from __future__ import annotations

import os


def _reserve_sms():
    """VARENA_RESERVE_SMS=n: vLLM's persistent matmul runs on NUM_SMS - n CTAs, so the
    single-CTA challenge kernel finds a free SM at once. Tile results don't depend on
    which CTA computes them, so outputs stay bit-identical. Set in baseline too."""
    n = int(os.environ.get("VARENA_RESERVE_SMS", "0") or 0)
    if n <= 0:
        return
    import vllm.model_executor.determinism.batch_invariant as bi
    orig = bi.num_compute_units
    if getattr(orig, "_varena", False):
        return

    def ncu(idx=None):
        return max(1, orig(idx) - n)

    ncu._varena = True
    bi.num_compute_units = ncu


def _no_uva():
    """VARENA_NO_UVA=1 (default): vLLM's v2 GPU model runner reads sampling parameters, staged writes and
    input-prep buffers straight from pinned host memory (UVA) inside its kernels. Under the PCIe noise the
    link is saturated host-to-device and those SM-issued host reads starve for seconds (the engine stalls:
    compute stream frozen between the logits processor and the sampler while copy-engine transfers and
    other kernels run normally). Its non-UVA buffers stage the same values with an explicit async H2D copy
    instead (same values, so outputs are unchanged). Set in baseline too."""
    if os.environ.get("VARENA_NO_UVA", "1") != "1":
        return
    from vllm.utils import platform_utils as pu

    def no_uva() -> bool:
        return False
    pu.is_uva_available = no_uva
    try:
        import torch
        from vllm.v1.worker.gpu import buffer_utils as bu
    except Exception:
        return
    bu.is_uva_available = no_uva
    if getattr(bu.NonUvaBuffer, "_varena", False):
        return

    def init(self, size, dtype):
        # pinned host staging (vLLM's fallback uses pageable memory, i.e. a synchronous staged copy);
        # buffer lifetime is the same as with UVA (the pool rotates buffers across in-flight steps)
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu").pin_memory()
        self.np = self.cpu.numpy()
        self._uva = torch.zeros(size, dtype=dtype, device="cuda")
    bu.NonUvaBuffer.__init__ = init
    bu.NonUvaBuffer._varena = True
    if os.environ.get("VARENA_SINGLE_QUEUE") == "1" and os.environ.get("VARENA") == "1":
        # single H2D queue: the input-buffer copy is issued by the noise streamer between noise windows; this stream
        # waits on its completion flag on the GPU (gpu/native.py sq_submit). Falls back to the plain copy when the
        # streamer is not running or a CUDA graph is being captured.
        orig_uva = bu.NonUvaBuffer.uva
        _nat = [None]

        def uva(self, n=None):
            if torch.cuda.is_current_stream_capturing():
                return orig_uva(self, n)
            if _nat[0] is None:
                from schemes.memory_link_exhaustion.gpu import native as _N
                _nat[0] = _N.mod()
            dst = self._uva if n is None else self._uva[:n]
            src = self.cpu if n is None else self.cpu[:n]
            if _nat[0].sq_submit(dst.data_ptr(), src.data_ptr(), dst.numel() * dst.element_size(),
                                 torch.cuda.current_stream().cuda_stream) < 0:
                return orig_uva(self, n)
            return dst
        bu.NonUvaBuffer.uva = uva


def _main_stream():
    """VARENA_MAIN_STREAM=1 (default): vLLM's v2 model runner uses the current stream at construction as its
    main stream, i.e. the legacy default stream. Under the PCIe noise, small pinned H2D input copies on the
    legacy stream sat for ~8 s at a time (engine stalls) while the same copies on another stream took
    0.2 ms. Make a dedicated non-blocking stream current in the worker thread before the runner is built,
    so every step (input copies, graph replays, eager kernels) runs on it. Set in baseline too."""
    if os.environ.get("VARENA_MAIN_STREAM", "1") != "1":
        return
    try:
        from vllm.v1.worker.gpu import model_runner as mr
    except Exception:
        return
    cls = getattr(mr, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_varena_stream", False):
        return
    init = cls.__init__

    def __init__(self, *a, **k):
        import torch
        torch.cuda.set_stream(torch.cuda.Stream(priority=0))
        init(self, *a, **k)
    cls.__init__ = __init__
    cls._varena_stream = True


def _runahead():
    """VARENA_RUNAHEAD=1 (noise-dip fix, default off): after every model step that is not
    decode-only (prefill or mixed tokens, which vLLM runs on the piecewise / eager path, many small launches), wait
    for the step's GPU work, so the CPU never fills the launch queue and blocks inside a launch while holding the
    CUDA driver lock (that block also stalled the native noise streamer's copies and reports for ~0.4-0.7 s once a
    run: a stall dump). Decode-only steps (full CUDA graphs) are untouched."""
    if os.environ.get("VARENA_RUNAHEAD", "0") != "1":
        return
    try:
        from vllm.v1.worker.gpu import model_runner as mr
    except Exception:
        return
    cls = getattr(mr, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_varena_runahead", False):
        return
    ex = cls.execute_model
    cnt = {"sync": 0, "steps": 0}

    def execute_model(self, scheduler_output, *a, **k):
        out = ex(self, scheduler_output, *a, **k)
        cnt["steps"] += 1
        try:
            tot = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
            nreq = len(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
        except Exception:
            tot, nreq = 0, 0
        if tot > nreq > 0 or (tot and not nreq):
            import torch
            torch.cuda.current_stream().synchronize()
            cnt["sync"] += 1
            if cnt["sync"] % 50 == 1:
                import sys, time
                print(f"{time.strftime('%H:%M:%S')} [varena-runahead] synced {cnt['sync']} of {cnt['steps']} steps "
                      f"(this one {tot} tokens / {nreq} requests)", file=sys.stderr, flush=True)
        return out
    cls.execute_model = execute_model
    cls._varena_runahead = True


def _launch_log():
    """VARENA_LAUNCH_LOG=1 (honest noise-dip diagnosis): log with timestamps
    - every Triton kernel specialisation compiled or loaded from the disk cache in this process (name, constexprs,
      warmup flag, seconds, launching thread),
    - every Triton launch whose run() takes > VARENA_LAUNCH_LOG_MS (20) ms of wall time in the launching thread
      (first launch of a specialisation loads its module and holds the driver: the captured stall),
    - every engine step > VARENA_STEP_LOG_MS (100) ms (scheduled tokens and requests), and the step context of
      each line above."""
    if os.environ.get("VARENA_LAUNCH_LOG", "0") != "1":
        return
    import sys
    import threading
    import time
    lim = float(os.environ.get("VARENA_LAUNCH_LOG_MS", "20")) / 1e3
    slim = float(os.environ.get("VARENA_STEP_LOG_MS", "100")) / 1e3
    ctx = {"step": -1, "tok": 0, "nreq": 0}

    def _p(msg):
        t = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime(t))}.{int(t % 1 * 1000):03d} [launch-log] {msg} "
              f"| step {ctx['step']} tok {ctx['tok']} req {ctx['nreq']} thread {threading.current_thread().name}",
              file=sys.stderr, flush=True)

    def _shapes(args):
        out = []
        for a in args[:8]:
            sh = getattr(a, "shape", None)
            out.append("x".join(map(str, sh)) if sh is not None else (repr(a)[:16] if isinstance(a, (int, float)) else "-"))
        return ",".join(out)
    try:
        from triton.runtime import jit as J
        JF = J.JITFunction
        if not getattr(JF, "_varena_ll", False):
            run0, comp0 = JF.run, JF._do_compile

            def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
                t0 = time.perf_counter()
                k = comp0(self, key, signature, device, constexprs, options, attrs, warmup)
                ce = {getattr(n, "name", n) if not isinstance(n, tuple) else n[-1]: v for n, v in
                      (constexprs.items() if hasattr(constexprs, "items") else [])}
                _p(f"COMPILE {getattr(self, '_fn_name', self)} {time.perf_counter() - t0:.3f}s warmup={warmup} "
                   f"constexprs={str(ce)[:300]}")
                return k

            def run(self, *args, grid, warmup, **kwargs):
                t0 = time.perf_counter()
                k = run0(self, *args, grid=grid, warmup=warmup, **kwargs)
                dt = time.perf_counter() - t0
                if dt > lim:
                    _p(f"SLOW_LAUNCH {getattr(self, '_fn_name', self)} {dt * 1e3:.1f}ms args {_shapes(args)} "
                       f"kw {str({kk: v for kk, v in kwargs.items() if isinstance(v, (int, bool))})[:200]}")
                return k
            JF.run, JF._do_compile, JF._varena_ll = run, _do_compile, True
    except Exception as e:                       # never break serving for a diagnostic
        print(f"[launch-log] triton hook failed: {e}", file=sys.stderr, flush=True)
    _runner_classes = []
    _alloc_log = os.environ.get("VARENA_ALLOC_LOG", "0") == "1"
    _alloc_prev = []
    try:
        import importlib
        hooked = []
        # both runner classes (vLLM 0.32 serves from vllm.v1.worker.gpu.model_runner; an earlier version hooked only
        # gpu_model_runner and every line said "step -1")
        for mod in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
            try:
                cls = getattr(importlib.import_module(mod), "GPUModelRunner", None)
            except Exception:
                cls = None
            if cls is None or getattr(cls, "_varena_ll", False):
                continue

            def _wrap(ex):
                def execute_model(self, scheduler_output, *a, **k):
                    try:
                        ctx["tok"] = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
                        ctx["nreq"] = len(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
                    except Exception:
                        pass
                    ctx["step"] += 1
                    t0 = time.perf_counter()
                    out = ex(self, scheduler_output, *a, **k)
                    dt = time.perf_counter() - t0
                    if dt > slim:
                        _p(f"SLOW_STEP {dt * 1e3:.0f}ms")
                    if _alloc_log:
                        # VARENA_ALLOC_LOG=1 (allocator hypothesis): counters that move
                        # only when PyTorch calls cudaMalloc / cudaFree; log every change with the step context
                        try:
                            import torch as _t
                            ms = _t.cuda.memory_stats()
                            cur = (ms.get("num_alloc_retries", 0), ms.get("num_device_alloc", 0),
                                   ms.get("num_device_free", 0), ms.get("segment.all.current", 0),
                                   ms.get("reserved_bytes.all.current", 0))
                            if _alloc_prev and cur != _alloc_prev[0]:
                                pr = _alloc_prev[0]
                                _p(f"ALLOC_CHANGE retries +{cur[0] - pr[0]} cudaMalloc +{cur[1] - pr[1]} cudaFree "
                                   f"+{cur[2] - pr[2]} segments {cur[3]} reserved {cur[4] / 2**20:.0f} MiB "
                                   f"({(cur[4] - pr[4]) / 2**20:+.0f}) step_ms {dt * 1e3:.0f}")
                            _alloc_prev[:] = [cur]
                        except Exception:
                            pass
                    return out
                return execute_model
            cls.execute_model, cls._varena_ll = _wrap(cls.execute_model), True
            hooked.append(mod)
            _runner_classes.append(cls)
        _p(f"step hook on {hooked}")
    except Exception as e:
        print(f"[launch-log] step hook failed: {e}", file=sys.stderr, flush=True)
    if os.environ.get("VARENA_GPU_STEP_LOG", "0") != "1":
        return
    # VARENA_GPU_STEP_LOG=1 (dip diagnosis): GPU-clock evidence for the ~0.5 s compute-stream stalls.
    # 1. every engine step: CUDA events at the start and end of execute_model on the launching stream; steps whose GPU
    #    span exceeds VARENA_GPU_STEP_MS (350) are logged with tokens/requests.
    # 2. a probe thread records an event on its own high-priority stream every 10 ms and compares host record gaps with
    #    GPU gaps: a GPU gap with no host gap = the device itself did not service the probe stream; a host gap = the
    #    record call was blocked (driver lock held).
    try:
        import collections
        import torch
        gl = float(os.environ.get("VARENA_GPU_STEP_MS", "350"))
        pend = collections.deque()
        lk = threading.Lock()

        def _reap():
            while True:
                time.sleep(0.05)
                with lk:
                    while pend and pend[0][4].query():
                        ht, st, tok, e0, e1 = pend.popleft()
                        ms = e0.elapsed_time(e1)
                        if ms > gl:
                            _p(f"GPU_STEP {ms:.0f}ms (host start {time.strftime('%H:%M:%S', time.localtime(ht))}."
                               f"{int(ht % 1 * 1000):03d}) step {st} tok {tok}")
        started = []

        def _sealed():
            try:
                from schemes.memory_link_exhaustion.inference import prover as _pv
                return _pv.STATE is not None and bool(getattr(_pv.STATE, "sealed", False))
            except Exception:
                return False

        def _start():                       # lazily, in the engine process only, once the arena is sealed (after
            if started or not _sealed():     # vLLM's CUDA-graph capture: an event record from another thread during
                return                       # capture aborts it), never at plugin load (aborted the registry subprocess)
            started.append(1)
            threading.Thread(target=_reap, daemon=True, name="gpu-step-reap").start()
            threading.Thread(target=_probe, daemon=True, name="gpu-probe").start()
            _p("GPU step log and probe on")
        for cls_ in _runner_classes:
            ex2 = cls_.execute_model

            def _wrap2(ex2):
                def execute_model(self, scheduler_output, *a, **k):
                    _start()
                    if not started or torch.cuda.is_current_stream_capturing():
                        return ex2(self, scheduler_output, *a, **k)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    out = ex2(self, scheduler_output, *a, **k)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e1.record()
                    with lk:
                        pend.append((time.time(), ctx["step"], ctx["tok"], e0, e1))
                    return out
                return execute_model
            cls_.execute_model = _wrap2(ex2)

        def _probe():
            torch.cuda.set_device(0)
            ps = torch.cuda.Stream(priority=-1)
            prev = None
            while True:
                time.sleep(0.01)
                h0 = time.perf_counter()
                e = torch.cuda.Event(enable_timing=True)
                e.record(ps)
                h1 = time.perf_counter()
                if (h1 - h0) > 0.05:
                    _p(f"PROBE_HOST record blocked {(h1 - h0) * 1e3:.0f}ms")
                if prev is not None:
                    e.synchronize()
                    g = prev[0].elapsed_time(e)
                    hg = (h1 - prev[1]) * 1e3
                    if g > 60 or hg > 60:
                        _p(f"PROBE_GAP gpu {g:.0f}ms host {hg:.0f}ms")
                prev = (e, h1)
    except Exception as e:
        print(f"[launch-log] step hook failed: {e}", file=sys.stderr, flush=True)


def register():
    _launch_log()
    _reserve_sms()
    _main_stream()
    _runahead()
    _no_uva()
    if os.environ.get("VARENA_TRITON_GEMM") == "1":
        from schemes.memory_link_exhaustion.inference.patches import install_triton_gemm_overrides
        install_triton_gemm_overrides()
    if os.environ.get("VARENA") != "1":
        return
    try:
        _patch_worker()
        _patch_scheduler()
    except ImportError:
        return


def _patch_worker():
    from vllm.v1.worker import gpu_worker as GW
    W = GW.Worker
    if getattr(W, "_varena", False):
        return
    load_model, init_cfg, warm = W.load_model, W.initialize_from_config, W.compile_or_warm_up_model

    def load_model_p(self, *a, **kw):
        from schemes.memory_link_exhaustion.inference import patches, prover
        patches.install_encoded()
        st = prover.get()
        st.connect(self.vllm_config.model_config.model)
        out = load_model(self, *a, **kw)
        st.encode_weights(self.model_runner.get_model())
        return out

    def init_cfg_p(self, *a, **kw):
        from schemes.memory_link_exhaustion.inference import prover
        out = init_cfg(self, *a, **kw)
        prover.get().tag_kv(self.vllm_config)
        return out

    def warm_p(self, *a, **kw):
        from schemes.memory_link_exhaustion.inference import prover
        out = warm(self, *a, **kw)
        prover.get().seal(self)
        return out

    W.load_model, W.initialize_from_config, W.compile_or_warm_up_model = load_model_p, init_cfg_p, warm_p
    W._varena = True


def _patch_scheduler():
    from vllm.v1.core import block_pool as BPm
    from vllm.v1.core.sched import scheduler as Sm
    BP, S = BPm.BlockPool, Sm.Scheduler
    if getattr(S, "_varena", False):
        return
    free_blocks, get_new = BP.free_blocks, BP.get_new_blocks
    s_init, s_sched = S.__init__, S.schedule

    def free_blocks_p(self, ordered_blocks):
        from schemes.memory_link_exhaustion.inference import prover
        st = prover.STATE
        if st is None or not st.sealed:
            return free_blocks(self, ordered_blocks)
        blocks = list(ordered_blocks)
        free_blocks(self, blocks)
        seen, released = set(), []
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null and b.block_id not in seen and \
                    (b.pool is None or b.pool is self):
                seen.add(b.block_id)
                released.append(b)
        for b in released:
            self.free_block_queue.remove(b)
        if released:
            st.on_free(self, released)

    def get_new_p(self, n):
        out = get_new(self, n)
        from schemes.memory_link_exhaustion.inference import prover
        if prover.STATE is not None:
            prover.STATE.on_alloc(out)
        return out

    def init_p(self, *a, **kw):
        s_init(self, *a, **kw)
        mgrs = self.kv_cache_manager.coordinator.single_type_managers
        assert len(mgrs) == 1, f"{len(mgrs)} KV cache groups"
        self._varena_blocks = mgrs[0].req_to_blocks
        # a step may still write a finished request's blocks; free only after it
        self.defer_block_free = True

    def sched_p(self, *a, **kw):
        from schemes.memory_link_exhaustion.inference import prover
        if prover.STATE is not None:
            prover.STATE.on_schedule(self)
        out = s_sched(self, *a, **kw)
        if prover.STATE is not None:
            # settled-KV rule: the block table for the step this call just scheduled must be on the device, and no
            # chase that read the previous table may still be running, before the step can write (no-op when off)
            prover.STATE.on_scheduled(self)
        return out

    BP.free_blocks, BP.get_new_blocks = free_blocks_p, get_new_p
    S.__init__, S.schedule = init_p, sched_p
    S._varena = True
    if os.environ.get("VARENA_SCHED_TRACE"):
        _patch_sched_trace(S, os.environ["VARENA_SCHED_TRACE"], s_init)


def _patch_sched_trace(S, path, true_init):
    """VARENA_SCHED_TRACE=<file>: record every call that changes the scheduler's state, in order (constructor
    settings, add_request, schedule, update_from_output, finish_requests), as a pickle stream of plain Python values
    (numbers, strings, lists, dicts: no engine object is pickled; pickling the live configuration hung the engine).
    The verifier replays the same calls on its own Scheduler (verifier/schedreplay.py) and derives the block table
    from that, so the table no longer has to be taken from the prover. What the trace adds to what the verifier
    already knows (its requests and the returned tokens) is only the order of the calls. A failure to record never
    touches the engine: it is written into the trace as an ERR record."""
    import dataclasses
    import inspect
    import pickle
    init0, add0, sched0, upd0, fin0 = S.__init__, S.add_request, S.schedule, S.update_from_output, S.finish_requests
    sig = inspect.signature(true_init)           # (init0 is the arena's wrapper: it only has *a, **kw)
    PLAIN = (int, float, bool, str, type(None))

    def plain(v):
        return v if isinstance(v, PLAIN) else ("repr", repr(v)[:200])

    def fields(o):
        if dataclasses.is_dataclass(o):
            return {f.name: plain(getattr(o, f.name, None)) for f in dataclasses.fields(o)}
        return {k: plain(v) for k, v in list(vars(o).items())[:200] if not k.startswith("_")}

    def state():
        from schemes.memory_link_exhaustion.inference import prover
        st = prover.STATE
        kt = getattr(st, "kvt", None) if st is not None else None
        ih = getattr(st, "_kvt_info_host", None) if kt is not None else None      # (version, digest) as published
        return (bool(st is not None and st.sealed), kt is not None, None if kt is None else int(kt.version),
                None if ih is None else int(ih[1]) & 0xFFFFFFFFFFFFFFFF)

    def put(self, rec):
        f = getattr(self, "_varena_trace", None)
        if f is None:
            return
        try:
            f.write(pickle.dumps(rec, protocol=4))
        except Exception as e:
            f.write(pickle.dumps(("ERR", rec[0], f"{type(e).__name__}: {e}"), protocol=4))

    def init_t(self, *a, **kw):
        init0(self, *a, **kw)
        try:
            self._varena_trace = open(f"{path}.{os.getpid()}", "wb")     # one file per engine process
            self._varena_so = {}
            self._varena_nsched = 0
            ba = sig.bind(self, *a, **kw).arguments
            kc = ba.get("kv_cache_config")
            kv = {"num_blocks": int(kc.num_blocks), "tensors": len(kc.kv_cache_tensors),
                  "groups": [{"layer_names": [str(x) for x in g.layer_names],
                              "spec_module": type(g.kv_cache_spec).__module__, "spec_cls": type(g.kv_cache_spec).__qualname__,
                              "spec": fields(g.kv_cache_spec),
                              "group": {k: v for k, v in fields(g).items() if k not in ("layer_names", "kv_cache_spec")}}
                             for g in kc.kv_cache_groups]}
            put(self, ("I", {"module": type(self).__module__, "cls": type(self).__qualname__,
                             "block_size": plain(ba.get("block_size")), "hash_block_size": plain(ba.get("hash_block_size")),
                             "include_finished_set": plain(ba.get("include_finished_set", False)),
                             "log_stats": plain(ba.get("log_stats", False)),
                             "defer_block_free": bool(self.defer_block_free), "kv": kv,
                             "scheduler_config": fields(self.scheduler_config),
                             "cache_config": fields(self.cache_config)}))
            self._varena_trace.flush()
        except Exception as e:
            print(f"[varena] scheduler trace off: {type(e).__name__}: {e}", flush=True)
            self._varena_trace = None

    def add_t(self, request):
        if getattr(self, "_varena_trace", None) is not None:
            sp = getattr(request, "sampling_params", None)
            put(self, ("A", state(),
                       {"request_id": str(request.request_id),
                        "prompt_token_ids": [int(t) for t in (request.prompt_token_ids or ())],
                        "max_tokens": plain(getattr(sp, "max_tokens", None)),
                        "arrival_time": plain(getattr(request, "arrival_time", None)),
                        "priority": plain(getattr(request, "priority", 0)),
                        "client_index": plain(getattr(request, "client_index", 0))}))
        return add0(self, request)

    def sched_t(self, *a, **kw):
        tr = getattr(self, "_varena_trace", None)
        st0 = state() if tr is not None else None
        out = sched0(self, *a, **kw)
        if tr is not None:
            k = self._varena_nsched
            self._varena_nsched += 1
            # which schedule() call an update belongs to: the index rides on the output object itself; the id map is
            # only the fallback, popped on use (an id is reused once an output is freed: a map pruned by insertion
            # order dropped the newest call's entry whenever its id had been seen before, and that update was lost)
            try:
                out._varena_k = k
            except Exception:
                self._varena_so.pop(id(out), None)
                self._varena_so[id(out)] = k
                while len(self._varena_so) > 256:
                    self._varena_so.pop(next(iter(self._varena_so)))
            put(self, ("S", st0, k, ([plain(x) for x in a], {str(x): plain(y) for x, y in kw.items()}), state(),
                       {str(r): int(n) for r, n in out.num_scheduled_tokens.items()}))
            tr.flush()
        return out

    def upd_t(self, scheduler_output, model_runner_output):
        if getattr(self, "_varena_trace", None) is not None:
            m = model_runner_output
            k_ = getattr(scheduler_output, "_varena_k", None)
            if k_ is None:
                k_ = self._varena_so.pop(id(scheduler_output), -1)
            put(self, ("U", state(), k_,
                       {"req_ids": [str(x) for x in (getattr(m, "req_ids", ()) or ())],
                        # (the scheduler looks requests up through this map, which can hold more than req_ids lists)
                        "req_id_to_index": {str(k): int(v) for k, v in (getattr(m, "req_id_to_index", {}) or {}).items()},
                        "sampled_token_ids": [[int(t) for t in x] for x in (getattr(m, "sampled_token_ids", ()) or ())]}))
        return upd0(self, scheduler_output, model_runner_output)

    def fin_t(self, request_ids, finished_status):
        if getattr(self, "_varena_trace", None) is not None:
            ids = request_ids if (request_ids is None or isinstance(request_ids, str)) else [str(x) for x in request_ids]
            put(self, ("F", state(), ids, int(finished_status)))
            self._varena_trace.flush()
            request_ids = ids
        return fin0(self, request_ids, finished_status)

    S.__init__, S.add_request, S.schedule, S.update_from_output, S.finish_requests = init_t, add_t, sched_t, upd_t, fin_t
