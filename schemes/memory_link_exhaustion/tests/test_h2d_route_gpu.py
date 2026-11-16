"""GPU self-test of the single H2D queue with tools/h2d_route.so (every host-to-device cudaMemcpy[Async] in the process
goes through the noise streamer's queue). No vLLM, no verifier: saturating inbound noise (32 MiB pinned->device
windows back to back, two in flight) from gpu/native.py's self-test driver, then plain torch copies of every kind.

  gcc -O2 -shared -fPIC -o /root/h2d_route.so schemes/memory_link_exhaustion/tools/h2d_route.c -ldl
  LD_PRELOAD=/root/h2d_route.so python -m schemes.memory_link_exhaustion.tests.test_h2d_route_gpu [--no-ref]

Two phases, each in its own process, queue phase first. "queue": the routed path (below). "ref" (skipped with
--no-ref): the same noise with the queue OFF on purpose, so it routes nothing; its small-copy latency is the reference.
Queue phase:
  1. correctness: 3,000 random copies (pinned non_blocking, pageable non_blocking, pageable blocking, torch.tensor(list,
     device="cuda"), .to("cuda")) on the default stream and a side stream, each checked against its source snapshot
  2. stream order: a kernel still reading the destination (torch.cuda._sleep, then clone) sees the old bytes
  3. small-copy latency (copy + sync) under saturating noise: p50 / p99 / max
  4. noise rate while copying (should stay near the link ceiling)
  5. interposer counters: everything queued, no fallback, guard not tripped
  6. head-of-line: a stream busy for ~2 s with a copy queued on it must not delay copies on another stream
  7. many streams (SELFTEST_STREAMS, default 24, default stream included) copying from as many threads
  8. CUDA graph capture (with an H2D inside it) and 200 replays while another thread keeps copying
Noise: SELFTEST_INFLIGHT windows in flight (default 4, like VARENA_SAT_LEAD=4); default 8 connections unless
CUDA_DEVICE_MAX_CONNECTIONS is set.
Prints one JSON line per phase and PASS / FAIL.
"""

from __future__ import annotations

import ctypes
import json
import os
import random
import subprocess
import sys
import time

W = 32 << 20


def _route_lib():
    for p in os.environ.get("LD_PRELOAD", "").split(":"):
        if "h2d_route" in p:
            return ctypes.CDLL(p)
    return None


def _stats(lib):
    if lib is None:
        return None
    a = (ctypes.c_uint64 * 4)()
    lib.varena_h2d_stats(a)
    return {"queued": a[0], "passed": a[1], "fallback": a[2], "over_max": a[3]}


def _say(msg):
    print(f"[selftest {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _watchdog(nm, period=30.0):
    """Every `period` s: queue counters and every Python thread's stack (a hang shows where it waits)."""
    import faulthandler
    import threading

    def run():
        while True:
            time.sleep(period)
            try:
                st = [int(x) for x in nm.sq_stats()] if hasattr(nm, "sq_stats") else None
            except Exception as e:
                st = f"err {e}"
            _say(f"watchdog: sq_stats [queued staged full stage_full err head tail] = {st}")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    threading.Thread(target=run, daemon=True).start()


def _capture_skips(lib):
    if lib is None or not hasattr(lib, "varena_h2d_capture_skips"):
        return None
    lib.varena_h2d_capture_skips.restype = ctypes.c_uint64
    return int(lib.varena_h2d_capture_skips())


def phase(use_queue: bool, n_copies: int, n_lat: int) -> dict:
    import torch
    from schemes.memory_link_exhaustion.gpu import native
    if use_queue:
        os.environ["VARENA_SINGLE_QUEUE"] = "1"
    _say(f"phase queue={use_queue}: building/loading native module")
    nm = native.mod()
    _watchdog(nm)
    dev = torch.cuda.current_device()
    pool = torch.empty(W, dtype=torch.uint8).pin_memory()
    ring = torch.empty(W, dtype=torch.uint8, device="cuda")
    s_noise = torch.cuda.Stream()
    lib = _route_lib()
    out = {"queue": use_queue, "interposer": lib is not None, "errors": []}
    side = torch.cuda.Stream()
    dst = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    inflight = int(os.environ.get("SELFTEST_INFLIGHT", "4"))     # windows in flight, like VARENA_SAT_LEAD=4
    out["inflight"] = inflight
    nm.sq_selftest_start(s_noise.cuda_stream, pool.data_ptr(), ring.data_ptr(), W, dev, use_queue, inflight)
    t_noise0 = time.perf_counter()
    time.sleep(1.0)
    if use_queue:
        st0 = [int(x) for x in nm.sq_stats()]
        _say(f"noise on; queue active={bool(nm.sq_active())} sq_stats={st0}")
        if not nm.sq_active():
            out["errors"].append("queue did not come on")
    else:
        _say("noise on (queue off)")                                   # noise at the ceiling before measuring

    # 3. latency first (clean distribution), with a timeout so the starved reference still finishes
    lat = []
    src = torch.ones(4096, dtype=torch.int32).pin_memory()
    d = torch.empty(4096, dtype=torch.int32, device="cuda")
    starved = 0
    for _ in range(n_lat):
        t0 = time.perf_counter()
        with torch.cuda.stream(side):
            d.copy_(src, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(side)
        while not ev.query():
            if time.perf_counter() - t0 > 10.0:
                starved += 1
                break
            time.sleep(0)
        lat.append(time.perf_counter() - t0)
        if starved >= 3:
            break
    _say(f"latency done: {len(lat)} copies, {starved} over 10 s, interposer {_stats(lib)}")
    lat.sort()
    out["lat_ms"] = {"n": len(lat), "p50": 1e3 * lat[len(lat) // 2], "p99": 1e3 * lat[min(len(lat) - 1, int(len(lat) * 0.99))],
                     "max": 1e3 * lat[-1], "over_10s": starved}

    if use_queue:
        # 0. smoke: one copy of each kind on a side stream and on the default stream, each in a thread with a 15 s
        # timeout, named before it starts, so a hang says exactly which kind and stream (once: the first pageable
        # .to("cuda") on the default stream never landed)
        import threading
        kinds = {0: "pinned non_blocking", 1: "pageable non_blocking", 2: "pageable blocking", 3: "torch.tensor(list)",
                 4: ".to(cuda) pageable"}
        smoke = {}
        for st_name, st in (("side", side), ("default", torch.cuda.default_stream())):
            for kind, kname in kinds.items():
                host = torch.randint(0, 256, (70000,), dtype=torch.uint8, generator=torch.Generator().manual_seed(kind))
                res = {}

                def one():
                    with torch.cuda.stream(st):
                        if kind == 0:
                            dst[:70000].copy_(host.pin_memory(), non_blocking=True); t = dst[:70000]
                        elif kind == 1:
                            dst[:70000].copy_(host, non_blocking=True); t = dst[:70000]
                        elif kind == 2:
                            dst[:70000].copy_(host); t = dst[:70000]
                        elif kind == 3:
                            t = torch.tensor(host[:4096].tolist(), dtype=torch.uint8, device="cuda")
                        else:
                            t = host.to("cuda")
                        res["ok"] = torch.equal(t.cpu(), host[: t.numel()])
                _say(f"smoke: {kname} on {st_name} stream ...")
                th = threading.Thread(target=one, daemon=True)
                th.start()
                th.join(timeout=15)
                key = f"{kname} @ {st_name}"
                smoke[key] = "HUNG" if th.is_alive() else ("ok" if res.get("ok") else "WRONG")
                _say(f"smoke: {key}: {smoke[key]}  interposer {_stats(lib)}")
                if th.is_alive():
                    out["smoke"] = smoke
                    out["errors"].append(f"smoke copy hung: {key}")
                    out["native_stats"] = [int(x) for x in nm.sq_stats()]
                    out["interposer_stats"] = _stats(lib)
                    return out                      # the GPU is stuck: report now, the caller exits hard
        out["smoke"] = smoke
        out["errors"] += [f"smoke {k}: {v}" for k, v in smoke.items() if v != "ok"]

        # 1. correctness
        rng = random.Random(1)
        t_c0 = time.perf_counter()
        for it in range(n_copies):
            n = rng.randint(1, 1 << 18)
            kind = rng.randrange(5)
            st = side if rng.random() < 0.5 else torch.cuda.default_stream()
            g = torch.Generator().manual_seed(it)
            host = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
            ref = host.clone()
            with torch.cuda.stream(st):
                if kind == 0:
                    h = host.pin_memory()
                    dst[:n].copy_(h, non_blocking=True)
                    t = dst[:n]
                elif kind == 1:
                    dst[:n].copy_(host, non_blocking=True)
                    host.fill_(0xEE)                 # pageable buffer reused at once
                    t = dst[:n]
                elif kind == 2:
                    dst[:n].copy_(host)
                    t = dst[:n]
                elif kind == 3:
                    t = torch.tensor(host[: min(n, 4096)].tolist(), dtype=torch.uint8, device="cuda")
                    ref = ref[: min(n, 4096)]
                else:
                    t = host.to("cuda")
                got = t.cpu()                         # D2H on the same stream: ordered after the copy
            if not torch.equal(got, ref):
                out["errors"].append(f"copy {it} kind {kind} n {n} mismatch")
                if len(out["errors"]) > 10:
                    break
            if it % 500 == 0:
                _say(f"correctness {it}/{n_copies}")
        _say(f"correctness done: errors {len(out['errors'])}, interposer {_stats(lib)}")
        out["copies"] = n_copies
        out["copies_s"] = time.perf_counter() - t_c0

        # 2. stream order (write-after-read on the destination)
        for k in range(20):
            a = torch.full((1 << 16,), k, dtype=torch.uint8).pin_memory()
            b = torch.full((1 << 16,), k + 100, dtype=torch.uint8).pin_memory()
            buf = torch.empty(1 << 16, dtype=torch.uint8, device="cuda")
            with torch.cuda.stream(side):
                buf.copy_(a, non_blocking=True)
                torch.cuda._sleep(20_000_000)         # ~10 ms of a kernel before the reader
                seen = buf.clone()
                buf.copy_(b, non_blocking=True)
                after = buf.clone()
            side.synchronize()
            if not (bool((seen == k).all()) and bool((after == k + 100).all())):
                out["errors"].append(f"stream order broken at {k}: seen {int(seen[0])} after {int(after[0])}")
                break

        # 4. head-of-line: stream A busy for ~2 s (as when vLLM's matmul spins on VARENA_MM_GATE), a copy queued on A;
        # copies on another stream must still land at once (one shared queue stream held them behind A)
        A = torch.cuda.Stream()
        ha = torch.full((4096,), 7, dtype=torch.uint8).pin_memory()
        da = torch.empty(4096, dtype=torch.uint8, device="cuda")
        with torch.cuda.stream(A):
            torch.cuda._sleep(int(os.environ.get("SELFTEST_HOL_CYCLES", "4000000000")))
            da.copy_(ha, non_blocking=True)
        hol = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.cuda.stream(side):
                d.copy_(src, non_blocking=True)
            side.synchronize()
            hol.append(time.perf_counter() - t0)
        a_busy = not A.query()
        A.synchronize()
        out["hol_ms_max"] = 1e3 * max(hol)
        out["hol_a_still_busy"] = a_busy
        _say(f"head-of-line: other-stream copy max {out['hol_ms_max']:.2f} ms while A busy={a_busy}")
        if a_busy and max(hol) > 0.5:
            out["errors"].append(f"copy on another stream waited {1e3 * max(hol):.0f} ms behind a busy stream")
        if not bool((da == 7).all()):
            out["errors"].append("copy queued behind the busy stream has wrong data")

        # 5. many streams and threads at once (more streams than CUDA_DEVICE_MAX_CONNECTIONS), default stream included
        import threading
        errs, lock = [], threading.Lock()

        def worker(k):
            st_ = torch.cuda.default_stream() if k == 0 else torch.cuda.Stream()
            buf = torch.empty(1 << 16, dtype=torch.uint8, device="cuda")
            r_ = random.Random(100 + k)
            for it in range(150):
                n = r_.randint(1, 1 << 16)
                h = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=torch.Generator().manual_seed(k * 1000 + it))
                with torch.cuda.stream(st_):
                    buf[:n].copy_(h.pin_memory() if it % 2 else h, non_blocking=True)
                    got = buf[:n].cpu()
                if not torch.equal(got, h):
                    with lock:
                        errs.append(f"stream worker {k} copy {it} mismatch")
                    return
        ths = [threading.Thread(target=worker, args=(k,)) for k in range(int(os.environ.get("SELFTEST_STREAMS", "24")))]
        t_m0 = time.perf_counter()
        for t_ in ths:
            t_.start()
        for t_ in ths:
            t_.join(timeout=120)
        hung = sum(t_.is_alive() for t_ in ths)
        out["many_streams_s"] = time.perf_counter() - t_m0
        _say(f"many streams: {len(ths)} threads, {len(errs)} errors, {hung} hung, {out['many_streams_s']:.1f} s")
        out["errors"] += errs[:5]
        if hung:
            out["errors"].append(f"{hung} stream workers hung")

        # 6. CUDA graph capture and replay while another thread keeps copying (vLLM warm-up and decode)
        stop_bg = [False]
        bg_err = []

        def bg():
            st_ = torch.cuda.Stream()
            buf = torch.empty(4096, dtype=torch.int32, device="cuda")
            i_ = 0
            hp = [torch.empty(4096, dtype=torch.int32).pin_memory() for _ in range(2)]   # no cudaHostAlloc in capture
            while not stop_bg[0]:
                h = hp[i_ % 2]
                h.fill_(i_)
                with torch.cuda.stream(st_):
                    buf.copy_(h, non_blocking=True)
                    got = buf.cpu()
                if not bool((got == i_).all()):
                    bg_err.append(f"copy during capture/replay mismatch at {i_}")
                    return
                i_ += 1
        tb = threading.Thread(target=bg)
        tb.start()
        try:
            x = torch.randn(256, 256, device="cuda")
            static_in = torch.zeros(256, dtype=torch.float32, device="cuda")
            host_in = torch.arange(256, dtype=torch.float32).pin_memory()
            cs = torch.cuda.Stream()
            cs.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(cs):                         # warm-up as vLLM does before capture
                y = x @ static_in
            torch.cuda.current_stream().wait_stream(cs)
            g = torch.cuda.CUDAGraph()
            # thread_local: the test's own copy thread keeps synchronising its stream during the capture
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                static_in.copy_(host_in, non_blocking=True)     # H2D inside the capture: never routed
                y = x @ static_in
            for r_ in range(200):
                host_in.fill_(float(r_))
                g.replay()
                torch.cuda.current_stream().synchronize()
                if not torch.allclose(y, x.sum(1) * float(r_)):
                    out["errors"].append(f"graph replay {r_} wrong")
                    break
        except Exception as e:
            out["errors"].append(f"graph capture/replay raised {type(e).__name__}: {e}")
        stop_bg[0] = True
        tb.join(timeout=60)
        if tb.is_alive():
            out["errors"].append("copy thread hung during capture/replay")
        out["errors"] += bg_err[:3]
        _say(f"capture/replay done; capture-time skips {_capture_skips(lib)}")
        out["capture_skips"] = _capture_skips(lib)

    t_end = time.perf_counter()
    _say("stopping noise")
    windows = nm.sq_selftest_stop()
    out["noise_gbs"] = windows * W / 1e9 / (t_end - t_noise0)
    out["interposer_stats"] = _stats(lib)
    if use_queue:
        out["native_stats"] = dict(zip(["queued", "staged", "full", "stage_full", "err", "head", "tail",
                                        "caller_streams", "guard_tripped"], [int(x) for x in nm.sq_stats()]))
        if out["native_stats"].get("guard_tripped"):
            out["errors"].append("queue guard tripped (a routed copy was stuck; routing turned itself off)")
        st = out["interposer_stats"]
        if lib is None:
            out["errors"].append("h2d_route.so not in LD_PRELOAD")
        elif st["fallback"]:
            out["errors"].append(f"{st['fallback']} copies fell back to the direct path")
        elif not st["queued"]:
            out["errors"].append("interposer queued nothing (router not registered?)")
        if out["lat_ms"]["over_10s"]:
            out["errors"].append("small copy starved > 10 s with the queue on")
    return out


def main():
    if "--phase" in sys.argv:
        q = sys.argv[sys.argv.index("--phase") + 1] == "queue"
        import torch
        torch.cuda.init()
        r = phase(q, n_copies=3000, n_lat=2000 if q else 200)
        print("RESULT " + json.dumps(r), flush=True)
        sys.stderr.flush()
        os._exit(0)                                     # never hang in teardown behind a stuck stream
    results = []
    # queue phase first (the one that matters); each phase in its own process (the queue is one-shot per process)
    phases = ["queue"] if "--no-ref" in sys.argv else ["queue", "ref"]
    tmo = float(os.environ.get("SELFTEST_PHASE_TIMEOUT", "360"))
    for ph in phases:
        cmd = [sys.executable, "-m", "schemes.memory_link_exhaustion.tests.test_h2d_route_gpu", "--phase", ph]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=tmo)
            so, se, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            so = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            se = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            rc = f"timeout {tmo:.0f} s"
        sys.stderr.write(f"----- phase {ph} stderr (tail) -----\n" + se[-12000:] + "\n")
        line = [ln for ln in so.splitlines() if ln.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"queue": ph == "queue", "errors": [f"phase {ph} exit {rc}"]}
        print(json.dumps(r), flush=True)
        results.append(r)
    q = [r for r in results if r.get("queue")][-1]
    print("PASS" if not q["errors"] else "FAIL: " + "; ".join(q["errors"]), flush=True)
    sys.exit(0 if not q["errors"] else 1)


if __name__ == "__main__":
    main()
