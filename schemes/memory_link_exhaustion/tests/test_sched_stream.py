"""Scheduled noise stream without vLLM: does the prover keep up with the verifier's clock?"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
from schemes.memory_link_exhaustion.link import pcie as P


def main():
    # In a function behind the __main__ guard: the Revealer is a spawned process, and spawn re-imports this module
    # in the child (at module level that started the test again in every child and never finished).
    gbs = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
    W = int(os.environ.get("W_MIB", "32")) << 20
    pool_mm, pool = P.create_pool("/dev/shm/t_pool", (2048 << 20) + 4096, 1)
    ck = P.Checker(pool, 1234, W, 3)
    delta = W / (gbs * 1e9)
    ck.sched = {"delta": delta, "slack": 0.03}
    rv = P.Revealer("/dev/shm/t_rev", 4096, 777, delta, 16 * delta, ck, link_frac=min(1.0, gbs / 55.5))
    st = P.Streamer("/dev/shm/t_pool", 1234, window_bytes=W, lag=3,
                    ring_bytes=1 << 30, sched={"rev_path": "/dev/shm/t_rev", "N": 4096})
    times = []
    st.on_window = lambda i, off, d0, d1, t: times.append((i, t))
    rv.start(); st.start()
    time.sleep(float(os.environ.get('SECS', '6')))
    st.stop(); time.sleep(0.05)
    print("revealer stats:", rv.stats(), flush=True)
    rv.stop()
    n = len(times)
    lat = [t - (rv.t0 + i * delta) for i, t in times] or [0.0]
    print(f"scheduled {gbs} GB/s: {n} windows in 6 s = {n * W / 6 / 1e9:.1f} GB/s; lateness p50 {np.median(lat)*1e3:.1f} ms "
          f"p99 {np.percentile(lat, 99)*1e3:.1f} ms")
    for f in ("/dev/shm/t_pool", "/dev/shm/t_rev", "/dev/shm/t_rev.dl"):
        os.unlink(f)
    if n == 0:
        print("FAIL: no window was streamed")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
