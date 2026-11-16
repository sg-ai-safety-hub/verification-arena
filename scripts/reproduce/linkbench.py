"""Raw PCIe ceiling on this host (no vLLM): pinned H2D / D2H copies, chunk sizes, streams, bidirectional."""
import torch, time
dev = torch.device("cuda")
N = 1 << 30
h = torch.empty(N, dtype=torch.uint8).pin_memory(); h2 = torch.empty(N, dtype=torch.uint8).pin_memory()
d = torch.empty(N, dtype=torch.uint8, device=dev); d2 = torch.empty(N, dtype=torch.uint8, device=dev)
def run(chunk, nstreams, direction, secs=3.0):
    ss = [torch.cuda.Stream() for _ in range(nstreams)]
    torch.cuda.synchronize(); t0 = time.perf_counter(); moved = {"h2d": 0, "d2h": 0}; o = 0
    while time.perf_counter() - t0 < secs:
        for k, s in enumerate(ss):
            with torch.cuda.stream(s):
                for _ in range(8):
                    a = o % (N - chunk)
                    if direction in ("h2d", "bi"):
                        d[a:a + chunk].copy_(h[a:a + chunk], non_blocking=True); moved["h2d"] += chunk
                    if direction in ("d2h", "bi"):
                        h2[a:a + chunk].copy_(d2[a:a + chunk], non_blocking=True); moved["d2h"] += chunk
                    o += chunk
        for s in ss: s.synchronize()
    dt = time.perf_counter() - t0
    return {k: round(v / dt / 1e9, 2) for k, v in moved.items() if v}
import sys
if "--quick" in sys.argv:                      # check_host.sh noise-path check: the 32 MiB H2D ceiling only
    print("RAW_H2D_32MIB", max(run(32 << 20, ns, "h2d")["h2d"] for ns in (1, 2, 4)), flush=True)
    sys.exit(0)
out = []
for direction in ("h2d", "d2h", "bi"):
    for chunk_mib in (4, 32, 256):
        for ns in (1, 2, 4):
            r = run(chunk_mib << 20, ns, direction)
            out.append((direction, chunk_mib, ns, r)); print("LINK", direction, chunk_mib, "MiB", ns, "streams", r, flush=True)
# the 32 MiB H2D ceiling the noise-path check compares against (same number --quick measures alone)
print("RAW_H2D_32MIB", max(r["h2d"] for d, c, ns, r in out if d == "h2d" and c == 32), flush=True)
