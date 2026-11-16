"""Deterministic (blocked-summation) pretraining baseline vs the default one, same box.
mode=default: cuBLAS + flash SDPA. mode=det: vLLM's batch-invariant Triton matmul for every mm/addmm (fixed BLOCK_K
summation order, forward and backward) + torch deterministic algorithms (deterministic SDPA backward).
Also: weight hash after N optimizer steps, twice from the same seed (bit-exact run to run)."""
import hashlib, json, os, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
mode = sys.argv[1]          # default | det (Triton batch-invariant mm) | det2 (cuBLAS, math SDPA, non-fused AdamW)
EXTRA = {"sdpa": "math", "fused_adam": False} if mode == "det2" else {}
if mode in ("det", "det2"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
if mode == "det":
    from schemes.memory_link_exhaustion.inference.patches import install_triton_gemm_overrides
    install_triton_gemm_overrides()
    import vllm.model_executor.determinism.batch_invariant as bi
    bi.enable_batch_invariant_mode()
if mode in ("det", "det2"):
    torch.use_deterministic_algorithms(True, warn_only=True)
from baselines.pretrain.pretrain_qwen import PretrainQwen
A = {"d": 5120, "heads": 40, "kv": 8, "hd": 128, "ffn": 27648}
out = {}
for L, mb in ((8, 1), (8, 2)):
    torch.cuda.empty_cache()
    wl = PretrainQwen({"layers": L, "mb": mb, "seq": 2048, "global_tokens": 65536, "arch": dict(A), **EXTRA}, budget=None)
    for _ in range(2):
        wl.step()
    torch.cuda.synchronize(); t0, u = time.perf_counter(), 0
    while time.perf_counter() - t0 < 30:
        u += wl.step()
    ts = u / (time.perf_counter() - t0)
    out[f"L{L}_mb{mb}"] = {"tok_s": round(ts, 1), "param_tok_s": ts * wl.P, "params": wl.P}
    print("DET", mode, f"L{L}_mb{mb}", json.dumps(out[f"L{L}_mb{mb}"]), flush=True)
    del wl
def wh(steps):
    torch.cuda.empty_cache(); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    wl = PretrainQwen({"layers": 2, "mb": 1, "seq": 2048, "global_tokens": 8192, "arch": dict(A), **EXTRA}, budget=None)
    for _ in range(steps * wl.accum):
        wl.step()
    torch.cuda.synchronize()
    h = hashlib.sha256()
    for p in wl.model.parameters():
        h.update(p.detach().float().cpu().numpy().tobytes())
    del wl
    return h.hexdigest()[:16]
h1, h2 = wh(3), wh(3)
print("DET", mode, "weight_hash_3steps", h1, h2, "identical" if h1 == h2 else "DIFFERENT", flush=True)
