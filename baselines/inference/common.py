"""Settings shared by baseline inference, inference (under the verifier) and the
reference engine, so all three run the same kernels on the same shapes."""

from __future__ import annotations

import json
import os

import numpy as np

PAGE = 65536

BASE_ENV = {
    "VLLM_BATCH_INVARIANT": "1",       # deterministic mode
    "VARENA_TRITON_GEMM": "1",         # Triton persistent matmul for aten::mm & co. too
    "VLLM_TRITON_USE_TD": "0",         # one K/V load path in TRITON_ATTN
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # vLLM's compile cache is keyed by config and traced source files, not by which varena
    # patches are active (encoded W through the custom op, plaintext custom op, stock). A graph
    # compiled in one configuration was silently reused by another (encoded weights fed to the
    # plaintext op: garbage tokens; plaintext run calling the decode op: KeyError). Inductor's
    # own content-addressed caches stay on.
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    # No OMP_NUM_THREADS here: when it is set, vLLM keeps torch at 16 CPU threads instead of
    # reducing to 1 for serving; with the verifier's CPU load that cost ~30% honest tok/s (v4 round).
}


def apply_env(extra: dict | None = None):
    for k, v in {**BASE_ENV, **(extra or {})}.items():
        os.environ[k] = str(v)
    if os.environ.get("VARENA_COMPILE_CACHE", "1") == "1":
        # vLLM's compile cache, but one cache root per patch configuration (hash of every VARENA_* /
        # batch-invariance variable), so a graph is never reused across configurations (the cause of
        # the garbage tokens that VLLM_DISABLE_COMPILE_CACHE=1 guarded against); saves ~30 s per start
        import hashlib
        cfg = sorted((k, v) for k, v in os.environ.items()
                     if (k.startswith("VARENA") and "SOCK" not in k and "PROF" not in k)
                     or k in ("VLLM_BATCH_INVARIANT", "VLLM_TRITON_USE_TD"))
        src = b""
        for f in ("patches.py", "prover.py"):           # the patch code itself is part of the key
            try:
                src += open(os.path.join(os.path.dirname(__file__), "..", "..", "schemes", "memory_link_exhaustion", "inference", f), "rb").read()
            except OSError:
                pass
        h = hashlib.sha1(repr(cfg).encode() + hashlib.sha1(src).digest()).hexdigest()[:16]
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(os.path.expanduser("~/.cache/vllm_cfg"), h)
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"


def block_size_for(model: str) -> int:
    """Tokens per KV block such that one layer's block is exactly one 64 KiB page."""
    c = json.load(open(os.path.join(model, "config.json")))
    hq = c["num_attention_heads"]
    hkv = c.get("num_key_value_heads", hq)
    d = c.get("head_dim") or c["hidden_size"] // hq
    per_tok = 2 * hkv * d * 2
    assert PAGE % per_tok == 0, per_tok
    return PAGE // per_tok


def llm_kwargs(a) -> dict:
    kw = dict(model=a.model, enable_prefix_caching=False, seed=0, attention_backend="TRITON_ATTN",
              block_size=block_size_for(a.model))
    if not getattr(a, "sched_defaults", 0):
        # earlier runs: fixed scheduler settings. --sched-defaults 1 leaves max_num_seqs,
        # max_num_batched_tokens, chunked prefill, max_model_len and gpu_memory_utilization at
        # vLLM's own defaults (same in baseline and inference)
        kw.update(gpu_memory_utilization=a.gpu_mem_util, max_model_len=a.max_model_len,
                  max_num_seqs=a.max_num_seqs)
    if getattr(a, "num_blocks", 0):
        kw["num_gpu_blocks_override"] = a.num_blocks
    if getattr(a, "max_batched_tokens", 0):
        # verifier-mandated per-step token budget: bounds the unchecked activation scratch
        kw["max_num_batched_tokens"] = int(a.max_batched_tokens)
    if os.environ.get("VARENA_RMSNORM", "op") == "op":
        # RMSNorm as vLLM's own op, not an inductor-generated reduction: inductor autotunes the fused
        # add+RMSNorm reduction block size per generated kernel, so any change to the graph around it
        # (e.g. an activation-encoding pass on the residual) changes the reduction order and breaks
        # bit-exactness. VARENA_RMSNORM=inductor restores vLLM's default.
        kw["compilation_config"] = {"custom_ops": ["+rms_norm"]}
    cg_sizes, cg_mode = os.environ.get("VARENA_CG_SIZES"), os.environ.get("VARENA_CG_MODE")
    if cg_sizes or cg_mode:
        # fewer captured CUDA graphs: vLLM's default capture list (51 piecewise + 35 full graphs at max_num_seqs
        # 256, token cap 1024) takes 0.62 GiB, most of it driver memory outside the arena that a cheater without
        # vLLM gets for free. Batch-invariant mode makes outputs independent of the padded batch size, so a
        # shorter list changes padding only. "pow2" = 1, 2, 4, ... up to max_num_seqs.
        cc = kw.setdefault("compilation_config", {})
        if cg_sizes:
            if cg_sizes == "pow2":
                cap = int(getattr(a, "max_num_seqs", 256) or 256)
                cc["cudagraph_capture_sizes"] = [1 << i for i in range(cap.bit_length()) if (1 << i) <= cap]
            else:
                cc["cudagraph_capture_sizes"] = sorted({int(x) for x in cg_sizes.split(",") if x.strip()})
        if cg_mode:
            cc["cudagraph_mode"] = cg_mode               # e.g. FULL_DECODE_ONLY: no piecewise graphs
    if os.environ.get("VARENA_LLM_EXTRA"):          # debugging only (e.g. cudagraph mode)
        kw.update(json.loads(os.environ["VARENA_LLM_EXTRA"]))
    return kw


def add_args(ap):
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-batched-tokens", type=int, default=0,
                    help="per-step token cap (max_num_batched_tokens) for baseline and inference alike; "
                         "0 = vLLM default. The verifier checks the engine runs with it")
    ap.add_argument("--num-prompts", type=int, default=4000)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--num-blocks", type=int, default=0,
                    help="fix the KV block count (same in every run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="random", choices=["random", "sharegpt"],
                    help="random: input-len random tokens, output-len each; sharegpt: vLLM's "
                         "benchmark_throughput sampling of ShareGPT V3 (first turn, reference "
                         "completion length, seed 0), greedy with ignore_eos")
    ap.add_argument("--dataset-path", default="/root/data/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--sched-defaults", type=int, default=0)


_WL = {}


def workload(a):
    """[(prompt token ids, output length)] for this run (cached per process and on disk)."""
    key = (a.dataset, a.num_prompts, a.seed, a.input_len, a.output_len)
    if key in _WL:
        return _WL[key]
    if a.dataset == "random":
        rng = np.random.default_rng(a.seed)
        wl = [(rng.integers(100, 30000, a.input_len).tolist(), a.output_len) for _ in range(a.num_prompts)]
    else:
        cache = f"{a.dataset_path}.{a.num_prompts}.s{a.seed}.ids.json"
        if os.path.exists(cache):
            wl = [tuple(x) for x in json.load(open(cache))]
        else:
            from transformers import AutoTokenizer
            from vllm.benchmarks.datasets.datasets import ShareGPTDataset
            tok = AutoTokenizer.from_pretrained(a.model)
            ds = ShareGPTDataset(dataset_path=a.dataset_path, random_seed=a.seed)
            reqs = ds.sample(tok, num_requests=a.num_prompts)
            wl = [(tok(r.prompt).input_ids, int(r.expected_output_len)) for r in reqs]
            tmp = f"{cache}.{os.getpid()}"
            json.dump(wl, open(tmp, "w"))
            os.replace(tmp, cache)
    _WL[key] = wl
    return wl


def prompts(a):
    return [p for p, _ in workload(a)]


def out_lens(a):
    return [n for _, n in workload(a)]


def sampling(a, idx=None):
    """Greedy, ignore_eos, per-request max_tokens (list aligned with prompts(a) or idx)."""
    from vllm import SamplingParams
    ns = out_lens(a)
    if idx is not None:
        ns = [ns[i] for i in idx]
    return [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in ns]


def sched_info(llm) -> dict:
    """Scheduler / cache values the engine actually uses (logged in every result)."""
    vc = llm.llm_engine.vllm_config
    sc, cc = vc.scheduler_config, vc.cache_config
    return {"max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,
            "chunked_prefill": getattr(sc, "enable_chunked_prefill", getattr(sc, "chunked_prefill_enabled", None)),
            "async_scheduling": sc.async_scheduling, "max_model_len": vc.model_config.max_model_len,
            "gpu_memory_utilization": cc.gpu_memory_utilization, "num_gpu_blocks": cc.num_gpu_blocks,
            "block_size": cc.block_size, "cudagraph_mode": str(vc.compilation_config.cudagraph_mode)}


def prompt_fp(ids) -> list:
    """Request fingerprint for KV claims: (length, crc32 of all prompt tokens, first, last). The first-4-tokens
    fingerprint collided across ShareGPT prompts (common openings), so claims were recomputed against the
    wrong prompt: the honest KV-recompute false positive (12-19% of pages)."""
    import zlib
    import numpy as _np
    a = _np.asarray(list(ids), dtype=_np.int32)
    return [int(a.size), zlib.crc32(a.tobytes()) & 0x7FFFFFFF, int(a[0]) if a.size else -1,
            int(a[-1]) if a.size else -1]
