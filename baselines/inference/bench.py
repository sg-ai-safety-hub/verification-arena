"""Baseline inference: deterministic vLLM, no verification (workload 1).

  python -m baselines.inference.bench --model DIR --tag base --out results.jsonl --save-tokens t.json

Same prompts, settings and kernels as inference under the verifier
(schemes.memory_link_exhaustion.verifier.verifier), which must reproduce these tokens bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from baselines.inference import common


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-tokens", default=None)
    ap.add_argument("--stock", action="store_true",
                    help="plain vLLM defaults (no batch invariance, default attention)")
    a = ap.parse_args()
    if not a.stock:
        common.apply_env({"VARENA": "0"})
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    t0 = time.time()
    kw = common.llm_kwargs(a)
    if a.stock:
        for k in ("attention_backend", "block_size", "compilation_config"):
            kw.pop(k, None)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR") and os.environ.get("VARENA_BENCH_PROF_N"):
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": os.environ["VLLM_TORCH_PROFILER_DIR"]}
    llm = LLM(**kw)
    startup = time.time() - t0
    ps = [TokensPrompt(prompt_token_ids=p) for p in common.prompts(a)]
    sp = common.sampling(a)
    llm.generate(ps[:8], sp[:8], use_tqdm=False)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR") and os.environ.get("VARENA_BENCH_PROF_N"):
        # kernel-time breakdown: a profiled slice of the workload (vLLM's own torch profiler in the engine process)
        n_ = int(os.environ["VARENA_BENCH_PROF_N"])
        llm.start_profile()
        llm.generate(ps[:n_], sp[:n_], use_tqdm=False)
        llm.stop_profile()
    t0 = time.time()
    outs = llm.generate(ps, sp, use_tqdm=False)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    vc = llm.llm_engine.vllm_config
    rec = {"tag": a.tag, "workload": "baseline_inference", "stock": a.stock, "tok_s": n / dt,
           "elapsed_s": dt, "output_tokens": n, "startup_s": startup,
           "num_blocks": vc.cache_config.num_gpu_blocks, "block_size": vc.cache_config.block_size,
           "async_scheduling": vc.scheduler_config.async_scheduling,
           "compile_mode": str(vc.compilation_config.mode),
           "cudagraph_mode": str(vc.compilation_config.cudagraph_mode),
           "sched": common.sched_info(llm), "args": vars(a)}
    print("RESULT", json.dumps(rec), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if a.save_tokens:
        json.dump([list(o.outputs[0].token_ids) for o in outs], open(a.save_tokens, "w"))
    os._exit(0)


if __name__ == "__main__":
    main()
