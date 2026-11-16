"""Reference engine for the verifier (runs in its own process, plain weights/KV).

  python -m schemes.memory_link_exhaustion.verifier.reference job.json out.json

For every settled K-page claim: prefill the claimed token prefix (max_tokens=1)
with deterministic vLLM, find the block that holds the claimed block index,
and return that layer's raw 64 KiB block (the verifier encodes it with the
claimed page's key and offset and compares digests). Also re-generates a
sample of the served prompts to compare output tokens, and for every
mutant_every-th claim recomputes a prefix with its last token changed (the
page must then differ: a match isn't automatic).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from baselines.inference import common


def main():
    job_path, out_path = sys.argv[1], sys.argv[2]
    job = json.load(open(job_path))
    common.apply_env({"VARENA": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.core import block_pool as BPm
    import torch

    allocs = []
    orig = BPm.BlockPool.get_new_blocks

    def get_new(self, n):
        out = orig(self, n)
        allocs.extend(b.block_id for b in out)
        return out

    BPm.BlockPool.get_new_blocks = get_new
    a = argparse.Namespace(model=job["model"], **job["args"])
    llm = LLM(**common.llm_kwargs(a))
    vc = llm.llm_engine.vllm_config
    ctx = vc.compilation_config.static_forward_context
    from vllm.model_executor.models.utils import extract_layer_index
    kvs = {}
    for name, mod in ctx.items():
        kv = getattr(mod, "kv_cache", None)
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        if isinstance(kv, torch.Tensor) and kv.numel():
            kvs[extract_layer_index(name)] = kv
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=list(range(100, 612)))], sp1, use_tqdm=False)

    def block_bytes(layer, block):
        kv = kvs[layer]
        t = torch.as_strided(kv, (kv.stride(0),), (1,), kv.storage_offset() + block * kv.stride(0))
        return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()

    def run_prefix(tokens):
        del allocs[:]
        llm.generate([TokensPrompt(prompt_token_ids=tokens)], sp1, use_tqdm=False)
        torch.cuda.synchronize()
        return list(allocs)

    t0 = time.time()
    cache = {}
    pages = []
    mut = {"n": 0, "identical": 0}
    for k, c in enumerate(job["claims"]):
        key = tuple(c["tokens"])
        if key not in cache:
            cache = {key: run_prefix(c["tokens"])}     # keep only the latest prefix's blocks
        blocks = cache[key]
        idx = c["block_index"]
        if idx >= len(blocks):
            pages.append(None)
            continue
        raw = block_bytes(c["layer"], blocks[idx])
        pages.append(raw.hex())
        if job.get("mutant_every") and k % job["mutant_every"] == 0:
            mt = list(c["tokens"])
            mt[-1] = mt[-1] + 1 if mt[-1] < 30000 else mt[-1] - 1
            mb = run_prefix(mt)
            cache = {}
            if idx < len(mb):
                mut["n"] += 1
                mut["identical"] += int(block_bytes(c["layer"], mb[idx]) == raw)
    t_claims = time.time() - t0
    ks = job.get("kv_store")
    n_store = 0
    if ks:
        # settled-KV replay: every request once (its prompt and the tokens the prover returned, as one prefix), all
        # of its blocks in every layer, raw, to <dir>/req<i>.bin as [block index][layer] x one 64 KiB block
        os.makedirs(ks["dir"], exist_ok=True)
        layers = sorted(kvs)
        bs_ = int(ks["block_tokens"])
        for ent in ks["requests"]:
            blocks = run_prefix(ent["tokens"])
            nbk = min(len(blocks), -(-len(ent["tokens"]) // bs_))
            with open(os.path.join(ks["dir"], f"req{ent['i']}.bin"), "wb") as f:
                for bi in blocks[:nbk]:
                    for l in layers:
                        f.write(block_bytes(l, bi))
            n_store += 1
        print("KV_STORE", n_store, "requests in", round(time.time() - t0 - t_claims, 1), "s", flush=True)
    prompts = common.prompts(a)
    ids = job["outputs"]
    sp = common.sampling(a, ids)
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in ids], sp, use_tqdm=False)
    json.dump({"pages": pages, "outputs": [list(o.outputs[0].token_ids) for o in outs],
               "mutants": mut, "claims_s": t_claims, "kv_store": n_store}, open(out_path, "w"))
    print("REFERENCE_DONE", len(pages), mut, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
