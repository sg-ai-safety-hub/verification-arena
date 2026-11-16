"""Verifier-side block table: replay the scheduler, derive the table, compare digests.

The prover's block table (inference/kvtable.py) is a function of the engine scheduler's state after each schedule()
call. The verifier knows the requests and the returned tokens; what it does not know is the order in which the
engine's calls happened (which requests had arrived before a schedule() call, which steps had finished). With that
order (VARENA_SCHED_TRACE, inference/plugin.py) the verifier runs the engine's own Scheduler class on the CPU, with
the same arena patches (block frees), and derives the table with its own copy of the rule (Deriver below, written
from the rule, not shared with the prover). The digest of the derived table at version v must equal the digest the
prover returned inside a timed answer for version v. Then nothing in a table is a claim: the prover only says where
in the call order an answer was computed.

    python -m schemes.memory_link_exhaustion.verifier.schedreplay <trace> --llm-kwargs <json> [--lag 2] [--kvtab <log>]

Stage 1 (this file): the engine configuration comes from the verifier's own settings; the trace carries, as plain
values, the KV cache layout, each request (id, prompt tokens, output length), the tokens each step sampled and the
order of the calls. The request contents and sampled tokens are things the verifier already has (its prompts, the
returned outputs) and can check against the trace; the call order stays the only prover-supplied input.
"""
from __future__ import annotations

import os
import pickle
from collections import deque



def read_trace(path):
    ev = []
    with open(path, "rb") as f:
        while True:
            try:
                ev.append(pickle.load(f))
            except EOFError:
                break
            except Exception as e:                       # a record cut off by the end of the run
                ev.append(("ERR", "read", f"{type(e).__name__}: {e}"))
                break
    return ev


class Deriver:
    """The table rule, from scheduler state only (no GPU, no prover code): what inference/prover.py does in
    _on_schedule (snapshot), _stale_owners / _on_free (stale-KV route) and on_scheduled."""

    def __init__(self, lag: int, tight: bool):
        self.lag, self.tight = int(lag), bool(tight)
        self.sealed = False
        self.kvt = None
        self.snaps = deque(maxlen=256)
        self.stale_owner = {}
        self.fresh, self.freed = set(), {}
        self.hist = deque(maxlen=self.lag + 1)
        self.prev = {}
        self.fp = {}
        self.digests = {}                               # version -> digest

    def start_table(self, n_blocks: int, block_tokens: int):
        from schemes.memory_link_exhaustion.inference import kvtable as KT
        self.kvt = KT.KVTable(n_blocks, block_tokens, tight=self.tight)
        self.fresh, self.freed = set(), {}

    def _fp(self, req):
        from baselines.inference.common import prompt_fp
        fp = self.fp.get(req.request_id)
        if fp is None:
            f = prompt_fp(req.prompt_token_ids)
            fp = self.fp[req.request_id] = (int(f[0]) << 32) | int(f[1])
        return fp

    # --- hooks the arena's scheduler patches call (inference/plugin.py _patch_scheduler)
    def on_schedule(self, sched):
        if not self.sealed:
            return
        blocks = sched._varena_blocks
        snap = {}
        for req in sched.running:
            bl = blocks.get(req.request_id)
            if bl is not None:
                # (the engine's own list object, as the prover keeps it: at a free it shows the blocks the request
                # held then, not only those it held at this snapshot)
                snap[req.request_id] = (req.num_computed_tokens - req.num_in_flight_tokens, req, bl)
        self.snaps.append(snap)

    def on_alloc(self, blocks):
        if self.sealed and self.kvt is not None:
            self.fresh.update(b.block_id for b in blocks)

    def on_free(self, pool, blocks):
        ids = [b.block_id for b in blocks]
        want = set(ids)
        if self.snaps:
            snap = self.snaps[-1]
            old = self.snaps[-1 - self.lag] if len(self.snaps) > self.lag else {}
            comp = {}
            for rid, (settled, req, bl) in snap.items():
                hit = [i for i, x in enumerate(bl) if x.block_id in want]
                for x in bl:
                    comp[x.block_id] = int(req.num_computed_tokens)
                if not hit:
                    continue
                o = old.get(rid) if self.lag else (settled,)
                st_old = min(settled, o[0]) if o is not None else 0
                fp = self._fp(req)
                for i in hit:
                    self.stale_owner[bl[i].block_id] = (fp, i, st_old)
                    want.discard(bl[i].block_id)
            for b in want:
                self.stale_owner[b] = None
            if self.kvt is not None:
                for b in ids:
                    o = self.stale_owner.get(b)
                    self.freed[b] = (o[0], o[1], o[2], comp.get(b, o[2])) if o is not None else None
        pool.free_block_queue.append_n(blocks)

    def on_scheduled(self, sched):
        if self.kvt is None or not self.sealed:
            return
        blocks = sched._varena_blocks
        now = {r.request_id: r.num_computed_tokens - r.num_in_flight_tokens for r in sched.running}
        self.hist.append(now)
        old = self.hist[0] if len(self.hist) > self.lag else {}
        live, rid_of = {}, {}
        for r in sched.running:
            bl = blocks.get(r.request_id)
            if bl is None:
                continue
            st = max(0, min(now[r.request_id], old.get(r.request_id, 0)))
            fp = self._fp(r)
            for i, x in enumerate(bl):
                live[x.block_id] = (fp, i, st)
                rid_of[x.block_id] = r.request_id
        kt = self.kvt
        if kt.tight:
            comp = {r.request_id: int(r.num_computed_tokens) for r in sched.running}
            live_t = {}
            for b, (fp, i, st) in live.items():
                rid = rid_of[b]
                if i >= self.prev.get(rid, 0) // kt.bs:
                    live_t[b] = (fp, i, st, max(st, comp[rid]))
            self.prev = {rid_of[b]: st for b, (fp, i, st) in live.items()}
            freed_t = {b: ((o[0], o[1], o[3]) if o is not None else None) for b, o in self.freed.items()}
            self.fresh, self.freed = set(), {}
            kt.update_tight(live_t, freed_t)
        else:
            fresh = [b for b in self.fresh if b not in live]
            freed = {b: ((o[0], o[1], o[2]) if o is not None else None) for b, o in self.freed.items() if b not in live}
            self.fresh, self.freed = set(), {}
            kt.update(live, fresh, freed)
        self.digests[int(kt.version)] = int(kt.digest())


def _plain(v):
    return not (isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == "repr")


def build_scheduler(init: dict, llm_kwargs: dict, log=print):
    """The engine's Scheduler, built by the verifier: the engine configuration from the verifier's own settings
    (llm_kwargs, what it passed to the engine), the KV cache layout from the trace's plain description (block count,
    layers, block size). Returns (scheduler, differences between its scheduler settings and the trace's)."""
    import dataclasses
    import importlib
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    names = {f.name for f in dataclasses.fields(EngineArgs)}
    dropped = sorted(k for k in llm_kwargs if k not in names)
    if dropped:
        log(f"schedreplay: engine settings not taken by EngineArgs: {dropped}")
    vcfg = EngineArgs(**{k: v for k, v in llm_kwargs.items() if k in names}).create_engine_config()
    # the engine fills this in after sizing the KV cache; the verifier's own block count (--num-blocks) is the same number
    vcfg.cache_config.num_gpu_blocks = int(init["kv"]["num_blocks"])
    groups = []
    for g in init["kv"]["groups"]:
        cls = getattr(importlib.import_module(g["spec_module"]), g["spec_cls"])
        ok = {f.name for f in dataclasses.fields(cls) if f.init}
        kw = {}
        for k, v in g["spec"].items():
            if k not in ok:
                continue
            if _plain(v):
                kw[k] = v
            elif str(v[1]).startswith("torch."):
                kw[k] = getattr(torch, str(v[1])[6:])
        gk = {k: v for k, v in g.get("group", {}).items() if _plain(v)}
        groups.append(KVCacheGroupSpec(layer_names=list(g["layer_names"]), kv_cache_spec=cls(**kw), **gk))
    kvc = KVCacheConfig(num_blocks=int(init["kv"]["num_blocks"]), kv_cache_tensors=[], kv_cache_groups=groups)
    try:
        from vllm.v1.structured_output import StructuredOutputManager
        som = StructuredOutputManager(vcfg)
    except Exception as e:
        from unittest import mock
        log(f"schedreplay: structured output manager stubbed ({type(e).__name__}: {e})")
        som = mock.MagicMock()
    cls = getattr(importlib.import_module(init["module"]), init["cls"])
    kw = {"block_size": int(init["block_size"]), "include_finished_set": bool(init["include_finished_set"]),
          "log_stats": bool(init["log_stats"])}
    if _plain(init.get("hash_block_size")) and init.get("hash_block_size") is not None:
        kw["hash_block_size"] = int(init["hash_block_size"])
    sched = cls(vllm_config=vcfg, kv_cache_config=kvc, structured_output_manager=som, **kw)
    diff = {}
    for part in ("scheduler_config", "cache_config"):
        mine = getattr(sched, part)
        for k, v in (init.get(part) or {}).items():
            if _plain(v) and getattr(mine, k, v) != v:
                diff[f"{part}.{k}"] = [v, getattr(mine, k, None)]
    return sched, diff


def _dump(f, body: bytes):
    """One table in the block-table log format (kvtable.read_log): uint32 length + zlib(body)."""
    import zlib
    z = zlib.compress(body, 1)
    f.write(len(z).to_bytes(4, "little") + z)


def expected_submissions(exp: dict):
    """What the verifier itself submitted to the engine, as a multiset {(prompt tokens, max_tokens): count}, and the
    output it knows for each prompt. exp: {"prompts": [[ids]], "out_lens": [n], "first": k, "warmup": w, "selftest":
    bool, "tokens": [[ids]]} = the verifier's calls (verifier.py run): the warm-up of the first w prompts at
    min(32, n), every prompt at its own length, and the self-test's four. "first": k (default 0) adds the first k
    prompts at their own lengths, for traces from before 16 Nov, when the verifier ran an 8-request pre-run first."""
    from collections import Counter
    pr = [tuple(int(t) for t in p) for p in exp["prompts"]]
    ol = [int(n) for n in exp["out_lens"]]
    c = Counter()
    for i in range(min(int(exp.get("first", 0)), len(pr))):
        c[(pr[i], ol[i])] += 1
    for i in range(min(int(exp.get("warmup", 0)), len(pr))):
        c[(pr[i], max(1, min(32, ol[i])))] += 1
    for i in range(len(pr)):
        c[(pr[i], ol[i])] += 1
    if exp.get("selftest"):
        for i in range(min(4, len(pr))):
            c[(pr[i], max(1, min(8, ol[i])))] += 1
    known = {}
    for i, t in enumerate(exp.get("tokens") or []):
        if i < len(pr) and len(t) >= len(known.get(pr[i], ())):
            known[pr[i]] = [int(x) for x in t]
    return c, known


def replay(path: str, llm_kwargs: dict, lag: int = 2, tight: bool = True, keep_tables: bool = False, log=print,
           dump: str = "", expect: dict | None = None):
    """Run the trace through the engine's Scheduler. Returns {"digests": {version: digest}, "tables": {version:
    KVTable} if keep_tables, "steps": n, "sched_mismatch": first schedule() call whose scheduled tokens differ from
    the trace's or None, "trace_digest_mismatches": versions whose derived digest differs from the one the trace
    recorded (the prover's own), "config_diff": scheduler settings that differ, "errors": [...]}."""
    from schemes.memory_link_exhaustion.inference import plugin, prover
    ev = read_trace(path)
    errs = [e for e in ev if e and e[0] == "ERR"]
    init = next(e for e in ev if e[0] == "I")[1]
    os.environ.pop("VARENA_SCHED_TRACE", None)
    plugin._patch_scheduler()
    der = Deriver(lag, tight)
    prover.STATE = der
    sched, diff = build_scheduler(init, llm_kwargs, log)
    from vllm import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    outs, tables = {}, {}
    out = {"steps": 0, "adds": 0, "updates": 0, "sched_mismatch": None, "trace_digest_mismatch": [],
           "trace_digest_mismatches": 0, "errors": errs[:5], "trace_versions": 0, "config_diff": diff}
    nb, bs = int(init["kv"]["num_blocks"]), int(init["block_size"])
    df = open(dump, "wb") if dump else None       # every derived table, for the verifier's chain replay and checks
    # the trace's requests and sampled tokens against what the verifier itself sent and received (expect): after
    # this the only thing taken from the prover is the order of the calls
    want, known = expected_submissions(expect) if expect else (None, None)
    reqs = {}
    if want is not None:
        out.update({"requests_unknown": 0, "requests_missing": 0, "tokens_bad": 0, "tokens_checked": 0})
    for e in ev:
        k = e[0]
        if k in ("I", "ERR"):
            continue
        der.sealed = bool(e[1][0])
        if k == "A":
            d = e[2]
            if want is not None:
                key = (tuple(int(t) for t in d["prompt_token_ids"]), int(d["max_tokens"]))
                if want.get(key, 0) > 0:
                    want[key] -= 1
                else:
                    out["requests_unknown"] += 1              # not a request of this verifier (or one too many)
            rq = Request(
                request_id=d["request_id"], prompt_token_ids=list(d["prompt_token_ids"]),
                sampling_params=SamplingParams(temperature=0.0, max_tokens=int(d["max_tokens"]), ignore_eos=True),
                pooling_params=None, client_index=int(d.get("client_index") or 0),
                arrival_time=d.get("arrival_time"), priority=int(d.get("priority") or 0))
            reqs[d["request_id"]] = rq
            sched.add_request(rq)
            out["adds"] += 1
        elif k == "S":
            _, st0, idx, (a, kwa), st1, nst = e
            if st0[1] and der.kvt is None:
                der.start_table(nb, bs)
                if df is not None:
                    _dump(df, der.kvt.body())                      # version 0: the sealed table
            v_before = None if der.kvt is None else int(der.kvt.version)
            so = sched.schedule(*a, **kwa)
            outs[idx] = so
            out["steps"] += 1
            if df is not None and der.kvt is not None and int(der.kvt.version) != v_before:
                _dump(df, der.kvt.body())
            got = {str(r): int(n) for r, n in so.num_scheduled_tokens.items()}
            if got != nst:
                # the replayed scheduler decided differently from the engine: everything after this call would be a
                # different history, so the derivation stops here and says where and how they differ
                d = sorted(set(got) ^ set(nst)) + sorted(r for r in got if r in nst and got[r] != nst[r])
                rq0 = sched.requests.get(d[0]) if d else None
                out["sched_mismatch"] = {
                    "call": idx, "trace_requests": len(nst), "replay_requests": len(got),
                    "trace_tokens": sum(nst.values()), "replay_tokens": sum(got.values()),
                    "differ": [(r, nst.get(r), got.get(r)) for r in d[:6]],
                    "first": None if rq0 is None else {
                        k: str(getattr(rq0, k, None))[:60] for k in ("status", "num_tokens", "num_prompt_tokens",
                        "num_computed_tokens", "num_in_flight_tokens", "num_output_placeholders", "max_tokens",
                        "num_stale_output_tokens", "drop_stale_output", "arrival_time", "priority")},
                    "running": len(sched.running), "waiting": len(sched.waiting), "state": [st0, st1]}
                break
            if st1[2] is not None and st1[3] is not None and st1[2] > 0:
                out["trace_versions"] += 1
                v = int(st1[2])
                if der.kvt is None or int(der.kvt.version) != v or der.digests.get(v) != int(st1[3]):
                    if len(out["trace_digest_mismatch"]) < 8:
                        out["trace_digest_mismatch"].append(
                            {"call": idx, "version_trace": v, "version_replay": None if der.kvt is None else int(der.kvt.version)})
                    out["trace_digest_mismatches"] += 1
                if keep_tables and der.kvt is not None:
                    from schemes.memory_link_exhaustion.inference import kvtable as KT
                    tables[int(der.kvt.version)] = KT.parse(der.kvt.body())
        elif k == "U":
            _, st, idx, f = e
            so = outs.pop(idx, None)
            if so is None:
                # an update the trace cannot place: the history from here on is unknown, never a pass
                out["errors"].append(f"update for unknown schedule call {idx}")
                break
            ids = list(f["req_ids"])
            sched.update_from_output(so, ModelRunnerOutput(
                req_ids=ids, req_id_to_index=dict(f.get("req_id_to_index") or {r: i for i, r in enumerate(ids)}),
                sampled_token_ids=[list(x) for x in f["sampled_token_ids"]]))
            out["updates"] += 1
        elif k == "F":
            _, st, ids, status = e
            sched.finish_requests(ids, RequestStatus(int(status)))
    if df is not None:
        df.close()
    if want is not None:
        out["requests_missing"] = int(sum(want.values()))       # submitted by the verifier, absent from the trace
        for rid, rq in reqs.items():
            ref_t = known.get(tuple(int(t) for t in rq.prompt_token_ids))
            if ref_t is None:
                continue
            if not hasattr(rq, "output_token_ids"):
                out["tokens_bad"] += 1                           # (cannot read the accepted tokens: never a pass)
                continue
            got_t = [int(t) for t in rq.output_token_ids]
            out["tokens_checked"] += 1
            # (the tokens the scheduler accepted for the request: a prefix of the output the verifier received for
            # that prompt, and complete unless the session ended first)
            if got_t != ref_t[:len(got_t)] or len(got_t) > int(rq.max_tokens):
                out["tokens_bad"] += 1
    out["digests"] = der.digests
    if keep_tables:
        out["tables"] = tables
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--lag", type=int, default=int(os.environ.get("VARENA_SETTLE_LAG", "0")))
    ap.add_argument("--tight", type=int, default=1)
    ap.add_argument("--kvtab", default="", help="the prover's table log: compare every logged table's digest too")
    ap.add_argument("--llm-kwargs", required=True, help="JSON file: the engine settings the verifier asked for")
    ap.add_argument("--dump", default="", help="write every derived table to this file (block-table log format)")
    ap.add_argument("--expect", default="", help="JSON: the verifier's own prompts, output lengths, warm-up settings "
                    "and received tokens; the trace's requests and sampled tokens are checked against them")
    a = ap.parse_args()
    r = replay(a.trace, json.load(open(a.llm_kwargs)), a.lag, bool(a.tight), dump=a.dump,
               expect=json.load(open(a.expect)) if a.expect else None)
    d = r.pop("digests")
    rep = {k: v for k, v in r.items()}
    rep["derived_versions"] = len(d)
    if a.kvtab:
        from schemes.memory_link_exhaustion.inference import kvtable as KT
        bodies = KT.read_log(a.kvtab)
        same = sum(1 for v, b in bodies.items() if d.get(v) == KT.body_digest(b))
        rep["kvtab"] = {"tables": len(bodies), "digest_equal": same}
    print(json.dumps(rep, default=str))


if __name__ == "__main__":
    main()
