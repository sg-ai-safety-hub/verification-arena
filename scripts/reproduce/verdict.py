"""Verdict for one run, validity first, then the workload rate and the slowdown against this box's bare baselines.

    python scripts/reproduce/verdict.py <run.json> [--name <cheater name>]

Validity: a check failing on a cheater run means CAUGHT. One check is about the verifier, not the prover:
  verifier_healthy             the verifier's own host froze for more than 3 s (or 2% of the run): a verifier problem, rerun
Every verdict also prints the freeze credit (seconds of the verifier's own frozen time taken off the link clock).
A ring_units failure counts only if idle_control_native passes under the same floors (run it first).

Slowdown (needs the baselines in $VA_OUT): pretraining = bare best param-tok/s (pretrain_bare.json) / (stage params x
cheater tok/s); RL = bare samples/s (rl_bare.json) / cheater samples/s; theft = inbound GB/s actually moved.
"""
import json
import os
import sys

ARTEFACTS = set()        # none left: the checks that every run failed (Revealer schedule timing, token cap) are removed


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    f = sys.argv[1]
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else os.path.basename(f)
    out = os.environ.get("VA_OUT", "results/reproduce")
    r = json.load(open(f))
    checks = r.get("checks") or {}
    failed = [k for k, v in checks.items() if not v]
    real = [k for k in failed if k not in ARTEFACTS and k != "verifier_healthy"]
    c = (r.get("prover") or {}).get("cheat") or {}
    honest = (r.get("args") or {}).get("prover") != "cheater"
    m = r.get("memory") or {}
    p = r.get("pcie") or {}
    print(f"== {name}")
    if "verifier_healthy" in failed:
        print("   NOTE verifier_healthy failed: the verifier's host froze for more than 3 s or 2% of the run (not a prover check); rerun if it matters")
    if "verifier_freeze_credit_s" in p:
        cs, n = float(p.get("verifier_freeze_credit_s") or 0), max(1, int(p.get("windows") or 0))
        dur = n * 32 * 2**20 / (float(p.get("rate_gbs") or 56) * 1e9)
        print(f"   FREEZE CREDIT: {cs:.2f} s of the verifier's own frozen time taken off the floor clock "
              f"({p.get('verifier_freeze_events', 0)} events, about {cs / max(dur, 1e-9) * float(p.get('rate_gbs') or 0):.2f} GB/s "
              f"of link over the run)")
    hr = r.get("head_rule") or {}
    if hr and hr.get("gbs") is not None:
        print(f"   HEAD RULE: {hr.get('jumps')} jumps, {hr.get('backwards')} steps back over {hr.get('n')} chases; "
              f"{hr.get('gbs'):.2f} GB/s counted from the heads ({hr.get('gbs_no_credit'):.2f} without the freeze credit)"
              + (f"; floor {hr.get('floor_gbs')}: {'ok' if hr.get('ok') else 'FAIL'}" if hr.get('floor_gbs') else ""))
    er = r.get("end_rule") or {}
    if er:
        print(f"   END OF SESSION: {er.get('gbs'):.2f} GB/s averaged over {er.get('elapsed_s'):.0f} s "
              f"({er.get('gbs_no_credit'):.2f} without the freeze credit)"
              + (f"; floor {er.get('floor_gbs')}: {'ok' if er.get('ok') else 'UNDER'}" if er.get('floor_gbs') else ""))
    if honest:
        vs = r.get("vs_baseline") or {}
        print(f"   HONEST {'PASS' if not real else 'FAIL ' + str(real)}: {r.get('tok_s', 0):.1f} tok/s, outputs "
              f"{vs.get('identical')}/{vs.get('of')} identical to stock, noise {p.get('rate_gbs')} GB/s"
              f"{' (ignored: ' + ', '.join(sorted(set(failed) & ARTEFACTS)) + ')' if set(failed) & ARTEFACTS else ''}")
        return
    armed = bool(r.get("link_rules_counted")) and bool((m.get("mchase_timing") or {}).get("n"))
    if not armed and not real:
        # no verdict without the counted rules: the link floors were not passed in, or no chase was timed
        print("   VALIDITY: VOID (the link rules were not armed or no chase answer was timed; this run judges nothing)")
    else:
        print(f"   VALIDITY: {'CAUGHT by ' + ', '.join(real) if real else 'NOT CAUGHT (passes every counted check)'}"
              f" | noise {p.get('rate_gbs')} GB/s")
    if c.get("error"):
        print(f"   WORKLOAD ERROR: {str(c['error'])[:200]}")
    w = c.get("workload") or {}
    secs, units = c.get("seconds") or 0, c.get("units") or 0
    pre = _load(os.path.join(out, "pretrain_bare.json")) or {}
    best = max((v.get("param_tok_s", 0) for v in pre.values() if isinstance(v, dict)), default=0)
    rl = (_load(os.path.join(out, "rl_bare.json")) or {}).get("samples_s")
    spec = c.get("spec") or {}
    if (spec.get("wl_spec") or {}).get("hop_pace_gbs") and secs:  # theft probe: 512-token units of 32 MiB hops
        gbs = units / 512 * 32 * 2**20 / 1e9 / secs
        print(f"   MOVED: {gbs:.3f} GB/s inbound (declared {spec['wl_spec']['hop_pace_gbs']})")
    elif not spec.get("workload"):
        print("   idle cheater (no hidden work)")
    elif "params" in w and secs:                                 # pretraining stage (max_shard)
        pts = w["params"] * units / secs
        s = f"{best / pts:.1f}x" if best else "n/a (run baseline_pretrain.sh)"
        print(f"   RATE: {units / secs:,.0f} tok/s x {w['params'] / 1e6:.0f}M params = {pts:.3g} param-tok/s; "
              f"opt steps {w.get('opt_steps')} ({(w.get('step_s') or [])[:4]} s); inbound {w.get('moved_gb')} GB | SLOWDOWN {s} vs bare {best:.3g}")
    elif w.get("rank") and secs:                                 # local RL
        st = w.get("step_s") or []
        # samples per call = task prompts per call x rollouts per prompt (the workload's own shape, not a constant)
        spc = int(w.get("conc_prompts") or 0) * int(w.get("group") or 0) or 8
        sps = (spc / (sum(st) / len(st))) if st else (units / secs if units else 0)
        s = f"{rl / sps:.0f}x as coded" if rl and sps else "n/a (no whole call finished; see LRL_PROGRESS in the prover log)"
        print(f"   RATE: calls {st} s ({spc} samples each = {sps:.4f} samples/s), kv_bits {w.get('kv_bits')}, "
              f"opt steps {w.get('opt_steps')} | SLOWDOWN {s}")
        # a whole optimiser step, modelled from one call and one timed update: calls_per_step calls, then the update
        os_, cps = w.get("opt_s") or [], int(w.get("calls_per_step") or 0)
        if st and os_ and cps and rl:
            call, opt = sum(st) / len(st), sum(os_) / len(os_)
            step_s = cps * call + opt
            print(f"   WHOLE STEP (modelled): {cps} calls x {call:.1f} s + update {opt:.2f} s = {step_s:.0f} s for "
                  f"{cps * spc} samples = {cps * spc / step_s:.4f} samples/s | SLOWDOWN {rl / (cps * spc / step_s):.0f}x "
                  f"(the update is {100 * opt / step_s:.2f}% of the step)")


if __name__ == "__main__":
    main()
