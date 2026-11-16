"""Arena sized from the honest run's measured peak (the "pad" of free HBM the prover leaves outside the arena).

The prover reports how much of its pad serving used (`pad_study`); the calibration turns that into the next
run's pad and pre-reserve; `--pad-mib auto` applies it. Pure functions (no torch) so they are testable on CPU.

Peak device use after seal is bounded by the larger of the non-torch (driver) use at seal and at the end, plus
torch's peak reserve over the run: non-torch memory (context, lazily loaded modules, graph execs) only grows in
practice, and torch's reserve is tracked exactly by its peak counter. An optional sampled minimum of the
driver's free memory tightens nothing but guards the assumption.
"""
from __future__ import annotations

import json
import math

MiB = 1 << 20


def pad_study(F: dict, end: dict, peak_reserved: int, min_free_sampled: int | None = None,
              arena_bytes: int = 0) -> dict:
    """F: device state right after the arena was sized: free, cap, reserved, allocated (bytes), pad (bytes).
    end: free, cap, reserved at the end of the run. peak_reserved: torch's peak reserve since serving began."""
    peak_res = max(int(peak_reserved), int(end["reserved"]))
    nt_F = (F["cap"] - F["free"]) - F["reserved"]
    nt_end = (end["cap"] - end["free"]) - end["reserved"]
    peak_used = max(nt_F, nt_end) + peak_res
    if min_free_sampled is not None:
        peak_used = max(peak_used, end["cap"] - int(min_free_sampled))
    used_F = F["cap"] - F["free"]
    r = lambda x: round(x / MiB, 1)
    return {"pad_mib": F["pad"] // MiB, "free_at_seal_mib": r(F["free"]),
            "pad_used_mib": r(peak_used - used_F),
            "pad_torch_growth_mib": r(max(0, peak_res - F["reserved"])),
            "pad_nontorch_growth_mib": r(max(0, nt_end - nt_F)),
            "min_free_mib": r(end["cap"] - peak_used), "free_end_mib": r(end["free"]),
            "nontorch_seal_mib": r(nt_F), "nontorch_end_mib": r(nt_end),
            "torch_reserved_seal_mib": r(F["reserved"]), "torch_peak_reserved_mib": r(peak_res),
            "torch_alloc_outside_arena_seal_mib": r(F.get("allocated", 0) - arena_bytes),
            "sampled": min_free_sampled is not None}


def pad_calib(stats: dict | None, a) -> dict:
    """Calibration fields for the next run's pad, from the prover's pad study (empty if it has none):
    pad = used x (1 + frac) + safety; pre-reserve = torch's growth (pre-grown at seal, so no cudaMalloc mid-run);
    the driver-side growth (+ the same safety) is kept free of the pre-reserve."""
    ps = (stats or {}).get("pad_study") or {}
    if "pad_used_mib" not in ps:
        return {}
    used = max(0.0, float(ps["pad_used_mib"]))
    nt = max(0.0, float(ps.get("pad_nontorch_growth_mib", 0)))
    tg = max(0.0, float(ps.get("pad_torch_growth_mib", 0)))
    return {"pad_mib_run": int(a.pad_mib), "pad_used_mib": used,
            "pad_mib_cal": int(math.ceil(used * (1 + a.pad_safety_frac) + a.pad_safety_mib)),
            "prereserve_mib_cal": int(math.ceil(tg)),
            "pad_nontorch_mib_cal": int(math.ceil(nt * (1 + a.pad_safety_frac) + a.pad_safety_mib)),
            "pad_min_free_mib": ps.get("min_free_mib")}


def resolve_pad(a, calib: dict | None) -> None:
    """--pad-mib auto: the arena is sized from the honest calibration's measured peak (pad_mib_cal), and torch's
    cache is pre-grown by its measured torch-side growth (prereserve_mib_cal), leaving the driver-side growth free.
    Sets a.pad_mib (int), a.prereserve_mib and a.pad_nontorch_mib (None unless auto)."""
    a.prereserve_mib = a.pad_nontorch_mib = None
    if str(a.pad_mib).lower() != "auto":
        a.pad_mib = int(a.pad_mib)
        return
    src = json.load(open(a.pad_calib)) if getattr(a, "pad_calib", None) else calib
    if not src or "pad_mib_cal" not in src:
        raise SystemExit("--pad-mib auto needs a calibration with pad_mib_cal (--pad-calib or --calib)")
    a.pad_mib = int(src["pad_mib_cal"])
    if src.get("prereserve_mib_cal") is not None:
        a.prereserve_mib = int(src["prereserve_mib_cal"])
        a.pad_nontorch_mib = int(src.get("pad_nontorch_mib_cal", 64))
    print(f"[verifier] pad auto: {a.pad_mib} MiB (pre-reserve {a.prereserve_mib} MiB) from "
          f"{getattr(a, 'pad_calib', None) or getattr(a, 'calib', None)}", flush=True)
