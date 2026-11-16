"""CPU tests for the arena sized from the measured peak (verifier/pad.py). Run: python -m pytest, or directly."""
import json
import os
import tempfile
from types import SimpleNamespace

from schemes.memory_link_exhaustion.verifier.pad import pad_calib, pad_study, resolve_pad

GiB, MiB = 1 << 30, 1 << 20
CAP = 150156607488                          # H200 NVL, a reference calibration run


def reference_run():
    """Reference run: seal free 1052442624 B with a 1 GiB pad, torch reserved 147568197632 B; end of run
    device used 139.0946 GiB, torch reserved 137.6621 GiB, torch peak reserve 147813564416 B."""
    F = {"free": 1052442624, "cap": CAP, "reserved": 147568197632, "allocated": 0, "pad": 1024 * MiB}
    end = {"free": CAP - int(139.0946044921875 * GiB), "cap": CAP, "reserved": int(137.662109375 * GiB)}
    return F, end, 147813564416


def test_b21_pad_use():
    F, end, peak = reference_run()
    s = pad_study(F, end, peak)
    # of the 0.98 GiB left free at seal, serving used ~236 MiB (torch's reserve +234, driver ~+2); 0.75 GiB
    # was never used
    assert 230 < s["pad_used_mib"] < 240, s
    assert 230 < s["pad_torch_growth_mib"] < 238, s
    assert s["pad_nontorch_growth_mib"] < 5, s
    assert abs(s["min_free_mib"] - s["free_end_mib"]) < 1, s       # peak at the end in that run
    assert abs(s["min_free_mib"] + s["pad_used_mib"] - s["free_at_seal_mib"]) < 0.5, s


def test_peak_reserve_counts_even_if_released():
    F, end, peak = reference_run()
    s0 = pad_study(F, end, peak)
    s1 = pad_study(F, end, peak + 300 * MiB)                       # a transient torch peak, released by the end
    assert abs(s1["pad_used_mib"] - s0["pad_used_mib"] - 300) < 0.2
    s2 = pad_study(F, end, peak, min_free_sampled=end["free"] - 500 * MiB)  # sampled dip below the bound
    assert abs(s2["pad_used_mib"] - s0["pad_used_mib"] - 500) < 0.2


def _args(**kw):
    a = SimpleNamespace(pad_mib="1024", pad_calib=None, calib=None, pad_safety_frac=0.25, pad_safety_mib=64)
    a.__dict__.update(kw)
    return a


def test_calib_and_auto_roundtrip():
    F, end, peak = reference_run()
    a = _args()
    resolve_pad(a, None)
    assert a.pad_mib == 1024 and a.prereserve_mib is None
    cal = pad_calib({"pad_study": pad_study(F, end, peak)}, a)
    # used ~236 MiB x 1.25 + 64 -> ~360 MiB: the arena grows by ~0.65 GiB versus the fixed 1 GiB pad
    assert 350 < cal["pad_mib_cal"] < 370, cal
    assert cal["pad_mib_cal"] >= cal["prereserve_mib_cal"] + cal["pad_nontorch_mib_cal"], cal
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cal, f)
    try:
        b = _args(pad_mib="auto", pad_calib=f.name)
        resolve_pad(b, None)
        assert b.pad_mib == cal["pad_mib_cal"] and b.prereserve_mib == cal["prereserve_mib_cal"]
        c = _args(pad_mib="auto")
        resolve_pad(c, cal)                                        # from --calib when no --pad-calib
        assert c.pad_mib == cal["pad_mib_cal"]
    finally:
        os.unlink(f.name)


def test_auto_without_calib_fails():
    try:
        resolve_pad(_args(pad_mib="auto"), {"allowance_gib": 3.4})
    except SystemExit:
        return
    raise AssertionError("auto without pad_mib_cal must stop")


def test_no_study_no_fields():
    assert pad_calib({}, _args(pad_mib=1024)) == {}
    assert pad_calib(None, _args(pad_mib=1024)) == {}


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
