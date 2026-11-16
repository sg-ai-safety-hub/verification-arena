"""CPU tests for the activation pool rule (verifier/act_pool.py) and the zero-weight cheater's declared-bytes sizing.
Run: python -m schemes.memory_link_exhaustion.tests.test_act_pool_cpu"""
from schemes.memory_link_exhaustion.verifier import act_pool as AP

GiB = 1 << 30
CAP = 150111977472                          # H200 NVL (verifier GPU_CAPACITY)
POOL = 1946157056                           # the 1.81 GiB pool the zero-weight cheater used to declare


def test_no_pool_passes():
    for calib in (None, {}, {"act_pool_bytes": 0}):
        ok, d, allowed = AP.check({"act_pool": None}, calib)
        assert ok and d == 0 and allowed == 0, (calib, ok, d, allowed)
    assert AP.check({}, {"allowance_gib": 1.984})[0]


def test_declared_pool_fails_against_poolless_calibration():
    info = {"act_pool": {"granules": [1 << 40], "bytes": POOL, "master": 1, "epoch": 1}}
    for calib in (None, {"allowance_gib": 1.984, "outside_gib": 1.9214},       # older calibrations: no field
                  {"act_pool_bytes": 0}):
        ok, d, allowed = AP.check(info, calib)
        assert not ok and d == POOL and allowed == 0, (calib, ok, d, allowed)
    # a calibration that had the same pool (an earlier design) would allow it
    assert AP.check(info, {"act_pool_bytes": POOL})[0]
    assert not AP.check(info, {"act_pool_bytes": POOL - 4096})[0]


def test_cheater_declared_min():
    allowance = int(1.984 * GiB)
    assert AP.cheater_declared_min(CAP, allowance) == CAP - allowance              # no pool term by default
    # the old behaviour: the pool was carved out of the arena (same total declared, less sloth filler)
    assert AP.cheater_declared_min(CAP, allowance, POOL) == CAP - allowance - POOL


if __name__ == "__main__":
    test_no_pool_passes()
    test_declared_pool_fails_against_poolless_calibration()
    test_cheater_declared_min()
    print("test_act_pool_cpu: PASS")
