"""Activation pool rule. A prover may declare an activation pool (info["act_pool"]: granule addresses and bytes) that
the mixed chase then covers in place of arena filler. Its content is a cheap address-keyed keystream, not sloth64
filler, so a declared pool is memory a cheater could borrow and restore cheaply. The honest prover declares none, so
a run fails `no_act_pool` when it declares more pool bytes than the calibration it is judged against
(calibration field act_pool_bytes; calibrations written before the field existed had no pool: 0)."""


def act_pool_bytes(info: dict | None) -> int:
    ap = (info or {}).get("act_pool") or {}
    return int(ap.get("bytes") or 0)


def check(info: dict | None, calib: dict | None):
    """(ok, declared bytes, allowed bytes). Without a calibration (an honest calibration run, check_host) the allowance
    is 0: the final configuration's honest prover has no pool."""
    declared = act_pool_bytes(info)
    allowed = int((calib or {}).get("act_pool_bytes", 0) or 0)
    return declared <= allowed, declared, allowed


def cheater_declared_min(cap: int, allowance: int, act_bytes: int = 0) -> int:
    """Bytes the zero-weight cheater must declare as arena pages: capacity minus the allowance, minus any pool it
    declares (none by default, so the whole remainder is arena: W, KV, sloth64 filler, ring)."""
    return cap - allowance - act_bytes
