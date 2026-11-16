"""Run one attack test_cpu.py module without pytest: python scripts/reproduce/cpu_suite.py <attack>  (max_shard, local_rl, ...)."""
import sys, traceback, inspect, importlib
sys.path.insert(0, ".")
class MP:
    def __init__(s): s.u = []
    def setattr(s, t, n, v=None, raising=True):
        if isinstance(t, str) and v is None:
            m, a = t.rsplit(".", 1); o = importlib.import_module(m); v = n; n = a; t = o
        s.u.append((t, n, getattr(t, n))); setattr(t, n, v)
    def undo(s):
        for t, n, v in reversed(s.u): setattr(t, n, v)
M = sys.argv[1]
T = importlib.import_module(f"schemes.memory_link_exhaustion.attacks.{M}.test_cpu")
ok = bad = 0
for k in sorted(dir(T)):
    if k.startswith("test_"):
        f = getattr(T, k); mp = MP()
        try:
            f(mp) if "monkeypatch" in inspect.signature(f).parameters else f(); ok += 1
        except Exception:
            bad += 1; print("FAIL", k); traceback.print_exc(limit=3)
        finally:
            mp.undo()
print(M, "passed", ok, "failed", bad)
