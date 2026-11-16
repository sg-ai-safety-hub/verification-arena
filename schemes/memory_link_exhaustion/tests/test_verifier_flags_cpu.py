"""The reference configuration parses: the verifier imports with GPU packages stubbed, accepts every flag that
scripts/reproduce/reference.env passes it (plus the floor flags lib.sh adds), and every `a.<name>` the verifier reads is a flag that exists.
Guards against a deleted flag that a script still passes or the code still reads.
Run: python -m schemes.memory_link_exhaustion.tests.test_verifier_flags_cpu"""
import argparse
import os
import sys
import types
import importlib.abc
import importlib.machinery

ROOTS = ("torch", "triton", "vllm", "transformers", "safetensors", "numba")


class _Any(types.ModuleType):
    __path__ = []

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        v = _Any(f"{self.__name__}.{k}")
        setattr(self, k, v)
        return v

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f=None, *aa, **kk: f

    def __getitem__(self, k):
        return _Any("item")

    def __or__(self, o):
        return self

    def __mro_entries__(self, bases):
        return (object,)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ROOTS:
            try:
                __import__("importlib.util").util.find_spec  # noqa
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _Any(spec.name)

    def exec_module(self, m):
        pass


if os.environ.get("VARENA_REAL_TORCH") != "1":
    sys.meta_path.insert(0, _Finder())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
from schemes.memory_link_exhaustion.verifier import verifier as V  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class _Stop(Exception):
    pass


def parse(argv):
    box = {}
    orig = argparse.ArgumentParser.parse_args

    def cap(self, args=None, namespace=None):
        box["a"] = orig(self, argv)
        raise _Stop
    argparse.ArgumentParser.parse_args = cap
    try:
        V.main()
    except _Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return box["a"]


def main():
    env = {**os.environ, "VA_MODEL": "x", "VA_DATA": "/dev/null"}
    sh = ("cd %s && bash -n scripts/reproduce/reference.env && . scripts/reproduce/reference.env >/dev/null; "
          "echo \"$VA_VERIFIER_FLAGS\"; echo \"--end-floor-gbs 56 --head-floor-gbs 56 --end-min-s 60 --hard-mult 1.5\"" % shlex.quote(REPO))
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    lines = [x for x in r.stdout.splitlines() if x.startswith("--")]
    assert len(lines) == 2, r.stdout[-400:]
    flags = shlex.split(lines[0]) + shlex.split(lines[1])
    assert "--kv-table" in flags and "--replay-all" in flags, flags
    a = parse(["--model", "x", "--out", "/dev/null"] + flags)
    assert a.kv_table == 3 and a.kv_table_derive == 1 and a.replay_all == 1 and a.noise_perm == 2, vars(a)
    src = open(V.__file__).read()
    read = set(re.findall(r"\b(?:self\.a|a)\.([a-z][a-z_0-9]*)\b", src))
    derived = {"prereserve_mib", "pad_nontorch_mib"}      # set at run time by verifier/pad.py, not flags
    missing = sorted(x for x in read - derived if not hasattr(a, x))
    assert not missing, f"the verifier reads flags that do not exist: {missing}"
    print("ok: reference flags parse (%d tokens), %d flag names read, none missing" % (len(flags), len(read)))


if __name__ == "__main__":
    main()
