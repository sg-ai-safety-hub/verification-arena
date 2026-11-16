"""JIT build of the custom Hopper GEMM extension (sm_90a)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def load(verbose=False):
    global _EXT
    if _EXT is not None:
        return _EXT
    subprocess.check_call([sys.executable, os.path.join(HERE, "gen_wgmma.py"), os.path.join(HERE, "wgmma.h")])
    from torch.utils.cpp_extension import load as tload
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mix = int(os.environ.get("VARENA_CHASE_MIX", "1") or 0)  # chase round function (chase.cu combine; ref.CH_MIX)
    wide = int(os.environ.get("VARENA_CHASE_WIDE", "1") or 1)  # honest chase probe width in 128-byte units (ref.CH_WIDE)
    _EXT = tload(name=f"varena_encgemm_m{mix}_p{wide}", sources=[os.path.join(HERE, "encgemm_torch.cu")],
                 extra_include_paths=[HERE],
                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                    "--ptxas-options=-v", f"-DVARENA_CHASE_MIX={mix}", f"-DVARENA_CHASE_WIDE={wide}"],
                 extra_ldflags=["-lcuda"], verbose=verbose)
    return _EXT


_RG = None


def load_regen(verbose=False):
    """Benchmark-only: the regenerate-on-probe chase kernel (regen_probe_torch.cu, tests/bench_regen_probe_gpu.py)."""
    global _RG
    if _RG is not None:
        return _RG
    from torch.utils.cpp_extension import load as tload
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mix = int(os.environ.get("VARENA_CHASE_MIX", "1") or 0)
    _RG = tload(name=f"varena_regen_probe_m{mix}", sources=[os.path.join(HERE, "regen_probe_torch.cu")],
                extra_include_paths=[HERE],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   f"-DVARENA_CHASE_MIX={mix}"],
                extra_ldflags=["-lcuda"], verbose=verbose)
    return _RG


_SL = None


def load_sl(verbose=False):
    """Fused sloth64 decode-on-load GEMMs (slothgemm*.cu): slgemm (v2), slgemm3 / slgemm4 (cluster variants)."""
    global _SL
    if _SL is not None:
        return _SL
    subprocess.check_call([sys.executable, os.path.join(HERE, "gen_wgmma.py"), os.path.join(HERE, "wgmma.h")])
    from torch.utils.cpp_extension import load as tload
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _SL = tload(name="varena_slgemm", sources=[os.path.join(HERE, "slothgemm_torch.cu")], extra_include_paths=[HERE],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "--ptxas-options=-v",
                                   ],
                extra_ldflags=["-lcuda"], verbose=verbose)
    return _SL
