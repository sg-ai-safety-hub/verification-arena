#!/usr/bin/env bash
# GPU tests for the code in the final configuration (run on an idle H200 after setup.sh; run after touching kernels).
#   bash scripts/reproduce/gpu_tests.sh            -> one PASS/FAIL line per test, exit 1 on any failure
#   GPU_TESTS_KNOWN_FAIL=1 bash scripts/reproduce/gpu_tests.sh   also runs test_h2d_route_gpu, a known failure (docs/REPRODUCE.md, "Known open items")
# Every test exits non-zero when one of its own checks fails, so PASS means the checks held, not only that it ran.
# Encodings (butterfly W/KV/filler; sloth64 kept for comparison), the custom decode-on-load GEMM (eager, CUDA graph, torch.compile, concurrent load,
# fused gate_up), sliced decode-ahead, the matmul gate, the mixed probe chase and its tagged in-flight reads, the kernels vs NumPy, the model paths
# vs HF, the single routed H2D queue and the noise stream.
cd "$(dirname "$0")/../.."
. scripts/reproduce/reference.env
rc=0
# test_h2d_route_gpu checks the routed queue, so it runs with the interposer preloaded and the queue settings of a
# verified run (scripts/reproduce/lib.sh ROUTE_ENV). Without the interposer its first side-stream copy never lands.
SO=$(pwd)/build/h2d_route.so
[ -e $SO ] || { mkdir -p build && gcc -O2 -shared -fPIC -o $SO schemes/memory_link_exhaustion/tools/h2d_route.c -ldl; }
ROUTE_ENV="LD_PRELOAD=$SO CUDA_DEVICE_MAX_CONNECTIONS=32 VARENA_SQ_YIELD=1 VARENA_SAT_LEAD=4"
for t in test_kernels test_model test_sloth64_gpu test_sloth_filler_gpu test_bfly_filler_gpu test_kvsloth_attn test_encgemm test_encgemm_paths \
         test_encgemm_graph test_encgemm_compile test_encgemm_race test_slgemm_integ_gpu test_wdec_slice_gpu test_wbfly_gpu test_encgemm_bfly_gpu test_kvbf_attn test_mm_gate_gpu \
         test_chase_tag_gpu test_chase_tag_race_gpu test_sched_stream ${GPU_TESTS_KNOWN_FAIL:+test_h2d_route_gpu}; do
  t0=$(date +%s)
  X=""; [ $t = test_h2d_route_gpu ] && X=$ROUTE_ENV
  if env $X timeout ${GPU_TEST_TIMEOUT:-900} python -m schemes.memory_link_exhaustion.tests.$t > /tmp/gpu_$t.out 2>&1; then
    echo "$t: PASS ($(( $(date +%s) - t0 )) s)"
  else echo "$t: FAIL rc $? ($(( $(date +%s) - t0 )) s; see /tmp/gpu_$t.out)"; rc=1; fi
done
exit $rc
