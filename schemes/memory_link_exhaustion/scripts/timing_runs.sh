#!/usr/bin/env bash
# Engine runs for scripts/reproduce: `base` (stock deterministic vLLM, the reference tokens) and `cfg:ref` (one honest
# prover run in the final configuration: calibration + R repeats).
#   OUT=<dir> bash schemes/memory_link_exhaustion/scripts/timing_runs.sh base cfg:ref
set -uo pipefail
cd "$(dirname "$0")/../../.."
[ -f /root/env.sh ] && . /root/env.sh
MODEL=${MODEL:-/root/models/Qwen2.5-7B-Instruct}; OUT=${OUT:-results/h200_timing}; NB=${NB:-8192}
NP=${NP:-4000}; R=${R:-3}
mkdir -p $OUT
export VARENA_TRITON_GEMM=1 VLLM_BATCH_INVARIANT=1
clean() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null; for i in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -lt 2000 ] && break; sleep 1; done; for q in $(pgrep -f "spawn import spawn_main"); do kill -9 $q; done 2>/dev/null; rm -f /dev/shm/varena_*; }
V() { T=$1; shift; echo "=== $T $(date +%T)"; clean
      export VARENA_STEPLOG=$PWD/$OUT/$T.steps.npy
      timeout 3600 python -m schemes.memory_link_exhaustion.verifier.verifier --model $MODEL --num-blocks $NB --shm /dev/shm --out $OUT/$T.json \
        --pad-mib ${PAD:-4096} ${PLAINW---plain-w 1} --num-prompts $NP ${DS:-} ${TOKCAP:+--max-batched-tokens $TOKCAP} "$@" ${XV:-} > $OUT/$T.log 2>&1; echo "rc $?"
      grep -a "verifier\] RESULT" $OUT/$T.log | cut -c1-300; clean; }
BT="--baseline-tokens $OUT/base1.tokens.json"
fix() {  # name, env assignments..., -- verifier args
  local name=$1; shift; local envs=(); while [[ $# -gt 0 && $1 != "--" ]]; do envs+=("$1"); shift; done; shift || true
  env "${envs[@]}" bash -c "true"
  ( export "${envs[@]}"
    if [ "${CALN:-1}" -gt 1 ]; then   # calibrate over CALN honest runs (max of p99s)
      for c in $(seq ${CALN}); do [ -e $OUT/calibq_${name}_c$c.json ] || V calib_${name}_c$c $BT --write-calib $OUT/calibq_${name}_c$c.json "$@"; done
      [ -e $OUT/calibq_$name.json ] || python schemes/memory_link_exhaustion/scripts/merge_calib.py $OUT/calibq_$name.json $(ls $OUT/calibq_${name}_c*.json)
    fi
    [ -e $OUT/calibq_$name.json ] || V calib_$name $BT --write-calib $OUT/calibq_$name.json "$@"
    for r in $(seq $R); do [ -e $OUT/honest_${name}_$r.json ] || V honest_${name}_$r --calib $OUT/calibq_$name.json $BT "$@"; done )
}
for step in "$@"; do
  case $step in
    base) [ -e $OUT/base1.tokens.json ] && continue; echo "=== base1 $(date +%T)"; clean
          timeout 3600 python -m baselines.inference.bench --model $MODEL --num-prompts $NP --num-blocks $NB --tag base1 \
            ${DS:-} ${TOKCAP:+--max-batched-tokens $TOKCAP} --out $OUT/baseline.jsonl --save-tokens $OUT/base1.tokens.json > $OUT/base1.log 2>&1; echo "rc $?"; clean ;;
    # the final configuration's honest run (scripts/reproduce/lib.sh RU sets the overrides): butterfly W / KV / filler,
    # fused gate_up + smoothed decode-ahead + pre-emptible matmul (scripts/reproduce/reference.env), stale KV, plain
    # activations (no activation pool), mixed chase over W / K / F, 2 GiB ring
    cfg:ref) PLAINW="" fix ref VARENA_ATT_CLAMP=${ATTCLAMP:-0} VARENA_ATT_BLOCK_M_PREFILL=${ATTBM:-32} VARENA_RESERVE_SMS=${RESERVESMS:-0} VARENA_MM_GATE=${MMGATE:-0} VARENA_STALE_KV=1 VARENA_KVENC=${KVENC:-bfly} VARENA_KV_BFLY_L=8 VARENA_PRERESERVE_MIB=${PRERES:-1024} VARENA_WENC=${WENC:-bfly} VARENA_FILLER=${FILLER:-bfly} VARENA_CUDA_GEMM=1 VARENA_PAUSE_VLLM=1 VARENA_ZEROCOPY=1 VARENA_BFLY_AHEAD_M=${BFLYAHEAD:-512} VARENA_BFLY_FUSE_KINDS=${FUSEKINDS-gate_up} VARENA_BFLY_XOR_SMEM_KB=${XORSMEM-45} VARENA_AHEAD_AE=${AHEADAE:-1} VARENA_AE_PREEMPT=1 VARENA_WDEC_SLICE_MB=${SLICEMB:-270} -- --noise-sched 1 --window-mib 32 --pool-mib 65536 --noise-gbs ${NGBS:-48} --lag ${LAG:-2} --ring-mib ${RINGMIB:-2048} --mchase-rate ${MRATE:-1} --check-max 2 --pad-mib ${PADMIB:-1024} ${PADCAL:+--pad-calib $PADCAL} --max-batched-tokens ${TOKCAP_REF:-1024} ${NOREF---no-reference} --allowance-margin-gib ${MARGIN:-0.25} ;;
    *) echo "unknown step $step" ;;
  esac
done
echo TIMING_DONE
