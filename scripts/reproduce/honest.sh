#!/usr/bin/env bash
# One honest run in the final configuration (routed saturating noise, unified challenge, memory cut).
#   bash scripts/reproduce/honest.sh <tag> [prompts] [full|sample]   (default tag h1, 300 prompts, sample; about 8 min on
#   one H200. sample replays one chain of 4 answers; full replays one chain of every answer)
# The first call on a box also runs the pad probe (tag pad: the memory-cut measurement the arena is sized from).
# Output: $VA_OUT/h_<tag>/calib_ref.json (+ calibq_ref.json, the timing calibration), one WINX line in $VA_OUT/arena.log.
# Expect (H200 NVL, reference config, Box W 16 Nov): about 600-610 tok/s on ShareGPT, outputs 300/300 identical to stock, every check passes
# (a harness artefact, see verdict.py).
cd "$(dirname "$0")/../.."
. scripts/reproduce/lib.sh
T=${1:-h1}; NP=${2:-300}; MODE=${3:-sample}      # sample (default) = 4 chains; full = every answer's chain
[ -f $VA_OUT/base_stockdet/baseline.jsonl ] || { echo "run scripts/reproduce/baseline_inference.sh first" >&2; exit 1; }
PC=$VA_ROOT/$VA_OUT/h_pad/calibq_ref.json
# The pad probe measures memory, so it replays a sample only. One probe serves every output folder of the same
# workload on this box: reuse another folder's if its stamp matches (FORCE_PAD=1 measures again).
if [ ! -f $PC ] && [ "${FORCE_PAD:-0}" != 1 ]; then
  for d in $VA_ROOT/results/*/h_pad; do
    [ -f $d/calibq_ref.json ] && [ "$(cat $d/workload 2>/dev/null)" = "$VA_DATASET" ] || continue
    mkdir -p $VA_OUT/h_pad && cp $d/calibq_ref.json $d/workload $VA_OUT/h_pad/ && echo "pad probe reused from $d" && break
  done
fi
if [ ! -f $PC ] || [ "${FORCE_PAD:-0}" = 1 ]; then
  HX="$SAMPLE_REPLAY" RU pad 300 PRERES=0 MARGIN=0.0625
  [ -f $PC ] && echo "$VA_DATASET" > $VA_OUT/h_pad/workload
fi
A0=$(bash scripts/reproduce/link_errors.sh)
[ "$MODE" = sample ] && HX="$SAMPLE_REPLAY"
RU $T $NP PADMIB=auto PADCAL=$PC MARGIN=0.0625
bash scripts/reproduce/link_errors.sh "$A0" "$(bash scripts/reproduce/link_errors.sh)"     # informational
grep "^WINX $T " $L | tail -1
python scripts/reproduce/verdict.py $VA_OUT/h_$T/calib_ref.json
