#!/usr/bin/env bash
# The whole final story on one H200, after setup.sh (~1 h 40 min; times measured on an H200 NVL, see docs/REPRODUCE.md):
#   host check, the three bare baselines, three honest runs + per-host floors, then the final cheaters, idle control
#   first. Every run appends a WINX line to $VA_OUT/arena.log; verdict.py prints validity first, then the slowdown.
set -u
cd "$(dirname "$0")"
# the host check takes ~15 min: skip it when it already passed on this box (FORCE_CHECK=1 runs it again)
. ./reference.env
case $VA_OUT in /*) HC=$VA_OUT;; *) HC=../../$VA_OUT;; esac; HC=$HC/check_host/result.txt
if [ "${FORCE_CHECK:-0}" != 1 ] && grep -q "^NOISE_PATH OK" $HC 2>/dev/null; then
  echo "host check already passed: $(cat $HC)"
else
  bash check_host.sh
fi
bash baseline_inference.sh
bash baseline_pretrain.sh
bash baseline_rl.sh
bash floors.sh h1 h2 h3
for n in ${CHEATERS:-idle_control_native max_shard_fitnh_loop_A_l05 local_rl_a16_g2_smooth_loop_nh}; do
  bash cheater.sh $n
done
