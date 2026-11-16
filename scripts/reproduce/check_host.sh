#!/usr/bin/env bash
# Hardware acceptance: GPU, PCIe link (gen/width along the path), NUMA, IOMMU, raw H2D/D2H ceiling (~3.5 min), then the noise-path check
# (one short session).
# Expect: H200 (NVL or SXM), Gen5 x16 on every hop, raw H2D at 32 MiB >= 55 GB/s. Hosts below ~50 GB/s cannot run the reference noise.
cd "$(dirname "$0")/../.."; . scripts/reproduce/reference.env
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader
B=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z | sed 's/^0000//')
P=$(readlink -f /sys/bus/pci/devices/$B); echo "path $P"
for d in $(echo $P | tr / '\n' | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:'); do echo "  $d $(cat /sys/bus/pci/devices/$d/current_link_speed) x$(cat /sys/bus/pci/devices/$d/current_link_width) numa $(cat /sys/bus/pci/devices/$d/numa_node)"; done
echo "iommu: $(tr ' ' '\n' < /proc/cmdline | grep -i iommu | tr '\n' ' ')"
nvidia-smi topo -m | head -3
mkdir -p $VA_OUT/check_host
python scripts/reproduce/linkbench.py | tee $VA_OUT/check_host/linkbench.txt
# host memory and NUMA (informational): automatic NUMA balancing and per-node free memory
echo "numa_balancing $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unknown)"
for n in /sys/devices/system/node/node[0-9]*; do echo "$(basename $n) $(awk '/MemFree/ {printf "free %.0f GB", $4/1e6}' $n/meminfo)"; done
AER0=$(bash scripts/reproduce/link_errors.sh); echo "$AER0"

# Noise-path check (CHECK_NOISE=0 skips): saturating inbound noise with an idle prover (no vLLM, no hidden
# work), pinned to the GPU's NUMA cores. One session of 75 prompts (about a minute of noise) with two answers
# content-checked: only the rate is used here. NOISE_RUNS=2 NOISE_PROMPTS=150 gives the earlier, longer form. The verifier's achieved noise rate must come within NOISE_TOL_GBS
# (default 2) of the raw 32 MiB H2D ceiling above. One host reached 55 raw but only ~43 here, and could not run the
# reference noise. Prints NOISE_PATH OK / WARN.
# Final configuration: the routed single H2D queue (LD_PRELOAD interposer) and saturating noise from
# lib.sh. Without the interposer this path measured only ~43-45 GB/s on hosts whose routed honest runs reach ~56.
if [ "${CHECK_NOISE:-1}" = 1 ]; then
  . scripts/reproduce/lib.sh
  RAW=$(awk '/^RAW_H2D_32MIB/ {print $2}' $VA_OUT/check_host/linkbench.txt)     # from the bench above, not a second run
  # the noise path only: no unified challenge (it needs a calibration, which does not exist yet at this step)
  FLB="${VA_VERIFIER_FLAGS% $VA_UNIFIED_FLAGS}"      # the noise stream and a plain mixed chase only
  mkdir -p $VA_OUT/check_host; R=""
  for i in $(seq ${NOISE_RUNS:-1}); do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done 2>/dev/null; sleep 3; rm -f /dev/shm/varena_*
    env $VA_PROVER_ENV $ROUTE_ENV timeout 900 $PIN python -m schemes.memory_link_exhaustion.verifier.verifier \
      --model $VA_MODEL --num-prompts ${NOISE_PROMPTS:-75} --num-blocks $VA_NB --dataset-path $VA_DATA --dataset $VA_DATASET --shm /dev/shm $FLB --mchase-verify 2 --zero-weights --no-reference \
      --prover cheater --cheater-spec '{"keep_w":true,"keep_k":true,"rate_tok_s":300}' --out $VA_OUT/check_host/noise$i.json > $VA_OUT/check_host/noise$i.log 2>&1
    R="$R $(python -c "import json;print(json.load(open('$VA_OUT/check_host/noise$i.json'))['pcie']['rate_gbs'])" 2>/dev/null)"
  done
  AER1=$(bash scripts/reproduce/link_errors.sh); echo "$AER1"; bash scripts/reproduce/link_errors.sh "$AER0" "$AER1"
  python - "$RAW" "${NOISE_TOL_GBS:-2}" $R <<'PY' | tee $VA_OUT/check_host/result.txt
import sys
raw, tol, runs = float(sys.argv[1] or 0), float(sys.argv[2]), [float(x) for x in sys.argv[3:]]
worst = min(runs) if runs else 0.0
ok = raw > 0 and len(runs) >= 1 and worst >= raw - tol
print("NOISE_PATH", "OK" if ok else "WARN", "raw_h2d_32mib", raw, "noise_runs", runs, "tol", tol,
      "" if ok else "(noise path cannot get within tol of the raw link: do not use this host for verdicts)")
PY
fi
