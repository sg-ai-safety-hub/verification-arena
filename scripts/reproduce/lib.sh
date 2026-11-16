# Shared helpers for scripts/reproduce/{honest,floors,cheater,check_host}.sh. Source from the repository root; it sets
# L, the run log, to $VA_OUT/arena.log, so each VA_OUT (one per workload) has its own tags. The configuration itself is scripts/reproduce/reference.env; on top of it every run here uses
#   the routed single H2D queue (LD_PRELOAD interposer, yielding) with saturating noise (ROUTE_ENV),
#   the memory cut: arena sized from the honest pad probe (PADMIB=auto), margin 1/16 GiB, CUDA graphs at powers of two,
#   per-host floors from this host's own honest runs (floors.sh): cumulative min mean - 0.3 GB/s, 10 s windowed
#   min w10 - 0.5 GB/s.
VA_ROOT=${VA_ROOT:-$(pwd)}
R=scripts/reproduce
step() { echo "=== $* $(date +%T)" >> $L; }
. $R/reference.env
case $VA_OUT in /*) echo "VA_OUT must be relative to the repository root: $VA_OUT" >&2; exit 1;; esac
mkdir -p $VA_OUT; L=$VA_ROOT/$VA_OUT/arena.log; touch $L
# noise windows in flight (VA_SAT_LEAD, default 4): a window is revealed as soon as fewer than this many are outstanding
export VARENA_SINGLE_QUEUE=1 VARENA_SAT_LEAD=${VA_SAT_LEAD:-4}
KILL() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done 2>/dev/null; sleep 3; rm -f /dev/shm/varena_*; }
TR=schemes/memory_link_exhaustion/scripts/timing_runs.sh
pin_cmd() { local B=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z | sed 's/^0000//')
  local N=$(cat /sys/bus/pci/devices/$B/numa_node 2>/dev/null || echo 0); [ "$N" -lt 0 ] 2>/dev/null && N=0
  echo "taskset -c $(cat /sys/devices/system/node/node$N/cpulist)"; }
PIN="$(pin_cmd)"; SO=$VA_ROOT/build/h2d_route.so
ROUTE_ENV="LD_PRELOAD=$SO CUDA_DEVICE_MAX_CONNECTIONS=32 VARENA_SINGLE_QUEUE=1 VARENA_SQ_YIELD=1 VARENA_SAT_LEAD=${VA_SAT_LEAD:-4}"
MEMCUT_ENV="VARENA_CG_SIZES=pow2"
export GUARD_S=${GUARD_S:-2700} PR_GUARD_S=${PR_GUARD_S:-2700}
build_route() { mkdir -p $VA_ROOT/build && gcc -O2 -shared -fPIC -o $SO schemes/memory_link_exhaustion/tools/h2d_route.c -ldl; }
[ -e $SO ] || build_route

# WINX tag json: one line per run with the verdict essentials. Windows are taken after the 60 s start-up: worst
# 1 / 2 / 10 / 60 s and the mean of the per-second noise series; failed checks; the cheater's workload rate.
WINX() { python - "$@" >> $L 2>&1 <<'PY'
import json, sys
tag, f = sys.argv[1], sys.argv[2]; r = json.load(open(f)); p = r.get("pcie") or {}; s = (p.get("rate_var") or {}).get("series_1s") or []
b = s[55:]; b = b[:-10] if len(b) > 20 else b
# freeze credit (always on): windows and the mean are rates over the time the verifier's Revealer was not frozen
# (verifier/floor.py), so floors and runs are measured on the same clock
cr = (p.get("rate_var") or {}).get("freeze_credit_1s") or []
cr = (cr[55:] + [0.0] * len(b))[:len(b)]
def sm(L): return round(min(sum(b[i:i+L]) / (L - min(sum(cr[i:i+L]), L / 2)) for i in range(len(b) - L + 1)), 2) if len(b) >= L else None
def mean_(): return round(sum(b) / (len(b) - min(sum(cr), len(b) / 2)), 2) if b else None
m = r["memory"]; c = (r.get("prover") or {}).get("cheat") or {}
t = {x: (m.get(f"{x}_timing") or {}) for x in ("mchase",)}
print("WINX", tag, "tok_s", round(r["tok_s"], 1), "noise", p.get("rate_gbs"), "w1", sm(1), "w2", sm(2), "w10", sm(10), "w60", sm(60),
      "mean", mean_(), "maxT", {x: round(t[x]["max_ms"] / t[x]["T_ms"], 2) for x in t if t[x].get("T_ms")},
      "fail", [x for x, y in r["checks"].items() if not y], "vs", r.get("vs_baseline"),
      "rate", c.get("rate"), "wl", json.dumps(c.get("workload"))[:300] if c.get("workload") else None,
      "end", (r.get("end_rule") or {}).get("gbs"),
      "fzc", p.get("verifier_freeze_credit_s", 0.0))
PY
}

# RU tag NP [ENV=... ...]: one routed honest run (timing_runs cfg:ref, unified flags, memory cut), guarded.
# HX: extra verifier flags for this run, last so they win (SAMPLE_REPLAY below).
# SAMPLE_REPLAY: content-check 4 randomly chosen challenges per kind after the session, replaying one chain of each
# of those answers, instead of every answer (--check-max defaults to 150 and --replay-all 1 queues the rest). Measured
# on Box S it saves only about 20 s: the 3 minutes after serving go to the reference job (a second engine start and a
# prefill of all 300 requests into a KV store), not to the replay itself. The default for every honest run (16 Nov).
SAMPLE_REPLAY="--replay-all 0 --check-max 4 --mchase-verify 4"   # 4 chains; MODE=full replays every answer
RU() { local T=$1 NP_=$2; shift 2; local O=$VA_ROOT/$VA_OUT/h_$T; rm -rf $O; mkdir -p $O
  cp $VA_OUT/base_stockdet/{base1.tokens.json,baseline.jsonl} $O/; step HONEST $T "$@"; KILL
  ( sleep $GUARD_S; grep -q "^WINX $T " $L || { echo "NOTE $T killed after $GUARD_S s" >> $L
      for p in $(ps -eo pid,args | awk "/timing_runs.sh|verifier.verifier/ && !/awk/ {print \$1}"); do kill -9 $p; done; } ) &
  env $ROUTE_ENV $MEMCUT_ENV MRATE=${MR:-4} NGBS=$VA_NOISE_GBS NP=$NP_ R=${R_:-0} CALN=1 RESERVESMS=2 MMGATE=1 \
    MODEL=$VA_MODEL NB=$VA_NB DS="--dataset-path $VA_DATA --dataset $VA_DATASET" OUT=$O XV="$VA_UNIFIED_FLAGS ${XF:-} ${HX:-}" "$@" $PIN \
    timeout $GUARD_S bash $TR cfg:ref >> $VA_ROOT/$VA_OUT/runs.log 2>&1
  WINX $T $O/calib_ref.json || echo "WINX $T NO_RESULT" >> $L
  echo "STALLS $T $(grep -ac 'no schedule()' $O/calib_ref.log) GUARD $(grep -ac 'GUARD TRIPPED' $O/*.log)" >> $L; }

# VA_XFLAGS: extra verifier flags for cheater runs, last so they win (e.g. "--mchase-verify 2" for link-calibration
# runs, where the post-run memory content check is not what is being measured).
# PR tag mode spec [prompts] [extra prover env]: one routed cheater run against the calibration in $C3 and the floors
# in $XF / $VARENA_SAT_FLOOR_GBS (load them with load_floors).
PR() { local N=$1 MODE=$2 SPEC=$3 NPR=${4:-300} XENV=${5:-}; step PROBE $N; KILL
  # cheaters that serve nothing report the sealed block table (nothing in flight); for real-weight ones a sample of
  # their chains is replayed from sources (every chain of a long session would take minutes), for zero-weight ones
  # the sampled chains are walked against the zero model's content
  # both kinds answer the full challenge: the table kernel with tagged reads (--kv-table 3), every ring slot read
  local X="--zero-weights --no-reference --kv-table ${VA_KV_TABLE_ZERO:-3}"
  [ "$MODE" = real ] && X="--max-claims 300 --kv-table ${VA_KV_TABLE_REAL:-3} --replay-all 0 --mchase-verify 200"
  local FLB="$VA_VERIFIER_FLAGS ${XF:-} ${VA_XFLAGS:-}" PV="$VA_PROVER_ENV $XENV"
  # no stale report: a run that dies must not leave an earlier run's verdict behind under the same name
  rm -f $VA_ROOT/$VA_OUT/f_$N.json $VA_ROOT/$VA_OUT/f_$N.json.prover.log $VA_ROOT/$VA_OUT/f_$N.json.replay.npz
  ( sleep $PR_GUARD_S; grep -q "^WINX $N " $L || { echo "NOTE $N killed after $PR_GUARD_S s" >> $L
      for p in $(ps -eo pid,args | awk "/verifier.verifier/ && !/awk/ {print \$1}"); do kill -9 $p; done; } ) &
  env $PV $ROUTE_ENV $PIN timeout $PR_GUARD_S python -m schemes.memory_link_exhaustion.verifier.verifier --model $VA_MODEL \
    --num-prompts $NPR --num-blocks $VA_NB --dataset-path $VA_DATA --dataset $VA_DATASET --shm /dev/shm --calib $C3/calibq_ref.json $FLB $X --prover cheater \
    --cheater-spec "$SPEC" --out $VA_ROOT/$VA_OUT/f_$N.json > $VA_ROOT/$VA_OUT/f_$N.log 2>&1
  WINX $N $VA_ROOT/$VA_OUT/f_$N.json || echo "WINX $N NO_RESULT" >> $L
  grep -ah "OutOfMemory\|Traceback\|Error:" $VA_ROOT/$VA_OUT/f_$N.json.prover.log 2>/dev/null | tail -2 | cut -c1-200 | sed "s/^/FLOG $N /" >> $L; }

# fin_thresh tag...: thresholds from this host's honest WINX lines in $L:
#   THR10 THR1 FLOOR HM_MCHASE HM_RING NRUNS FZMAX  (w10 floor = min w10 - VA_W10_MARGIN, default 0.5; cumulative =
#   min mean - VA_FLOOR_MARGIN, default 0.3; tails
#   >= 1.5 / 2.5; FZMAX = the largest freeze credit (s) among these honest runs, for the record, or "off" for
#   floors that predate the freeze credit)
fin_thresh() { python - "$L" "$@" <<'PY'
import sys, re, ast
L, tags = sys.argv[1], set(sys.argv[2:])
w1, w10, mean, mc, rg, fzc, end = [], [], [], [1.0], [1.0], [], []
for line in open(L, errors="replace"):
    if not line.startswith("WINX "): continue
    t = line.split()[1]
    if t not in tags or not re.search(r" w10 [0-9.]+", line): continue
    g = lambda k: re.search(rf" {k} ([0-9.]+|None)", line).group(1)
    if g("w10") != "None": w10.append(float(g("w10"))); w1.append(float(g("w1"))); mean.append(float(g("mean")))
    m = re.search(r"maxT (\{[^}]*\})", line)
    if m:
        d = ast.literal_eval(m.group(1)); mc.append(d.get("mchase", 1.0)); rg.append(d.get("ring", 1.0))
    m = re.search(r" fzc ([0-9.]+)\s*$", line)
    if m: fzc.append(float(m.group(1)))
    m = re.search(r" end ([0-9.]+)", line)
    if m: end.append(float(m.group(1)))
import os, json
# Tail multiplier from the honest runs' own slowest answers. Honest runs are calibration runs, so their WINX lines
# carry no max / T; read each run's calibration file instead: slowest answer of every honest run over T, where T is
# the FIRST tag's p99 (the calibration cheater runs are judged against). Until 15 Nov this list stayed at [1.0], so
# the multiplier was always 1.5, and on a host whose honest serving reaches 1.5 to 1.8 T the idle control failed.
out = os.environ.get("VA_OUT", "results/reproduce")
try:
    cal = lambda t: json.load(open(os.path.join(out, f"h_{t}", "calibq_ref.json")))
    T0 = cal(sys.argv[2])["lat_mchase_ms_p99"]
    for t in sys.argv[2:]:
        try:
            mc.append(round(cal(t)["lat_mchase_ms_max"] / T0, 3))
        except Exception:
            pass
except Exception:
    pass
# mean rule reference (16 Nov): the slowest honest floor run's mean answer time without its slowest answer
means = []
for t in sys.argv[2:]:
    try:
        r = json.load(open(os.path.join(out, f"h_{t}", "calib_ref.json")))
        mt = ((r.get("memory") or {}).get("mchase_timing") or {}).get("mean1_ms")
        if mt is None:
            ls = sorted(((r.get("memory") or {}).get("mchase_lat_ms") or {}).get("all") or [])
            mt = sum(ls[:-1]) / (len(ls) - 1) if len(ls) >= 2 else None
        if mt is not None:
            means.append(mt)
    except Exception:
        pass
m10, mc_ = float(os.environ.get("VA_W10_MARGIN", "0.5")), float(os.environ.get("VA_FLOOR_MARGIN", "0.3"))
print(round(min(w10) - m10, 2), round(min(w1) - 2.0, 2), round(min(mean) - mc_, 2),
      round(max(1.5, 1.2 * max(mc)), 2), round(max(2.5, 1.2 * max(rg)), 2), len(w10),
      round(max(fzc), 3) if len(fzc) == len(w10) and fzc else "off",
      # end-of-session floor (the counted link rule of the simplified mode): lowest honest end-of-session rate minus
      # VA_END_MARGIN (default 0.12 GB/s); "off" when the honest runs predate the rule
      round(min(end) - float(os.environ.get("VA_END_MARGIN", "0.12")), 2) if len(end) == len(w10) and end else "off",
      round(max(means), 4) if len(means) == len(sys.argv[2:]) and means else "off")
PY
}

# floors file written by floors.sh; load_floors arms them for PR
load_floors() { [ -f $VA_OUT/floors.env ] || { echo "no $VA_OUT/floors.env: run scripts/reproduce/floors.sh first" >&2; return 1; }
  . $VA_OUT/floors.env; export VARENA_SAT_FLOOR_GBS=$FLOOR
  # floors from before the freeze credit became permanent are on a different clock
  [ "${FZMAX:-off}" = off ] && { echo "floors.env predates the freeze credit: rerun scripts/reproduce/floors.sh" >&2; return 1; }
  # link rules: the end-of-session rate and the head rule against this host's floor (floors.env files from before
  # the end floor carry only THR10 / FLOOR: rerun floors.sh)
  [ "${ENDF:-off}" = off ] && { echo "floors.env has no end-of-session floor: rerun scripts/reproduce/floors.sh" >&2; return 1; }
  XF="--end-floor-gbs $ENDF --head-floor-gbs $ENDF --end-min-s 60 --hard-mult $HMC"
  # mean rule (16 Nov): within VA_MEAN_TOL (default 5%) of the slowest honest floor run's mean answer time
  [ "${MEANREF:-off}" != off ] && XF="$XF --mean-ref-ms $MEANREF --mean-tol ${VA_MEAN_TOL:-0.05}"
  # the closed-loop (paced) cheaters hold their own noise average here: just above the floor unless set by hand
  export VARENA_PACE_TARGET_GBS=${VARENA_PACE_TARGET_GBS:-$(python3 -c "print(round($ENDF + 0.03, 3))")}
  C3=$VA_ROOT/$VA_OUT/h_$CALIB; }
