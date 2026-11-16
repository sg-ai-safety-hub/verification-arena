#!/usr/bin/env bash
# Per-host floors from this box's own honest runs (never another host's):
#   bash scripts/reproduce/floors.sh [tags...]     (default: h1 h2 h3; missing runs are run first, ~6-8 min each)
# end-of-session floor ENDF = lowest honest end-of-session rate - 0.12 GB/s (VA_END_MARGIN): the counted link rule,
# with the head rule, of the simplified mode. (THR10 / FLOOR, the windowed and cumulative floors of the old rules, are
# still written for the record; cheater.sh no longer passes them.) Chase tail limit 1.5 x.
# Writes $VA_OUT/floors.env, which cheater.sh loads. The first tag's run is the timing calibration.
cd "$(dirname "$0")/../.."
. scripts/reproduce/lib.sh
TAGS=${*:-h1 h2 h3}
# floor runs check 4 sampled chains like every other run; FLOOR_REPLAY=full replays every answer's chain
M=${FLOOR_REPLAY:-sample}
for t in $TAGS; do grep -q "^WINX $t .* w10 [0-9]" $L || bash scripts/reproduce/honest.sh $t 300 $M; M=${FLOOR_REPLAY:-sample}; done
read THR10 THR1 FLOOR HMC HMR NRUNS FZMAX ENDF MEANREF < <(fin_thresh $TAGS)
set -- $TAGS
cat > $VA_OUT/floors.env <<F
# per-host floors from honest runs: $TAGS ($(date -u +%FT%TZ), $(hostname))
THR10=$THR10 THR1=$THR1 FLOOR=$FLOOR HMC=$HMC HMR=$HMR NRUNS=$NRUNS CALIB=$1 FZMAX=$FZMAX ENDF=$ENDF MEANREF=$MEANREF
F
cat $VA_OUT/floors.env
