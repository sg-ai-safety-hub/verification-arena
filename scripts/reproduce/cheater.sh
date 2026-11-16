#!/usr/bin/env bash
# One cheater against this box's floors:  bash scripts/reproduce/cheater.sh <name> [prompts]
# Names: scripts/reproduce/cheaters.json. Run idle_control_native first:
# a ring-content failure only counts as caught if the idle control passes under the same floors.
# Output: $VA_OUT/f_<name>.json (+ .log, .json.prover.log), a WINX line in $VA_OUT/arena.log, the verdict.
cd "$(dirname "$0")/../.."
F=scripts/reproduce/cheaters.json
. scripts/reproduce/lib.sh
load_floors || exit 1
N=$1
read MODE NPR ENVX SPEC < <(python - "$N" "$F" <<'PY'
import json, sys
c = json.load(open(sys.argv[2]))[sys.argv[1]]
print(c["mode"], c.get("prompts", 300), (c.get("env") or "-").replace(" ", "|"), json.dumps(c["spec"]).replace(" ", ""))
PY
)
[ "$ENVX" = - ] && ENVX="" || ENVX=${ENVX//|/ }
PR $N $MODE "$SPEC" ${2:-$NPR} "$ENVX"
grep "^WINX $N " $L | tail -1
[ -f $VA_OUT/f_$N.json ] || { echo "== $N"; echo "   VALIDITY: VOID (the run produced no report; see $VA_OUT/f_$N.log and the prover log next to it)"; exit 1; }
python scripts/reproduce/verdict.py $VA_OUT/f_$N.json --name $N
