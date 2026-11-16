#!/usr/bin/env bash
# PCIe correctable-error counters (AER, sysfs, no privilege) on every hop of the GPU's path, one line:
#   bash scripts/reproduce/link_errors.sh            -> "AER <bdf>:<TOTAL_ERR_COR> ..." or "AER UNKNOWN"
#   bash scripts/reproduce/link_errors.sh A B        -> compare two such lines: LINK_ERRORS OK | WARN (deltas) | UNKNOWN
# Informational only (on one host about 3 of 7 deep noise dips came within seconds of a correctable-error
# step on the last two hops; no verdict depends on it).
if [ $# -ge 2 ]; then
  python3 - "$1" "$2" <<'PY'
import sys
def parse(s):
    return {k: int(v) for k, v in (x.rsplit(":", 1) for x in s.split()[1:] if ":" in x)} if not s.endswith("UNKNOWN") else None
a, b = parse(sys.argv[1]), parse(sys.argv[2])
if a is None or b is None: print("LINK_ERRORS UNKNOWN (sysfs AER not readable)")
else:
    d = {k: b[k] - a.get(k, 0) for k in b if b[k] != a.get(k, 0)}
    print("LINK_ERRORS OK" if not d else "LINK_ERRORS WARN deltas " + " ".join(f"{k}:+{v}" for k, v in d.items()))
PY
  exit 0
fi
B=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr A-Z a-z | sed 's/^0000//')
P=$(readlink -f /sys/bus/pci/devices/$B 2>/dev/null)
out="AER"; n=0
for d in $(echo $P | tr / '\n' | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:'); do
  f=/sys/bus/pci/devices/$d/aer_dev_correctable
  [ -r $f ] && { out="$out $d:$(awk '/TOTAL_ERR_COR/ {print $2}' $f)"; n=$((n+1)); }
done
[ $n -gt 0 ] && echo "$out" || echo "AER UNKNOWN"
