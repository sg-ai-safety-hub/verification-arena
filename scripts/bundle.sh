#!/usr/bin/env bash
# Pack the code a rented GPU box needs into one tarball (tracked files only, from an allowlist).
#   bash scripts/bundle.sh [out.tgz]            default: /tmp/varena-bundle.tgz
# Then on the box: mkdir -p /root/varena && tar xzf varena-bundle.tgz -C /root/varena
#                  cd /root/varena && pip install --no-deps -e .      (re-run after a layout change: the vLLM plugin entry point moves)
# Only allowlisted, tracked paths are shipped. results/ isn't shipped; runs write their own results/ on the box.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/varena-bundle.tgz}
ALLOW=(README.md LICENSE pyproject.toml .gitignore baselines schemes scripts docs)
FILES=$(git ls-files -- "${ALLOW[@]}")
tar czf "$OUT" $FILES
echo "$OUT: $(wc -l <<<"$FILES") files, $(du -h "$OUT" | cut -f1), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (+uncommitted changes)')"
