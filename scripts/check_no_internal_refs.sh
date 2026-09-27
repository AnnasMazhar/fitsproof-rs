#!/usr/bin/env bash
# Fail if internal system names or host paths appear in tracked files.
# Same rule as the Python edition's scripts/check_no_internal_refs.py, no Python required.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# Tokens are assembled from fragments so this file does not itself contain them.
T1="plu""tus"; T2="oly""mpus"; T3="annas""claw"; T4="sa""ts"
PATTERN="\\b(${T1}|${T2}|${T3}|${T4}|v3)\\b|/home/open""claw/|\\.open""claw|workspace-${T1}"

hits=$(git ls-files -z | xargs -0 grep -nIE "$PATTERN" 2>/dev/null \
  | grep -v '^scripts/check_no_internal_refs.sh:' || true)

if [ -n "$hits" ]; then
  echo "FORBIDDEN internal reference(s) found:"
  echo "$hits"
  exit 1
fi
echo "check_no_internal_refs: CLEAN"
