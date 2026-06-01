#!/usr/bin/env bash
# One-command NIDS runner — wrapper around scripts/run_experiment.py
#
# Usage:
#   ./run.sh                          # both splits, full data, 30% sample
#   ./run.sh both full 0.3            # same, positional args
#   ./run.sh file full 0.3            # file-based only -> output2/
#   ./run.sh random full 0.3          # random only -> output/
#   ./run.sh both small 0.1 --fast    # quick test
#
# Positional: [split] [size] [sample]
#   split  = random | file | both   (default: both)
#   size   = small | medium | full  (default: full)
#   sample = 0.05 - 1.0             (default: 0.3)

set -e
cd "$(dirname "$0")"

SPLIT="${1:-both}"
SIZE="${2:-full}"
SAMPLE="${3:-0.3}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true

if [[ -f env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source env/bin/activate
fi

exec python scripts/run_experiment.py --split "$SPLIT" --size "$SIZE" --sample "$SAMPLE" "$@"
