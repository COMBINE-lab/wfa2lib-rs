#!/usr/bin/env bash
# Generate benchmark data files once. Does NOT regenerate if files already exist.
# Re-run manually with --force to regenerate (this will invalidate prior C vs Rust comparisons).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
GENERATOR="$SCRIPT_DIR/../WFA2-lib/bin/generate_dataset"

FORCE=0
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE=1
done

if [[ ! -x "$GENERATOR" ]]; then
  echo "ERROR: $GENERATOR not found. Build WFA2-lib first: cd WFA2-lib && make"
  exit 1
fi

mkdir -p "$DATA_DIR"

generate_if_missing() {
  local file="$DATA_DIR/$1"
  local args=("${@:2}")
  if [[ -f "$file" && "$FORCE" -eq 0 ]]; then
    echo "  [exists] $1"
  else
    echo "  [generating] $1 ..."
    "$GENERATOR" "${args[@]}" -o "$file"
    echo "  [done] $1 ($(wc -l < "$file") lines)"
  fi
}

echo "Benchmark data generation (stable — run with --force to regenerate)"
echo "===================================================================="

# Standard benchmark: 1000bp sequences, 5% error, 50k pairs
# (C uses ~1-10s range; enough pairs for stable timing)
generate_if_missing bench_1k_5pct_50k.seq  -n 50000 -l 1000 -e 0.05

# Longer sequences for BiWFA (more interesting divide-and-conquer)
generate_if_missing bench_5k_2pct_10k.seq  -n 10000 -l 5000 -e 0.02

echo ""
echo "Data files are in: $DATA_DIR"
echo "To benchmark, run: bench/run.sh"
