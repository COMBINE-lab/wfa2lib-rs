#!/usr/bin/env bash
# Run all benchmarks: C reference vs Rust, on identical data files.
# Both tools read the same .seq file, ensuring fair comparison.
#
# Usage:
#   bench/run.sh                # all modes, standard dataset
#   bench/run.sh --fast         # 5000 pairs for quick iteration
#   bench/run.sh edit affine    # run only specified modes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
DATA_DIR="$SCRIPT_DIR/data"

RUST_BIN="$ROOT/target/release/align_benchmark"
C_BIN="$ROOT/WFA2-lib/bin/align_benchmark"

SEQ_FILE="$DATA_DIR/bench_1k_5pct_50k.seq"
CLEANUP_TMP=""

# Parse args
FAST=0
MODES=()
for arg in "$@"; do
  if [[ "$arg" == "--fast" ]]; then FAST=1
  else MODES+=("$arg")
  fi
done

# Default: all modes
if [[ ${#MODES[@]} -eq 0 ]]; then
  MODES=(edit linear affine affine-cigar biwfa affine2p affine2p-cigar)
fi

# Validate prerequisites
if [[ ! -f "$SEQ_FILE" ]]; then
  echo "ERROR: Benchmark data missing. Run: bench/generate_data.sh"
  exit 1
fi
if [[ ! -x "$RUST_BIN" ]]; then
  echo "ERROR: Rust binary not found. Build with:"
  echo "  RUSTFLAGS='-C target-cpu=native' cargo build --release"
  exit 1
fi
if [[ ! -x "$C_BIN" ]]; then
  echo "ERROR: C binary not found. Build with: cd WFA2-lib && make"
  exit 1
fi

# Optionally limit sequence count for fast runs
if [[ "$FAST" -eq 1 ]]; then
  TMPF=$(mktemp /tmp/wfa_bench_fast.XXXXXX)
  CLEANUP_TMP="$TMPF"
  head -10000 "$SEQ_FILE" > "$TMPF"
  SEQ_FILE="$TMPF"
  PAIRS=5000
  trap "rm -f $CLEANUP_TMP" EXIT
  echo "[fast mode: using 5000 pairs]"
else
  PAIRS=$(( $(wc -l < "$SEQ_FILE") / 2 ))
fi

echo "Sequence file: $SEQ_FILE ($PAIRS pairs)"
echo ""

# Extract timing in ms from output.
# Rust: "Time.Alignment 2308.18 ms ..."   → $1=key $2=value $3=unit
# C:    "=> Time.Benchmark 243.78 ms ..."  → $1="=>" $2=key $3=value $4=unit
#        or "=> Time.Benchmark 2.43 s ..."  → same positions, unit may be s or ms
extract_ms() {
  awk '
    /^Time\.Alignment/ {
      # Rust: first field is key, second is value, third is unit
      val = $2 + 0
      if ($3 == "s") val *= 1000
      printf "%.2f", val; exit
    }
    /Time\.Benchmark/ {
      # C: "=> Time.Benchmark VALUE UNIT ..."
      val = $3 + 0
      if ($4 == "s") val *= 1000
      printf "%.2f", val; exit
    }
  '
}

# Run benchmark, return minimum of two runs (ms).
bench_ms() {
  local bin="$1"; shift
  local args=("$@")
  local t1="" t2=""
  t1=$("$bin" -i "$SEQ_FILE" "${args[@]}" 2>&1 | extract_ms)
  t2=$("$bin" -i "$SEQ_FILE" "${args[@]}" 2>&1 | extract_ms)
  awk -v a="$t1" -v b="$t2" 'BEGIN { print (a < b) ? a : b }'
}

printf "%-22s %10s %10s %8s\n" "Mode" "C (ms)" "Rust (ms)" "Ratio"
printf "%-22s %10s %10s %8s\n" "----" "------" "---------" "-----"

run_mode() {
  local label="$1"; shift
  local args=("$@")
  local c_ms rust_ms ratio
  c_ms=$(bench_ms "$C_BIN"    "${args[@]}")
  rust_ms=$(bench_ms "$RUST_BIN" "${args[@]}")
  ratio=$(awk -v c="$c_ms" -v r="$rust_ms" 'BEGIN { printf "%.3fx", r/c }')
  printf "%-22s %10.0f %10.0f %8s\n" "$label" "$c_ms" "$rust_ms" "$ratio"
}

for mode in "${MODES[@]}"; do
  case "$mode" in
    edit)
      run_mode "Edit Score"         -a edit-wfa --wfa-score-only ;;
    linear)
      run_mode "Linear Score"       -a gap-linear-wfa --wfa-score-only ;;
    affine)
      run_mode "Affine Score"       -a gap-affine-wfa --wfa-score-only ;;
    affine-cigar)
      run_mode "Affine CIGAR"       -a gap-affine-wfa ;;
    biwfa)
      run_mode "BiWFA (ultralow)"   -a gap-affine-wfa --wfa-memory ultralow ;;
    affine2p)
      run_mode "Affine2p Score"     -a gap-affine2p-wfa --wfa-score-only ;;
    affine2p-cigar)
      run_mode "Affine2p CIGAR"     -a gap-affine2p-wfa ;;
    *)
      echo "Unknown mode: $mode (valid: edit linear affine affine-cigar biwfa affine2p affine2p-cigar)"
      exit 1 ;;
  esac
done
