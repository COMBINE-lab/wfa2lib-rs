#!/bin/bash
# Rust WFA2 unitary tests — compares align_benchmark output against C reference
# Usage: ./tests/wfa_utest.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$PROJECT_DIR/target/release/align_benchmark"
INPUT="$PROJECT_DIR/WFA2-lib/tests/wfa.utest.seq"
CHECK="$PROJECT_DIR/WFA2-lib/tests/wfa.utest.check"
TMPDIR=$(mktemp -d)

trap "rm -rf $TMPDIR" EXIT

# Build
echo ">>> Building release binary..."
cargo build --release --quiet 2>&1

if [ ! -f "$BIN" ]; then
    echo "[Error] Binary not found at $BIN"
    exit 1
fi

PASS=0
FAIL=0
SCORE_ONLY_OK=0
ERRORS=""

# Compare function: exact diff first, then score-only
compare() {
    local output="$1"
    local reference="$2"
    local name="$3"

    if [ ! -f "$reference" ]; then
        echo "  SKIP $name (no reference file)"
        return
    fi

    if diff -q "$output" "$reference" > /dev/null 2>&1; then
        echo "  OK   $name"
        PASS=$((PASS + 1))
    else
        # Try score-only comparison (absolute values of first column)
        if diff <(awk '{if ($1<0) print -$1; else print $1}' "$output") \
                <(awk '{if ($1<0) print -$1; else print $1}' "$reference") > /dev/null 2>&1; then
            echo "  ok   $name (scores match, CIGARs differ)"
            SCORE_ONLY_OK=$((SCORE_ONLY_OK + 1))
            PASS=$((PASS + 1))
        else
            echo "  FAIL $name"
            FAIL=$((FAIL + 1))
            ERRORS="$ERRORS\n  $name"
            # Show first few diffs
            diff "$output" "$reference" | head -6
        fi
    fi
}

# Run one test combo
run_test() {
    local mode_flags="$1"
    local name="$2"
    local algo="$3"
    local suffix="$4"
    local extra_flags="$5"

    local outfile="$TMPDIR/${name}.${suffix}.alg"
    local reffile="$CHECK/${name}.${suffix}.alg"

    $BIN -i "$INPUT" -o "$outfile" -a "$algo" $mode_flags $extra_flags 2>/dev/null
    compare "$outfile" "$reffile" "${name}.${suffix}"
}

# Run all 12 combos for a given mode
run_mode() {
    local mode_flags="$1"
    local name="$2"
    echo ">>> Testing '$name' ($mode_flags)"

    # Distance functions
    run_test "$mode_flags" "$name" "indel-wfa" "indel"
    run_test "$mode_flags" "$name" "edit-wfa" "edit"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine"
    run_test "$mode_flags" "$name" "gap-affine2p-wfa" "affine2p"

    # Custom penalties
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p0" "--affine-penalties=0,1,2,1"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p1" "--affine-penalties=0,3,1,4"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p2" "--affine-penalties=0,5,3,2"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p3" "--affine-penalties=-5,1,2,1"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p4" "--affine-penalties=-2,3,1,4"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.p5" "--affine-penalties=-3,5,3,2"

    # Heuristics
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.wfapt0" "--wfa-heuristic=wfa-adaptive --wfa-heuristic-parameters=10,50,1"
    run_test "$mode_flags" "$name" "gap-affine-wfa" "affine.wfapt1" "--wfa-heuristic=wfa-adaptive --wfa-heuristic-parameters=10,50,10"
}

# Run all 5 modes
run_mode "--check=correct" "test"
run_mode "--wfa-score-only" "test.score"
run_mode "--wfa-memory=med --check=correct" "test.pb"
run_mode "--wfa-memory=ultralow --check=correct" "test.biwfa"
run_mode "--wfa-memory=ultralow --wfa-score-only" "test.biwfa.score"

# Summary
echo ""
echo "========================================="
echo "Results: $PASS passed, $FAIL failed"
if [ $SCORE_ONLY_OK -gt 0 ]; then
    echo "  ($SCORE_ONLY_OK passed with score-only match)"
fi
if [ $FAIL -gt 0 ]; then
    echo -e "Failed tests:$ERRORS"
    exit 1
else
    echo ">>> ALL GOOD!"
    exit 0
fi
