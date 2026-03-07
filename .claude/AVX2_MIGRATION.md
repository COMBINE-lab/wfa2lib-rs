# WFA2-lib Rust — AVX2 Migration Guide

## Current State (2026-03-06, Apple Silicon / NEON)

All modes at parity or faster than C reference (WFA2-lib):

| Benchmark | Ratio | Notes |
|-----------|-------|-------|
| Edit Score | **0.59x** | 1.7x faster (NEON SIMD) |
| Linear Score | **0.69x** | 1.4x faster |
| Affine Score | **0.94x** | 1.06x faster |
| Affine CIGAR | **0.95x** | 1.06x faster |
| BiWFA (ultralow) | **0.92x** | 1.09x faster |
| Affine2p Score | **1.00x** | Parity |
| Affine2p CIGAR | 1.03x | ~3% structural gap |

Ratio < 1.0 = Rust faster. Benchmark: 50k pairs, 1000bp, 5% divergence.

## Key Optimizations Already Applied

1. **SIMD compute kernels** — NEON (aarch64) and AVX2 (x86_64) in all four kernels
2. **Affine kernel dispatcher** — affine2p falls back to 3-component affine kernel when O2/E2 inputs are null (saves ~40% kernel work per step)
3. **Conditional I2/D2 allocation** — skips allocation when both inputs null, uses victim wavefront
4. **Blockwise 64-bit extend** — XOR + trailing zeros, 8 chars/iteration
5. **Chunked OffsetArena** — bump allocator for wavefront offsets
6. **Slab reuse** — `reuse_or_allocate_ptr` eliminates free→allocate round-trip in modular mode
7. **BiWFA sub-aligner caching** — lazy Option<Box<...>> avoids repeated heap allocation
8. **Const-generic WavefrontComponents<N>** — N=1/3/5 for edit/affine/affine2p
9. **Interleaved flat array** — `flat[score] = [*mut Wavefront; N]` for cache-friendly access
10. **Partial clear optimization** — only clears `flat[0..prev_score_count]` between alignments

## AVX2 Kernel Files

All SIMD code is in `src/compute/`:

- `src/compute/edit.rs` — Edit distance kernel (NEON + AVX2 + scalar)
- `src/compute/linear.rs` — Gap-linear kernel (NEON + AVX2 + scalar)
- `src/compute/affine.rs` — Gap-affine kernel (NEON + AVX2 + scalar), ~337 lines
- `src/compute/affine2p.rs` — Gap-affine 2-piece kernel (NEON + AVX2 + scalar), ~290 lines

Each kernel has the same structure:
```
#[cfg(target_arch = "aarch64")] { NEON block, 4 diagonals/iter }
#[cfg(target_arch = "x86_64")] { AVX2 block, 8 diagonals/iter, runtime feature check }
// Scalar tail handles remaining diagonals
```

The extend kernel (`src/extend.rs`) does NOT have AVX2 — it uses blockwise 64-bit scalar comparison which is already very fast. SIMD extend would require gather/scatter for diagonal-indexed sequence access and is unlikely to help.

## AVX2-Specific Notes

- AVX2 processes **8 diagonals per iteration** (vs NEON's 4)
- Uses `_mm256_loadu_si256` / `_mm256_storeu_si256` (unaligned) for offset loads/stores
- Unsigned comparison trick: XOR with `i32::MIN` to convert signed `_mm256_cmpgt_epi32` to unsigned compare (needed for bounds checking)
- `_mm256_blendv_epi8` for conditional null replacement (vs NEON's `vbslq_s32`)
- Runtime check: `is_x86_feature_detected!("avx2")` — falls back to scalar if not available
- The AVX2 kernels were written but **never profiled or optimized** on real AVX2 hardware

## What to Test on AVX2 Machine

### 1. Correctness
```bash
cargo test                    # 228 tests
bash tests/wfa_utest.sh      # 60/60 reference tests (needs WFA2-lib submodule)
```

### 2. Benchmark
```bash
# Build optimized
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build C reference
cd WFA2-lib && make clean && make all && cd ..

# Run benchmarks (needs bench/data/ files — see below)
bash bench/run.sh
```

### 3. Profile
```bash
# Linux perf (much better than macOS `sample`)
perf record -g ./target/release/align_benchmark -a gap-affine2p-wfa -g 0,7,4,2,24,1 -i bench/data/bench_1k_5pct_50k.seq --wfa-score-only
perf report

# Or flamegraph
cargo install flamegraph
RUSTFLAGS="-C target-cpu=native" cargo flamegraph --release --bin align_benchmark -- -a gap-affine2p-wfa -g 0,7,4,2,24,1 -i bench/data/bench_1k_5pct_50k.seq --wfa-score-only
```

## Potential AVX2 Optimizations to Investigate

1. **AVX2 extend kernel** — the extend phase is the hottest function. Could use `_mm256_cmpeq_epi8` for 32-byte comparison blocks. Challenge: sequences are indexed diagonally (different offsets per diagonal), so would need gather or careful alignment.

2. **AVX-512** — if available, 16 diagonals/iter. Check `is_x86_feature_detected!("avx512f")`. Would need new kernel variants.

3. **Prefetching** — `_mm_prefetch` for upcoming wavefront data. Affine2p's 130-wavefront working set exceeds L1; prefetch could help.

4. **Alignment of offset arrays** — currently unaligned loads. If arena allocations are 32-byte aligned, could use `_mm256_load_si256` (may help on some microarchitectures).

## Remaining Performance Gap

The ~3% affine2p CIGAR gap is structural:
- Score-only working set: ~20KB (fits L1)
- CIGAR working set: ~700KB (L2/L3)
- Non-modular mode allocates every wavefront (no reuse), increasing allocation overhead
- This is inherent to the algorithm, not a code issue

## Build Requirements

- Rust 1.91+ (edition 2024)
- `Cargo.toml` profile: `lto = true, codegen-units = 1, panic = "abort"`
- C reference: `cd WFA2-lib && make all` (may need `make` flags for Linux)
- WFA2-lib is a git submodule — `git submodule update --init`
