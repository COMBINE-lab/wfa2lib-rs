# wfa2lib-rs

**A pure Rust port of [WFA2-lib](https://github.com/smarco/WFA2-lib)**, the wavefront alignment library for pairwise sequence alignment.

This project was created with [Claude Code](https://claude.ai/claude-code).

## Overview

wfa2lib-rs is a from-scratch Rust implementation of the Wavefront Alignment (WFA) algorithm. WFA exploits the structure of sequence similarity to achieve O(ns) time complexity — proportional to the sequence length (n) and the alignment score (s) — making it significantly faster than traditional O(n²) dynamic programming approaches for similar sequences.

This implementation covers the full WFA feature set:

- **Distance metrics**: indel (LCS), edit (Levenshtein), gap-linear (Needleman-Wunsch), gap-affine (Smith-Waterman-Gotoh), and gap-affine 2-piece (concave penalties)
- **Alignment modes**: end-to-end (global), ends-free (semi-global), and extension alignment
- **BiWFA**: bidirectional WFA for O(s) memory alignment with full CIGAR output
- **Heuristics**: WF-adaptive, X-drop, Z-drop, static banding, and adaptive banding for pruning divergent sequences
- **Sequence representations**: ASCII byte sequences, packed 2-bit DNA encoding, and a lambda callback interface for custom sequence access
- **SIMD acceleration**: NEON-vectorized compute kernels on aarch64 (4 diagonals/iteration), AVX2 on x86_64 (8 diagonals/iteration), with scalar fallback
- **Score and alignment**: score-only mode for speed, or full CIGAR traceback with support for match rewards via Eizenga's formula

## Performance

Benchmarked against the C reference implementation (WFA2-lib), 50,000 sequence pairs of 1000 bp at 5% divergence:

### x86_64 (Intel Broadwell, AVX2)

| Benchmark | Rust | C | Ratio |
|---|---|---|---|
| Edit distance (score) | 405 ms | 446 ms | **0.91x** |
| Gap-linear (score) | 682 ms | 1090 ms | **0.63x** |
| Gap-affine (score) | 3980 ms | 3970 ms | **1.00x** |
| Gap-affine (CIGAR) | 5440 ms | 5380 ms | 1.01x |
| Gap-affine 2-piece (score) | 16800 ms | 19710 ms | **0.85x** |
| Gap-affine 2-piece (CIGAR) | 20620 ms | 22070 ms | **0.93x** |
| BiWFA (ultralow memory) | 7133 ms | 6660 ms | 1.07x |

### aarch64 (Apple M-series, NEON)

| Benchmark | Rust vs C | Notes |
|---|---|---|
| Edit distance (score) | **0.56x** | 1.8× faster |
| Gap-linear (score) | **0.68x** | 1.5× faster |
| Gap-affine (score) | **0.85x** | 1.2× faster |
| Gap-affine (CIGAR) | **0.90x** | 1.1× faster |
| Gap-affine 2-piece (score) | **0.88x** | 1.1× faster |
| Gap-affine 2-piece (CIGAR) | **0.86x** | 1.2× faster |
| BiWFA (ultralow memory) | **0.86x** | 1.2× faster |

Ratio < 1.0 means Rust is faster. Key optimizations include hand-written AVX2/NEON SIMD compute kernels (8 or 4 diagonals/iteration) with software prefetching, pre-centered wavefront offset pointers (matching C's `offsets = offsets_mem - min_lo` pattern), separate per-component wavefront arrays for optimal cache stride, arena bump allocation with cache coloring to eliminate L1 conflict misses, conditional kernel dispatch (affine fallback when O2/E2 inputs are null), wavefront reuse between alignments, and blockwise 64-bit sequence extension.

## Building

Requires Rust 1.91+ (edition 2024).

```bash
# Debug build
cargo build

# Optimized release build (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run tests (228 tests: 210 unit + 18 integration)
cargo test
```

## Usage

### As a library

```rust
use wfa2lib_rs::aligner::{AlignmentScope, WavefrontAligner};
use wfa2lib_rs::penalties::{AffinePenalties, DistanceMetric, WavefrontPenalties};

// Configure gap-affine penalties (mismatch=4, gap_open=6, gap_extend=2)
let penalties = WavefrontPenalties::new_affine(AffinePenalties {
    match_: 0,
    mismatch: 4,
    gap_opening: 6,
    gap_extension: 2,
});

let mut aligner = WavefrontAligner::new(penalties);
aligner.alignment_scope = AlignmentScope::ComputeAlignment;

let pattern = b"TCTTTACTCGCGCGTTGGAGAAATACAATAGT";
let text    = b"TCTATACTGCGCGTTTGGAGAAATAAAATAGT";

let score = aligner.align_end2end(pattern, text);
println!("Score: {}", score);
println!("CIGAR: {}", aligner.cigar);
```

### CLI benchmark tool

The included `align_benchmark` binary is compatible with the WFA2-lib reference interface:

```bash
# Gap-affine alignment
cargo run --release --bin align_benchmark -- \
    -a gap-affine-wfa \
    -g 0,4,6,2 \
    -i sequences.seq

# BiWFA (memory-efficient)
cargo run --release --bin align_benchmark -- \
    -a gap-affine-wfa \
    -g 0,4,6,2 \
    --wfa-memory ultralow \
    -i sequences.seq

# Score-only mode
cargo run --release --bin align_benchmark -- \
    -a gap-affine-wfa \
    -g 0,4,6,2 \
    --wfa-score-only \
    -i sequences.seq
```

Input files use the WFA2-lib format: one alignment pair per line, with pattern and text separated by a `>` character.

## Project structure

```
src/
  lib.rs              # Public module exports
  aligner.rs          # Main WavefrontAligner struct and alignment loops
  penalties.rs        # Distance metrics and penalty configurations
  sequences.rs        # Sequence representations (ASCII, packed 2-bit, lambda)
  wavefront.rs        # Wavefront data structure
  wavefront_set.rs    # Grouped wavefront sets (M, I, D components)
  slab.rs             # Wavefront memory pool with arena allocation
  arena.rs            # Byte-oriented bump allocator for wavefront arrays
  compute/            # WFA compute kernels (edit, linear, affine, affine2p)
  extend.rs           # Wavefront extend with SIMD acceleration
  backtrace.rs        # CIGAR traceback from wavefronts
  bialign.rs          # Bidirectional WFA (BiWFA)
  cigar.rs            # CIGAR operations and scoring
  heuristic.rs        # Pruning heuristics
  offset.rs           # Wavefront offset type and coordinate helpers
  pcigar.rs           # Packed CIGAR for BiWFA backtrace
  bt_buffer.rs        # Backtrace buffer for BiWFA
  components.rs       # Wavefront component management
  termination.rs      # Alignment termination conditions
  bin/
    align_benchmark.rs  # CLI binary
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Citations

If you use this software, please cite the original WFA papers:

**WFA — Wavefront Alignment**:
> Santiago Marco-Sola, Juan Carlos Moure, Miquel Moreto, Antonio Espinosa. "Fast gap-affine pairwise alignment using the wavefront algorithm." *Bioinformatics*, Volume 37, Issue 4, February 2021, Pages 456–463. https://doi.org/10.1093/bioinformatics/btaa777

**BiWFA — Bidirectional WFA**:
> Santiago Marco-Sola, Jordan M Eizenga, Andrea Guarracino, Benedict Paten, Erik Garrison, Miquel Moreto. "Optimal gap-affine alignment in O(s) space." *Bioinformatics*, Volume 39, Issue 2, February 2023, btad074. https://doi.org/10.1093/bioinformatics/btad074
