# WFA2-lib Rust Port: Implementation Plan

## Context

This project ports [WFA2-lib](https://github.com/smarco/WFA2-lib) — a C/C++ library for time/space-efficient sequence alignment using the Wavefront Alignment (WFA) algorithm — to idiomatic Rust. The goal is **output-identical**, **performance-competitive**, **feature-complete** Rust code with an ergonomic API. The C reference lives in `WFA2-lib/` and its test suite (`WFA2-lib/tests/wfa.utest.sh`) provides our correctness oracle.

---

## Rust Module Structure

```
src/
  lib.rs                      -- Public API re-exports
  penalties.rs                 -- DistanceMetric, penalty structs
  cigar.rs                     -- CIGAR struct, scoring, display
  pcigar.rs                    -- Packed CIGAR (u32-based)
  offset.rs                    -- WfOffset type, coordinate helpers
  sequences.rs                 -- Sequence abstraction (ascii, lambda, packed2bits)
  wavefront.rs                 -- Single Wavefront (offsets, bt fields)
  wavefront_set.rs             -- WavefrontSet (input/output refs by index)
  slab.rs                      -- WavefrontSlab (allocator/recycler)
  components.rs                -- WavefrontComponents (M/I/D arrays, bt_buffer)
  bt_buffer.rs                 -- BacktraceBuffer (bt_block segments, compaction)
  compute/
    mod.rs                     -- Shared helpers (fetch, allocate, trim)
    edit.rs                    -- Edit/indel compute kernels
    linear.rs                  -- Gap-linear compute kernel
    affine.rs                  -- Gap-affine compute kernel
    affine2p.rs                -- Gap-affine 2-piece compute kernel
  extend.rs                    -- Extend kernels (packed e2e, endsfree, custom)
  termination.rs               -- End2end and endsfree termination checks
  backtrace.rs                 -- Linear and affine wavefront backtrace
  backtrace_offload.rs         -- Piggyback backtrace offloading
  heuristic.rs                 -- Heuristic strategies (banded, adaptive, xdrop, zdrop)
  aligner.rs                   -- WavefrontAligner (main state + public API)
  unialign.rs                  -- Unidirectional alignment loop
  bialigner.rs                 -- BiWFA bidirectional aligner
  bialign.rs                   -- BiWFA breakpoint detection + divide-and-conquer
  plot.rs                      -- Wavefront plotting (optional)
  display.rs                   -- Debug display utilities
  utils/
    mod.rs
    dna.rs                     -- 2-bit packed DNA utilities
  bin/
    align_benchmark.rs         -- CLI driver binary
```

---

## Key Design Decisions

| Decision | Approach | Rationale |
|----------|----------|-----------|
| **Wavefront ownership** | Arena/slab pattern: `WavefrontSlab` owns all `Wavefront` objects; other structures hold indices (`usize`) | Avoids borrow-checker fights while staying safe |
| **K-centered indexing** | `offsets[(k - base_k) as usize]` with `#[inline(always)]` accessors; `unsafe` unchecked indexing in validated hot loops | Mirrors C pointer arithmetic safely |
| **Function dispatch** | **Generics (monomorphization)** over an `AlignKernel` trait; one-time `match` at top level selects the monomorphized path | C uses function pointers (indirect call per loop iteration). Generics let the compiler specialize + inline the compute/extend kernels into the hot loop — **zero-overhead dispatch**, beating C's indirect-call model. ~10 specializations (5 metrics x 2 spans) is modest code bloat. |
| **Error handling** | `Result<AlignStatus, AlignError>` public API; internal status codes match C | Clean Rust API wrapping C-style status |
| **Memory modes** | `high`/`med`/`low` = modular + piggyback flags; `ultralow` = BiWFA (separate aligner) | Same architecture as C |

---

## Phases

### Phase 1: Foundation — Types, CIGAR, Project Scaffolding

**Create:**
- `Cargo.toml` — add `tracing`, `tracing-subscriber`
- `.cargo/config.toml` — `target-cpu=native`
- `src/offset.rs` — `WfOffset = i32`, `OFFSET_NULL`, coordinate helpers (`wavefront_v`, `wavefront_h`)
- `src/penalties.rs` — `DistanceMetric` enum, `LinearPenalties`, `AffinePenalties`, `Affine2pPenalties`, unified `WavefrontPenalties`
- `src/cigar.rs` — `Cigar` struct with `Vec<u8>` ops buffer, scoring under all metrics, `check_alignment`, `Display`
- `src/pcigar.rs` — `Pcigar = u32`, push/pop/extract operations
- `src/lib.rs` — module declarations

**Validate:** `cargo build && cargo clippy && cargo test` — unit tests for penalty construction, CIGAR scoring, pcigar round-trips

**Reference files:** `WFA2-lib/alignment/cigar.c`, `WFA2-lib/alignment/*_penalties.h`, `WFA2-lib/wavefront/wavefront_offset.h`, `WFA2-lib/wavefront/wavefront_pcigar.h`

---

### Phase 2: Wavefront Data Structures

**Create:**
- `src/wavefront.rs` — `Wavefront` struct (offsets vec, lo/hi, bt_pcigar, bt_prev, status)
- `src/slab.rs` — `WavefrontSlab` (allocation, free-list recycling, reap)
- `src/bt_buffer.rs` — `BacktraceBuffer`, `BtBlock`, block allocation and traceback
- `src/wavefront_set.rs` — `WavefrontSet` (slab indices for input/output wavefront groups)
- `src/components.rs` — `WavefrontComponents` (M/I/D wavefront arrays as `Vec<Option<usize>>`, historic min/max)
- `src/sequences.rs` — `WavefrontSequences` (ASCII mode first; lambda/packed2bits stubs)

**Validate:** Unit tests for slab alloc/free cycles, k-centered indexing, bt_buffer block operations, sequence comparison

**Reference files:** `WFA2-lib/wavefront/wavefront.h`, `WFA2-lib/wavefront/wavefront_slab.c`, `WFA2-lib/wavefront/wavefront_backtrace_buffer.c`, `WFA2-lib/wavefront/wavefront_components.c`

---

### Phase 3: Edit Distance — Score Only, End-to-End

**First end-to-end alignment path.** Create:
- `src/compute/mod.rs` — shared helpers: trim ends, compute limits, fetch input, allocate output
- `src/compute/edit.rs` — `wavefront_compute_edit_idm`, `wavefront_compute_indel_idm`
- `src/extend.rs` — `wavefront_extend_matches_packed_end2end` (8-byte-at-a-time XOR matching)
- `src/termination.rs` — `wavefront_termination_end2end`
- `src/aligner.rs` — `WavefrontAligner::new()`, `align()`, `score()`
- `src/unialign.rs` — main loop: init → extend → terminate? → compute next score → check limits

**Validate:** Parse `WFA2-lib/tests/wfa.utest.seq`, run edit-distance score-only on all pairs, compare against `test.score.edit.alg` reference output. **This is the first end-to-end correctness checkpoint.**

**Reference files:** `WFA2-lib/wavefront/wavefront_compute_edit.c`, `WFA2-lib/wavefront/wavefront_extend.c`, `WFA2-lib/wavefront/wavefront_unialign.c`

---

### Phase 4: Edit Distance with CIGAR (Backtrace)

**Add:**
- `src/backtrace.rs` — `wavefront_backtrace_linear` (walk backwards through M-wavefronts)
- `src/backtrace_offload.rs` — piggyback block offloading for linear distance
- Update `compute/edit.rs` — add piggyback variants of compute kernels
- Update `unialign.rs` — dispatch to backtrace or pcigar-unpack depending on memory mode

**Validate:** Compare full CIGAR output against `test.edit.alg` (high memory) and `test.pb.edit.alg` (piggyback). Verify `cigar.check_alignment()` passes. Also validate indel mode (`test.indel.alg`, `test.score.indel.alg`).

---

### Phase 5: Gap-Linear and Gap-Affine Kernels

**Create:**
- `src/compute/linear.rs` — gap-linear kernel (M + I/D wavefronts)
- `src/compute/affine.rs` — gap-affine kernel (M + I1 + D1 wavefronts) — **most performance-critical inner loop**
- Update `src/backtrace.rs` — `wavefront_backtrace_affine` (walk M/I/D wavefronts)
- Update `src/bt_buffer.rs` — `unpack_cigar_affine`

**Validate:** Compare against `test.affine.alg` (default penalties M=0,X=4,O=6,E=2), `test.affine.p0.alg` through `test.affine.p5.alg` (varied penalties), `test.score.affine.alg`, `test.pb.affine.alg`. Also gap-linear equivalents.

**Reference files:** `WFA2-lib/wavefront/wavefront_compute_affine.c`, `WFA2-lib/wavefront/wavefront_backtrace.c`

---

### Phase 6: Gap-Affine 2-Piece Kernel

**Create:**
- `src/compute/affine2p.rs` — 5-component kernel (M, I1, I2, D1, D2)

**Validate:** Compare against `test.affine2p.alg`, `test.score.affine2p.alg`, `test.pb.affine2p.alg`

---

### Phase 7: Ends-Free and Extension Alignment

**Modify:**
- `src/extend.rs` — add `wavefront_extend_matches_packed_endsfree`
- `src/termination.rs` — add `wavefront_termination_endsfree`
- `src/aligner.rs` — add `set_alignment_free_ends(...)`, `set_alignment_extension()`
- `src/unialign.rs` — handle extension CIGAR maxtrimming

**Validate:** Create test cases with known ends-free alignments; compare against C version output

---

### Phase 8: Heuristics

**Create:**
- `src/heuristic.rs` — `HeuristicStrategy` enum (None, BandedStatic, BandedAdaptive, WfAdaptive, WfMash, XDrop, ZDrop), `wavefront_heuristic_cutoff`

**Validate:** Compare against `test.affine.wfapt0.alg` (wfa-adaptive 10,50,1) and `test.affine.wfapt1.alg` (wfa-adaptive 10,50,10)

---

### Phase 9: BiWFA (Ultralow Memory)

**Create:**
- `src/bialigner.rs` — forward + reverse + subsidiary aligners
- `src/bialign.rs` — breakpoint detection, divide-and-conquer recursion (~largest module)

**Validate:** Compare against `test.biwfa.edit.alg`, `test.biwfa.indel.alg`, `test.biwfa.affine.alg`, `test.biwfa.affine2p.alg` and all penalty variants. BiWFA scores must match regular scores exactly.

**Reference files:** `WFA2-lib/wavefront/wavefront_bialign.c`, `WFA2-lib/wavefront/wavefront_bialigner.c`

---

### Phase 10: Lambda/Custom Match and Packed 2-Bit DNA

**Modify:**
- `src/sequences.rs` — complete lambda and packed2bits modes
- `src/extend.rs` — add `wavefront_extend_matches_custom`
- `src/utils/dna.rs` — 2-bit packed encoding/decoding

**Validate:** Lambda with byte-identical comparator must produce same output as ASCII. Packed2bits with known DNA pairs must match ASCII results.

---

### Phase 11: CLI Binary Driver

**Create:**
- `src/bin/align_benchmark.rs` — `clap`-based CLI matching C version's interface
- Add `clap` to `Cargo.toml`, `[[bin]]` target

**CLI interface:** `-a ALGORITHM -i INPUT [--wfa-score-only] [--wfa-span] [--wfa-memory] [-p M,X,I] [-g M,X,O,E] [--affine2p-penalties ...] [--wfa-heuristic ...] [-c check] [-v verbose]`

**Validate:** Port `wfa.utest.sh` to use Rust binary; diff all `.alg` outputs against reference. **This is the comprehensive correctness gate.**

---

### Phase 12: Performance Optimization

**Focus areas:**
1. **Extend kernel** — ensure 8-byte XOR matching is optimized; consider `unsafe` unchecked indexing in hot loops after bounds validation; inspect generated asm with `cargo asm`
2. **Slab allocation** — profile reuse efficiency, pre-size vectors
3. **Memory layout** — profile cache misses, ensure struct layout is cache-friendly
4. **Benchmarking** — use `criterion` for microbenchmarks, `align_benchmark` for end-to-end; log to `.claude/PERFORMANCE_LOG.md`
5. **(Stretch) SIMD extend** — AVX2/NEON via `core::arch` intrinsics if scalar Rust is slower than scalar C

**Targets:** Runtime within 2x of C version; memory within 1.5x; no correctness regressions.

---

### Phase 13: Polish

- Wavefront plotting (`src/plot.rs`)
- Debug display utilities (`src/display.rs`)
- Rustdoc on all public types/methods
- `examples/` directory
- Final `clippy` + `cargo fmt` pass
- Set up all tracking files (`.claude/PROGRESS.md`, `.claude/FIXED_BUGS.md`, `.claude/PERFORMANCE_LOG.md`)

---

## Dependency Graph

```
Phase 1 (Foundation)
  → Phase 2 (Data Structures)
    → Phase 3 (Edit score-only)
      → Phase 4 (Edit + CIGAR)
        → Phase 5 (Linear + Affine)
          → Phase 6 (Affine 2-piece)
            → Phase 7 (Ends-free/Extension)  ─┐
            → Phase 8 (Heuristics)            ─┤ (partially parallelizable)
            → Phase 9 (BiWFA)                 ─┘
              → Phase 10 (Lambda/Packed2bits)
                → Phase 11 (CLI Binary)
                  → Phase 12 (Performance)
                    → Phase 13 (Polish)
```

## Verification Strategy

Each phase validates against the C reference outputs in `WFA2-lib/tests/wfa.utest.check/`. The final gate (Phase 11) runs the full `wfa.utest.sh` test suite with our Rust binary. Performance is tracked in `.claude/PERFORMANCE_LOG.md` with git commit references.
