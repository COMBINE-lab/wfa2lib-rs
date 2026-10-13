# Performance Log

Benchmarks tied to git commits, as asked for in `PROJECT.md`. Check here before
merging a change that touches the compute, extend or allocation paths, to make
sure it doesn't introduce a regression.

**Method:** `bench/run.sh` (C vs Rust on the same `.seq` file; minimum of two
runs, `Time.Alignment` for Rust and `Time.Benchmark` for C). Dataset:
`bench/generate_data.sh`, which produces 50,000 pairs of 1000 bp at 5% error.
Ratio = Rust / C; below 1.0 means Rust is faster. Also record:
- the machine;
- the C commit;
- max RSS (`/usr/bin/time -l` on macOS, `-v` on Linux) when memory may change.

Run on an otherwise idle machine. Concurrent jobs, or the machine sleeping,
invalidate the numbers.

> `generate_dataset` output may vary between machines and C versions, so ratios
> are only comparable within one entry. Compare commits by re-running both on
> the same data.

---

## Historical (initial port, 2026-03)

Copied from the README and `AVX2_MIGRATION.md`; raw logs weren't kept.

### 2026-03-07 — `2e7ab34` / `3ac6766` (README numbers)

aarch64 (Apple M-series, NEON):

| Benchmark | Rust / C |
|---|---|
| Edit (score) | 0.56x |
| Gap-linear (score) | 0.68x |
| Gap-affine (score) | 0.85x |
| Gap-affine (CIGAR) | 0.90x |
| Gap-affine 2-piece (score) | 0.88x |
| Gap-affine 2-piece (CIGAR) | 0.86x |
| BiWFA (ultralow) | 0.86x |

x86_64 (Intel Broadwell, AVX2):

| Benchmark | Rust | C | Ratio |
|---|---|---|---|
| Edit (score) | 405 ms | 446 ms | 0.91x |
| Gap-linear (score) | 682 ms | 1090 ms | 0.63x |
| Gap-affine (score) | 3980 ms | 3970 ms | 1.00x |
| Gap-affine (CIGAR) | 5440 ms | 5380 ms | 1.01x |
| Gap-affine 2-piece (score) | 16800 ms | 19710 ms | 0.85x |
| Gap-affine 2-piece (CIGAR) | 20620 ms | 22070 ms | 0.93x |
| BiWFA (ultralow) | 7133 ms | 6660 ms | 1.07x |

### 2026-03-06 — `bc838bb` (`AVX2_MIGRATION.md`, NEON)

Edit 0.59x, Linear 0.69x, Affine score 0.94x, Affine CIGAR 0.95x, BiWFA 0.92x,
Affine2p score 1.00x, Affine2p CIGAR 1.03x.

---

## 2026-10 — C-parity fixes (Phase 14)

Machine: Apple M2 Pro (aarch64, NEON), macOS 26.6.2. C: WFA2-lib `3282f83`,
built with `make` minus OpenMP (Apple clang has no `-fopenmp`; strip it from
`tools/align_benchmark/Makefile`).

Data: `bench/generate_data.sh` settings (50,000 × 1000 bp, 5% error),
regenerated with C `3282f83`'s `generate_dataset`. Timing as in `bench/run.sh`
(minimum of 2 runs). Max RSS from `/usr/bin/time -l` (the larger of the 2 runs).
The machine was otherwise idle apart from an endpoint-backup agent (~0.5 core).
Within each mode, C and all three commits ran back to back.

### 2026-10-12 — `6d284c8` (main, HPCBio/wfa2lib-rs#1 and HPCBio/wfa2lib-rs#2), `06c593c` (HPCBio/wfa2lib-rs#3), `29a9c89` (HPCBio/wfa2lib-rs#4)

| Mode | C (ms) | `6d284c8` | `06c593c` | `29a9c89` | 29a9c89 vs 06c593c |
|---|---|---|---|---|---|
| Edit score | 403 | 223 (0.55x) | 223 (0.55x) | 228 (0.57x) | +2.0% (noise) |
| Linear score | 555 | 369 (0.66x) | 328 (0.59x) | 323 (0.58x) | -1.3% |
| Affine score | 2370 | 1881 (0.79x) | 1887 (0.80x) | 1940 (0.82x) | +2.8% |
| Affine CIGAR | 2610 | 1938 (0.74x) | 1973 (0.76x) | 2074 (0.79x) | +5.2% |
| BiWFA (ultralow) | 4110 | 3106 (0.76x) | 3079 (0.75x) | 3181 (0.77x) | +3.3% |
| Affine2p score | 8430 | 6975 (0.83x) | 6919 (0.82x) | 7580 (0.90x) | +9.6% |
| Affine2p CIGAR | 8910 | 7433 (0.83x) | 7462 (0.84x) | 7899 (0.89x) | +5.9% |

- **`06c593c` (HPCBio/wfa2lib-rs#3):** no regression.
- **`29a9c89` (HPCBio/wfa2lib-rs#4):** costs 3–10% on affine/affine2p. Rust is still faster than
  C in every mode.

**Where `29a9c89`'s cost comes from** (minimum of 3 runs, the two most affected modes):

| Commit | Affine2p score | Affine CIGAR |
|---|---|---|
| `06c593c` (before `29a9c89`) | 7016 | 1969 |
| `320611d` cutoff equates I/D only when M narrowed | 6993 (±0%) | 1973 (±0%) |
| `ddae4d8` I/D `trim_ends` restored | 7388 (+5.3%) | 2022 (+2.7%) |
| `29a9c89` null-input handling | 7419 (+0.4%) | 2041 (+1.0%) |

Nearly all of it comes from restoring C's independent I/D trimming, which
2c112b2 removed for speed (C pays the same cost). That removal changed results,
and output-identical comes first in the project goals, so the trimming stays.
**Optimization target:** make I/D trimming cheaper without changing which
diagonals survive.

**Memory:** Rust max RSS is about 200 MB in every mode, at all three commits,
vs 1.6–9.5 MB for C. That points to the `align_benchmark` CLI rather than the
library: it reads the whole input file (about 100 MB here) into memory and
buffers all output lines, while C streams. It fails PLAN's "memory within 1.5x"
target as measured. Tracked as a CLI gap (HPCBio/wfa2lib-rs#8). Library-level
memory should be measured separately, without CLI buffering.

### 2026-10-12 — `6a06d8f` (HPCBio/wfa2lib-rs#9 ends-free begin gaps) vs `eee854b` (main)

End-to-end, same machine and data, Rust only (minimum of 3 runs, ms):

| Mode | `eee854b` | `6a06d8f` |
|---|---|---|
| Edit score | 220 | 201 |
| Linear score | 321 | 324 |
| Affine score | 1939 | 1937 |
| Affine CIGAR | 2074 | 2070 |
| Affine2p score | 7420 | 7416 |

No regression from the per-step begin-free check. Edit is about 8% faster,
plausibly because the edit step now writes C's four boundary NULLs directly
instead of calling `init_ends`.

