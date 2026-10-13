# Fixed Bugs

Non-trivial bugs found and fixed in the Rust port, as asked for in
`PROJECT.md`. Each entry records the symptom, the root cause, the C reference
behaviour, which commit introduced the bug, and the fix.

C reference: WFA2-lib `16969b5b` (2026-02-28), the commit the port was made
from (see "Port baseline" in `PROGRESS.md`), plus current C `main` where noted.
Issue and PR numbers refer to the HPCBio/wfa2lib-rs fork unless marked
COMBINE-lab.

| # | Bug | Introduced | Status |
|---|---|---|---|
| 1 | Out-of-bounds write past score-indexed component arrays | d235d45 | Fixed: `d3618de` (HPCBio/wfa2lib-rs#11, merged); upstream COMBINE-lab/wfa2lib-rs#6 (open) |
| 2 | `init_ends_{lower,higher}` negative length wraps to a huge slice | 4d99296 | Fixed: `e494181` (HPCBio/wfa2lib-rs#11, merged); upstream COMBINE-lab/wfa2lib-rs#6 (open) |
| 3 | Edit/indel/linear kernels overrun the wavefront allocation | original port | Fixed: `06c593c` (HPCBio/wfa2lib-rs#12, merged `12ba05b`) |
| 4 | Unreachable banded alignments never terminate; affine banded output differs from C | 2c112b2, 35bfcde + original port | Fixed: `320611d`, `ddae4d8`, `29a9c89` (HPCBio/wfa2lib-rs#14, merged `59db5a4`) |
| 5 | CLI prints `0` and an empty CIGAR field when no alignment is found (C: `-1  -`) | original port (CLI) | Fixed: `5e6ce55` (HPCBio/wfa2lib-rs#15, merged) |
| 6 | Ends-free begin gaps ignored (output equals global) | original port | Fix on branch `feat/endsfree-begin-gaps` |

All four were found by differential testing against C (HPCBio/wfa2lib-rs#10): the
Rust and C `align_benchmark` binaries run over the same configuration matrix,
which includes banded heuristics that the stock `wfa.utest.sh` doesn't cover.

---

## 1. Out-of-bounds write past the score-indexed component arrays

- **Symptom:** segfault (release) or UB-check abort (debug) with
  `banded-static` whenever the band excludes the end diagonal, in full-memory
  (CIGAR) mode. Reproduce: `>` 120 bp / `<` its first 90 bp, `-a edit-wfa
  --wfa-heuristic=banded-static --wfa-heuristic-parameters=-5,5`.
- **Root cause:** an unreachable alignment keeps raising the score past
  `WavefrontComponents::num_wavefronts`. `set_*_ptr` writes through
  `get_unchecked_mut`. Running past `num_wavefronts` also exceeds the slab
  reservation (`num_wavefronts * num_types`) that keeps the stored
  `*mut Wavefront` pointers valid.
- **Introduced:** d235d45 removed the original grow-on-demand bounds checks, on
  the assumption (stated in a `// SAFETY:` comment) that `compute_dimensions()`
  pre-sizes the arrays to the maximum reachable score. That only holds when the
  alignment terminates.
- **C:** `wavefront_compute_allocate_output` checks
  `score_mod >= num_wavefronts` ("Maximum allocated wavefronts reached") and
  calls `exit(1)`.
- **Fix (`d3618de`):** `WavefrontComponents::has_slot()` is checked once per
  score step in the end-to-end, ends-free and BiWFA-base loops, stopping with
  `AlignStatus::MaxStepsReached`. The hot accessors stay unchecked.
- **Test:** `test_banded_unreachable_does_not_overflow_components`.

## 2. `Wavefront::init_ends_{lower,higher}` wrapped slice length

- **Symptom:** UB-check abort at `wavefront.rs` (`slice::from_raw_parts_mut`)
  with gap-linear + banded-static; silent memory corruption in release.
- **Root cause:** when a cutoff leaves `lo < min_lo < wf_elements_init_min`,
  `(min_init - start) as usize` is negative and wraps.
  `init_ends_higher` has the mirror-image case.
- **Introduced:** the original `for k in start..min_init` loop matched C.
  cbe976e changed it to a checked slice range (which would panic instead), and
  4d99296 to `from_raw_parts_mut` with an unchecked cast.
- **C:** `wavefront_compute_init_ends_wf_{lower,higher}` use `for` loops, which
  do nothing when the range is empty.
- **Fix (`e494181`):** fill only when the range is non-empty.
- **Tests:** `test_init_ends_{lower,higher}_empty_range`,
  `test_banded_linear_init_ends_empty_range`.

## 3. Edit/indel/linear kernels overrun the wavefront allocation

- **Symptom:** heap corruption with banded-static ±10.
  - gap-linear: on `wfa.utest.seq` pair 251 alone (`lo = -1265`).
  - edit: depends on order, appearing on pair 261 only after pair 251.
  - score-only mode never terminated.
- **Root cause:** once trimming empties M (`null = true`, `lo > hi`), the edit
  and linear steps kept computing from it, because they only checked for a
  missing *pointer*. Each step widened the range by one diagonal. The band
  cutoff skips `lo > hi` wavefronts, so `lo/hi` drifted past the historic range
  the wavefronts are allocated with.
- **C:** linear/affine/affine2p check `->null` on every input. If all are null
  they store a NULL output and count a null step, and extend reports
  `END_UNREACHABLE` after `max_score_scope` such steps. (C's edit step lacks
  the check and never terminates: a C bug, still to be reported upstream.)
- **Fix (`06c593c`, HPCBio/wfa2lib-rs#12, merged):**
  - null output when every input is null;
  - edit/linear `lo/hi` clamped to the historic range minus the kernel's k±1
    margin, as affine already does;
  - null outputs release the modular slot.
- **Test:** `test_banded_unreachable_edit_linear_end_unreachable`.

## 4. Banded alignments: no termination and divergent results (affine/affine2p)

- **Symptoms:**
  - affine/affine2p score-only mode hung on unreachable banded alignments;
  - CIGAR mode gave up on pairs that C completes (utest pairs 233 and 293);
  - give-up scores differed from C.
- **Root causes** (three; each one alone left differences from C):
  1. `apply_heuristic_cutoff` equated I/D to M on every call. C's
     `wavefront_heuristic_cufoff` skips an empty or unchanged M, and marks M
     null if the cutoff emptied it. Equating unconditionally empties I/D
     whenever M is empty.
  2. 2c112b2 dropped `trim_ends` on I/D wavefronts ("matching C's
     process_ends"), and 35bfcde replaced it with clamping I/D to M's range.
     C's `wavefront_compute_process_ends` trims every output independently, so
     I/D keep valid diagonals outside M and can be recognised as null.
  3. Inputs that were present but trimmed empty were read as live, and the
     range union skipped missing inputs. C's `wavefront_compute_get_*wavefront`
     substitutes `wavefront_null` for missing *or* `->null` inputs, and
     `wavefront_compute_limits_input` includes its bounds (lo = 1, hi = -1) in
     the union.
- **How it was found:** per-score traces of M/I/D `lo/hi/null` and the null-step
  count, from Rust and from an instrumented copy of C, on utest pair 202. The
  traces first diverge at s = 980 (M = [-11,-5] in Rust vs [-11,0] in C).
- **Fix (HPCBio/wfa2lib-rs#14, merged):** `320611d` (cutoff equate),
  `ddae4d8` (I/D trim), `29a9c89` (null inputs).
- **Result:** banded-static output for affine, affine2p and gap-linear matches
  C `16969b5b` exactly on `wfa.utest.seq` and a 300 × 2 kb / 8% set (bands
  ±10 and ±50, alignment and score-only modes), including the give-up score for
  unreachable pairs.
- **Tests:** `test_banded_unreachable_affine_end_unreachable`,
  `test_banded_affine_equates_id_only_when_m_narrowed`.

## 5. CLI output when no alignment is found

- **Symptom:** in alignment (CIGAR) mode, `align_benchmark` printed
  `0<TAB>` (score 0, empty CIGAR field) for pairs with no alignment, e.g.
  END_UNREACHABLE under a band. C prints `-1<TAB>-`. Because of this, banded
  output never diffed clean against C, even where the alignments matched.
- **Root cause:** the CLI always re-scored the CIGAR (an empty CIGAR scores 0)
  and printed its RLE string (empty).
- **C:** `benchmark_print_output` uses score `-1` when the CIGAR is empty
  (`begin_offset >= end_offset`) and prints `-` for an empty CIGAR.
- **Fix (`5e6ce55`, HPCBio/wfa2lib-rs#15, merged):** the same rule in
  `compute_output_score` and the CIGAR column. Banded-static output is now
  byte-identical to C `16969b5b` in high, med and score-only modes (30/30
  cells).
- **Tests:** `tests/cli_output.rs` runs the binary: an empty CIGAR prints
  `-1  -`, and the completed and score-only formats are unchanged.

## 6. Ends-free begin gaps ignored

- **Symptom:** with `set_alignment_free_ends(P0, Pf, T0, Tf)` and P0 or T0 > 0,
  the result was identical to the global alignment. Visible once the CLI
  honoured `--wfa-span` (HPCBio/wfa2lib-rs#16): every begin-gap cell differed from C,
  for every metric.
- **Root cause:** `pattern_begin_free` / `text_begin_free` were stored but
  never read. C has two mechanisms, and neither was ported:
  - `match == 0`: `wavefront_aligner_init_wf_m` seeds M0 over `[-P0, T0]`,
    putting cell (h,0) on k = h and (0,v) on k = -v.
  - `match != 0`: `wavefront_compute_endsfree_{required,limits,init,allocate_null}`
    add cell (d,0) / (0,d) at score d * -match, before the ends are trimmed.
- **Fix (branch `feat/endsfree-begin-gaps`):** `28cb857` (match == 0) and
  `6a06d8f` (match != 0). Also C's extension preset (P0 = T0 = 0) and smarco/WFA2-lib#126's
  restriction of the seeding to the ends-free span.
- **Parity details reproduced:**
  - C sets only `lo/hi` on M0; the initialized range stays `[0, 0]`.
  - C's edit step nulls the boundary cells directly instead of calling
    `init_ends`.
  - Together these mean the edit/indel backtrace can't step onto M0's
    begin-free diagonals. It falls back to `min(v,h)` unchecked `M`s, so
    **C can emit an invalid CIGAR for edit/indel with begin gaps** (its own
    `--check=correct` flags it; e.g. `ACGTTCGTACGA` vs `CCACGTACGTACGA`,
    T0 = 3 → `2I12M`).
  - Scores are correct. Gap metrics are unaffected, because their
    `init_ends` widens the initialized range.
  - It's a C regression from `380eb31b` (backtrace bounds checks moved
    from `lo/hi` to the initialized range). C at `380eb31b^` gives valid
    CIGARs.
  - Rust reproduces it for output identity, documented on
    `set_alignment_free_ends`. **Known issue HPCBio/wfa2lib-rs#18.**
  - Reported upstream as smarco/WFA2-lib#142, with a reproducer and a
    verified one-line fix: set `wf_elements_init_min/max = lo/hi` on M0 in
    `wavefront_aligner_init_wf_m`.
- **Result:** ends-free (begin and/or end gaps) and extension are
  byte-identical to C for all metrics, match == 0 and match < 0, in
  alignment, score-only and med modes.
- **Tests:** `test_endsfree_begin_free_match_zero`,
  `test_extension_ignores_begin_free`, and
  `tests/cli_output.rs::ends_free_begin_gaps_match_c`.
