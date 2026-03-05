//! WavefrontAligner: main state machine and public API for wavefront alignment.

use crate::backtrace;
use crate::bialign::{self, BIALIGN_FALLBACK_MIN_SCORE, BiAlignBreakpoint, ComponentType};
use crate::cigar::Cigar;
use crate::components::WavefrontComponents;
use crate::compute;
use crate::compute::{affine, affine2p, edit, linear};
use crate::extend;
use crate::heuristic::{self, HeuristicState, HeuristicStrategy};
use crate::offset::{wavefront_h, wavefront_v};
use crate::penalties::{DistanceMetric, WavefrontPenalties};
use crate::sequences::{SequenceMode, WavefrontSequences};
use crate::slab::{SlabMode, WAVEFRONT_IDX_NONE, WavefrontSlab};
use crate::termination;
use crate::wavefront::WavefrontPos;

/// Alignment computation scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentScope {
    /// Only compute the alignment score.
    ComputeScore,
    /// Compute score and full alignment (CIGAR).
    ComputeAlignment,
}

/// Alignment span mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentSpan {
    /// End-to-end (global) alignment.
    End2End,
    /// Ends-free (semi-global) alignment.
    EndsFree,
}

/// Alignment result status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignStatus {
    /// Initial state / in progress.
    Ok,
    /// Alignment target reached (internal, before finalization).
    EndReached,
    /// Target unreachable (heuristic cutoff).
    EndUnreachable,
    /// Maximum alignment steps exceeded.
    MaxStepsReached,
    /// Out of memory.
    Oom,
    /// Alignment completed successfully.
    Completed,
    /// Partial alignment (extension or heuristic).
    Partial,
}

/// Main wavefront aligner.
pub struct WavefrontAligner {
    // Sequences
    pub sequences: WavefrontSequences,

    // Wavefront data
    pub wf_components: WavefrontComponents,
    pub wavefront_slab: WavefrontSlab,

    // Configuration
    pub penalties: WavefrontPenalties,
    pub alignment_scope: AlignmentScope,
    pub alignment_span: AlignmentSpan,

    // Alignment state
    pub cigar: Cigar,
    pub alignment_end_pos: WavefrontPos,
    pub status: AlignStatus,
    pub num_null_steps: i32,

    // Ends-free configuration
    pub pattern_begin_free: i32,
    pub pattern_end_free: i32,
    pub text_begin_free: i32,
    pub text_end_free: i32,
    pub extension: bool,

    // Heuristic
    pub heuristic: HeuristicStrategy,
    pub heuristic_state: HeuristicState,

    // System limits
    pub max_alignment_steps: i32,
}

impl WavefrontAligner {
    /// Create a new aligner with the given penalties.
    pub fn new(penalties: WavefrontPenalties) -> Self {
        let wf_components = WavefrontComponents::new(10, 10, &penalties, false, false);
        let init_wf_length = 10;

        Self {
            sequences: WavefrontSequences::new(),
            wf_components,
            wavefront_slab: WavefrontSlab::new(init_wf_length, false, SlabMode::Reuse),
            penalties,
            alignment_scope: AlignmentScope::ComputeScore,
            alignment_span: AlignmentSpan::End2End,
            cigar: Cigar::new(0),
            alignment_end_pos: WavefrontPos::default(),
            status: AlignStatus::Ok,
            num_null_steps: 0,
            pattern_begin_free: 0,
            pattern_end_free: 0,
            text_begin_free: 0,
            text_end_free: 0,
            extension: false,
            heuristic: HeuristicStrategy::None,
            heuristic_state: HeuristicState::default(),
            max_alignment_steps: i32::MAX,
        }
    }

    /// Set the maximum alignment steps (score limit).
    pub fn set_max_alignment_steps(&mut self, max_steps: i32) {
        self.max_alignment_steps = max_steps;
    }

    /// Configure ends-free alignment with the given free-gap parameters.
    pub fn set_alignment_free_ends(
        &mut self,
        pattern_begin_free: i32,
        pattern_end_free: i32,
        text_begin_free: i32,
        text_end_free: i32,
    ) {
        self.alignment_span = AlignmentSpan::EndsFree;
        self.extension = false;
        self.pattern_begin_free = pattern_begin_free;
        self.pattern_end_free = pattern_end_free;
        self.text_begin_free = text_begin_free;
        self.text_end_free = text_end_free;
    }

    /// Configure extension alignment mode.
    ///
    /// Extension alignment uses ends-free with full end-free parameters
    /// and applies maxtrim post-processing to find the maximal-scoring prefix.
    pub fn set_alignment_extension(&mut self) {
        self.alignment_span = AlignmentSpan::EndsFree;
        self.extension = true;
    }

    /// Set the heuristic strategy for wavefront pruning.
    pub fn set_heuristic(&mut self, strategy: HeuristicStrategy) {
        self.heuristic = strategy;
    }

    /// Align two sequences end-to-end and return the alignment score.
    ///
    /// For edit distance, the score equals the edit distance (Levenshtein distance).
    /// For indel distance, the score equals the indel distance (LCS-based).
    /// For gap-linear/affine, the score equals the WFA score (sum of penalties).
    pub fn align_end2end(&mut self, pattern: &[u8], text: &[u8]) -> i32 {
        // Init sequences
        self.sequences.init_ascii(pattern, text, false);
        self.align_end2end_inner()
    }

    /// Shared end-to-end alignment loop (sequences must already be initialized).
    fn align_end2end_inner(&mut self) -> i32 {
        // Init alignment
        self.init_alignment();

        // Main alignment loop
        let mut score = 0;
        loop {
            // Extend
            if self.extend_end2end(score) {
                break;
            }

            // Next score
            score += 1;

            // Compute next wavefront based on distance metric
            self.compute_step(score);

            // Check limits
            if score >= self.max_alignment_steps {
                self.status = AlignStatus::MaxStepsReached;
                break;
            }
        }

        // Finalize
        if self.status == AlignStatus::EndReached {
            // Backtrace to produce CIGAR if requested
            if self.alignment_scope == AlignmentScope::ComputeAlignment {
                let pos = self.alignment_end_pos;
                match self.penalties.distance_metric {
                    DistanceMetric::GapAffine => {
                        backtrace::backtrace_affine(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            ComponentType::M,
                            ComponentType::M,
                            &mut self.cigar,
                        );
                    }
                    DistanceMetric::GapAffine2p => {
                        backtrace::backtrace_affine2p(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            ComponentType::M,
                            ComponentType::M,
                            &mut self.cigar,
                        );
                    }
                    _ => {
                        backtrace::backtrace_linear(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            &mut self.cigar,
                        );
                    }
                }
            }
            self.status = AlignStatus::Completed;
        }

        score
    }

    /// Align two sequences with ends-free (semi-global) alignment.
    ///
    /// Ends-free alignment allows free gaps at the ends of sequences, controlled by
    /// the `pattern_end_free` and `text_end_free` parameters (set via
    /// `set_alignment_free_ends` or `set_alignment_extension`).
    ///
    /// For extension mode, the CIGAR is trimmed to the maximal-scoring prefix.
    pub fn align_endsfree(&mut self, pattern: &[u8], text: &[u8]) -> i32 {
        // Init sequences
        self.sequences.init_ascii(pattern, text, false);

        // For extension mode, override end_free to full sequence lengths
        if self.extension {
            self.pattern_end_free = self.sequences.pattern_length;
            self.text_end_free = self.sequences.text_length;
        }

        // Init alignment
        self.init_alignment();

        // Main alignment loop
        let mut score = 0;
        loop {
            // Extend with ends-free termination
            if self.extend_endsfree(score) {
                break;
            }

            // Next score
            score += 1;

            // Compute next wavefront based on distance metric
            match self.penalties.distance_metric {
                DistanceMetric::Edit | DistanceMetric::Indel => {
                    self.compute_edit_step(score);
                }
                DistanceMetric::GapLinear => {
                    self.compute_linear_step(score);
                }
                DistanceMetric::GapAffine => {
                    self.compute_affine_step(score);
                }
                DistanceMetric::GapAffine2p => {
                    self.compute_affine2p_step(score);
                }
            }

            // Check limits
            if score >= self.max_alignment_steps {
                self.status = AlignStatus::MaxStepsReached;
                break;
            }
        }

        // Finalize
        if self.status == AlignStatus::EndReached || self.status == AlignStatus::EndUnreachable {
            let unreachable = self.status == AlignStatus::EndUnreachable;

            // Backtrace to produce CIGAR if requested
            if self.alignment_scope == AlignmentScope::ComputeAlignment
                && self.status == AlignStatus::EndReached
            {
                let pos = self.alignment_end_pos;
                match self.penalties.distance_metric {
                    DistanceMetric::GapAffine => {
                        backtrace::backtrace_affine(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            ComponentType::M,
                            ComponentType::M,
                            &mut self.cigar,
                        );
                    }
                    DistanceMetric::GapAffine2p => {
                        backtrace::backtrace_affine2p(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            ComponentType::M,
                            ComponentType::M,
                            &mut self.cigar,
                        );
                    }
                    _ => {
                        backtrace::backtrace_linear(
                            &self.wf_components,
                            &self.wavefront_slab,
                            &self.penalties,
                            self.sequences.pattern_length,
                            self.sequences.text_length,
                            pos.score,
                            pos.k,
                            pos.offset,
                            &mut self.cigar,
                        );
                    }
                }

                // Post-processing: extension mode or unreachable → maxtrim
                if self.extension || unreachable {
                    let trimmed = self.maxtrim_cigar();
                    self.status = if trimmed || unreachable {
                        AlignStatus::Partial
                    } else {
                        AlignStatus::Completed
                    };
                } else {
                    self.status = AlignStatus::Completed;
                }
            } else {
                self.status = if unreachable {
                    AlignStatus::Partial
                } else {
                    AlignStatus::Completed
                };
            }
        }

        score
    }

    /// Get the alignment status after the last alignment.
    pub fn status(&self) -> AlignStatus {
        self.status
    }

    /// Get the CIGAR from the last alignment (only valid after ComputeAlignment).
    pub fn cigar(&self) -> &Cigar {
        &self.cigar
    }

    // --- Internal methods ---

    /// Initialize for a new alignment.
    fn init_alignment(&mut self) {
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        // Resize CIGAR buffer if computing alignment
        if self.alignment_scope == AlignmentScope::ComputeAlignment {
            let max_ops = (plen + tlen + 1) as usize;
            self.cigar.resize(max_ops);
        }

        // Resize components for sequence dimensions
        self.wf_components.resize(plen, tlen, &self.penalties);

        // Clear slab
        self.wavefront_slab.clear();

        // Reset state
        self.num_null_steps = 0;
        self.status = AlignStatus::Ok;
        self.alignment_end_pos = WavefrontPos::default();
        self.heuristic_state.reset(&self.heuristic);

        // Compute allocation limits based on sequence lengths.
        // k ranges from -plen to +tlen, so we need wavefronts that cover this
        // full range. Add padding for the kernel's k±1 reads and score scope.
        let max_score_scope = self.wf_components.max_score_scope as i32;
        let eff_lo = -(plen + max_score_scope + 2);
        let eff_hi = tlen + max_score_scope + 2;

        // Update historic bounds — these define the min base_k for all wavefronts
        self.wf_components.historic_min_lo = eff_lo;
        self.wf_components.historic_max_hi = eff_hi;

        // Ensure the slab allocates wavefronts large enough for the full range
        self.wavefront_slab.ensure_min_length(eff_lo, eff_hi);

        // Resize null/victim wavefronts
        self.wf_components.resize_null_victim(eff_lo, eff_hi);

        // Allocate initial M-wavefront at score 0
        let idx = self.wavefront_slab.allocate(eff_lo, eff_hi);
        {
            let wf = self.wavefront_slab.get_mut(idx);
            wf.set_offset(0, 0);
            wf.set_limits(0, 0);
        }
        self.wf_components.set_m_idx(0, idx);
    }

    /// Extend the M-wavefront at the given score, check termination.
    /// Returns true if alignment is finished.
    fn extend_end2end(&mut self, score: i32) -> bool {
        let score_idx = score as usize;
        let m_idx = self.wf_components.get_m_idx(score_idx);

        if m_idx == WAVEFRONT_IDX_NONE {
            self.num_null_steps += 1;
            if self.num_null_steps > self.wf_components.max_score_scope as i32 {
                self.status = AlignStatus::EndUnreachable;
                return true;
            }
            return false;
        }
        self.num_null_steps = 0;

        // Extend matches
        let wf = self.wavefront_slab.get_mut(m_idx);
        if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_end2end(&self.sequences, wf);
        } else {
            extend::extend_matches_packed_end2end(&self.sequences, wf);
        }

        // Check termination
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;
        let wf = self.wavefront_slab.get(m_idx);
        if termination::termination_end2end(plen, tlen, wf) {
            let alignment_k = tlen - plen;
            self.alignment_end_pos = WavefrontPos {
                score,
                k: alignment_k,
                offset: tlen,
            };
            self.status = AlignStatus::EndReached;
            return true;
        }

        // Apply heuristic cutoff
        if !matches!(self.heuristic, HeuristicStrategy::None)
            && self.apply_heuristic_cutoff(score, m_idx)
        {
            self.status = AlignStatus::EndUnreachable;
            return true;
        }

        false
    }

    /// Extend the M-wavefront at the given score with ends-free termination.
    /// Returns true if alignment is finished.
    fn extend_endsfree(&mut self, score: i32) -> bool {
        let score_idx = score as usize;
        let m_idx = self.wf_components.get_m_idx(score_idx);

        if m_idx == WAVEFRONT_IDX_NONE {
            self.num_null_steps += 1;
            if self.num_null_steps > self.wf_components.max_score_scope as i32 {
                self.status = AlignStatus::EndUnreachable;
                return true;
            }
            return false;
        }
        self.num_null_steps = 0;

        // Extend matches with ends-free termination check
        let wf = self.wavefront_slab.get_mut(m_idx);
        let result = if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_endsfree(
                &self.sequences,
                wf,
                self.pattern_end_free,
                self.text_end_free,
            )
        } else {
            extend::extend_matches_packed_endsfree(
                &self.sequences,
                wf,
                self.pattern_end_free,
                self.text_end_free,
            )
        };

        if let Some((k, offset)) = result {
            self.alignment_end_pos = WavefrontPos { score, k, offset };
            self.status = AlignStatus::EndReached;
            return true;
        }

        // Apply heuristic cutoff
        if !matches!(self.heuristic, HeuristicStrategy::None)
            && self.apply_heuristic_cutoff(score, m_idx)
        {
            self.status = AlignStatus::EndUnreachable;
            return true;
        }

        false
    }

    /// Apply maxtrim to the CIGAR based on the distance metric.
    /// Returns true if the CIGAR was trimmed.
    fn maxtrim_cigar(&mut self) -> bool {
        match self.penalties.distance_metric {
            DistanceMetric::GapLinear => {
                let p = self.penalties.linear_penalties.as_ref().unwrap();
                self.cigar.maxtrim_gap_linear(p)
            }
            DistanceMetric::GapAffine => {
                let p = self.penalties.affine_penalties.as_ref().unwrap();
                self.cigar.maxtrim_gap_affine(p)
            }
            DistanceMetric::GapAffine2p => {
                let p = self.penalties.affine2p_penalties.as_ref().unwrap();
                self.cigar.maxtrim_gap_affine2p(p)
            }
            _ => false, // Edit/indel don't support maxtrim
        }
    }

    /// Apply heuristic cutoff to the M-wavefront (and I/D wavefronts if affine).
    /// Returns true if alignment should be terminated (z-drop).
    fn apply_heuristic_cutoff(&mut self, score: i32, m_idx: usize) -> bool {
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        // Use the victim wavefront's offsets as scratch buffer for distances
        let distances_base_k = self.wf_components.wavefront_victim.base_k();
        let distances = self.wf_components.wavefront_victim.offsets_slice_mut();

        let wf = self.wavefront_slab.get_mut(m_idx);
        let zdropped = heuristic::heuristic_cutoff(
            &self.heuristic,
            &mut self.heuristic_state,
            wf,
            distances,
            distances_base_k,
            plen,
            tlen,
            score,
        );

        if zdropped {
            return true;
        }

        // Propagate narrowed M-wavefront bounds to I/D wavefronts
        let score_idx = score as usize;
        match self.penalties.distance_metric {
            DistanceMetric::GapAffine => {
                let m_wf = self.wavefront_slab.get(m_idx);
                let m_lo = m_wf.lo;
                let m_hi = m_wf.hi;

                let i1_idx = self.wf_components.get_i1_idx(score_idx);
                if i1_idx != WAVEFRONT_IDX_NONE {
                    let i1_wf = self.wavefront_slab.get_mut(i1_idx);
                    if m_lo > i1_wf.lo {
                        i1_wf.lo = m_lo;
                    }
                    if m_hi < i1_wf.hi {
                        i1_wf.hi = m_hi;
                    }
                    if i1_wf.lo > i1_wf.hi {
                        i1_wf.null = true;
                    }
                }

                let d1_idx = self.wf_components.get_d1_idx(score_idx);
                if d1_idx != WAVEFRONT_IDX_NONE {
                    let d1_wf = self.wavefront_slab.get_mut(d1_idx);
                    if m_lo > d1_wf.lo {
                        d1_wf.lo = m_lo;
                    }
                    if m_hi < d1_wf.hi {
                        d1_wf.hi = m_hi;
                    }
                    if d1_wf.lo > d1_wf.hi {
                        d1_wf.null = true;
                    }
                }
            }
            DistanceMetric::GapAffine2p => {
                let m_wf = self.wavefront_slab.get(m_idx);
                let m_lo = m_wf.lo;
                let m_hi = m_wf.hi;

                for get_idx_fn in [
                    WavefrontComponents::get_i1_idx,
                    WavefrontComponents::get_d1_idx,
                    WavefrontComponents::get_i2_idx,
                    WavefrontComponents::get_d2_idx,
                ] {
                    let idx = get_idx_fn(&self.wf_components, score_idx);
                    if idx != WAVEFRONT_IDX_NONE {
                        let wf = self.wavefront_slab.get_mut(idx);
                        if m_lo > wf.lo {
                            wf.lo = m_lo;
                        }
                        if m_hi < wf.hi {
                            wf.hi = m_hi;
                        }
                        if wf.lo > wf.hi {
                            wf.null = true;
                        }
                    }
                }
            }
            _ => {} // Edit/indel/linear only have M-wavefront
        }

        false
    }

    /// Compute the next wavefront for edit/indel distance.
    fn compute_edit_step(&mut self, score: i32) {
        let score_prev = (score - 1) as usize;
        let score_curr = score as usize;

        let prev_idx = self.wf_components.get_m_idx(score_prev);
        // Get prev wavefront dimensions
        let (prev_lo, prev_hi) = {
            let wf = self.wavefront_slab.get(prev_idx);
            (wf.lo, wf.hi)
        };

        let lo = prev_lo - 1;
        let hi = prev_hi + 1;

        // Initialize ends on previous wavefront (like C's wavefront_compute_init_ends)
        // Edit kernel reads prev[k-1], prev[k], prev[k+1] for k in [lo, hi]
        // so prev needs [lo-1, hi+1]
        {
            let wf_prev = self.wavefront_slab.get_mut(prev_idx);
            wf_prev.init_ends_higher(hi + 1);
            wf_prev.init_ends_lower(lo - 1);
        }

        // Allocate output wavefront using historic bounds
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        let curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        {
            let wf_curr = self.wavefront_slab.get_mut(curr_idx);
            wf_curr.set_limits(lo, hi);
        }

        // Compute kernel
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;
        {
            let (wf_prev, wf_curr) = self.wavefront_slab.get_two_mut(prev_idx, curr_idx);
            match self.penalties.distance_metric {
                DistanceMetric::Edit => {
                    edit::compute_edit_idm(plen, tlen, wf_prev, wf_curr, lo, hi);
                }
                DistanceMetric::Indel => {
                    edit::compute_indel_idm(plen, tlen, wf_prev, wf_curr, lo, hi);
                }
                _ => unreachable!("compute_edit_step called for non-edit metric"),
            }
        }

        // Store in components
        self.wf_components.set_m_idx(score_curr, curr_idx);

        // Trim ends
        {
            let wf_curr = self.wavefront_slab.get_mut(curr_idx);
            compute::trim_ends(plen, tlen, wf_curr);
            if wf_curr.null {
                self.num_null_steps = i32::MAX;
            }
        }
    }
    /// Compute the next wavefront for gap-linear distance.
    fn compute_linear_step(&mut self, score: i32) {
        let score_curr = score as usize;
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        // Fetch input wavefronts
        let misms_score = score - self.penalties.mismatch;
        let gap_score = score - self.penalties.gap_opening1;

        let misms_idx = if misms_score >= 0 {
            self.wf_components.get_m_idx(misms_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let gap_idx = if gap_score >= 0 {
            self.wf_components.get_m_idx(gap_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };

        // Get effective lo/hi from input wavefronts
        let (misms_lo, misms_hi) = if misms_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get(misms_idx);
            (wf.lo, wf.hi)
        } else {
            (0, -1) // empty range
        };
        let (gap_lo, gap_hi) = if gap_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get(gap_idx);
            (wf.lo, wf.hi)
        } else {
            (0, -1) // empty range
        };

        // Union of ranges (gap expands by ±1 for ins/del k-shifts)
        let lo = misms_lo.min(gap_lo - 1);
        let hi = misms_hi.max(gap_hi + 1);

        if lo > hi {
            // No valid inputs
            return;
        }

        // Ensure null/victim wavefronts cover the range
        self.wf_components.resize_null_victim(lo, hi);

        // Initialize input wavefront ends (like C's wavefront_compute_init_ends)
        // misms needs [lo, hi]
        if misms_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(misms_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo);
        }
        // gap needs [lo-1, hi+1] (kernel reads k-1 and k+1)
        if gap_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(gap_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo - 1);
        }
        let use_misms_idx = misms_idx;

        // Allocate output wavefront using historic bounds
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        let curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        {
            let wf_curr = self.wavefront_slab.get_mut(curr_idx);
            wf_curr.set_limits(lo, hi);
        }

        // Compute kernel using raw pointers for multi-borrow
        // SAFETY: curr_idx is freshly allocated and distinct from all input indices.
        // Input wavefronts are either the null wavefront (not in slab) or at
        // prior score indices which are guaranteed different from curr_idx.
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;

            let wf_misms_ptr = if use_misms_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(use_misms_idx) as *const _
            } else {
                null_wf
            };
            let wf_gap_ptr = if gap_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(gap_idx) as *const _
            } else {
                null_wf
            };
            let curr_ptr = self.wavefront_slab.get_raw_mut(curr_idx);

            linear::compute_linear_idm(
                plen,
                tlen,
                &*wf_misms_ptr,
                &*wf_gap_ptr,
                &mut *curr_ptr,
                lo,
                hi,
            );
        }

        // Store in components
        self.wf_components.set_m_idx(score_curr, curr_idx);

        // Trim ends
        {
            let wf_curr = self.wavefront_slab.get_mut(curr_idx);
            compute::trim_ends(plen, tlen, wf_curr);
            if wf_curr.null {
                self.num_null_steps = i32::MAX;
            }
        }
    }

    /// Compute the next wavefront for gap-affine distance.
    fn compute_affine_step(&mut self, score: i32) {
        let score_curr = score as usize;
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        let x = self.penalties.mismatch;
        let o_plus_e = self.penalties.gap_opening1 + self.penalties.gap_extension1;
        let e = self.penalties.gap_extension1;

        // Fetch input wavefront indices
        let m_misms_score = score - x;
        let m_open_score = score - o_plus_e;
        let i1_ext_score = score - e;
        let d1_ext_score = score - e;

        let m_misms_idx = if m_misms_score >= 0 {
            self.wf_components.get_m_idx(m_misms_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let m_open_idx = if m_open_score >= 0 {
            self.wf_components.get_m_idx(m_open_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let i1_ext_idx = if i1_ext_score >= 0 {
            self.wf_components.get_i1_idx(i1_ext_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let d1_ext_idx = if d1_ext_score >= 0 {
            self.wf_components.get_d1_idx(d1_ext_score as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };

        // Compute effective lo/hi from all input wavefronts
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;

        // Each input contributes output diagonals based on how the kernel reads it:
        // - m_misms: read at k → output [lo, hi]
        // - m_open: read at k-1 and k+1 → output [lo-1, hi+1]
        // - i1_ext: read at k-1 → output [lo+1, hi+1]
        // - d1_ext: read at k+1 → output [lo-1, hi-1]
        for (idx, lo_delta, hi_delta) in [
            (m_misms_idx, 0, 0),
            (m_open_idx, -1, 1),
            (i1_ext_idx, 1, 1),
            (d1_ext_idx, -1, -1),
        ] {
            if idx != WAVEFRONT_IDX_NONE {
                let wf = self.wavefront_slab.get(idx);
                lo = lo.min(wf.lo + lo_delta);
                hi = hi.max(wf.hi + hi_delta);
            }
        }

        if lo > hi {
            // No valid inputs
            return;
        }

        // Clamp lo/hi to leave room for kernel's k±1 reads within allocated range.
        // All slab wavefronts have base_k = hist_lo, so k-1 ≥ hist_lo and k+1 ≤ hist_hi.
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        lo = lo.max(hist_lo + 1);
        hi = hi.min(hist_hi - 1);

        if lo > hi {
            return;
        }

        // Ensure null/victim wavefronts cover the range
        self.wf_components.resize_null_victim(lo, hi);

        // Initialize input wavefront ends (like C's wavefront_compute_init_ends)
        // m_misms: kernel reads at k → needs [lo, hi]
        if m_misms_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(m_misms_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo);
        }
        // m_open: kernel reads at k-1 and k+1 → needs [lo-1, hi+1]
        if m_open_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(m_open_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo - 1);
        }
        // i1_ext: kernel reads at k-1 → needs [lo-1, hi-1]
        if i1_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(i1_ext_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo - 1);
        }
        // d1_ext: kernel reads at k+1 → needs [lo+1, hi+1]
        if d1_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(d1_ext_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo);
        }

        // Allocate 3 output wavefronts using historic bounds
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        let m_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let i1_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let d1_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);

        {
            let wf = self.wavefront_slab.get_mut(m_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(i1_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(d1_curr_idx);
            wf.set_limits(lo, hi);
        }

        // Compute kernel using raw pointers for multi-borrow
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;

            let wf_m_misms = if m_misms_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(m_misms_idx) as *const _
            } else {
                null_wf
            };
            let wf_m_open = if m_open_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(m_open_idx) as *const _
            } else {
                null_wf
            };
            let wf_i1_ext = if i1_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(i1_ext_idx) as *const _
            } else {
                null_wf
            };
            let wf_d1_ext = if d1_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(d1_ext_idx) as *const _
            } else {
                null_wf
            };

            let m_curr_ptr = self.wavefront_slab.get_raw_mut(m_curr_idx);
            let i1_curr_ptr = self.wavefront_slab.get_raw_mut(i1_curr_idx);
            let d1_curr_ptr = self.wavefront_slab.get_raw_mut(d1_curr_idx);

            affine::compute_affine_idm(
                plen,
                tlen,
                &*wf_m_misms,
                &*wf_m_open,
                &*wf_i1_ext,
                &*wf_d1_ext,
                &mut *m_curr_ptr,
                &mut *i1_curr_ptr,
                &mut *d1_curr_ptr,
                lo,
                hi,
            );
        }

        // Store in components
        self.wf_components.set_m_idx(score_curr, m_curr_idx);
        self.wf_components.set_i1_idx(score_curr, i1_curr_idx);
        self.wf_components.set_d1_idx(score_curr, d1_curr_idx);

        // Trim ends (on M-wavefront)
        {
            let wf_curr = self.wavefront_slab.get_mut(m_curr_idx);
            compute::trim_ends(plen, tlen, wf_curr);
            if wf_curr.null {
                self.num_null_steps = i32::MAX;
            }
        }
    }

    /// Compute the next wavefront for gap-affine 2-piece distance.
    fn compute_affine2p_step(&mut self, score: i32) {
        let score_curr = score as usize;
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        let x = self.penalties.mismatch;
        let o1_plus_e1 = self.penalties.gap_opening1 + self.penalties.gap_extension1;
        let e1 = self.penalties.gap_extension1;
        let o2_plus_e2 = self.penalties.gap_opening2 + self.penalties.gap_extension2;
        let e2 = self.penalties.gap_extension2;

        // Fetch input wavefront indices
        let m_misms_idx = if score - x >= 0 {
            self.wf_components.get_m_idx((score - x) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let m_open1_idx = if score - o1_plus_e1 >= 0 {
            self.wf_components.get_m_idx((score - o1_plus_e1) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let m_open2_idx = if score - o2_plus_e2 >= 0 {
            self.wf_components.get_m_idx((score - o2_plus_e2) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let i1_ext_idx = if score - e1 >= 0 {
            self.wf_components.get_i1_idx((score - e1) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let i2_ext_idx = if score - e2 >= 0 {
            self.wf_components.get_i2_idx((score - e2) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let d1_ext_idx = if score - e1 >= 0 {
            self.wf_components.get_d1_idx((score - e1) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };
        let d2_ext_idx = if score - e2 >= 0 {
            self.wf_components.get_d2_idx((score - e2) as usize)
        } else {
            WAVEFRONT_IDX_NONE
        };

        // Compute effective lo/hi from all input wavefronts
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;

        for (idx, lo_delta, hi_delta) in [
            (m_misms_idx, 0, 0),
            (m_open1_idx, -1, 1),
            (m_open2_idx, -1, 1),
            (i1_ext_idx, 1, 1),
            (i2_ext_idx, 1, 1),
            (d1_ext_idx, -1, -1),
            (d2_ext_idx, -1, -1),
        ] {
            if idx != WAVEFRONT_IDX_NONE {
                let wf = self.wavefront_slab.get(idx);
                lo = lo.min(wf.lo + lo_delta);
                hi = hi.max(wf.hi + hi_delta);
            }
        }

        if lo > hi {
            return;
        }

        // Clamp lo/hi to leave room for kernel's k±1 reads
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        lo = lo.max(hist_lo + 1);
        hi = hi.min(hist_hi - 1);

        if lo > hi {
            return;
        }

        // Ensure null/victim wavefronts cover the range
        self.wf_components.resize_null_victim(lo, hi);

        // Initialize input wavefront ends
        // m_misms: kernel reads at k → needs [lo, hi]
        if m_misms_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(m_misms_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo);
        }
        // m_open1: kernel reads at k-1 and k+1 → needs [lo-1, hi+1]
        if m_open1_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(m_open1_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo - 1);
        }
        // m_open2: kernel reads at k-1 and k+1 → needs [lo-1, hi+1]
        if m_open2_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(m_open2_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo - 1);
        }
        // i1_ext: kernel reads at k-1 → needs [lo-1, hi]
        if i1_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(i1_ext_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo - 1);
        }
        // i2_ext: kernel reads at k-1 → needs [lo-1, hi]
        if i2_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(i2_ext_idx);
            wf.init_ends_higher(hi);
            wf.init_ends_lower(lo - 1);
        }
        // d1_ext: kernel reads at k+1 → needs [lo, hi+1]
        if d1_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(d1_ext_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo);
        }
        // d2_ext: kernel reads at k+1 → needs [lo, hi+1]
        if d2_ext_idx != WAVEFRONT_IDX_NONE {
            let wf = self.wavefront_slab.get_mut(d2_ext_idx);
            wf.init_ends_higher(hi + 1);
            wf.init_ends_lower(lo);
        }

        // Allocate 5 output wavefronts using historic bounds
        let m_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let i1_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let i2_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let d1_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);
        let d2_curr_idx = self.wavefront_slab.allocate(hist_lo, hist_hi);

        {
            let wf = self.wavefront_slab.get_mut(m_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(i1_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(i2_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(d1_curr_idx);
            wf.set_limits(lo, hi);
        }
        {
            let wf = self.wavefront_slab.get_mut(d2_curr_idx);
            wf.set_limits(lo, hi);
        }

        // Compute kernel using raw pointers for multi-borrow
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;

            let wf_m_misms = if m_misms_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(m_misms_idx) as *const _
            } else {
                null_wf
            };
            let wf_m_open1 = if m_open1_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(m_open1_idx) as *const _
            } else {
                null_wf
            };
            let wf_m_open2 = if m_open2_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(m_open2_idx) as *const _
            } else {
                null_wf
            };
            let wf_i1_ext = if i1_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(i1_ext_idx) as *const _
            } else {
                null_wf
            };
            let wf_i2_ext = if i2_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(i2_ext_idx) as *const _
            } else {
                null_wf
            };
            let wf_d1_ext = if d1_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(d1_ext_idx) as *const _
            } else {
                null_wf
            };
            let wf_d2_ext = if d2_ext_idx != WAVEFRONT_IDX_NONE {
                self.wavefront_slab.get(d2_ext_idx) as *const _
            } else {
                null_wf
            };

            let m_curr_ptr = self.wavefront_slab.get_raw_mut(m_curr_idx);
            let i1_curr_ptr = self.wavefront_slab.get_raw_mut(i1_curr_idx);
            let i2_curr_ptr = self.wavefront_slab.get_raw_mut(i2_curr_idx);
            let d1_curr_ptr = self.wavefront_slab.get_raw_mut(d1_curr_idx);
            let d2_curr_ptr = self.wavefront_slab.get_raw_mut(d2_curr_idx);

            affine2p::compute_affine2p_idm(
                plen,
                tlen,
                &*wf_m_misms,
                &*wf_m_open1,
                &*wf_m_open2,
                &*wf_i1_ext,
                &*wf_i2_ext,
                &*wf_d1_ext,
                &*wf_d2_ext,
                &mut *m_curr_ptr,
                &mut *i1_curr_ptr,
                &mut *i2_curr_ptr,
                &mut *d1_curr_ptr,
                &mut *d2_curr_ptr,
                lo,
                hi,
            );
        }

        // Store in components
        self.wf_components.set_m_idx(score_curr, m_curr_idx);
        self.wf_components.set_i1_idx(score_curr, i1_curr_idx);
        self.wf_components.set_i2_idx(score_curr, i2_curr_idx);
        self.wf_components.set_d1_idx(score_curr, d1_curr_idx);
        self.wf_components.set_d2_idx(score_curr, d2_curr_idx);

        // Trim ends (on M-wavefront)
        {
            let wf_curr = self.wavefront_slab.get_mut(m_curr_idx);
            compute::trim_ends(plen, tlen, wf_curr);
            if wf_curr.null {
                self.num_null_steps = i32::MAX;
            }
        }
    }

    // =========================================================================
    // BiWFA: Bidirectional Wavefront Alignment
    // =========================================================================

    /// Bidirectional wavefront alignment (BiWFA).
    ///
    /// Uses divide-and-conquer with O(s) memory via bidirectional
    /// wavefront expansion and breakpoint detection.
    pub fn align_biwfa(&mut self, pattern: &[u8], text: &[u8]) -> i32 {
        self.sequences.init_ascii(pattern, text, false);
        self.alignment_scope = AlignmentScope::ComputeAlignment;

        let plen = pattern.len() as i32;
        let tlen = text.len() as i32;

        let max_ops = (plen + tlen + 1) as usize;
        self.cigar.resize(max_ops);
        self.status = AlignStatus::Ok;

        // Create sub-aligners: fwd/rev use modular storage (score-only),
        // base uses full storage (compute alignment for fallback).
        let mut fwd = Self::new_biwfa_sub(self.penalties.clone());
        let mut rev = Self::new_biwfa_sub(self.penalties.clone());
        let mut base = WavefrontAligner::new(self.penalties.clone());
        base.alignment_scope = AlignmentScope::ComputeAlignment;

        let score = Self::bialign_recursive(
            &mut self.cigar,
            &self.penalties,
            self.max_alignment_steps,
            &mut fwd,
            &mut rev,
            &mut base,
            pattern,
            text,
            0,
            plen,
            0,
            tlen,
            ComponentType::M,
            ComponentType::M,
            i32::MAX,
            0,
        );

        self.cigar.end_v = plen;
        self.cigar.end_h = tlen;
        self.status = AlignStatus::Completed;

        score
    }

    /// End-to-end alignment using a custom match function (Lambda mode).
    pub fn align_lambda(
        &mut self,
        match_funct: Box<dyn Fn(i32, i32) -> bool>,
        pattern_length: i32,
        text_length: i32,
    ) -> i32 {
        self.sequences
            .init_lambda(match_funct, pattern_length, text_length, false);
        self.align_end2end_inner()
    }

    /// End-to-end alignment using packed 2-bit DNA sequences.
    pub fn align_packed2bits(
        &mut self,
        pattern: &[u8],
        pattern_length: usize,
        text: &[u8],
        text_length: usize,
    ) -> i32 {
        self.sequences
            .init_packed2bits(pattern, pattern_length, text, text_length, false);
        self.align_end2end_inner()
    }

    /// Create a sub-aligner for BiWFA with modular wavefront storage (score-only).
    fn new_biwfa_sub(penalties: WavefrontPenalties) -> Self {
        let wf_components = WavefrontComponents::new(10, 10, &penalties, true, false);
        Self {
            sequences: WavefrontSequences::new(),
            wf_components,
            wavefront_slab: WavefrontSlab::new(10, false, SlabMode::Reuse),
            penalties,
            alignment_scope: AlignmentScope::ComputeScore,
            alignment_span: AlignmentSpan::End2End,
            cigar: Cigar::new(0),
            alignment_end_pos: WavefrontPos::default(),
            status: AlignStatus::Ok,
            num_null_steps: 0,
            pattern_begin_free: 0,
            pattern_end_free: 0,
            text_begin_free: 0,
            text_end_free: 0,
            extension: false,
            heuristic: HeuristicStrategy::None,
            heuristic_state: HeuristicState::default(),
            max_alignment_steps: i32::MAX,
        }
    }

    /// Recursive BiWFA divide-and-conquer alignment.
    #[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
    fn bialign_recursive(
        cigar: &mut Cigar,
        penalties: &WavefrontPenalties,
        max_steps: i32,
        fwd: &mut WavefrontAligner,
        rev: &mut WavefrontAligner,
        base: &mut WavefrontAligner,
        pattern: &[u8],
        text: &[u8],
        pb: i32,
        pe: i32,
        tb: i32,
        te: i32,
        component_begin: ComponentType,
        component_end: ComponentType,
        score_remaining: i32,
        level: i32,
    ) -> i32 {
        let plen = pe - pb;
        let tlen = te - tb;

        // Trivial cases: empty sequences
        if tlen == 0 {
            if plen > 0 {
                cigar.append_deletion(plen as usize);
            }
            return 0;
        }
        if plen == 0 {
            if tlen > 0 {
                cigar.append_insertion(tlen as usize);
            }
            return 0;
        }

        // Fallback to standard WFA for small subproblems
        const MIN_LENGTH: i32 = 100;
        if score_remaining <= BIALIGN_FALLBACK_MIN_SCORE || plen.max(tlen) <= MIN_LENGTH {
            return Self::bialign_base(
                cigar,
                penalties,
                base,
                pattern,
                text,
                pb,
                pe,
                tb,
                te,
                component_begin,
                component_end,
            );
        }

        // Find breakpoint via bidirectional expansion
        let p_sub = &pattern[pb as usize..pe as usize];
        let t_sub = &text[tb as usize..te as usize];
        let mut breakpoint = BiAlignBreakpoint::new();

        let found = Self::bialign_find_breakpoint(
            penalties,
            max_steps,
            fwd,
            rev,
            p_sub,
            t_sub,
            &mut breakpoint,
        );

        if !found {
            return Self::bialign_base(
                cigar,
                penalties,
                base,
                pattern,
                text,
                pb,
                pe,
                tb,
                te,
                component_begin,
                component_end,
            );
        }

        // Split at breakpoint (forward h,v coordinates)
        let bp_h = wavefront_h(breakpoint.k_forward, breakpoint.offset_forward);
        let bp_v = wavefront_v(breakpoint.k_forward, breakpoint.offset_forward);

        // Recurse on first half [pb..pb+bp_v, tb..tb+bp_h]
        Self::bialign_recursive(
            cigar,
            penalties,
            max_steps,
            fwd,
            rev,
            base,
            pattern,
            text,
            pb,
            pb + bp_v,
            tb,
            tb + bp_h,
            component_begin,
            breakpoint.component,
            breakpoint.score_forward,
            level + 1,
        );

        // Recurse on second half [pb+bp_v..pe, tb+bp_h..te]
        Self::bialign_recursive(
            cigar,
            penalties,
            max_steps,
            fwd,
            rev,
            base,
            pattern,
            text,
            pb + bp_v,
            pe,
            tb + bp_h,
            te,
            breakpoint.component,
            component_end,
            breakpoint.score_reverse,
            level + 1,
        );

        breakpoint.score
    }

    /// Two-phase breakpoint detection for BiWFA.
    #[allow(clippy::too_many_arguments)]
    fn bialign_find_breakpoint(
        penalties: &WavefrontPenalties,
        max_steps: i32,
        fwd: &mut WavefrontAligner,
        rev: &mut WavefrontAligner,
        pattern: &[u8],
        text: &[u8],
        breakpoint: &mut BiAlignBreakpoint,
    ) -> bool {
        let plen = pattern.len() as i32;
        let tlen = text.len() as i32;

        // Init sub-aligners
        fwd.sequences.init_ascii(pattern, text, false);
        fwd.init_alignment();
        rev.sequences.init_ascii(pattern, text, true);
        rev.init_alignment();

        let has_affine = matches!(
            penalties.distance_metric,
            DistanceMetric::GapAffine | DistanceMetric::GapAffine2p
        );
        let has_affine2p = matches!(penalties.distance_metric, DistanceMetric::GapAffine2p);
        let gap_opening = match penalties.distance_metric {
            DistanceMetric::GapAffine => penalties.gap_opening1,
            DistanceMetric::GapAffine2p => penalties.gap_opening1.max(penalties.gap_opening2),
            _ => 0,
        };

        let max_antidiag = plen + tlen - 1;
        let mut score_fwd = 0i32;
        let mut score_rev = 0i32;
        let mut last_was_forward = true;

        // Extend at score 0
        let mut fwd_max_ak = fwd.extend_biwfa_max(0);
        let mut rev_max_ak = rev.extend_biwfa_max(0);

        // Phase 1: Alternate compute+extend until antidiagonals meet
        while fwd_max_ak + rev_max_ak < max_antidiag {
            // Forward
            score_fwd += 1;
            fwd.compute_step(score_fwd);
            let ak = fwd.extend_biwfa_max(score_fwd);
            if ak > fwd_max_ak {
                fwd_max_ak = ak;
            }
            last_was_forward = true;
            if fwd_max_ak + rev_max_ak >= max_antidiag {
                break;
            }

            // Reverse
            score_rev += 1;
            rev.compute_step(score_rev);
            let ak = rev.extend_biwfa_max(score_rev);
            if ak > rev_max_ak {
                rev_max_ak = ak;
            }
            last_was_forward = false;

            if score_fwd + score_rev >= max_steps {
                return false;
            }
        }

        // First overlap check (on whichever side was last extended)
        if last_was_forward {
            bialign::bialign_overlap(
                &fwd.wf_components,
                &fwd.wavefront_slab,
                &rev.wf_components,
                &rev.wavefront_slab,
                score_fwd,
                score_rev,
                plen,
                tlen,
                penalties.gap_opening1,
                penalties.gap_opening2,
                has_affine,
                has_affine2p,
                true,
                breakpoint,
            );
        } else {
            bialign::bialign_overlap(
                &rev.wf_components,
                &rev.wavefront_slab,
                &fwd.wf_components,
                &fwd.wavefront_slab,
                score_rev,
                score_fwd,
                plen,
                tlen,
                penalties.gap_opening1,
                penalties.gap_opening2,
                has_affine,
                has_affine2p,
                false,
                breakpoint,
            );
        }

        // Phase 2: Continue alternating with overlap checks
        while score_fwd + score_rev - gap_opening < breakpoint.score {
            // Forward
            score_fwd += 1;
            fwd.compute_step(score_fwd);
            fwd.extend_biwfa(score_fwd);
            bialign::bialign_overlap(
                &fwd.wf_components,
                &fwd.wavefront_slab,
                &rev.wf_components,
                &rev.wavefront_slab,
                score_fwd,
                score_rev,
                plen,
                tlen,
                penalties.gap_opening1,
                penalties.gap_opening2,
                has_affine,
                has_affine2p,
                true,
                breakpoint,
            );
            if score_fwd + score_rev - gap_opening >= breakpoint.score {
                break;
            }

            // Reverse
            score_rev += 1;
            rev.compute_step(score_rev);
            rev.extend_biwfa(score_rev);
            bialign::bialign_overlap(
                &rev.wf_components,
                &rev.wavefront_slab,
                &fwd.wf_components,
                &fwd.wavefront_slab,
                score_rev,
                score_fwd,
                plen,
                tlen,
                penalties.gap_opening1,
                penalties.gap_opening2,
                has_affine,
                has_affine2p,
                false,
                breakpoint,
            );

            if score_fwd + score_rev >= max_steps {
                return false;
            }
        }

        breakpoint.found()
    }

    /// Base case: run standard WFA with component-aware backtrace.
    #[allow(clippy::too_many_arguments)]
    fn bialign_base(
        cigar: &mut Cigar,
        penalties: &WavefrontPenalties,
        base: &mut WavefrontAligner,
        pattern: &[u8],
        text: &[u8],
        pb: i32,
        pe: i32,
        tb: i32,
        te: i32,
        component_begin: ComponentType,
        component_end: ComponentType,
    ) -> i32 {
        let p_sub = &pattern[pb as usize..pe as usize];
        let t_sub = &text[tb as usize..te as usize];

        // Init sequences and alignment with the specified starting component
        base.sequences.init_ascii(p_sub, t_sub, false);
        base.init_alignment_component(component_begin);

        // Alignment loop: extend M for match propagation, but check
        // the component_end wavefront for termination.
        let mut score = 0;
        loop {
            // Always extend M wavefront (needed for correct propagation)
            let m_idx = base.wf_components.get_m_idx(score as usize);
            if m_idx != WAVEFRONT_IDX_NONE {
                let wf = base.wavefront_slab.get_mut(m_idx);
                if base.sequences.mode == SequenceMode::Lambda {
                    extend::extend_matches_custom_end2end(&base.sequences, wf);
                } else {
                    extend::extend_matches_packed_end2end(&base.sequences, wf);
                }
            }

            // Check termination on the component_end wavefront
            if Self::check_component_termination(base, score, component_end) {
                break;
            }

            score += 1;
            base.compute_step(score);

            // For non-M components, also check right after compute
            // (since I/D wavefronts are finalized by compute, not extend)
            if component_end != ComponentType::M
                && Self::check_component_termination(base, score, component_end)
            {
                break;
            }

            if score >= base.max_alignment_steps {
                base.status = AlignStatus::MaxStepsReached;
                break;
            }
        }

        // Backtrace with component-aware start/end
        if base.status == AlignStatus::EndReached {
            let pos = base.alignment_end_pos;
            match penalties.distance_metric {
                DistanceMetric::GapAffine => {
                    backtrace::backtrace_affine(
                        &base.wf_components,
                        &base.wavefront_slab,
                        penalties,
                        base.sequences.pattern_length,
                        base.sequences.text_length,
                        pos.score,
                        pos.k,
                        pos.offset,
                        component_begin,
                        component_end,
                        &mut base.cigar,
                    );
                }
                DistanceMetric::GapAffine2p => {
                    backtrace::backtrace_affine2p(
                        &base.wf_components,
                        &base.wavefront_slab,
                        penalties,
                        base.sequences.pattern_length,
                        base.sequences.text_length,
                        pos.score,
                        pos.k,
                        pos.offset,
                        component_begin,
                        component_end,
                        &mut base.cigar,
                    );
                }
                _ => {
                    backtrace::backtrace_linear(
                        &base.wf_components,
                        &base.wavefront_slab,
                        penalties,
                        base.sequences.pattern_length,
                        base.sequences.text_length,
                        pos.score,
                        pos.k,
                        pos.offset,
                        &mut base.cigar,
                    );
                }
            }
            cigar.append_forward(&base.cigar);
        }

        score
    }

    /// Initialize alignment with a specific starting component (for BiWFA base case).
    fn init_alignment_component(&mut self, component: ComponentType) {
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        if self.alignment_scope == AlignmentScope::ComputeAlignment {
            let max_ops = (plen + tlen + 1) as usize;
            self.cigar.resize(max_ops);
        }

        self.wf_components.resize(plen, tlen, &self.penalties);
        self.wavefront_slab.clear();

        self.num_null_steps = 0;
        self.status = AlignStatus::Ok;
        self.alignment_end_pos = WavefrontPos::default();
        self.heuristic_state.reset(&self.heuristic);

        let max_score_scope = self.wf_components.max_score_scope as i32;
        let eff_lo = -(plen + max_score_scope + 2);
        let eff_hi = tlen + max_score_scope + 2;

        self.wf_components.historic_min_lo = eff_lo;
        self.wf_components.historic_max_hi = eff_hi;
        self.wavefront_slab.ensure_min_length(eff_lo, eff_hi);
        self.wf_components.resize_null_victim(eff_lo, eff_hi);

        // Allocate initial wavefront at score 0
        let idx = self.wavefront_slab.allocate(eff_lo, eff_hi);
        {
            let wf = self.wavefront_slab.get_mut(idx);
            wf.set_offset(0, 0);
            wf.set_limits(0, 0);
        }

        // Set the wavefront for the specified starting component
        match component {
            ComponentType::M => self.wf_components.set_m_idx(0, idx),
            ComponentType::I1 => self.wf_components.set_i1_idx(0, idx),
            ComponentType::D1 => self.wf_components.set_d1_idx(0, idx),
            ComponentType::I2 => self.wf_components.set_i2_idx(0, idx),
            ComponentType::D2 => self.wf_components.set_d2_idx(0, idx),
        }
    }

    /// Check if the specified wavefront component has reached the end diagonal.
    fn check_component_termination(
        base: &mut WavefrontAligner,
        score: i32,
        component: ComponentType,
    ) -> bool {
        let plen = base.sequences.pattern_length;
        let tlen = base.sequences.text_length;
        let k_end = tlen - plen;

        let idx = match component {
            ComponentType::M => base.wf_components.get_m_idx(score as usize),
            ComponentType::I1 => base.wf_components.get_i1_idx(score as usize),
            ComponentType::D1 => base.wf_components.get_d1_idx(score as usize),
            ComponentType::I2 => base.wf_components.get_i2_idx(score as usize),
            ComponentType::D2 => base.wf_components.get_d2_idx(score as usize),
        };

        if idx == WAVEFRONT_IDX_NONE {
            return false;
        }

        let wf = base.wavefront_slab.get(idx);
        if k_end >= wf.lo && k_end <= wf.hi && wf.get_offset(k_end) >= tlen {
            base.alignment_end_pos = WavefrontPos {
                score,
                k: k_end,
                offset: tlen,
            };
            base.status = AlignStatus::EndReached;
            true
        } else {
            false
        }
    }

    /// Dispatch compute step based on distance metric.
    fn compute_step(&mut self, score: i32) {
        match self.penalties.distance_metric {
            DistanceMetric::Edit | DistanceMetric::Indel => self.compute_edit_step(score),
            DistanceMetric::GapLinear => self.compute_linear_step(score),
            DistanceMetric::GapAffine => self.compute_affine_step(score),
            DistanceMetric::GapAffine2p => self.compute_affine2p_step(score),
        }
    }

    /// Extend wavefront for BiWFA, returning max antidiagonal.
    fn extend_biwfa_max(&mut self, score: i32) -> i32 {
        let m_idx = self.wf_components.get_m_idx(score as usize);
        if m_idx == WAVEFRONT_IDX_NONE {
            return 0;
        }
        let wf = self.wavefront_slab.get_mut(m_idx);
        if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_end2end_max(&self.sequences, wf)
        } else {
            extend::extend_matches_packed_end2end_max(&self.sequences, wf)
        }
    }

    /// Extend wavefront for BiWFA (no max tracking).
    fn extend_biwfa(&mut self, score: i32) {
        let m_idx = self.wf_components.get_m_idx(score as usize);
        if m_idx == WAVEFRONT_IDX_NONE {
            return;
        }
        let wf = self.wavefront_slab.get_mut(m_idx);
        if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_end2end(&self.sequences, wf);
        } else {
            extend::extend_matches_packed_end2end(&self.sequences, wf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penalties::WavefrontPenalties;

    #[test]
    fn test_identical_sequences() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_one_mismatch() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 1);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_one_insertion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 1);
    }

    #[test]
    fn test_one_deletion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACGT", b"ACT");
        assert_eq!(score, 1);
    }

    #[test]
    fn test_multiple_edits() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        // kitten → sitting: 3 edits (k→s, e→i, +g)
        let score = aligner.align_end2end(b"kitten", b"sitting");
        assert_eq!(score, 3);
    }

    #[test]
    fn test_empty_vs_nonempty() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"", b"ACGT");
        assert_eq!(score, 4);
    }

    #[test]
    fn test_nonempty_vs_empty() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACGT", b"");
        assert_eq!(score, 4);
    }

    #[test]
    fn test_completely_different() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"AAAA", b"TTTT");
        assert_eq!(score, 4);
    }

    #[test]
    fn test_longer_sequences() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_end2end(b"ACGTACGTACGTACGT", b"ACGTACGTACGTACGT");
        assert_eq!(score, 0);
    }

    #[test]
    fn test_indel_distance() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
        // Indel distance: only insertions and deletions, no substitutions
        // "ACGT" vs "ACTT": need to delete G, insert T → indel dist = 2
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 2);
    }

    #[test]
    fn test_indel_identical() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
    }

    #[test]
    fn test_indel_insertion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 1);
    }

    #[test]
    fn test_reuse_aligner() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score1 = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score1, 1);

        let score2 = aligner.align_end2end(b"AAA", b"BBB");
        assert_eq!(score2, 3);
    }

    #[test]
    fn test_max_steps_limit() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.set_max_alignment_steps(2);
        let score = aligner.align_end2end(b"AAAA", b"TTTT");
        assert_eq!(aligner.status(), AlignStatus::MaxStepsReached);
        // Score should be 2 (the limit), not the actual edit distance 4
        assert!(score <= 4);
    }

    // --- Gap-linear tests ---

    #[test]
    fn test_linear_identical() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_linear(
            crate::penalties::LinearPenalties {
                match_: 0,
                mismatch: 4,
                indel: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_linear_one_mismatch() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_linear(
            crate::penalties::LinearPenalties {
                match_: 0,
                mismatch: 4,
                indel: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        // Mismatch cost = 4, or ins+del = 2+2 = 4. Either way score = 4.
        assert_eq!(score, 4);
    }

    #[test]
    fn test_linear_one_insertion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_linear(
            crate::penalties::LinearPenalties {
                match_: 0,
                mismatch: 4,
                indel: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 2);
    }

    #[test]
    fn test_linear_with_cigar() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_linear(
            crate::penalties::LinearPenalties {
                match_: 0,
                mismatch: 4,
                indel: 2,
            },
        ));
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 2);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACT", b"ACGT").unwrap();
    }

    // --- Gap-affine tests ---

    #[test]
    fn test_affine_identical() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_affine_one_mismatch() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 4); // One mismatch = X=4
    }

    #[test]
    fn test_affine_one_insertion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 8); // One gap = O+E = 6+2 = 8
    }

    #[test]
    fn test_affine_with_cigar() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 4);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACGT", b"ACTT").unwrap();
    }

    #[test]
    fn test_affine_gap_extend() {
        // Two consecutive insertions should cost O + 2*E = 6 + 4 = 10
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"AT", b"ACGT");
        assert_eq!(score, 10); // O + 2*E = 6 + 4 = 10
        let cigar = aligner.cigar();
        cigar.check_alignment(b"AT", b"ACGT").unwrap();
    }

    #[test]
    fn test_affine_reuse() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        let score1 = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score1, 0);
        let score2 = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score2, 4);
    }

    // --- Gap-affine 2-piece tests ---

    fn make_affine2p_aligner() -> WavefrontAligner {
        WavefrontAligner::new(WavefrontPenalties::new_affine2p(
            crate::penalties::Affine2pPenalties {
                match_: 0,
                mismatch: 4,
                gap_opening1: 6,
                gap_extension1: 2,
                gap_opening2: 24,
                gap_extension2: 1,
            },
        ))
    }

    #[test]
    fn test_affine2p_identical() {
        let mut aligner = make_affine2p_aligner();
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_affine2p_one_mismatch() {
        let mut aligner = make_affine2p_aligner();
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 4); // One mismatch = X=4
    }

    #[test]
    fn test_affine2p_one_insertion() {
        let mut aligner = make_affine2p_aligner();
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 8); // One gap = O1+E1 = 6+2 = 8
    }

    #[test]
    fn test_affine2p_with_cigar() {
        let mut aligner = make_affine2p_aligner();
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 4);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACGT", b"ACTT").unwrap();
    }

    #[test]
    fn test_affine2p_reuse() {
        let mut aligner = make_affine2p_aligner();
        let score1 = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score1, 0);
        let score2 = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score2, 4);
    }

    // --- Ends-free tests ---

    #[test]
    fn test_endsfree_text_suffix_free() {
        // Pattern "ACGT" (4) vs Text "ACGTXXXX" (8)
        // With text_end_free=4: pattern fully consumed, remaining text (4) is free
        // Edit distance: score should be 0 (perfect match of pattern against text prefix)
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.set_alignment_free_ends(0, 0, 0, 4);
        let score = aligner.align_endsfree(b"ACGT", b"ACGTXXXX");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_endsfree_pattern_suffix_free() {
        // Pattern "ACGTXXXX" (8) vs Text "ACGT" (4)
        // With pattern_end_free=4: text fully consumed, remaining pattern (4) is free
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.set_alignment_free_ends(0, 4, 0, 0);
        let score = aligner.align_endsfree(b"ACGTXXXX", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    #[test]
    fn test_endsfree_with_cigar() {
        // Ends-free with CIGAR computation
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        aligner.set_alignment_free_ends(0, 0, 0, 4);
        let score = aligner.align_endsfree(b"ACGT", b"ACGTXXXX");
        assert_eq!(score, 0);
        // CIGAR should cover pattern fully but not all of text
        let cigar = aligner.cigar();
        assert!(!cigar.is_null());
    }

    #[test]
    fn test_endsfree_with_mismatch() {
        // Pattern "ACGT" vs Text "ACTTXXXX"
        // With text_end_free=4: one mismatch at position 2
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.set_alignment_free_ends(0, 0, 0, 4);
        let score = aligner.align_endsfree(b"ACGT", b"ACTTXXXX");
        assert_eq!(score, 1);
    }

    #[test]
    fn test_endsfree_affine() {
        // Affine ends-free: pattern fully consumed with text suffix free
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        aligner.set_alignment_free_ends(0, 0, 0, 4);
        let score = aligner.align_endsfree(b"ACGT", b"ACGTXXXX");
        assert_eq!(score, 0);
    }

    // --- Extension alignment tests ---

    #[test]
    fn test_extension_basic() {
        // Extension mode: find maximal-scoring prefix
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        aligner.set_alignment_extension();
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        // "ACGTXXXX" vs "ACGTTTTT": first 4 match, then mismatches
        // Maxtrim should keep the matching prefix
        let _score = aligner.align_endsfree(b"ACGTXXXX", b"ACGTTTTT");
        let cigar = aligner.cigar();
        assert!(!cigar.is_null());
        // The maximal-scoring prefix should be the 4 matches
        assert_eq!(cigar.end_v, 4);
        assert_eq!(cigar.end_h, 4);
        assert_eq!(aligner.status(), AlignStatus::Partial);
    }

    #[test]
    fn test_extension_identical() {
        // Extension with identical sequences: no trim needed
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        aligner.set_alignment_extension();
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_endsfree(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
    }

    // --- BiWFA tests ---

    #[test]
    fn test_biwfa_base_case_matches_standard() {
        // This tests the base case fallback specifically (short sequences)
        let penalties = WavefrontPenalties::new_affine(crate::penalties::AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        let pattern = b"GCTGTTTGCCAGAATATGGTTTGACTCGAGAGGCTAGTCTACATGGCCGGTGTCACTTAGAAACAAGCGCCGTCCCTAGATACACCTGCCCACCACGGCTCGT";
        let text = b"GCCTGGTTCAAAACTACTGTGACTGCCAGAGCCTCTCAGGCTTGAGGGCAGACTTACAAAAGCGCGGTTCGAACTATCGCGACAACACCACGGTAACGTC";

        // Standard alignment
        let mut std_aligner = WavefrontAligner::new(penalties.clone());
        std_aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let std_score = std_aligner.align_end2end(pattern, text);
        let std_cigar = std_aligner.cigar().to_string_rle(true);
        std_aligner.cigar().check_alignment(pattern, text).unwrap();

        // BiWFA
        let mut biwfa_aligner = WavefrontAligner::new(penalties);
        let biwfa_score = biwfa_aligner.align_biwfa(pattern, text);
        let biwfa_cigar = biwfa_aligner.cigar().to_string_rle(true);

        assert_eq!(biwfa_score, std_score, "Scores differ");

        if let Err(e) = biwfa_aligner.cigar().check_alignment(pattern, text) {
            panic!(
                "BiWFA CIGAR invalid: {}\nStd CIGAR: {}\nBiWFA CIGAR: {}\nStd begin/end: {}/{}, BiWFA begin/end: {}/{}",
                e,
                std_cigar,
                biwfa_cigar,
                std_aligner.cigar().begin_offset,
                std_aligner.cigar().end_offset,
                biwfa_aligner.cigar().begin_offset,
                biwfa_aligner.cigar().end_offset,
            );
        }
    }

    #[test]
    fn test_biwfa_identical_sequences() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let score = aligner.align_biwfa(b"ACGTACGT", b"ACGTACGT");
        assert_eq!(score, 0);
        aligner
            .cigar()
            .check_alignment(b"ACGTACGT", b"ACGTACGT")
            .unwrap();
    }

    #[test]
    fn test_biwfa_edit_simple() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let pattern = b"ACGTACGT";
        let text = b"ACTTACGT";
        let biwfa_score = aligner.align_biwfa(pattern, text);
        aligner.cigar().check_alignment(pattern, text).unwrap();

        // Compare with standard WFA
        let mut std_aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        std_aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let std_score = std_aligner.align_end2end(pattern, text);
        assert_eq!(biwfa_score, std_score);
    }

    #[test]
    fn test_biwfa_edit_with_indels() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        let pattern = b"ACGTACGTACGT";
        let text = b"ACGACGTAGT";
        let biwfa_score = aligner.align_biwfa(pattern, text);
        aligner.cigar().check_alignment(pattern, text).unwrap();

        let mut std_aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        std_aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let std_score = std_aligner.align_end2end(pattern, text);
        assert_eq!(biwfa_score, std_score);
    }

    #[test]
    fn test_biwfa_affine_simple() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        let pattern = b"ACGTACGTACGTACGT";
        let text = b"ACTTACGTACGTACTT";
        let biwfa_score = aligner.align_biwfa(pattern, text);
        aligner.cigar().check_alignment(pattern, text).unwrap();

        let mut std_aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(
            crate::penalties::AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2,
            },
        ));
        std_aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let std_score = std_aligner.align_end2end(pattern, text);
        assert_eq!(biwfa_score, std_score);
    }
}
