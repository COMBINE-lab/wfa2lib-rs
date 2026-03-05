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
use crate::components::WF_PTR_NONE;
use crate::slab::{SlabMode, WavefrontSlab};
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

    // BiWFA sub-aligners (lazily initialized, reused across align_biwfa calls)
    biwfa_fwd: Option<Box<WavefrontAligner>>,
    biwfa_rev: Option<Box<WavefrontAligner>>,
    biwfa_base: Option<Box<WavefrontAligner>>,
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
            biwfa_fwd: None,
            biwfa_rev: None,
            biwfa_base: None,
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

        // Enable modular wavefront storage for score-only mode (like C reference).
        // This keeps only max_score_scope wavefronts alive, dramatically reducing
        // memory usage and improving cache performance.
        let score_only = self.alignment_scope == AlignmentScope::ComputeScore;
        self.wf_components.memory_modular = score_only;

        // Resize components for sequence dimensions
        self.wf_components.resize(plen, tlen, &self.penalties);

        // Clear slab
        self.wavefront_slab.clear();

        // Pre-size slab to avoid Vec reallocations that would invalidate stored raw pointers.
        // In modular (score-only) mode, num_wavefronts == max_score_scope (circular buffer).
        // In non-modular (CIGAR) mode, num_wavefronts == full score range.
        {
            let num_types = match self.penalties.distance_metric {
                DistanceMetric::Edit | DistanceMetric::Indel => 1,
                DistanceMetric::GapLinear => 1,
                DistanceMetric::GapAffine => 3,
                DistanceMetric::GapAffine2p => 5,
            };
            self.wavefront_slab
                .reserve(self.wf_components.num_wavefronts * num_types);
        }

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
        let init_ptr = self.wavefront_slab.allocate_ptr(eff_lo, eff_hi);
        unsafe {
            (*init_ptr).set_offset(0, 0);
            (*init_ptr).set_limits(0, 0);
        }
        self.wf_components.set_m_ptr(0, init_ptr);
    }

    /// Extend the M-wavefront at the given score, check termination.
    /// Returns true if alignment is finished.
    fn extend_end2end(&mut self, score: i32) -> bool {
        let score_idx = score as usize;
        let m_ptr = self.wf_components.get_m_ptr(score_idx);

        if m_ptr.is_null() {
            self.num_null_steps += 1;
            if self.num_null_steps > self.wf_components.max_score_scope as i32 {
                self.status = AlignStatus::EndUnreachable;
                return true;
            }
            return false;
        }
        self.num_null_steps = 0;

        // Extend matches
        let wf = unsafe { &mut *m_ptr };
        if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_end2end(&self.sequences, wf);
        } else {
            extend::extend_matches_packed_end2end(&self.sequences, wf);
        }

        // Check termination
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;
        let wf = unsafe { &*m_ptr };
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
            && self.apply_heuristic_cutoff(score, m_ptr)
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
        let m_ptr = self.wf_components.get_m_ptr(score_idx);

        if m_ptr.is_null() {
            self.num_null_steps += 1;
            if self.num_null_steps > self.wf_components.max_score_scope as i32 {
                self.status = AlignStatus::EndUnreachable;
                return true;
            }
            return false;
        }
        self.num_null_steps = 0;

        // Extend matches with ends-free termination check
        let wf = unsafe { &mut *m_ptr };
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
            && self.apply_heuristic_cutoff(score, m_ptr)
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
    fn apply_heuristic_cutoff(&mut self, score: i32, m_ptr: *mut crate::wavefront::Wavefront) -> bool {
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;

        // Use the victim wavefront's offsets as scratch buffer for distances
        let distances_base_k = self.wf_components.wavefront_victim.base_k();
        let distances = self.wf_components.wavefront_victim.offsets_slice_mut();

        let wf = unsafe { &mut *m_ptr };
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
                let m_lo = unsafe { (*m_ptr).lo };
                let m_hi = unsafe { (*m_ptr).hi };

                let i1_ptr = self.wf_components.get_i1_ptr(score_idx);
                if !i1_ptr.is_null() {
                    let i1_wf = unsafe { &mut *i1_ptr };
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

                let d1_ptr = self.wf_components.get_d1_ptr(score_idx);
                if !d1_ptr.is_null() {
                    let d1_wf = unsafe { &mut *d1_ptr };
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
                let m_lo = unsafe { (*m_ptr).lo };
                let m_hi = unsafe { (*m_ptr).hi };

                for get_ptr_fn in [
                    WavefrontComponents::get_i1_ptr,
                    WavefrontComponents::get_d1_ptr,
                    WavefrontComponents::get_i2_ptr,
                    WavefrontComponents::get_d2_ptr,
                ] {
                    let wf_ptr = get_ptr_fn(&self.wf_components, score_idx);
                    if !wf_ptr.is_null() {
                        let wf = unsafe { &mut *wf_ptr };
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

        let prev_ptr = self.wf_components.get_m_ptr(score_prev);
        // Get prev wavefront dimensions
        let (prev_lo, prev_hi) = unsafe { ((*prev_ptr).lo, (*prev_ptr).hi) };

        let lo = prev_lo - 1;
        let hi = prev_hi + 1;

        // Initialize ends on previous wavefront (like C's wavefront_compute_init_ends)
        // Edit kernel reads prev[k-1], prev[k], prev[k+1] for k in [lo, hi]
        // so prev needs [lo-1, hi+1]
        unsafe { (*prev_ptr).init_ends_higher(hi + 1); (*prev_ptr).init_ends_lower(lo - 1); }

        // Allocate or reuse output wavefront
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        let curr_ptr = if self.wf_components.memory_modular {
            let old_ptr = self.wf_components.get_m_ptr(score_curr);
            self.wavefront_slab.reuse_or_allocate_ptr(old_ptr, hist_lo, hist_hi)
        } else {
            self.wavefront_slab.allocate_ptr(hist_lo, hist_hi)
        };
        unsafe { (*curr_ptr).set_limits(lo, hi); }

        // Compute kernel
        let plen = self.sequences.pattern_length;
        let tlen = self.sequences.text_length;
        unsafe {
            match self.penalties.distance_metric {
                DistanceMetric::Edit => {
                    edit::compute_edit_idm(plen, tlen, &mut *prev_ptr, &mut *curr_ptr, lo, hi);
                }
                DistanceMetric::Indel => {
                    edit::compute_indel_idm(plen, tlen, &mut *prev_ptr, &mut *curr_ptr, lo, hi);
                }
                _ => unreachable!("compute_edit_step called for non-edit metric"),
            }
        }

        // Store in components
        self.wf_components.set_m_ptr(score_curr, curr_ptr);

        // Trim ends
        {
            let wf_curr = unsafe { &mut *curr_ptr };
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

        let misms_ptr = if misms_score >= 0 {
            self.wf_components.get_m_ptr(misms_score as usize)
        } else {
            WF_PTR_NONE
        };
        let gap_ptr = if gap_score >= 0 {
            self.wf_components.get_m_ptr(gap_score as usize)
        } else {
            WF_PTR_NONE
        };

        // Get effective lo/hi from input wavefronts
        let (misms_lo, misms_hi) = if !misms_ptr.is_null() {
            unsafe { ((*misms_ptr).lo, (*misms_ptr).hi) }
        } else {
            (0, -1) // empty range
        };
        let (gap_lo, gap_hi) = if !gap_ptr.is_null() {
            unsafe { ((*gap_ptr).lo, (*gap_ptr).hi) }
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
        if !misms_ptr.is_null() {
            unsafe { (*misms_ptr).init_ends_higher(hi); (*misms_ptr).init_ends_lower(lo); }
        }
        // gap needs [lo-1, hi+1] (kernel reads k-1 and k+1)
        if !gap_ptr.is_null() {
            unsafe { (*gap_ptr).init_ends_higher(hi + 1); (*gap_ptr).init_ends_lower(lo - 1); }
        }

        // Allocate or reuse output wavefront
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        let curr_ptr = if self.wf_components.memory_modular {
            let old_ptr = self.wf_components.get_m_ptr(score_curr);
            self.wavefront_slab.reuse_or_allocate_ptr(old_ptr, hist_lo, hist_hi)
        } else {
            self.wavefront_slab.allocate_ptr(hist_lo, hist_hi)
        };
        unsafe { (*curr_ptr).set_limits(lo, hi); }

        // Compute kernel using raw pointers for multi-borrow
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;

            let wf_misms = if !misms_ptr.is_null() { misms_ptr as *const _ } else { null_wf };
            let wf_gap = if !gap_ptr.is_null() { gap_ptr as *const _ } else { null_wf };

            linear::compute_linear_idm(
                plen,
                tlen,
                &*wf_misms,
                &*wf_gap,
                &mut *curr_ptr,
                lo,
                hi,
            );
        }

        // Store in components
        self.wf_components.set_m_ptr(score_curr, curr_ptr);

        // Trim ends
        {
            let wf_curr = unsafe { &mut *curr_ptr };
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

        // Fetch input wavefront pointers directly from components
        let ptr_m_misms = if score - x >= 0 { self.wf_components.get_m_ptr((score - x) as usize) } else { WF_PTR_NONE };
        let ptr_m_open  = if score - o_plus_e >= 0 { self.wf_components.get_m_ptr((score - o_plus_e) as usize) } else { WF_PTR_NONE };
        let ptr_i1_ext  = if score - e >= 0 { self.wf_components.get_i1_ptr((score - e) as usize) } else { WF_PTR_NONE };
        let ptr_d1_ext  = if score - e >= 0 { self.wf_components.get_d1_ptr((score - e) as usize) } else { WF_PTR_NONE };

        // Compute effective lo/hi from cached pointers (no slab re-lookup)
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        unsafe {
            if !ptr_m_misms.is_null() { lo = lo.min((*ptr_m_misms).lo);     hi = hi.max((*ptr_m_misms).hi);     }
            if !ptr_m_open.is_null()  { lo = lo.min((*ptr_m_open).lo - 1);  hi = hi.max((*ptr_m_open).hi + 1);  }
            if !ptr_i1_ext.is_null()  { lo = lo.min((*ptr_i1_ext).lo + 1);  hi = hi.max((*ptr_i1_ext).hi + 1);  }
            if !ptr_d1_ext.is_null()  { lo = lo.min((*ptr_d1_ext).lo - 1);  hi = hi.max((*ptr_d1_ext).hi - 1);  }
        }

        if lo > hi {
            // No valid inputs — free old modular slot and set to none
            if self.wf_components.memory_modular {
                self.free_output_wavefronts_affine(score_curr);
            }
            return;
        }

        // Clamp lo/hi to leave room for kernel's k±1 reads within allocated range.
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        lo = lo.max(hist_lo + 1);
        hi = hi.min(hist_hi - 1);

        if lo > hi {
            if self.wf_components.memory_modular {
                self.free_output_wavefronts_affine(score_curr);
            }
            return;
        }

        // Ensure null/victim wavefronts cover the range
        self.wf_components.resize_null_victim(lo, hi);

        // Initialize input wavefront ends using cached raw pointers (no slab re-lookup)
        unsafe {
            if !ptr_m_misms.is_null() { (*ptr_m_misms).init_ends_higher(hi);     (*ptr_m_misms).init_ends_lower(lo);     }
            if !ptr_m_open.is_null()  { (*ptr_m_open).init_ends_higher(hi + 1);  (*ptr_m_open).init_ends_lower(lo - 1);  }
            if !ptr_i1_ext.is_null()  { (*ptr_i1_ext).init_ends_higher(hi);      (*ptr_i1_ext).init_ends_lower(lo - 1);  }
            if !ptr_d1_ext.is_null()  { (*ptr_d1_ext).init_ends_higher(hi + 1);  (*ptr_d1_ext).init_ends_lower(lo);      }
        }

        // Allocate or reuse 3 output wavefronts
        let (m_curr_ptr, i1_curr_ptr, d1_curr_ptr) = if self.wf_components.memory_modular {
            let old_m = self.wf_components.get_m_ptr(score_curr);
            let old_i1 = self.wf_components.get_i1_ptr(score_curr);
            let old_d1 = self.wf_components.get_d1_ptr(score_curr);
            (
                self.wavefront_slab.reuse_or_allocate_ptr(old_m, hist_lo, hist_hi),
                self.wavefront_slab.reuse_or_allocate_ptr(old_i1, hist_lo, hist_hi),
                self.wavefront_slab.reuse_or_allocate_ptr(old_d1, hist_lo, hist_hi),
            )
        } else {
            (
                self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
            )
        };

        // Compute kernel using raw pointers for multi-borrow
        // Input ptrs come directly from components; outputs are fresh.
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;
            let wf_m_misms = if !ptr_m_misms.is_null() { ptr_m_misms as *const _ } else { null_wf };
            let wf_m_open  = if !ptr_m_open.is_null()  { ptr_m_open  as *const _ } else { null_wf };
            let wf_i1_ext  = if !ptr_i1_ext.is_null()  { ptr_i1_ext  as *const _ } else { null_wf };
            let wf_d1_ext  = if !ptr_d1_ext.is_null()  { ptr_d1_ext  as *const _ } else { null_wf };

            // set_limits on output wavefronts
            (*m_curr_ptr).set_limits(lo, hi);
            (*i1_curr_ptr).set_limits(lo, hi);
            (*d1_curr_ptr).set_limits(lo, hi);

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
        self.wf_components.set_m_ptr(score_curr, m_curr_ptr);
        self.wf_components.set_i1_ptr(score_curr, i1_curr_ptr);
        self.wf_components.set_d1_ptr(score_curr, d1_curr_ptr);

        // Trim ends on M wavefront to detect null steps.
        // I1/D1 don't need trimming: the NEON/scalar kernel already clamps
        // out-of-bounds M offsets to OFFSET_NULL. I1/D1 values are only read
        // at k±1 (within allocated range) and never used for termination.
        {
            let wf_curr = unsafe { &mut *m_curr_ptr };
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

        // Fetch input wavefront pointers directly from components
        // SAFETY: slab.wavefronts is pre-reserved; resize_null_victim does not touch the slab.
        let ptr_m_misms = if score - x >= 0 { self.wf_components.get_m_ptr((score - x) as usize) } else { WF_PTR_NONE };
        let ptr_m_open1 = if score - o1_plus_e1 >= 0 { self.wf_components.get_m_ptr((score - o1_plus_e1) as usize) } else { WF_PTR_NONE };
        let ptr_m_open2 = if score - o2_plus_e2 >= 0 { self.wf_components.get_m_ptr((score - o2_plus_e2) as usize) } else { WF_PTR_NONE };
        let ptr_i1_ext  = if score - e1 >= 0 { self.wf_components.get_i1_ptr((score - e1) as usize) } else { WF_PTR_NONE };
        let ptr_i2_ext  = if score - e2 >= 0 { self.wf_components.get_i2_ptr((score - e2) as usize) } else { WF_PTR_NONE };
        let ptr_d1_ext  = if score - e1 >= 0 { self.wf_components.get_d1_ptr((score - e1) as usize) } else { WF_PTR_NONE };
        let ptr_d2_ext  = if score - e2 >= 0 { self.wf_components.get_d2_ptr((score - e2) as usize) } else { WF_PTR_NONE };

        // Compute effective lo/hi from cached pointers (no slab re-lookup)
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        unsafe {
            if !ptr_m_misms.is_null() { lo = lo.min((*ptr_m_misms).lo);     hi = hi.max((*ptr_m_misms).hi);     }
            if !ptr_m_open1.is_null() { lo = lo.min((*ptr_m_open1).lo - 1); hi = hi.max((*ptr_m_open1).hi + 1); }
            if !ptr_m_open2.is_null() { lo = lo.min((*ptr_m_open2).lo - 1); hi = hi.max((*ptr_m_open2).hi + 1); }
            if !ptr_i1_ext.is_null()  { lo = lo.min((*ptr_i1_ext).lo  + 1); hi = hi.max((*ptr_i1_ext).hi  + 1); }
            if !ptr_i2_ext.is_null()  { lo = lo.min((*ptr_i2_ext).lo  + 1); hi = hi.max((*ptr_i2_ext).hi  + 1); }
            if !ptr_d1_ext.is_null()  { lo = lo.min((*ptr_d1_ext).lo  - 1); hi = hi.max((*ptr_d1_ext).hi  - 1); }
            if !ptr_d2_ext.is_null()  { lo = lo.min((*ptr_d2_ext).lo  - 1); hi = hi.max((*ptr_d2_ext).hi  - 1); }
        }

        if lo > hi {
            if self.wf_components.memory_modular {
                self.free_output_wavefronts_affine2p(score_curr);
            }
            return;
        }

        // Clamp lo/hi to leave room for kernel's k±1 reads
        let hist_lo = self.wf_components.historic_min_lo;
        let hist_hi = self.wf_components.historic_max_hi;
        lo = lo.max(hist_lo + 1);
        hi = hi.min(hist_hi - 1);

        if lo > hi {
            if self.wf_components.memory_modular {
                self.free_output_wavefronts_affine2p(score_curr);
            }
            return;
        }

        // Ensure null/victim wavefronts cover the range
        self.wf_components.resize_null_victim(lo, hi);

        // Initialize input wavefront ends using cached raw pointers (no slab re-lookup)
        unsafe {
            if !ptr_m_misms.is_null() { (*ptr_m_misms).init_ends_higher(hi);     (*ptr_m_misms).init_ends_lower(lo);     }
            if !ptr_m_open1.is_null() { (*ptr_m_open1).init_ends_higher(hi + 1); (*ptr_m_open1).init_ends_lower(lo - 1); }
            if !ptr_m_open2.is_null() { (*ptr_m_open2).init_ends_higher(hi + 1); (*ptr_m_open2).init_ends_lower(lo - 1); }
            if !ptr_i1_ext.is_null()  { (*ptr_i1_ext).init_ends_higher(hi);      (*ptr_i1_ext).init_ends_lower(lo - 1);  }
            if !ptr_i2_ext.is_null()  { (*ptr_i2_ext).init_ends_higher(hi);      (*ptr_i2_ext).init_ends_lower(lo - 1);  }
            if !ptr_d1_ext.is_null()  { (*ptr_d1_ext).init_ends_higher(hi + 1);  (*ptr_d1_ext).init_ends_lower(lo);      }
            if !ptr_d2_ext.is_null()  { (*ptr_d2_ext).init_ends_higher(hi + 1);  (*ptr_d2_ext).init_ends_lower(lo);      }
        }

        // Allocate or reuse 5 output wavefronts
        let (m_curr_ptr, i1_curr_ptr, i2_curr_ptr, d1_curr_ptr, d2_curr_ptr) =
            if self.wf_components.memory_modular {
                let old_m = self.wf_components.get_m_ptr(score_curr);
                let old_i1 = self.wf_components.get_i1_ptr(score_curr);
                let old_i2 = self.wf_components.get_i2_ptr(score_curr);
                let old_d1 = self.wf_components.get_d1_ptr(score_curr);
                let old_d2 = self.wf_components.get_d2_ptr(score_curr);
                (
                    self.wavefront_slab.reuse_or_allocate_ptr(old_m, hist_lo, hist_hi),
                    self.wavefront_slab.reuse_or_allocate_ptr(old_i1, hist_lo, hist_hi),
                    self.wavefront_slab.reuse_or_allocate_ptr(old_i2, hist_lo, hist_hi),
                    self.wavefront_slab.reuse_or_allocate_ptr(old_d1, hist_lo, hist_hi),
                    self.wavefront_slab.reuse_or_allocate_ptr(old_d2, hist_lo, hist_hi),
                )
            } else {
                (
                    self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                    self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                    self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                    self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                    self.wavefront_slab.allocate_ptr(hist_lo, hist_hi),
                )
            };

        // Compute kernel using raw pointers for multi-borrow
        // Input ptrs come directly from components; outputs are fresh.
        unsafe {
            let null_wf = &self.wf_components.wavefront_null as *const _;
            let wf_m_misms = if !ptr_m_misms.is_null() { ptr_m_misms as *const _ } else { null_wf };
            let wf_m_open1 = if !ptr_m_open1.is_null() { ptr_m_open1 as *const _ } else { null_wf };
            let wf_m_open2 = if !ptr_m_open2.is_null() { ptr_m_open2 as *const _ } else { null_wf };
            let wf_i1_ext  = if !ptr_i1_ext.is_null()  { ptr_i1_ext  as *const _ } else { null_wf };
            let wf_i2_ext  = if !ptr_i2_ext.is_null()  { ptr_i2_ext  as *const _ } else { null_wf };
            let wf_d1_ext  = if !ptr_d1_ext.is_null()  { ptr_d1_ext  as *const _ } else { null_wf };
            let wf_d2_ext  = if !ptr_d2_ext.is_null()  { ptr_d2_ext  as *const _ } else { null_wf };

            // set_limits on output wavefronts
            (*m_curr_ptr).set_limits(lo, hi);
            (*i1_curr_ptr).set_limits(lo, hi);
            (*i2_curr_ptr).set_limits(lo, hi);
            (*d1_curr_ptr).set_limits(lo, hi);
            (*d2_curr_ptr).set_limits(lo, hi);

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
        self.wf_components.set_m_ptr(score_curr, m_curr_ptr);
        self.wf_components.set_i1_ptr(score_curr, i1_curr_ptr);
        self.wf_components.set_i2_ptr(score_curr, i2_curr_ptr);
        self.wf_components.set_d1_ptr(score_curr, d1_curr_ptr);
        self.wf_components.set_d2_ptr(score_curr, d2_curr_ptr);

        // Trim ends on M wavefront only
        {
            let wf_curr = unsafe { &mut *m_curr_ptr };
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

        // Lazily create sub-aligners and reuse them across calls (avoids 3×N alloc/free
        // per alignment). fwd/rev use modular storage; base uses full CIGAR storage.
        if self.biwfa_fwd.is_none() {
            self.biwfa_fwd = Some(Box::new(Self::new_biwfa_sub(self.penalties.clone())));
            self.biwfa_rev = Some(Box::new(Self::new_biwfa_sub(self.penalties.clone())));
            let mut b = WavefrontAligner::new(self.penalties.clone());
            b.alignment_scope = AlignmentScope::ComputeAlignment;
            self.biwfa_base = Some(Box::new(b));
        }

        // Take sub-aligners out of self so we can borrow self.cigar/penalties separately.
        let mut fwd = self.biwfa_fwd.take().unwrap();
        let mut rev = self.biwfa_rev.take().unwrap();
        let mut base = self.biwfa_base.take().unwrap();

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

        // Return sub-aligners to self for reuse on the next call.
        self.biwfa_fwd = Some(fwd);
        self.biwfa_rev = Some(rev);
        self.biwfa_base = Some(base);

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
            biwfa_fwd: None,
            biwfa_rev: None,
            biwfa_base: None,
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
            component_begin,
            component_end,
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
        component_begin: ComponentType,
        component_end: ComponentType,
        breakpoint: &mut BiAlignBreakpoint,
    ) -> bool {
        let plen = pattern.len() as i32;
        let tlen = text.len() as i32;

        // Init sub-aligners with correct starting components
        // Forward starts from component_begin, reverse starts from component_end
        fwd.sequences.init_ascii(pattern, text, false);
        fwd.init_alignment_component(component_begin);
        rev.sequences.init_ascii(pattern, text, true);
        rev.init_alignment_component(component_end);

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

        // Phase 2: Alternate compute+extend with overlap checks
        // Match C reference: overlap check first, then compute opposite direction
        let max_score_scope = fwd.wf_components.max_score_scope as i32;

        loop {
            if last_was_forward {
                // Check forward overlap
                let min_score_rev = if score_rev > max_score_scope - 1 {
                    score_rev - (max_score_scope - 1)
                } else {
                    0
                };
                if score_fwd + min_score_rev - gap_opening >= breakpoint.score {
                    break;
                }
                bialign::bialign_overlap(
                    &fwd.wf_components,
                    &rev.wf_components,
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

                // Compute+extend reverse
                score_rev += 1;
                rev.compute_step(score_rev);
                rev.extend_biwfa(score_rev);
            }

            // Check reverse overlap
            let min_score_fwd = if score_fwd > max_score_scope - 1 {
                score_fwd - (max_score_scope - 1)
            } else {
                0
            };
            if min_score_fwd + score_rev - gap_opening >= breakpoint.score {
                break;
            }
            bialign::bialign_overlap(
                &rev.wf_components,
                &fwd.wf_components,
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

            // Compute+extend forward
            score_fwd += 1;
            fwd.compute_step(score_fwd);
            fwd.extend_biwfa(score_fwd);

            if score_fwd + score_rev >= max_steps {
                return false;
            }

            last_was_forward = true;
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
            let m_ptr_base = base.wf_components.get_m_ptr(score as usize);
            if !m_ptr_base.is_null() {
                let wf = unsafe { &mut *m_ptr_base };
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

        // Pre-size slab to prevent Vec reallocation that would invalidate stored pointers.
        // In modular mode, num_wavefronts == max_score_scope (circular buffer size).
        // In non-modular mode, num_wavefronts == full score range.
        {
            let num_types = match self.penalties.distance_metric {
                DistanceMetric::Edit | DistanceMetric::Indel => 1,
                DistanceMetric::GapLinear => 1,
                DistanceMetric::GapAffine => 3,
                DistanceMetric::GapAffine2p => 5,
            };
            self.wavefront_slab
                .reserve(self.wf_components.num_wavefronts * num_types);
        }

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
        let init_ptr = self.wavefront_slab.allocate_ptr(eff_lo, eff_hi);
        unsafe {
            (*init_ptr).set_offset(0, 0);
            (*init_ptr).set_limits(0, 0);
        }

        // Set the wavefront for the specified starting component
        match component {
            ComponentType::M => self.wf_components.set_m_ptr(0, init_ptr),
            ComponentType::I1 => self.wf_components.set_i1_ptr(0, init_ptr),
            ComponentType::D1 => self.wf_components.set_d1_ptr(0, init_ptr),
            ComponentType::I2 => self.wf_components.set_i2_ptr(0, init_ptr),
            ComponentType::D2 => self.wf_components.set_d2_ptr(0, init_ptr),
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

        let wf_ptr = match component {
            ComponentType::M => base.wf_components.get_m_ptr(score as usize),
            ComponentType::I1 => base.wf_components.get_i1_ptr(score as usize),
            ComponentType::D1 => base.wf_components.get_d1_ptr(score as usize),
            ComponentType::I2 => base.wf_components.get_i2_ptr(score as usize),
            ComponentType::D2 => base.wf_components.get_d2_ptr(score as usize),
        };

        if wf_ptr.is_null() {
            return false;
        }

        let wf = unsafe { &*wf_ptr };
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

    /// Free old wavefronts at the modular slot for affine (M, I1, D1).
    fn free_output_wavefronts_affine(&mut self, score: usize) {
        let score_mod = score % self.wf_components.max_score_scope;
        let old_m = self.wf_components.get_m_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_m);
        let old_i1 = self.wf_components.get_i1_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_i1);
        let old_d1 = self.wf_components.get_d1_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_d1);
        self.wf_components.set_m_ptr(score, WF_PTR_NONE);
        self.wf_components.set_i1_ptr(score, WF_PTR_NONE);
        self.wf_components.set_d1_ptr(score, WF_PTR_NONE);
    }

    fn free_output_wavefronts_affine2p(&mut self, score: usize) {
        let score_mod = score % self.wf_components.max_score_scope;
        let old_m = self.wf_components.get_m_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_m);
        let old_i1 = self.wf_components.get_i1_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_i1);
        let old_d1 = self.wf_components.get_d1_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_d1);
        let old_i2 = self.wf_components.get_i2_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_i2);
        let old_d2 = self.wf_components.get_d2_ptr(score_mod);
        self.wavefront_slab.free_ptr(old_d2);
        self.wf_components.set_m_ptr(score, WF_PTR_NONE);
        self.wf_components.set_i1_ptr(score, WF_PTR_NONE);
        self.wf_components.set_d1_ptr(score, WF_PTR_NONE);
        self.wf_components.set_i2_ptr(score, WF_PTR_NONE);
        self.wf_components.set_d2_ptr(score, WF_PTR_NONE);
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
        let m_ptr = self.wf_components.get_m_ptr(score as usize);
        if m_ptr.is_null() {
            return 0;
        }
        let wf = unsafe { &mut *m_ptr };
        if self.sequences.mode == SequenceMode::Lambda {
            extend::extend_matches_custom_end2end_max(&self.sequences, wf)
        } else {
            extend::extend_matches_packed_end2end_max(&self.sequences, wf)
        }
    }

    /// Extend wavefront for BiWFA (no max tracking).
    fn extend_biwfa(&mut self, score: i32) {
        let m_ptr = self.wf_components.get_m_ptr(score as usize);
        if m_ptr.is_null() {
            return;
        }
        let wf = unsafe { &mut *m_ptr };
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
