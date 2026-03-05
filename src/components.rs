//! Wavefront components: manages the arrays of wavefronts for each matrix type
//! (M, I1, D1, I2, D2) across all scores.
//!
//! In non-modular mode, wavefronts are stored for all scores (index = score).
//! In modular mode, wavefronts are stored modulo `max_score_scope` (circular buffer).

use crate::bt_buffer::BacktraceBuffer;
use crate::offset::wavefront_length;
use crate::penalties::{DistanceMetric, WavefrontPenalties};
use crate::wavefront::Wavefront;

/// Sentinel raw pointer indicating no wavefront.
pub const WF_PTR_NONE: *mut Wavefront = std::ptr::null_mut();

/// Initial size for null/victim wavefronts.
const WF_NULL_INIT_LO: i32 = -1024;
const WF_NULL_INIT_HI: i32 = 1024;

/// Manages all wavefront arrays and the null/victim wavefronts.
pub struct WavefrontComponents {
    // Configuration
    /// Whether to use modular (circular) wavefront storage.
    pub memory_modular: bool,
    /// Whether to use piggyback backtrace.
    pub bt_piggyback: bool,

    // Dimensions
    /// Total number of wavefront slots allocated.
    pub num_wavefronts: usize,
    /// Maximum score-difference between dependent wavefronts.
    pub max_score_scope: usize,
    /// Maximum hi-limit seen during current alignment.
    pub historic_max_hi: i32,
    /// Minimum lo-limit seen during current alignment.
    pub historic_min_lo: i32,

    // Wavefront arrays (indexed by score or score % max_score_scope)
    /// M-wavefronts (match/mismatch).
    pub mwavefronts: Vec<*mut Wavefront>,
    /// I1-wavefronts (insertion piece 1). Empty if not needed.
    pub i1wavefronts: Vec<*mut Wavefront>,
    /// D1-wavefronts (deletion piece 1). Empty if not needed.
    pub d1wavefronts: Vec<*mut Wavefront>,
    /// I2-wavefronts (insertion piece 2, affine2p only). Empty if not needed.
    pub i2wavefronts: Vec<*mut Wavefront>,
    /// D2-wavefronts (deletion piece 2, affine2p only). Empty if not needed.
    pub d2wavefronts: Vec<*mut Wavefront>,

    // Special wavefronts (owned directly, not in slab)
    /// Null wavefront: reads return OFFSET_NULL.
    pub wavefront_null: Wavefront,
    /// Victim wavefront: writes are discarded.
    pub wavefront_victim: Wavefront,

    // BT-Buffer
    /// Backtrace buffer (only if bt_piggyback is true).
    pub bt_buffer: Option<BacktraceBuffer>,
}

// SAFETY: Raw pointers make WavefrontComponents !Send by default.
// The pointers are only valid during an alignment (slab is pre-reserved and stable).
// The aligner is not shared across threads during alignment.
unsafe impl Send for WavefrontComponents {}

impl WavefrontComponents {
    /// Allocate wavefront components for the given parameters.
    pub fn new(
        max_pattern_length: i32,
        max_text_length: i32,
        penalties: &WavefrontPenalties,
        memory_modular: bool,
        bt_piggyback: bool,
    ) -> Self {
        let (max_score_scope, num_wavefronts) = Self::compute_dimensions(
            penalties,
            max_pattern_length,
            max_text_length,
            memory_modular,
        );

        let (i1wavefronts, d1wavefronts, i2wavefronts, d2wavefronts) =
            Self::allocate_wf_arrays(num_wavefronts, penalties.distance_metric);

        // Allocate null wavefront
        let null_len = wavefront_length(WF_NULL_INIT_LO, WF_NULL_INIT_HI);
        let mut wavefront_null = Wavefront::allocate(null_len, bt_piggyback);
        wavefront_null.init_null(WF_NULL_INIT_LO, WF_NULL_INIT_HI);

        // Allocate victim wavefront
        let mut wavefront_victim = Wavefront::allocate(null_len, bt_piggyback);
        wavefront_victim.init_victim(WF_NULL_INIT_LO, WF_NULL_INIT_HI);

        let bt_buffer = if bt_piggyback {
            Some(BacktraceBuffer::new())
        } else {
            None
        };

        Self {
            memory_modular,
            bt_piggyback,
            num_wavefronts,
            max_score_scope,
            historic_max_hi: 0,
            historic_min_lo: 0,
            mwavefronts: vec![WF_PTR_NONE; num_wavefronts],
            i1wavefronts,
            d1wavefronts,
            i2wavefronts,
            d2wavefronts,
            wavefront_null,
            wavefront_victim,
            bt_buffer,
        }
    }

    /// Clear all wavefront slots and reset historic limits.
    pub fn clear(&mut self) {
        self.mwavefronts.fill(WF_PTR_NONE);
        if !self.i1wavefronts.is_empty() {
            self.i1wavefronts.fill(WF_PTR_NONE);
        }
        if !self.d1wavefronts.is_empty() {
            self.d1wavefronts.fill(WF_PTR_NONE);
        }
        if !self.i2wavefronts.is_empty() {
            self.i2wavefronts.fill(WF_PTR_NONE);
        }
        if !self.d2wavefronts.is_empty() {
            self.d2wavefronts.fill(WF_PTR_NONE);
        }
        self.historic_max_hi = 0;
        self.historic_min_lo = 0;
        if let Some(ref mut bt_buffer) = self.bt_buffer {
            bt_buffer.clear();
        }
    }

    /// Reap: free BT buffer.
    pub fn reap(&mut self) {
        if let Some(ref mut bt_buffer) = self.bt_buffer {
            bt_buffer.reap();
        }
    }

    /// Resize the wavefront components for new sequence lengths.
    pub fn resize(
        &mut self,
        max_pattern_length: i32,
        max_text_length: i32,
        penalties: &WavefrontPenalties,
    ) {
        let (max_score_scope, num_wavefronts) = Self::compute_dimensions(
            penalties,
            max_pattern_length,
            max_text_length,
            self.memory_modular,
        );
        self.max_score_scope = max_score_scope;

        if num_wavefronts > self.num_wavefronts {
            self.num_wavefronts = num_wavefronts;
            let (i1, d1, i2, d2) =
                Self::allocate_wf_arrays(num_wavefronts, penalties.distance_metric);
            self.mwavefronts = vec![WF_PTR_NONE; num_wavefronts];
            self.i1wavefronts = i1;
            self.d1wavefronts = d1;
            self.i2wavefronts = i2;
            self.d2wavefronts = d2;
            if let Some(ref mut bt_buffer) = self.bt_buffer {
                bt_buffer.clear();
            }
        } else {
            self.clear();
        }
    }

    /// Resize null and victim wavefronts if the diagonal range exceeds their bounds.
    pub fn resize_null_victim(&mut self, lo: i32, hi: i32) {
        if lo - 1 < self.wavefront_null.wf_elements_init_min
            || hi + 1 > self.wavefront_null.wf_elements_init_max
        {
            let wf_len = wavefront_length(lo, hi);
            let wf_inc = (wf_len * 3) / 2;
            let proposed_lo = lo - wf_inc / 2;
            let proposed_hi = hi + wf_inc / 2;
            let proposed_length = wavefront_length(proposed_lo, proposed_hi);

            self.wavefront_victim.resize(proposed_length);
            self.wavefront_victim.init_victim(proposed_lo, proposed_hi);

            self.wavefront_null.resize(proposed_length);
            self.wavefront_null.init_null(proposed_lo, proposed_hi);
        }
    }

    /// Get the raw pointer to the M-wavefront at a given score.
    #[inline(always)]
    pub fn get_m_ptr(&self, score: usize) -> *mut Wavefront {
        if self.memory_modular {
            unsafe { *self.mwavefronts.get_unchecked(score % self.max_score_scope) }
        } else {
            // SAFETY: Vecs pre-sized to upper-bound score by resize()/compute_dimensions().
            unsafe { *self.mwavefronts.get_unchecked(score) }
        }
    }

    /// Set the raw pointer to the M-wavefront at a given score.
    #[inline(always)]
    pub fn set_m_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        if self.memory_modular {
            unsafe { *self.mwavefronts.get_unchecked_mut(score % self.max_score_scope) = ptr; }
        } else {
            // SAFETY: Vecs pre-sized to upper-bound score by resize()/compute_dimensions().
            unsafe { *self.mwavefronts.get_unchecked_mut(score) = ptr; }
        }
    }

    /// Get the raw pointer to the I1-wavefront at a given score.
    #[inline(always)]
    pub fn get_i1_ptr(&self, score: usize) -> *mut Wavefront {
        if self.memory_modular {
            unsafe { *self.i1wavefronts.get_unchecked(score % self.max_score_scope) }
        } else {
            unsafe { *self.i1wavefronts.get_unchecked(score) }
        }
    }

    /// Set the raw pointer to the I1-wavefront at a given score.
    #[inline(always)]
    pub fn set_i1_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        if self.memory_modular {
            unsafe { *self.i1wavefronts.get_unchecked_mut(score % self.max_score_scope) = ptr; }
        } else {
            unsafe { *self.i1wavefronts.get_unchecked_mut(score) = ptr; }
        }
    }

    /// Get the raw pointer to the D1-wavefront at a given score.
    #[inline(always)]
    pub fn get_d1_ptr(&self, score: usize) -> *mut Wavefront {
        if self.memory_modular {
            unsafe { *self.d1wavefronts.get_unchecked(score % self.max_score_scope) }
        } else {
            unsafe { *self.d1wavefronts.get_unchecked(score) }
        }
    }

    /// Set the raw pointer to the D1-wavefront at a given score.
    #[inline(always)]
    pub fn set_d1_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        if self.memory_modular {
            unsafe { *self.d1wavefronts.get_unchecked_mut(score % self.max_score_scope) = ptr; }
        } else {
            unsafe { *self.d1wavefronts.get_unchecked_mut(score) = ptr; }
        }
    }

    /// Get the raw pointer to the I2-wavefront at a given score.
    #[inline(always)]
    pub fn get_i2_ptr(&self, score: usize) -> *mut Wavefront {
        if self.memory_modular {
            unsafe { *self.i2wavefronts.get_unchecked(score % self.max_score_scope) }
        } else {
            unsafe { *self.i2wavefronts.get_unchecked(score) }
        }
    }

    /// Set the raw pointer to the I2-wavefront at a given score.
    #[inline(always)]
    pub fn set_i2_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        if self.memory_modular {
            unsafe { *self.i2wavefronts.get_unchecked_mut(score % self.max_score_scope) = ptr; }
        } else {
            unsafe { *self.i2wavefronts.get_unchecked_mut(score) = ptr; }
        }
    }

    /// Get the raw pointer to the D2-wavefront at a given score.
    #[inline(always)]
    pub fn get_d2_ptr(&self, score: usize) -> *mut Wavefront {
        if self.memory_modular {
            unsafe { *self.d2wavefronts.get_unchecked(score % self.max_score_scope) }
        } else {
            unsafe { *self.d2wavefronts.get_unchecked(score) }
        }
    }

    /// Set the raw pointer to the D2-wavefront at a given score.
    #[inline(always)]
    pub fn set_d2_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        if self.memory_modular {
            unsafe { *self.d2wavefronts.get_unchecked_mut(score % self.max_score_scope) = ptr; }
        } else {
            unsafe { *self.d2wavefronts.get_unchecked_mut(score) = ptr; }
        }
    }

    // --- Internal dimension computation ---

    fn compute_dimensions(
        penalties: &WavefrontPenalties,
        max_pattern_length: i32,
        max_text_length: i32,
        memory_modular: bool,
    ) -> (usize, usize) {
        match penalties.distance_metric {
            DistanceMetric::Indel | DistanceMetric::Edit => {
                let max_score_scope = 2;
                let num_wavefronts = if memory_modular {
                    2
                } else {
                    max_pattern_length.max(max_text_length) as usize + 1
                };
                (max_score_scope, num_wavefronts)
            }
            DistanceMetric::GapLinear => {
                let max_score_scope = penalties.mismatch.max(penalties.gap_opening1) as usize + 1;
                let num_wavefronts = if memory_modular {
                    max_score_scope
                } else {
                    let abs_diff = (max_pattern_length - max_text_length).unsigned_abs() as i32;
                    let max_score_misms =
                        max_pattern_length.min(max_text_length) * penalties.mismatch;
                    let max_score_indel = penalties.gap_opening1 * abs_diff;
                    (max_score_misms + max_score_indel + 1) as usize
                };
                (max_score_scope, num_wavefronts)
            }
            DistanceMetric::GapAffine => {
                let max_scope_indel = penalties.gap_opening1 + penalties.gap_extension1;
                let max_score_scope = max_scope_indel.max(penalties.mismatch) as usize + 1;
                let num_wavefronts = if memory_modular {
                    max_score_scope
                } else {
                    let abs_diff = (max_pattern_length - max_text_length).unsigned_abs() as i32;
                    let max_score_misms =
                        max_pattern_length.min(max_text_length) * penalties.mismatch;
                    let max_score_indel =
                        penalties.gap_opening1 + abs_diff * penalties.gap_extension1;
                    (max_score_misms + max_score_indel + 1) as usize
                };
                (max_score_scope, num_wavefronts)
            }
            DistanceMetric::GapAffine2p => {
                let max_scope_indel = (penalties.gap_opening1 + penalties.gap_extension1)
                    .max(penalties.gap_opening2 + penalties.gap_extension2);
                let max_score_scope = max_scope_indel.max(penalties.mismatch) as usize + 1;
                let num_wavefronts = if memory_modular {
                    max_score_scope
                } else {
                    let abs_diff = (max_pattern_length - max_text_length).unsigned_abs() as i32;
                    let max_score_misms =
                        max_pattern_length.min(max_text_length) * penalties.mismatch;
                    let max_score_indel1 =
                        penalties.gap_opening1 + abs_diff * penalties.gap_extension1;
                    let max_score_indel2 =
                        penalties.gap_opening2 + abs_diff * penalties.gap_extension2;
                    let max_score_indel = max_score_indel1.min(max_score_indel2);
                    (max_score_misms + max_score_indel + 1) as usize
                };
                (max_score_scope, num_wavefronts)
            }
        }
    }

    fn allocate_wf_arrays(
        num_wavefronts: usize,
        distance_metric: DistanceMetric,
    ) -> (
        Vec<*mut Wavefront>,
        Vec<*mut Wavefront>,
        Vec<*mut Wavefront>,
        Vec<*mut Wavefront>,
    ) {
        match distance_metric {
            DistanceMetric::Indel | DistanceMetric::Edit | DistanceMetric::GapLinear => {
                (vec![], vec![], vec![], vec![])
            }
            DistanceMetric::GapAffine => (
                vec![WF_PTR_NONE; num_wavefronts],
                vec![WF_PTR_NONE; num_wavefronts],
                vec![],
                vec![],
            ),
            DistanceMetric::GapAffine2p => (
                vec![WF_PTR_NONE; num_wavefronts],
                vec![WF_PTR_NONE; num_wavefronts],
                vec![WF_PTR_NONE; num_wavefronts],
                vec![WF_PTR_NONE; num_wavefronts],
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_components() {
        let penalties = WavefrontPenalties::new_edit();
        let comp = WavefrontComponents::new(100, 100, &penalties, false, false);
        assert_eq!(comp.max_score_scope, 2);
        assert!(comp.i1wavefronts.is_empty());
        assert!(comp.d1wavefronts.is_empty());
        assert!(!comp.mwavefronts.is_empty());
    }

    #[test]
    fn test_affine_components() {
        let penalties = WavefrontPenalties::new_affine(crate::penalties::AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        let comp = WavefrontComponents::new(100, 100, &penalties, false, false);
        // max_score_scope = max(6+2, 4) + 1 = 9
        assert_eq!(comp.max_score_scope, 9);
        assert!(!comp.i1wavefronts.is_empty());
        assert!(!comp.d1wavefronts.is_empty());
        assert!(comp.i2wavefronts.is_empty());
        assert!(comp.d2wavefronts.is_empty());
    }

    #[test]
    fn test_affine2p_components() {
        let penalties = WavefrontPenalties::new_affine2p(crate::penalties::Affine2pPenalties {
            match_: 0,
            mismatch: 4,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 24,
            gap_extension2: 1,
        });
        let comp = WavefrontComponents::new(100, 100, &penalties, false, false);
        assert!(!comp.i2wavefronts.is_empty());
        assert!(!comp.d2wavefronts.is_empty());
    }

    #[test]
    fn test_modular_mode() {
        let penalties = WavefrontPenalties::new_edit();
        let comp = WavefrontComponents::new(100, 100, &penalties, true, false);
        assert_eq!(comp.num_wavefronts, 2); // modular edit: 2 slots
    }

    #[test]
    fn test_set_get_m_ptr() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::new(100, 100, &penalties, false, false);
        // Use a non-null sentinel for testing (we just test the pointer round-trip)
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
    }

    #[test]
    fn test_set_get_m_ptr_modular() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::new(100, 100, &penalties, true, false);
        // max_score_scope = 2, so score 5 maps to 5 % 2 = 1
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
        assert_eq!(comp.get_m_ptr(7), fake_ptr); // 7 % 2 = 1 same slot
    }

    #[test]
    fn test_clear() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::new(100, 100, &penalties, true, false);
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(0, fake_ptr);
        comp.clear();
        assert_eq!(comp.get_m_ptr(0), WF_PTR_NONE);
    }

    #[test]
    fn test_resize_null_victim() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::new(100, 100, &penalties, false, false);
        // Initially covers [-1024, 1024]
        // Request something within bounds — no resize
        comp.resize_null_victim(-100, 100);

        // Request something out of bounds — should resize
        comp.resize_null_victim(-2000, 2000);
        assert!(comp.wavefront_null.wf_elements_init_min <= -2000);
        assert!(comp.wavefront_null.wf_elements_init_max >= 2000);
    }

    #[test]
    fn test_with_bt_buffer() {
        let penalties = WavefrontPenalties::new_edit();
        let comp = WavefrontComponents::new(100, 100, &penalties, false, true);
        assert!(comp.bt_buffer.is_some());
        assert!(comp.wavefront_null.has_backtrace());
    }

    #[test]
    fn test_access_within_bounds_non_modular() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::new(10, 10, &penalties, false, false);
        // Pre-sized to max(10,10)+1 = 11 slots
        let fake_ptr = 0x5678usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
    }
}
