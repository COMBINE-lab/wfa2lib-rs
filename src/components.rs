//! Wavefront components: manages the arrays of wavefronts for each matrix type
//! (M, I1, D1, I2, D2) across all scores.
//!
//! In non-modular mode, wavefronts are stored for all scores (index = score).
//! In modular mode, wavefronts are stored modulo `max_score_scope` (circular buffer).
//!
//! The generic parameter `N` controls how many component slots are stored per score:
//! - N=1: Edit, Indel, GapLinear (only M wavefront used)
//! - N=3: GapAffine (M + I1 + D1)
//! - N=5: GapAffine2p (M + I1 + D1 + I2 + D2)
//!
//! `flat[score]` = `[m_ptr, i1_ptr, d1_ptr, i2_ptr, d2_ptr]` — all N pointers for a given
//! score are contiguous in memory, fitting in a single cache line for N=5 (40 bytes).

use crate::bt_buffer::BacktraceBuffer;
use crate::offset::wavefront_length;
use crate::penalties::{DistanceMetric, WavefrontPenalties};
use crate::wavefront::Wavefront;

/// Sentinel pointer indicating no wavefront is allocated.
pub const WF_PTR_NONE: *mut Wavefront = std::ptr::null_mut();

/// Initial size for null/victim wavefronts.
const WF_NULL_INIT_LO: i32 = -1024;
const WF_NULL_INIT_HI: i32 = 1024;

/// Component index constants.
pub const COMP_M: usize = 0;
pub const COMP_I1: usize = 1;
pub const COMP_D1: usize = 2;
pub const COMP_I2: usize = 3;
pub const COMP_D2: usize = 4;

/// Manages all wavefront arrays and the null/victim wavefronts.
///
/// The const generic `N` is the number of component slots per score:
/// 1 for edit/indel/linear, 3 for affine, 5 for affine2p.
pub struct WavefrontComponents<const N: usize> {
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
    /// Number of flat slots used by the previous alignment (= last_score + 1).
    /// Used in non-modular mode to clear only the needed range on the next alignment.
    pub prev_score_count: usize,

    // Interleaved wavefront pointer array.
    // flat[score] = [m_ptr, i1_ptr, d1_ptr, i2_ptr, d2_ptr] (first N are valid).
    // All N pointers for a given score are contiguous — fits in one cache line for N=5.
    pub flat: Vec<[*mut Wavefront; N]>,

    // Special wavefronts (owned directly, not in slab)
    /// Null wavefront: reads return OFFSET_NULL.
    pub wavefront_null: Wavefront,
    /// Victim wavefront: writes are discarded.
    pub wavefront_victim: Wavefront,

    // BT-Buffer
    /// Backtrace buffer (only if bt_piggyback is true).
    pub bt_buffer: Option<BacktraceBuffer>,
}

// Raw pointers in Wavefront (null/victim) are owned by this struct, and
// WavefrontComponents is never shared between threads while mutated.
unsafe impl<const N: usize> Send for WavefrontComponents<N> {}

impl<const N: usize> WavefrontComponents<N> {
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
            prev_score_count: 0,
            flat: vec![[WF_PTR_NONE; N]; num_wavefronts],
            wavefront_null,
            wavefront_victim,
            bt_buffer,
        }
    }

    /// Clear all wavefront slots and reset historic limits.
    ///
    /// In non-modular (CIGAR) mode: skip filling the flat array.
    /// The forward pass always overwrites every entry [0..final_score] before
    /// the backtrace reads them, so nulling stale entries is unnecessary.
    /// This matches C's wavefront_components_clear() which only clears in
    /// modular mode (where the circular buffer needs to be reset).
    ///
    /// In modular (score-only) mode: only clear the circular buffer entries
    /// (max_score_scope entries), not the full oversized flat array.
    pub fn clear(&mut self) {
        if self.memory_modular {
            // Modular: clear only the slots used by the circular buffer.
            let n = self.max_score_scope.min(self.flat.len());
            self.flat[..n].fill([WF_PTR_NONE; N]);
        } else {
            // Non-modular: clear only the slots used by the previous alignment.
            // Invariant: flat[prev_score_count..] is already null, so clearing
            // [0..prev_score_count] leaves the entire flat array null.
            // This avoids the 160KB fill for the full num_wavefronts array.
            let n = self.prev_score_count.min(self.flat.len());
            if n > 0 {
                self.flat[..n].fill([WF_PTR_NONE; N]);
            }
            self.prev_score_count = 0;
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
            self.prev_score_count = 0; // new array is already null-filled
            self.flat = vec![[WF_PTR_NONE; N]; num_wavefronts];
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

    // --- Accessor helpers ---

    #[inline(always)]
    fn slot(&self, score: usize) -> usize {
        if self.memory_modular {
            score % self.max_score_scope
        } else {
            score
        }
    }

    /// Get the wavefront pointer for M-wavefronts at a given score.
    #[inline(always)]
    pub fn get_m_ptr(&self, score: usize) -> *mut Wavefront {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked(s)[COMP_M] }
    }

    /// Set the wavefront pointer for M-wavefronts at a given score.
    #[inline(always)]
    pub fn set_m_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked_mut(s)[COMP_M] = ptr; }
    }

    /// Get the wavefront pointer for I1-wavefronts at a given score.
    #[inline(always)]
    pub fn get_i1_ptr(&self, score: usize) -> *mut Wavefront {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked(s)[COMP_I1] }
    }

    /// Set the wavefront pointer for I1-wavefronts at a given score.
    #[inline(always)]
    pub fn set_i1_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked_mut(s)[COMP_I1] = ptr; }
    }

    /// Get the wavefront pointer for D1-wavefronts at a given score.
    #[inline(always)]
    pub fn get_d1_ptr(&self, score: usize) -> *mut Wavefront {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked(s)[COMP_D1] }
    }

    /// Set the wavefront pointer for D1-wavefronts at a given score.
    #[inline(always)]
    pub fn set_d1_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked_mut(s)[COMP_D1] = ptr; }
    }

    /// Get the wavefront pointer for I2-wavefronts at a given score.
    #[inline(always)]
    pub fn get_i2_ptr(&self, score: usize) -> *mut Wavefront {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked(s)[COMP_I2] }
    }

    /// Set the wavefront pointer for I2-wavefronts at a given score.
    #[inline(always)]
    pub fn set_i2_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked_mut(s)[COMP_I2] = ptr; }
    }

    /// Get the wavefront pointer for D2-wavefronts at a given score.
    #[inline(always)]
    pub fn get_d2_ptr(&self, score: usize) -> *mut Wavefront {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked(s)[COMP_D2] }
    }

    /// Set the wavefront pointer for D2-wavefronts at a given score.
    #[inline(always)]
    pub fn set_d2_ptr(&mut self, score: usize, ptr: *mut Wavefront) {
        let s = self.slot(score);
        unsafe { self.flat.get_unchecked_mut(s)[COMP_D2] = ptr; }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_components() {
        let penalties = WavefrontPenalties::new_edit();
        let comp = WavefrontComponents::<1>::new(100, 100, &penalties, false, false);
        assert_eq!(comp.max_score_scope, 2);
        assert_eq!(comp.flat.len(), 101);
        assert!(!comp.flat.is_empty());
    }

    #[test]
    fn test_affine_components() {
        let penalties = WavefrontPenalties::new_affine(crate::penalties::AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        let comp = WavefrontComponents::<3>::new(100, 100, &penalties, false, false);
        // max_score_scope = max(6+2, 4) + 1 = 9
        assert_eq!(comp.max_score_scope, 9);
        assert_eq!(comp.flat[0].len(), 3);
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
        let comp = WavefrontComponents::<5>::new(100, 100, &penalties, false, false);
        assert_eq!(comp.flat[0].len(), 5);
    }

    #[test]
    fn test_modular_mode() {
        let penalties = WavefrontPenalties::new_edit();
        let comp = WavefrontComponents::<1>::new(100, 100, &penalties, true, false);
        assert_eq!(comp.num_wavefronts, 2); // modular edit: 2 slots
    }

    #[test]
    fn test_set_get_m_ptr() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::<1>::new(100, 100, &penalties, false, false);
        // Use a non-null sentinel pointer for testing
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
    }

    #[test]
    fn test_set_get_m_ptr_modular() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::<1>::new(100, 100, &penalties, true, false);
        // max_score_scope = 2, so score 5 maps to 5 % 2 = 1
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
        assert_eq!(comp.get_m_ptr(7), fake_ptr); // 7 % 2 = 1 same slot
    }

    #[test]
    fn test_clear() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::<1>::new(100, 100, &penalties, true, false);
        let fake_ptr = 0x1234usize as *mut Wavefront;
        comp.set_m_ptr(0, fake_ptr);
        comp.clear();
        assert!(comp.get_m_ptr(0).is_null());
    }

    #[test]
    fn test_resize_null_victim() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::<1>::new(100, 100, &penalties, false, false);
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
        let comp = WavefrontComponents::<1>::new(100, 100, &penalties, false, true);
        assert!(comp.bt_buffer.is_some());
        assert!(comp.wavefront_null.has_backtrace());
    }

    #[test]
    fn test_access_within_bounds_non_modular() {
        let penalties = WavefrontPenalties::new_edit();
        let mut comp = WavefrontComponents::<1>::new(10, 10, &penalties, false, false);
        // Pre-sized to max(10,10)+1 = 11 slots
        let fake_ptr = 0x5678usize as *mut Wavefront;
        comp.set_m_ptr(5, fake_ptr);
        assert_eq!(comp.get_m_ptr(5), fake_ptr);
    }

    #[test]
    fn test_affine_i1_d1_accessors() {
        let penalties = WavefrontPenalties::new_affine(crate::penalties::AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        let mut comp = WavefrontComponents::<3>::new(100, 100, &penalties, false, false);
        let ptr10 = 0x10usize as *mut Wavefront;
        let ptr20 = 0x20usize as *mut Wavefront;
        comp.set_i1_ptr(3, ptr10);
        comp.set_d1_ptr(3, ptr20);
        assert_eq!(comp.get_i1_ptr(3), ptr10);
        assert_eq!(comp.get_d1_ptr(3), ptr20);
    }

    #[test]
    fn test_cache_line_layout() {
        // For N=5, each flat[score] is 5 * 8 = 40 bytes — fits in one cache line.
        assert_eq!(std::mem::size_of::<[*mut Wavefront; 5]>(), 40);
        // For N=3: 24 bytes.
        assert_eq!(std::mem::size_of::<[*mut Wavefront; 3]>(), 24);
        // For N=1: 8 bytes.
        assert_eq!(std::mem::size_of::<[*mut Wavefront; 1]>(), 8);
    }
}
