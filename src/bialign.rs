//! BiWFA: Bidirectional Wavefront Alignment.
//!
//! Implements divide-and-conquer bidirectional alignment for O(s) memory usage.
//! Forward and reverse wavefronts expand toward each other until they overlap,
//! producing a breakpoint that splits the problem into two halves.

use crate::offset::{WfOffset, wavefront_h, wavefront_k_inverse, wavefront_v};
use crate::slab::WAVEFRONT_IDX_NONE;
use crate::wavefront::Wavefront;

/// DP matrix component type for BiWFA breakpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    M,
    I1,
    D1,
    I2,
    D2,
}

/// Breakpoint where forward and reverse wavefronts meet.
#[derive(Debug, Clone)]
pub struct BiAlignBreakpoint {
    /// Best total score found so far (i32::MAX = no breakpoint yet).
    pub score: i32,
    pub score_forward: i32,
    pub score_reverse: i32,
    pub k_forward: i32,
    pub k_reverse: i32,
    pub offset_forward: WfOffset,
    pub offset_reverse: WfOffset,
    pub component: ComponentType,
}

impl Default for BiAlignBreakpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl BiAlignBreakpoint {
    pub fn new() -> Self {
        Self {
            score: i32::MAX,
            score_forward: 0,
            score_reverse: 0,
            k_forward: 0,
            k_reverse: 0,
            offset_forward: 0,
            offset_reverse: 0,
            component: ComponentType::M,
        }
    }

    pub fn found(&self) -> bool {
        self.score < i32::MAX
    }
}

/// Get the gap opening penalty for overlap early-exit checks.
///
/// Returns the minimum gap opening penalty for the distance metric.
/// For edit/indel/linear: 0 (no gap opening concept or it's 0).
/// For affine: gap_opening1.
/// For affine2p: min(gap_opening1, gap_opening2).
pub fn overlap_gopen_adjust(gap_opening1: i32, gap_opening2: i32, has_affine2p: bool) -> i32 {
    if has_affine2p {
        gap_opening1.min(gap_opening2)
    } else {
        gap_opening1
    }
}

/// Check M-to-M wavefront overlap between forward (wf_0) and reverse (wf_1) wavefronts.
///
/// If `h_0 + h_1 >= text_length` for any overlapping diagonal, a breakpoint is found.
/// `breakpoint_forward`: if true, wf_0 is the forward aligner; if false, wf_0 is reverse.
#[allow(clippy::too_many_arguments)]
pub fn breakpoint_m2m(
    wf_0: &Wavefront,
    wf_1: &Wavefront,
    pattern_length: i32,
    text_length: i32,
    score_0: i32,
    score_1: i32,
    breakpoint_forward: bool,
    breakpoint: &mut BiAlignBreakpoint,
) {
    if score_0 + score_1 >= breakpoint.score {
        return;
    }

    let lo_0 = wf_0.lo;
    let hi_0 = wf_0.hi;

    // Convert reverse k-range to forward coordinates
    let lo_1_inv = wavefront_k_inverse(wf_1.hi, pattern_length, text_length);
    let hi_1_inv = wavefront_k_inverse(wf_1.lo, pattern_length, text_length);

    // Check k-range overlap
    if hi_1_inv < lo_0 || hi_0 < lo_1_inv {
        return;
    }

    let max_lo = lo_0.max(lo_1_inv);
    let min_hi = hi_0.min(hi_1_inv);

    let base_k_0 = wf_0.base_k();
    let base_k_1 = wf_1.base_k();
    let offsets_0 = wf_0.offsets_slice();
    let offsets_1 = wf_1.offsets_slice();

    for k_0 in max_lo..=min_hi {
        let k_1 = wavefront_k_inverse(k_0, pattern_length, text_length);

        let offset_0 = offsets_0[(k_0 - base_k_0) as usize];
        let offset_1 = offsets_1[(k_1 - base_k_1) as usize];

        let h_0 = wavefront_h(k_0, offset_0);
        let h_1 = wavefront_h(k_1, offset_1);

        if h_0 + h_1 >= text_length {
            if breakpoint_forward {
                breakpoint.score_forward = score_0;
                breakpoint.score_reverse = score_1;
                breakpoint.k_forward = k_0;
                breakpoint.k_reverse = k_1;
                breakpoint.offset_forward = offset_0;
                breakpoint.offset_reverse = offset_1;
            } else {
                breakpoint.score_forward = score_1;
                breakpoint.score_reverse = score_0;
                breakpoint.k_forward = k_1;
                breakpoint.k_reverse = k_0;
                breakpoint.offset_forward = offset_1;
                breakpoint.offset_reverse = offset_0;
            }
            breakpoint.score = score_0 + score_1;
            breakpoint.component = ComponentType::M;
            return;
        }
    }
}

/// Check I/D-to-I/D wavefront overlap between forward and reverse wavefronts.
///
/// Same as m2m but subtracts gap_opening from the combined score.
#[allow(clippy::too_many_arguments)]
pub fn breakpoint_indel2indel(
    wf_0: &Wavefront,
    wf_1: &Wavefront,
    pattern_length: i32,
    text_length: i32,
    score_0: i32,
    score_1: i32,
    gap_opening: i32,
    component: ComponentType,
    breakpoint_forward: bool,
    breakpoint: &mut BiAlignBreakpoint,
) {
    if score_0 + score_1 - gap_opening >= breakpoint.score {
        return;
    }

    let lo_0 = wf_0.lo;
    let hi_0 = wf_0.hi;

    let lo_1_inv = wavefront_k_inverse(wf_1.hi, pattern_length, text_length);
    let hi_1_inv = wavefront_k_inverse(wf_1.lo, pattern_length, text_length);

    if hi_1_inv < lo_0 || hi_0 < lo_1_inv {
        return;
    }

    let max_lo = lo_0.max(lo_1_inv);
    let min_hi = hi_0.min(hi_1_inv);

    let base_k_0 = wf_0.base_k();
    let base_k_1 = wf_1.base_k();
    let offsets_0 = wf_0.offsets_slice();
    let offsets_1 = wf_1.offsets_slice();

    for k_0 in max_lo..=min_hi {
        let k_1 = wavefront_k_inverse(k_0, pattern_length, text_length);

        let offset_0 = offsets_0[(k_0 - base_k_0) as usize];
        let offset_1 = offsets_1[(k_1 - base_k_1) as usize];

        let h_0 = wavefront_h(k_0, offset_0);
        let h_1 = wavefront_h(k_1, offset_1);

        if h_0 + h_1 >= text_length {
            // Extra bounds check for indel breakpoints
            let (check_k, check_h) = if breakpoint_forward {
                (k_0, h_0)
            } else {
                (k_1, h_1)
            };
            let v = wavefront_v(check_k, check_h);
            if v > pattern_length || check_h > text_length {
                continue;
            }

            if breakpoint_forward {
                breakpoint.score_forward = score_0;
                breakpoint.score_reverse = score_1;
                breakpoint.k_forward = k_0;
                breakpoint.k_reverse = k_1;
                breakpoint.offset_forward = offset_0;
                breakpoint.offset_reverse = offset_1;
            } else {
                breakpoint.score_forward = score_1;
                breakpoint.score_reverse = score_0;
                breakpoint.k_forward = k_1;
                breakpoint.k_reverse = k_0;
                breakpoint.offset_forward = offset_1;
                breakpoint.offset_reverse = offset_0;
            }
            breakpoint.score = score_0 + score_1 - gap_opening;
            breakpoint.component = component;
            return;
        }
    }
}

/// Wavefront access helper: get a wavefront from a slab by component index function.
/// Returns None if the index is WAVEFRONT_IDX_NONE or the wavefront is null.
fn get_wf_if_valid(slab: &crate::slab::WavefrontSlab, idx: usize) -> Option<&Wavefront> {
    if idx == WAVEFRONT_IDX_NONE {
        return None;
    }
    let wf = slab.get(idx);
    if wf.null {
        return None;
    }
    Some(wf)
}

/// Check all wavefront overlaps between two aligners at given scores.
///
/// `aligner_0` is at `score_0`, `aligner_1` is checked from `score_1` downward.
/// `breakpoint_forward`: true if aligner_0 is the forward aligner.
#[allow(clippy::too_many_arguments)]
pub fn bialign_overlap(
    components_0: &crate::components::WavefrontComponents,
    slab_0: &crate::slab::WavefrontSlab,
    components_1: &crate::components::WavefrontComponents,
    slab_1: &crate::slab::WavefrontSlab,
    score_0: i32,
    score_1: i32,
    pattern_length: i32,
    text_length: i32,
    gap_opening1: i32,
    gap_opening2: i32,
    has_affine: bool,
    has_affine2p: bool,
    breakpoint_forward: bool,
    breakpoint: &mut BiAlignBreakpoint,
) {
    let max_score_scope = components_1.max_score_scope;

    // Get M-wavefront at score_0
    let m_idx_0 = components_0.get_m_idx(score_0 as usize);
    let mwf_0 = match get_wf_if_valid(slab_0, m_idx_0) {
        Some(wf) => wf,
        None => return,
    };

    // Also get I/D wavefronts at score_0 (for indel-to-indel checks)
    let d1wf_0 = if has_affine {
        get_wf_if_valid(slab_0, components_0.get_d1_idx(score_0 as usize))
    } else {
        None
    };
    let i1wf_0 = if has_affine {
        get_wf_if_valid(slab_0, components_0.get_i1_idx(score_0 as usize))
    } else {
        None
    };
    let d2wf_0 = if has_affine2p {
        get_wf_if_valid(slab_0, components_0.get_d2_idx(score_0 as usize))
    } else {
        None
    };
    let i2wf_0 = if has_affine2p {
        get_wf_if_valid(slab_0, components_0.get_i2_idx(score_0 as usize))
    } else {
        None
    };

    // Iterate backwards through scores of aligner_1
    for i in 0..max_score_scope {
        let score_i = score_1 - i as i32;
        if score_i < 0 {
            break;
        }

        // Check D2/I2 breakpoints (affine2p only)
        if has_affine2p && score_0 + score_i - gap_opening2 < breakpoint.score {
            if let (Some(d2wf_0), Some(d2wf_1)) = (
                d2wf_0,
                get_wf_if_valid(slab_1, components_1.get_d2_idx(score_i as usize)),
            ) {
                breakpoint_indel2indel(
                    d2wf_0,
                    d2wf_1,
                    pattern_length,
                    text_length,
                    score_0,
                    score_i,
                    gap_opening2,
                    ComponentType::D2,
                    breakpoint_forward,
                    breakpoint,
                );
            }
            if let (Some(i2wf_0), Some(i2wf_1)) = (
                i2wf_0,
                get_wf_if_valid(slab_1, components_1.get_i2_idx(score_i as usize)),
            ) {
                breakpoint_indel2indel(
                    i2wf_0,
                    i2wf_1,
                    pattern_length,
                    text_length,
                    score_0,
                    score_i,
                    gap_opening2,
                    ComponentType::I2,
                    breakpoint_forward,
                    breakpoint,
                );
            }
        }

        // Check D1/I1 breakpoints (affine and above)
        if has_affine && score_0 + score_i - gap_opening1 < breakpoint.score {
            if let (Some(d1wf_0), Some(d1wf_1)) = (
                d1wf_0,
                get_wf_if_valid(slab_1, components_1.get_d1_idx(score_i as usize)),
            ) {
                breakpoint_indel2indel(
                    d1wf_0,
                    d1wf_1,
                    pattern_length,
                    text_length,
                    score_0,
                    score_i,
                    gap_opening1,
                    ComponentType::D1,
                    breakpoint_forward,
                    breakpoint,
                );
            }
            if let (Some(i1wf_0), Some(i1wf_1)) = (
                i1wf_0,
                get_wf_if_valid(slab_1, components_1.get_i1_idx(score_i as usize)),
            ) {
                breakpoint_indel2indel(
                    i1wf_0,
                    i1wf_1,
                    pattern_length,
                    text_length,
                    score_0,
                    score_i,
                    gap_opening1,
                    ComponentType::I1,
                    breakpoint_forward,
                    breakpoint,
                );
            }
        }

        // Check M-to-M breakpoints (all metrics)
        if score_0 + score_i >= breakpoint.score {
            continue;
        }
        if let Some(mwf_1) = get_wf_if_valid(slab_1, components_1.get_m_idx(score_i as usize)) {
            breakpoint_m2m(
                mwf_0,
                mwf_1,
                pattern_length,
                text_length,
                score_0,
                score_i,
                breakpoint_forward,
                breakpoint,
            );
        }
    }
}

/// Fallback score threshold: below this, use standard unidirectional WFA.
pub const BIALIGN_FALLBACK_MIN_SCORE: i32 = 250;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavefront::Wavefront;

    #[test]
    fn test_breakpoint_new() {
        let bp = BiAlignBreakpoint::new();
        assert!(!bp.found());
        assert_eq!(bp.score, i32::MAX);
    }

    #[test]
    fn test_breakpoint_m2m_found() {
        // Forward: k=0, offset=6, h=6. Reverse: k_inv=0, offset=5, h=5.
        // plen=10, tlen=10. h_0 + h_1 = 11 >= 10 → breakpoint
        let mut wf_fwd = Wavefront::allocate(7, false);
        wf_fwd.init(-3, 3);
        wf_fwd.set_limits(0, 0);
        wf_fwd.set_offset(0, 6);

        let mut wf_rev = Wavefront::allocate(7, false);
        wf_rev.init(-3, 3);
        wf_rev.set_limits(0, 0);
        wf_rev.set_offset(0, 5);

        let mut bp = BiAlignBreakpoint::new();
        breakpoint_m2m(&wf_fwd, &wf_rev, 10, 10, 3, 2, true, &mut bp);

        assert!(bp.found());
        assert_eq!(bp.score, 5);
        assert_eq!(bp.score_forward, 3);
        assert_eq!(bp.score_reverse, 2);
        assert_eq!(bp.k_forward, 0);
        assert_eq!(bp.offset_forward, 6);
        assert_eq!(bp.component, ComponentType::M);
    }

    #[test]
    fn test_breakpoint_m2m_not_found() {
        // h_0 + h_1 = 4 + 4 = 8 < 10 → no breakpoint
        let mut wf_fwd = Wavefront::allocate(7, false);
        wf_fwd.init(-3, 3);
        wf_fwd.set_limits(0, 0);
        wf_fwd.set_offset(0, 4);

        let mut wf_rev = Wavefront::allocate(7, false);
        wf_rev.init(-3, 3);
        wf_rev.set_limits(0, 0);
        wf_rev.set_offset(0, 4);

        let mut bp = BiAlignBreakpoint::new();
        breakpoint_m2m(&wf_fwd, &wf_rev, 10, 10, 3, 2, true, &mut bp);

        assert!(!bp.found());
    }

    #[test]
    fn test_breakpoint_m2m_different_lengths() {
        // plen=8, tlen=12 → alignment_k = 4
        // Forward: k=2, offset=9 → h=9
        // Reverse: k_inv for k=2: (12-8)-2 = 2. So reverse at k=2, offset=4 → h=4
        // h_0 + h_1 = 9 + 4 = 13 >= 12 → breakpoint
        let mut wf_fwd = Wavefront::allocate(11, false);
        wf_fwd.init(-5, 5);
        wf_fwd.set_limits(2, 2);
        wf_fwd.set_offset(2, 9);

        let mut wf_rev = Wavefront::allocate(11, false);
        wf_rev.init(-5, 5);
        wf_rev.set_limits(2, 2);
        wf_rev.set_offset(2, 4);

        let mut bp = BiAlignBreakpoint::new();
        breakpoint_m2m(&wf_fwd, &wf_rev, 8, 12, 5, 3, true, &mut bp);

        assert!(bp.found());
        assert_eq!(bp.score, 8);
    }

    #[test]
    fn test_breakpoint_m2m_no_k_overlap() {
        // Forward covers k=[0,2], reverse covers k=[-3,-1] → k_inv=[-1,1] for plen=tlen=10
        // Actually k_inv of k=[-3,-1] = (10-10)-k = [1,3]
        // Forward [0,2], reverse inv [1,3] → overlap [1,2]
        let mut wf_fwd = Wavefront::allocate(11, false);
        wf_fwd.init(-5, 5);
        wf_fwd.set_limits(4, 5); // k=[4,5]

        let mut wf_rev = Wavefront::allocate(11, false);
        wf_rev.init(-5, 5);
        wf_rev.set_limits(-5, -4); // k=[-5,-4], inv=[4,5] for plen=tlen=10
        wf_rev.set_offset(-5, 1);
        wf_rev.set_offset(-4, 1);

        wf_fwd.set_offset(4, 1);
        wf_fwd.set_offset(5, 1);

        let mut bp = BiAlignBreakpoint::new();
        breakpoint_m2m(&wf_fwd, &wf_rev, 10, 10, 3, 2, true, &mut bp);

        // h_0 + h_1 = 1 + 1 = 2 < 10 → no breakpoint
        assert!(!bp.found());
    }

    #[test]
    fn test_breakpoint_indel2indel_found() {
        // Same as m2m test but with gap_opening subtraction
        let mut wf_fwd = Wavefront::allocate(7, false);
        wf_fwd.init(-3, 3);
        wf_fwd.set_limits(0, 0);
        wf_fwd.set_offset(0, 6);

        let mut wf_rev = Wavefront::allocate(7, false);
        wf_rev.init(-3, 3);
        wf_rev.set_limits(0, 0);
        wf_rev.set_offset(0, 5);

        let mut bp = BiAlignBreakpoint::new();
        breakpoint_indel2indel(
            &wf_fwd,
            &wf_rev,
            10,
            10,
            3,
            2,
            6, // gap_opening
            ComponentType::D1,
            true,
            &mut bp,
        );

        assert!(bp.found());
        // score = 3 + 2 - 6 = -1
        assert_eq!(bp.score, -1);
        assert_eq!(bp.component, ComponentType::D1);
    }

    #[test]
    fn test_overlap_gopen_adjust() {
        assert_eq!(overlap_gopen_adjust(6, 24, false), 6);
        assert_eq!(overlap_gopen_adjust(6, 24, true), 6);
        assert_eq!(overlap_gopen_adjust(6, 3, true), 3);
    }
}
