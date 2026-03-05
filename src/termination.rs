//! Wavefront termination detection.
//!
//! Checks whether the alignment has reached its target position
//! (end of both sequences for end-to-end, or boundary conditions for ends-free).

use crate::offset::{WfOffset, dpmatrix_diagonal, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Check if end-to-end alignment has completed.
///
/// Returns true if the M-wavefront's offset at the target diagonal
/// (text_length - pattern_length) has reached or exceeded text_length.
#[inline(always)]
pub fn termination_end2end(pattern_length: i32, text_length: i32, wf: &Wavefront) -> bool {
    let alignment_k = dpmatrix_diagonal(text_length, pattern_length);
    let alignment_offset = text_length; // dpmatrix_offset

    if wf.lo > alignment_k || alignment_k > wf.hi {
        return false;
    }

    wf.get_offset(alignment_k) >= alignment_offset
}

/// Check if ends-free alignment has reached a valid termination point.
///
/// Called for a specific diagonal k and offset during extension. Returns true if:
/// - Text is consumed (h >= text_length) and remaining pattern <= pattern_end_free
/// - Pattern is consumed (v >= pattern_length) and remaining text <= text_end_free
#[inline(always)]
pub fn termination_endsfree(
    pattern_length: i32,
    text_length: i32,
    pattern_end_free: i32,
    text_end_free: i32,
    k: i32,
    offset: WfOffset,
) -> bool {
    let h = wavefront_h(k, offset);
    let v = wavefront_v(k, offset);

    // Check if text is fully consumed
    if h >= text_length {
        let pattern_left = pattern_length - v;
        if pattern_left <= pattern_end_free {
            return true;
        }
    }

    // Check if pattern is fully consumed
    if v >= pattern_length {
        let text_left = text_length - h;
        if text_left <= text_end_free {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_reached() {
        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(0, 0);
        wf.set_offset(0, 2); // offset=2 < text_length=4

        assert!(!termination_end2end(4, 4, &wf));
    }

    #[test]
    fn test_reached_equal_lengths() {
        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(0, 0);
        wf.set_offset(0, 4); // offset=4 >= text_length=4

        assert!(termination_end2end(4, 4, &wf));
    }

    #[test]
    fn test_reached_different_lengths() {
        // pattern=3, text=5 → alignment_k = 5-3 = 2, alignment_offset = 5
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(-1, 3);
        wf.set_offset(2, 5);

        assert!(termination_end2end(3, 5, &wf));
    }

    #[test]
    fn test_diagonal_out_of_range() {
        // alignment_k = 0 but wf only covers [1, 3]
        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(1, 3);
        wf.set_offset(1, 100);

        assert!(!termination_end2end(4, 4, &wf));
    }

    // --- Ends-free termination tests ---

    #[test]
    fn test_endsfree_text_consumed() {
        // h=10 >= text_length=10, pattern_left = 8 - 0 = 8 <= pattern_end_free=10
        // k=10, offset=10 → h=10, v=0
        assert!(termination_endsfree(8, 10, 10, 0, 10, 10));
    }

    #[test]
    fn test_endsfree_pattern_consumed() {
        // k=-8, offset=0 → h=0, v=8 >= pattern_length=8
        // text_left = 10 - 0 = 10 <= text_end_free=10
        assert!(termination_endsfree(8, 10, 0, 10, -8, 0));
    }

    #[test]
    fn test_endsfree_not_reached() {
        // k=0, offset=5 → h=5, v=5
        // h < text_length=10, v < pattern_length=10 → neither consumed
        assert!(!termination_endsfree(10, 10, 5, 5, 0, 5));
    }

    #[test]
    fn test_endsfree_text_consumed_but_pattern_left_too_large() {
        // k=5, offset=10 → h=10 >= text_length=10, v=5
        // pattern_left = 10 - 5 = 5, but pattern_end_free=3 → not enough
        assert!(!termination_endsfree(10, 10, 3, 0, 5, 10));
    }

    #[test]
    fn test_endsfree_exact_boundary() {
        // Both sequences exactly consumed: k=0, offset=10 → h=10, v=10
        // pattern_left = 0 <= 0, text_left = 0 <= 0
        assert!(termination_endsfree(10, 10, 0, 0, 0, 10));
    }
}
