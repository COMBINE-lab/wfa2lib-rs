//! Wavefront extension: extends exact matches along diagonals.
//!
//! The extend kernel compares pattern and text characters starting from each
//! wavefront offset, advancing the offset while characters match. Sentinel
//! characters at the end of each sequence guarantee termination without
//! explicit bounds checks.

use crate::offset::{OFFSET_NULL, WfOffset};
use crate::sequences::WavefrontSequences;
use crate::termination;
use crate::wavefront::Wavefront;

/// Extend matches for end-to-end alignment (character-by-character).
///
/// For each diagonal k in [lo, hi], extends the offset by comparing
/// pattern[v..] with text[h..] where v = offset - k, h = offset.
/// Sentinels in the padded sequence buffer guarantee safe termination.
#[inline(never)]
pub fn extend_matches_packed_end2end(sequences: &WavefrontSequences, wf: &mut Wavefront) {
    let lo = wf.lo;
    let hi = wf.hi;
    let pattern = sequences.pattern_ptr();
    let text = sequences.text_ptr();
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = (offset - k) as usize;
        let mut h = offset as usize;

        // Character-by-character comparison.
        // Safe because sentinel chars differ between pattern ('!') and text ('?').
        while pattern[v] == text[h] {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;
    }
}

/// Extend matches and return the maximum antidiagonal reached.
#[inline(never)]
pub fn extend_matches_packed_end2end_max(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
) -> WfOffset {
    let lo = wf.lo;
    let hi = wf.hi;
    let pattern = sequences.pattern_ptr();
    let text = sequences.text_ptr();
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();
    let mut max_antidiag: WfOffset = 0;

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = (offset - k) as usize;
        let mut h = offset as usize;

        while pattern[v] == text[h] {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;

        // antidiagonal = 2*offset - k
        let antidiag = 2 * offset - k;
        if antidiag > max_antidiag {
            max_antidiag = antidiag;
        }
    }

    max_antidiag
}

/// Extend matches for ends-free alignment, checking termination at each diagonal.
///
/// Returns `Some((k, offset))` if a valid termination point is found, `None` otherwise.
#[inline(never)]
pub fn extend_matches_packed_endsfree(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
    pattern_end_free: i32,
    text_end_free: i32,
) -> Option<(i32, WfOffset)> {
    let lo = wf.lo;
    let hi = wf.hi;
    let pattern = sequences.pattern_ptr();
    let text = sequences.text_ptr();
    let pattern_length = sequences.pattern_length;
    let text_length = sequences.text_length;
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = (offset - k) as usize;
        let mut h = offset as usize;

        // Character-by-character comparison (same sentinel-based loop)
        while pattern[v] == text[h] {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;

        // Check ends-free termination at this diagonal
        if termination::termination_endsfree(
            pattern_length,
            text_length,
            pattern_end_free,
            text_end_free,
            k,
            offset,
        ) {
            return Some((k, offset));
        }
    }

    None
}

// --- Custom extend functions for Lambda mode (bounds-checked, no sentinels) ---

/// Extend matches for Lambda mode end-to-end alignment (bounds-checked).
#[inline(never)]
pub fn extend_matches_custom_end2end(sequences: &WavefrontSequences, wf: &mut Wavefront) {
    let lo = wf.lo;
    let hi = wf.hi;
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = offset - k;
        let mut h = offset;

        while sequences.cmp_lambda(v, h) {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;
    }
}

/// Extend matches for Lambda mode and return the maximum antidiagonal reached.
#[inline(never)]
pub fn extend_matches_custom_end2end_max(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
) -> WfOffset {
    let lo = wf.lo;
    let hi = wf.hi;
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();
    let mut max_antidiag: WfOffset = 0;

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = offset - k;
        let mut h = offset;

        while sequences.cmp_lambda(v, h) {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;

        let antidiag = 2 * offset - k;
        if antidiag > max_antidiag {
            max_antidiag = antidiag;
        }
    }

    max_antidiag
}

/// Extend matches for Lambda mode ends-free alignment, checking termination.
#[inline(never)]
pub fn extend_matches_custom_endsfree(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
    pattern_end_free: i32,
    text_end_free: i32,
) -> Option<(i32, WfOffset)> {
    let lo = wf.lo;
    let hi = wf.hi;
    let pattern_length = sequences.pattern_length;
    let text_length = sequences.text_length;
    let base_k = wf.base_k();
    let offsets = wf.offsets_slice_mut();

    for k in lo..=hi {
        let idx = (k - base_k) as usize;
        let mut offset = offsets[idx];
        if offset == OFFSET_NULL {
            continue;
        }

        let mut v = offset - k;
        let mut h = offset;

        while sequences.cmp_lambda(v, h) {
            v += 1;
            h += 1;
            offset += 1;
        }

        offsets[idx] = offset;

        if termination::termination_endsfree(
            pattern_length,
            text_length,
            pattern_end_free,
            text_end_free,
            k,
            offset,
        ) {
            return Some((k, offset));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extend_identical() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACGT", false);

        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(0, 0);
        wf.set_offset(0, 0);

        extend_matches_packed_end2end(&seq, &mut wf);

        // Should extend all 4 characters
        assert_eq!(wf.get_offset(0), 4);
    }

    #[test]
    fn test_extend_no_match() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"AAAA", b"TTTT", false);

        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(0, 0);
        wf.set_offset(0, 0);

        extend_matches_packed_end2end(&seq, &mut wf);

        // No characters match at position 0
        assert_eq!(wf.get_offset(0), 0);
    }

    #[test]
    fn test_extend_partial_match() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACTT", false);

        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(0, 0);
        wf.set_offset(0, 0);

        extend_matches_packed_end2end(&seq, &mut wf);

        // Matches: A, C then stops (G != T)
        assert_eq!(wf.get_offset(0), 2);
    }

    #[test]
    fn test_extend_null_offset_skipped() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACGT", false);

        let mut wf = Wavefront::allocate(7, false);
        wf.init(-3, 3);
        wf.set_limits(-1, 1);
        wf.set_offset(-1, OFFSET_NULL);
        wf.set_offset(0, 0);
        wf.set_offset(1, OFFSET_NULL);

        extend_matches_packed_end2end(&seq, &mut wf);

        // Only k=0 should be extended
        assert_eq!(wf.get_offset(0), 4);
        assert_eq!(wf.get_offset(-1), OFFSET_NULL);
        assert_eq!(wf.get_offset(1), OFFSET_NULL);
    }

    // --- Ends-free extend tests ---

    #[test]
    fn test_endsfree_terminates_on_text_end() {
        // pattern="ACGT" (4), text="ACGTXXXX" (8)
        // With text_end_free=4, once pattern is fully consumed (v=4),
        // remaining text (4) <= text_end_free → terminate
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACGTXXXX", false);

        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(0, 0);
        wf.set_offset(0, 0);

        // k=0, after extending: offset=4, v=4=pattern_length
        // text_left = 8 - 4 = 4 <= text_end_free=4 → terminate
        let result = extend_matches_packed_endsfree(&seq, &mut wf, 0, 4);
        assert!(result.is_some());
        let (k, offset) = result.unwrap();
        assert_eq!(k, 0);
        assert_eq!(offset, 4);
    }

    #[test]
    fn test_endsfree_no_termination() {
        // pattern="ACGTACGT" (8), text="ACGTXXXX" (8)
        // With pattern_end_free=0, text_end_free=0 (same as end2end)
        // After extending: offset=4 (matches ACGT then stops)
        // h=4 < 8, v=4 < 8 → no termination
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGTACGT", b"ACGTXXXX", false);

        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(0, 0);
        wf.set_offset(0, 0);

        let result = extend_matches_packed_endsfree(&seq, &mut wf, 0, 0);
        assert!(result.is_none());
        assert_eq!(wf.get_offset(0), 4);
    }
}
