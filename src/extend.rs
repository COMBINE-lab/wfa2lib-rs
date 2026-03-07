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

/// Scalar extend kernel for a single diagonal. Used as fallback from AVX2.
#[inline(always)]
unsafe fn extend_scalar_one(
    pattern_base: *const u8,
    text_base: *const u8,
    k: i32,
    mut offset: WfOffset,
) -> WfOffset {
    let v = (offset - k) as usize;
    let h = offset as usize;
    let mut p_ptr = pattern_base.add(v);
    let mut t_ptr = text_base.add(h);
    let mut cmp = (p_ptr as *const u64).read_unaligned()
        ^ (t_ptr as *const u64).read_unaligned();
    while cmp == 0 {
        offset += 8;
        p_ptr = p_ptr.add(8);
        t_ptr = t_ptr.add(8);
        cmp = (p_ptr as *const u64).read_unaligned()
            ^ (t_ptr as *const u64).read_unaligned();
    }
    offset += (cmp.trailing_zeros() / 8) as i32;
    offset
}

/// Extend matches for end-to-end alignment using 64-bit blockwise comparison.
///
/// On x86_64 with AVX2, uses gather-based SIMD to process 8 diagonals at once
/// for the initial 4-byte comparison, falling back to scalar for full extensions.
/// On other architectures, uses scalar 64-bit XOR blocks.
#[inline]
pub fn extend_matches_packed_end2end(sequences: &WavefrontSequences, wf: &mut Wavefront) {
    let lo = wf.lo;
    let hi = wf.hi;
    let ptrs = sequences.ptrs();
    let pattern_base = ptrs.pattern;
    let text_base = ptrs.text;

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            let count = hi - lo + 1;
            if count >= 256
                && (cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2"))
            {
                let offsets = wf.offsets_centered_mut_ptr();
                extend_end2end_avx2(pattern_base, text_base, offsets, lo, hi);
                return;
            }
        }

        // Scalar fallback using pre-centered pointer
        let centered = wf.offsets_centered_mut_ptr();
        let mut off_ptr = centered.offset(lo as isize);
        let off_end = centered.offset(hi as isize).add(1);
        let mut k = lo;

        while off_ptr < off_end {
            let offset = *off_ptr;
            if offset >= 0 {
                *off_ptr = extend_scalar_one(pattern_base, text_base, k, offset);
            }
            off_ptr = off_ptr.add(1);
            k += 1;
        }
    }
}

/// AVX2 extend kernel for end-to-end alignment.
///
/// Processes 8 diagonals at a time using gather instructions to load 4 bytes
/// from each diagonal's pattern/text position simultaneously. For diagonals
/// where all 4 bytes match, falls back to scalar to continue the extension.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extend_end2end_avx2(
    pattern_base: *const u8,
    text_base: *const u8,
    offsets: *mut WfOffset,
    lo: i32,
    hi: i32,
) {
    use std::arch::x86_64::*;

    let count = hi - lo + 1;
    if count <= 0 {
        return;
    }

    let vector_null = _mm256_set1_epi32(-1);
    let eights = _mm256_set1_epi32(8);
    let offsets_null = _mm256_set1_epi32(OFFSET_NULL);
    // Byte-reverse within each 32-bit lane (for CLZ-based mismatch counting)
    let vec_shuffle = _mm256_set_epi8(
        28, 29, 30, 31, 24, 25, 26, 27,
        20, 21, 22, 23, 16, 17, 18, 19,
        12, 13, 14, 15,  8,  9, 10, 11,
         4,  5,  6,  7,  0,  1,  2,  3,
    );

    let mut k = lo;

    // Loop peeling: handle num_diagonals % 8 with scalar
    let peel = count % 8;
    let peel_end = lo + peel;
    while k < peel_end {
        let offset = *offsets.offset(k as isize);
        if offset >= 0 {
            *offsets.offset(k as isize) =
                extend_scalar_one(pattern_base, text_base, k, offset);
        }
        k += 1;
    }
    if count < 8 {
        return;
    }

    let mut ks = _mm256_add_epi32(
        _mm256_set1_epi32(k),
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    );

    while k <= hi {
        let offsets_v = _mm256_loadu_si256(offsets.offset(k as isize) as *const __m256i);
        let h_vector = offsets_v;
        let v_vector = _mm256_sub_epi32(offsets_v, ks);

        // Mask NULL offsets → gather reads index 0 (safe, sentinel-padded)
        let null_mask = _mm256_cmpgt_epi32(offsets_v, vector_null);
        let v_masked = _mm256_and_si256(null_mask, v_vector);
        let h_masked = _mm256_and_si256(null_mask, h_vector);

        // Gather 4 bytes from pattern[v] and text[h] for 8 diagonals
        let pattern_v =
            _mm256_i32gather_epi32(pattern_base as *const i32, v_masked, 1);
        let text_v =
            _mm256_i32gather_epi32(text_base as *const i32, h_masked, 1);

        // Which diagonals had all 4 bytes matching?
        let eq_mask = _mm256_cmpeq_epi32(text_v, pattern_v);
        let mask = _mm256_movemask_epi8(eq_mask);

        // XOR → byte-reverse within lanes → CLZ → divide by 8 = equal chars (0–4)
        let xor_result = _mm256_xor_si256(pattern_v, text_v);
        let xor_shuffled = _mm256_shuffle_epi8(xor_result, vec_shuffle);
        let clz = avx2_lzcnt_epi32(xor_shuffled);

        let equal_chars = _mm256_srli_epi32(clz, 3);
        let offsets_updated = _mm256_add_epi32(offsets_v, equal_chars);
        let offsets_final =
            _mm256_blendv_epi8(offsets_null, offsets_updated, null_mask);
        ks = _mm256_add_epi32(ks, eights);

        _mm256_storeu_si256(
            offsets.offset(k as isize) as *mut __m256i,
            offsets_final,
        );

        // For diagonals where all 4 bytes matched, continue with scalar
        if mask != 0 {
            let mut remaining = mask as u32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros();
                let curr_k = k + (tz / 4) as i32;
                let offset = *offsets.offset(curr_k as isize);
                if offset >= 0 {
                    *offsets.offset(curr_k as isize) =
                        extend_scalar_one(pattern_base, text_base, curr_k, offset);
                } else {
                    *offsets.offset(curr_k as isize) = OFFSET_NULL;
                }
                remaining &= 0xfffffff0_u32.wrapping_shl(tz);
            }
        }

        k += 8;
    }
}

/// Emulated count-leading-zeros for 8x i32 lanes (AVX2 lacks native LZCNT).
/// Uses float conversion trick: convert to float, extract exponent, undo bias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_lzcnt_epi32(v: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Keep only the 8 MSB of each lane
    let v = _mm256_andnot_si256(_mm256_srli_epi32(v, 8), v);
    // Convert integer to float (exponent encodes magnitude)
    let v = _mm256_castps_si256(_mm256_cvtepi32_ps(v));
    // Shift down the exponent field
    let v = _mm256_srli_epi32(v, 23);
    // Undo IEEE754 bias (127 + 31 = 158)
    let v = _mm256_subs_epu16(_mm256_set1_epi32(158), v);
    // Clamp at 32 (for zero input)
    _mm256_min_epi16(v, _mm256_set1_epi32(32))
}

/// Extend matches and return the maximum antidiagonal reached.
#[inline]
pub fn extend_matches_packed_end2end_max(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
) -> WfOffset {
    let lo = wf.lo;
    let hi = wf.hi;
    let ptrs = sequences.ptrs();
    let pattern_base = ptrs.pattern;
    let text_base = ptrs.text;

    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            let count = hi - lo + 1;
            if count >= 256
                && (cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2"))
            {
                let offsets = wf.offsets_centered_mut_ptr();
                return extend_end2end_max_avx2(
                    pattern_base, text_base, offsets, lo, hi,
                );
            }
        }

        // Scalar fallback using pre-centered pointer
        let centered = wf.offsets_centered_mut_ptr();
        let mut off_ptr = centered.offset(lo as isize);
        let off_end = centered.offset(hi as isize).add(1);
        let mut k = lo;
        let mut max_antidiag: WfOffset = 0;

        while off_ptr < off_end {
            let offset = *off_ptr;
            if offset >= 0 {
                let extended = extend_scalar_one(pattern_base, text_base, k, offset);
                *off_ptr = extended;
                let antidiag = 2 * extended - k;
                if antidiag > max_antidiag {
                    max_antidiag = antidiag;
                }
            }
            off_ptr = off_ptr.add(1);
            k += 1;
        }

        max_antidiag
    }
}

/// AVX2 extend kernel with max antidiagonal tracking (for BiWFA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extend_end2end_max_avx2(
    pattern_base: *const u8,
    text_base: *const u8,
    offsets: *mut WfOffset,
    lo: i32,
    hi: i32,
) -> WfOffset {
    use std::arch::x86_64::*;

    let count = hi - lo + 1;
    if count <= 0 {
        return 0;
    }

    let vector_null = _mm256_set1_epi32(-1);
    let eights = _mm256_set1_epi32(8);
    let offsets_null = _mm256_set1_epi32(OFFSET_NULL);
    let vec_shuffle = _mm256_set_epi8(
        28, 29, 30, 31, 24, 25, 26, 27,
        20, 21, 22, 23, 16, 17, 18, 19,
        12, 13, 14, 15,  8,  9, 10, 11,
         4,  5,  6,  7,  0,  1,  2,  3,
    );

    let mut k = lo;
    let mut max_antidiag: WfOffset = 0;

    // Loop peeling
    let peel = count % 8;
    let peel_end = lo + peel;
    while k < peel_end {
        let offset = *offsets.offset(k as isize);
        if offset >= 0 {
            let extended = extend_scalar_one(pattern_base, text_base, k, offset);
            *offsets.offset(k as isize) = extended;
            let antidiag = 2 * extended - k;
            if antidiag > max_antidiag {
                max_antidiag = antidiag;
            }
        }
        k += 1;
    }
    if count < 8 {
        return max_antidiag;
    }

    let mut max_antidiag_v = _mm256_set1_epi32(max_antidiag);
    let mut ks = _mm256_add_epi32(
        _mm256_set1_epi32(k),
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    );

    while k <= hi {
        let offsets_v = _mm256_loadu_si256(offsets.offset(k as isize) as *const __m256i);
        let h_vector = offsets_v;
        let v_vector = _mm256_sub_epi32(offsets_v, ks);

        let null_mask = _mm256_cmpgt_epi32(offsets_v, vector_null);
        let v_masked = _mm256_and_si256(null_mask, v_vector);
        let h_masked = _mm256_and_si256(null_mask, h_vector);

        let pattern_v =
            _mm256_i32gather_epi32(pattern_base as *const i32, v_masked, 1);
        let text_v =
            _mm256_i32gather_epi32(text_base as *const i32, h_masked, 1);

        let eq_mask = _mm256_cmpeq_epi32(text_v, pattern_v);
        let mask = _mm256_movemask_epi8(eq_mask);

        let xor_result = _mm256_xor_si256(pattern_v, text_v);
        let xor_shuffled = _mm256_shuffle_epi8(xor_result, vec_shuffle);
        let clz = avx2_lzcnt_epi32(xor_shuffled);

        let equal_chars = _mm256_srli_epi32(clz, 3);
        let offsets_updated = _mm256_add_epi32(offsets_v, equal_chars);
        let offsets_final =
            _mm256_blendv_epi8(offsets_null, offsets_updated, null_mask);

        _mm256_storeu_si256(
            offsets.offset(k as isize) as *mut __m256i,
            offsets_final,
        );

        // Track max antidiagonal: 2*offset - k (only for non-null)
        let offset_max = _mm256_and_si256(null_mask, offsets_final);
        let offset_max = _mm256_slli_epi32(offset_max, 1);
        let offset_max = _mm256_sub_epi32(offset_max, ks);
        max_antidiag_v = _mm256_max_epi32(max_antidiag_v, offset_max);
        ks = _mm256_add_epi32(ks, eights);

        if mask != 0 {
            // Extract current max from vector for scalar fallback
            let buf: [i32; 8] = unsafe { std::mem::transmute(max_antidiag_v) };
            for &v in &buf {
                if v > max_antidiag {
                    max_antidiag = v;
                }
            }

            let mut remaining = mask as u32;
            while remaining != 0 {
                let tz = remaining.trailing_zeros();
                let curr_k = k + (tz / 4) as i32;
                let offset = *offsets.offset(curr_k as isize);
                if offset >= 0 {
                    let extended =
                        extend_scalar_one(pattern_base, text_base, curr_k, offset);
                    *offsets.offset(curr_k as isize) = extended;
                    let antidiag = 2 * extended - curr_k;
                    if antidiag > max_antidiag {
                        max_antidiag = antidiag;
                    }
                } else {
                    *offsets.offset(curr_k as isize) = OFFSET_NULL;
                }
                remaining &= 0xfffffff0_u32.wrapping_shl(tz);
            }
            max_antidiag_v = _mm256_set1_epi32(max_antidiag);
        }

        k += 8;
    }

    // Final reduction of max antidiagonal vector
    let buf: [i32; 8] = unsafe { std::mem::transmute(max_antidiag_v) };
    for &v in &buf {
        if v > max_antidiag {
            max_antidiag = v;
        }
    }

    max_antidiag
}

/// Extend matches for ends-free alignment, checking termination at each diagonal.
///
/// Returns `Some((k, offset))` if a valid termination point is found, `None` otherwise.
#[inline]
pub fn extend_matches_packed_endsfree(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
    pattern_end_free: i32,
    text_end_free: i32,
) -> Option<(i32, WfOffset)> {
    let lo = wf.lo;
    let hi = wf.hi;
    let ptrs = sequences.ptrs();
    let pattern_base = ptrs.pattern;
    let text_base = ptrs.text;
    let pattern_length = sequences.pattern_length;
    let text_length = sequences.text_length;

    // SAFETY: Same as extend_matches_packed_end2end.
    unsafe {
        let centered = wf.offsets_centered_mut_ptr();
        let mut off_ptr = centered.offset(lo as isize);
        let off_end = centered.offset(hi as isize).add(1);
        let mut k = lo;

        while off_ptr < off_end {
            let mut offset = *off_ptr;
            if offset != OFFSET_NULL {
                let v = (offset - k) as usize;
                let h = offset as usize;
                let mut p_ptr = pattern_base.add(v);
                let mut t_ptr = text_base.add(h);
                let mut cmp = (p_ptr as *const u64).read_unaligned()
                    ^ (t_ptr as *const u64).read_unaligned();
                while cmp == 0 {
                    offset += 8;
                    p_ptr = p_ptr.add(8);
                    t_ptr = t_ptr.add(8);
                    cmp = (p_ptr as *const u64).read_unaligned()
                        ^ (t_ptr as *const u64).read_unaligned();
                }
                offset += (cmp.trailing_zeros() / 8) as i32;
                *off_ptr = offset;

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
            off_ptr = off_ptr.add(1);
            k += 1;
        }
    }

    None
}

// --- Custom extend functions for Lambda mode (bounds-checked, no sentinels) ---

/// Extend matches for Lambda mode end-to-end alignment (bounds-checked).
#[inline]
pub fn extend_matches_custom_end2end(sequences: &WavefrontSequences, wf: &mut Wavefront) {
    let lo = wf.lo;
    let hi = wf.hi;
    let centered = unsafe { wf.offsets_centered_mut_ptr() };

    for k in lo..=hi {
        let mut offset = unsafe { *centered.offset(k as isize) };
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

        unsafe { *centered.offset(k as isize) = offset; }
    }
}

/// Extend matches for Lambda mode and return the maximum antidiagonal reached.
#[inline]
pub fn extend_matches_custom_end2end_max(
    sequences: &WavefrontSequences,
    wf: &mut Wavefront,
) -> WfOffset {
    let lo = wf.lo;
    let hi = wf.hi;
    let centered = unsafe { wf.offsets_centered_mut_ptr() };
    let mut max_antidiag: WfOffset = 0;

    for k in lo..=hi {
        let mut offset = unsafe { *centered.offset(k as isize) };
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

        unsafe { *centered.offset(k as isize) = offset; }

        let antidiag = 2 * offset - k;
        if antidiag > max_antidiag {
            max_antidiag = antidiag;
        }
    }

    max_antidiag
}

/// Extend matches for Lambda mode ends-free alignment, checking termination.
#[inline]
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
    let centered = unsafe { wf.offsets_centered_mut_ptr() };

    for k in lo..=hi {
        let mut offset = unsafe { *centered.offset(k as isize) };
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

        unsafe { *centered.offset(k as isize) = offset; }

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
