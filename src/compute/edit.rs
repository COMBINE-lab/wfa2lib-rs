//! Wavefront compute kernels for edit distance and indel distance.

use crate::offset::{OFFSET_NULL, WfOffset, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Edit distance compute kernel (NEON-vectorized on aarch64).
///
/// For each diagonal k in [lo, hi]:
///   ins   = prev[k-1]     (insertion: advance text)
///   del   = prev[k+1]     (deletion: advance pattern)
///   misms = prev[k]       (mismatch: advance both)
///   curr[k] = max(del, max(ins, misms) + 1)
///
/// Out-of-bounds offsets are set to OFFSET_NULL.
#[inline]
pub fn compute_edit_idm(
    pattern_length: i32,
    text_length: i32,
    wf_prev: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    // SAFETY: Bounds guaranteed by caller (lo/hi clamped, init_ends called).
    unsafe {
        let prev = wf_prev.offsets_centered_ptr();
        let curr = wf_curr.offsets_centered_mut_ptr();

        let count = hi - lo + 1;
        let mut k = lo;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            if count >= 4 {
                {
                    let v_one = vdupq_n_s32(1);
                    let v_four = vdupq_n_s32(4);
                    let v_null = vdupq_n_s32(OFFSET_NULL);
                    let v_tlen = vdupq_n_u32(text_length as u32);
                    let v_plen = vdupq_n_u32(pattern_length as u32);
                    let k_base: [i32; 4] = [0, 1, 2, 3];
                    let mut v_k = vaddq_s32(vdupq_n_s32(lo), vld1q_s32(k_base.as_ptr()));
                    let neon_end = lo + (count & !3);

                    while k < neon_end {
                        let v_ins = vld1q_s32(prev.offset(k as isize - 1));
                        let v_del = vld1q_s32(prev.offset(k as isize + 1));
                        let v_misms = vld1q_s32(prev.offset(k as isize));
                        let v_max = vmaxq_s32(v_del, vaddq_s32(vmaxq_s32(v_ins, v_misms), v_one));

                        let v_h = vreinterpretq_u32_s32(v_max);
                        let v_v = vreinterpretq_u32_s32(vsubq_s32(v_max, v_k));
                        let oob = vorrq_u32(vcgtq_u32(v_h, v_tlen), vcgtq_u32(v_v, v_plen));
                        let v_result = vbslq_s32(oob, v_null, v_max);

                        vst1q_s32(curr.offset(k as isize), v_result);
                        k += 4;
                        v_k = vaddq_s32(v_k, v_four);
                    }
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            if count >= 8 && (cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2")) {
                unsafe {
                    let v_one = _mm256_set1_epi32(1);
                    let v_eight = _mm256_set1_epi32(8);
                    let v_null = _mm256_set1_epi32(OFFSET_NULL);
                    let v_sign = _mm256_set1_epi32(i32::MIN);
                    let v_tlen_biased = _mm256_xor_si256(_mm256_set1_epi32(text_length), v_sign);
                    let v_plen_biased = _mm256_xor_si256(_mm256_set1_epi32(pattern_length), v_sign);
                    let mut v_k = _mm256_add_epi32(
                        _mm256_set1_epi32(lo),
                        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
                    );
                    let avx_end = lo + (count & !7);

                    while k < avx_end {
                        let v_ins = _mm256_loadu_si256(prev.offset(k as isize - 1) as *const __m256i);
                        let v_del = _mm256_loadu_si256(prev.offset(k as isize + 1) as *const __m256i);
                        let v_misms = _mm256_loadu_si256(prev.offset(k as isize) as *const __m256i);
                        let v_max = _mm256_max_epi32(v_del, _mm256_add_epi32(_mm256_max_epi32(v_ins, v_misms), v_one));

                        let v_h_biased = _mm256_xor_si256(v_max, v_sign);
                        let v_v_biased = _mm256_xor_si256(_mm256_sub_epi32(v_max, v_k), v_sign);
                        let oob = _mm256_or_si256(
                            _mm256_cmpgt_epi32(v_h_biased, v_tlen_biased),
                            _mm256_cmpgt_epi32(v_v_biased, v_plen_biased),
                        );
                        let v_result = _mm256_blendv_epi8(v_max, v_null, oob);

                        _mm256_storeu_si256(curr.offset(k as isize) as *mut __m256i, v_result);
                        k += 8;
                        v_k = _mm256_add_epi32(v_k, v_eight);
                    }
                }
            }
        }

        while k <= hi {
            let ins = *prev.offset(k as isize - 1);
            let del = *prev.offset(k as isize + 1);
            let misms = *prev.offset(k as isize);
            let mut max = del.max(ins.max(misms) + 1);

            let h = wavefront_h(k, max) as u32;
            let v = wavefront_v(k, max) as u32;
            if h > text_length as u32 || v > pattern_length as u32 {
                max = OFFSET_NULL;
            }

            *curr.offset(k as isize) = max;
            k += 1;
        }
    }
}

/// Indel distance compute kernel (NEON-vectorized on aarch64).
///
/// For each diagonal k in [lo, hi]:
///   ins   = prev[k-1] + 1  (insertion)
///   del   = prev[k+1]      (deletion)
///   curr[k] = max(del, ins)
#[inline]
pub fn compute_indel_idm(
    pattern_length: i32,
    text_length: i32,
    wf_prev: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    // SAFETY: Bounds guaranteed by caller (lo/hi clamped, init_ends called).
    unsafe {
        let prev = wf_prev.offsets_centered_ptr();
        let curr = wf_curr.offsets_centered_mut_ptr();

        let count = hi - lo + 1;
        let mut k = lo;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            if count >= 4 {
                {
                    let v_one = vdupq_n_s32(1);
                    let v_four = vdupq_n_s32(4);
                    let v_null = vdupq_n_s32(OFFSET_NULL);
                    let v_tlen = vdupq_n_u32(text_length as u32);
                    let v_plen = vdupq_n_u32(pattern_length as u32);
                    let k_base: [i32; 4] = [0, 1, 2, 3];
                    let mut v_k = vaddq_s32(vdupq_n_s32(lo), vld1q_s32(k_base.as_ptr()));
                    let neon_end = lo + (count & !3);

                    while k < neon_end {
                        let v_ins = vaddq_s32(vld1q_s32(prev.offset(k as isize - 1)), v_one);
                        let v_del = vld1q_s32(prev.offset(k as isize + 1));
                        let v_max = vmaxq_s32(v_del, v_ins);

                        let v_h = vreinterpretq_u32_s32(v_max);
                        let v_v = vreinterpretq_u32_s32(vsubq_s32(v_max, v_k));
                        let oob = vorrq_u32(vcgtq_u32(v_h, v_tlen), vcgtq_u32(v_v, v_plen));
                        let v_result = vbslq_s32(oob, v_null, v_max);

                        vst1q_s32(curr.offset(k as isize), v_result);
                        k += 4;
                        v_k = vaddq_s32(v_k, v_four);
                    }
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            if count >= 8 && (cfg!(target_feature = "avx2") || is_x86_feature_detected!("avx2")) {
                unsafe {
                    let v_one = _mm256_set1_epi32(1);
                    let v_eight = _mm256_set1_epi32(8);
                    let v_null = _mm256_set1_epi32(OFFSET_NULL);
                    let v_sign = _mm256_set1_epi32(i32::MIN);
                    let v_tlen_biased = _mm256_xor_si256(_mm256_set1_epi32(text_length), v_sign);
                    let v_plen_biased = _mm256_xor_si256(_mm256_set1_epi32(pattern_length), v_sign);
                    let mut v_k = _mm256_add_epi32(
                        _mm256_set1_epi32(lo),
                        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
                    );
                    let avx_end = lo + (count & !7);

                    while k < avx_end {
                        let v_ins = _mm256_add_epi32(
                            _mm256_loadu_si256(prev.offset(k as isize - 1) as *const __m256i),
                            v_one,
                        );
                        let v_del = _mm256_loadu_si256(prev.offset(k as isize + 1) as *const __m256i);
                        let v_max = _mm256_max_epi32(v_del, v_ins);

                        let v_h_biased = _mm256_xor_si256(v_max, v_sign);
                        let v_v_biased = _mm256_xor_si256(_mm256_sub_epi32(v_max, v_k), v_sign);
                        let oob = _mm256_or_si256(
                            _mm256_cmpgt_epi32(v_h_biased, v_tlen_biased),
                            _mm256_cmpgt_epi32(v_v_biased, v_plen_biased),
                        );
                        let v_result = _mm256_blendv_epi8(v_max, v_null, oob);

                        _mm256_storeu_si256(curr.offset(k as isize) as *mut __m256i, v_result);
                        k += 8;
                        v_k = _mm256_add_epi32(v_k, v_eight);
                    }
                }
            }
        }

        while k <= hi {
            let ins = *prev.offset(k as isize - 1) + 1;
            let del = *prev.offset(k as isize + 1);
            let mut max: WfOffset = del.max(ins);

            let h = wavefront_h(k, max) as u32;
            let v = wavefront_v(k, max) as u32;
            if h > text_length as u32 || v > pattern_length as u32 {
                max = OFFSET_NULL;
            }

            *curr.offset(k as isize) = max;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_kernel_basic() {
        // Simulate score 0 → score 1 for edit distance
        // Score 0: k=0, offset=0 (after extend, no matches)
        let mut wf_prev = Wavefront::allocate(7, false); // covers [-3, 3]
        wf_prev.init(-3, 3);
        wf_prev.set_limits(0, 0);
        wf_prev.set_offset(0, 0);

        // Set padding for the compute kernel
        wf_prev.set_offset(-2, OFFSET_NULL);
        wf_prev.set_offset(-1, OFFSET_NULL);
        wf_prev.set_offset(1, OFFSET_NULL);
        wf_prev.set_offset(2, OFFSET_NULL);

        let mut wf_curr = Wavefront::allocate(7, false);
        wf_curr.init(-3, 3);
        wf_curr.set_limits(-1, 1);

        compute_edit_idm(4, 4, &wf_prev, &mut wf_curr, -1, 1);

        // k=-1: ins=prev[-2]=NULL, del=prev[0]=0, misms=prev[-1]=NULL
        //        max = max(0, max(NULL, NULL)+1) = max(0, NULL+1)
        //        NULL+1 overflows but stays very negative, so max = 0
        // Actually OFFSET_NULL = i32::MIN/2, so NULL+1 is still very negative
        // max(0, very_negative) = 0
        // h = wavefront_h(-1, 0) = 0, v = wavefront_v(-1, 0) = 1
        // h=0 <= 4, v=1 <= 4: valid
        assert_eq!(wf_curr.get_offset(-1), 0);

        // k=0: ins=prev[-1]=NULL, del=prev[1]=NULL, misms=prev[0]=0
        //       max = max(NULL, max(NULL, 0)+1) = max(NULL, 1) = 1
        assert_eq!(wf_curr.get_offset(0), 1);

        // k=1: ins=prev[0]=0, del=prev[2]=NULL, misms=prev[1]=NULL
        //       max = max(NULL, max(0, NULL)+1) = max(NULL, 1) = 1
        assert_eq!(wf_curr.get_offset(1), 1);
    }

    #[test]
    fn test_indel_kernel_basic() {
        let mut wf_prev = Wavefront::allocate(7, false);
        wf_prev.init(-3, 3);
        wf_prev.set_limits(0, 0);
        wf_prev.set_offset(0, 0);

        wf_prev.set_offset(-2, OFFSET_NULL);
        wf_prev.set_offset(-1, OFFSET_NULL);
        wf_prev.set_offset(1, OFFSET_NULL);
        wf_prev.set_offset(2, OFFSET_NULL);

        let mut wf_curr = Wavefront::allocate(7, false);
        wf_curr.init(-3, 3);
        wf_curr.set_limits(-1, 1);

        compute_indel_idm(4, 4, &wf_prev, &mut wf_curr, -1, 1);

        // k=-1: ins=prev[-2]+1=NULL+1≈NULL, del=prev[0]=0
        //        max = max(0, NULL) = 0
        assert_eq!(wf_curr.get_offset(-1), 0);

        // k=0: ins=prev[-1]+1=NULL+1≈NULL, del=prev[1]=NULL
        //       max = max(NULL, NULL) = NULL
        // But bounds check: h and v are huge negative → set to NULL
        assert_eq!(wf_curr.get_offset(0), OFFSET_NULL);

        // k=1: ins=prev[0]+1=1, del=prev[2]=NULL
        //       max = max(NULL, 1) = 1
        assert_eq!(wf_curr.get_offset(1), 1);
    }
}
