//! Wavefront compute kernel for gap-affine 2-piece distance.
//!
//! Gap-affine 2-piece uses a concave gap penalty model with two gap-cost
//! functions (O1+E1 for short gaps, O2+E2 for long gaps) and five wavefront
//! components: M, I1, I2, D1, D2.
//!
//! Recurrence:
//!   I1[s,k] = max(M[s-(O1+E1), k-1], I1[s-E1, k-1]) + 1
//!   I2[s,k] = max(M[s-(O2+E2), k-1], I2[s-E2, k-1]) + 1
//!   D1[s,k] = max(M[s-(O1+E1), k+1], D1[s-E1, k+1])
//!   D2[s,k] = max(M[s-(O2+E2), k+1], D2[s-E2, k+1])
//!   M[s,k]  = max(M[s-X, k] + 1, I1[s,k], I2[s,k], D1[s,k], D2[s,k])

use crate::offset::{OFFSET_NULL, WfOffset, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Gap-affine 2-piece compute kernel.
///
/// Input wavefronts (read-only):
/// - `wf_m_misms`: M at score - X
/// - `wf_m_open1`: M at score - (O1+E1)
/// - `wf_m_open2`: M at score - (O2+E2)
/// - `wf_i1_ext`: I1 at score - E1
/// - `wf_i2_ext`: I2 at score - E2
/// - `wf_d1_ext`: D1 at score - E1
/// - `wf_d2_ext`: D2 at score - E2
///
/// Output wavefronts (write):
/// - `wf_m_curr`, `wf_i1_curr`, `wf_i2_curr`, `wf_d1_curr`, `wf_d2_curr`
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn compute_affine2p_idm(
    pattern_length: i32,
    text_length: i32,
    wf_m_misms: &Wavefront,
    wf_m_open1: &Wavefront,
    wf_m_open2: &Wavefront,
    wf_i1_ext: &Wavefront,
    wf_i2_ext: &Wavefront,
    wf_d1_ext: &Wavefront,
    wf_d2_ext: &Wavefront,
    wf_m_curr: &mut Wavefront,
    wf_i1_curr: &mut Wavefront,
    wf_i2_curr: &mut Wavefront,
    wf_d1_curr: &mut Wavefront,
    wf_d2_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    // SAFETY: Bounds guaranteed by caller (lo/hi clamped, init_ends called).
    unsafe {
        let m_misms = wf_m_misms.offsets_centered_ptr();
        let m_open1 = wf_m_open1.offsets_centered_ptr();
        let m_open2 = wf_m_open2.offsets_centered_ptr();
        let i1_ext = wf_i1_ext.offsets_centered_ptr();
        let i2_ext = wf_i2_ext.offsets_centered_ptr();
        let d1_ext = wf_d1_ext.offsets_centered_ptr();
        let d2_ext = wf_d2_ext.offsets_centered_ptr();
        let m_curr = wf_m_curr.offsets_centered_mut_ptr();
        let i1_curr = wf_i1_curr.offsets_centered_mut_ptr();
        let i2_curr = wf_i2_curr.offsets_centered_mut_ptr();
        let d1_curr = wf_d1_curr.offsets_centered_mut_ptr();
        let d2_curr = wf_d2_curr.offsets_centered_mut_ptr();

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
                        // I1, I2 (shifted by -1)
                        let v_i1 = vaddq_s32(vmaxq_s32(
                            vld1q_s32(m_open1.offset(k as isize - 1)),
                            vld1q_s32(i1_ext.offset(k as isize - 1)),
                        ), v_one);
                        let v_i2 = vaddq_s32(vmaxq_s32(
                            vld1q_s32(m_open2.offset(k as isize - 1)),
                            vld1q_s32(i2_ext.offset(k as isize - 1)),
                        ), v_one);
                        // D1, D2 (shifted by +1)
                        let v_d1 = vmaxq_s32(
                            vld1q_s32(m_open1.offset(k as isize + 1)),
                            vld1q_s32(d1_ext.offset(k as isize + 1)),
                        );
                        let v_d2 = vmaxq_s32(
                            vld1q_s32(m_open2.offset(k as isize + 1)),
                            vld1q_s32(d2_ext.offset(k as isize + 1)),
                        );
                        // M = max(misms+1, max(i1,i2), max(d1,d2))
                        let v_misms = vaddq_s32(vld1q_s32(m_misms.offset(k as isize)), v_one);
                        let v_ins = vmaxq_s32(v_i1, v_i2);
                        let v_del = vmaxq_s32(v_d1, v_d2);
                        let v_m = vmaxq_s32(v_misms, vmaxq_s32(v_ins, v_del));

                        // Bounds check
                        let v_h = vreinterpretq_u32_s32(v_m);
                        let v_v = vreinterpretq_u32_s32(vsubq_s32(v_m, v_k));
                        let oob = vorrq_u32(vcgtq_u32(v_h, v_tlen), vcgtq_u32(v_v, v_plen));
                        let v_m_final = vbslq_s32(oob, v_null, v_m);

                        // Store all 5 outputs
                        vst1q_s32(m_curr.offset(k as isize), v_m_final);
                        vst1q_s32(i1_curr.offset(k as isize), v_i1);
                        vst1q_s32(i2_curr.offset(k as isize), v_i2);
                        vst1q_s32(d1_curr.offset(k as isize), v_d1);
                        vst1q_s32(d2_curr.offset(k as isize), v_d2);

                        k += 4;
                        v_k = vaddq_s32(v_k, v_four);
                    }
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            if count >= 8 && is_x86_feature_detected!("avx2") {
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
                        // I1, I2 (shifted by -1)
                        let v_i1 = _mm256_add_epi32(_mm256_max_epi32(
                            _mm256_loadu_si256(m_open1.offset(k as isize - 1) as *const __m256i),
                            _mm256_loadu_si256(i1_ext.offset(k as isize - 1) as *const __m256i),
                        ), v_one);
                        let v_i2 = _mm256_add_epi32(_mm256_max_epi32(
                            _mm256_loadu_si256(m_open2.offset(k as isize - 1) as *const __m256i),
                            _mm256_loadu_si256(i2_ext.offset(k as isize - 1) as *const __m256i),
                        ), v_one);
                        // D1, D2 (shifted by +1)
                        let v_d1 = _mm256_max_epi32(
                            _mm256_loadu_si256(m_open1.offset(k as isize + 1) as *const __m256i),
                            _mm256_loadu_si256(d1_ext.offset(k as isize + 1) as *const __m256i),
                        );
                        let v_d2 = _mm256_max_epi32(
                            _mm256_loadu_si256(m_open2.offset(k as isize + 1) as *const __m256i),
                            _mm256_loadu_si256(d2_ext.offset(k as isize + 1) as *const __m256i),
                        );
                        // M = max(misms+1, max(i1,i2), max(d1,d2))
                        let v_misms = _mm256_add_epi32(
                            _mm256_loadu_si256(m_misms.offset(k as isize) as *const __m256i),
                            v_one,
                        );
                        let v_ins = _mm256_max_epi32(v_i1, v_i2);
                        let v_del = _mm256_max_epi32(v_d1, v_d2);
                        let v_m = _mm256_max_epi32(v_misms, _mm256_max_epi32(v_ins, v_del));

                        // Bounds check
                        let v_h_biased = _mm256_xor_si256(v_m, v_sign);
                        let v_v_biased = _mm256_xor_si256(_mm256_sub_epi32(v_m, v_k), v_sign);
                        let oob = _mm256_or_si256(
                            _mm256_cmpgt_epi32(v_h_biased, v_tlen_biased),
                            _mm256_cmpgt_epi32(v_v_biased, v_plen_biased),
                        );
                        let v_m_final = _mm256_blendv_epi8(v_m, v_null, oob);

                        // Store all 5 outputs
                        _mm256_storeu_si256(m_curr.offset(k as isize) as *mut __m256i, v_m_final);
                        _mm256_storeu_si256(i1_curr.offset(k as isize) as *mut __m256i, v_i1);
                        _mm256_storeu_si256(i2_curr.offset(k as isize) as *mut __m256i, v_i2);
                        _mm256_storeu_si256(d1_curr.offset(k as isize) as *mut __m256i, v_d1);
                        _mm256_storeu_si256(d2_curr.offset(k as isize) as *mut __m256i, v_d2);

                        k += 8;
                        v_k = _mm256_add_epi32(v_k, v_eight);
                    }
                }
            }
        }

        // Scalar tail
        while k <= hi {
            let i1_val: WfOffset = (*m_open1.offset(k as isize - 1))
                .max(*i1_ext.offset(k as isize - 1)) + 1;
            let i2_val: WfOffset = (*m_open2.offset(k as isize - 1))
                .max(*i2_ext.offset(k as isize - 1)) + 1;
            let d1_val: WfOffset = (*m_open1.offset(k as isize + 1))
                .max(*d1_ext.offset(k as isize + 1));
            let d2_val: WfOffset = (*m_open2.offset(k as isize + 1))
                .max(*d2_ext.offset(k as isize + 1));
            let misms_val = *m_misms.offset(k as isize) + 1;
            let ins = i1_val.max(i2_val);
            let del = d1_val.max(d2_val);
            let mut m: WfOffset = misms_val.max(ins.max(del));

            let h = wavefront_h(k, m) as u32;
            let v = wavefront_v(k, m) as u32;
            if h > text_length as u32 || v > pattern_length as u32 {
                m = OFFSET_NULL;
            }

            *m_curr.offset(k as isize) = m;
            *i1_curr.offset(k as isize) = i1_val;
            *i2_curr.offset(k as isize) = i2_val;
            *d1_curr.offset(k as isize) = d1_val;
            *d2_curr.offset(k as isize) = d2_val;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_null_wf() -> Wavefront {
        let mut wf = Wavefront::allocate(11, false);
        wf.init_null(-5, 5);
        wf
    }

    #[test]
    fn test_affine2p_kernel_gap_open() {
        // Test gap opening: only m_open1 contributes (M[0] at k=0)
        let mut wf_m_open1 = Wavefront::allocate(11, false);
        wf_m_open1.init(-5, 5);
        wf_m_open1.set_limits(0, 0);
        wf_m_open1.set_offset(0, 0);
        for k in [-2, -1, 1, 2] {
            wf_m_open1.set_offset(k, OFFSET_NULL);
        }

        let wf_m_misms = make_null_wf();
        let wf_m_open2 = make_null_wf();
        let wf_i1_ext = make_null_wf();
        let wf_i2_ext = make_null_wf();
        let wf_d1_ext = make_null_wf();
        let wf_d2_ext = make_null_wf();

        let mut wf_m_curr = Wavefront::allocate(11, false);
        wf_m_curr.init(-5, 5);
        let mut wf_i1_curr = Wavefront::allocate(11, false);
        wf_i1_curr.init(-5, 5);
        let mut wf_i2_curr = Wavefront::allocate(11, false);
        wf_i2_curr.init(-5, 5);
        let mut wf_d1_curr = Wavefront::allocate(11, false);
        wf_d1_curr.init(-5, 5);
        let mut wf_d2_curr = Wavefront::allocate(11, false);
        wf_d2_curr.init(-5, 5);

        compute_affine2p_idm(
            4,
            4,
            &wf_m_misms,
            &wf_m_open1,
            &wf_m_open2,
            &wf_i1_ext,
            &wf_i2_ext,
            &wf_d1_ext,
            &wf_d2_ext,
            &mut wf_m_curr,
            &mut wf_i1_curr,
            &mut wf_i2_curr,
            &mut wf_d1_curr,
            &mut wf_d2_curr,
            -1,
            1,
        );

        // k=1: I1 = max(m_open1[0]=0, NULL)+1 = 1
        assert_eq!(wf_i1_curr.get_offset(1), 1);
        // k=-1: D1 = max(m_open1[0]=0, NULL) = 0
        assert_eq!(wf_d1_curr.get_offset(-1), 0);
    }
}
