//! Wavefront compute kernel for gap-affine distance.
//!
//! Gap-affine (Smith-Waterman-Gotoh) uses separate gap opening and extension
//! penalties with three wavefront components: M, I1, D1.
//!
//! Recurrence:
//!   I1[s,k] = max(M[s-(o+e), k-1], I1[s-e, k-1]) + 1
//!   D1[s,k] = max(M[s-(o+e), k+1], D1[s-e, k+1])
//!   M[s,k]  = max(M[s-x, k] + 1, I1[s,k], D1[s,k])

use crate::offset::{OFFSET_NULL, WfOffset, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Gap-affine compute kernel (NEON-vectorized on aarch64, scalar fallback elsewhere).
///
/// Input wavefronts (read-only):
/// - `wf_m_misms`: M-wavefront at score - mismatch (for M→M mismatch transition)
/// - `wf_m_open`: M-wavefront at score - (O+E) (for M→I1/D1 gap open)
/// - `wf_i1_ext`: I1-wavefront at score - E (for I1→I1 gap extend)
/// - `wf_d1_ext`: D1-wavefront at score - E (for D1→D1 gap extend)
///
/// Output wavefronts (write):
/// - `wf_m_curr`: M-wavefront at current score
/// - `wf_i1_curr`: I1-wavefront at current score
/// - `wf_d1_curr`: D1-wavefront at current score
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn compute_affine_idm(
    pattern_length: i32,
    text_length: i32,
    wf_m_misms: &Wavefront,
    wf_m_open: &Wavefront,
    wf_i1_ext: &Wavefront,
    wf_d1_ext: &Wavefront,
    wf_m_curr: &mut Wavefront,
    wf_i1_curr: &mut Wavefront,
    wf_d1_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    // Pre-adjusted raw pointers: ptr[k] accesses diagonal k directly,
    // matching the C pattern `offsets = offsets_mem - min_lo`.
    // SAFETY: lo/hi are clamped to [hist_lo+1, hist_hi-1] by the caller.
    // Input wavefronts have init_ends called to cover [lo-1, hi+1].
    // Output wavefronts are allocated with full [hist_lo, hist_hi] range.
    // All final pointer dereferences are within allocated bounds.
    unsafe {
        let m_misms = wf_m_misms.offsets_slice().as_ptr().wrapping_offset(-(wf_m_misms.base_k() as isize));
        let m_open = wf_m_open.offsets_slice().as_ptr().wrapping_offset(-(wf_m_open.base_k() as isize));
        let i1_ext = wf_i1_ext.offsets_slice().as_ptr().wrapping_offset(-(wf_i1_ext.base_k() as isize));
        let d1_ext = wf_d1_ext.offsets_slice().as_ptr().wrapping_offset(-(wf_d1_ext.base_k() as isize));
        let m_curr = wf_m_curr.offsets_slice_mut().as_mut_ptr().wrapping_offset(-(wf_m_curr.base_k() as isize));
        let i1_curr = wf_i1_curr.offsets_slice_mut().as_mut_ptr().wrapping_offset(-(wf_i1_curr.base_k() as isize));
        let d1_curr = wf_d1_curr.offsets_slice_mut().as_mut_ptr().wrapping_offset(-(wf_d1_curr.base_k() as isize));

        #[cfg(target_arch = "aarch64")]
        {
            compute_affine_idm_neon(
                pattern_length, text_length,
                m_misms, m_open, i1_ext, d1_ext,
                m_curr, i1_curr, d1_curr,
                lo, hi,
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            compute_affine_idm_scalar(
                pattern_length, text_length,
                m_misms, m_open, i1_ext, d1_ext,
                m_curr, i1_curr, d1_curr,
                lo, hi,
            );
        }
    }
}

/// NEON-vectorized inner loop: processes 4 diagonals at a time.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
unsafe fn compute_affine_idm_neon(
    pattern_length: i32,
    text_length: i32,
    m_misms: *const WfOffset,
    m_open: *const WfOffset,
    i1_ext: *const WfOffset,
    d1_ext: *const WfOffset,
    m_curr: *mut WfOffset,
    i1_curr: *mut WfOffset,
    d1_curr: *mut WfOffset,
    lo: i32,
    hi: i32,
) {
    use std::arch::aarch64::*;

    let count = hi - lo + 1;
    if count <= 0 {
        return;
    }

    let mut k = lo;

    // NEON: process 4 diagonals at a time
    if count >= 4 {
        unsafe {
            let v_one = vdupq_n_s32(1);
            let v_four = vdupq_n_s32(4);
            let v_null = vdupq_n_s32(OFFSET_NULL);
            let v_tlen = vdupq_n_u32(text_length as u32);
            let v_plen = vdupq_n_u32(pattern_length as u32);
            // k_vec = [lo, lo+1, lo+2, lo+3]
            let k_base: [i32; 4] = [0, 1, 2, 3];
            let mut v_k = vaddq_s32(vdupq_n_s32(lo), vld1q_s32(k_base.as_ptr()));

            let neon_end = lo + (count & !3); // round down to multiple of 4
            while k < neon_end {
                // I1: load m_open[k-1] and i1_ext[k-1] (unaligned, shifted by -1)
                let v_m_open_km1 = vld1q_s32(m_open.offset(k as isize - 1));
                let v_i1_ext_km1 = vld1q_s32(i1_ext.offset(k as isize - 1));
                let v_i1 = vaddq_s32(vmaxq_s32(v_m_open_km1, v_i1_ext_km1), v_one);

                // D1: load m_open[k+1] and d1_ext[k+1] (unaligned, shifted by +1)
                let v_m_open_kp1 = vld1q_s32(m_open.offset(k as isize + 1));
                let v_d1_ext_kp1 = vld1q_s32(d1_ext.offset(k as isize + 1));
                let v_d1 = vmaxq_s32(v_m_open_kp1, v_d1_ext_kp1);

                // M: load m_misms[k], add 1, then max with i1, d1
                let v_misms = vaddq_s32(vld1q_s32(m_misms.offset(k as isize)), v_one);
                let v_m = vmaxq_s32(v_d1, vmaxq_s32(v_misms, v_i1));

                // Bounds check: h = m (unsigned), v = m - k (unsigned)
                // if h > text_length || v > pattern_length → OFFSET_NULL
                let v_h = vreinterpretq_u32_s32(v_m);
                let v_v = vreinterpretq_u32_s32(vsubq_s32(v_m, v_k));
                let h_oob = vcgtq_u32(v_h, v_tlen);
                let v_oob = vcgtq_u32(v_v, v_plen);
                let oob = vorrq_u32(h_oob, v_oob);
                // Select: if oob, use OFFSET_NULL, else use v_m
                let v_m_final = vbslq_s32(oob, v_null, v_m);

                // Store results
                vst1q_s32(m_curr.offset(k as isize), v_m_final);
                vst1q_s32(i1_curr.offset(k as isize), v_i1);
                vst1q_s32(d1_curr.offset(k as isize), v_d1);

                k += 4;
                v_k = vaddq_s32(v_k, v_four);
            }
        }
    }

    // Scalar tail for remaining diagonals
    while k <= hi {
        let i1_val: WfOffset = (*m_open.offset(k as isize - 1))
            .max(*i1_ext.offset(k as isize - 1)) + 1;
        let d1_val: WfOffset = (*m_open.offset(k as isize + 1))
            .max(*d1_ext.offset(k as isize + 1));
        let misms = *m_misms.offset(k as isize) + 1;
        let mut m: WfOffset = misms.max(i1_val.max(d1_val));

        let h = wavefront_h(k, m) as u32;
        let v = wavefront_v(k, m) as u32;
        if h > text_length as u32 || v > pattern_length as u32 {
            m = OFFSET_NULL;
        }

        *m_curr.offset(k as isize) = m;
        *i1_curr.offset(k as isize) = i1_val;
        *d1_curr.offset(k as isize) = d1_val;
        k += 1;
    }
}

/// Scalar fallback for non-aarch64 platforms.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
#[allow(clippy::too_many_arguments, unsafe_op_in_unsafe_fn)]
unsafe fn compute_affine_idm_scalar(
    pattern_length: i32,
    text_length: i32,
    m_misms: *const WfOffset,
    m_open: *const WfOffset,
    i1_ext: *const WfOffset,
    d1_ext: *const WfOffset,
    m_curr: *mut WfOffset,
    i1_curr: *mut WfOffset,
    d1_curr: *mut WfOffset,
    lo: i32,
    hi: i32,
) {
    let mut k = lo;
    while k <= hi {
        let i1_val: WfOffset = (*m_open.offset(k as isize - 1))
            .max(*i1_ext.offset(k as isize - 1)) + 1;
        let d1_val: WfOffset = (*m_open.offset(k as isize + 1))
            .max(*d1_ext.offset(k as isize + 1));
        let misms = *m_misms.offset(k as isize) + 1;
        let mut m: WfOffset = misms.max(i1_val.max(d1_val));

        let h = wavefront_h(k, m) as u32;
        let v = wavefront_v(k, m) as u32;
        if h > text_length as u32 || v > pattern_length as u32 {
            m = OFFSET_NULL;
        }

        *m_curr.offset(k as isize) = m;
        *i1_curr.offset(k as isize) = i1_val;
        *d1_curr.offset(k as isize) = d1_val;
        k += 1;
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
    fn test_affine_kernel_gap_open() {
        // Score = O+E (e.g., 6+2=8 for default affine penalties)
        // Only wf_m_open (score=0) contributes gap opens

        let mut wf_m_open = Wavefront::allocate(11, false);
        wf_m_open.init(-5, 5);
        wf_m_open.set_limits(0, 0);
        wf_m_open.set_offset(0, 0);
        // Padding
        for k in [-2, -1, 1, 2] {
            wf_m_open.set_offset(k, OFFSET_NULL);
        }

        let wf_m_misms = make_null_wf(); // score - X < 0 for early scores
        let wf_i1_ext = make_null_wf(); // no I1 yet
        let wf_d1_ext = make_null_wf(); // no D1 yet

        let mut wf_m_curr = Wavefront::allocate(11, false);
        wf_m_curr.init(-5, 5);
        wf_m_curr.lo = -1;
        wf_m_curr.hi = 1;

        let mut wf_i1_curr = Wavefront::allocate(11, false);
        wf_i1_curr.init(-5, 5);

        let mut wf_d1_curr = Wavefront::allocate(11, false);
        wf_d1_curr.init(-5, 5);

        compute_affine_idm(
            4,
            4,
            &wf_m_misms,
            &wf_m_open,
            &wf_i1_ext,
            &wf_d1_ext,
            &mut wf_m_curr,
            &mut wf_i1_curr,
            &mut wf_d1_curr,
            -1,
            1,
        );

        // k=-1: I1 = max(m_open[-2]=NULL, i1_ext[-2]=NULL)+1 = NULL
        //        D1 = max(m_open[0]=0, d1_ext[0]=NULL) = 0
        //        M = max(misms[-1]=NULL+1, I1=NULL, D1=0) = 0
        assert_eq!(wf_d1_curr.get_offset(-1), 0);
        assert_eq!(wf_m_curr.get_offset(-1), 0);

        // k=1:  I1 = max(m_open[0]=0, i1_ext[0]=NULL)+1 = 1
        //        D1 = max(m_open[2]=NULL, d1_ext[2]=NULL) = NULL
        //        M = max(misms[1]=NULL+1, I1=1, D1=NULL) = 1
        assert_eq!(wf_i1_curr.get_offset(1), 1);
        assert_eq!(wf_m_curr.get_offset(1), 1);
    }
}
