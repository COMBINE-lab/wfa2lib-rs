//! Wavefront compute kernel for gap-linear distance.
//!
//! Gap-linear (Needleman-Wunsch) uses separate mismatch and indel penalties
//! but no gap open/extend distinction. Only M-wavefronts are needed.
//!
//! Recurrence:
//!   M[s,k] = max(M[s-x, k] + 1, M[s-o, k-1] + 1, M[s-o, k+1])
//! where x = mismatch penalty, o = gap_opening1 (indel penalty).

use crate::offset::{OFFSET_NULL, WfOffset, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Gap-linear compute kernel.
///
/// Reads from two input M-wavefronts:
/// - `wf_misms`: M-wavefront at score - mismatch
/// - `wf_gap`: M-wavefront at score - indel
///
/// Writes to `wf_curr` for diagonals [lo, hi].
#[inline]
pub fn compute_linear_idm(
    pattern_length: i32,
    text_length: i32,
    wf_misms: &Wavefront,
    wf_gap: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    // SAFETY: Bounds guaranteed by caller (lo/hi clamped, init_ends called).
    unsafe {
        let misms_ptr = wf_misms.offsets_slice().as_ptr().wrapping_offset(-(wf_misms.base_k() as isize));
        let gap_ptr = wf_gap.offsets_slice().as_ptr().wrapping_offset(-(wf_gap.base_k() as isize));
        let curr_ptr = wf_curr.offsets_slice_mut().as_mut_ptr().wrapping_offset(-(wf_curr.base_k() as isize));

        let count = hi - lo + 1;
        let mut k = lo;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            if count >= 4 {
                unsafe {
                    let v_one = vdupq_n_s32(1);
                    let v_four = vdupq_n_s32(4);
                    let v_null = vdupq_n_s32(OFFSET_NULL);
                    let v_tlen = vdupq_n_u32(text_length as u32);
                    let v_plen = vdupq_n_u32(pattern_length as u32);
                    let k_base: [i32; 4] = [0, 1, 2, 3];
                    let mut v_k = vaddq_s32(vdupq_n_s32(lo), vld1q_s32(k_base.as_ptr()));
                    let neon_end = lo + (count & !3);

                    while k < neon_end {
                        let v_misms = vaddq_s32(vld1q_s32(misms_ptr.offset(k as isize)), v_one);
                        let v_ins = vaddq_s32(vld1q_s32(gap_ptr.offset(k as isize - 1)), v_one);
                        let v_del = vld1q_s32(gap_ptr.offset(k as isize + 1));
                        let v_max = vmaxq_s32(v_del, vmaxq_s32(v_ins, v_misms));

                        let v_h = vreinterpretq_u32_s32(v_max);
                        let v_v = vreinterpretq_u32_s32(vsubq_s32(v_max, v_k));
                        let oob = vorrq_u32(vcgtq_u32(v_h, v_tlen), vcgtq_u32(v_v, v_plen));
                        let v_result = vbslq_s32(oob, v_null, v_max);

                        vst1q_s32(curr_ptr.offset(k as isize), v_result);
                        k += 4;
                        v_k = vaddq_s32(v_k, v_four);
                    }
                }
            }
        }

        while k <= hi {
            let misms = *misms_ptr.offset(k as isize) + 1;
            let ins = *gap_ptr.offset(k as isize - 1) + 1;
            let del = *gap_ptr.offset(k as isize + 1);
            let mut max: WfOffset = del.max(ins.max(misms));

            let h = wavefront_h(k, max) as u32;
            let v = wavefront_v(k, max) as u32;
            if h > text_length as u32 || v > pattern_length as u32 {
                max = OFFSET_NULL;
            }

            *curr_ptr.offset(k as isize) = max;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kernel_basic() {
        // Simulate gap-linear with mismatch=4, indel=2
        // At score=2 (indel cost), we need wf_gap (score=0) and wf_misms (score=-2 → null)

        // wf_gap = M[score=0]: k=0, offset=0
        let mut wf_gap = Wavefront::allocate(7, false);
        wf_gap.init(-3, 3);
        wf_gap.set_limits(0, 0);
        wf_gap.set_offset(0, 0);
        // Padding
        wf_gap.set_offset(-2, OFFSET_NULL);
        wf_gap.set_offset(-1, OFFSET_NULL);
        wf_gap.set_offset(1, OFFSET_NULL);
        wf_gap.set_offset(2, OFFSET_NULL);

        // wf_misms = null wavefront (score - mismatch < 0)
        let mut wf_misms = Wavefront::allocate(7, false);
        wf_misms.init(-3, 3);
        wf_misms.init_null(-3, 3);

        let mut wf_curr = Wavefront::allocate(7, false);
        wf_curr.init(-3, 3);
        wf_curr.lo = -1;
        wf_curr.hi = 1;

        compute_linear_idm(4, 4, &wf_misms, &wf_gap, &mut wf_curr, -1, 1);

        // k=-1: misms=NULL+1≈NULL, ins=gap[-2]+1=NULL+1≈NULL, del=gap[0]=0
        // max = 0
        assert_eq!(wf_curr.get_offset(-1), 0);

        // k=0: misms=NULL+1≈NULL, ins=gap[-1]+1=NULL+1≈NULL, del=gap[1]=NULL
        // max = NULL (all null) → but bounds check will also catch it
        assert_eq!(wf_curr.get_offset(0), OFFSET_NULL);

        // k=1: misms=NULL+1≈NULL, ins=gap[0]+1=1, del=gap[2]=NULL
        // max = 1
        assert_eq!(wf_curr.get_offset(1), 1);
    }
}
