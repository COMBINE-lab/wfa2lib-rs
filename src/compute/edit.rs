//! Wavefront compute kernels for edit distance and indel distance.

use crate::offset::{OFFSET_NULL, WfOffset, wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Edit distance compute kernel (no backtrace).
///
/// For each diagonal k in [lo, hi]:
///   ins   = prev[k-1]     (insertion: advance text)
///   del   = prev[k+1]     (deletion: advance pattern)
///   misms = prev[k]       (mismatch: advance both)
///   curr[k] = max(del, max(ins, misms) + 1)
///
/// Out-of-bounds offsets are set to OFFSET_NULL.
#[inline(never)]
pub fn compute_edit_idm(
    pattern_length: i32,
    text_length: i32,
    wf_prev: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    let prev_offsets = wf_prev.offsets_slice();
    let prev_base = wf_prev.base_k();
    let curr_base = wf_curr.base_k();
    let curr_offsets = wf_curr.offsets_slice_mut();

    for k in lo..=hi {
        let ins = prev_offsets[((k - 1) - prev_base) as usize];
        let del = prev_offsets[((k + 1) - prev_base) as usize];
        let misms = prev_offsets[(k - prev_base) as usize];
        let mut max = del.max(ins.max(misms) + 1);

        // Bounds check using unsigned cast (catches negative values too)
        let h = wavefront_h(k, max) as u32;
        let v = wavefront_v(k, max) as u32;
        if h > text_length as u32 {
            max = OFFSET_NULL;
        }
        if v > pattern_length as u32 {
            max = OFFSET_NULL;
        }

        curr_offsets[(k - curr_base) as usize] = max;
    }
}

/// Indel distance compute kernel (no backtrace).
///
/// For each diagonal k in [lo, hi]:
///   ins   = prev[k-1] + 1  (insertion)
///   del   = prev[k+1]      (deletion)
///   curr[k] = max(del, ins)
#[inline(never)]
pub fn compute_indel_idm(
    pattern_length: i32,
    text_length: i32,
    wf_prev: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    let prev_offsets = wf_prev.offsets_slice();
    let prev_base = wf_prev.base_k();
    let curr_base = wf_curr.base_k();
    let curr_offsets = wf_curr.offsets_slice_mut();

    for k in lo..=hi {
        let ins = prev_offsets[((k - 1) - prev_base) as usize] + 1;
        let del = prev_offsets[((k + 1) - prev_base) as usize];
        let mut max: WfOffset = del.max(ins);

        let h = wavefront_h(k, max) as u32;
        let v = wavefront_v(k, max) as u32;
        if h > text_length as u32 {
            max = OFFSET_NULL;
        }
        if v > pattern_length as u32 {
            max = OFFSET_NULL;
        }

        curr_offsets[(k - curr_base) as usize] = max;
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
