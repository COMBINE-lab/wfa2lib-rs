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
#[inline(never)]
pub fn compute_linear_idm(
    pattern_length: i32,
    text_length: i32,
    wf_misms: &Wavefront,
    wf_gap: &Wavefront,
    wf_curr: &mut Wavefront,
    lo: i32,
    hi: i32,
) {
    let misms_offsets = wf_misms.offsets_slice();
    let misms_base = wf_misms.base_k();
    let gap_offsets = wf_gap.offsets_slice();
    let gap_base = wf_gap.base_k();
    let curr_base = wf_curr.base_k();
    let curr_offsets = wf_curr.offsets_slice_mut();

    for k in lo..=hi {
        // Mismatch: M[s-x, k] + 1
        let misms = misms_offsets[(k - misms_base) as usize] + 1;
        // Insertion: M[s-o, k-1] + 1
        let ins = gap_offsets[((k - 1) - gap_base) as usize] + 1;
        // Deletion: M[s-o, k+1]
        let del = gap_offsets[((k + 1) - gap_base) as usize];

        let mut max: WfOffset = del.max(ins.max(misms));

        // Bounds check using unsigned cast
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
