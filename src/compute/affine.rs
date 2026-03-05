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

/// Gap-affine compute kernel.
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
#[inline(never)]
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
    let m_misms_offsets = wf_m_misms.offsets_slice();
    let m_misms_base = wf_m_misms.base_k();
    let m_open_offsets = wf_m_open.offsets_slice();
    let m_open_base = wf_m_open.base_k();
    let i1_ext_offsets = wf_i1_ext.offsets_slice();
    let i1_ext_base = wf_i1_ext.base_k();
    let d1_ext_offsets = wf_d1_ext.offsets_slice();
    let d1_ext_base = wf_d1_ext.base_k();

    let m_curr_base = wf_m_curr.base_k();
    let m_curr_offsets = wf_m_curr.offsets_slice_mut();
    let i1_curr_base = wf_i1_curr.base_k();
    let i1_curr_offsets = wf_i1_curr.offsets_slice_mut();
    let d1_curr_base = wf_d1_curr.base_k();
    let d1_curr_offsets = wf_d1_curr.offsets_slice_mut();

    for k in lo..=hi {
        // I1[s,k] = max(M[s-(o+e), k-1], I1[s-e, k-1]) + 1
        let i1_m_open = m_open_offsets[((k - 1) - m_open_base) as usize];
        let i1_ext = i1_ext_offsets[((k - 1) - i1_ext_base) as usize];
        let i1: WfOffset = i1_m_open.max(i1_ext) + 1;

        // D1[s,k] = max(M[s-(o+e), k+1], D1[s-e, k+1])
        let d1_m_open = m_open_offsets[((k + 1) - m_open_base) as usize];
        let d1_ext = d1_ext_offsets[((k + 1) - d1_ext_base) as usize];
        let d1: WfOffset = d1_m_open.max(d1_ext);

        // M[s,k] = max(M[s-x, k] + 1, I1[s,k], D1[s,k])
        let misms = m_misms_offsets[(k - m_misms_base) as usize] + 1;
        let mut m: WfOffset = misms.max(i1.max(d1));

        // Bounds check using unsigned cast
        let h = wavefront_h(k, m) as u32;
        let v = wavefront_v(k, m) as u32;
        if h > text_length as u32 || v > pattern_length as u32 {
            m = OFFSET_NULL;
        }

        m_curr_offsets[(k - m_curr_base) as usize] = m;
        i1_curr_offsets[(k - i1_curr_base) as usize] = i1;
        d1_curr_offsets[(k - d1_curr_base) as usize] = d1;
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
