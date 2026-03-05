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
#[inline(never)]
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
    let m_misms_offsets = wf_m_misms.offsets_slice();
    let m_misms_base = wf_m_misms.base_k();
    let m_open1_offsets = wf_m_open1.offsets_slice();
    let m_open1_base = wf_m_open1.base_k();
    let m_open2_offsets = wf_m_open2.offsets_slice();
    let m_open2_base = wf_m_open2.base_k();
    let i1_ext_offsets = wf_i1_ext.offsets_slice();
    let i1_ext_base = wf_i1_ext.base_k();
    let i2_ext_offsets = wf_i2_ext.offsets_slice();
    let i2_ext_base = wf_i2_ext.base_k();
    let d1_ext_offsets = wf_d1_ext.offsets_slice();
    let d1_ext_base = wf_d1_ext.base_k();
    let d2_ext_offsets = wf_d2_ext.offsets_slice();
    let d2_ext_base = wf_d2_ext.base_k();

    let m_curr_base = wf_m_curr.base_k();
    let m_curr_offsets = wf_m_curr.offsets_slice_mut();
    let i1_curr_base = wf_i1_curr.base_k();
    let i1_curr_offsets = wf_i1_curr.offsets_slice_mut();
    let i2_curr_base = wf_i2_curr.base_k();
    let i2_curr_offsets = wf_i2_curr.offsets_slice_mut();
    let d1_curr_base = wf_d1_curr.base_k();
    let d1_curr_offsets = wf_d1_curr.offsets_slice_mut();
    let d2_curr_base = wf_d2_curr.base_k();
    let d2_curr_offsets = wf_d2_curr.offsets_slice_mut();

    for k in lo..=hi {
        // I1[s,k] = max(M[s-(O1+E1), k-1], I1[s-E1, k-1]) + 1
        let i1_m_open = m_open1_offsets[((k - 1) - m_open1_base) as usize];
        let i1_ext = i1_ext_offsets[((k - 1) - i1_ext_base) as usize];
        let i1: WfOffset = i1_m_open.max(i1_ext) + 1;

        // I2[s,k] = max(M[s-(O2+E2), k-1], I2[s-E2, k-1]) + 1
        let i2_m_open = m_open2_offsets[((k - 1) - m_open2_base) as usize];
        let i2_ext = i2_ext_offsets[((k - 1) - i2_ext_base) as usize];
        let i2: WfOffset = i2_m_open.max(i2_ext) + 1;

        // D1[s,k] = max(M[s-(O1+E1), k+1], D1[s-E1, k+1])
        let d1_m_open = m_open1_offsets[((k + 1) - m_open1_base) as usize];
        let d1_ext = d1_ext_offsets[((k + 1) - d1_ext_base) as usize];
        let d1: WfOffset = d1_m_open.max(d1_ext);

        // D2[s,k] = max(M[s-(O2+E2), k+1], D2[s-E2, k+1])
        let d2_m_open = m_open2_offsets[((k + 1) - m_open2_base) as usize];
        let d2_ext = d2_ext_offsets[((k + 1) - d2_ext_base) as usize];
        let d2: WfOffset = d2_m_open.max(d2_ext);

        // M[s,k] = max(M[s-X, k] + 1, I1, I2, D1, D2)
        let misms = m_misms_offsets[(k - m_misms_base) as usize] + 1;
        let ins = i1.max(i2);
        let del = d1.max(d2);
        let mut m: WfOffset = misms.max(ins.max(del));

        // Bounds check using unsigned cast
        let h = wavefront_h(k, m) as u32;
        let v = wavefront_v(k, m) as u32;
        if h > text_length as u32 || v > pattern_length as u32 {
            m = OFFSET_NULL;
        }

        m_curr_offsets[(k - m_curr_base) as usize] = m;
        i1_curr_offsets[(k - i1_curr_base) as usize] = i1;
        i2_curr_offsets[(k - i2_curr_base) as usize] = i2;
        d1_curr_offsets[(k - d1_curr_base) as usize] = d1;
        d2_curr_offsets[(k - d2_curr_base) as usize] = d2;
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
