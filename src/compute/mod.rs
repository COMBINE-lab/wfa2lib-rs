//! Shared wavefront compute helpers.

pub mod affine;
pub mod affine2p;
pub mod edit;
pub mod linear;

use crate::offset::{wavefront_h, wavefront_v};
use crate::wavefront::Wavefront;

/// Trim wavefront ends: remove diagonals where the offset is out of bounds.
/// Sets `wf.null = true` if all diagonals are trimmed.
pub fn trim_ends(pattern_length: i32, text_length: i32, wf: &mut Wavefront) {
    // Use pre-centered pointer for direct k-indexed access
    let centered = unsafe { wf.offsets_centered_mut_ptr() };

    // Trim from hi
    let lo = wf.lo;
    let mut k = wf.hi;
    while k >= lo {
        let offset = unsafe { *centered.offset(k as isize) };
        let h = wavefront_h(k, offset) as u32;
        let v = wavefront_v(k, offset) as u32;
        if h <= text_length as u32 && v <= pattern_length as u32 {
            break;
        }
        k -= 1;
    }
    wf.hi = k;
    wf.wf_elements_init_max = k;

    // Trim from lo
    let hi = wf.hi;
    k = wf.lo;
    while k <= hi {
        let offset = unsafe { *centered.offset(k as isize) };
        let h = wavefront_h(k, offset) as u32;
        let v = wavefront_v(k, offset) as u32;
        if h <= text_length as u32 && v <= pattern_length as u32 {
            break;
        }
        k += 1;
    }
    wf.lo = k;
    wf.wf_elements_init_min = k;

    wf.null = wf.lo > wf.hi;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::OFFSET_NULL;

    #[test]
    fn test_trim_ends_no_trim() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(-2, 2);

        // Set valid offsets: for k in [-2, 2], offset = k + 3 (so v=3, h=k+3)
        for k in -2..=2 {
            wf.set_offset(k, k + 3);
        }

        trim_ends(10, 10, &mut wf);
        assert_eq!(wf.lo, -2);
        assert_eq!(wf.hi, 2);
        assert!(!wf.null);
    }

    #[test]
    fn test_trim_ends_all_null() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(-2, 2);

        // Set all offsets to NULL
        for k in -2..=2 {
            wf.set_offset(k, OFFSET_NULL);
        }

        trim_ends(10, 10, &mut wf);
        assert!(wf.null);
    }

    #[test]
    fn test_trim_ends_partial() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(-3, 3);

        // Set offsets: valid in middle, out-of-bounds at edges
        wf.set_offset(-3, OFFSET_NULL);
        wf.set_offset(-2, 3); // v=5, h=3 — valid for plen=10, tlen=10
        wf.set_offset(-1, 3); // v=4, h=3
        wf.set_offset(0, 3); // v=3, h=3
        wf.set_offset(1, 3); // v=2, h=3
        wf.set_offset(2, 3); // v=1, h=3
        wf.set_offset(3, OFFSET_NULL);

        trim_ends(10, 10, &mut wf);
        assert_eq!(wf.lo, -2);
        assert_eq!(wf.hi, 2);
    }
}
