//! Wavefront offset type and coordinate utilities.
//!
//! The wavefront alignment algorithm operates on diagonals `k = h - v`,
//! where `v` is the position in the pattern and `h` is the position in the text.
//! Each diagonal stores an offset value that represents the furthest-reaching
//! point along that diagonal.

/// Wavefront offset type (signed 32-bit integer).
pub type WfOffset = i32;

/// Unsigned wavefront offset type, used for comparisons.
pub type WfUnsignedOffset = u32;

/// Sentinel value indicating an uninitialized or invalid offset.
pub const OFFSET_NULL: WfOffset = i32::MIN / 2;

/// Compute the number of elements in a wavefront with the given lo/hi bounds (inclusive).
#[inline(always)]
pub const fn wavefront_length(lo: i32, hi: i32) -> i32 {
    hi - lo + 1
}

/// Convert diagonal `k` and offset to pattern coordinate `v`.
/// v = offset - k
#[inline(always)]
pub const fn wavefront_v(k: i32, offset: WfOffset) -> i32 {
    offset - k
}

/// Convert diagonal `k` and offset to text coordinate `h`.
/// h = offset
#[inline(always)]
pub const fn wavefront_h(_k: i32, offset: WfOffset) -> i32 {
    offset
}

/// Compute the anti-diagonal from diagonal `k` and offset.
/// antidiag = 2*offset - k = h + v
#[inline(always)]
pub const fn wavefront_antidiagonal(k: i32, offset: WfOffset) -> i32 {
    2 * offset - k
}

/// Sentinel value for a null DP-matrix diagonal.
pub const DPMATRIX_DIAGONAL_NULL: i32 = i32::MAX;

/// Compute the diagonal from DP-matrix coordinates (h, v).
/// k = h - v
#[inline(always)]
pub const fn dpmatrix_diagonal(h: i32, v: i32) -> i32 {
    h - v
}

/// Compute the anti-diagonal from DP-matrix coordinates (h, v).
#[inline(always)]
pub const fn dpmatrix_antidiagonal(h: i32, v: i32) -> i32 {
    h + v
}

/// Compute the offset from DP-matrix coordinates (h, v).
#[inline(always)]
pub const fn dpmatrix_offset(h: i32, _v: i32) -> i32 {
    h
}

/// Compute the inverse diagonal.
/// k_inverse = (tlen - plen) - k
#[inline(always)]
pub const fn wavefront_k_inverse(k: i32, plen: i32, tlen: i32) -> i32 {
    (tlen - plen) - k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavefront_coordinates() {
        // On diagonal k=2, offset=5: h=5, v=3
        assert_eq!(wavefront_h(2, 5), 5);
        assert_eq!(wavefront_v(2, 5), 3);
        assert_eq!(wavefront_antidiagonal(2, 5), 8);
    }

    #[test]
    fn test_wavefront_length() {
        assert_eq!(wavefront_length(-3, 3), 7);
        assert_eq!(wavefront_length(0, 0), 1);
        assert_eq!(wavefront_length(-1, 1), 3);
    }

    #[test]
    fn test_dpmatrix_conversions() {
        // h=5, v=3 => diagonal=2, antidiagonal=8, offset=5
        assert_eq!(dpmatrix_diagonal(5, 3), 2);
        assert_eq!(dpmatrix_antidiagonal(5, 3), 8);
        assert_eq!(dpmatrix_offset(5, 3), 5);
    }

    #[test]
    fn test_k_inverse() {
        // plen=10, tlen=12, k=3 => k_inverse = (12-10)-3 = -1
        assert_eq!(wavefront_k_inverse(3, 10, 12), -1);
    }

    #[test]
    fn test_offset_null() {
        // Adding a reasonable value to OFFSET_NULL should not overflow
        let result = OFFSET_NULL + 1000;
        assert!(result < 0);
    }
}
