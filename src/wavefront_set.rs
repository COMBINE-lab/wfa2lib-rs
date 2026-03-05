//! Wavefront set: references to input/output wavefronts for a compute step.
//!
//! During each WFA compute step, we need to read from input wavefronts at
//! previous scores and write to output wavefronts at the current score.
//! This module stores raw pointers to these wavefronts.

use crate::components::WF_PTR_NONE;
use crate::wavefront::Wavefront;

/// A set of wavefront pointers for a single compute step.
///
/// Input wavefronts are read from (previous scores), output wavefronts
/// are written to (current score). `WF_PTR_NONE` (null) means the
/// wavefront is null/unused.
#[derive(Debug, Clone)]
pub struct WavefrontSet {
    // Input wavefronts
    /// M-wavefront for mismatch input.
    pub in_mwavefront_misms: *mut Wavefront,
    /// M-wavefront for gap-open1 input.
    pub in_mwavefront_open1: *mut Wavefront,
    /// M-wavefront for gap-open2 input (affine2p only).
    pub in_mwavefront_open2: *mut Wavefront,
    /// I1-wavefront for gap-extend1 input.
    pub in_i1wavefront_ext: *mut Wavefront,
    /// I2-wavefront for gap-extend2 input (affine2p only).
    pub in_i2wavefront_ext: *mut Wavefront,
    /// D1-wavefront for gap-extend1 input.
    pub in_d1wavefront_ext: *mut Wavefront,
    /// D2-wavefront for gap-extend2 input (affine2p only).
    pub in_d2wavefront_ext: *mut Wavefront,

    // Output wavefronts
    /// Output M-wavefront.
    pub out_mwavefront: *mut Wavefront,
    /// Output I1-wavefront.
    pub out_i1wavefront: *mut Wavefront,
    /// Output I2-wavefront (affine2p only).
    pub out_i2wavefront: *mut Wavefront,
    /// Output D1-wavefront.
    pub out_d1wavefront: *mut Wavefront,
    /// Output D2-wavefront (affine2p only).
    pub out_d2wavefront: *mut Wavefront,
}

impl Default for WavefrontSet {
    fn default() -> Self {
        Self::new()
    }
}

impl WavefrontSet {
    /// Create a new empty wavefront set (all pointers set to null).
    pub fn new() -> Self {
        Self {
            in_mwavefront_misms: WF_PTR_NONE,
            in_mwavefront_open1: WF_PTR_NONE,
            in_mwavefront_open2: WF_PTR_NONE,
            in_i1wavefront_ext: WF_PTR_NONE,
            in_i2wavefront_ext: WF_PTR_NONE,
            in_d1wavefront_ext: WF_PTR_NONE,
            in_d2wavefront_ext: WF_PTR_NONE,
            out_mwavefront: WF_PTR_NONE,
            out_i1wavefront: WF_PTR_NONE,
            out_i2wavefront: WF_PTR_NONE,
            out_d1wavefront: WF_PTR_NONE,
            out_d2wavefront: WF_PTR_NONE,
        }
    }
}
