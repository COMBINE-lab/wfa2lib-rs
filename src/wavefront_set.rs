//! Wavefront set: references to input/output wavefronts for a compute step.
//!
//! During each WFA compute step, we need to read from input wavefronts at
//! previous scores and write to output wavefronts at the current score.
//! This module stores indices into the slab for these wavefronts.

use crate::slab::WavefrontIdx;

/// A set of wavefront indices for a single compute step.
///
/// Input wavefronts are read from (previous scores), output wavefronts
/// are written to (current score). `WAVEFRONT_IDX_NONE` means the
/// wavefront is null/unused.
#[derive(Debug, Clone)]
pub struct WavefrontSet {
    // Input wavefronts
    /// M-wavefront for mismatch input.
    pub in_mwavefront_misms: WavefrontIdx,
    /// M-wavefront for gap-open1 input.
    pub in_mwavefront_open1: WavefrontIdx,
    /// M-wavefront for gap-open2 input (affine2p only).
    pub in_mwavefront_open2: WavefrontIdx,
    /// I1-wavefront for gap-extend1 input.
    pub in_i1wavefront_ext: WavefrontIdx,
    /// I2-wavefront for gap-extend2 input (affine2p only).
    pub in_i2wavefront_ext: WavefrontIdx,
    /// D1-wavefront for gap-extend1 input.
    pub in_d1wavefront_ext: WavefrontIdx,
    /// D2-wavefront for gap-extend2 input (affine2p only).
    pub in_d2wavefront_ext: WavefrontIdx,

    // Output wavefronts
    /// Output M-wavefront.
    pub out_mwavefront: WavefrontIdx,
    /// Output I1-wavefront.
    pub out_i1wavefront: WavefrontIdx,
    /// Output I2-wavefront (affine2p only).
    pub out_i2wavefront: WavefrontIdx,
    /// Output D1-wavefront.
    pub out_d1wavefront: WavefrontIdx,
    /// Output D2-wavefront (affine2p only).
    pub out_d2wavefront: WavefrontIdx,
}

impl Default for WavefrontSet {
    fn default() -> Self {
        Self::new()
    }
}

impl WavefrontSet {
    /// Create a new empty wavefront set (all indices set to NONE).
    pub fn new() -> Self {
        use crate::slab::WAVEFRONT_IDX_NONE;
        Self {
            in_mwavefront_misms: WAVEFRONT_IDX_NONE,
            in_mwavefront_open1: WAVEFRONT_IDX_NONE,
            in_mwavefront_open2: WAVEFRONT_IDX_NONE,
            in_i1wavefront_ext: WAVEFRONT_IDX_NONE,
            in_i2wavefront_ext: WAVEFRONT_IDX_NONE,
            in_d1wavefront_ext: WAVEFRONT_IDX_NONE,
            in_d2wavefront_ext: WAVEFRONT_IDX_NONE,
            out_mwavefront: WAVEFRONT_IDX_NONE,
            out_i1wavefront: WAVEFRONT_IDX_NONE,
            out_i2wavefront: WAVEFRONT_IDX_NONE,
            out_d1wavefront: WAVEFRONT_IDX_NONE,
            out_d2wavefront: WAVEFRONT_IDX_NONE,
        }
    }
}
