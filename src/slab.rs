//! Wavefront slab allocator for fast pre-allocated wavefront memory management.
//!
//! The slab owns all wavefronts and recycles them through a free list.
//! Two modes are supported:
//! - `Reuse`: keeps all wavefronts, grows slab size dynamically
//! - `Tight`: reaps wavefronts when size changes, keeps only init-sized ones

use crate::arena::OffsetArena;
use crate::offset::wavefront_length;
use crate::wavefront::{Wavefront, WavefrontStatus};

const WF_SLAB_EXPAND_FACTOR: f32 = 1.5;

/// Slab allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabMode {
    /// Keep all wavefronts; only reap by explicit demand.
    Reuse,
    /// Reap all wavefronts when they're resized.
    Tight,
}

/// Index into the slab's wavefront storage.
pub type WavefrontIdx = usize;

/// Sentinel index indicating no wavefront.
pub const WAVEFRONT_IDX_NONE: WavefrontIdx = usize::MAX;

/// Wavefront slab allocator.
pub struct WavefrontSlab {
    /// Whether wavefronts need backtrace vectors.
    allocate_backtrace: bool,
    /// Slab strategy.
    slab_mode: SlabMode,
    /// Initial wavefront element count.
    init_wf_length: i32,
    /// Current wavefront element count (may grow in Reuse mode).
    current_wf_length: i32,
    /// All wavefronts owned by this slab.
    wavefronts: Vec<Wavefront>,
    /// Indices of free wavefronts (stack).
    free_list: Vec<WavefrontIdx>,
    /// Total memory used in bytes.
    memory_used: u64,
    /// Arena for wavefront offset arrays (contiguous bump allocation).
    arena: OffsetArena,
}

impl WavefrontSlab {
    /// Create a new slab.
    pub fn new(init_wf_length: i32, allocate_backtrace: bool, slab_mode: SlabMode) -> Self {
        // Initial arena: enough for ~500 wavefronts at init size.
        // Large initial chunk avoids chunk splits in CIGAR mode.
        let arena_capacity = (init_wf_length as usize) * 500;
        Self {
            allocate_backtrace,
            slab_mode,
            init_wf_length,
            current_wf_length: init_wf_length,
            wavefronts: Vec::with_capacity(100),
            free_list: Vec::with_capacity(100),
            memory_used: 0,
            arena: OffsetArena::new(arena_capacity.max(4096)),
        }
    }

    /// Pre-allocate capacity for the wavefronts Vec to avoid reallocations.
    pub fn reserve(&mut self, num_wavefronts: usize) {
        self.wavefronts.reserve(num_wavefronts);
    }

    /// Get the slab mode.
    pub fn mode(&self) -> SlabMode {
        self.slab_mode
    }

    /// Set the slab mode; reaps and repurposes if changed.
    pub fn set_mode(&mut self, mode: SlabMode) {
        if mode != self.slab_mode {
            self.slab_mode = mode;
            self.current_wf_length = self.init_wf_length;
            self.reap_repurpose();
        }
    }

    /// Allocate a wavefront for the given diagonal range [min_lo, max_hi].
    /// Returns the index of the wavefront in the slab.
    pub fn allocate(&mut self, min_lo: i32, max_hi: i32) -> WavefrontIdx {
        let wf_length = wavefront_length(min_lo, max_hi);

        match self.slab_mode {
            SlabMode::Reuse => {
                // Grow current_wf_length if needed
                if wf_length > self.current_wf_length {
                    let proposed = (wf_length as f32 * WF_SLAB_EXPAND_FACTOR) as i32;
                    self.current_wf_length = proposed;
                    self.reap_free();
                }
                // Try to reuse a free wavefront
                if let Some(idx) = self.free_list.pop() {
                    self.wavefronts[idx].status = WavefrontStatus::Busy;
                    self.wavefronts[idx].init(min_lo, max_hi);
                    idx
                } else {
                    self.allocate_new(self.current_wf_length, min_lo, max_hi)
                }
            }
            SlabMode::Tight => {
                if wf_length <= self.init_wf_length {
                    if let Some(idx) = self.free_list.pop() {
                        self.wavefronts[idx].status = WavefrontStatus::Busy;
                        self.wavefronts[idx].init(min_lo, max_hi);
                        idx
                    } else {
                        self.allocate_new(self.init_wf_length, min_lo, max_hi)
                    }
                } else {
                    self.allocate_new(wf_length, min_lo, max_hi)
                }
            }
        }
    }

    /// Reuse an existing wavefront in-place (skip free-list round-trip).
    /// Returns the same index after reinitializing the wavefront.
    #[inline(always)]
    pub fn reuse_inplace(&mut self, idx: WavefrontIdx, min_lo: i32, max_hi: i32) {
        let wf = unsafe { self.wavefronts.get_unchecked_mut(idx) };
        wf.init(min_lo, max_hi);
        wf.status = WavefrontStatus::Busy;
    }

    /// Reuse an existing wavefront or allocate a new one.
    /// If `old_idx` is valid, reuses it in-place (avoids free-list round-trip).
    /// Otherwise allocates from the free list or creates new.
    #[inline(always)]
    pub fn reuse_or_allocate(
        &mut self,
        old_idx: WavefrontIdx,
        min_lo: i32,
        max_hi: i32,
    ) -> WavefrontIdx {
        if old_idx != WAVEFRONT_IDX_NONE {
            self.reuse_inplace(old_idx, min_lo, max_hi);
            old_idx
        } else {
            self.allocate(min_lo, max_hi)
        }
    }

    /// Free (return) a wavefront to the slab.
    pub fn free(&mut self, idx: WavefrontIdx) {
        let wf_length = self.wavefronts[idx].wf_elements_allocated;

        let repurpose_reuse =
            self.slab_mode == SlabMode::Reuse && wf_length == self.current_wf_length;
        let repurpose_tight = self.slab_mode == SlabMode::Tight && wf_length == self.init_wf_length;

        if repurpose_reuse || repurpose_tight {
            self.wavefronts[idx].status = WavefrontStatus::Free;
            self.free_list.push(idx);
        } else {
            self.memory_used -= self.wavefronts[idx].get_size();
            self.wavefronts[idx].status = WavefrontStatus::Deallocated;
        }
    }

    /// Reset to initial size and repurpose all wavefronts.
    pub fn reap(&mut self) {
        self.current_wf_length = self.init_wf_length;
        // With arena, just clear everything and reset
        self.wavefronts.clear();
        self.free_list.clear();
        self.memory_used = 0;
        self.arena.reset();
    }

    /// Clear the slab according to its mode. Resets the arena.
    pub fn clear(&mut self) {
        // Drop all wavefronts (arena-backed offsets will NOT be dealloc'd).
        self.wavefronts.clear();
        self.free_list.clear();
        self.memory_used = 0;
        self.arena.reset();
        match self.slab_mode {
            SlabMode::Reuse => {}
            SlabMode::Tight => {
                self.current_wf_length = self.init_wf_length;
            }
        }
    }

    /// Ensure the slab's current wavefront length is at least large enough
    /// to cover the given diagonal range.
    pub fn ensure_min_length(&mut self, min_lo: i32, max_hi: i32) {
        let wf_length = wavefront_length(min_lo, max_hi);
        if wf_length > self.current_wf_length {
            let proposed = (wf_length as f32 * WF_SLAB_EXPAND_FACTOR) as i32;
            self.current_wf_length = proposed;
            self.reap_free();
        }
    }

    /// Get a reference to a wavefront by index.
    #[inline(always)]
    pub fn get(&self, idx: WavefrontIdx) -> &Wavefront {
        unsafe { self.wavefronts.get_unchecked(idx) }
    }

    /// Get a mutable reference to a wavefront by index.
    #[inline(always)]
    pub fn get_mut(&mut self, idx: WavefrontIdx) -> &mut Wavefront {
        unsafe { self.wavefronts.get_unchecked_mut(idx) }
    }

    /// Get a raw mutable pointer to a wavefront by index.
    ///
    /// # Safety
    /// The caller must ensure that no two mutable references to the same
    /// wavefront exist simultaneously.
    #[inline(always)]
    pub unsafe fn get_raw_mut(&mut self, idx: WavefrontIdx) -> *mut Wavefront {
        unsafe { self.wavefronts.as_mut_ptr().add(idx) }
    }

    /// Get two mutable references to different wavefronts (panics if same index).
    #[inline(always)]
    pub fn get_two_mut(
        &mut self,
        a: WavefrontIdx,
        b: WavefrontIdx,
    ) -> (&mut Wavefront, &mut Wavefront) {
        assert!(
            a != b,
            "cannot get two mutable references to the same wavefront"
        );
        if a < b {
            let (left, right) = self.wavefronts.split_at_mut(b);
            (&mut left[a], &mut right[0])
        } else {
            let (left, right) = self.wavefronts.split_at_mut(a);
            (&mut right[0], &mut left[b])
        }
    }

    /// Allocate a wavefront and return a raw pointer into the slab's Vec.
    ///
    /// # Safety
    /// The slab Vec must have been pre-reserved (via reserve()) before this call,
    /// so no reallocation occurs and all previously stored pointers remain valid.
    pub fn allocate_ptr(&mut self, min_lo: i32, max_hi: i32) -> *mut Wavefront {
        let idx = self.allocate(min_lo, max_hi);
        unsafe { self.wavefronts.as_mut_ptr().add(idx) }
    }

    /// Re-initialize an existing wavefront in place (already allocated, just reset it).
    ///
    /// # Safety
    /// `ptr` must be a valid pointer into `self.wavefronts`.
    pub unsafe fn reuse_inplace_ptr(&mut self, ptr: *mut Wavefront, min_lo: i32, max_hi: i32) {
        unsafe {
            (*ptr).init(min_lo, max_hi);
            (*ptr).status = WavefrontStatus::Busy;
        }
    }

    /// Reuse an existing pointer in-place if non-null, otherwise allocate fresh.
    ///
    /// # Safety
    /// If non-null, `old_ptr` must be a valid pointer into `self.wavefronts`.
    pub fn reuse_or_allocate_ptr(&mut self, old_ptr: *mut Wavefront, min_lo: i32, max_hi: i32) -> *mut Wavefront {
        if old_ptr.is_null() {
            return self.allocate_ptr(min_lo, max_hi);
        }
        let base = self.wavefronts.as_ptr();
        let idx = unsafe { (old_ptr as *const Wavefront).offset_from(base) as usize };
        let reused_idx = self.reuse_or_allocate(idx, min_lo, max_hi);
        unsafe { self.wavefronts.as_mut_ptr().add(reused_idx) }
    }

    /// Free a wavefront by pointer.
    ///
    /// # Safety
    /// If non-null, `ptr` must be a valid pointer into `self.wavefronts`.
    pub fn free_ptr(&mut self, ptr: *mut Wavefront) {
        if ptr.is_null() {
            return;
        }
        let base = self.wavefronts.as_ptr();
        let idx = unsafe { (ptr as *const Wavefront).offset_from(base) as usize };
        self.free(idx);
    }

    /// Get total memory used.
    pub fn get_size(&self) -> u64 {
        self.memory_used
    }

    /// Get the number of wavefronts.
    pub fn num_wavefronts(&self) -> usize {
        self.wavefronts.len()
    }

    /// Get the number of free wavefronts.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    // --- Internal ---

    fn allocate_new(&mut self, wf_length: i32, min_lo: i32, max_hi: i32) -> WavefrontIdx {
        let mut wf =
            Wavefront::allocate_from_arena(&mut self.arena, wf_length, self.allocate_backtrace);
        wf.status = WavefrontStatus::Busy;
        wf.init(min_lo, max_hi);
        self.memory_used += wf.get_size();
        let idx = self.wavefronts.len();
        self.wavefronts.push(wf);
        idx
    }

    /// Remove deallocated and free wavefronts (Reuse mode reap).
    /// With arena, keep busy wavefronts and their offset pointers (still valid).
    /// Dead space from freed wavefronts is reclaimed on the next clear().
    fn reap_free(&mut self) {
        self.free_list.clear();
        let mut valid_idx = 0;
        for i in 0..self.wavefronts.len() {
            if self.wavefronts[i].status == WavefrontStatus::Busy {
                if valid_idx != i {
                    self.wavefronts.swap(valid_idx, i);
                }
                valid_idx += 1;
            }
            // Free and Deallocated wavefronts: metadata dropped, offset memory
            // becomes arena dead space (reclaimed on arena.reset()).
        }
        self.wavefronts.truncate(valid_idx);
    }

    /// Repurpose wavefronts matching current_wf_length; remove others.
    /// Offset pointers remain valid in the arena (not reset here).
    fn reap_repurpose(&mut self) {
        self.free_list.clear();
        let current_wf_length = self.current_wf_length;
        let mut valid_idx = 0;
        for i in 0..self.wavefronts.len() {
            match self.wavefronts[i].status {
                WavefrontStatus::Deallocated => {}
                WavefrontStatus::Busy | WavefrontStatus::Free => {
                    if self.wavefronts[i].wf_elements_allocated == current_wf_length {
                        self.wavefronts[i].status = WavefrontStatus::Free;
                        if valid_idx != i {
                            self.wavefronts.swap(valid_idx, i);
                        }
                        self.free_list.push(valid_idx);
                        valid_idx += 1;
                    }
                    // Non-matching wavefronts: dropped, arena space reclaimed on reset
                }
            }
        }
        self.wavefronts.truncate(valid_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_slab() {
        let slab = WavefrontSlab::new(100, false, SlabMode::Reuse);
        assert_eq!(slab.num_wavefronts(), 0);
        assert_eq!(slab.num_free(), 0);
    }

    #[test]
    fn test_allocate_and_access() {
        let mut slab = WavefrontSlab::new(100, false, SlabMode::Reuse);
        let idx = slab.allocate(-5, 5);

        let wf = slab.get(idx);
        assert_eq!(wf.status, WavefrontStatus::Busy);
        assert!(!wf.null);

        let wf = slab.get_mut(idx);
        wf.set_offset(0, 42);
        assert_eq!(wf.get_offset(0), 42);
    }

    #[test]
    fn test_free_and_reuse() {
        let mut slab = WavefrontSlab::new(100, false, SlabMode::Reuse);
        let idx1 = slab.allocate(-5, 5);
        assert_eq!(slab.num_wavefronts(), 1);
        assert_eq!(slab.num_free(), 0);

        slab.free(idx1);
        assert_eq!(slab.num_free(), 1);

        // Next allocation should reuse the free wavefront
        let idx2 = slab.allocate(-3, 3);
        assert_eq!(idx2, idx1);
        assert_eq!(slab.num_wavefronts(), 1);
        assert_eq!(slab.num_free(), 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut slab = WavefrontSlab::new(20, false, SlabMode::Reuse);
        let mut indices = Vec::new();
        for _ in 0..10 {
            indices.push(slab.allocate(-5, 5));
        }
        assert_eq!(slab.num_wavefronts(), 10);

        // Free all
        for idx in &indices {
            slab.free(*idx);
        }
        assert_eq!(slab.num_free(), 10);

        // Reallocate — should reuse
        for _ in 0..10 {
            slab.allocate(-5, 5);
        }
        assert_eq!(slab.num_wavefronts(), 10);
        assert_eq!(slab.num_free(), 0);
    }

    #[test]
    fn test_reap() {
        let mut slab = WavefrontSlab::new(20, false, SlabMode::Reuse);
        for _ in 0..5 {
            slab.allocate(-5, 5);
        }
        slab.reap();
    }

    #[test]
    fn test_tight_mode() {
        let mut slab = WavefrontSlab::new(20, false, SlabMode::Tight);
        let idx = slab.allocate(-5, 5);
        slab.free(idx);

        let idx2 = slab.allocate(-3, 3);
        assert_eq!(idx2, idx);
    }

    #[test]
    fn test_tight_mode_oversized() {
        let mut slab = WavefrontSlab::new(10, false, SlabMode::Tight);

        let idx = slab.allocate(-10, 10);
        let wf = slab.get(idx);
        assert!(wf.wf_elements_allocated >= 21);

        slab.free(idx);
        assert_eq!(slab.num_free(), 0); // Deallocated, not free
    }

    #[test]
    fn test_with_backtrace() {
        let mut slab = WavefrontSlab::new(20, true, SlabMode::Reuse);
        let idx = slab.allocate(-5, 5);
        let wf = slab.get(idx);
        assert!(wf.has_backtrace());
    }

    #[test]
    fn test_clear() {
        let mut slab = WavefrontSlab::new(20, false, SlabMode::Reuse);
        for _ in 0..5 {
            slab.allocate(-5, 5);
        }
        slab.clear();
        // Arena clear drops all wavefronts and resets arena
        assert_eq!(slab.num_wavefronts(), 0);
        assert_eq!(slab.num_free(), 0);
        // Can allocate again from fresh arena
        let idx = slab.allocate(-5, 5);
        assert_eq!(slab.get(idx).status, WavefrontStatus::Busy);
    }
}
