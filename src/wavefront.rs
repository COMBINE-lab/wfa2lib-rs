//! Individual wavefront data structure.
//!
//! A wavefront stores offsets along a range of diagonals [lo, hi].
//! Offsets are k-centered: internally stored in a raw buffer but accessed
//! via diagonal index `k` using a base offset.
//!
//! Uses raw pointers instead of `Vec` to minimize struct size (56 bytes)
//! for better cache utilization when accessing many wavefronts per step.

use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::ptr::null_mut;

use crate::arena::ByteArena;
use crate::offset::{OFFSET_NULL, WfOffset};
use crate::pcigar::{PCIGAR_NULL, Pcigar};

/// Index into the backtrace buffer.
pub type BtBlockIdx = u32;

/// Null backtrace block index.
pub const BT_BLOCK_IDX_NULL: BtBlockIdx = u32::MAX;

/// Alignment position in the wavefront.
#[derive(Debug, Clone, Copy, Default)]
pub struct WavefrontPos {
    pub score: i32,
    pub k: i32,
    pub offset: WfOffset,
}

/// Status of a wavefront in the slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavefrontStatus {
    Free,
    Busy,
    Deallocated,
}

/// A single wavefront storing offsets for diagonals [lo, hi].
///
/// Layout is optimized for cache: ~56 bytes fits in a single cache line.
/// Raw pointers replace `Vec`/`Option<Vec>` to eliminate fat-pointer overhead.
pub struct Wavefront {
    // Pointers first (8-byte aligned)
    /// Raw offset storage buffer.
    pub(crate) offsets_ptr: *mut WfOffset,
    /// Pre-centered offset pointer: `centered[k]` accesses diagonal `k` directly.
    /// Equals `offsets_ptr.wrapping_offset(-(base_k as isize))`.
    /// Matches C's `offsets = offsets_mem - min_lo` pattern.
    pub(crate) centered_offsets: *mut WfOffset,
    /// Backtrace pcigar values (null if no backtrace).
    bt_pcigar_ptr: *mut Pcigar,
    /// Backtrace previous-index values (null if no backtrace).
    bt_prev_ptr: *mut BtBlockIdx,

    // Dimensions
    /// Lowest diagonal (inclusive).
    pub lo: i32,
    /// Highest diagonal (inclusive).
    pub hi: i32,
    /// The k-value corresponding to offsets_ptr[0].
    base_k: i32,

    // Slab/metadata
    /// Total elements allocated (max wavefront size).
    pub wf_elements_allocated: i32,
    /// Minimum diagonal element initialized (inclusive).
    pub wf_elements_init_min: i32,
    /// Maximum diagonal element initialized (inclusive).
    pub wf_elements_init_max: i32,
    /// Maximum pcigar-ops stored in any backtrace block.
    pub bt_occupancy_max: i32,

    /// Whether this is a null (empty/sentinel) wavefront.
    pub null: bool,
    /// Memory state of this wavefront.
    pub status: WavefrontStatus,
    /// Whether offset array is arena-backed (skip dealloc on drop).
    arena_backed: bool,
    /// Whether bt_pcigar and bt_prev arrays are arena-backed (skip dealloc on drop).
    /// True when allocated via `allocate_from_arena` with `allocate_backtrace=true`.
    bt_arena_backed: bool,
    // Layout: 4×ptr(32B) + 7×i32(28B) + 4×bool(4B) = 64B = 1 cache line.
}

// Raw pointers are not Send/Sync by default, but our wavefronts are
// single-owner (never shared across threads).
unsafe impl Send for Wavefront {}

impl Drop for Wavefront {
    fn drop(&mut self) {
        let size = self.wf_elements_allocated as usize;
        // Arena-backed offsets are freed when the arena resets — skip dealloc.
        if !self.arena_backed && size > 0 && !self.offsets_ptr.is_null() {
            unsafe {
                let layout = Layout::array::<WfOffset>(size).unwrap();
                dealloc(self.offsets_ptr as *mut u8, layout);
            }
        }
        // Arena-backed bt arrays are freed when the arena resets — skip dealloc.
        if !self.bt_arena_backed && !self.bt_pcigar_ptr.is_null() {
            unsafe {
                let layout = Layout::array::<Pcigar>(size).unwrap();
                dealloc(self.bt_pcigar_ptr as *mut u8, layout);
                let layout = Layout::array::<BtBlockIdx>(size).unwrap();
                dealloc(self.bt_prev_ptr as *mut u8, layout);
            }
        }
    }
}

impl Wavefront {
    /// Allocate a new wavefront with the given number of elements.
    ///
    /// # Safety invariant
    /// The offsets array is left uninitialized for performance. Callers must
    /// ensure all positions are written (by compute kernels) or filled (by
    /// `init_null`/`init_ends_lower`/`init_ends_higher`) before being read.
    /// The slab allocator + compute pipeline guarantee this invariant.
    pub fn allocate(wf_elements_allocated: i32, allocate_backtrace: bool) -> Self {
        let size = wf_elements_allocated as usize;
        // SAFETY: offsets buffer is not read until init_null/init_ends fills
        // the relevant range, or the compute kernel writes every position.
        let offsets_ptr = unsafe {
            let layout = Layout::array::<WfOffset>(size).unwrap();
            alloc(layout) as *mut WfOffset
        };
        // bt_pcigar (PCIGAR_NULL=0) and bt_prev (0) use zero-init.
        let (bt_pcigar_ptr, bt_prev_ptr) = if allocate_backtrace {
            unsafe {
                let p = alloc_zeroed(Layout::array::<Pcigar>(size).unwrap()) as *mut Pcigar;
                let b = alloc_zeroed(Layout::array::<BtBlockIdx>(size).unwrap()) as *mut BtBlockIdx;
                (p, b)
            }
        } else {
            (null_mut(), null_mut())
        };
        Self {
            offsets_ptr,
            centered_offsets: offsets_ptr, // base_k=0, so centered = offsets_ptr
            bt_pcigar_ptr,
            bt_prev_ptr,
            null: false,
            lo: 1,
            hi: -1,
            base_k: 0,
            bt_occupancy_max: 0,
            status: WavefrontStatus::Free,
            wf_elements_allocated,
            wf_elements_init_min: 0,
            wf_elements_init_max: 0,
            arena_backed: false,
            bt_arena_backed: false,
        }
    }

    /// Allocate a wavefront with all arrays from the arena (bump allocator).
    ///
    /// All three arrays (offsets, bt_pcigar, bt_prev) live in the arena's
    /// contiguous buffer. None are zeroed — matching C's `mm_allocator_calloc`
    /// with `zero_mem=false`. Correctness is maintained because:
    /// - Offset positions [lo,hi] are always written by compute kernels before read.
    /// - bt positions [lo,hi] are always written by CIGAR compute kernels before
    ///   the backtrace reads them. Positions outside [lo,hi] are never accessed.
    /// - `init_null()` and `init_ends_lower/higher()` explicitly fill any
    ///   null/extended positions including their bt values.
    pub fn allocate_from_arena(
        arena: &mut ByteArena,
        wf_elements_allocated: i32,
        allocate_backtrace: bool,
    ) -> Self {
        let size = wf_elements_allocated as usize;
        let offsets_ptr = arena.alloc_bytes(size * 4) as *mut WfOffset;
        let (bt_pcigar_ptr, bt_prev_ptr) = if allocate_backtrace {
            // No zeroing needed — compute kernels write before backtrace reads.
            let p = arena.alloc_bytes(size * 4) as *mut Pcigar;
            let b = arena.alloc_bytes(size * 4) as *mut BtBlockIdx;
            (p, b)
        } else {
            (null_mut(), null_mut())
        };
        Self {
            offsets_ptr,
            centered_offsets: offsets_ptr, // base_k=0, so centered = offsets_ptr
            bt_pcigar_ptr,
            bt_prev_ptr,
            null: false,
            lo: 1,
            hi: -1,
            base_k: 0,
            bt_occupancy_max: 0,
            status: WavefrontStatus::Free,
            wf_elements_allocated,
            wf_elements_init_min: 0,
            wf_elements_init_max: 0,
            arena_backed: true,
            bt_arena_backed: allocate_backtrace,
        }
    }

    /// Initialize a `Wavefront` at an arena-allocated address.
    ///
    /// `ptr` must point to a block allocated by `ByteArena::alloc_bytes()` with
    /// at least `size_of::<Wavefront>() + wf_elements_allocated * size_of::<WfOffset>()`
    /// bytes and 8-byte alignment. The offset array immediately follows the struct
    /// at `ptr.add(1) as *mut WfOffset`.
    ///
    /// # Safety
    /// `ptr` must be valid, writable, 8-byte aligned, and followed by
    /// at least `wf_elements_allocated` `WfOffset` (i32) slots.
    pub unsafe fn new_at_ptr(
        ptr: *mut Wavefront,
        wf_elements_allocated: i32,
        allocate_backtrace: bool,
    ) {
        let size = wf_elements_allocated as usize;
        // Offset array immediately follows the struct in the same arena block.
        let offsets_ptr = unsafe { ptr.add(1) as *mut WfOffset };
        let (bt_pcigar_ptr, bt_prev_ptr) = if allocate_backtrace {
            unsafe {
                let p = alloc_zeroed(Layout::array::<Pcigar>(size).unwrap()) as *mut Pcigar;
                let b = alloc_zeroed(Layout::array::<BtBlockIdx>(size).unwrap()) as *mut BtBlockIdx;
                (p, b)
            }
        } else {
            (null_mut(), null_mut())
        };
        unsafe {
            ptr.write(Wavefront {
                offsets_ptr,
                centered_offsets: offsets_ptr, // base_k=0, so centered = offsets_ptr
                bt_pcigar_ptr,
                bt_prev_ptr,
                null: false,
                lo: 1,
                hi: -1,
                base_k: 0,
                bt_occupancy_max: 0,
                status: WavefrontStatus::Free,
                wf_elements_allocated,
                wf_elements_init_min: 0,
                wf_elements_init_max: 0,
                arena_backed: true,
                bt_arena_backed: false, // new_at_ptr keeps bt as heap (or null)
            });
        }
    }

    /// Resize the wavefront (content is lost). Heap-allocated only.
    pub fn resize(&mut self, wf_elements_allocated: i32) {
        debug_assert!(!self.arena_backed, "arena-backed wavefronts cannot be resized in place");
        debug_assert!(!self.bt_arena_backed, "arena-backed bt wavefronts cannot be resized in place");
        let old_size = self.wf_elements_allocated as usize;
        let new_size = wf_elements_allocated as usize;
        self.wf_elements_allocated = wf_elements_allocated;
        unsafe {
            // Free old offsets
            if old_size > 0 && !self.offsets_ptr.is_null() {
                dealloc(self.offsets_ptr as *mut u8, Layout::array::<WfOffset>(old_size).unwrap());
            }
            // Allocate new offsets (uninitialized)
            self.offsets_ptr = alloc(Layout::array::<WfOffset>(new_size).unwrap()) as *mut WfOffset;
            // Reset centered pointer (base_k unchanged, will be set by next init())
            self.centered_offsets = self.offsets_ptr.wrapping_offset(-(self.base_k as isize));
            // Resize backtrace if present
            if !self.bt_pcigar_ptr.is_null() {
                dealloc(self.bt_pcigar_ptr as *mut u8, Layout::array::<Pcigar>(old_size).unwrap());
                dealloc(self.bt_prev_ptr as *mut u8, Layout::array::<BtBlockIdx>(old_size).unwrap());
                self.bt_pcigar_ptr = alloc_zeroed(Layout::array::<Pcigar>(new_size).unwrap()) as *mut Pcigar;
                self.bt_prev_ptr = alloc_zeroed(Layout::array::<BtBlockIdx>(new_size).unwrap()) as *mut BtBlockIdx;
            }
        }
    }

    /// Initialize the wavefront for a new alignment step.
    /// Sets base_k = min_lo (matching C's `offsets = offsets_mem - min_lo`).
    /// Accessible range is [min_lo, min_lo + wf_elements_allocated - 1].
    pub fn init(&mut self, min_lo: i32, max_hi: i32) {
        self.null = false;
        self.lo = 1;
        self.hi = -1;
        self.base_k = min_lo;
        self.centered_offsets = self.offsets_ptr.wrapping_offset(-(min_lo as isize));
        if !self.bt_pcigar_ptr.is_null() {
            self.bt_occupancy_max = 0;
        }
        self.wf_elements_init_min = 0;
        self.wf_elements_init_max = 0;
        debug_assert!(
            max_hi <= min_lo + self.wf_elements_allocated - 1,
            "max_hi={} exceeds allocated_max={} (base_k={}, alloc={})",
            max_hi,
            min_lo + self.wf_elements_allocated - 1,
            self.base_k,
            self.wf_elements_allocated
        );
    }

    /// Initialize as a null wavefront (all offsets set to OFFSET_NULL).
    pub fn init_null(&mut self, min_lo: i32, max_hi: i32) {
        self.null = true;
        self.lo = 1;
        self.hi = -1;
        self.base_k = min_lo;
        self.centered_offsets = self.offsets_ptr.wrapping_offset(-(min_lo as isize));
        let wf_elements = (max_hi - min_lo + 1) as usize;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.offsets_ptr, wf_elements);
            slice.fill(OFFSET_NULL);
            if !self.bt_pcigar_ptr.is_null() {
                self.bt_occupancy_max = 0;
                let pcigar_slice = std::slice::from_raw_parts_mut(self.bt_pcigar_ptr, wf_elements);
                pcigar_slice.fill(PCIGAR_NULL);
                let prev_slice = std::slice::from_raw_parts_mut(self.bt_prev_ptr, wf_elements);
                prev_slice.fill(0);
            }
        }
        self.wf_elements_init_min = min_lo;
        self.wf_elements_init_max = max_hi;
    }

    /// Initialize as a victim wavefront (for writing that gets discarded).
    pub fn init_victim(&mut self, min_lo: i32, max_hi: i32) {
        self.init(min_lo, max_hi);
        self.null = true;
    }

    /// Set the effective diagonal limits.
    pub fn set_limits(&mut self, lo: i32, hi: i32) {
        self.lo = lo;
        self.hi = hi;
        self.wf_elements_init_min = lo;
        self.wf_elements_init_max = hi;
    }

    /// Extend the initialized range downward to `min_lo`.
    /// Fills positions [min_lo, wf_elements_init_min) with OFFSET_NULL.
    /// Only initializes within the allocated range.
    #[inline(always)]
    pub fn init_ends_lower(&mut self, min_lo: i32) {
        if self.wf_elements_init_min <= min_lo {
            return;
        }
        let min_init = self.wf_elements_init_min.min(self.lo);
        let allocated_min = self.base_k;
        let start = min_lo.max(allocated_min);
        let count = (min_init - start) as usize;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                self.centered_offsets.offset(start as isize),
                count,
            );
            slice.fill(OFFSET_NULL);
        }
        self.wf_elements_init_min = start;
    }

    /// Extend the initialized range upward to `max_hi`.
    /// Fills positions (wf_elements_init_max, max_hi] with OFFSET_NULL.
    /// Only initializes within the allocated range.
    #[inline(always)]
    pub fn init_ends_higher(&mut self, max_hi: i32) {
        if self.wf_elements_init_max >= max_hi {
            return;
        }
        let max_init = self.wf_elements_init_max.max(self.hi);
        let allocated_max = self.base_k + self.wf_elements_allocated - 1;
        let end = max_hi.min(allocated_max);
        let start = max_init + 1;
        let count = (end - max_init) as usize;
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                self.centered_offsets.offset(start as isize),
                count,
            );
            slice.fill(OFFSET_NULL);
        }
        self.wf_elements_init_max = end;
    }

    // --- K-centered accessors ---

    /// Get the offset at diagonal `k`.
    #[inline(always)]
    pub fn get_offset(&self, k: i32) -> WfOffset {
        unsafe { *self.centered_offsets.offset(k as isize) }
    }

    /// Set the offset at diagonal `k`.
    #[inline(always)]
    pub fn set_offset(&mut self, k: i32, value: WfOffset) {
        unsafe { *self.centered_offsets.offset(k as isize) = value; }
    }

    /// Set the offset at diagonal `k` if `k` is within the allocated range.
    /// No-op if `k` is outside the allocated bounds.
    #[inline(always)]
    pub fn set_offset_if_allocated(&mut self, k: i32, value: WfOffset) {
        let allocated_min = self.base_k;
        let allocated_max = self.base_k + self.wf_elements_allocated - 1;
        if k >= allocated_min && k <= allocated_max {
            unsafe { *self.centered_offsets.offset(k as isize) = value; }
        }
    }

    /// Get the bt_pcigar at diagonal `k`.
    #[inline(always)]
    pub fn get_bt_pcigar(&self, k: i32) -> Pcigar {
        debug_assert!(!self.bt_pcigar_ptr.is_null());
        unsafe { *self.bt_pcigar_ptr.add((k - self.base_k) as usize) }
    }

    /// Set the bt_pcigar at diagonal `k`.
    #[inline(always)]
    pub fn set_bt_pcigar(&mut self, k: i32, value: Pcigar) {
        debug_assert!(!self.bt_pcigar_ptr.is_null());
        unsafe { *self.bt_pcigar_ptr.add((k - self.base_k) as usize) = value; }
    }

    /// Get the bt_prev at diagonal `k`.
    #[inline(always)]
    pub fn get_bt_prev(&self, k: i32) -> BtBlockIdx {
        debug_assert!(!self.bt_prev_ptr.is_null());
        unsafe { *self.bt_prev_ptr.add((k - self.base_k) as usize) }
    }

    /// Set the bt_prev at diagonal `k`.
    #[inline(always)]
    pub fn set_bt_prev(&mut self, k: i32, value: BtBlockIdx) {
        debug_assert!(!self.bt_prev_ptr.is_null());
        unsafe { *self.bt_prev_ptr.add((k - self.base_k) as usize) = value; }
    }

    /// Get the base k value for raw slice access.
    #[inline(always)]
    pub fn base_k(&self) -> i32 {
        self.base_k
    }

    /// Get a raw slice of the offsets (for performance-critical inner loops).
    #[inline(always)]
    pub fn offsets_slice(&self) -> &[WfOffset] {
        unsafe { std::slice::from_raw_parts(self.offsets_ptr, self.wf_elements_allocated as usize) }
    }

    /// Get a mutable raw slice of the offsets.
    #[inline(always)]
    pub fn offsets_slice_mut(&mut self) -> &mut [WfOffset] {
        unsafe { std::slice::from_raw_parts_mut(self.offsets_ptr, self.wf_elements_allocated as usize) }
    }

    /// Get a k-centered const pointer: `ptr[k]` accesses diagonal `k` directly.
    ///
    /// # Safety
    /// Only valid for `k` in `[base_k, base_k + wf_elements_allocated - 1]`.
    #[inline(always)]
    pub unsafe fn offsets_centered_ptr(&self) -> *const WfOffset {
        self.centered_offsets
    }

    /// Get a k-centered mutable pointer: `ptr[k]` accesses diagonal `k` directly.
    ///
    /// # Safety
    /// Only valid for `k` in `[base_k, base_k + wf_elements_allocated - 1]`.
    #[inline(always)]
    pub unsafe fn offsets_centered_mut_ptr(&mut self) -> *mut WfOffset {
        self.centered_offsets
    }

    /// Get a raw slice of bt_pcigar.
    #[inline(always)]
    pub fn bt_pcigar_slice(&self) -> Option<&[Pcigar]> {
        if self.bt_pcigar_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.bt_pcigar_ptr, self.wf_elements_allocated as usize) })
        }
    }

    /// Get a mutable raw slice of bt_pcigar.
    #[inline(always)]
    pub fn bt_pcigar_slice_mut(&mut self) -> Option<&mut [Pcigar]> {
        if self.bt_pcigar_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(self.bt_pcigar_ptr, self.wf_elements_allocated as usize) })
        }
    }

    /// Get a raw slice of bt_prev.
    #[inline(always)]
    pub fn bt_prev_slice(&self) -> Option<&[BtBlockIdx]> {
        if self.bt_prev_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.bt_prev_ptr, self.wf_elements_allocated as usize) })
        }
    }

    /// Get a mutable raw slice of bt_prev.
    #[inline(always)]
    pub fn bt_prev_slice_mut(&mut self) -> Option<&mut [BtBlockIdx]> {
        if self.bt_prev_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts_mut(self.bt_prev_ptr, self.wf_elements_allocated as usize) })
        }
    }

    /// Whether this wavefront has backtrace storage.
    #[inline(always)]
    pub fn has_backtrace(&self) -> bool {
        !self.bt_pcigar_ptr.is_null()
    }

    /// Get the memory size of this wavefront in bytes.
    pub fn get_size(&self) -> u64 {
        let mut total =
            (self.wf_elements_allocated as u64) * std::mem::size_of::<WfOffset>() as u64;
        if !self.bt_pcigar_ptr.is_null() {
            total += (self.wf_elements_allocated as u64)
                * (std::mem::size_of::<Pcigar>() + std::mem::size_of::<BtBlockIdx>()) as u64;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_size() {
        let size = std::mem::size_of::<Wavefront>();
        eprintln!("sizeof(Wavefront) = {size}");
        assert!(size <= 72, "Wavefront should fit in ~1 cache line, got {size}");
    }

    #[test]
    fn test_allocate_and_init() {
        let mut wf = Wavefront::allocate(10, false);
        wf.init(-3, 6);
        assert!(!wf.null);
        assert_eq!(wf.lo, 1);
        assert_eq!(wf.hi, -1);
        assert_eq!(wf.base_k(), -3);
        assert!(!wf.has_backtrace());
    }

    #[test]
    fn test_k_centered_access() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);

        wf.set_offset(-5, 10);
        wf.set_offset(0, 20);
        wf.set_offset(5, 30);

        assert_eq!(wf.get_offset(-5), 10);
        assert_eq!(wf.get_offset(0), 20);
        assert_eq!(wf.get_offset(5), 30);
    }

    #[test]
    fn test_init_null() {
        let mut wf = Wavefront::allocate(11, true);
        wf.init_null(-5, 5);
        assert!(wf.null);

        for k in -5..=5 {
            assert_eq!(wf.get_offset(k), OFFSET_NULL);
        }

        for k in -5..=5 {
            assert_eq!(wf.get_bt_pcigar(k), PCIGAR_NULL);
            assert_eq!(wf.get_bt_prev(k), 0);
        }
    }

    #[test]
    fn test_backtrace_access() {
        let mut wf = Wavefront::allocate(5, true);
        wf.init(0, 4);

        wf.set_bt_pcigar(2, 42);
        wf.set_bt_prev(2, 99);

        assert_eq!(wf.get_bt_pcigar(2), 42);
        assert_eq!(wf.get_bt_prev(2), 99);
    }

    #[test]
    fn test_set_limits() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);
        wf.set_limits(-2, 3);

        assert_eq!(wf.lo, -2);
        assert_eq!(wf.hi, 3);
        assert_eq!(wf.wf_elements_init_min, -2);
        assert_eq!(wf.wf_elements_init_max, 3);
    }

    #[test]
    fn test_resize() {
        let mut wf = Wavefront::allocate(5, true);
        wf.init(0, 4);
        wf.set_offset(2, 100);

        wf.resize(20);
        assert_eq!(wf.wf_elements_allocated, 20);
    }

    #[test]
    fn test_get_size() {
        let wf = Wavefront::allocate(100, false);
        assert_eq!(wf.get_size(), 100 * 4);

        let wf_bt = Wavefront::allocate(100, true);
        assert_eq!(wf_bt.get_size(), 100 * (4 + 4 + 4));
    }

    #[test]
    fn test_init_victim() {
        let mut wf = Wavefront::allocate(5, false);
        wf.init_victim(-2, 2);
        assert!(wf.null);
        assert_eq!(wf.lo, 1);
        assert_eq!(wf.hi, -1);
    }

    #[test]
    fn test_raw_slice_access() {
        let mut wf = Wavefront::allocate(11, false);
        wf.init(-5, 5);

        let base = wf.base_k();
        let slice = wf.offsets_slice_mut();
        slice[(3 - base) as usize] = 42;

        assert_eq!(wf.get_offset(3), 42);
    }

    #[test]
    fn test_new_at_ptr() {
        use std::alloc::{Layout, alloc, dealloc};
        use std::mem::size_of;
        let n: usize = 11;
        // Allocate a block large enough for Wavefront + n WfOffsets, 8-aligned.
        let block_size = size_of::<Wavefront>() + n * size_of::<WfOffset>();
        let block_size_aligned = (block_size + 7) & !7;
        let layout = Layout::from_size_align(block_size_aligned, 8).unwrap();
        let raw = unsafe { alloc(layout) } as *mut Wavefront;
        assert!(!raw.is_null());

        unsafe { Wavefront::new_at_ptr(raw, n as i32, false); }

        let wf = unsafe { &mut *raw };
        assert_eq!(wf.wf_elements_allocated, n as i32);
        assert!(wf.arena_backed);
        assert!(!wf.null);
        // offsets_ptr should be immediately after the struct
        let expected_offsets = unsafe { raw.add(1) as *mut WfOffset };
        assert_eq!(wf.offsets_ptr, expected_offsets);

        // Can write/read offsets through the wavefront API
        wf.init(-5, 5);
        wf.set_offset(0, 42);
        assert_eq!(wf.get_offset(0), 42);

        // Leak: drop won't free offsets (arena_backed) or bt arrays (null).
        // Just free the backing block.
        unsafe { std::mem::forget(std::ptr::read(raw)); dealloc(raw as *mut u8, layout); }
    }
}
