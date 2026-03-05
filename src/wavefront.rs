//! Individual wavefront data structure.
//!
//! A wavefront stores offsets along a range of diagonals [lo, hi].
//! Offsets are k-centered: internally stored in a Vec but accessed
//! via diagonal index `k` using a base offset.

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
pub struct Wavefront {
    // Dimensions
    /// Whether this is a null (empty/sentinel) wavefront.
    pub null: bool,
    /// Lowest diagonal (inclusive).
    pub lo: i32,
    /// Highest diagonal (inclusive).
    pub hi: i32,

    // Wavefront elements (k-centered storage)
    /// The raw offset storage.
    offsets_mem: Vec<WfOffset>,
    /// The k-value corresponding to offsets_mem[0].
    base_k: i32,

    // Piggyback backtrace
    /// Maximum pcigar-ops stored in any backtrace block.
    pub bt_occupancy_max: i32,
    /// Backtrace pcigar values (k-centered, parallel to offsets).
    bt_pcigar_mem: Option<Vec<Pcigar>>,
    /// Backtrace previous-index values (k-centered, parallel to offsets).
    bt_prev_mem: Option<Vec<BtBlockIdx>>,

    // Slab internals
    /// Memory state of this wavefront.
    pub status: WavefrontStatus,
    /// Total elements allocated (max wavefront size).
    pub wf_elements_allocated: i32,
    /// Minimum diagonal element allocated.
    pub wf_elements_allocated_min: i32,
    /// Maximum diagonal element allocated.
    pub wf_elements_allocated_max: i32,
    /// Minimum diagonal element initialized (inclusive).
    pub wf_elements_init_min: i32,
    /// Maximum diagonal element initialized (inclusive).
    pub wf_elements_init_max: i32,
}

impl Wavefront {
    /// Allocate a new wavefront with the given number of elements.
    pub fn allocate(wf_elements_allocated: i32, allocate_backtrace: bool) -> Self {
        let size = wf_elements_allocated as usize;
        let bt_pcigar_mem = if allocate_backtrace {
            Some(vec![PCIGAR_NULL; size])
        } else {
            None
        };
        let bt_prev_mem = if allocate_backtrace {
            Some(vec![0u32; size])
        } else {
            None
        };
        Self {
            null: false,
            lo: 1,
            hi: -1,
            offsets_mem: vec![OFFSET_NULL; size],
            base_k: 0,
            bt_occupancy_max: 0,
            bt_pcigar_mem,
            bt_prev_mem,
            status: WavefrontStatus::Free,
            wf_elements_allocated,
            wf_elements_allocated_min: 0,
            wf_elements_allocated_max: 0,
            wf_elements_init_min: 0,
            wf_elements_init_max: 0,
        }
    }

    /// Resize the wavefront (content is lost).
    pub fn resize(&mut self, wf_elements_allocated: i32) {
        let size = wf_elements_allocated as usize;
        self.wf_elements_allocated = wf_elements_allocated;
        self.offsets_mem = vec![OFFSET_NULL; size];
        if let Some(ref mut bt_pcigar) = self.bt_pcigar_mem {
            *bt_pcigar = vec![PCIGAR_NULL; size];
            *self.bt_prev_mem.as_mut().unwrap() = vec![0u32; size];
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
        if self.bt_pcigar_mem.is_some() {
            self.bt_occupancy_max = 0;
        }
        self.wf_elements_allocated_min = min_lo;
        self.wf_elements_allocated_max = min_lo + self.wf_elements_allocated - 1;
        self.wf_elements_init_min = 0;
        self.wf_elements_init_max = 0;
        debug_assert!(
            max_hi <= self.wf_elements_allocated_max,
            "max_hi={} exceeds allocated_max={} (base_k={}, alloc={})",
            max_hi,
            self.wf_elements_allocated_max,
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
        let wf_elements = (max_hi - min_lo + 1) as usize;
        for i in 0..wf_elements {
            self.offsets_mem[i] = OFFSET_NULL;
        }
        if let Some(ref mut bt_pcigar) = self.bt_pcigar_mem {
            self.bt_occupancy_max = 0;
            bt_pcigar[..wf_elements].fill(PCIGAR_NULL);
            self.bt_prev_mem.as_mut().unwrap()[..wf_elements].fill(0);
        }
        self.wf_elements_allocated_min = min_lo;
        self.wf_elements_allocated_max = min_lo + self.wf_elements_allocated - 1;
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
    pub fn init_ends_lower(&mut self, min_lo: i32) {
        if self.wf_elements_init_min <= min_lo {
            return;
        }
        let min_init = self.wf_elements_init_min.min(self.lo);
        let start = min_lo.max(self.wf_elements_allocated_min);
        for k in start..min_init {
            self.offsets_mem[(k - self.base_k) as usize] = OFFSET_NULL;
        }
        self.wf_elements_init_min = start;
    }

    /// Extend the initialized range upward to `max_hi`.
    /// Fills positions (wf_elements_init_max, max_hi] with OFFSET_NULL.
    /// Only initializes within the allocated range.
    pub fn init_ends_higher(&mut self, max_hi: i32) {
        if self.wf_elements_init_max >= max_hi {
            return;
        }
        let max_init = self.wf_elements_init_max.max(self.hi);
        let end = max_hi.min(self.wf_elements_allocated_max);
        for k in (max_init + 1)..=end {
            self.offsets_mem[(k - self.base_k) as usize] = OFFSET_NULL;
        }
        self.wf_elements_init_max = end;
    }

    // --- K-centered accessors ---

    /// Get the offset at diagonal `k`.
    #[inline(always)]
    pub fn get_offset(&self, k: i32) -> WfOffset {
        self.offsets_mem[(k - self.base_k) as usize]
    }

    /// Set the offset at diagonal `k`.
    #[inline(always)]
    pub fn set_offset(&mut self, k: i32, value: WfOffset) {
        self.offsets_mem[(k - self.base_k) as usize] = value;
    }

    /// Set the offset at diagonal `k` if `k` is within the allocated range.
    /// No-op if `k` is outside the allocated bounds.
    #[inline(always)]
    pub fn set_offset_if_allocated(&mut self, k: i32, value: WfOffset) {
        if k >= self.wf_elements_allocated_min && k <= self.wf_elements_allocated_max {
            self.offsets_mem[(k - self.base_k) as usize] = value;
        }
    }

    /// Get the bt_pcigar at diagonal `k`.
    #[inline(always)]
    pub fn get_bt_pcigar(&self, k: i32) -> Pcigar {
        self.bt_pcigar_mem.as_ref().unwrap()[(k - self.base_k) as usize]
    }

    /// Set the bt_pcigar at diagonal `k`.
    #[inline(always)]
    pub fn set_bt_pcigar(&mut self, k: i32, value: Pcigar) {
        self.bt_pcigar_mem.as_mut().unwrap()[(k - self.base_k) as usize] = value;
    }

    /// Get the bt_prev at diagonal `k`.
    #[inline(always)]
    pub fn get_bt_prev(&self, k: i32) -> BtBlockIdx {
        self.bt_prev_mem.as_ref().unwrap()[(k - self.base_k) as usize]
    }

    /// Set the bt_prev at diagonal `k`.
    #[inline(always)]
    pub fn set_bt_prev(&mut self, k: i32, value: BtBlockIdx) {
        self.bt_prev_mem.as_mut().unwrap()[(k - self.base_k) as usize] = value;
    }

    /// Get the base k value for raw slice access.
    #[inline(always)]
    pub fn base_k(&self) -> i32 {
        self.base_k
    }

    /// Get a raw slice of the offsets (for performance-critical inner loops).
    #[inline(always)]
    pub fn offsets_slice(&self) -> &[WfOffset] {
        &self.offsets_mem
    }

    /// Get a mutable raw slice of the offsets.
    #[inline(always)]
    pub fn offsets_slice_mut(&mut self) -> &mut [WfOffset] {
        &mut self.offsets_mem
    }

    /// Get a raw slice of bt_pcigar.
    #[inline(always)]
    pub fn bt_pcigar_slice(&self) -> Option<&[Pcigar]> {
        self.bt_pcigar_mem.as_deref()
    }

    /// Get a mutable raw slice of bt_pcigar.
    #[inline(always)]
    pub fn bt_pcigar_slice_mut(&mut self) -> Option<&mut [Pcigar]> {
        self.bt_pcigar_mem.as_deref_mut()
    }

    /// Get a raw slice of bt_prev.
    #[inline(always)]
    pub fn bt_prev_slice(&self) -> Option<&[BtBlockIdx]> {
        self.bt_prev_mem.as_deref()
    }

    /// Get a mutable raw slice of bt_prev.
    #[inline(always)]
    pub fn bt_prev_slice_mut(&mut self) -> Option<&mut [BtBlockIdx]> {
        self.bt_prev_mem.as_deref_mut()
    }

    /// Whether this wavefront has backtrace storage.
    pub fn has_backtrace(&self) -> bool {
        self.bt_pcigar_mem.is_some()
    }

    /// Get the memory size of this wavefront in bytes.
    pub fn get_size(&self) -> u64 {
        let mut total =
            (self.wf_elements_allocated as u64) * std::mem::size_of::<WfOffset>() as u64;
        if self.bt_pcigar_mem.is_some() {
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

        // Set and get offsets at various diagonals
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

        // All offsets should be OFFSET_NULL
        for k in -5..=5 {
            assert_eq!(wf.get_offset(k), OFFSET_NULL);
        }

        // BT fields should be zeroed
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
        // Content is lost after resize
    }

    #[test]
    fn test_get_size() {
        let wf = Wavefront::allocate(100, false);
        assert_eq!(wf.get_size(), 100 * 4); // 4 bytes per WfOffset

        let wf_bt = Wavefront::allocate(100, true);
        assert_eq!(wf_bt.get_size(), 100 * (4 + 4 + 4)); // offset + pcigar + bt_prev
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
}
