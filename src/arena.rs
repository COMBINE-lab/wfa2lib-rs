//! Byte-oriented bump allocator for wavefront offset arrays.
//!
//! Provides 4-byte-aligned (WfOffset-granularity) heap-stable memory regions
//! via pointer-bump from a list of fixed-size chunks. Existing pointers remain
//! valid when new chunks are added (unlike a single Vec that may reallocate).
//! `reset()` reclaims all memory at once (between alignments) without freeing.
//!
//! This matches C's `mm_allocator` pattern for offset array storage.

use std::alloc::{Layout, alloc, dealloc};

/// Byte-oriented bump allocator with 8-byte alignment.
///
/// All allocations are padded to an 8-byte boundary so that `Wavefront`
/// structs (which contain pointer fields) are always suitably aligned.
pub struct ByteArena {
    /// Heap-allocated chunks. Each is a raw `*mut u8` + capacity in bytes.
    /// Pointers remain stable because chunks are never moved.
    chunks: Vec<(*mut u8, usize)>,
    /// Index of the current chunk being allocated from.
    current_chunk: usize,
    /// Next free byte offset within the current chunk (always 8-aligned).
    cursor: usize,
    /// Default size in bytes for new chunks.
    default_chunk_size: usize,
}

// Raw pointer storage: ByteArena is single-owner and never shared.
unsafe impl Send for ByteArena {}

impl ByteArena {
    /// Create a new arena with the given initial chunk capacity (bytes).
    /// Actual capacity is rounded up to a multiple of 8 and at least 1024.
    pub fn new(capacity_bytes: usize) -> Self {
        let cap = (capacity_bytes.max(1024) + 7) & !7;
        let mut arena = Self {
            chunks: Vec::with_capacity(8),
            current_chunk: 0,
            cursor: 0,
            default_chunk_size: cap,
        };
        arena.add_chunk(cap);
        arena
    }

    /// Allocate `size` bytes at 8-byte alignment.
    ///
    /// Returns a raw `*mut u8` pointer. The returned pointer is valid until
    /// `drop` (chunks are never reallocated). `reset()` invalidates all
    /// pointers by reusing the same memory for the next alignment.
    ///
    /// `size` is rounded up to the next 8-byte boundary internally so the
    /// *next* allocation also starts 8-byte aligned.
    #[inline]
    pub fn alloc_bytes(&mut self, size: usize) -> *mut u8 {
        // Round size up to 8-byte boundary for alignment of next alloc.
        let size_aligned = (size + 7) & !7;
        let (chunk_ptr, chunk_cap) = self.chunks[self.current_chunk];
        let new_cursor = self.cursor + size_aligned;
        if new_cursor <= chunk_cap {
            let ptr = unsafe { chunk_ptr.add(self.cursor) };
            self.cursor = new_cursor;
            return ptr;
        }
        self.alloc_slow(size_aligned)
    }

    /// Slow path: current chunk is full, try next or allocate a new one.
    #[cold]
    fn alloc_slow(&mut self, size_aligned: usize) -> *mut u8 {
        let next = self.current_chunk + 1;
        if next < self.chunks.len() {
            let (chunk_ptr, chunk_cap) = self.chunks[next];
            if size_aligned <= chunk_cap {
                self.current_chunk = next;
                self.cursor = size_aligned;
                return chunk_ptr;
            }
        }
        let new_cap = size_aligned.max(self.default_chunk_size);
        self.add_chunk(new_cap);
        self.current_chunk = self.chunks.len() - 1;
        self.cursor = size_aligned;
        self.chunks[self.current_chunk].0
    }

    /// Reset the arena for a new alignment. Existing chunks are reused.
    #[inline]
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.cursor = 0;
    }

    /// Current memory used in bytes (approximate).
    pub fn used_bytes(&self) -> usize {
        let mut total = 0;
        for i in 0..self.current_chunk {
            total += self.chunks[i].1;
        }
        total += self.cursor;
        total
    }

    /// Total capacity in bytes across all chunks.
    pub fn capacity_bytes(&self) -> usize {
        self.chunks.iter().map(|(_, cap)| *cap).sum()
    }

    fn add_chunk(&mut self, capacity: usize) {
        // SAFETY: capacity is > 0; Layout is valid with 8-byte alignment.
        let layout = Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "arena chunk allocation failed");
        self.chunks.push((ptr, capacity));
    }
}

impl Drop for ByteArena {
    fn drop(&mut self) {
        for &(ptr, cap) in &self.chunks {
            if !ptr.is_null() && cap > 0 {
                let layout = Layout::from_size_align(cap, 8).unwrap();
                unsafe { dealloc(ptr, layout); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mut arena = ByteArena::new(1024);
        let p1 = arena.alloc_bytes(16);
        let p2 = arena.alloc_bytes(16);
        assert!(!p1.is_null());
        assert!(!p2.is_null());
        // p2 should be 16 bytes after p1 (same chunk)
        assert_eq!(unsafe { p2.offset_from(p1) }, 16);
    }

    #[test]
    fn test_alignment() {
        let mut arena = ByteArena::new(1024);
        // Alloc odd sizes — next should still be 8-aligned
        let p1 = arena.alloc_bytes(5);
        let p2 = arena.alloc_bytes(3);
        assert_eq!(p1 as usize % 8, 0);
        assert_eq!(p2 as usize % 8, 0);
        // p2 is 8 bytes after p1 (5 rounded up to 8)
        assert_eq!(unsafe { p2.offset_from(p1) }, 8);
    }

    #[test]
    fn test_reset() {
        let mut arena = ByteArena::new(1024);
        arena.alloc_bytes(512);
        arena.reset();
        let p1 = arena.alloc_bytes(16);
        let base = arena.chunks[0].0;
        assert_eq!(p1, base);
    }

    #[test]
    fn test_cross_chunk_alloc() {
        // Use an allocation larger than the initial chunk to force a new chunk.
        let mut arena = ByteArena::new(1024);
        let p1 = arena.alloc_bytes(1024);
        assert!(!p1.is_null());
        assert_eq!(arena.chunks.len(), 1);
        let p2 = arena.alloc_bytes(16);
        assert!(!p2.is_null());
        assert_eq!(arena.chunks.len(), 2);
        // p1 is still valid
        unsafe { *p1 = 42; assert_eq!(*p1, 42); }
    }

    #[test]
    fn test_large_alloc() {
        let mut arena = ByteArena::new(64);
        let p = arena.alloc_bytes(512);
        assert!(!p.is_null());
    }

    #[test]
    fn test_pointer_stability() {
        let mut arena = ByteArena::new(64);
        let ptrs: Vec<*mut u8> = (0..20).map(|_| arena.alloc_bytes(8)).collect();
        for (i, &p) in ptrs.iter().enumerate() {
            unsafe { *p = i as u8; }
        }
        for (i, &p) in ptrs.iter().enumerate() {
            assert_eq!(unsafe { *p }, i as u8);
        }
    }

    #[test]
    fn test_reset_reuses_chunks() {
        let mut arena = ByteArena::new(64);
        arena.alloc_bytes(64);
        arena.alloc_bytes(64); // triggers second chunk
        let num_chunks = arena.chunks.len();
        arena.reset();
        arena.alloc_bytes(64);
        arena.alloc_bytes(64);
        assert_eq!(arena.chunks.len(), num_chunks);
    }

    #[test]
    fn test_write_read() {
        let mut arena = ByteArena::new(256);
        let p = arena.alloc_bytes(32);
        unsafe {
            *(p as *mut u32) = 0xDEADBEEF;
            assert_eq!(*(p as *mut u32), 0xDEADBEEF);
        }
    }
}
