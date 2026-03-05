//! Bump allocator for wavefront offset arrays.
//!
//! Provides heap-stable memory regions from which wavefront offset arrays are
//! allocated via pointer-bump. Uses a list of fixed-size chunks so that
//! existing pointers remain valid when new chunks are added (unlike a single
//! `Vec` that may reallocate). This eliminates per-wavefront malloc/free
//! overhead while keeping pointer stability.
//!
//! The arena resets between alignments (cursors back to 0) — no individual
//! frees needed. This matches C's `mm_allocator` pattern.

use std::alloc::{Layout, alloc, dealloc};

use crate::offset::WfOffset;

/// Bump allocator for wavefront offset arrays.
///
/// Allocations return raw pointers into heap-allocated chunks.
/// Chunks are never reallocated, so pointers remain valid until `drop`.
/// `reset()` reclaims all memory at once (between alignments) without freeing.
pub struct OffsetArena {
    /// Heap-allocated chunks. Each is a raw pointer + capacity.
    /// Pointers remain stable because chunks are never moved.
    chunks: Vec<(*mut WfOffset, usize)>,
    /// Index of the current chunk being allocated from.
    current_chunk: usize,
    /// Next free position within the current chunk.
    cursor: usize,
    /// Default size for new chunks.
    default_chunk_size: usize,
}

impl OffsetArena {
    /// Create a new arena with the given initial chunk capacity (in elements).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1024);
        let mut arena = Self {
            chunks: Vec::with_capacity(8),
            current_chunk: 0,
            cursor: 0,
            default_chunk_size: cap,
        };
        arena.add_chunk(cap);
        arena
    }

    /// Allocate `len` elements from the arena. Returns a raw mutable pointer
    /// to the start of the allocated region.
    ///
    /// If the current chunk can't fit the allocation, a new chunk is added.
    /// Existing pointers remain valid.
    #[inline]
    pub fn alloc(&mut self, len: usize) -> *mut WfOffset {
        let (chunk_ptr, chunk_cap) = self.chunks[self.current_chunk];
        let new_cursor = self.cursor + len;
        if new_cursor <= chunk_cap {
            let ptr = unsafe { chunk_ptr.add(self.cursor) };
            self.cursor = new_cursor;
            return ptr;
        }
        self.alloc_slow(len)
    }

    /// Slow path: current chunk is full, try next chunk or allocate new one.
    #[cold]
    fn alloc_slow(&mut self, len: usize) -> *mut WfOffset {
        // Try the next existing chunk (from a previous alignment that used more chunks)
        let next = self.current_chunk + 1;
        if next < self.chunks.len() {
            let (chunk_ptr, chunk_cap) = self.chunks[next];
            if len <= chunk_cap {
                self.current_chunk = next;
                self.cursor = len;
                return chunk_ptr;
            }
        }
        // Need a new chunk — at least as big as requested, or default size
        let new_cap = len.max(self.default_chunk_size);
        self.add_chunk(new_cap);
        self.current_chunk = self.chunks.len() - 1;
        self.cursor = len;
        self.chunks[self.current_chunk].0
    }

    /// Reset the arena for a new alignment. Reuse existing chunks.
    #[inline]
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.cursor = 0;
    }

    /// Current memory usage in bytes (approximate — counts used portion of current chunk
    /// plus all prior chunks).
    pub fn used_bytes(&self) -> usize {
        let mut total = 0;
        for i in 0..self.current_chunk {
            total += self.chunks[i].1;
        }
        total += self.cursor;
        total * std::mem::size_of::<WfOffset>()
    }

    /// Total capacity in bytes across all chunks.
    pub fn capacity_bytes(&self) -> usize {
        let total: usize = self.chunks.iter().map(|(_, cap)| *cap).sum();
        total * std::mem::size_of::<WfOffset>()
    }

    /// Allocate a new chunk and add it to the chunk list.
    fn add_chunk(&mut self, capacity: usize) {
        let layout = Layout::array::<WfOffset>(capacity).unwrap();
        let ptr = unsafe { alloc(layout) as *mut WfOffset };
        assert!(!ptr.is_null(), "arena chunk allocation failed");
        self.chunks.push((ptr, capacity));
    }
}

impl Drop for OffsetArena {
    fn drop(&mut self) {
        for &(ptr, cap) in &self.chunks {
            if !ptr.is_null() && cap > 0 {
                let layout = Layout::array::<WfOffset>(cap).unwrap();
                unsafe { dealloc(ptr as *mut u8, layout); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mut arena = OffsetArena::new(100);
        let p1 = arena.alloc(10);
        let p2 = arena.alloc(20);
        assert!(!p1.is_null());
        assert!(!p2.is_null());
        // p2 should be 10 elements after p1 (same chunk)
        assert_eq!(unsafe { p2.offset_from(p1) }, 10);
    }

    #[test]
    fn test_reset() {
        let mut arena = OffsetArena::new(100);
        arena.alloc(50);
        arena.reset();
        // Allocate again — should reuse the same chunk
        let p1 = arena.alloc(10);
        let base = arena.chunks[0].0;
        assert_eq!(p1, base);
    }

    #[test]
    fn test_cross_chunk_alloc() {
        let mut arena = OffsetArena::new(1024);
        // Fill first chunk
        let p1 = arena.alloc(1024);
        assert!(!p1.is_null());
        // Next alloc goes to a new chunk
        let p2 = arena.alloc(50);
        assert!(!p2.is_null());
        assert_eq!(arena.chunks.len(), 2);
        // p1 is still valid (stable pointer)
        unsafe {
            *p1 = 42;
            assert_eq!(*p1, 42);
        }
    }

    #[test]
    fn test_large_alloc() {
        let mut arena = OffsetArena::new(100);
        // Request more than default chunk size
        let p = arena.alloc(500);
        assert!(!p.is_null());
    }

    #[test]
    fn test_pointer_stability() {
        let mut arena = OffsetArena::new(50);
        let ptrs: Vec<*mut WfOffset> = (0..20).map(|_| arena.alloc(10)).collect();
        // Write to all pointers
        for (i, p) in ptrs.iter().enumerate() {
            unsafe { **p = i as WfOffset; }
        }
        // Read back — all should still be valid
        for (i, p) in ptrs.iter().enumerate() {
            assert_eq!(unsafe { **p }, i as WfOffset);
        }
    }

    #[test]
    fn test_reset_reuses_chunks() {
        let mut arena = OffsetArena::new(50);
        arena.alloc(50);
        arena.alloc(50); // triggers second chunk
        let num_chunks = arena.chunks.len();
        arena.reset();
        arena.alloc(50);
        arena.alloc(50);
        // Should reuse existing chunks, not create new ones
        assert_eq!(arena.chunks.len(), num_chunks);
    }

    #[test]
    fn test_write_read() {
        let mut arena = OffsetArena::new(100);
        let p = arena.alloc(5);
        unsafe {
            *p.add(0) = 42;
            *p.add(4) = 99;
            assert_eq!(*p.add(0), 42);
            assert_eq!(*p.add(4), 99);
        }
    }
}
