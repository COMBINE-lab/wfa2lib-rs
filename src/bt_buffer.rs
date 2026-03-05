//! Backtrace buffer for storing backtrace blocks (packed CIGAR chains).
//!
//! The buffer uses segmented storage: large fixed-size segments of `BtBlock`s,
//! with new segments allocated on demand. Blocks form linked lists via `prev_idx`
//! pointers, allowing reconstruction of the full alignment CIGAR.

use crate::pcigar::Pcigar;
use crate::wavefront::BtBlockIdx;

/// Null block index sentinel.
pub const BT_BLOCK_IDX_NULL: BtBlockIdx = u32::MAX;

/// Size of each segment in blocks.
const BT_BUFFER_SEGMENT_LENGTH: usize = 1 << 23; // 8M blocks (~48MB per segment)

/// A single backtrace block.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BtBlock {
    /// Packed CIGAR for this block.
    pub pcigar: Pcigar,
    /// Index of the previous block in the chain.
    pub prev_idx: BtBlockIdx,
}

/// Initial position for a backtrace chain.
#[derive(Clone, Copy, Debug)]
pub struct BtInitPos {
    pub v: i32,
    pub h: i32,
}

/// Segmented backtrace buffer.
pub struct BacktraceBuffer {
    /// Current segment index.
    segment_idx: usize,
    /// Current offset within the current segment.
    segment_offset: usize,
    /// Memory segments.
    segments: Vec<Vec<BtBlock>>,
    /// Initial positions for alignment chains.
    pub alignment_init_pos: Vec<BtInitPos>,
    /// Total compacted blocks (dense from 0..num_compacted_blocks-1).
    pub num_compacted_blocks: BtBlockIdx,
    /// Total compactions performed.
    pub num_compactions: u32,
    /// Temporal buffer for packed alignment (pcigar values, front to back).
    pub alignment_packed: Vec<Pcigar>,
}

impl BacktraceBuffer {
    /// Create a new backtrace buffer.
    pub fn new() -> Self {
        let initial_segment = vec![
            BtBlock {
                pcigar: 0,
                prev_idx: BT_BLOCK_IDX_NULL,
            };
            BT_BUFFER_SEGMENT_LENGTH
        ];
        Self {
            segment_idx: 0,
            segment_offset: 0,
            segments: vec![initial_segment],
            alignment_init_pos: Vec::with_capacity(100),
            num_compacted_blocks: 0,
            num_compactions: 0,
            alignment_packed: Vec::with_capacity(100),
        }
    }

    /// Clear the buffer for reuse (keeps first segment allocated).
    pub fn clear(&mut self) {
        self.segment_idx = 0;
        self.segment_offset = 0;
        self.num_compacted_blocks = 0;
        self.num_compactions = 0;
        self.alignment_init_pos.clear();
    }

    /// Reap: free all segments beyond the first, reset.
    pub fn reap(&mut self) {
        self.segments.truncate(1);
        self.clear();
    }

    /// Get the current global block index (total blocks used).
    pub fn get_used(&self) -> u64 {
        (self.segment_idx * BT_BUFFER_SEGMENT_LENGTH + self.segment_offset) as u64
    }

    /// Get total allocated memory in bytes.
    pub fn get_size_allocated(&self) -> u64 {
        (self.segments.len() * BT_BUFFER_SEGMENT_LENGTH) as u64
            * std::mem::size_of::<BtBlock>() as u64
    }

    /// Get used memory in bytes.
    pub fn get_size_used(&self) -> u64 {
        self.get_used() * std::mem::size_of::<BtBlock>() as u64
    }

    /// Get a block by global index.
    #[inline(always)]
    pub fn get_block(&self, block_idx: BtBlockIdx) -> BtBlock {
        let seg_idx = block_idx as usize / BT_BUFFER_SEGMENT_LENGTH;
        let seg_off = block_idx as usize % BT_BUFFER_SEGMENT_LENGTH;
        self.segments[seg_idx][seg_off]
    }

    /// Set a block at a global index.
    #[inline(always)]
    pub fn set_block(&mut self, block_idx: BtBlockIdx, block: BtBlock) {
        let seg_idx = block_idx as usize / BT_BUFFER_SEGMENT_LENGTH;
        let seg_off = block_idx as usize % BT_BUFFER_SEGMENT_LENGTH;
        self.segments[seg_idx][seg_off] = block;
    }

    /// Store a block and return its global index.
    pub fn store_block(&mut self, pcigar: Pcigar, prev_idx: BtBlockIdx) -> BtBlockIdx {
        let global_idx =
            (self.segment_idx * BT_BUFFER_SEGMENT_LENGTH + self.segment_offset) as BtBlockIdx;

        self.segments[self.segment_idx][self.segment_offset] = BtBlock { pcigar, prev_idx };

        self.segment_offset += 1;
        if self.segment_offset >= BT_BUFFER_SEGMENT_LENGTH {
            self.reserve_next_segment();
        }

        global_idx
    }

    /// Initialize a new backtrace chain with a starting position.
    /// Returns the global index of the initial block.
    pub fn init_block(&mut self, v: i32, h: i32) -> BtBlockIdx {
        let init_pos_offset = self.alignment_init_pos.len() as u32;
        self.alignment_init_pos.push(BtInitPos { v, h });
        self.store_block(init_pos_offset, BT_BLOCK_IDX_NULL)
    }

    /// Get the current write position and available space in the current segment.
    /// Returns (global_idx, segment_slice, available_count).
    pub fn get_mem(&mut self) -> (BtBlockIdx, &mut [BtBlock], usize) {
        let global_idx =
            (self.segment_idx * BT_BUFFER_SEGMENT_LENGTH + self.segment_offset) as BtBlockIdx;
        let available = BT_BUFFER_SEGMENT_LENGTH - self.segment_offset;
        let offset = self.segment_offset;
        let slice = &mut self.segments[self.segment_idx][offset..];
        (global_idx, slice, available)
    }

    /// Advance the write position by `used` blocks.
    pub fn add_used(&mut self, used: usize) {
        self.segment_offset += used;
        if self.segment_offset >= BT_BUFFER_SEGMENT_LENGTH {
            self.reserve_next_segment();
        }
    }

    /// Trace back from a block, collecting all pcigars into `alignment_packed`.
    /// Returns the initial block (the one with prev_idx == NULL).
    pub fn traceback_pcigar(&mut self, start_block_idx: BtBlockIdx) -> BtBlock {
        self.alignment_packed.clear();

        let mut block = self.get_block(start_block_idx);
        while block.prev_idx != BT_BLOCK_IDX_NULL {
            self.alignment_packed.push(block.pcigar);
            block = self.get_block(block.prev_idx);
        }

        // Return the initial block (contains init_pos index in pcigar field)
        block
    }

    /// Reset compaction state.
    pub fn reset_compaction(&mut self) {
        self.num_compactions = 0;
        self.num_compacted_blocks = 0;
    }

    // --- Internal ---

    fn reserve_next_segment(&mut self) {
        self.segment_offset = 0;
        self.segment_idx += 1;

        if self.segment_idx >= self.segments.len() {
            // Check we haven't exceeded max addressable
            let block_idx = ((self.segment_idx + 1) * BT_BUFFER_SEGMENT_LENGTH) as u64;
            assert!(
                block_idx < u32::MAX as u64,
                "BacktraceBuffer: reached maximum addressable index"
            );

            self.segments.push(vec![
                BtBlock {
                    pcigar: 0,
                    prev_idx: BT_BLOCK_IDX_NULL,
                };
                BT_BUFFER_SEGMENT_LENGTH
            ]);
        }
    }
}

impl Default for BacktraceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = BacktraceBuffer::new();
        assert_eq!(buf.get_used(), 0);
        assert_eq!(buf.segments.len(), 1);
    }

    #[test]
    fn test_store_and_get_block() {
        let mut buf = BacktraceBuffer::new();
        let idx = buf.store_block(42, BT_BLOCK_IDX_NULL);
        assert_eq!(idx, 0);

        let block = buf.get_block(idx);
        let pcigar = block.pcigar;
        let prev_idx = block.prev_idx;
        assert_eq!(pcigar, 42);
        assert_eq!(prev_idx, BT_BLOCK_IDX_NULL);
        assert_eq!(buf.get_used(), 1);
    }

    #[test]
    fn test_init_block() {
        let mut buf = BacktraceBuffer::new();
        let idx = buf.init_block(10, 20);

        assert_eq!(buf.alignment_init_pos.len(), 1);
        assert_eq!(buf.alignment_init_pos[0].v, 10);
        assert_eq!(buf.alignment_init_pos[0].h, 20);

        let block = buf.get_block(idx);
        let pcigar = block.pcigar;
        let prev_idx = block.prev_idx;
        assert_eq!(pcigar, 0); // init_pos_offset = 0
        assert_eq!(prev_idx, BT_BLOCK_IDX_NULL);
    }

    #[test]
    fn test_chain_and_traceback() {
        let mut buf = BacktraceBuffer::new();

        // Create a chain: init -> block1 -> block2
        let init_idx = buf.init_block(0, 0);
        let block1_idx = buf.store_block(100, init_idx);
        let block2_idx = buf.store_block(200, block1_idx);

        // Traceback from block2
        let init_block = buf.traceback_pcigar(block2_idx);

        // alignment_packed should have [200, 100] (most recent first)
        assert_eq!(buf.alignment_packed.len(), 2);
        assert_eq!(buf.alignment_packed[0], 200);
        assert_eq!(buf.alignment_packed[1], 100);

        // Init block's pcigar is the init_pos_offset
        let pcigar = init_block.pcigar;
        let prev_idx = init_block.prev_idx;
        assert_eq!(pcigar, 0);
        assert_eq!(prev_idx, BT_BLOCK_IDX_NULL);
    }

    #[test]
    fn test_clear() {
        let mut buf = BacktraceBuffer::new();
        buf.store_block(1, BT_BLOCK_IDX_NULL);
        buf.store_block(2, 0);

        buf.clear();
        assert_eq!(buf.get_used(), 0);
        assert_eq!(buf.num_compacted_blocks, 0);
    }

    #[test]
    fn test_reap() {
        let mut buf = BacktraceBuffer::new();
        // Store enough to potentially trigger segment growth
        for i in 0..100 {
            buf.store_block(i, BT_BLOCK_IDX_NULL);
        }

        buf.reap();
        assert_eq!(buf.segments.len(), 1);
        assert_eq!(buf.get_used(), 0);
    }

    #[test]
    fn test_get_mem() {
        let mut buf = BacktraceBuffer::new();
        let (idx, _slice, available) = buf.get_mem();
        assert_eq!(idx, 0);
        assert_eq!(available, BT_BUFFER_SEGMENT_LENGTH);
    }

    #[test]
    fn test_add_used() {
        let mut buf = BacktraceBuffer::new();
        buf.add_used(5);
        assert_eq!(buf.get_used(), 5);
    }

    #[test]
    fn test_memory_sizes() {
        let buf = BacktraceBuffer::new();
        assert!(buf.get_size_allocated() > 0);
        assert_eq!(buf.get_size_used(), 0);
    }
}
