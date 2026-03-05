//! Wavefront backtrace: reconstructs alignment CIGAR from stored wavefronts.
//!
//! The "high memory" backtrace walks backwards through all M-wavefronts,
//! using a piggyback encoding trick (offset + type packed into i64) to
//! determine which operation (mismatch, insertion, deletion) was taken
//! at each step.

use crate::bialign::ComponentType;
use crate::cigar::Cigar;
use crate::components::WavefrontComponents;
use crate::offset::{OFFSET_NULL, wavefront_h, wavefront_v};
use crate::penalties::{DistanceMetric, WavefrontPenalties};

// Backtrace type constants (packed into low 4 bits of i64).
// Higher values win ties when offsets are equal.
const BACKTRACE_TYPE_BITS: u32 = 4;
const BACKTRACE_TYPE_MASK: i64 = 0xF;
const BT_I1_OPEN: i64 = 1;
const BT_I1_EXT: i64 = 2;
const BT_I2_OPEN: i64 = 3;
const BT_I2_EXT: i64 = 4;
const BT_D1_OPEN: i64 = 5;
const BT_D1_EXT: i64 = 6;
const BT_D2_OPEN: i64 = 7;
const BT_D2_EXT: i64 = 8;
const BT_M: i64 = 9;

#[inline(always)]
fn piggyback_set(offset: i32, bt_type: i64) -> i64 {
    ((offset as i64) << BACKTRACE_TYPE_BITS) | bt_type
}

#[inline(always)]
fn piggyback_get_offset(value: i64) -> i32 {
    (value >> BACKTRACE_TYPE_BITS) as i32
}

#[inline(always)]
fn piggyback_get_type(value: i64) -> i64 {
    value & BACKTRACE_TYPE_MASK
}

/// Look up mismatch predecessor: M-wavefront at (score, k), offset + 1.
#[inline(always)]
fn bt_misms(wf_components: &WavefrontComponents, score: i32, k: i32) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let m_ptr = wf_components.get_m_ptr(score as usize);
    if m_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*m_ptr };
    if wf.wf_elements_init_min <= k && k <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k) + 1, BT_M)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up insertion predecessor: M-wavefront at (score, k-1), offset + 1.
#[inline(always)]
fn bt_ins1_open(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let m_ptr = wf_components.get_m_ptr(score as usize);
    if m_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*m_ptr };
    if wf.wf_elements_init_min < k && k - 1 <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k - 1) + 1, BT_I1_OPEN)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up deletion predecessor: M-wavefront at (score, k+1), offset unchanged.
#[inline(always)]
fn bt_del1_open(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let m_ptr = wf_components.get_m_ptr(score as usize);
    if m_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*m_ptr };
    if wf.wf_elements_init_min <= k + 1 && k < wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k + 1), BT_D1_OPEN)
    } else {
        OFFSET_NULL as i64
    }
}

/// Write `num_matches` 'M' operations into the CIGAR (backwards).
#[inline(always)]
fn write_matches(cigar: &mut Cigar, num_matches: i32) {
    if num_matches <= 0 {
        return;
    }
    let n = num_matches as usize;
    let start = cigar.begin_offset + 1 - n;
    cigar.operations[start..cigar.begin_offset + 1].fill(b'M');
    cigar.begin_offset = start - 1;
}

/// Perform backtrace for linear (edit/indel) distance, building a CIGAR.
///
/// Walks backwards through all stored M-wavefronts from the alignment endpoint
/// to reconstruct the full alignment. The CIGAR is built from end to beginning.
#[allow(clippy::too_many_arguments)]
pub fn backtrace_linear(
    wf_components: &WavefrontComponents,
    penalties: &WavefrontPenalties,
    pattern_length: i32,
    text_length: i32,
    alignment_score: i32,
    alignment_k: i32,
    alignment_offset: i32,
    cigar: &mut Cigar,
) {
    let distance_metric = penalties.distance_metric;

    // Prepare cigar — write backwards from the end of the buffer
    let max_ops = cigar.operations.len();
    cigar.end_offset = max_ops;
    cigar.begin_offset = max_ops - 1;

    let mut score = alignment_score;
    let mut k = alignment_k;
    let mut offset = alignment_offset;
    let mut v = wavefront_v(k, offset);
    let mut h = wavefront_h(k, offset);

    // Account for ending insertions/deletions (ends-free)
    if v < pattern_length {
        for _ in 0..(pattern_length - v) {
            cigar.operations[cigar.begin_offset] = b'D';
            cigar.begin_offset -= 1;
        }
    }
    if h < text_length {
        for _ in 0..(text_length - h) {
            cigar.operations[cigar.begin_offset] = b'I';
            cigar.begin_offset -= 1;
        }
    }

    // Main backtrace loop
    while v > 0 && h > 0 && score > 0 {
        // Compute predecessor scores
        let mismatch_score = score - penalties.mismatch;
        let gap_score = score - penalties.gap_opening1;

        // Look up predecessor offsets (packed with type)
        let misms = if distance_metric != DistanceMetric::Indel {
            bt_misms(wf_components, mismatch_score, k)
        } else {
            OFFSET_NULL as i64
        };
        let ins = bt_ins1_open(wf_components, gap_score, k);
        let del = bt_del1_open(wf_components, gap_score, k);

        // Take max — higher type wins ties (M > D > I)
        let max_all = misms.max(ins.max(del));

        if max_all < 0 {
            break; // No valid predecessor
        }

        // Emit matches (gap between current offset and predecessor offset)
        let max_offset = piggyback_get_offset(max_all);
        let num_matches = offset - max_offset;
        write_matches(cigar, num_matches);
        offset = max_offset;

        // Check boundary
        v = wavefront_v(k, offset);
        h = wavefront_h(k, offset);
        if v <= 0 || h <= 0 {
            break;
        }

        // Emit the operation
        let bt_type = piggyback_get_type(max_all);
        match bt_type {
            BT_M => {
                score = mismatch_score;
                cigar.operations[cigar.begin_offset] = b'X';
                cigar.begin_offset -= 1;
                offset -= 1;
            }
            BT_I1_OPEN => {
                score = gap_score;
                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;
                k -= 1;
                offset -= 1;
            }
            BT_D1_OPEN => {
                score = gap_score;
                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;
                k += 1;
            }
            _ => panic!("Invalid backtrace type: {}", bt_type),
        }

        // Update coordinates
        v = wavefront_v(k, offset);
        h = wavefront_h(k, offset);
    }

    // Account for beginning matches
    if v > 0 && h > 0 {
        let num_matches = v.min(h);
        write_matches(cigar, num_matches);
        v -= num_matches;
        h -= num_matches;
    }

    // Account for beginning insertions/deletions
    while v > 0 {
        cigar.operations[cigar.begin_offset] = b'D';
        cigar.begin_offset -= 1;
        v -= 1;
    }
    while h > 0 {
        cigar.operations[cigar.begin_offset] = b'I';
        cigar.begin_offset -= 1;
        h -= 1;
    }

    // Adjust to point to first valid operation
    cigar.begin_offset += 1;
    cigar.score = alignment_score;
}

// --- Affine backtrace helpers ---

/// Look up I1-wavefront at (score, k): returns piggyback(offset, BT_I1_EXT).
#[inline(always)]
fn bt_i1_ext(wf_components: &WavefrontComponents, score: i32, k: i32) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let i1_ptr = wf_components.get_i1_ptr(score as usize);
    if i1_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*i1_ptr };
    if wf.wf_elements_init_min <= k && k <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k), BT_I1_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up D1-wavefront at (score, k): returns piggyback(offset, BT_D1_EXT).
#[inline(always)]
fn bt_d1_ext(wf_components: &WavefrontComponents, score: i32, k: i32) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let d1_ptr = wf_components.get_d1_ptr(score as usize);
    if d1_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*d1_ptr };
    if wf.wf_elements_init_min <= k && k <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k), BT_D1_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up both M→I1 open (k-1, offset+1) and M→D1 open (k+1, offset) from
/// the same M-wavefront at the given score. Returns (ins1_open, del1_open).
/// This avoids a redundant get_m_ptr double-fetch.
#[inline(always)]
fn bt_open1_pair(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> (i64, i64) {
    if score < 0 {
        return (OFFSET_NULL as i64, OFFSET_NULL as i64);
    }
    let m_ptr = wf_components.get_m_ptr(score as usize);
    if m_ptr.is_null() {
        return (OFFSET_NULL as i64, OFFSET_NULL as i64);
    }
    let wf = unsafe { &*m_ptr };
    let min = wf.wf_elements_init_min;
    let max = wf.wf_elements_init_max;

    let ins = if min < k && k - 1 <= max {
        piggyback_set(wf.get_offset(k - 1) + 1, BT_I1_OPEN)
    } else {
        OFFSET_NULL as i64
    };
    let del = if min <= k + 1 && k < max {
        piggyback_set(wf.get_offset(k + 1), BT_D1_OPEN)
    } else {
        OFFSET_NULL as i64
    };
    (ins, del)
}

/// Affine backtrace: look up I1 extend at (score, k-1), offset + 1.
#[inline(always)]
fn bt_affine_ins1_ext(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let i1_ptr = wf_components.get_i1_ptr(score as usize);
    if i1_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*i1_ptr };
    if wf.wf_elements_init_min < k && k - 1 <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k - 1) + 1, BT_I1_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Affine backtrace: look up D1 extend at (score, k+1), offset unchanged.
#[inline(always)]
fn bt_affine_del1_ext(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let d1_ptr = wf_components.get_d1_ptr(score as usize);
    if d1_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*d1_ptr };
    if wf.wf_elements_init_min <= k + 1 && k < wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k + 1), BT_D1_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Perform backtrace for gap-affine distance, building a CIGAR.
///
/// Uses a state machine tracking current matrix type (M, I1, D1).
/// In M-state: predecessors are mismatch (M→M), gap open (M→I1/D1), or I1/D1→M.
/// In I1-state: predecessors are gap open (M→I1) or gap extend (I1→I1).
/// In D1-state: predecessors are gap open (M→D1) or gap extend (D1→D1).
#[allow(clippy::too_many_arguments)]
pub fn backtrace_affine(
    wf_components: &WavefrontComponents,
    penalties: &WavefrontPenalties,
    pattern_length: i32,
    text_length: i32,
    alignment_score: i32,
    alignment_k: i32,
    alignment_offset: i32,
    component_begin: ComponentType,
    component_end: ComponentType,
    cigar: &mut Cigar,
) {
    // Prepare cigar — write backwards from the end of the buffer
    let max_ops = cigar.operations.len();
    cigar.end_offset = max_ops;
    cigar.begin_offset = max_ops - 1;

    let mut score = alignment_score;
    let mut k = alignment_k;
    let mut offset = alignment_offset;
    let mut v = wavefront_v(k, offset);
    let mut h = wavefront_h(k, offset);

    // Account for ending insertions/deletions (ends-free, only when ending in M)
    if component_end == ComponentType::M {
        if v < pattern_length {
            for _ in 0..(pattern_length - v) {
                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;
            }
        }
        if h < text_length {
            for _ in 0..(text_length - h) {
                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;
            }
        }
    }

    // Penalty shortcuts
    let x = penalties.mismatch;
    let o_plus_e = penalties.gap_opening1 + penalties.gap_extension1;
    let e = penalties.gap_extension1;

    // State machine: start in component_end matrix
    // 0 = M, 1 = I1, 2 = D1
    let mut matrix = match component_end {
        ComponentType::M => 0u8,
        ComponentType::I1 => 1u8,
        ComponentType::D1 => 2u8,
        _ => 0u8,
    };

    // Main backtrace loop
    while v > 0 && h > 0 && score > 0 {
        match matrix {
            0 => {
                // In M-matrix: predecessors are mismatch, I1→M, D1→M
                let misms = bt_misms(wf_components, score - x, k);
                let ins_from_i1 = bt_i1_ext(wf_components, score, k);
                let del_from_d1 = bt_d1_ext(wf_components, score, k);

                let max_all = misms.max(ins_from_i1.max(del_from_d1));

                if max_all < 0 {
                    break;
                }

                let max_offset = piggyback_get_offset(max_all);
                let num_matches = offset - max_offset;
                write_matches(cigar, num_matches);
                offset = max_offset;

                v = wavefront_v(k, offset);
                h = wavefront_h(k, offset);
                if v <= 0 || h <= 0 {
                    break;
                }

                let bt_type = piggyback_get_type(max_all);
                match bt_type {
                    BT_M => {
                        // Mismatch: stay in M, emit X
                        score -= x;
                        cigar.operations[cigar.begin_offset] = b'X';
                        cigar.begin_offset -= 1;
                        offset -= 1;
                    }
                    BT_I1_EXT => {
                        // Came from I1: transition to I1 state
                        matrix = 1;
                    }
                    BT_D1_EXT => {
                        // Came from D1: transition to D1 state
                        matrix = 2;
                    }
                    _ => panic!("Invalid backtrace type in M-state: {}", bt_type),
                }
            }
            1 => {
                // In I1-matrix: predecessors are gap open (M→I1) or gap extend (I1→I1)
                // bt_open1_pair fetches M-wavefront once for both ins_open and del_open
                let (ins_open, _) = bt_open1_pair(wf_components, score - o_plus_e, k);
                let ins_ext = bt_affine_ins1_ext(wf_components, score - e, k);

                let max_all = ins_open.max(ins_ext);

                if max_all < 0 {
                    break;
                }

                // Emit 'I' operation
                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                // Insertion: undo the k-1 shift and offset+1 from forward pass
                k -= 1;
                offset -= 1;

                match bt_type {
                    BT_I1_OPEN => {
                        // Gap open: transition back to M
                        score -= o_plus_e;
                        matrix = 0;
                    }
                    BT_I1_EXT => {
                        // Gap extend: stay in I1
                        score -= e;
                    }
                    _ => panic!("Invalid backtrace type in I1-state: {}", bt_type),
                }
            }
            2 => {
                // In D1-matrix: predecessors are gap open (M→D1) or gap extend (D1→D1)
                let (_, del_open) = bt_open1_pair(wf_components, score - o_plus_e, k);
                let del_ext = bt_affine_del1_ext(wf_components, score - e, k);

                let max_all = del_open.max(del_ext);

                if max_all < 0 {
                    break;
                }

                // Emit 'D' operation
                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                // Deletion: undo the k+1 shift (offset unchanged in forward pass)
                k += 1;

                match bt_type {
                    BT_D1_OPEN => {
                        // Gap open: transition back to M
                        score -= o_plus_e;
                        matrix = 0;
                    }
                    BT_D1_EXT => {
                        // Gap extend: stay in D1
                        score -= e;
                    }
                    _ => panic!("Invalid backtrace type in D1-state: {}", bt_type),
                }
            }
            _ => unreachable!(),
        }

        v = wavefront_v(k, offset);
        h = wavefront_h(k, offset);
    }

    // Account for beginning matches (only when beginning in M)
    if component_begin == ComponentType::M && v > 0 && h > 0 {
        let num_matches = v.min(h);
        write_matches(cigar, num_matches);
        v -= num_matches;
        h -= num_matches;
    }

    // Account for beginning insertions/deletions (always emit remaining I/D)
    while v > 0 {
        cigar.operations[cigar.begin_offset] = b'D';
        cigar.begin_offset -= 1;
        v -= 1;
    }
    while h > 0 {
        cigar.operations[cigar.begin_offset] = b'I';
        cigar.begin_offset -= 1;
        h -= 1;
    }

    // Adjust to point to first valid operation
    cigar.begin_offset += 1;
    cigar.score = alignment_score;
}

// --- Affine2p backtrace helpers ---

/// Look up I2-wavefront at (score, k): returns piggyback(offset, BT_I2_EXT).
#[inline(always)]
fn bt_i2_ext(wf_components: &WavefrontComponents, score: i32, k: i32) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let i2_ptr = wf_components.get_i2_ptr(score as usize);
    if i2_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*i2_ptr };
    if wf.wf_elements_init_min <= k && k <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k), BT_I2_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up D2-wavefront at (score, k): returns piggyback(offset, BT_D2_EXT).
#[inline(always)]
fn bt_d2_ext(wf_components: &WavefrontComponents, score: i32, k: i32) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let d2_ptr = wf_components.get_d2_ptr(score as usize);
    if d2_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*d2_ptr };
    if wf.wf_elements_init_min <= k && k <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k), BT_D2_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Look up both M→I2 open (k-1, offset+1) and M→D2 open (k+1, offset) from
/// the same M-wavefront at the given score. Returns (ins2_open, del2_open).
#[inline(always)]
fn bt_open2_pair(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> (i64, i64) {
    if score < 0 {
        return (OFFSET_NULL as i64, OFFSET_NULL as i64);
    }
    let m_ptr = wf_components.get_m_ptr(score as usize);
    if m_ptr.is_null() {
        return (OFFSET_NULL as i64, OFFSET_NULL as i64);
    }
    let wf = unsafe { &*m_ptr };
    let min = wf.wf_elements_init_min;
    let max = wf.wf_elements_init_max;

    let ins = if min < k && k - 1 <= max {
        piggyback_set(wf.get_offset(k - 1) + 1, BT_I2_OPEN)
    } else {
        OFFSET_NULL as i64
    };
    let del = if min <= k + 1 && k < max {
        piggyback_set(wf.get_offset(k + 1), BT_D2_OPEN)
    } else {
        OFFSET_NULL as i64
    };
    (ins, del)
}

/// Affine2p backtrace: look up I2 extend at (score, k-1), offset + 1.
#[inline(always)]
fn bt_affine_ins2_ext(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let i2_ptr = wf_components.get_i2_ptr(score as usize);
    if i2_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*i2_ptr };
    if wf.wf_elements_init_min < k && k - 1 <= wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k - 1) + 1, BT_I2_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Affine2p backtrace: look up D2 extend at (score, k+1), offset unchanged.
#[inline(always)]
fn bt_affine_del2_ext(
    wf_components: &WavefrontComponents,
    score: i32,
    k: i32,
) -> i64 {
    if score < 0 {
        return OFFSET_NULL as i64;
    }
    let d2_ptr = wf_components.get_d2_ptr(score as usize);
    if d2_ptr.is_null() {
        return OFFSET_NULL as i64;
    }
    let wf = unsafe { &*d2_ptr };
    if wf.wf_elements_init_min <= k + 1 && k < wf.wf_elements_init_max {
        piggyback_set(wf.get_offset(k + 1), BT_D2_EXT)
    } else {
        OFFSET_NULL as i64
    }
}

/// Perform backtrace for gap-affine 2-piece distance, building a CIGAR.
///
/// Uses a 5-state machine tracking current matrix type (M, I1, D1, I2, D2).
#[allow(clippy::too_many_arguments)]
pub fn backtrace_affine2p(
    wf_components: &WavefrontComponents,
    penalties: &WavefrontPenalties,
    pattern_length: i32,
    text_length: i32,
    alignment_score: i32,
    alignment_k: i32,
    alignment_offset: i32,
    component_begin: ComponentType,
    component_end: ComponentType,
    cigar: &mut Cigar,
) {
    // Prepare cigar — write backwards from the end of the buffer
    let max_ops = cigar.operations.len();
    cigar.end_offset = max_ops;
    cigar.begin_offset = max_ops - 1;

    let mut score = alignment_score;
    let mut k = alignment_k;
    let mut offset = alignment_offset;
    let mut v = wavefront_v(k, offset);
    let mut h = wavefront_h(k, offset);

    // Account for ending insertions/deletions (ends-free, only when ending in M)
    if component_end == ComponentType::M {
        if v < pattern_length {
            for _ in 0..(pattern_length - v) {
                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;
            }
        }
        if h < text_length {
            for _ in 0..(text_length - h) {
                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;
            }
        }
    }

    // Penalty shortcuts
    let x = penalties.mismatch;
    let o1_plus_e1 = penalties.gap_opening1 + penalties.gap_extension1;
    let e1 = penalties.gap_extension1;
    let o2_plus_e2 = penalties.gap_opening2 + penalties.gap_extension2;
    let e2 = penalties.gap_extension2;

    // State machine: 0=M, 1=I1, 2=D1, 3=I2, 4=D2
    let mut matrix = match component_end {
        ComponentType::M => 0u8,
        ComponentType::I1 => 1u8,
        ComponentType::D1 => 2u8,
        ComponentType::I2 => 3u8,
        ComponentType::D2 => 4u8,
    };

    // Main backtrace loop
    while v > 0 && h > 0 && score > 0 {
        match matrix {
            0 => {
                // M-state: predecessors are mismatch, I1→M, D1→M, I2→M, D2→M
                let misms = bt_misms(wf_components, score - x, k);
                let ins_from_i1 = bt_i1_ext(wf_components, score, k);
                let del_from_d1 = bt_d1_ext(wf_components, score, k);
                let ins_from_i2 = bt_i2_ext(wf_components, score, k);
                let del_from_d2 = bt_d2_ext(wf_components, score, k);

                let max_all = misms
                    .max(ins_from_i1)
                    .max(del_from_d1)
                    .max(ins_from_i2)
                    .max(del_from_d2);

                if max_all < 0 {
                    break;
                }

                let max_offset = piggyback_get_offset(max_all);
                let num_matches = offset - max_offset;
                write_matches(cigar, num_matches);
                offset = max_offset;

                v = wavefront_v(k, offset);
                h = wavefront_h(k, offset);
                if v <= 0 || h <= 0 {
                    break;
                }

                let bt_type = piggyback_get_type(max_all);
                match bt_type {
                    BT_M => {
                        score -= x;
                        cigar.operations[cigar.begin_offset] = b'X';
                        cigar.begin_offset -= 1;
                        offset -= 1;
                    }
                    BT_I1_EXT => {
                        matrix = 1;
                    }
                    BT_D1_EXT => {
                        matrix = 2;
                    }
                    BT_I2_EXT => {
                        matrix = 3;
                    }
                    BT_D2_EXT => {
                        matrix = 4;
                    }
                    _ => panic!("Invalid backtrace type in M-state: {}", bt_type),
                }
            }
            1 => {
                // I1-state: gap open (M→I1) or gap extend (I1→I1)
                let (ins_open, _) = bt_open1_pair(wf_components, score - o1_plus_e1, k);
                let ins_ext = bt_affine_ins1_ext(wf_components, score - e1, k);

                let max_all = ins_open.max(ins_ext);
                if max_all < 0 {
                    break;
                }

                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                k -= 1;
                offset -= 1;

                match bt_type {
                    BT_I1_OPEN => {
                        score -= o1_plus_e1;
                        matrix = 0;
                    }
                    BT_I1_EXT => {
                        score -= e1;
                    }
                    _ => panic!("Invalid backtrace type in I1-state: {}", bt_type),
                }
            }
            2 => {
                // D1-state: gap open (M→D1) or gap extend (D1→D1)
                let (_, del_open) = bt_open1_pair(wf_components, score - o1_plus_e1, k);
                let del_ext = bt_affine_del1_ext(wf_components, score - e1, k);

                let max_all = del_open.max(del_ext);
                if max_all < 0 {
                    break;
                }

                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                k += 1;

                match bt_type {
                    BT_D1_OPEN => {
                        score -= o1_plus_e1;
                        matrix = 0;
                    }
                    BT_D1_EXT => {
                        score -= e1;
                    }
                    _ => panic!("Invalid backtrace type in D1-state: {}", bt_type),
                }
            }
            3 => {
                // I2-state: gap open (M→I2) or gap extend (I2→I2)
                let (ins_open, _) = bt_open2_pair(wf_components, score - o2_plus_e2, k);
                let ins_ext = bt_affine_ins2_ext(wf_components, score - e2, k);

                let max_all = ins_open.max(ins_ext);
                if max_all < 0 {
                    break;
                }

                cigar.operations[cigar.begin_offset] = b'I';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                k -= 1;
                offset -= 1;

                match bt_type {
                    BT_I2_OPEN => {
                        score -= o2_plus_e2;
                        matrix = 0;
                    }
                    BT_I2_EXT => {
                        score -= e2;
                    }
                    _ => panic!("Invalid backtrace type in I2-state: {}", bt_type),
                }
            }
            4 => {
                // D2-state: gap open (M→D2) or gap extend (D2→D2)
                let (_, del_open) = bt_open2_pair(wf_components, score - o2_plus_e2, k);
                let del_ext = bt_affine_del2_ext(wf_components, score - e2, k);

                let max_all = del_open.max(del_ext);
                if max_all < 0 {
                    break;
                }

                cigar.operations[cigar.begin_offset] = b'D';
                cigar.begin_offset -= 1;

                let bt_type = piggyback_get_type(max_all);
                k += 1;

                match bt_type {
                    BT_D2_OPEN => {
                        score -= o2_plus_e2;
                        matrix = 0;
                    }
                    BT_D2_EXT => {
                        score -= e2;
                    }
                    _ => panic!("Invalid backtrace type in D2-state: {}", bt_type),
                }
            }
            _ => unreachable!(),
        }

        v = wavefront_v(k, offset);
        h = wavefront_h(k, offset);
    }

    // Account for beginning matches (only when beginning in M)
    if component_begin == ComponentType::M && v > 0 && h > 0 {
        let num_matches = v.min(h);
        write_matches(cigar, num_matches);
        v -= num_matches;
        h -= num_matches;
    }

    // Account for beginning insertions/deletions (always emit remaining I/D)
    while v > 0 {
        cigar.operations[cigar.begin_offset] = b'D';
        cigar.begin_offset -= 1;
        v -= 1;
    }
    while h > 0 {
        cigar.operations[cigar.begin_offset] = b'I';
        cigar.begin_offset -= 1;
        h -= 1;
    }

    // Adjust to point to first valid operation
    cigar.begin_offset += 1;
    cigar.score = alignment_score;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligner::{AlignStatus, AlignmentScope, WavefrontAligner};

    #[test]
    fn test_backtrace_identical() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACGT");
        assert_eq!(score, 0);
        assert_eq!(aligner.status(), AlignStatus::Completed);
        assert_eq!(aligner.cigar().to_string_rle(true), "4M");
        aligner.cigar().check_alignment(b"ACGT", b"ACGT").unwrap();
    }

    #[test]
    fn test_backtrace_one_mismatch() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 1);
        assert_eq!(aligner.cigar().to_string_rle(true), "2M1X1M");
        aligner.cigar().check_alignment(b"ACGT", b"ACTT").unwrap();
    }

    #[test]
    fn test_backtrace_one_insertion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACT", b"ACGT");
        assert_eq!(score, 1);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACT", b"ACGT").unwrap();
        assert_eq!(cigar.score_edit(), 1);
    }

    #[test]
    fn test_backtrace_one_deletion() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACT");
        assert_eq!(score, 1);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACGT", b"ACT").unwrap();
        assert_eq!(cigar.score_edit(), 1);
    }

    #[test]
    fn test_backtrace_multiple_edits() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"kitten", b"sitting");
        assert_eq!(score, 3);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"kitten", b"sitting").unwrap();
        assert_eq!(cigar.score_edit(), 3);
    }

    #[test]
    fn test_backtrace_empty_vs_nonempty() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"", b"ACGT");
        assert_eq!(score, 4);
        assert_eq!(aligner.cigar().to_string_rle(true), "4I");
    }

    #[test]
    fn test_backtrace_nonempty_vs_empty() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"");
        assert_eq!(score, 4);
        assert_eq!(aligner.cigar().to_string_rle(true), "4D");
    }

    #[test]
    fn test_backtrace_indel_distance() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"ACGT", b"ACTT");
        assert_eq!(score, 2);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"ACGT", b"ACTT").unwrap();
        // Indel distance doesn't allow mismatches, so no X in CIGAR
        let rle = cigar.to_string_rle(true);
        assert!(
            !rle.contains('X'),
            "Indel CIGAR should not contain X: {}",
            rle
        );
    }

    #[test]
    fn test_backtrace_completely_different() {
        let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
        let score = aligner.align_end2end(b"AAAA", b"TTTT");
        assert_eq!(score, 4);
        let cigar = aligner.cigar();
        cigar.check_alignment(b"AAAA", b"TTTT").unwrap();
        assert_eq!(cigar.score_edit(), 4);
    }
}
