//! CIGAR data structure for representing sequence alignments.
//!
//! A CIGAR string encodes the alignment between a pattern and text as a sequence
//! of operations:
//! - `M`: match (characters are equal)
//! - `X`: mismatch (characters differ)
//! - `I`: insertion (character in text, gap in pattern)
//! - `D`: deletion (character in pattern, gap in text)

use std::fmt;

use crate::penalties::{Affine2pPenalties, AffinePenalties, LinearPenalties};

/// CIGAR alignment representation.
#[derive(Clone)]
pub struct Cigar {
    /// Raw alignment operations (M, X, I, D).
    pub operations: Vec<u8>,
    /// Start offset into operations buffer (inclusive).
    pub begin_offset: usize,
    /// End offset into operations buffer (exclusive).
    pub end_offset: usize,
    /// Computed alignment score.
    pub score: i32,
    /// Alignment end position in pattern (v coordinate).
    pub end_v: i32,
    /// Alignment end position in text (h coordinate).
    pub end_h: i32,
}

impl Cigar {
    /// Create a new CIGAR with the given capacity.
    pub fn new(max_operations: usize) -> Self {
        Self {
            operations: vec![0u8; max_operations],
            begin_offset: 0,
            end_offset: 0,
            score: i32::MIN,
            end_v: -1,
            end_h: -1,
        }
    }

    /// Clear the CIGAR, resetting all fields.
    pub fn clear(&mut self) {
        self.begin_offset = 0;
        self.end_offset = 0;
        self.score = i32::MIN;
        self.end_v = -1;
        self.end_h = -1;
    }

    /// Resize the CIGAR buffer if needed.
    pub fn resize(&mut self, max_operations: usize) {
        if max_operations > self.operations.len() {
            self.operations = vec![0u8; max_operations];
        }
        self.clear();
    }

    /// Check if the CIGAR is null (empty).
    pub fn is_null(&self) -> bool {
        self.begin_offset >= self.end_offset
    }

    /// Get the operations slice.
    pub fn operations_slice(&self) -> &[u8] {
        &self.operations[self.begin_offset..self.end_offset]
    }

    /// Count the number of match operations.
    pub fn count_matches(&self) -> usize {
        self.operations_slice()
            .iter()
            .filter(|&&op| op == b'M')
            .count()
    }

    /// Append another CIGAR's operations forward.
    pub fn append_forward(&mut self, other: &Cigar) {
        let src = other.operations_slice();
        let dst_start = self.end_offset;
        self.operations[dst_start..dst_start + src.len()].copy_from_slice(src);
        self.end_offset += src.len();
    }

    /// Append another CIGAR's operations in reverse.
    pub fn append_reverse(&mut self, other: &Cigar) {
        let src = other.operations_slice();
        let dst_start = self.end_offset;
        for (i, &op) in src.iter().rev().enumerate() {
            self.operations[dst_start + i] = op;
        }
        self.end_offset += src.len();
    }

    /// Append `length` deletion operations.
    pub fn append_deletion(&mut self, length: usize) {
        for i in 0..length {
            self.operations[self.end_offset + i] = b'D';
        }
        self.end_offset += length;
    }

    /// Append `length` insertion operations.
    pub fn append_insertion(&mut self, length: usize) {
        for i in 0..length {
            self.operations[self.end_offset + i] = b'I';
        }
        self.end_offset += length;
    }

    /// Compute edit distance score (count of non-match operations).
    pub fn score_edit(&self) -> i32 {
        self.operations_slice()
            .iter()
            .map(|&op| match op {
                b'M' => 0,
                b'X' | b'D' | b'I' => 1,
                _ => panic!("Unknown CIGAR operation: {}", op as char),
            })
            .sum()
    }

    /// Compute gap-linear score.
    pub fn score_gap_linear(&self, penalties: &LinearPenalties) -> i32 {
        self.operations_slice()
            .iter()
            .map(|&op| match op {
                b'M' => -penalties.match_,
                b'X' => -penalties.mismatch,
                b'I' | b'D' => -penalties.indel,
                _ => panic!("Unknown CIGAR operation: {}", op as char),
            })
            .sum()
    }

    /// Compute gap-affine score.
    pub fn score_gap_affine(&self, penalties: &AffinePenalties) -> i32 {
        let ops = self.operations_slice();
        let mut score = 0i32;
        let mut last_op = 0u8;
        for &op in ops {
            match op {
                b'M' => score -= penalties.match_,
                b'X' => score -= penalties.mismatch,
                b'D' => {
                    score -= penalties.gap_extension
                        + if last_op == b'D' {
                            0
                        } else {
                            penalties.gap_opening
                        };
                }
                b'I' => {
                    score -= penalties.gap_extension
                        + if last_op == b'I' {
                            0
                        } else {
                            penalties.gap_opening
                        };
                }
                _ => panic!("Unknown CIGAR operation: {}", op as char),
            }
            last_op = op;
        }
        score
    }

    /// Compute gap-affine 2-piece score.
    pub fn score_gap_affine2p(&self, penalties: &Affine2pPenalties) -> i32 {
        let ops = self.operations_slice();
        let mut score = 0i32;
        let mut last_op = 0u8;
        let mut op_length = 0i32;

        for &op in ops {
            if op != last_op && last_op != 0 {
                score -= Self::score_affine2p_op(last_op, op_length, penalties);
                op_length = 0;
            }
            last_op = op;
            op_length += 1;
        }
        if last_op != 0 {
            score -= Self::score_affine2p_op(last_op, op_length, penalties);
        }
        score
    }

    fn score_affine2p_op(operation: u8, length: i32, penalties: &Affine2pPenalties) -> i32 {
        match operation {
            b'M' => penalties.match_ * length,
            b'X' => penalties.mismatch * length,
            b'D' | b'I' => {
                let score1 = penalties.gap_opening1 + penalties.gap_extension1 * length;
                let score2 = penalties.gap_opening2 + penalties.gap_extension2 * length;
                score1.min(score2)
            }
            _ => panic!("Unknown CIGAR operation: {}", operation as char),
        }
    }

    /// Trim CIGAR to the maximal-scoring prefix (gap-linear penalties).
    ///
    /// Returns `true` if the CIGAR was trimmed (alignment is partial).
    pub fn maxtrim_gap_linear(&mut self, penalties: &LinearPenalties) -> bool {
        let begin = self.begin_offset;
        let end = self.end_offset;
        if begin >= end {
            return false;
        }
        let match_score: i32 = if penalties.match_ != 0 {
            penalties.match_
        } else {
            -1
        };
        let mut max_score = 0i32;
        let mut max_score_offset = begin;
        let mut max_end_v = 0i32;
        let mut max_end_h = 0i32;

        let mut score = 0i32;
        let mut end_v = 0i32;
        let mut end_h = 0i32;

        for i in begin..end {
            match self.operations[i] {
                b'M' => {
                    score -= match_score;
                    end_v += 1;
                    end_h += 1;
                }
                b'X' => {
                    score -= penalties.mismatch;
                    end_v += 1;
                    end_h += 1;
                }
                b'I' => {
                    score -= penalties.indel;
                    end_h += 1;
                }
                b'D' => {
                    score -= penalties.indel;
                    end_v += 1;
                }
                _ => {}
            }
            if max_score < score {
                max_score = score;
                max_score_offset = i;
                max_end_v = end_v;
                max_end_h = end_h;
            }
        }

        let trimmed = max_score_offset != end - 1;
        if max_score == 0 {
            self.clear();
        } else {
            self.end_offset = max_score_offset + 1;
            self.score = max_score;
            self.end_v = max_end_v;
            self.end_h = max_end_h;
        }
        trimmed
    }

    /// Trim CIGAR to the maximal-scoring prefix (gap-affine penalties).
    ///
    /// Returns `true` if the CIGAR was trimmed (alignment is partial).
    pub fn maxtrim_gap_affine(&mut self, penalties: &AffinePenalties) -> bool {
        let begin = self.begin_offset;
        let end = self.end_offset;
        if begin >= end {
            return false;
        }
        let match_score: i32 = if penalties.match_ != 0 {
            penalties.match_
        } else {
            -1
        };
        let mut max_score = 0i32;
        let mut max_score_offset = begin;
        let mut max_end_v = 0i32;
        let mut max_end_h = 0i32;

        let mut last_op = 0u8;
        let mut score = 0i32;
        let mut end_v = 0i32;
        let mut end_h = 0i32;

        for i in begin..end {
            let op = self.operations[i];
            match op {
                b'M' => {
                    score -= match_score;
                    end_v += 1;
                    end_h += 1;
                }
                b'X' => {
                    score -= penalties.mismatch;
                    end_v += 1;
                    end_h += 1;
                }
                b'I' => {
                    score -= penalties.gap_extension
                        + if last_op == b'I' {
                            0
                        } else {
                            penalties.gap_opening
                        };
                    end_h += 1;
                }
                b'D' => {
                    score -= penalties.gap_extension
                        + if last_op == b'D' {
                            0
                        } else {
                            penalties.gap_opening
                        };
                    end_v += 1;
                }
                _ => {}
            }
            last_op = op;
            if max_score < score {
                max_score = score;
                max_score_offset = i;
                max_end_v = end_v;
                max_end_h = end_h;
            }
        }

        let trimmed = max_score_offset != end - 1;
        if max_score == 0 {
            self.clear();
        } else {
            self.end_offset = max_score_offset + 1;
            self.score = max_score;
            self.end_v = max_end_v;
            self.end_h = max_end_h;
        }
        trimmed
    }

    /// Trim CIGAR to the maximal-scoring prefix (gap-affine 2-piece penalties).
    ///
    /// Returns `true` if the CIGAR was trimmed (alignment is partial).
    pub fn maxtrim_gap_affine2p(&mut self, penalties: &Affine2pPenalties) -> bool {
        let begin = self.begin_offset;
        let end = self.end_offset;
        if begin >= end {
            return false;
        }
        let mut max_score = 0i32;
        let mut max_score_offset = begin;
        let mut max_end_v = 0i32;
        let mut max_end_h = 0i32;

        let mut last_op = 0u8;
        let mut score = 0i32;
        let mut end_v = 0i32;
        let mut end_h = 0i32;
        let mut op_length = 0i32;

        for i in begin..end {
            let operation = self.operations[i];
            if operation != last_op && last_op != 0 {
                score -= Self::maxtrim_affine2p_op_score(
                    last_op, op_length, penalties, &mut end_v, &mut end_h,
                );
                op_length = 0;
                if max_score < score {
                    max_score = score;
                    max_score_offset = i - 1;
                    max_end_v = end_v;
                    max_end_h = end_h;
                }
            }
            last_op = operation;
            op_length += 1;
        }
        // Account for last operation run
        score -=
            Self::maxtrim_affine2p_op_score(last_op, op_length, penalties, &mut end_v, &mut end_h);
        if max_score < score {
            max_score = score;
            max_score_offset = end - 1;
            max_end_v = end_v;
            max_end_h = end_h;
        }

        let trimmed = max_score_offset != end - 1;
        if max_score == 0 {
            self.clear();
        } else {
            self.end_offset = max_score_offset + 1;
            self.score = max_score;
            self.end_v = max_end_v;
            self.end_h = max_end_h;
        }
        trimmed
    }

    fn maxtrim_affine2p_op_score(
        operation: u8,
        length: i32,
        penalties: &Affine2pPenalties,
        end_v: &mut i32,
        end_h: &mut i32,
    ) -> i32 {
        match operation {
            b'M' => {
                *end_v += length;
                *end_h += length;
                let match_score = if penalties.match_ != 0 {
                    penalties.match_
                } else {
                    -1
                };
                match_score * length
            }
            b'X' => {
                *end_v += length;
                *end_h += length;
                penalties.mismatch * length
            }
            b'D' => {
                *end_v += length;
                let s1 = penalties.gap_opening1 + penalties.gap_extension1 * length;
                let s2 = penalties.gap_opening2 + penalties.gap_extension2 * length;
                s1.min(s2)
            }
            b'I' => {
                *end_h += length;
                let s1 = penalties.gap_opening1 + penalties.gap_extension1 * length;
                let s2 = penalties.gap_opening2 + penalties.gap_extension2 * length;
                s1.min(s2)
            }
            _ => 0,
        }
    }

    /// Check that a CIGAR correctly aligns the given pattern and text.
    ///
    /// Returns `Ok(())` if valid, or `Err(message)` describing the problem.
    pub fn check_alignment(&self, pattern: &[u8], text: &[u8]) -> Result<(), String> {
        let ops = self.operations_slice();
        let mut pattern_pos = 0usize;
        let mut text_pos = 0usize;

        for (i, &op) in ops.iter().enumerate() {
            match op {
                b'M' => {
                    if pattern_pos >= pattern.len() || text_pos >= text.len() {
                        return Err(format!(
                            "Match at op {} out of bounds (ppos={}, tpos={})",
                            i, pattern_pos, text_pos
                        ));
                    }
                    if pattern[pattern_pos] != text[text_pos] {
                        return Err(format!(
                            "Alignment not matching (pattern[{}]={} != text[{}]={})",
                            pattern_pos,
                            pattern[pattern_pos] as char,
                            text_pos,
                            text[text_pos] as char
                        ));
                    }
                    pattern_pos += 1;
                    text_pos += 1;
                }
                b'X' => {
                    if pattern_pos >= pattern.len() || text_pos >= text.len() {
                        return Err(format!(
                            "Mismatch at op {} out of bounds (ppos={}, tpos={})",
                            i, pattern_pos, text_pos
                        ));
                    }
                    if pattern[pattern_pos] == text[text_pos] {
                        return Err(format!(
                            "Alignment not mismatching (pattern[{}]={} == text[{}]={})",
                            pattern_pos,
                            pattern[pattern_pos] as char,
                            text_pos,
                            text[text_pos] as char
                        ));
                    }
                    pattern_pos += 1;
                    text_pos += 1;
                }
                b'I' => {
                    text_pos += 1;
                }
                b'D' => {
                    pattern_pos += 1;
                }
                _ => {
                    return Err(format!("Unknown edit operation '{}'", op as char));
                }
            }
        }

        if pattern_pos != pattern.len() {
            return Err(format!(
                "Alignment incorrect length (pattern-aligned={}, pattern-length={})",
                pattern_pos,
                pattern.len()
            ));
        }
        if text_pos != text.len() {
            return Err(format!(
                "Alignment incorrect length (text-aligned={}, text-length={})",
                text_pos,
                text.len()
            ));
        }
        Ok(())
    }

    /// Format the CIGAR as a run-length encoded string (e.g., "3M1X2I").
    /// If `print_matches` is false, match runs are omitted.
    pub fn to_string_rle(&self, print_matches: bool) -> String {
        if self.is_null() {
            return String::new();
        }
        let ops = self.operations_slice();
        let mut result = String::new();
        let mut last_op = ops[0];
        let mut last_op_length = 1u32;

        for &op in &ops[1..] {
            if op == last_op {
                last_op_length += 1;
            } else {
                if print_matches || last_op != b'M' {
                    result.push_str(&format!("{}{}", last_op_length, last_op as char));
                }
                last_op = op;
                last_op_length = 1;
            }
        }
        if print_matches || last_op != b'M' {
            result.push_str(&format!("{}{}", last_op_length, last_op as char));
        }
        result
    }

    /// Format the CIGAR as a pretty-printed alignment.
    pub fn print_pretty(&self, pattern: &[u8], text: &[u8]) -> String {
        let ops = self.operations_slice();
        let max_len = pattern.len() + text.len() + 1;
        let mut pattern_alg = Vec::with_capacity(max_len);
        let mut ops_alg = Vec::with_capacity(max_len);
        let mut text_alg = Vec::with_capacity(max_len);
        let mut pattern_pos = 0usize;
        let mut text_pos = 0usize;

        for &op in ops {
            match op {
                b'M' => {
                    pattern_alg.push(pattern[pattern_pos]);
                    text_alg.push(text[text_pos]);
                    ops_alg.push(if pattern[pattern_pos] != text[text_pos] {
                        b'X'
                    } else {
                        b'|'
                    });
                    pattern_pos += 1;
                    text_pos += 1;
                }
                b'X' => {
                    pattern_alg.push(pattern[pattern_pos]);
                    text_alg.push(text[text_pos]);
                    ops_alg.push(if pattern[pattern_pos] != text[text_pos] {
                        b' '
                    } else {
                        b'X'
                    });
                    pattern_pos += 1;
                    text_pos += 1;
                }
                b'I' => {
                    pattern_alg.push(b'-');
                    ops_alg.push(b' ');
                    text_alg.push(text[text_pos]);
                    text_pos += 1;
                }
                b'D' => {
                    pattern_alg.push(pattern[pattern_pos]);
                    ops_alg.push(b' ');
                    text_alg.push(b'-');
                    pattern_pos += 1;
                }
                _ => {}
            }
        }

        let p_str = String::from_utf8_lossy(&pattern_alg);
        let o_str = String::from_utf8_lossy(&ops_alg);
        let t_str = String::from_utf8_lossy(&text_alg);

        format!(
            "      ALIGNMENT {}\n      PATTERN    {}\n                 {}\n      TEXT       {}",
            self.to_string_rle(true),
            p_str,
            o_str,
            t_str
        )
    }
}

impl fmt::Display for Cigar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_rle(true))
    }
}

impl fmt::Debug for Cigar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cigar")
            .field("ops", &self.to_string_rle(true))
            .field("score", &self.score)
            .field("end_v", &self.end_v)
            .field("end_h", &self.end_h)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cigar(ops: &[u8]) -> Cigar {
        let mut c = Cigar::new(ops.len());
        c.operations[..ops.len()].copy_from_slice(ops);
        c.end_offset = ops.len();
        c
    }

    #[test]
    fn test_is_null() {
        let c = Cigar::new(10);
        assert!(c.is_null());
    }

    #[test]
    fn test_count_matches() {
        let c = make_cigar(b"MMXMID");
        assert_eq!(c.count_matches(), 3);
    }

    #[test]
    fn test_score_edit() {
        let c = make_cigar(b"MMXMID");
        // X=1, I=1, D=1 => 3
        assert_eq!(c.score_edit(), 3);
    }

    #[test]
    fn test_score_edit_all_matches() {
        let c = make_cigar(b"MMMM");
        assert_eq!(c.score_edit(), 0);
    }

    #[test]
    fn test_score_gap_linear() {
        let p = LinearPenalties {
            match_: 0,
            mismatch: 4,
            indel: 2,
        };
        // MMXID: M=0, M=0, X=-(-4)=4, I=-(-2)=2, D=-(-2)=2 => 8
        // Wait, score -= penalties.X means score += X (since we negate)
        // Actually: score = sum of (-penalties.op) for each op
        // For M: -match_ = -0 = 0
        // For X: -mismatch = -4... no wait
        // Let me re-read: score -= penalties.match_ means score += (-match_)
        // The C code does: score -= penalties->match for M
        // With match_=0: 0; with mismatch=4: score -= 4 => score goes negative
        // But the C function returns a positive score...
        // Actually the C code scores penalties as costs. "score -= penalties->mismatch"
        // means "subtract the negative of the penalty" ... no.
        // C: score -= penalties->match; => score = score - match
        // With match=0: score -= 0 => no change
        // With mismatch=4: score -= 4 => score becomes -4
        // But wait, penalties are positive costs in WFA convention
        // Looking at C: result is sum of (-penalty) for each op? No:
        // C code: score -= penalties->match; for 'M', so score decreases by match
        // If match=0, no change. If match=-1 (reward), score increases by 1.
        // For 'X': score -= penalties->mismatch = score -= 4, so score = -4
        // Hmm, but the test results show positive scores...
        // Actually looking more carefully at the C convention:
        // penalties use "penalty representation" where match <= 0 and costs > 0
        // score -= penalty means we're accumulating negative scores
        // But edit distance returns positive... let me check cigar_score_edit:
        // It does: case 'X': ++score; => just counts, returns positive
        // For gap_linear: score -= penalties->mismatch where mismatch is positive
        // So the result is negative! The WFA convention is that penalties make
        // scores go negative/down.
        // Our Rust implementation matches: score -= penalties.match_ etc.
        let c = make_cigar(b"MMXID");
        let s = c.score_gap_linear(&p);
        // M: -0 + M: -0 + X: -4 + I: -2 + D: -2 = -8
        assert_eq!(s, -8);
    }

    #[test]
    fn test_score_gap_affine() {
        let p = AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        };
        // Ops: M M X I I D
        // M: -0, M: -0, X: -4
        // I (first): -(2+6) = -8, I (continued): -(2+0) = -2
        // D (first): -(2+6) = -8
        // Total: -22
        let c = make_cigar(b"MMXIID");
        let s = c.score_gap_affine(&p);
        assert_eq!(s, -22);
    }

    #[test]
    fn test_score_gap_affine2p() {
        let p = Affine2pPenalties {
            match_: 0,
            mismatch: 4,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 24,
            gap_extension2: 1,
        };
        // Ops: M X I I I
        // M run (1): match*1 = 0
        // X run (1): mismatch*1 = 4
        // I run (3): min(6+2*3, 24+1*3) = min(12, 27) = 12
        // Total: -(0 + 4 + 12) = -16
        let c = make_cigar(b"MXIII");
        let s = c.score_gap_affine2p(&p);
        assert_eq!(s, -16);
    }

    #[test]
    fn test_check_alignment_valid() {
        let pattern = b"ACGT";
        let text = b"ACGT";
        let c = make_cigar(b"MMMM");
        assert!(c.check_alignment(pattern, text).is_ok());
    }

    #[test]
    fn test_check_alignment_with_mismatch() {
        let pattern = b"ACGT";
        let text = b"ACTT";
        let c = make_cigar(b"MMXM");
        assert!(c.check_alignment(pattern, text).is_ok());
    }

    #[test]
    fn test_check_alignment_with_indels() {
        let pattern = b"ACGT";
        let text = b"ACT";
        // pattern: A C G T
        // text:    A C - T
        // ops:     M M D M
        let c = make_cigar(b"MMDM");
        assert!(c.check_alignment(pattern, text).is_ok());
    }

    #[test]
    fn test_check_alignment_insertion() {
        let pattern = b"ACT";
        let text = b"ACGT";
        // pattern: A C - T
        // text:    A C G T
        // ops:     M M I M
        let c = make_cigar(b"MMIM");
        assert!(c.check_alignment(pattern, text).is_ok());
    }

    #[test]
    fn test_check_alignment_invalid_match() {
        let pattern = b"ACGT";
        let text = b"ACTT";
        // Claiming match at pos 2 but G != T
        let c = make_cigar(b"MMMM");
        assert!(c.check_alignment(pattern, text).is_err());
    }

    #[test]
    fn test_check_alignment_wrong_length() {
        let pattern = b"ACG";
        let text = b"ACGT";
        let c = make_cigar(b"MMM");
        assert!(c.check_alignment(pattern, text).is_err());
    }

    #[test]
    fn test_to_string_rle() {
        let c = make_cigar(b"MMMXXIID");
        assert_eq!(c.to_string_rle(true), "3M2X2I1D");
        assert_eq!(c.to_string_rle(false), "2X2I1D");
    }

    #[test]
    fn test_display() {
        let c = make_cigar(b"MMXD");
        assert_eq!(format!("{}", c), "2M1X1D");
    }

    #[test]
    fn test_append_forward() {
        let mut dst = Cigar::new(20);
        dst.operations[0] = b'M';
        dst.operations[1] = b'M';
        dst.end_offset = 2;

        let src = make_cigar(b"XID");
        dst.append_forward(&src);
        assert_eq!(dst.end_offset, 5);
        assert_eq!(&dst.operations[..5], b"MMXID");
    }

    #[test]
    fn test_append_reverse() {
        let mut dst = Cigar::new(20);
        dst.end_offset = 0;

        let src = make_cigar(b"XID");
        dst.append_reverse(&src);
        assert_eq!(dst.end_offset, 3);
        assert_eq!(&dst.operations[..3], b"DIX");
    }

    #[test]
    fn test_append_deletion_insertion() {
        let mut c = Cigar::new(20);
        c.append_deletion(3);
        c.append_insertion(2);
        assert_eq!(c.end_offset, 5);
        assert_eq!(&c.operations[..5], b"DDDII");
    }

    #[test]
    fn test_clear() {
        let mut c = make_cigar(b"MMXD");
        c.score = 42;
        c.clear();
        assert!(c.is_null());
        assert_eq!(c.score, i32::MIN);
    }

    #[test]
    fn test_print_pretty() {
        let pattern = b"ACGT";
        let text = b"ACTT";
        let c = make_cigar(b"MMXM");
        let pretty = c.print_pretty(pattern, text);
        assert!(pretty.contains("PATTERN"));
        assert!(pretty.contains("TEXT"));
    }

    // --- Maxtrim tests ---

    #[test]
    fn test_maxtrim_linear_all_matches() {
        let p = LinearPenalties {
            match_: 0,
            mismatch: 4,
            indel: 2,
        };
        let mut c = make_cigar(b"MMMM");
        // All matches → score goes up each step → no trim
        let trimmed = c.maxtrim_gap_linear(&p);
        assert!(!trimmed);
        assert_eq!(c.end_offset, 4);
        assert_eq!(c.score, 4); // 4 matches, each +1
    }

    #[test]
    fn test_maxtrim_linear_matches_then_mismatches() {
        let p = LinearPenalties {
            match_: 0,
            mismatch: 4,
            indel: 2,
        };
        // MMMXX: scores after each op: 1, 2, 3, -1, -5
        // Max score = 3 at offset 2 (0-indexed)
        let mut c = make_cigar(b"MMMXX");
        let trimmed = c.maxtrim_gap_linear(&p);
        assert!(trimmed);
        assert_eq!(c.end_offset, 3); // trimmed to 3 ops (MMM)
        assert_eq!(c.score, 3);
        assert_eq!(c.end_v, 3);
        assert_eq!(c.end_h, 3);
    }

    #[test]
    fn test_maxtrim_linear_all_mismatches() {
        let p = LinearPenalties {
            match_: 0,
            mismatch: 4,
            indel: 2,
        };
        // XXX: scores: -4, -8, -12 → max_score = 0, cigar cleared
        let mut c = make_cigar(b"XXX");
        let trimmed = c.maxtrim_gap_linear(&p);
        assert!(trimmed);
        assert!(c.is_null());
    }

    #[test]
    fn test_maxtrim_affine_with_gap() {
        let p = AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        };
        // MMMMMIIM: scores after each op: 1,2,3,4,5,-3,-5,-4
        // Max score = 5 at offset 4
        let mut c = make_cigar(b"MMMMMIIM");
        let trimmed = c.maxtrim_gap_affine(&p);
        assert!(trimmed);
        assert_eq!(c.end_offset, 5);
        assert_eq!(c.score, 5);
        assert_eq!(c.end_v, 5);
        assert_eq!(c.end_h, 5);
    }

    #[test]
    fn test_maxtrim_affine_no_trim() {
        let p = AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        };
        // MMMM: all matches → monotonically increasing
        let mut c = make_cigar(b"MMMM");
        let trimmed = c.maxtrim_gap_affine(&p);
        assert!(!trimmed);
        assert_eq!(c.end_offset, 4);
    }

    #[test]
    fn test_maxtrim_affine2p_basic() {
        let p = Affine2pPenalties {
            match_: 0,
            mismatch: 4,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 24,
            gap_extension2: 1,
        };
        // MMMXX: M run(3) → score += 3, X run(2) → score -= 4*2 = 8 → score = -5
        // Max score = 3 after M run
        let mut c = make_cigar(b"MMMXX");
        let trimmed = c.maxtrim_gap_affine2p(&p);
        assert!(trimmed);
        assert_eq!(c.score, 3);
        assert_eq!(c.end_v, 3);
        assert_eq!(c.end_h, 3);
    }

    #[test]
    fn test_maxtrim_affine2p_no_trim() {
        let p = Affine2pPenalties {
            match_: 0,
            mismatch: 4,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 24,
            gap_extension2: 1,
        };
        let mut c = make_cigar(b"MMMMMM");
        let trimmed = c.maxtrim_gap_affine2p(&p);
        assert!(!trimmed);
        assert_eq!(c.end_offset, 6);
    }
}
