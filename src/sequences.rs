//! Wavefront sequences: encapsulates input sequences for alignment.
//!
//! Supports ASCII mode (the primary mode). Lambda and packed2bits modes
//! will be added in Phase 10.
//!
//! Key design: sequences are copied into an internal padded buffer with
//! sentinel characters at the end of each sequence. This allows the
//! extend kernel to safely read beyond sequence bounds without explicit
//! bounds checks in the inner loop (the sentinel characters will cause
//! mismatches, terminating extension).

/// Padding bytes around sequences for safe over-read in extend kernel.
const WF_SEQUENCES_PADDING: usize = 64;

/// Sentinel character at end of pattern (must differ from any text character).
const WF_SEQUENCES_PATTERN_EOS: u8 = b'!';

/// Sentinel character at end of text (must differ from any pattern character).
const WF_SEQUENCES_TEXT_EOS: u8 = b'?';

/// Sequence input mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceMode {
    Ascii,
    Lambda,
}

// --- 2-bit DNA decode helpers ---

const DNA_DECODE: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Decode a packed 2-bit DNA sequence into ASCII bytes.
/// Each input byte contains 4 nucleotides (2 bits each, LSB-first).
fn decode_packed2bits(packed: &[u8], length: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(length);
    for &byte in packed {
        let letters = [
            DNA_DECODE[(byte & 3) as usize],
            DNA_DECODE[((byte >> 2) & 3) as usize],
            DNA_DECODE[((byte >> 4) & 3) as usize],
            DNA_DECODE[((byte >> 6) & 3) as usize],
        ];
        for &ch in &letters {
            if result.len() >= length {
                break;
            }
            result.push(ch);
        }
    }
    result
}

/// Encode ASCII DNA sequence into packed 2-bit format.
/// A=0, C=1, G=2, T=3. Each byte holds 4 nucleotides (LSB-first).
pub fn encode_packed2bits(ascii: &[u8]) -> Vec<u8> {
    let encode = |ch: u8| -> u8 {
        match ch {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 0, // N or unknown → A
        }
    };
    let mut result = Vec::with_capacity(ascii.len().div_ceil(4));
    for chunk in ascii.chunks(4) {
        let mut byte = 0u8;
        for (i, &ch) in chunk.iter().enumerate() {
            byte |= encode(ch) << (i * 2);
        }
        result.push(byte);
    }
    result
}

/// Encapsulates input sequences for alignment.
pub struct WavefrontSequences {
    /// Current mode.
    pub mode: SequenceMode,
    /// Whether sequences are stored in reverse.
    pub reverse: bool,

    // Internal padded buffer
    seq_buffer: Vec<u8>,
    pattern_start: usize,
    text_start: usize,
    pattern_buffer_length: usize,
    text_buffer_length: usize,

    // Current view (may be a sub-range for BiWFA)
    /// Pointer offset into seq_buffer for current pattern.
    pattern_offset: usize,
    /// Pointer offset into seq_buffer for current text.
    text_offset: usize,
    /// Pattern begin coordinate (for BiWFA sub-ranges).
    pub pattern_begin: i32,
    /// Current pattern length.
    pub pattern_length: i32,
    /// Text begin coordinate (for BiWFA sub-ranges).
    pub text_begin: i32,
    /// Current text length.
    pub text_length: i32,

    // Saved EOS chars (for set_bounds restore)
    pattern_eos_saved: u8,
    text_eos_saved: u8,

    /// Custom match function for Lambda mode.
    match_funct: Option<Box<dyn Fn(i32, i32) -> bool>>,
}

impl WavefrontSequences {
    /// Create a new empty sequences holder.
    pub fn new() -> Self {
        Self {
            mode: SequenceMode::Ascii,
            reverse: false,
            seq_buffer: Vec::new(),
            pattern_start: 0,
            text_start: 0,
            pattern_buffer_length: 0,
            text_buffer_length: 0,
            pattern_offset: 0,
            text_offset: 0,
            pattern_begin: 0,
            pattern_length: 0,
            text_begin: 0,
            text_length: 0,
            pattern_eos_saved: WF_SEQUENCES_PATTERN_EOS,
            text_eos_saved: WF_SEQUENCES_TEXT_EOS,
            match_funct: None,
        }
    }

    /// Initialize with ASCII sequences.
    pub fn init_ascii(&mut self, pattern: &[u8], text: &[u8], reverse: bool) {
        self.mode = SequenceMode::Ascii;
        self.reverse = reverse;

        let plen = pattern.len();
        let tlen = text.len();

        // Allocate buffer: padding + pattern + padding + text + padding
        let buffer_size = plen + tlen + 3 * WF_SEQUENCES_PADDING;
        if self.seq_buffer.len() < buffer_size {
            let proposed = buffer_size + buffer_size / 2;
            self.seq_buffer = vec![0u8; proposed];
        }

        self.pattern_start = WF_SEQUENCES_PADDING;
        self.text_start = WF_SEQUENCES_PADDING + plen + WF_SEQUENCES_PADDING;

        // Copy pattern
        if reverse {
            for i in 0..plen {
                self.seq_buffer[self.pattern_start + i] = pattern[plen - 1 - i];
            }
        } else {
            self.seq_buffer[self.pattern_start..self.pattern_start + plen].copy_from_slice(pattern);
        }
        // Add pattern EOS sentinel
        self.seq_buffer[self.pattern_start + plen] = WF_SEQUENCES_PATTERN_EOS;
        self.pattern_buffer_length = plen;

        // Copy text
        if reverse {
            for i in 0..tlen {
                self.seq_buffer[self.text_start + i] = text[tlen - 1 - i];
            }
        } else {
            self.seq_buffer[self.text_start..self.text_start + tlen].copy_from_slice(text);
        }
        // Add text EOS sentinel
        self.seq_buffer[self.text_start + tlen] = WF_SEQUENCES_TEXT_EOS;
        self.text_buffer_length = tlen;

        // Set current view
        self.pattern_offset = self.pattern_start;
        self.text_offset = self.text_start;
        self.pattern_begin = 0;
        self.pattern_length = plen as i32;
        self.text_begin = 0;
        self.text_length = tlen as i32;

        self.pattern_eos_saved = self.seq_buffer[self.pattern_offset + plen];
        self.text_eos_saved = self.seq_buffer[self.text_offset + tlen];
    }

    /// Initialize with packed 2-bit DNA sequences.
    /// Each byte contains 4 nucleotides (2 bits each, LSB-first: A=0, C=1, G=2, T=3).
    pub fn init_packed2bits(
        &mut self,
        pattern: &[u8],
        pattern_length: usize,
        text: &[u8],
        text_length: usize,
        reverse: bool,
    ) {
        let pat_ascii = decode_packed2bits(pattern, pattern_length);
        let txt_ascii = decode_packed2bits(text, text_length);
        self.init_ascii(&pat_ascii, &txt_ascii, reverse);
        // Mode stays Ascii — packed2bits is just an input format
    }

    /// Initialize with a custom match function (Lambda mode).
    pub fn init_lambda(
        &mut self,
        match_funct: Box<dyn Fn(i32, i32) -> bool>,
        pattern_length: i32,
        text_length: i32,
        reverse: bool,
    ) {
        self.mode = SequenceMode::Lambda;
        self.reverse = reverse;
        self.match_funct = Some(match_funct);
        self.pattern_begin = 0;
        self.pattern_length = pattern_length;
        self.text_begin = 0;
        self.text_length = text_length;
        // No buffer allocation — no sentinels
    }

    /// Compare pattern[v] == text[h] using the Lambda match function.
    /// Returns false for out-of-bounds positions (replaces sentinel mechanism).
    #[inline(always)]
    pub fn cmp_lambda(&self, pattern_pos: i32, text_pos: i32) -> bool {
        if pattern_pos >= self.pattern_length || text_pos >= self.text_length {
            return false;
        }
        let mf = self.match_funct.as_ref().unwrap();
        if self.reverse {
            let p = self.pattern_begin + self.pattern_length - 1 - pattern_pos;
            let t = self.text_begin + self.text_length - 1 - text_pos;
            mf(p, t)
        } else {
            mf(self.pattern_begin + pattern_pos, self.text_begin + text_pos)
        }
    }

    /// Compare pattern[pattern_pos] == text[text_pos].
    #[inline(always)]
    pub fn cmp(&self, pattern_pos: i32, text_pos: i32) -> bool {
        self.seq_buffer[self.pattern_offset + pattern_pos as usize]
            == self.seq_buffer[self.text_offset + text_pos as usize]
    }

    /// Get pattern character at position.
    #[inline(always)]
    pub fn get_pattern(&self, position: i32) -> u8 {
        self.seq_buffer[self.pattern_offset + position as usize]
    }

    /// Get text character at position.
    #[inline(always)]
    pub fn get_text(&self, position: i32) -> u8 {
        self.seq_buffer[self.text_offset + position as usize]
    }

    /// Get a pointer to the pattern for bulk comparison.
    /// The returned slice has at least `pattern_length + WF_SEQUENCES_PADDING` bytes.
    #[inline(always)]
    pub fn pattern_ptr(&self) -> &[u8] {
        &self.seq_buffer[self.pattern_offset..]
    }

    /// Get a pointer to the text for bulk comparison.
    #[inline(always)]
    pub fn text_ptr(&self) -> &[u8] {
        &self.seq_buffer[self.text_offset..]
    }

    /// Set sub-range bounds (used by BiWFA).
    pub fn set_bounds(
        &mut self,
        pattern_begin: i32,
        pattern_end: i32,
        text_begin: i32,
        text_end: i32,
    ) {
        if self.mode != SequenceMode::Lambda {
            // Restore previous EOS chars
            let old_plen = self.pattern_length as usize;
            let old_tlen = self.text_length as usize;
            self.seq_buffer[self.pattern_offset + old_plen] = self.pattern_eos_saved;
            self.seq_buffer[self.text_offset + old_tlen] = self.text_eos_saved;

            // Set new view
            if self.reverse {
                self.pattern_offset =
                    self.pattern_start + (self.pattern_buffer_length - pattern_end as usize);
                self.text_offset = self.text_start + (self.text_buffer_length - text_end as usize);
            } else {
                self.pattern_offset = self.pattern_start + pattern_begin as usize;
                self.text_offset = self.text_start + text_begin as usize;
            }

            // Save and set new EOS
            let new_plen = (pattern_end - pattern_begin) as usize;
            let new_tlen = (text_end - text_begin) as usize;
            self.pattern_eos_saved = self.seq_buffer[self.pattern_offset + new_plen];
            self.text_eos_saved = self.seq_buffer[self.text_offset + new_tlen];
            self.seq_buffer[self.pattern_offset + new_plen] = WF_SEQUENCES_PATTERN_EOS;
            self.seq_buffer[self.text_offset + new_tlen] = WF_SEQUENCES_TEXT_EOS;
        }

        self.pattern_begin = pattern_begin;
        self.pattern_length = pattern_end - pattern_begin;
        self.text_begin = text_begin;
        self.text_length = text_end - text_begin;
    }
}

impl Default for WavefrontSequences {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_ascii() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACTT", false);

        assert_eq!(seq.pattern_length, 4);
        assert_eq!(seq.text_length, 4);
        assert_eq!(seq.mode, SequenceMode::Ascii);
    }

    #[test]
    fn test_cmp() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACTT", false);

        assert!(seq.cmp(0, 0)); // A == A
        assert!(seq.cmp(1, 1)); // C == C
        assert!(!seq.cmp(2, 2)); // G != T
        assert!(seq.cmp(3, 3)); // T == T
    }

    #[test]
    fn test_get_chars() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"TTTT", false);

        assert_eq!(seq.get_pattern(0), b'A');
        assert_eq!(seq.get_pattern(3), b'T');
        assert_eq!(seq.get_text(0), b'T');
    }

    #[test]
    fn test_eos_sentinel() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"ACGT", false);

        // Reading past pattern end should hit sentinel '!'
        assert_eq!(seq.get_pattern(4), WF_SEQUENCES_PATTERN_EOS);
        // Reading past text end should hit sentinel '?'
        assert_eq!(seq.get_text(4), WF_SEQUENCES_TEXT_EOS);
        // Sentinel comparison should fail
        assert!(!seq.cmp(4, 4));
    }

    #[test]
    fn test_reverse() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"TTTT", true);

        // Reversed pattern: TGCA
        assert_eq!(seq.get_pattern(0), b'T');
        assert_eq!(seq.get_pattern(1), b'G');
        assert_eq!(seq.get_pattern(2), b'C');
        assert_eq!(seq.get_pattern(3), b'A');
    }

    #[test]
    fn test_set_bounds() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"AACCGGTT", b"AACCGGTT", false);

        // Focus on sub-range [2,6) of both sequences: "CCGG"
        seq.set_bounds(2, 6, 2, 6);

        assert_eq!(seq.pattern_length, 4);
        assert_eq!(seq.text_length, 4);
        assert_eq!(seq.get_pattern(0), b'C');
        assert_eq!(seq.get_pattern(3), b'G');

        // EOS sentinel should be at position 4
        assert_eq!(seq.get_pattern(4), WF_SEQUENCES_PATTERN_EOS);
    }

    #[test]
    fn test_set_bounds_reverse() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"AACCGGTT", b"AACCGGTT", true);

        // In reverse mode, buffer contains "TTGGCCAA"
        // set_bounds(2, 6, 2, 6) => sub-range of original [2,6)
        // In reverse, this maps to buffer positions [8-6, 8-2) = [2, 6) => "GGCC"
        seq.set_bounds(2, 6, 2, 6);

        assert_eq!(seq.pattern_length, 4);
        assert_eq!(seq.get_pattern(0), b'G');
        assert_eq!(seq.get_pattern(3), b'C');
    }

    #[test]
    fn test_pattern_text_ptr() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"ACGT", b"TTGG", false);

        let p = seq.pattern_ptr();
        assert_eq!(p[0], b'A');
        assert_eq!(p[1], b'C');

        let t = seq.text_ptr();
        assert_eq!(t[0], b'T');
        assert_eq!(t[1], b'T');
    }

    #[test]
    fn test_reuse_buffer() {
        let mut seq = WavefrontSequences::new();
        seq.init_ascii(b"AAAA", b"TTTT", false);
        seq.init_ascii(b"CCCC", b"GGGG", false);

        assert_eq!(seq.get_pattern(0), b'C');
        assert_eq!(seq.get_text(0), b'G');
    }

    // --- Packed 2-bit DNA tests ---

    #[test]
    fn test_decode_2bits() {
        // A=0b00, C=0b01, G=0b10, T=0b11
        // Byte 0b11_10_01_00 = 0xE4 → A, C, G, T
        let packed = [0xE4u8];
        let decoded = decode_packed2bits(&packed, 4);
        assert_eq!(decoded, b"ACGT");
    }

    #[test]
    fn test_decode_2bits_partial() {
        // 3 nucleotides from one byte
        let packed = [0xE4u8]; // A, C, G, T but we only want 3
        let decoded = decode_packed2bits(&packed, 3);
        assert_eq!(decoded, b"ACG");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = b"ACGTACGT";
        let packed = encode_packed2bits(original);
        let decoded = decode_packed2bits(&packed, original.len());
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_odd_length() {
        let original = b"ACGTA";
        let packed = encode_packed2bits(original);
        let decoded = decode_packed2bits(&packed, original.len());
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_init_packed2bits() {
        let pattern = b"ACGT";
        let text = b"ACTT";
        let ppacked = encode_packed2bits(pattern);
        let tpacked = encode_packed2bits(text);

        let mut seq = WavefrontSequences::new();
        seq.init_packed2bits(&ppacked, 4, &tpacked, 4, false);

        assert_eq!(seq.mode, SequenceMode::Ascii); // stays Ascii after decode
        assert_eq!(seq.pattern_length, 4);
        assert!(seq.cmp(0, 0)); // A == A
        assert!(seq.cmp(1, 1)); // C == C
        assert!(!seq.cmp(2, 2)); // G != T
        assert!(seq.cmp(3, 3)); // T == T
    }

    #[test]
    fn test_packed2bits_reverse() {
        let pattern = b"ACGT";
        let ppacked = encode_packed2bits(pattern);

        let mut seq = WavefrontSequences::new();
        seq.init_packed2bits(&ppacked, 4, &ppacked, 4, true);

        // Reversed: TGCA
        assert_eq!(seq.get_pattern(0), b'T');
        assert_eq!(seq.get_pattern(1), b'G');
        assert_eq!(seq.get_pattern(2), b'C');
        assert_eq!(seq.get_pattern(3), b'A');
    }

    // --- Lambda mode tests ---

    #[test]
    fn test_init_lambda() {
        let pattern = b"ACGT";
        let text = b"ACTT";
        let mf = move |p: i32, t: i32| -> bool { pattern[p as usize] == text[t as usize] };

        let mut seq = WavefrontSequences::new();
        seq.init_lambda(Box::new(mf), 4, 4, false);

        assert_eq!(seq.mode, SequenceMode::Lambda);
        assert_eq!(seq.pattern_length, 4);
        assert_eq!(seq.text_length, 4);
    }

    #[test]
    fn test_cmp_lambda() {
        let pattern = b"ACGT".to_vec();
        let text = b"ACTT".to_vec();
        let p = pattern.clone();
        let t = text.clone();
        let mf = move |ppos: i32, tpos: i32| -> bool { p[ppos as usize] == t[tpos as usize] };

        let mut seq = WavefrontSequences::new();
        seq.init_lambda(Box::new(mf), 4, 4, false);

        assert!(seq.cmp_lambda(0, 0)); // A == A
        assert!(seq.cmp_lambda(1, 1)); // C == C
        assert!(!seq.cmp_lambda(2, 2)); // G != T
        assert!(seq.cmp_lambda(3, 3)); // T == T
    }

    #[test]
    fn test_cmp_lambda_bounds() {
        let mf = |_p: i32, _t: i32| -> bool { true };
        let mut seq = WavefrontSequences::new();
        seq.init_lambda(Box::new(mf), 4, 4, false);

        // Out of bounds returns false (replaces sentinel)
        assert!(!seq.cmp_lambda(4, 0));
        assert!(!seq.cmp_lambda(0, 4));
        assert!(!seq.cmp_lambda(4, 4));
    }

    #[test]
    fn test_cmp_lambda_reverse() {
        // pattern "ACGT", reversed positions: pos 0→3, pos 1→2, pos 2→1, pos 3→0
        let pattern = b"ACGT".to_vec();
        let text = b"TGCA".to_vec();
        let p = pattern.clone();
        let t = text.clone();
        let mf = move |ppos: i32, tpos: i32| -> bool { p[ppos as usize] == t[tpos as usize] };

        let mut seq = WavefrontSequences::new();
        seq.init_lambda(Box::new(mf), 4, 4, true);

        // In reverse mode, cmp_lambda(0, 0) maps to pattern[3], text[3]
        // pattern[3]=T, text[3]=A → false
        assert!(!seq.cmp_lambda(0, 0));
        // cmp_lambda(0, 0): pattern[3]=T, text[3]=A → mismatch
        // To verify reverse works, check that sequential matches work:
        // cmp_lambda(0, 0): pat[3]='T' vs txt[3]='A' → false
        // cmp_lambda(3, 3): pat[0]='A' vs txt[0]='T' → false
        // With matching reversed seqs:
        let p2 = b"ACGT".to_vec();
        let t2 = b"ACGT".to_vec(); // same as pattern
        let mf2 = move |ppos: i32, tpos: i32| -> bool { p2[ppos as usize] == t2[tpos as usize] };
        let mut seq2 = WavefrontSequences::new();
        seq2.init_lambda(Box::new(mf2), 4, 4, true);
        // reverse cmp_lambda(0, 0) → pat[3], txt[3] → T==T → true
        assert!(seq2.cmp_lambda(0, 0));
        // reverse cmp_lambda(3, 3) → pat[0], txt[0] → A==A → true
        assert!(seq2.cmp_lambda(3, 3));
    }

    #[test]
    fn test_lambda_set_bounds() {
        let pattern = b"AACCGGTT".to_vec();
        let text = b"AACCGGTT".to_vec();
        let p = pattern.clone();
        let t = text.clone();
        let mf = move |ppos: i32, tpos: i32| -> bool { p[ppos as usize] == t[tpos as usize] };

        let mut seq = WavefrontSequences::new();
        seq.init_lambda(Box::new(mf), 8, 8, false);
        seq.set_bounds(2, 6, 2, 6);

        assert_eq!(seq.pattern_length, 4);
        assert_eq!(seq.text_length, 4);
        assert_eq!(seq.pattern_begin, 2);
        assert_eq!(seq.text_begin, 2);
        // cmp_lambda(0, 0) should map to original positions (2, 2) → C==C
        assert!(seq.cmp_lambda(0, 0));
        // Out of bounds at 4
        assert!(!seq.cmp_lambda(4, 0));
    }
}
