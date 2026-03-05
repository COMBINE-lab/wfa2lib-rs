//! Packed CIGAR (alignment operations encoded in 2-bit fields within a u32).
//!
//! Each operation is encoded as 2 bits:
//! - 0b00: (unused/match placeholder — matches are implicit in WFA)
//! - 0b01: Deletion
//! - 0b10: Mismatch
//! - 0b11: Insertion
//!
//! Operations are packed from MSB to LSB, with `push_back` shifting left
//! and inserting at the least significant bits.

/// Packed CIGAR type (32-bit, holds up to 16 operations).
pub type Pcigar = u32;

/// Null/empty packed CIGAR.
pub const PCIGAR_NULL: Pcigar = 0;

/// Maximum number of operations in a 32-bit packed CIGAR.
pub const PCIGAR_MAX_LENGTH: usize = 16;

// Operation codes (2-bit)
pub const PCIGAR_DELETION: u32 = 1;
pub const PCIGAR_MISMATCH: u32 = 2;
pub const PCIGAR_INSERTION: u32 = 3;

/// Mask: pcigar is completely full (16 operations).
pub const PCIGAR_FULL_MASK: u32 = 0x4000_0000;
/// Mask: pcigar has 15+ slots used.
pub const PCIGAR_ALMOST_FULL_MASK: u32 = 0x1000_0000;
/// Mask: pcigar has 9+ slots used.
pub const PCIGAR_HALF_FULL_MASK: u32 = 0x0001_0000;

/// Push an operation onto the back of the packed CIGAR.
#[inline(always)]
pub const fn pcigar_push_back(pcigar: Pcigar, operation: u32) -> Pcigar {
    (pcigar << 2) | operation
}

/// Push an insertion onto the back.
#[inline(always)]
pub const fn pcigar_push_back_ins(pcigar: Pcigar) -> Pcigar {
    (pcigar << 2) | PCIGAR_INSERTION
}

/// Push a deletion onto the back.
#[inline(always)]
pub const fn pcigar_push_back_del(pcigar: Pcigar) -> Pcigar {
    (pcigar << 2) | PCIGAR_DELETION
}

/// Push a mismatch onto the back.
#[inline(always)]
pub const fn pcigar_push_back_misms(pcigar: Pcigar) -> Pcigar {
    (pcigar << 2) | PCIGAR_MISMATCH
}

/// Pop (discard) the front operation by shifting left 2 bits.
#[inline(always)]
pub const fn pcigar_pop_front(pcigar: Pcigar) -> Pcigar {
    pcigar << 2
}

/// Extract the front operation (the 2 MSBs).
#[inline(always)]
pub const fn pcigar_extract(pcigar: Pcigar) -> u32 {
    pcigar >> 30
}

/// Check if the pcigar utilization meets or exceeds the given mask threshold.
#[inline(always)]
pub const fn pcigar_is_utilised(pcigar: Pcigar, mask: u32) -> bool {
    pcigar >= mask
}

/// Count the number of free slots remaining.
#[inline(always)]
pub const fn pcigar_free_slots(pcigar: Pcigar) -> usize {
    if pcigar != 0 {
        (pcigar.leading_zeros() / 2) as usize
    } else {
        PCIGAR_MAX_LENGTH
    }
}

/// Get the number of operations stored in the packed CIGAR.
pub const fn pcigar_get_length(pcigar: Pcigar) -> usize {
    PCIGAR_MAX_LENGTH - pcigar_free_slots(pcigar)
}

/// Unpack a packed CIGAR into a character buffer.
/// Returns the number of characters written.
pub fn pcigar_unpack(pcigar: Pcigar, buffer: &mut Vec<u8>) -> usize {
    let length = pcigar_get_length(pcigar);
    if length == 0 {
        return 0;
    }
    // The operations are stored with the first-pushed at the MSB end.
    // We extract from MSB to LSB.
    let mut p = pcigar;
    // Shift so the first operation is at bits 30-31
    let free = pcigar_free_slots(pcigar);
    p <<= (free * 2) as u32;
    for _ in 0..length {
        let op = p >> 30;
        let ch = match op {
            PCIGAR_DELETION => b'D',
            PCIGAR_MISMATCH => b'X',
            PCIGAR_INSERTION => b'I',
            _ => b'?',
        };
        buffer.push(ch);
        p <<= 2;
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_pcigar() {
        assert_eq!(pcigar_get_length(PCIGAR_NULL), 0);
        assert_eq!(pcigar_free_slots(PCIGAR_NULL), PCIGAR_MAX_LENGTH);
    }

    #[test]
    fn test_push_and_length() {
        let mut p = PCIGAR_NULL;
        p = pcigar_push_back_ins(p);
        assert_eq!(pcigar_get_length(p), 1);
        p = pcigar_push_back_del(p);
        assert_eq!(pcigar_get_length(p), 2);
        p = pcigar_push_back_misms(p);
        assert_eq!(pcigar_get_length(p), 3);
        assert_eq!(pcigar_free_slots(p), PCIGAR_MAX_LENGTH - 3);
    }

    #[test]
    fn test_extract() {
        // Push insertion (0b11), then deletion (0b01)
        let mut p = PCIGAR_NULL;
        p = pcigar_push_back_ins(p); // 0b11
        p = pcigar_push_back_del(p); // 0b01
        // Extract should give the first pushed operation (insertion = 0b11)
        // But we need to shift to get it to bits 30-31
        // With 2 ops and 14 free slots, the MSB contains the first op
        // pcigar = 0b...1101 (ins=11, del=01 at LSB)
        // After shifting by free_slots*2 = 28, first op at bits 30-31
        let free = pcigar_free_slots(p);
        let shifted = p << ((free * 2) as u32);
        assert_eq!(shifted >> 30, PCIGAR_INSERTION);
    }

    #[test]
    fn test_unpack_round_trip() {
        let mut p = PCIGAR_NULL;
        p = pcigar_push_back_ins(p);
        p = pcigar_push_back_del(p);
        p = pcigar_push_back_misms(p);
        p = pcigar_push_back_ins(p);

        let mut buf = Vec::new();
        let len = pcigar_unpack(p, &mut buf);
        assert_eq!(len, 4);
        assert_eq!(&buf, b"IDXI");
    }

    #[test]
    fn test_full_pcigar() {
        let mut p = PCIGAR_NULL;
        for _ in 0..PCIGAR_MAX_LENGTH {
            p = pcigar_push_back_misms(p);
        }
        assert_eq!(pcigar_get_length(p), PCIGAR_MAX_LENGTH);
        assert_eq!(pcigar_free_slots(p), 0);
        assert!(pcigar_is_utilised(p, PCIGAR_FULL_MASK));
    }

    #[test]
    fn test_utilisation_masks() {
        let mut p = PCIGAR_NULL;
        // Push 9 operations to test half-full
        for _ in 0..9 {
            p = pcigar_push_back_del(p);
        }
        assert!(pcigar_is_utilised(p, PCIGAR_HALF_FULL_MASK));
        assert!(!pcigar_is_utilised(p, PCIGAR_ALMOST_FULL_MASK));
    }

    #[test]
    fn test_pop_front() {
        let p = pcigar_push_back_ins(PCIGAR_NULL);
        // After pop_front, shifting left discards MSB content but value changes
        let popped = pcigar_pop_front(p);
        // The original 2-bit op gets shifted out of range
        assert_ne!(popped, p);
    }
}
