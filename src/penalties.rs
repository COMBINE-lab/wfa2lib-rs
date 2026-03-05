//! Penalty/distance metric types for wavefront alignment.
//!
//! WFA supports multiple distance metrics, each with different penalty structures.
//! Penalties use the convention: match <= 0, all others > 0 (cost representation).
//! When the match score is negative (reward), Eizenga's formula is used to convert
//! to the internal WFA representation.

use std::fmt;

/// Distance metric used for alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Longest Common Subsequence (indel only, no mismatches).
    Indel,
    /// Levenshtein edit distance (unit cost for substitutions and indels).
    Edit,
    /// Gap-linear (Needleman-Wunsch): separate mismatch and indel costs.
    GapLinear,
    /// Gap-affine (Smith-Waterman-Gotoh): gap opening + extension.
    GapAffine,
    /// Gap-affine 2-piece: two-piece concave gap penalty.
    GapAffine2p,
}

/// Gap-linear penalties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearPenalties {
    /// Match score (typically M <= 0; 0 = no reward, negative = reward).
    pub match_: i32,
    /// Mismatch penalty (X > 0).
    pub mismatch: i32,
    /// Indel penalty (I > 0).
    pub indel: i32,
}

/// Gap-affine penalties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffinePenalties {
    /// Match score (typically M <= 0).
    pub match_: i32,
    /// Mismatch penalty (X > 0).
    pub mismatch: i32,
    /// Gap opening penalty (O >= 0).
    pub gap_opening: i32,
    /// Gap extension penalty (E > 0).
    pub gap_extension: i32,
}

/// Gap-affine 2-piece penalties.
/// Usually concave: O1 + E1 < O2 + E2 and E1 > E2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Affine2pPenalties {
    /// Match score (typically M <= 0).
    pub match_: i32,
    /// Mismatch penalty (X > 0).
    pub mismatch: i32,
    /// First piece gap opening (O1 >= 0).
    pub gap_opening1: i32,
    /// First piece gap extension (E1 > 0).
    pub gap_extension1: i32,
    /// Second piece gap opening (O2 >= 0).
    pub gap_opening2: i32,
    /// Second piece gap extension (E2 > 0).
    pub gap_extension2: i32,
}

/// Affine matrix type for backtrace (gap-affine model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineMatrixType {
    M,
    I,
    D,
}

/// Affine 2-piece matrix type for backtrace (gap-affine 2-piece model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affine2pMatrixType {
    M,
    I1,
    I2,
    D1,
    D2,
}

/// Unified wavefront penalties structure.
///
/// Stores the distance metric, the internal (possibly Eizenga-adjusted) penalty values,
/// and the original user-supplied penalties.
#[derive(Debug, Clone)]
pub struct WavefrontPenalties {
    /// Distance metric in use.
    pub distance_metric: DistanceMetric,
    // Internal penalty values (after Eizenga adjustment if match < 0)
    /// Match score (M <= 0).
    pub match_: i32,
    /// Mismatch penalty (X > 0).
    pub mismatch: i32,
    /// Gap opening 1 (O1 >= 0).
    pub gap_opening1: i32,
    /// Gap extension 1 (E1 > 0).
    pub gap_extension1: i32,
    /// Gap opening 2 (O2 >= 0), -1 if unused.
    pub gap_opening2: i32,
    /// Gap extension 2 (E2 > 0), -1 if unused.
    pub gap_extension2: i32,
    // Original user-supplied penalties
    /// Original linear penalties (if applicable).
    pub linear_penalties: Option<LinearPenalties>,
    /// Original affine penalties (if applicable).
    pub affine_penalties: Option<AffinePenalties>,
    /// Original affine2p penalties (if applicable).
    pub affine2p_penalties: Option<Affine2pPenalties>,
    /// Original gap-extension value (used for z-drop).
    pub internal_gap_e: i32,
}

impl WavefrontPenalties {
    /// Create penalties for indel (LCS) distance.
    pub fn new_indel() -> Self {
        Self {
            distance_metric: DistanceMetric::Indel,
            match_: 0,
            mismatch: -1,
            gap_opening1: 1,
            gap_extension1: -1,
            gap_opening2: -1,
            gap_extension2: -1,
            linear_penalties: None,
            affine_penalties: None,
            affine2p_penalties: None,
            internal_gap_e: 1,
        }
    }

    /// Create penalties for edit (Levenshtein) distance.
    pub fn new_edit() -> Self {
        Self {
            distance_metric: DistanceMetric::Edit,
            match_: 0,
            mismatch: 1,
            gap_opening1: 1,
            gap_extension1: -1,
            gap_opening2: -1,
            gap_extension2: -1,
            linear_penalties: None,
            affine_penalties: None,
            affine2p_penalties: None,
            internal_gap_e: 1,
        }
    }

    /// Create penalties for gap-linear distance.
    ///
    /// # Panics
    /// Panics if match > 0, mismatch <= 0, or indel <= 0.
    pub fn new_linear(penalties: LinearPenalties) -> Self {
        assert!(
            penalties.match_ <= 0,
            "Match score must be negative or zero (M={})",
            penalties.match_
        );
        assert!(
            penalties.mismatch > 0 && penalties.indel > 0,
            "Penalties (X={},I={}) must be (X>0,I>0)",
            penalties.mismatch,
            penalties.indel
        );
        let (match_, mismatch, gap_opening1) = if penalties.match_ < 0 {
            // Eizenga's formula adjustment
            (
                penalties.match_,
                2 * penalties.mismatch - 2 * penalties.match_,
                2 * penalties.indel - penalties.match_,
            )
        } else {
            (0, penalties.mismatch, penalties.indel)
        };
        Self {
            distance_metric: DistanceMetric::GapLinear,
            match_,
            mismatch,
            gap_opening1,
            gap_extension1: -1,
            gap_opening2: -1,
            gap_extension2: -1,
            linear_penalties: Some(penalties),
            affine_penalties: None,
            affine2p_penalties: None,
            internal_gap_e: penalties.indel,
        }
    }

    /// Create penalties for gap-affine distance.
    ///
    /// # Panics
    /// Panics if match > 0, mismatch <= 0, gap_opening < 0, or gap_extension <= 0.
    pub fn new_affine(penalties: AffinePenalties) -> Self {
        assert!(
            penalties.match_ <= 0,
            "Match score must be negative or zero (M={})",
            penalties.match_
        );
        assert!(
            penalties.mismatch > 0 && penalties.gap_opening >= 0 && penalties.gap_extension > 0,
            "Penalties (X={},O={},E={}) must be (X>0,O>=0,E>0)",
            penalties.mismatch,
            penalties.gap_opening,
            penalties.gap_extension
        );
        let (match_, mismatch, gap_opening1, gap_extension1) = if penalties.match_ < 0 {
            (
                penalties.match_,
                2 * penalties.mismatch - 2 * penalties.match_,
                2 * penalties.gap_opening,
                2 * penalties.gap_extension - penalties.match_,
            )
        } else {
            (
                0,
                penalties.mismatch,
                penalties.gap_opening,
                penalties.gap_extension,
            )
        };
        Self {
            distance_metric: DistanceMetric::GapAffine,
            match_,
            mismatch,
            gap_opening1,
            gap_extension1,
            gap_opening2: -1,
            gap_extension2: -1,
            linear_penalties: None,
            affine_penalties: Some(penalties),
            affine2p_penalties: None,
            internal_gap_e: penalties.gap_extension,
        }
    }

    /// Create penalties for gap-affine 2-piece distance.
    ///
    /// # Panics
    /// Panics if penalty constraints are violated.
    pub fn new_affine2p(penalties: Affine2pPenalties) -> Self {
        assert!(
            penalties.match_ <= 0,
            "Match score must be negative or zero (M={})",
            penalties.match_
        );
        assert!(
            penalties.mismatch > 0
                && penalties.gap_opening1 >= 0
                && penalties.gap_extension1 > 0
                && penalties.gap_opening2 >= 0
                && penalties.gap_extension2 > 0,
            "Penalties (X={},O1={},E1={},O2={},E2={}) must be (X>0,O1>=0,E1>0,O2>=0,E2>0)",
            penalties.mismatch,
            penalties.gap_opening1,
            penalties.gap_extension1,
            penalties.gap_opening2,
            penalties.gap_extension2
        );
        let (match_, mismatch, gap_opening1, gap_extension1, gap_opening2, gap_extension2) =
            if penalties.match_ < 0 {
                (
                    penalties.match_,
                    2 * penalties.mismatch - 2 * penalties.match_,
                    2 * penalties.gap_opening1,
                    2 * penalties.gap_extension1 - penalties.match_,
                    2 * penalties.gap_opening2,
                    2 * penalties.gap_extension2 - penalties.match_,
                )
            } else {
                (
                    0,
                    penalties.mismatch,
                    penalties.gap_opening1,
                    penalties.gap_extension1,
                    penalties.gap_opening2,
                    penalties.gap_extension2,
                )
            };
        Self {
            distance_metric: DistanceMetric::GapAffine2p,
            match_,
            mismatch,
            gap_opening1,
            gap_extension1,
            gap_opening2,
            gap_extension2,
            linear_penalties: None,
            affine_penalties: None,
            affine2p_penalties: Some(penalties),
            internal_gap_e: penalties.gap_extension1,
        }
    }

    /// Compute SW-score equivalent using Eizenga's formula.
    /// sw_score = (swg_match * (plen + tlen) - wf_score) / 2
    pub fn wf_score_to_sw_score(&self, plen: i32, tlen: i32, wf_score: i32) -> i32 {
        let swg_match = -self.match_;
        (swg_match * (plen + tlen) - wf_score) / 2
    }
}

impl fmt::Display for WavefrontPenalties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.distance_metric {
            DistanceMetric::Indel => write!(f, "(Indel,0,inf,1)"),
            DistanceMetric::Edit => write!(f, "(Edit,0,1,1)"),
            DistanceMetric::GapLinear => {
                let p = self.linear_penalties.as_ref().unwrap();
                write!(f, "(GapLinear,{},{},{})", p.match_, p.mismatch, p.indel)
            }
            DistanceMetric::GapAffine => {
                let p = self.affine_penalties.as_ref().unwrap();
                write!(
                    f,
                    "(GapAffine,{},{},{},{})",
                    p.match_, p.mismatch, p.gap_opening, p.gap_extension
                )
            }
            DistanceMetric::GapAffine2p => {
                let p = self.affine2p_penalties.as_ref().unwrap();
                write!(
                    f,
                    "(GapAffine2p,{},{},{},{},{},{})",
                    p.match_,
                    p.mismatch,
                    p.gap_opening1,
                    p.gap_extension1,
                    p.gap_opening2,
                    p.gap_extension2
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indel_penalties() {
        let p = WavefrontPenalties::new_indel();
        assert_eq!(p.distance_metric, DistanceMetric::Indel);
        assert_eq!(p.match_, 0);
        assert_eq!(p.gap_opening1, 1);
    }

    #[test]
    fn test_edit_penalties() {
        let p = WavefrontPenalties::new_edit();
        assert_eq!(p.distance_metric, DistanceMetric::Edit);
        assert_eq!(p.mismatch, 1);
        assert_eq!(p.gap_opening1, 1);
    }

    #[test]
    fn test_linear_penalties_no_match_reward() {
        let p = WavefrontPenalties::new_linear(LinearPenalties {
            match_: 0,
            mismatch: 4,
            indel: 2,
        });
        assert_eq!(p.mismatch, 4);
        assert_eq!(p.gap_opening1, 2);
        assert_eq!(p.gap_extension1, -1);
    }

    #[test]
    fn test_linear_penalties_with_match_reward() {
        // match=-1, mismatch=4, indel=2
        // Eizenga: X' = 2*4 - 2*(-1) = 10, O' = 2*2 - (-1) = 5
        let p = WavefrontPenalties::new_linear(LinearPenalties {
            match_: -1,
            mismatch: 4,
            indel: 2,
        });
        assert_eq!(p.match_, -1);
        assert_eq!(p.mismatch, 10);
        assert_eq!(p.gap_opening1, 5);
    }

    #[test]
    fn test_affine_penalties_no_match_reward() {
        let p = WavefrontPenalties::new_affine(AffinePenalties {
            match_: 0,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        assert_eq!(p.mismatch, 4);
        assert_eq!(p.gap_opening1, 6);
        assert_eq!(p.gap_extension1, 2);
    }

    #[test]
    fn test_affine_penalties_with_match_reward() {
        // match=-1, mismatch=4, O=6, E=2
        // Eizenga: X'=2*4-2*(-1)=10, O'=2*6=12, E'=2*2-(-1)=5
        let p = WavefrontPenalties::new_affine(AffinePenalties {
            match_: -1,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        assert_eq!(p.match_, -1);
        assert_eq!(p.mismatch, 10);
        assert_eq!(p.gap_opening1, 12);
        assert_eq!(p.gap_extension1, 5);
    }

    #[test]
    fn test_affine2p_penalties() {
        let p = WavefrontPenalties::new_affine2p(Affine2pPenalties {
            match_: 0,
            mismatch: 4,
            gap_opening1: 6,
            gap_extension1: 2,
            gap_opening2: 24,
            gap_extension2: 1,
        });
        assert_eq!(p.mismatch, 4);
        assert_eq!(p.gap_opening1, 6);
        assert_eq!(p.gap_extension1, 2);
        assert_eq!(p.gap_opening2, 24);
        assert_eq!(p.gap_extension2, 1);
    }

    #[test]
    fn test_display() {
        assert_eq!(
            WavefrontPenalties::new_indel().to_string(),
            "(Indel,0,inf,1)"
        );
        assert_eq!(WavefrontPenalties::new_edit().to_string(), "(Edit,0,1,1)");
        assert_eq!(
            WavefrontPenalties::new_linear(LinearPenalties {
                match_: 0,
                mismatch: 4,
                indel: 2
            })
            .to_string(),
            "(GapLinear,0,4,2)"
        );
        assert_eq!(
            WavefrontPenalties::new_affine(AffinePenalties {
                match_: 0,
                mismatch: 4,
                gap_opening: 6,
                gap_extension: 2
            })
            .to_string(),
            "(GapAffine,0,4,6,2)"
        );
    }

    #[test]
    #[should_panic(expected = "Match score must be negative or zero")]
    fn test_linear_positive_match_panics() {
        WavefrontPenalties::new_linear(LinearPenalties {
            match_: 1,
            mismatch: 4,
            indel: 2,
        });
    }

    #[test]
    #[should_panic(expected = "Penalties")]
    fn test_affine_zero_mismatch_panics() {
        WavefrontPenalties::new_affine(AffinePenalties {
            match_: 0,
            mismatch: 0,
            gap_opening: 6,
            gap_extension: 2,
        });
    }

    #[test]
    fn test_sw_score_conversion() {
        // With match reward of -2: sw_match = 2
        // sw_score = (2 * (10 + 12) - 100) / 2 = (44 - 100) / 2 = -28
        let p = WavefrontPenalties::new_affine(AffinePenalties {
            match_: -2,
            mismatch: 4,
            gap_opening: 6,
            gap_extension: 2,
        });
        let sw = p.wf_score_to_sw_score(10, 12, 100);
        assert_eq!(sw, (2 * (10 + 12) - 100) / 2);
    }
}
