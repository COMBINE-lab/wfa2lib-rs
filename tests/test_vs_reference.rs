//! Integration tests: validate Rust output against C reference outputs.

use std::fs;
use wfa2lib_rs::aligner::{AlignmentScope, WavefrontAligner};
use wfa2lib_rs::heuristic::HeuristicStrategy;
use wfa2lib_rs::penalties::{Affine2pPenalties, AffinePenalties, WavefrontPenalties};
use wfa2lib_rs::sequences::encode_packed2bits;

/// Parse the test sequence file: alternating lines starting with '>' (pattern) and '<' (text).
fn parse_sequences(path: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    let content = fs::read_to_string(path).expect("failed to read sequence file");
    let lines: Vec<&str> = content.lines().collect();
    assert!(
        lines.len() % 2 == 0,
        "sequence file must have even number of lines"
    );

    let mut pairs = Vec::new();
    for chunk in lines.chunks(2) {
        let pattern = chunk[0]
            .strip_prefix('>')
            .expect("expected '>' prefix")
            .as_bytes()
            .to_vec();
        let text = chunk[1]
            .strip_prefix('<')
            .expect("expected '<' prefix")
            .as_bytes()
            .to_vec();
        pairs.push((pattern, text));
    }
    pairs
}

/// Parse a reference alignment file: each line is "score\tCIGAR_RLE".
fn parse_alignments(path: &str) -> Vec<(i32, String)> {
    let content = fs::read_to_string(path).expect("failed to read alignment file");
    content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let mut parts = line.split('\t');
            let score: i32 = parts
                .next()
                .unwrap()
                .parse()
                .expect("failed to parse score");
            let cigar = parts.next().unwrap().to_string();
            (score, cigar)
        })
        .collect()
}

/// Parse the reference score file: one score per line, tab-separated (score\t-).
fn parse_scores(path: &str) -> Vec<i32> {
    let content = fs::read_to_string(path).expect("failed to read score file");
    content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let score_str = line.split('\t').next().unwrap();
            score_str.parse::<i32>().expect("failed to parse score")
        })
        .collect()
}

#[test]
fn test_edit_distance_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.edit.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        if score != *expected {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Edit distance score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_indel_distance_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.indel.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        if score != *expected {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Indel distance score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_edit_distance_cigars() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.edit.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);

        // Verify CIGAR is valid
        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != *exp_score || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Edit distance CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_indel_distance_cigars() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.indel.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_indel());
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);

        // Verify CIGAR is valid
        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != *exp_score || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Indel distance CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine_distance_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    // Default affine penalties: mismatch=4, gap_opening=6, gap_extension=2
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        // Reference scores are negative (-WFA_score), so negate for comparison
        let expected_positive = -expected;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine distance score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine_distance_cigars() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.affine.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    // Default affine penalties: mismatch=4, gap_opening=6, gap_extension=2
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);
        // Reference scores are negative
        let exp_score_positive = -exp_score;

        // Verify CIGAR is valid
        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != exp_score_positive || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score_positive,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine distance CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine2p_distance_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine2p.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    // Default affine2p penalties: M=0, X=4, O1=6, E1=2, O2=24, E2=1
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine2p(Affine2pPenalties {
        match_: 0,
        mismatch: 4,
        gap_opening1: 6,
        gap_extension1: 2,
        gap_opening2: 24,
        gap_extension2: 1,
    }));
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        // Reference scores are negative (-WFA_score), so negate for comparison
        let expected_positive = -expected;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine2p distance score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine2p_distance_cigars() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.affine2p.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    // Default affine2p penalties: M=0, X=4, O1=6, E1=2, O2=24, E2=1
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine2p(Affine2pPenalties {
        match_: 0,
        mismatch: 4,
        gap_opening1: 6,
        gap_extension1: 2,
        gap_opening2: 24,
        gap_extension2: 1,
    }));
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);
        // Reference scores are negative
        let exp_score_positive = -exp_score;

        // Verify CIGAR is valid
        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != exp_score_positive || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score_positive,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine2p distance CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// --- WF-Adaptive heuristic tests ---

#[test]
fn test_affine_wfadaptive_scores_pt0() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine.wfapt0.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    // Affine penalties: M=0, X=4, O=6, E=2 with WfAdaptive(10, 50, 1)
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    aligner.set_heuristic(HeuristicStrategy::WfAdaptive {
        min_wavefront_length: 10,
        max_distance_threshold: 50,
        steps_between_cutoffs: 1,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        let expected_positive = -expected;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine WfAdaptive(10,50,1) score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine_wfadaptive_cigars_pt0() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.affine.wfapt0.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    aligner.set_heuristic(HeuristicStrategy::WfAdaptive {
        min_wavefront_length: 10,
        max_distance_threshold: 50,
        steps_between_cutoffs: 1,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);
        let exp_score_positive = -exp_score;

        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != exp_score_positive || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score_positive,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine WfAdaptive(10,50,1) CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine_wfadaptive_scores_pt1() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine.wfapt1.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    // Affine penalties: M=0, X=4, O=6, E=2 with WfAdaptive(10, 50, 10)
    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    aligner.set_heuristic(HeuristicStrategy::WfAdaptive {
        min_wavefront_length: 10,
        max_distance_threshold: 50,
        steps_between_cutoffs: 10,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_end2end(pattern, text);
        let expected_positive = -expected;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine WfAdaptive(10,50,10) score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_affine_wfadaptive_cigars_pt1() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.affine.wfapt1.alg";

    let pairs = parse_sequences(seq_path);
    let expected = parse_alignments(ref_path);
    assert_eq!(pairs.len(), expected.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    aligner.set_heuristic(HeuristicStrategy::WfAdaptive {
        min_wavefront_length: 10,
        max_distance_threshold: 50,
        steps_between_cutoffs: 10,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), (exp_score, exp_cigar))) in
        pairs.iter().zip(expected.iter()).enumerate()
    {
        let score = aligner.align_end2end(pattern, text);
        let cigar = aligner.cigar();
        let cigar_rle = cigar.to_string_rle(true);
        let exp_score_positive = -exp_score;

        if let Err(e) = cigar.check_alignment(pattern, text) {
            failures.push(format!(
                "  pair {}: CIGAR validation failed: {} (cigar={})",
                i + 1,
                e,
                cigar_rle,
            ));
            continue;
        }

        if score != exp_score_positive || cigar_rle != *exp_cigar {
            failures.push(format!(
                "  pair {}: score={}/{}, cigar={}/{}",
                i + 1,
                score,
                exp_score_positive,
                cigar_rle,
                exp_cigar,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Affine WfAdaptive(10,50,10) CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// --- BiWFA (Bidirectional Wavefront Alignment) tests ---

#[test]
fn test_biwfa_edit_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.edit.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_edit());
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_biwfa(pattern, text);
        if score != *expected {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected,
                pattern.len(),
                text.len()
            ));
        }
        // Also verify CIGAR
        if let Err(e) = aligner.cigar().check_alignment(pattern, text) {
            failures.push(format!("  pair {}: CIGAR invalid: {}", i + 1, e));
        }
    }

    if !failures.is_empty() {
        panic!(
            "BiWFA edit score/CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_biwfa_affine_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    }));
    let mut failures = Vec::new();

    let mut score_failures = Vec::new();
    let mut cigar_failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_biwfa(pattern, text);
        let expected_positive = -expected;
        if score != expected_positive {
            score_failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
        if let Err(e) = aligner.cigar().check_alignment(pattern, text) {
            cigar_failures.push(format!("  pair {}: CIGAR invalid: {}", i + 1, e));
        }
    }
    failures.extend(score_failures.iter().cloned());
    failures.extend(cigar_failures.iter().cloned());

    if !failures.is_empty() {
        panic!(
            "BiWFA affine: {} score mismatches, {} CIGAR failures (out of {}):\n{}",
            score_failures.len(),
            cigar_failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_biwfa_affine2p_scores() {
    let seq_path = "WFA2-lib/tests/wfa.utest.seq";
    let ref_path = "WFA2-lib/tests/wfa.utest.check/test.score.affine2p.alg";

    let pairs = parse_sequences(seq_path);
    let expected_scores = parse_scores(ref_path);
    assert_eq!(pairs.len(), expected_scores.len());

    let mut aligner = WavefrontAligner::new(WavefrontPenalties::new_affine2p(Affine2pPenalties {
        match_: 0,
        mismatch: 4,
        gap_opening1: 6,
        gap_extension1: 2,
        gap_opening2: 24,
        gap_extension2: 1,
    }));
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected)) in pairs.iter().zip(expected_scores.iter()).enumerate() {
        let score = aligner.align_biwfa(pattern, text);
        let expected_positive = -expected;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: got {}, expected {} (plen={}, tlen={})",
                i + 1,
                score,
                expected_positive,
                pattern.len(),
                text.len()
            ));
        }
        if let Err(e) = aligner.cigar().check_alignment(pattern, text) {
            failures.push(format!("  pair {}: CIGAR invalid: {}", i + 1, e));
        }
    }

    if !failures.is_empty() {
        panic!(
            "BiWFA affine2p score/CIGAR mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// =============================================================================
// Lambda mode tests: custom match function must produce same scores as ASCII
// =============================================================================

#[test]
fn test_lambda_edit_scores() {
    let pairs = parse_sequences("WFA2-lib/tests/wfa.utest.seq");
    let ref_scores = parse_scores("WFA2-lib/tests/wfa.utest.check/test.score.edit.alg");

    let penalties = WavefrontPenalties::new_edit();
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected_score)) in pairs.iter().zip(ref_scores.iter()).enumerate() {
        let p = pattern.clone();
        let t = text.clone();
        let plen = p.len() as i32;
        let tlen = t.len() as i32;

        let match_funct =
            move |ppos: i32, tpos: i32| -> bool { p[ppos as usize] == t[tpos as usize] };

        let mut aligner = WavefrontAligner::new(penalties.clone());
        let score = aligner.align_lambda(Box::new(match_funct), plen, tlen);

        if score != *expected_score {
            failures.push(format!(
                "  pair {}: lambda={} expected={}",
                i, score, expected_score
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Lambda edit score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_lambda_affine_scores() {
    let pairs = parse_sequences("WFA2-lib/tests/wfa.utest.seq");
    let ref_scores = parse_scores("WFA2-lib/tests/wfa.utest.check/test.score.affine.alg");

    let penalties = WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected_score)) in pairs.iter().zip(ref_scores.iter()).enumerate() {
        let p = pattern.clone();
        let t = text.clone();
        let plen = p.len() as i32;
        let tlen = t.len() as i32;

        let match_funct =
            move |ppos: i32, tpos: i32| -> bool { p[ppos as usize] == t[tpos as usize] };

        let mut aligner = WavefrontAligner::new(penalties.clone());
        let score = aligner.align_lambda(Box::new(match_funct), plen, tlen);

        let expected_positive = -expected_score;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: lambda={} expected={}",
                i, score, expected_positive
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Lambda affine score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// =============================================================================
// Packed 2-bit DNA tests: packed input must produce same scores as ASCII
// =============================================================================

#[test]
fn test_packed2bits_affine_scores() {
    let pairs = parse_sequences("WFA2-lib/tests/wfa.utest.seq");
    let ref_scores = parse_scores("WFA2-lib/tests/wfa.utest.check/test.score.affine.alg");

    let penalties = WavefrontPenalties::new_affine(AffinePenalties {
        match_: 0,
        mismatch: 4,
        gap_opening: 6,
        gap_extension: 2,
    });
    let mut failures = Vec::new();

    for (i, ((pattern, text), expected_score)) in pairs.iter().zip(ref_scores.iter()).enumerate() {
        let ppacked = encode_packed2bits(pattern);
        let tpacked = encode_packed2bits(text);

        let mut aligner = WavefrontAligner::new(penalties.clone());
        let score = aligner.align_packed2bits(&ppacked, pattern.len(), &tpacked, text.len());

        let expected_positive = -expected_score;
        if score != expected_positive {
            failures.push(format!(
                "  pair {}: packed2bits={} expected={}",
                i, score, expected_positive
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Packed2bits affine score mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}
