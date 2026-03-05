use std::fs;
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
use wfa2lib_rs::aligner::{AlignmentScope, WavefrontAligner};
use wfa2lib_rs::heuristic::HeuristicStrategy;
use wfa2lib_rs::penalties::{
    Affine2pPenalties, AffinePenalties, DistanceMetric, LinearPenalties, WavefrontPenalties,
};

#[derive(Parser)]
#[command(name = "align_benchmark")]
struct Cli {
    /// Input sequence file
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Output alignment file
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Algorithm
    #[arg(short = 'a', long, default_value = "gap-affine-wfa")]
    algorithm: String,

    /// Compute score only (no CIGAR)
    #[arg(long)]
    wfa_score_only: bool,

    /// Alignment span: global, extension, ends-free
    #[arg(long)]
    wfa_span: Option<String>,

    /// Memory mode: high, med, low, ultralow
    #[arg(long)]
    wfa_memory: Option<String>,

    /// Linear penalties: M,X,I
    #[arg(short = 'p', long)]
    linear_penalties: Option<String>,

    /// Affine penalties: M,X,O,E
    #[arg(short = 'g', long)]
    affine_penalties: Option<String>,

    /// Affine 2-piece penalties: M,X,O1,E1,O2,E2
    #[arg(long)]
    affine2p_penalties: Option<String>,

    /// Heuristic strategy
    #[arg(long)]
    wfa_heuristic: Option<String>,

    /// Heuristic parameters
    #[arg(long)]
    wfa_heuristic_parameters: Option<String>,

    /// Maximum alignment steps
    #[arg(long)]
    wfa_max_steps: Option<i32>,

    /// Check alignment correctness
    #[arg(short = 'c', long)]
    check: Option<String>,

    /// Verbosity level
    #[arg(short = 'v', long)]
    verbose: Option<i32>,
}

fn parse_sequences(path: &PathBuf) -> Vec<(Vec<u8>, Vec<u8>)> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading input file {:?}: {}", path, e);
        std::process::exit(1);
    });
    let lines: Vec<&str> = content.lines().collect();
    let mut pairs = Vec::new();
    for chunk in lines.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let pattern = chunk[0]
            .strip_prefix('>')
            .unwrap_or(chunk[0])
            .as_bytes()
            .to_vec();
        let text = chunk[1]
            .strip_prefix('<')
            .unwrap_or(chunk[1])
            .as_bytes()
            .to_vec();
        pairs.push((pattern, text));
    }
    pairs
}

fn parse_penalty_values(s: &str) -> Vec<i32> {
    s.split(',')
        .map(|v| v.trim().parse::<i32>().expect("Invalid penalty value"))
        .collect()
}

fn build_penalties(cli: &Cli) -> WavefrontPenalties {
    match cli.algorithm.as_str() {
        "indel-wfa" => WavefrontPenalties::new_indel(),
        "edit-wfa" => WavefrontPenalties::new_edit(),
        "gap-linear-wfa" => {
            if let Some(ref p) = cli.linear_penalties {
                let v = parse_penalty_values(p);
                WavefrontPenalties::new_linear(LinearPenalties {
                    match_: v[0],
                    mismatch: v[1],
                    indel: v[2],
                })
            } else {
                WavefrontPenalties::new_linear(LinearPenalties {
                    match_: 0,
                    mismatch: 4,
                    indel: 2,
                })
            }
        }
        "gap-affine-wfa" => {
            if let Some(ref p) = cli.affine_penalties {
                let v = parse_penalty_values(p);
                WavefrontPenalties::new_affine(AffinePenalties {
                    match_: v[0],
                    mismatch: v[1],
                    gap_opening: v[2],
                    gap_extension: v[3],
                })
            } else {
                WavefrontPenalties::new_affine(AffinePenalties {
                    match_: 0,
                    mismatch: 4,
                    gap_opening: 6,
                    gap_extension: 2,
                })
            }
        }
        "gap-affine2p-wfa" => {
            if let Some(ref p) = cli.affine2p_penalties {
                let v = parse_penalty_values(p);
                WavefrontPenalties::new_affine2p(Affine2pPenalties {
                    match_: v[0],
                    mismatch: v[1],
                    gap_opening1: v[2],
                    gap_extension1: v[3],
                    gap_opening2: v[4],
                    gap_extension2: v[5],
                })
            } else {
                WavefrontPenalties::new_affine2p(Affine2pPenalties {
                    match_: 0,
                    mismatch: 4,
                    gap_opening1: 6,
                    gap_extension1: 2,
                    gap_opening2: 24,
                    gap_extension2: 1,
                })
            }
        }
        other => {
            eprintln!("Unsupported algorithm: {}", other);
            std::process::exit(1);
        }
    }
}

fn parse_heuristic(name: &str, params: Option<&str>) -> HeuristicStrategy {
    let p: Vec<i32> = if let Some(params) = params {
        params
            .split(',')
            .map(|s| s.trim().parse().expect("Invalid heuristic parameter"))
            .collect()
    } else {
        Vec::new()
    };
    match name {
        "none" => HeuristicStrategy::None,
        "wfa-adaptive" => HeuristicStrategy::WfAdaptive {
            min_wavefront_length: p[0],
            max_distance_threshold: p[1],
            steps_between_cutoffs: p[2],
        },
        "xdrop" => HeuristicStrategy::XDrop { xdrop: p[0] },
        "zdrop" => HeuristicStrategy::ZDrop { zdrop: p[0] },
        "banded-static" => HeuristicStrategy::BandedStatic {
            min_k: p[0],
            max_k: p[1],
        },
        "banded-adaptive" => HeuristicStrategy::BandedAdaptive {
            min_k: p[0],
            max_k: p[1],
            steps_between_cutoffs: p[2],
        },
        other => {
            eprintln!("Unknown heuristic: {}", other);
            std::process::exit(1);
        }
    }
}

fn compute_output_score(
    aligner: &WavefrontAligner,
    wfa_score: i32,
    penalties: &WavefrontPenalties,
    score_only: bool,
    pattern_len: i32,
    text_len: i32,
) -> i32 {
    if !score_only {
        // CIGAR mode: compute score from CIGAR using original penalties
        match penalties.distance_metric {
            DistanceMetric::Indel | DistanceMetric::Edit => aligner.cigar().score_edit(),
            DistanceMetric::GapLinear => aligner
                .cigar()
                .score_gap_linear(penalties.linear_penalties.as_ref().unwrap()),
            DistanceMetric::GapAffine => aligner
                .cigar()
                .score_gap_affine(penalties.affine_penalties.as_ref().unwrap()),
            DistanceMetric::GapAffine2p => aligner
                .cigar()
                .score_gap_affine2p(penalties.affine2p_penalties.as_ref().unwrap()),
        }
    } else {
        // Score-only mode
        match penalties.distance_metric {
            DistanceMetric::Indel | DistanceMetric::Edit => wfa_score,
            _ => {
                if penalties.match_ < 0 {
                    // Non-zero match reward: convert WFA score to SW score via Eizenga's formula
                    penalties.wf_score_to_sw_score(pattern_len, text_len, wfa_score)
                } else {
                    -wfa_score
                }
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let pairs = parse_sequences(&cli.input);
    let penalties = build_penalties(&cli);
    let score_only = cli.wfa_score_only;
    let use_biwfa = cli.wfa_memory.as_deref() == Some("ultralow");
    let do_check = cli.check.is_some() && !score_only;

    let mut aligner = WavefrontAligner::new(penalties.clone());

    if !score_only {
        aligner.alignment_scope = AlignmentScope::ComputeAlignment;
    }

    if let Some(ref heuristic) = cli.wfa_heuristic {
        let strategy = parse_heuristic(heuristic, cli.wfa_heuristic_parameters.as_deref());
        aligner.set_heuristic(strategy);
    }

    if let Some(max_steps) = cli.wfa_max_steps {
        aligner.set_max_alignment_steps(max_steps);
    }

    let mut output_lines: Vec<String> = Vec::with_capacity(pairs.len());
    let mut correct_count = 0u32;
    let total = pairs.len() as u32;

    let mut align_nanos: u64 = 0;

    for (pattern, text) in &pairs {
        let t0 = std::time::Instant::now();
        let wfa_score = if use_biwfa && !score_only {
            aligner.align_biwfa(pattern, text)
        } else {
            aligner.align_end2end(pattern, text)
        };
        align_nanos += t0.elapsed().as_nanos() as u64;

        let output_score = compute_output_score(
            &aligner,
            wfa_score,
            &penalties,
            score_only,
            pattern.len() as i32,
            text.len() as i32,
        );

        let cigar_str = if score_only {
            "-".to_string()
        } else {
            aligner.cigar().to_string_rle(true)
        };

        output_lines.push(format!("{}\t{}", output_score, cigar_str));

        if do_check && !score_only {
            match aligner.cigar().check_alignment(pattern, text) {
                Ok(()) => correct_count += 1,
                Err(e) => {
                    if cli.verbose.unwrap_or(0) > 0 {
                        eprintln!("Alignment check failed: {}", e);
                    }
                }
            }
        }
    }

    let align_ms = align_nanos as f64 / 1_000_000.0;
    eprintln!(
        "Time.Alignment {:.2} ms ({} calls, {:.2} us/call)",
        align_ms,
        total,
        align_ms * 1000.0 / total as f64
    );

    // Write output
    if let Some(ref output_path) = cli.output {
        let mut f = fs::File::create(output_path).unwrap_or_else(|e| {
            eprintln!("Error creating output file {:?}: {}", output_path, e);
            std::process::exit(1);
        });
        for line in &output_lines {
            writeln!(f, "{}", line).unwrap();
        }
    } else {
        for line in &output_lines {
            println!("{}", line);
        }
    }

    // Print check results
    if do_check {
        eprintln!("Alignments.Correct {}/{}", correct_count, total);
    }
}
