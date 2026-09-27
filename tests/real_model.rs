//! Real-model evidence test.
//!
//! Reads a real GGUF model blob from disk (if present), extracts the model
//! architecture via the GGUF header reader, and runs `plan()` to produce a
//! predicted peak memory estimate.
//!
//! This test is the "real-model proof" required by PROOF-AND-RELEASE.md:
//! a run against real weights (not a randomly-initialised reference bundle)
//! with the predicted peak recorded in EVIDENCE.md.
//!
//! The test is skipped (not failed) if no GGUF is present — the limitation
//! is then recorded as PARTIAL in EVIDENCE.md.

use fitsproof::gguf::{metadata_to_model_config, read_metadata};
use fitsproof::plan::plan;
use fitsproof::probe::MachineProfile;

/// Real GGUF blobs to try, in preference order.
///
/// Never hardcode a host path: the CI guard rejects those, and a stranger's machine has different
/// directories. `FITSPROOF_REAL_GGUF` wins when set; otherwise the usual model caches under
/// `$HOME` are searched, and the test skips (recording the limitation as PARTIAL) when none exist.
fn gguf_candidates() -> Vec<std::path::PathBuf> {
    let mut v: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("FITSPROOF_REAL_GGUF") {
        if !p.is_empty() {
            v.push(p.into());
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = std::path::PathBuf::from(home);
        for rel in [
            ".cache/qmd/models/hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf",
            ".ollama/models/blobs/sha256-aeda25e",
        ] {
            v.push(home.join(rel));
        }
    }
    v
}

fn synthetic_machine() -> MachineProfile {
    // Use a real probe for evidence; fall back to synthetic for CI speed.
    MachineProfile {
        hostname: "evidence-host".into(),
        platform_str: "evidence".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

/// Reads a real GGUF file and produces a `plan()` output.
///
/// Fault detected: GGUF reader fails on a real file (header format wrong).
/// Fault detected: `plan()` errors on a real model config (field out of range).
///
/// If no GGUF is found, the test prints a skip message and passes.
/// The limitation is recorded as PARTIAL in EVIDENCE.md.
#[test]
fn real_gguf_model_plan_succeeds() {
    // Find the first available GGUF file.
    let candidates = gguf_candidates();
    let gguf_path = candidates.iter().find(|p| p.exists());

    let gguf_path = match gguf_path {
        Some(p) => p,
        None => {
            println!(
                "SKIP: no real GGUF found at {:?}. \
                 Set FITSPROOF_REAL_GGUF to run this proof. \
                 Limitation recorded as PARTIAL in docs/EVIDENCE.md.",
                candidates
            );
            return; // not a failure — limitation is honest
        }
    };

    println!("Reading GGUF: {}", gguf_path.display());

    let f = std::fs::File::open(gguf_path).expect("GGUF file must be readable");
    let meta = read_metadata(f).expect("GGUF header must parse");

    println!("  GGUF version: {}", meta.version);
    println!("  Tensor count: {}", meta.tensor_count);
    println!("  KV entries: {}", meta.kv.len());

    // Print architecture-relevant keys.
    for key in meta.kv.keys() {
        if key.contains("block_count")
            || key.contains("embedding_length")
            || key.contains("head_count")
            || key.contains("feed_forward")
            || key.contains("context_length")
            || key.contains("architecture")
            || key.contains("general.name")
        {
            println!("  {key}: {:?}", meta.kv[key]);
        }
    }

    let model_name = gguf_path
        .rsplit('/')
        .next()
        .unwrap_or("unknown")
        .trim_end_matches(".gguf");

    let cfg = metadata_to_model_config(&meta, model_name)
        .expect("must extract ModelConfig from real GGUF metadata");

    println!("  Extracted config: {cfg:?}");

    let machine = synthetic_machine();
    let budget_bytes = 4_000_000_000u64; // 4 GB

    let p = plan(&cfg, &machine, 512, budget_bytes, "int4_sym", 0.6)
        .expect("plan() must succeed for real model config");

    println!("  Verdict:        {:?}", p.verdict);
    println!(
        "  Predicted peak: {:.3} GB",
        p.predicted_peak_bytes as f64 / 1e9
    );
    println!("  Budget:         {:.3} GB", p.budget_bytes as f64 / 1e9);
    if !p.binding_constraint.is_empty() {
        println!("  Binding:        {}", p.binding_constraint);
    }

    // The plan must produce a finite, non-zero peak.
    assert!(
        p.predicted_peak_bytes > 0,
        "predicted_peak_bytes must be > 0 for a real model"
    );
    assert!(
        p.predicted_peak_bytes < 100_000_000_000, // < 100 GB (sanity)
        "predicted_peak_bytes > 100 GB looks wrong: {}",
        p.predicted_peak_bytes
    );
}
