//! Adversarial / byzantine test suite.
//!
//! Every test in this file is named for the fault it detects. The QUALITY-CONTRACT §1
//! (vacuity ban) requires each test to describe a realistic fault a naive implementation
//! would miss. The fault is described in the module-level table and repeated above each test.
//!
//! # Fault index
//!
//! | Test | Fault detected |
//! |------|----------------|
//! | `budget_zero_returns_invalid_budget_error` | A budget of 0 bytes must return PlanError::InvalidBudget, never silently produce DoesNotFit or Fits. |
//! | `unknown_quant_returns_unknown_quant_error` | An unrecognised quant string must return PlanError::UnknownQuant, never fall through to fp32 silently. |
//! | `context_len_zero_returns_invalid_context_error` | context_len=0 must return PlanError::InvalidContextLen. |
//! | `budget_exactly_at_predicted_peak_admits` | Budget == predicted peak must not return DoesNotFit (rounding error with > vs >= causes incorrect refusal). |
//! | `budget_one_byte_below_peak_not_fits` | Budget strictly below predicted peak must not return Verdict::Fits. |
//! | `integer_overflow_context_len` | context_len near u32::MAX must not overflow or wrap to a small value in kv_cache_bytes. |
//! | `large_vocab_weight_bytes_not_zero` | A vocab of 1M must produce weight_bytes ≥ 256 MB, not overflow to 0. |
//! | `zero_layers_does_not_crash` | A 0-layer model must not panic or divide-by-zero anywhere in cost pipeline. |
//! | `zero_hidden_size_does_not_crash` | hidden_size=0 must not panic in weight_bytes or activation_bytes. |
//! | `quant_unknown_in_cost_panics_not_silent` | Unknown quant in cost::weight_bytes must panic loudly (pre-validated by plan()), not silently return wrong bytes. |
//! | `refused_record_has_refusal_reason` | A refused AdmitRecord must carry a non-empty refusal_reason string. |
//! | `refused_record_no_applied_degradation` | A refused AdmitRecord must have applied_degradation == None. |
//! | `fits_with_degradation_applied_degradation_is_some` | A Degraded AdmitRecord must have applied_degradation == Some(_). |
//! | `gguf_truncated_header_returns_error` | Truncated GGUF (magic + version only) must return Err, not Ok with garbage. |
//! | `gguf_wrong_magic_returns_error` | Wrong magic bytes must return Err immediately. |
//! | `gguf_version_zero_returns_error` | GGUF version 0 is undefined; must return Err. |
//! | `gguf_empty_file_returns_error` | Empty file must return Err, not panic. |
//! | `zero_bandwidth_decode_tok_s_not_nan` | MachineProfile with 0 bandwidth must not produce NaN in decode_tok_s. |
//! | `plan_verdict_monotone_in_budget` | Increasing budget must never produce a strictly worse verdict. |

use fitsproof::admit::{admit, AdmitStatus};
use fitsproof::cost;
use fitsproof::model::ModelConfig;
use fitsproof::plan::{plan, PlanError, Verdict};
use fitsproof::probe::MachineProfile;

fn small_model() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        hidden_size: 64,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 32,
        intermediate_size: 256,
        vocab_size: 512,
        max_seq_len: 128,
        name: "adversarial-ref".into(),
    }
}

fn synthetic_machine() -> MachineProfile {
    MachineProfile {
        hostname: "adversarial-test".into(),
        platform_str: "adversarial-test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 4,
    }
}

// ── plan() error-path faults ─────────────────────────────────────────────────

/// Fault: budget=0 silently produces DoesNotFit instead of a typed error.
/// A naive implementation that returns DoesNotFit would let callers treat 0 as a valid budget,
/// hiding configuration errors.
#[test]
fn budget_zero_returns_invalid_budget_error() {
    let model = small_model();
    let machine = synthetic_machine();
    let result = plan(&model, &machine, 128, 0, "fp32", 1.0);
    assert!(
        matches!(result, Err(PlanError::InvalidBudget(_))),
        "budget=0 must return PlanError::InvalidBudget; got {:?}",
        result
    );
}

/// Fault: unknown quant string falls through to fp32 silently, producing an overly optimistic plan.
#[test]
fn unknown_quant_returns_unknown_quant_error() {
    let model = small_model();
    let machine = synthetic_machine();
    let result = plan(&model, &machine, 128, 1_000_000_000, "not_a_quant", 1.0);
    assert!(
        matches!(result, Err(PlanError::UnknownQuant(_))),
        "unknown quant must return PlanError::UnknownQuant; got {:?}",
        result
    );
}

/// Fault: context_len=0 silently plans a zero-context run instead of returning an error.
#[test]
fn context_len_zero_returns_invalid_context_error() {
    let model = small_model();
    let machine = synthetic_machine();
    let result = plan(&model, &machine, 0, 1_000_000_000, "fp32", 1.0);
    assert!(
        matches!(result, Err(PlanError::InvalidContextLen(_))),
        "context_len=0 must return PlanError::InvalidContextLen; got {:?}",
        result
    );
}

// ── Budget boundary faults ───────────────────────────────────────────────────

/// Fault: rounding error (> vs >=) causes DoesNotFit when budget exactly equals predicted peak.
#[test]
fn budget_exactly_at_predicted_peak_admits() {
    let model = small_model();
    let machine = synthetic_machine();
    let peak = cost::estimate(&model, &machine, 128, "fp32", 1.0).total_peak_bytes;
    let result = plan(&model, &machine, 128, peak, "fp32", 1.0);
    let p = result.expect("plan must succeed with valid inputs");
    assert_ne!(
        p.verdict,
        Verdict::DoesNotFit,
        "plan with budget == predicted peak must not return DoesNotFit; got {:?}",
        p.verdict
    );
}

/// Fault: naïve >= instead of > allows a budget of (peak - 1) to return Verdict::Fits.
#[test]
fn budget_one_byte_below_peak_not_fits() {
    let model = small_model();
    let machine = synthetic_machine();
    let peak = cost::estimate(&model, &machine, 128, "fp32", 1.0).total_peak_bytes;
    if peak == 0 {
        return; // degenerate model — skip
    }
    let result = plan(&model, &machine, 128, peak - 1, "fp32", 1.0);
    if let Ok(p) = result {
        assert_ne!(
            p.verdict,
            Verdict::Fits,
            "plan with budget < peak must not return Verdict::Fits"
        );
    }
}

// ── Integer overflow / numerical faults ──────────────────────────────────────

/// Fault: integer overflow in context_len arithmetic wraps kv_cache_bytes to a small value.
#[test]
fn integer_overflow_context_len() {
    let model = small_model();
    // u32::MAX context — must produce a large, positive byte count, never wrap to 0
    let bytes = cost::kv_cache_bytes(&model, u32::MAX as usize, "fp32");
    assert!(
        bytes > 1_000_000,
        "kv_cache_bytes with u32::MAX context must exceed 1 MB; got {}",
        bytes
    );
}

/// Fault: large vocab_size causes weight_bytes to silently overflow to 0 or a tiny value.
#[test]
fn large_vocab_weight_bytes_not_zero() {
    let mut model = small_model();
    model.vocab_size = 1_000_000;
    let bytes = cost::weight_bytes(&model, "fp32");
    // Embedding table alone: 1_000_000 × 64 × 4 bytes = 256 MB
    assert!(
        bytes >= 256_000_000,
        "weight_bytes with 1M vocab and hidden=64 must be ≥256 MB; got {}",
        bytes
    );
}

/// Fault: 0-layer model causes divide-by-zero or panic in cost pipeline.
#[test]
fn zero_layers_does_not_crash() {
    let mut model = small_model();
    model.num_layers = 0;
    let machine = synthetic_machine();
    // Must reach here without panicking
    let _ = cost::weight_bytes(&model, "fp32");
    let _ = cost::kv_cache_bytes(&model, 128, "fp32");
    let _ = cost::estimate(&model, &machine, 128, "fp32", 1.0);
    let _ = plan(&model, &machine, 128, 1_000_000, "fp32", 1.0);
}

/// Fault: hidden_size=0 causes divide-by-zero in head_dim derivation or activation formula.
#[test]
fn zero_hidden_size_does_not_crash() {
    let mut model = small_model();
    model.hidden_size = 0;
    // Must not panic — reaching here = no divide-by-zero
    let _ = cost::weight_bytes(&model, "fp32");
    let _ = cost::activation_bytes(&model);
}

/// Fault: unknown quant in cost::weight_bytes produces 0 (empty model) or panics.
/// The actual contract: cost::weight_bytes asserts that quant is known (validated upstream by plan()).
/// This test verifies that the assertion fires rather than silently continuing with wrong bytes.
/// Correct behavior: panic with "weight_bytes: unknown quant — caller must validate".
/// Wrong behavior: silently return fp32 or 0 bytes.
#[test]
fn quant_unknown_in_cost_panics_not_silent() {
    use std::panic;
    let model = small_model();
    // cost::weight_bytes requires pre-validated quant; an unknown quant must panic loudly,
    // not silently return fp32/0 bytes (which would make the contract unenforceable).
    let result = panic::catch_unwind(|| cost::weight_bytes(&model, "completely_unknown_format"));
    assert!(
        result.is_err(),
        "cost::weight_bytes with unknown quant must panic (loud failure), not silently return bytes"
    );
}

/// Fault: zero memory bandwidth causes divide-by-zero in decode_tok_s, producing NaN.
#[test]
fn zero_bandwidth_decode_tok_s_not_nan() {
    let mut machine = synthetic_machine();
    machine.memory_bandwidth_bps = 0.0;
    let model = small_model();
    let toks = cost::decode_tok_s(&model, &machine, "fp32", 1.0);
    assert!(
        !toks.is_nan(),
        "decode_tok_s with 0 bandwidth must not return NaN"
    );
}

// ── AdmitRecord contract faults ──────────────────────────────────────────────

/// Fault: Refused AdmitRecord has empty refusal_reason — caller cannot explain why the run was blocked.
#[test]
fn refused_record_has_refusal_reason() {
    let model = small_model();
    let machine = synthetic_machine();
    // A tiny budget that can never fit anything — will produce DoesNotFit
    let p = plan(&model, &machine, 128, 1, "fp32", 1.0);
    match p {
        Err(_) => {
            // PlanError::InvalidBudget is also fine — the contract rejects 0 early
        }
        Ok(p) => {
            if p.verdict == Verdict::DoesNotFit {
                let record = admit(p);
                assert_eq!(record.status, AdmitStatus::Refused);
                assert!(
                    !record.refusal_reason.is_empty(),
                    "refused AdmitRecord must carry a non-empty refusal_reason"
                );
            }
        }
    }
}

/// Fault: Refused AdmitRecord has Some(applied_degradation) — implies a degradation was silently applied.
#[test]
fn refused_record_no_applied_degradation() {
    let model = small_model();
    let machine = synthetic_machine();
    // Budget of 1 byte — below every degradation option
    let p = plan(&model, &machine, 128, 1, "fp32", 1.0);
    match p {
        Err(_) => {}
        Ok(p) => {
            if p.verdict == Verdict::DoesNotFit {
                let record = admit(p);
                assert_eq!(record.status, AdmitStatus::Refused);
                assert!(
                    record.applied_degradation.is_none(),
                    "refused AdmitRecord must not claim a degradation was applied"
                );
            }
        }
    }
}

/// Fault: Degraded AdmitRecord has None applied_degradation — a silent mode change.
#[test]
fn fits_with_degradation_applied_degradation_is_some() {
    let model = small_model();
    let machine = synthetic_machine();
    // Budget: above int4 peak but below fp32 peak
    let fp32_peak = cost::estimate(&model, &machine, 128, "fp32", 1.0).total_peak_bytes;
    let int4_peak = cost::estimate(&model, &machine, 128, "int4", 1.0).total_peak_bytes;
    if fp32_peak <= int4_peak {
        return; // model too small to create a gap — skip
    }
    let budget = int4_peak + (fp32_peak - int4_peak) / 2;
    let p = plan(&model, &machine, 128, budget, "fp32", 1.0).unwrap();
    if p.verdict != Verdict::FitsWithDegradation {
        return; // budget window did not produce a degradation path — skip
    }
    let record = admit(p);
    if record.status == AdmitStatus::Degraded {
        assert!(
            record.applied_degradation.is_some(),
            "Degraded AdmitRecord must carry Some(applied_degradation); silent mode change detected"
        );
    }
}

// ── GGUF malformed input ─────────────────────────────────────────────────────

/// Fault: truncated GGUF (magic + version, then EOF) returns Ok with garbage ModelConfig.
#[test]
fn gguf_truncated_header_returns_error() {
    use fitsproof::gguf::read_metadata;
    use std::io::Cursor;
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    // Deliberately omit tensor_count and kv_count fields
    let result = read_metadata(Cursor::new(data));
    assert!(
        result.is_err(),
        "truncated GGUF must return Err, not Ok with garbage"
    );
}

/// Fault: wrong magic bytes accepted silently, leading to misparse downstream.
#[test]
fn gguf_wrong_magic_returns_error() {
    use fitsproof::gguf::read_metadata;
    use std::io::Cursor;
    let mut data = vec![0u8; 32];
    data[0..4].copy_from_slice(b"WXYZ"); // wrong magic
    data[4..8].copy_from_slice(&3u32.to_le_bytes());
    let result = read_metadata(Cursor::new(data));
    assert!(result.is_err(), "wrong GGUF magic must return Err");
}

/// Fault: GGUF version 0 accepted as a valid format version.
#[test]
fn gguf_version_zero_returns_error() {
    use fitsproof::gguf::read_metadata;
    use std::io::Cursor;
    let mut data = vec![0u8; 32];
    data[0..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&0u32.to_le_bytes()); // version = 0
    let result = read_metadata(Cursor::new(data));
    assert!(result.is_err(), "GGUF version 0 must return Err");
}

/// Fault: empty file causes panic instead of returning Err.
#[test]
fn gguf_empty_file_returns_error() {
    use fitsproof::gguf::read_metadata;
    use std::io::Cursor;
    let result = read_metadata(Cursor::new(Vec::<u8>::new()));
    assert!(result.is_err(), "empty file must return Err, not panic");
}

// ── Verdict monotonicity ─────────────────────────────────────────────────────

/// Fault: plan() with a larger budget returns a strictly worse verdict than a smaller budget.
/// The ordering is: DoesNotFit < FitsWithDegradation < Fits.
/// A larger budget must never produce a worse verdict than a smaller one.
#[test]
fn plan_verdict_monotone_in_budget() {
    let model = small_model();
    let machine = synthetic_machine();

    fn verdict_rank(v: Verdict) -> u8 {
        match v {
            Verdict::DoesNotFit => 0,
            Verdict::FitsWithDegradation => 1,
            Verdict::Fits => 2,
        }
    }

    let budgets: &[u64] = &[1, 1_000_000, 100_000_000, 10_000_000_000];
    let mut prev_rank = 0u8;
    for &budget in budgets {
        let p = match plan(&model, &machine, 128, budget, "fp32", 1.0) {
            Ok(p) => p,
            Err(_) => continue, // skip invalid budget values
        };
        let rank = verdict_rank(p.verdict);
        assert!(
            rank >= prev_rank,
            "plan verdict must be monotone: at budget={budget}, got {:?} (rank {rank}) after rank {prev_rank}",
            p.verdict
        );
        prev_rank = rank;
    }
}
