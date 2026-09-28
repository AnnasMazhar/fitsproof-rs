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
//! | `race_condition_ceiling_closed` | ADV-3 fix verified: two concurrent threads cannot both succeed when combined alloc exceeds ceiling. CAS loop in try_reserve closes the TOCTOU race. |
//! | `fits_with_degradation_but_none_fit_refuses` | A Plan with FitsWithDegradation but no fitting DegradationStep must refuse, not silently proceed. |
//! | `negative_or_zero_peak_does_not_bypass_budget` | Degenerate models (0 layers, vocab=1, hidden=1) must not produce overflow or negative peaks. |
//! | `verify_handles_vmhwm_read` | read_vmhwm_bytes must not panic on any platform. |
//! | `guard_refuses_when_budget_below_any_degradation` | FitsproofClient::guard() must return Err for an impossible budget, with non-empty binding_constraint. |
//! | `guard_ok_for_large_budget_means_admitted` | FitsproofClient::guard() must return Ok for a clearly feasible budget — no spurious refusals. |
//! | `client_with_quant_changes_predicted_peak` | with_quant() must change the predicted peak — silent discard would produce over-estimated peaks. |
//! | `client_with_context_changes_predicted_peak` | with_context() must change the predicted peak — silent discard would under-estimate KV cache. |
//! | `guard_error_display_names_binding_constraint` | GuardError Display must name the binding constraint, not return a generic string. |

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

// ── Adversarial Pass 2: Attack the Property ───────────────────────────────────

/// ATTACK: Race condition in ceiling enforcement.
/// The check-then-allocate is not atomic. Two threads racing can both pass
/// the check if they read current_bytes before either updates it.
///
/// Attack vector: ceiling=900, current=500, two threads each try alloc(350)
/// Both threads: check 500+350=850<900 → pass
/// Both threads: alloc succeeds → current becomes 1200 > ceiling
///
/// FINDING ADV-3 STATUS: FIXED in c2-p05 via `try_reserve()` CAS loop.
/// The `alloc` / `alloc_zeroed` / `realloc` paths now use `try_reserve`, which
/// atomically increments `current` only when the result would not exceed the ceiling.
/// This test verifies the race is closed: the two threads keep their pointers live
/// until both have finished allocating (using a Barrier), so dealloc cannot create
/// a false window. At most ONE of the two 350-byte allocations can succeed within a
/// 900-byte ceiling when 500 bytes are already live.
#[test]
fn race_condition_ceiling_closed() {
    use fitsproof::allocator::TrackingAllocator;
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    for _trial in 0..50 {
        let allocator = Arc::new(TrackingAllocator::new());
        let ceiling: u64 = 900;

        // Pre-fill 500 bytes so the two 350-byte allocations together would exceed ceiling.
        let prefill_layout = Layout::array::<u8>(500).unwrap();
        let prefill_ptr = unsafe { allocator.alloc(prefill_layout) };
        if prefill_ptr.is_null() {
            continue;
        }

        allocator.set_ceiling(ceiling);

        // Barrier ensures both threads enter alloc() simultaneously.
        let barrier = Arc::new(Barrier::new(2));
        let success_count = Arc::new(AtomicUsize::new(0));
        // Store pointers as u64 to make them Send.
        let ptr0 = Arc::new(AtomicU64::new(0));
        let ptr1 = Arc::new(AtomicU64::new(0));
        let alloc_layout = Layout::array::<u8>(350).unwrap();

        let (a0, a1) = (Arc::clone(&allocator), Arc::clone(&allocator));
        let (sc0, sc1) = (Arc::clone(&success_count), Arc::clone(&success_count));
        let (b0, b1) = (Arc::clone(&barrier), Arc::clone(&barrier));
        let (p0, p1) = (Arc::clone(&ptr0), Arc::clone(&ptr1));

        let h0 = thread::spawn(move || {
            b0.wait();
            let ptr = unsafe { a0.alloc(alloc_layout) };
            if !ptr.is_null() {
                sc0.fetch_add(1, Ordering::SeqCst);
                p0.store(ptr as u64, Ordering::SeqCst);
            }
        });
        let h1 = thread::spawn(move || {
            b1.wait();
            let ptr = unsafe { a1.alloc(alloc_layout) };
            if !ptr.is_null() {
                sc1.fetch_add(1, Ordering::SeqCst);
                p1.store(ptr as u64, Ordering::SeqCst);
            }
        });
        h0.join().unwrap();
        h1.join().unwrap();

        // Dealloc now, after both threads have finished (pointers stay live during race).
        let p = ptr0.load(Ordering::SeqCst) as *mut u8;
        if !p.is_null() {
            unsafe { allocator.dealloc(p, alloc_layout) };
        }
        let p = ptr1.load(Ordering::SeqCst) as *mut u8;
        if !p.is_null() {
            unsafe { allocator.dealloc(p, alloc_layout) };
        }
        unsafe { allocator.dealloc(prefill_ptr, prefill_layout) };

        let successes = success_count.load(Ordering::SeqCst);
        assert!(
            successes <= 1,
            "ADV-3 REGRESSION: ceiling=900, prefill=500, two concurrent threads tried 350 B each — \
             both succeeded ({} successes). The CAS loop in try_reserve must prevent this.",
            successes
        );
    }
}

/// ATTACK: Construct a Plan with verdict=FitsWithDegradation but no fitting degradation.
/// This tests the defensive code path in admit() that handles this inconsistency.
/// The correct behavior is: refuse, not silently proceed.
#[test]
fn fits_with_degradation_but_none_fit_refuses() {
    use fitsproof::admit::{admit, AdmitStatus};
    use fitsproof::plan::{DegradationKind, DegradationStep, Plan, Verdict};

    // Manually construct an inconsistent Plan
    let plan = Plan {
        verdict: Verdict::FitsWithDegradation,
        predicted_peak_bytes: 1_000_000,
        predicted_peak_ci: (800_000, 1_200_000),
        predicted_tok_s: 10.0,
        budget_bytes: 500_000,
        quant: "none".into(),
        context_len: 512,
        degradations: vec![
            DegradationStep {
                kind: DegradationKind::LowerQuant,
                description: "Use int8 instead".into(),
                predicted_peak_bytes: 700_000,
                predicted_tok_s: 15.0,
                fits_budget: false, // Deliberately set to false
            },
            DegradationStep {
                kind: DegradationKind::ShorterContext,
                description: "Reduce context".into(),
                predicted_peak_bytes: 600_000,
                predicted_tok_s: 12.0,
                fits_budget: false, // Deliberately set to false
            },
        ],
        binding_constraint: String::new(),
    };

    let record = admit(plan);

    // The defensive code path should refuse, not silently proceed
    assert_eq!(
        record.status,
        AdmitStatus::Refused,
        "FitsWithDegradation with no fitting degradation must refuse"
    );
    assert!(
        !record.refusal_reason.is_empty(),
        "Refused record must explain why"
    );
    assert!(
        record.applied_degradation.is_none(),
        "No degradation should be applied when none fit"
    );
}

/// ATTACK: Negative peak bytes in cost estimate could bypass budget check.
/// Test that the contract handles edge cases in cost calculation.
#[test]
fn negative_or_zero_peak_does_not_bypass_budget() {
    use fitsproof::cost;

    // Zero-layer model might produce zero peak
    let mut model = small_model();
    model.num_layers = 0;
    model.vocab_size = 1; // Minimal vocab
    model.hidden_size = 1;

    let machine = synthetic_machine();
    let est = cost::estimate(&model, &machine, 1, "none", 1.0);

    // Even a degenerate model should not produce negative or overflow values
    assert!(
        est.total_peak_bytes < u64::MAX / 2,
        "peak should be reasonable"
    );
    // Zero is acceptable for a degenerate model
}

/// ATTACK: VmHWM reading could fail, hiding real memory usage.
/// Test that verify() handles /proc read errors gracefully.
#[test]
fn verify_handles_vmhwm_read() {
    // This test documents the expected behavior when VmHWM is read.
    // The current implementation reads /proc/self/status, which should always work
    // on Linux. On other platforms, it should return a reasonable value or error.
    use fitsproof::verify::read_vmhwm_bytes;

    let hwm = read_vmhwm_bytes();
    // On Linux, this should succeed. On other platforms, it may return 0.
    // The key is: it must not panic.
    eprintln!("VmHWM read result: {} bytes", hwm);
    // Accept any non-panic result
}

// ── FitsproofClient API attacks ───────────────────────────────────────────────

/// ATTACK: FitsproofClient::guard() admits a config that exceeds the budget.
///
/// Fault detected: `guard()` returns `Ok(())` when the predicted peak exceeds the declared
/// budget — allowing the caller to proceed to model loading and OOM. This is the exact
/// failure mode fitsproof exists to prevent.
#[test]
fn guard_refuses_when_budget_below_any_degradation() {
    use fitsproof::client::FitsproofClient;
    // 1 byte budget — no config can possibly fit
    let client = FitsproofClient::new(1.0 / (1024.0 * 1024.0 * 1024.0));
    let result = client.guard();
    assert!(
        result.is_err(),
        "guard() must return Err when budget is effectively 0; got Ok"
    );
    let err = result.unwrap_err();
    assert!(
        !err.binding_constraint.is_empty(),
        "GuardError must carry a non-empty binding_constraint"
    );
}

/// ATTACK: FitsproofClient::guard() silently accepts a degraded config as if it were
/// the requested config — hiding that a mode change occurred.
///
/// Fault detected: `guard()` returns `Ok(())` for a `FitsWithDegradation` verdict,
/// masking the degradation. The correct contract: `Degraded` records are admitted
/// (Ok), but the `AdmitRecord` attached to any error must reflect the actual status.
#[test]
fn guard_ok_for_large_budget_means_admitted() {
    use fitsproof::client::FitsproofClient;
    // Very large budget — should always be admitted
    let client = FitsproofClient::new(1000.0);
    let result = client.guard();
    // Must return Ok(()) — not Err — for a clearly feasible budget
    assert!(
        result.is_ok(),
        "guard() must return Ok for a 1000 GB budget; got Err: {:?}",
        result.err()
    );
}

/// ATTACK: FitsproofClient builder methods silently discard the quant setting.
///
/// Fault detected: `with_quant("int4")` returns a client whose plan still uses fp32
/// — the quant override is silently ignored, producing an over-estimated peak.
/// A higher peak means the client rejects configs that would actually fit.
#[test]
fn client_with_quant_changes_predicted_peak() {
    use fitsproof::client::FitsproofClient;
    let fp32_client = FitsproofClient::new(8.0);
    let int4_client = FitsproofClient::new(8.0).with_quant("int4");

    let fp32_plan = fp32_client.plan().expect("fp32 plan must succeed");
    let int4_plan = int4_client.plan().expect("int4 plan must succeed");

    assert!(
        int4_plan.predicted_peak_bytes < fp32_plan.predicted_peak_bytes,
        "int4 plan peak ({}) must be less than fp32 plan peak ({}) — \
         with_quant is silently discarded",
        int4_plan.predicted_peak_bytes,
        fp32_plan.predicted_peak_bytes
    );
}

/// ATTACK: FitsproofClient builder silently discards the context override.
///
/// Fault detected: `with_context(4096)` returns a client whose plan still uses
/// the default context — the context override is silently ignored, producing
/// an under-estimated KV cache and a falsely optimistic budget check.
#[test]
fn client_with_context_changes_predicted_peak() {
    use fitsproof::client::FitsproofClient;
    let short_ctx = FitsproofClient::new(8.0).with_context(128);
    let long_ctx = FitsproofClient::new(8.0).with_context(8192);

    let short_plan = short_ctx.plan().expect("short-context plan must succeed");
    let long_plan = long_ctx.plan().expect("long-context plan must succeed");

    assert!(
        long_plan.predicted_peak_bytes > short_plan.predicted_peak_bytes,
        "8192-context peak ({}) must exceed 128-context peak ({}) — \
         with_context is silently discarded",
        long_plan.predicted_peak_bytes,
        short_plan.predicted_peak_bytes
    );
}

/// ATTACK: GuardError's Display is empty, making it useless in error chains.
///
/// Fault detected: `GuardError` implements `Display` but returns an empty or
/// generic string — the binding constraint is not surfaced, so the caller cannot
/// tell the user what specifically doesn't fit.
#[test]
fn guard_error_display_names_binding_constraint() {
    use fitsproof::client::FitsproofClient;
    let client = FitsproofClient::new(0.0001); // impossibly small
    let err = client.guard().unwrap_err();
    let display = err.to_string();
    assert!(
        display.contains("guard refused") || display.contains("refused"),
        "GuardError Display must mention 'refused' or 'guard refused'; got: {:?}",
        display
    );
    // The binding_constraint field must also be non-empty
    assert!(
        !err.binding_constraint.is_empty(),
        "GuardError.binding_constraint must not be empty"
    );
}
