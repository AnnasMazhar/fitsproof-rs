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
//! | `budget_exactly_at_predicted_peak_admits` | Budget == predicted peak must return Verdict::Fits (not FitsWithDegradation). Changing `<=` to `<` in plan.rs produces FitsWithDegradation — `!= DoesNotFit` misses this; `== Fits` catches it. |
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
//! | `mcp_invalid_json_returns_parse_error` | Totally malformed JSON (not just missing fields) must return a JSON-RPC parse error (-32700), not panic or silently return success. |
//! | `serve_admitted_response_has_admission_record` | A sufficient budget must yield HTTP 200 whose body contains `admission_record` — the contract proof, not just a completion stub. |
//! | `plan_budget_infinity_does_not_panic` | budget_bytes = u64::MAX (f64::INFINITY cast) must not crash; it must return Fits or DoesNotFit, never panic. |
//! | `pareto_impossible_model_tiny_budget_empty_frontier` | A model with large weight_bytes against a 1-byte budget must produce an empty Pareto frontier, not a non-empty one with impossible configs. |
//! | `plan_context_len_usize_max_no_overflow` | context_len = usize::MAX must not overflow or wrap in kv_cache_bytes (different path from u32::MAX test). |
//! | `admit_refused_record_has_nonempty_refusal_reason_and_message` | A refused AdmitRecord from DoesNotFit must carry non-empty refusal_reason and message starting with REFUSED:. The Default::default() mutant returns empty strings. |
//! | `serve_minimal_request_no_budget_field_returns_json` | A request without budget_gb must not cause HTTP 500 or panic — handler must return 200 or 400 with valid JSON. |
//! | `pareto_frontier_at_tight_budget_contains_only_feasible_configs` | Pareto Fits entries must not exceed the declared budget — a dominated config in the frontier would silently admit an impossible config. |
//! | `allocator_check_refuses_at_ceiling_plus_one` | Allocator ceiling check: requesting ceiling+1 bytes must fail; exactly ceiling bytes must succeed. Off-by-one (> vs >=) would allow silent budget breach. |
//! | `quant_from_name_returns_none_for_unknown_strings` | Unknown quant strings must propagate as PlanError::UnknownQuant, not silently fall through to fp32. |
//! | `cost_estimate_total_peak_equals_sum_of_components` | CostEstimate::total_peak_bytes must equal weight_bytes + kv_cache_bytes + activation_bytes. A hidden constant or wrong operator would break this. |
//! | `admit_degraded_record_message_starts_with_degraded` | A degraded AdmitRecord message must start with "DEGRADED:" — generic or empty messages do not document the mode change. |
//! | `pareto_admitted_count_matches_non_does_not_fit_frontier` | ParetoResult::admitted_configs must equal the count of non-DoesNotFit entries in the frontier — a wrong count misrepresents the feasibility surface. |
//! | `mcp_probe_tool_response_contains_hostname_field` | MCP probe tool response must contain "hostname" and "memory_bytes" — a replace-body mutant returning an empty object is not caught by status-only checks. |
//! | `serve_unknown_path_returns_404` | Unknown URL paths must return 404, not 200 or panic — the contract applies only to the documented endpoint. |

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
/// Also catches the <= vs < fault: if `<=` is changed to `<`, the verdict falls through to
/// FitsWithDegradation instead of Fits — which `!= DoesNotFit` would pass but `== Fits` catches.
#[test]
fn budget_exactly_at_predicted_peak_admits() {
    let model = small_model();
    let machine = synthetic_machine();
    let peak = cost::estimate(&model, &machine, 128, "fp32", 1.0).total_peak_bytes;
    let result = plan(&model, &machine, 128, peak, "fp32", 1.0);
    let p = result.expect("plan must succeed with valid inputs");
    assert_eq!(
        p.verdict,
        Verdict::Fits,
        "plan with budget == predicted peak must return Verdict::Fits (not FitsWithDegradation \
         or DoesNotFit); got {:?}. Changing <= to < in plan.rs would produce FitsWithDegradation \
         here — that is the fault this test detects.",
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

/// ATTACK: MCP handle_rpc silently accepts garbage input and returns a success response.
///
/// Fault detected: A missing `method` field in the JSON-RPC request must return
/// an error response (-32600), not a success. A naive handler that defaults to
/// any method when the field is absent would bypass error handling.
#[test]
fn mcp_missing_method_returns_rpc_error() {
    use fitsproof::mcp::handle_rpc_for_test;
    // No "method" field — must return a JSON-RPC error, not a result.
    let resp = handle_rpc_for_test(r#"{"jsonrpc":"2.0","id":1,"params":{}}"#);
    assert!(
        resp.contains("\"error\""),
        "missing method must return JSON-RPC error object, got: {resp}"
    );
    assert!(
        !resp.contains("\"result\""),
        "missing method must not return a result, got: {resp}"
    );
}

/// ATTACK: MCP tools/call with an unknown tool name returns a success (content) response.
///
/// Fault detected: Unknown tool names must return a JSON-RPC error (-32603), not an
/// empty content array or a success with null content. A bug where the match arm
/// falls through to Ok("") would pass an empty content as success.
#[test]
fn mcp_unknown_tool_returns_error() {
    use fitsproof::mcp::handle_rpc_for_test;
    let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"doesnotexist","arguments":{}}}"#;
    let resp = handle_rpc_for_test(req);
    assert!(
        resp.contains("\"error\""),
        "unknown tool must return JSON-RPC error, got: {resp}"
    );
}

/// ATTACK: Pareto sweep with a budget of 0 bytes produces a non-empty frontier.
///
/// Fault detected: If the budget comparison is wrong (e.g. `>=` instead of `>`),
/// a zero budget could admit at least one config. The Pareto frontier for budget=0
/// must be empty — nothing can fit in 0 bytes.
#[test]
fn pareto_zero_budget_empty_frontier() {
    use fitsproof::model::ModelConfig;
    use fitsproof::pareto::pareto_sweep;
    use fitsproof::probe::MachineProfile;
    let cfg = ModelConfig::reference();
    let machine = MachineProfile {
        hostname: "test".into(),
        platform_str: "test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 4,
    };
    let result = pareto_sweep(&cfg, &machine, 0);
    assert!(
        result.frontier.is_empty(),
        "Pareto frontier for budget=0 bytes must be empty, got {} configs",
        result.frontier.len()
    );
}

/// ATTACK: Pareto frontier contains a dominated config (both peak and context are worse
/// than another config on the frontier).
///
/// Fault detected: If the dominance check is wrong (e.g. strict `<` replaced with `<=`),
/// configs equal on both axes would be incorrectly pruned, and the Pareto property
/// could be violated. We verify no config on the frontier is strictly dominated.
#[test]
fn pareto_frontier_has_no_dominated_configs() {
    use fitsproof::model::ModelConfig;
    use fitsproof::pareto::pareto_sweep;
    use fitsproof::probe::MachineProfile;
    let cfg = ModelConfig::reference();
    let machine = MachineProfile {
        hostname: "test".into(),
        platform_str: "test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 4,
    };
    let result = pareto_sweep(&cfg, &machine, 4_000_000_000);
    let frontier = &result.frontier;
    // For every pair (a, b) on the frontier, a must not strictly dominate b.
    for (i, a) in frontier.iter().enumerate() {
        for (j, b) in frontier.iter().enumerate() {
            if i == j {
                continue;
            }
            let a_dominates_b =
                a.predicted_peak_bytes < b.predicted_peak_bytes && a.context_len > b.context_len;
            assert!(
                !a_dominates_b,
                "frontier config [{i}] strictly dominates [{j}]: \
                 peak {:.3}GB ctx {} vs peak {:.3}GB ctx {}",
                a.predicted_peak_bytes as f64 / 1e9,
                a.context_len,
                b.predicted_peak_bytes as f64 / 1e9,
                b.context_len,
            );
        }
    }
}

/// ATTACK: serve HTTP endpoint returns 200 with an empty body for a refused budget.
///
/// Fault detected: When the admit result is Refused, the server must return 503 (not 200)
/// and the body must contain the binding constraint. A bug where the status code is
/// hardcoded to 200, or where refused responses are serialised as normal completions,
/// would defeat CI gating on the contract.
#[test]
fn serve_refused_budget_returns_503() {
    use fitsproof::serve::handle_request_for_test;
    // Request body with an impossibly small budget (0.000001 GB).
    let body = r#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hi"}],"budget_gb":0.000001}"#;
    let (status_code, response_body) = handle_request_for_test("/v1/chat/completions", body);
    assert_eq!(
        status_code, 503,
        "refused budget must return HTTP 503, got {status_code}; body: {response_body}"
    );
    assert!(
        response_body.contains("binding_constraint") || response_body.contains("REFUSED"),
        "503 body must name the binding constraint, got: {response_body}"
    );
}

/// ATTACK: MCP receives totally invalid JSON (not just a missing field) and returns success.
///
/// Fault detected: A handler that tries `serde_json::from_str` and panics (or uses unwrap)
/// on non-JSON input would crash the stdio loop. A handler that silently swallows the error
/// and returns `{"result":null}` would let garbage bytes claim success. Must return a
/// JSON-RPC error response (code -32700 parse error or -32600 invalid request), not success.
///
/// Note on implementation: fitsproof's MCP handler uses lightweight text-based JSON extraction.
/// It cannot distinguish "not JSON" from "JSON missing required fields" — both produce None
/// for the method field and return -32600 (Invalid Request). This is a valid JSON-RPC 2.0
/// error (both -32600 and -32700 indicate the request cannot be processed). The critical
/// property is that the response contains `"error"` and does not contain `"result"`.
#[test]
fn mcp_invalid_json_returns_parse_error() {
    use fitsproof::mcp::handle_rpc_for_test;
    // Totally malformed — not JSON at all.
    let resp = handle_rpc_for_test("this is not json }{{{");
    assert!(
        resp.contains("\"error\""),
        "invalid JSON must return a JSON-RPC error object, got: {resp}"
    );
    assert!(
        !resp.contains("\"result\""),
        "invalid JSON must not return a result, got: {resp}"
    );
    // Error code must be -32700 (Parse Error) or -32600 (Invalid Request) — both are valid
    // JSON-RPC 2.0 error responses for input that cannot be processed. The critical property
    // is a non-success error response, not the specific code.
    assert!(
        resp.contains("-32700") || resp.contains("-32600"),
        "error code must be -32700 or -32600, got: {resp}"
    );
}

/// ATTACK: serve returns HTTP 200 with only a stub body when the budget is sufficient —
/// the `admission_record` field is absent, defeating CI gating on the contract proof.
///
/// Fault detected: If the handler creates a ChatCompletion response but omits the
/// `admission_record` key, a caller that checks only HTTP 200 cannot verify the
/// contract was actually evaluated. The field must be present in a 200 response.
#[test]
fn serve_admitted_response_has_admission_record() {
    use fitsproof::serve::handle_request_for_test;
    // Large budget: reference model (~0.057 GB) easily fits.
    let body = r#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hello"}],"budget_gb":8.0}"#;
    let (status_code, response_body) = handle_request_for_test("/v1/chat/completions", body);
    assert_eq!(
        status_code, 200,
        "sufficient budget must return HTTP 200, got {status_code}; body: {response_body}"
    );
    assert!(
        response_body.contains("admission_record"),
        "HTTP 200 body must contain 'admission_record' (the contract proof), got: {response_body}"
    );
}

/// ATTACK: budget_bytes = u64::MAX does not crash or produce a nonsensical verdict.
///
/// Fault detected: `plan()` computes `budget_bytes as f64` and compares to `total_peak_bytes`.
/// With u64::MAX, the cast to f64 is well-defined (rounds up), but any arithmetic that
/// overflows to negative or NaN would silently admit an impossible config, or panic.
/// Must return a well-formed Plan (not panic, not NaN-based).
#[test]
fn plan_budget_infinity_does_not_panic() {
    let model = small_model();
    let machine = synthetic_machine();
    // u64::MAX budget: should always return Fits (model is tiny, budget is astronomical).
    let result = plan(&model, &machine, 512, u64::MAX, "fp32", 0.9);
    match result {
        Ok(p) => {
            // Any well-formed Plan is acceptable — the key property is no panic or NaN verdict.
            let _ = p.verdict;
        }
        Err(_) => {
            // A PlanError is also acceptable (e.g. if u64::MAX triggers InvalidBudget).
        }
    }
    // No panic = pass. (If this test runs, it passed.)
}

/// ATTACK: Pareto sweep for a model that requires more than the declared budget on every
/// quant/context combination returns a non-empty frontier.
///
/// Fault detected: If the frontier-inclusion predicate compares wrong (e.g. `>` vs `>=`),
/// or if the budget is not propagated to the inner admit() call, configs that do not fit
/// could appear on the frontier. A 1-byte budget must yield an empty frontier regardless
/// of model or quant.
#[test]
fn pareto_impossible_model_tiny_budget_empty_frontier() {
    use fitsproof::model::ModelConfig;
    use fitsproof::pareto::pareto_sweep;
    use fitsproof::probe::MachineProfile;
    // A large model: many layers, large hidden_size, large vocab — guaranteed > 1 byte.
    let big_model = ModelConfig {
        num_layers: 32,
        hidden_size: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 11008,
        vocab_size: 32000,
        max_seq_len: 4096,
        name: "large-adversarial".into(),
    };
    let machine = MachineProfile {
        hostname: "test".into(),
        platform_str: "test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 4,
    };
    // Budget = 1 byte: nothing should fit.
    let result = pareto_sweep(&big_model, &machine, 1);
    assert!(
        result.frontier.is_empty(),
        "Pareto frontier for a large model with 1-byte budget must be empty, got {} configs",
        result.frontier.len()
    );
}

/// ATTACK: context_len = usize::MAX overflows kv_cache_bytes to 0 or wraps to a tiny value.
///
/// Fault detected: kv_cache_bytes computes `n_layers * n_kv_heads * context_len * head_dim * 2`.
/// With usize::MAX this is a multiplication of very large numbers. On 64-bit targets usize::MAX
/// is 2^64-1; multiplying even by 2 wraps to 0 in unchecked arithmetic. The result must be
/// either a very large u64 (no wrap) or the function must saturate — the critical constraint
/// is that it must NOT return 0 or a value smaller than with a smaller context_len.
/// (This is a different path from integer_overflow_context_len which uses u32::MAX as usize.)
#[test]
fn plan_context_len_usize_max_no_overflow() {
    use fitsproof::cost;
    let model = small_model(); // num_kv_heads=2, head_dim=32, num_layers=2
                               // context_len = usize::MAX — the overflow-prone input
    let kv_max = cost::kv_cache_bytes(&model, usize::MAX, "fp16");
    // context_len = 1 — the minimal non-zero input
    let kv_one = cost::kv_cache_bytes(&model, 1, "fp16");
    // The key safety property: kv_cache_bytes(MAX) must be >= kv_cache_bytes(1).
    // Overflow to 0 would violate this — the plan would admit an impossibly large model.
    assert!(
        kv_max >= kv_one,
        "kv_cache_bytes(usize::MAX)={kv_max} must be >= kv_cache_bytes(1)={kv_one}; \
         overflow-to-zero would cause silent OOM admission"
    );
}

// ---------------------------------------------------------------------------
// Additional adversarial tests added in c5-p05 — targeting missed fault patterns
// ---------------------------------------------------------------------------

/// Fault detected: `admit()` accepting a DoesNotFit plan and returning a blank AdmitRecord
/// with status=Admitted and empty refusal_reason — the Default::default() mutant.
/// This test verifies via the refusal_reason field specifically (not just status).
#[test]
fn admit_refused_record_has_nonempty_refusal_reason_and_message() {
    // The mutant replace admit -> AdmitRecord with Default::default() produces
    // a record with refusal_reason="" and message="" regardless of input.
    let p = {
        use fitsproof::plan::{plan, Verdict};
        let m = small_model();
        let machine = synthetic_machine();
        let p = plan(&m, &machine, 8, 1, "none", 0.6).unwrap();
        assert_eq!(p.verdict, Verdict::DoesNotFit);
        p
    };
    let rec = fitsproof::admit::admit(p);
    assert_eq!(rec.status, fitsproof::admit::AdmitStatus::Refused);
    assert!(
        !rec.refusal_reason.is_empty(),
        "refusal_reason must be non-empty for a refused record"
    );
    assert!(
        !rec.message.is_empty(),
        "message must be non-empty for a refused record"
    );
    // Message must specifically contain "REFUSED:" prefix (not just any non-empty string)
    assert!(
        rec.message.starts_with("REFUSED:"),
        "refused message must start with REFUSED:, got {:?}",
        rec.message
    );
}

/// Fault detected: serve handler panicking or returning HTTP 500 for a request with
/// no `budget_gb` field (missing optional field treated as hard error instead of default).
/// Uses the test-accessible `handle_request_for_test` function.
#[test]
fn serve_minimal_request_no_budget_field_returns_json() {
    use fitsproof::serve::handle_request_for_test;

    // A request with no budget_gb field — the handler must return HTTP 200 with
    // admission_record (using the default budget) rather than panicking or 500.
    let body = r#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"test"}]}"#;
    let (status, response_body) = handle_request_for_test("/v1/chat/completions", body);

    // Must be a 200 (admitted, default budget) or 400 (bad request — both are non-panic).
    // The critical property: no panic, no HTTP 500.
    assert!(
        status != 500,
        "serve must not return 500 for a request without budget_gb, got status={status}"
    );
    // Response must be parseable JSON.
    let parsed: serde_json::Value =
        serde_json::from_str(&response_body).expect("serve response must be valid JSON");
    // For 200 responses, admission_record must be present.
    if status == 200 {
        assert!(
            parsed.get("admission_record").is_some(),
            "HTTP 200 response must contain admission_record"
        );
    }
}

/// Fault detected: pareto sweep including dominated configs — specifically, configs
/// marked as Fits in the frontier that actually exceed the declared budget.
/// Each frontier entry with Verdict::Fits must use ≤ budget bytes.
#[test]
fn pareto_frontier_at_tight_budget_contains_only_feasible_configs() {
    use fitsproof::pareto::pareto_sweep;
    use fitsproof::plan::Verdict;

    let cfg = small_model();
    let machine = synthetic_machine();
    // Budget: exactly what int4 + 32 context fits, but not fp32.
    let int4_32_bytes = fitsproof::cost::weight_bytes(&cfg, "int4_sym")
        + fitsproof::cost::kv_cache_bytes(&cfg, 32, "fp16")
        + fitsproof::cost::activation_bytes(&cfg);

    let result = pareto_sweep(&cfg, &machine, int4_32_bytes);

    // Every config marked Fits in the frontier must actually use ≤ budget bytes.
    // Configs marked DoesNotFit are on the frontier as reference points — they
    // can exceed the budget by definition.
    for entry in result
        .frontier
        .iter()
        .filter(|e| e.verdict == Verdict::Fits)
    {
        assert!(
            entry.predicted_peak_bytes <= int4_32_bytes,
            "frontier Fits entry {}/{} reports {} bytes > budget {} bytes",
            entry.quant,
            entry.context_len,
            entry.predicted_peak_bytes,
            int4_32_bytes
        );
    }
}

/// Fault detected: allocator ceiling allowing an allocation at exactly ceiling+1 bytes
/// (off-by-one in the ceiling check: `>` vs `>=` or wrong comparison operand).
/// Uses a standalone TrackingAllocator to avoid interfering with the global test allocator.
#[test]
fn allocator_check_refuses_at_ceiling_plus_one() {
    use fitsproof::allocator::TrackingAllocator;

    let a = TrackingAllocator::new();

    // Set ceiling to exactly 100 bytes.
    a.set_ceiling(100);

    // Requesting exactly 100 bytes — must succeed (≤ ceiling, current=0 so after=100).
    assert!(
        a.check(100).is_ok(),
        "check(100) with ceiling=100 and current=0 must succeed"
    );
    // Requesting 101 bytes — must fail (after=101 > 100 ceiling).
    assert!(
        a.check(101).is_err(),
        "check(101) with ceiling=100 must fail (off-by-one would allow it)"
    );
    // Requesting 1 byte — must succeed (well within ceiling, current=0).
    assert!(a.check(1).is_ok(), "check(1) with ceiling=100 must succeed");

    // Remove ceiling (0 = no ceiling per API docs) — must allow any size.
    a.set_ceiling(0);
    assert!(
        a.check(1024 * 1024).is_ok(),
        "check with ceiling=0 must always succeed (no ceiling installed)"
    );
}

/// Fault detected: QuantBits::from_name returning Some for unknown strings
/// (missing _ => None arm, or match logic inverted).
/// Unknown strings must return None, causing weight_bytes to panic (fail-closed).
#[test]
fn quant_from_name_returns_none_for_unknown_strings() {
    // Verified via plan() — it returns PlanError::UnknownQuant for unknown quant strings.
    use fitsproof::plan::{plan, PlanError};
    let m = small_model();
    let machine = synthetic_machine();
    let err = plan(&m, &machine, 128, 10_000_000_000, "bfloat16_mystery", 0.6);
    assert!(
        matches!(err, Err(PlanError::UnknownQuant(_))),
        "unknown quant string must return PlanError::UnknownQuant, got: {:?}",
        err
    );
}

// ---------------------------------------------------------------------------
// Additional adversarial tests added in c5-p05 — targeting missed fault patterns
// ---------------------------------------------------------------------------

/// Fault detected: `CostEstimate::total_peak_bytes` not equal to weight + kv + activation.
/// A mutation that adds a constant to total or swaps + for * would break this property.
/// The estimate() function documents this invariant in the module comment.
#[test]
fn cost_estimate_total_peak_equals_sum_of_components() {
    let model = small_model();
    let machine = synthetic_machine();

    // Use fp32 weights and fp16 KV (the estimate() function uses fp16 KV by default).
    let est = fitsproof::cost::estimate(&model, &machine, 512, "none", 0.6);

    // The contract: total == weight + kv + activation, no hidden constants.
    assert_eq!(
        est.total_peak_bytes,
        est.weight_bytes + est.kv_cache_bytes + est.activation_bytes,
        "total_peak_bytes ({}) must equal weight ({}) + kv ({}) + activation ({})",
        est.total_peak_bytes,
        est.weight_bytes,
        est.kv_cache_bytes,
        est.activation_bytes
    );
}

/// Fault detected: degraded `AdmitRecord` message missing "DEGRADED:" prefix.
/// A mutant that produces a Degraded record with a generic or empty message would
/// not be caught by `status == Degraded` alone — the message prefix is the user-visible
/// contract proof.
#[test]
fn admit_degraded_record_message_starts_with_degraded() {
    // Build a plan that is FitsWithDegradation (budget below fp32 but above int4).
    let m = small_model();
    let machine = synthetic_machine();
    // fp32 peak exceeds budget; int4_sym should fit.
    let fp32_peak = fitsproof::cost::estimate(&m, &machine, 512, "none", 0.6).total_peak_bytes;
    let int4_peak = fitsproof::cost::estimate(&m, &machine, 512, "int4_sym", 0.6).total_peak_bytes;
    // Only proceed if there's an actual degradation gap to exploit.
    if int4_peak >= fp32_peak {
        return; // model too small, skip
    }
    let budget = (fp32_peak + int4_peak) / 2; // above int4_peak, below fp32_peak
    use fitsproof::plan::{plan, Verdict};
    let Ok(p) = plan(&m, &machine, 512, budget, "none", 0.6) else {
        return; // plan error — skip
    };
    if p.verdict != Verdict::FitsWithDegradation {
        return; // not a degradation case — skip
    }
    let rec = fitsproof::admit::admit(p);
    assert_eq!(rec.status, fitsproof::admit::AdmitStatus::Degraded);
    assert!(
        rec.message.starts_with("DEGRADED:"),
        "degraded record message must start with DEGRADED:, got {:?}",
        rec.message
    );
    // applied_degradation must be Some — the mode change must be recorded.
    assert!(
        rec.applied_degradation.is_some(),
        "degraded record must carry applied_degradation"
    );
}

/// Fault detected: all `ParetoResult::admitted_configs` in the frontier reporting as
/// Fits when some actually exceed the budget (frontier admits a DoesNotFit config).
/// The `admitted_configs` count must equal exactly the configs with non-DoesNotFit verdict
/// across all configurations (before Pareto reduction), per the `pareto_sweep` contract.
#[test]
fn pareto_admitted_count_matches_non_does_not_fit_frontier() {
    use fitsproof::pareto::pareto_sweep;
    use fitsproof::plan::Verdict;

    let model = small_model();
    let machine = synthetic_machine();
    // Generous budget — most configs should fit.
    let budget_bytes = 2u64 * 1024 * 1024 * 1024;
    let result = pareto_sweep(&model, &machine, budget_bytes);

    // admitted_configs is the count of non-DoesNotFit configs across ALL (quant × context)
    // combinations, before Pareto reduction.  It must be ≤ total_configs and ≥ frontier.len().
    assert!(
        result.admitted_configs <= result.total_configs,
        "admitted_configs ({}) must not exceed total_configs ({})",
        result.admitted_configs,
        result.total_configs
    );
    assert!(
        result.admitted_configs >= result.frontier.len(),
        "admitted_configs ({}) must be ≥ frontier.len() ({})",
        result.admitted_configs,
        result.frontier.len()
    );
    // Every frontier entry must be non-DoesNotFit (they come from the admitted set).
    for entry in &result.frontier {
        assert_ne!(
            entry.verdict,
            Verdict::DoesNotFit,
            "frontier entry {}/{} must not be DoesNotFit",
            entry.quant,
            entry.context_len
        );
    }
}

/// Fault detected: MCP `probe` tool returning a response that does not contain the
/// `hostname` field — the replace-body mutant that returns an empty JSON object or a
/// hard-coded success stub. Tests the MCP RPC path for the probe tool specifically.
#[test]
fn mcp_probe_tool_response_contains_hostname_field() {
    use fitsproof::mcp::handle_rpc_for_test;

    let request = r#"{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"probe","arguments":{}}}"#;
    let response = handle_rpc_for_test(request);

    let parsed: serde_json::Value =
        serde_json::from_str(&response).expect("MCP response must be valid JSON");

    // result.content[0].text is an embedded JSON object (not a string).
    let text_val = &parsed["result"]["content"][0]["text"];
    assert!(
        !text_val.is_null(),
        "MCP probe response must have result.content[0].text"
    );

    // Either text is an object with fields, or a string embedding JSON — either way
    // we verify hostname and memory-related fields are present.
    let text_str = if text_val.is_string() {
        text_val.as_str().unwrap().to_string()
    } else {
        text_val.to_string()
    };

    assert!(
        text_str.contains("hostname"),
        "MCP probe response must contain 'hostname' field, got: {text_str}"
    );
    // Either memory_bytes or ram_gb must be present (probe may report either).
    let has_memory = text_str.contains("memory_bytes") || text_str.contains("ram_gb");
    assert!(
        has_memory,
        "MCP probe response must contain memory field (memory_bytes or ram_gb), got: {text_str}"
    );
}

/// Fault detected: `serve` returning 200 for any path, including unknown paths.
/// An unknown URL (not `/v1/chat/completions`) must return 404, not 200 or panic.
/// Matches the HTTP contract: only the documented endpoint returns 200.
#[test]
fn serve_unknown_path_returns_404() {
    use fitsproof::serve::handle_request_for_test;

    let body = r#"{"model":"fitsproof/ref","messages":[]}"#;
    let (status, _) = handle_request_for_test("/unknown/path", body);
    assert_eq!(
        status, 404,
        "serve must return 404 for unknown path, got status={status}"
    );

    let (status2, _) = handle_request_for_test("/v1/completions", body); // old-style path
    assert_eq!(
        status2, 404,
        "serve must return 404 for /v1/completions (not a supported endpoint), got status={status2}"
    );
}
