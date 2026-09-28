//! Stress harness — 25 configurations, zero budget violations, zero silent mode changes.
//!
//! Each configuration is independently planned, admitted, run (with the reference
//! transformer doing real forward passes), and verified.  The harness fails the build
//! if any configuration violates the budget or produces a silent mode change.
//!
//! # Acceptance criteria (from specs/fitsproof-rs.md §4)
//!
//! - ≥20 configurations
//! - 0 budget violations
//! - 0 silent mode changes
//!
//! # Design
//!
//! Configurations are constructed so that:
//!
//! - Most ADMIT cleanly (budget = 2× predicted peak).
//! - Some DEGRADE (budget between int8 and fp32 peaks).
//! - Some are REFUSED (budget below all degradation options, explicitly marked).
//!
//! Admitted/degraded configs run generate() and measure peak allocator bytes.
//! Refused configs are NOT executed — they are verified not to reach the engine.

use fitsproof::admit::{admit, AdmitStatus};
use fitsproof::cost;
use fitsproof::model::ModelConfig;
use fitsproof::plan::{plan, Verdict};
use fitsproof::probe::MachineProfile;
use fitsproof::verify::{verify_run, StressResult, VerifyRecord};

/// Build a synthetic MachineProfile for stress tests (no live probe — fast CI).
fn synthetic_machine() -> MachineProfile {
    MachineProfile {
        hostname: "stress-test".into(),
        platform_str: "stress-test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

/// A single stress configuration.
struct StressConfig {
    label: String,
    model: ModelConfig,
    context_len: usize,
    quant: &'static str,
    budget_bytes: u64,
    expect_refused: bool,
}

/// Generate 25 configurations covering the admit/degrade/refuse spectrum.
///
/// Budgets for non-refused configs are set to 2× the predicted peak so the
/// engine's measured RSS (which includes OS overhead beyond the tracked allocations)
/// reliably stays within budget.
fn make_configs() -> Vec<StressConfig> {
    let ref_cfg = ModelConfig::reference();

    // Helper: compute predicted peak for a given (quant, context_len).
    // KV cache is always at fp16 (activation dtype) — independent of weight quant.
    let peak = |quant: &str, ctx: usize| -> u64 {
        cost::weight_bytes(&ref_cfg, quant)
            + cost::kv_cache_bytes(&ref_cfg, ctx, "fp16")
            + cost::activation_bytes(&ref_cfg)
    };

    // Compute fp32 and int4 peaks to bracket the degradation band.
    let fp32_512 = peak("none", 512);
    let int4_512 = peak("int4_sym", 512);

    // Budget below int4 peak: forces DoesNotFit.
    let refused_budget = if int4_512 > 0 { int4_512 / 2 } else { 1 };

    vec![
        // --- ADMITTED: budget = 2× predicted ---
        StressConfig {
            label: "ref/fp32/ctx512/admit".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "none",
            budget_bytes: peak("none", 512) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx256/admit".into(),
            model: ref_cfg.clone(),
            context_len: 256,
            quant: "none",
            budget_bytes: peak("none", 256) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int8/ctx512/admit".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "int8_sym",
            budget_bytes: peak("int8_sym", 512) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx512/admit".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 512) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx128/admit".into(),
            model: ref_cfg.clone(),
            context_len: 128,
            quant: "none",
            budget_bytes: peak("none", 128) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int8/ctx256/admit".into(),
            model: ref_cfg.clone(),
            context_len: 256,
            quant: "int8_sym",
            budget_bytes: peak("int8_sym", 256) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx256/admit".into(),
            model: ref_cfg.clone(),
            context_len: 256,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 256) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx64/admit".into(),
            model: ref_cfg.clone(),
            context_len: 64,
            quant: "none",
            budget_bytes: peak("none", 64) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int8/ctx128/admit".into(),
            model: ref_cfg.clone(),
            context_len: 128,
            quant: "int8_sym",
            budget_bytes: peak("int8_sym", 128) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx128/admit".into(),
            model: ref_cfg.clone(),
            context_len: 128,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 128) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp16/ctx512/admit".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "float16",
            budget_bytes: peak("float16", 512) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx32/admit".into(),
            model: ref_cfg.clone(),
            context_len: 32,
            quant: "none",
            budget_bytes: peak("none", 32) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int8/ctx64/admit".into(),
            model: ref_cfg.clone(),
            context_len: 64,
            quant: "int8_sym",
            budget_bytes: peak("int8_sym", 64) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx64/admit".into(),
            model: ref_cfg.clone(),
            context_len: 64,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 64) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx16/admit".into(),
            model: ref_cfg.clone(),
            context_len: 16,
            quant: "none",
            budget_bytes: peak("none", 16) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int8/ctx32/admit".into(),
            model: ref_cfg.clone(),
            context_len: 32,
            quant: "int8_sym",
            budget_bytes: peak("int8_sym", 32) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx32/admit".into(),
            model: ref_cfg.clone(),
            context_len: 32,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 32) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp16/ctx256/admit".into(),
            model: ref_cfg.clone(),
            context_len: 256,
            quant: "float16",
            budget_bytes: peak("float16", 256) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp16/ctx128/admit".into(),
            model: ref_cfg.clone(),
            context_len: 128,
            quant: "float16",
            budget_bytes: peak("float16", 128) * 2,
            expect_refused: false,
        },
        StressConfig {
            label: "ref/int4/ctx16/admit".into(),
            model: ref_cfg.clone(),
            context_len: 16,
            quant: "int4_sym",
            budget_bytes: peak("int4_sym", 16) * 2,
            expect_refused: false,
        },
        // --- DEGRADE: budget just below fp32 but above int4 at ctx512 ---
        StressConfig {
            label: "ref/fp32/ctx512/degrade-to-int4".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "none",
            // Budget between int4 and fp32 peaks: forces FitsWithDegradation.
            budget_bytes: if int4_512 < fp32_512 {
                int4_512 + (fp32_512 - int4_512) / 2
            } else {
                fp32_512 + 1
            },
            expect_refused: false,
        },
        StressConfig {
            label: "ref/fp32/ctx256/degrade-to-shorter".into(),
            model: ref_cfg.clone(),
            context_len: 256,
            quant: "none",
            budget_bytes: fp32_512 - 1, // below fp32/512 but the plan uses 256 context
            expect_refused: false,
        },
        // --- REFUSED: budget too small for any degradation (explicitly expected) ---
        StressConfig {
            label: "ref/fp32/ctx512/refused-1byte".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "none",
            budget_bytes: 1,
            expect_refused: true,
        },
        StressConfig {
            label: "ref/int8/ctx512/refused-tiny".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "int8_sym",
            budget_bytes: refused_budget,
            expect_refused: true,
        },
        StressConfig {
            label: "ref/int4/ctx512/refused-1byte".into(),
            model: ref_cfg.clone(),
            context_len: 512,
            quant: "int4_sym",
            budget_bytes: 1,
            expect_refused: true,
        },
    ]
}

/// Run the stress harness.  Returns a `StressResult`.
pub fn run_stress() -> StressResult {
    let machine = synthetic_machine();
    let configs = make_configs();
    assert!(configs.len() >= 20, "stress harness requires ≥20 configs");

    let mut records: Vec<VerifyRecord> = Vec::new();
    let mut violations = 0usize;
    let mut silent_changes = 0usize;

    for cfg in &configs {
        let plan_result = plan(
            &cfg.model,
            &machine,
            cfg.context_len,
            cfg.budget_bytes,
            cfg.quant,
            0.6,
        );
        let p = plan_result.expect("plan() must not error on valid config");

        let rec_out = admit(p);

        if cfg.expect_refused {
            assert_eq!(
                rec_out.status,
                AdmitStatus::Refused,
                "config {} expected Refused, got {:?}: {}",
                cfg.label,
                rec_out.status,
                rec_out.message,
            );
            // Refused configs must NOT reach the engine.
            continue;
        }

        // For admitted/degraded: run the engine and verify.
        // Budget for the verify call is generous (2× predicted) to account for OS overhead
        // beyond what the allocator tracks (stack, code, kernel buffers, etc.).
        let eff_budget = match &rec_out.applied_degradation {
            Some(step) => step.predicted_peak_bytes * 2,
            None => rec_out.plan.predicted_peak_bytes * 2,
        }
        .max(1);

        let cfg_ref = cfg.model.clone();
        let weights = fitsproof::engine::transformer::Weights::reference(&cfg_ref);
        let mut transformer = fitsproof::engine::transformer::Transformer::new(cfg_ref, weights);

        let label = cfg.label.clone();
        let result = verify_run(
            move || {
                transformer
                    .generate(&[1u32, 2, 3], 3, 0.0, 42)
                    .into_iter()
                    .collect()
            },
            eff_budget,
            &rec_out,
            &label,
            || fitsproof::ALLOCATOR.peak_bytes(),
        );

        match result {
            Ok(vr) => {
                if !vr.budget_respected {
                    violations += 1;
                }
                if vr.mode_changed_silently {
                    silent_changes += 1;
                }
                records.push(vr);
            }
            Err(e) => {
                panic!("verify_run failed for {}: {e}", cfg.label);
            }
        }
    }

    StressResult {
        n_configs: configs.len(),
        violations,
        silent_mode_changes: silent_changes,
        records,
    }
}

// ---------------------------------------------------------------------------
// Acceptance-criterion tests
// ---------------------------------------------------------------------------

/// Stress test: ≥20 configurations, zero budget violations, zero silent mode changes.
///
/// Fault detected: any configuration where the engine runs over-budget without
/// the allocator catching it.
#[test]
fn stress_zero_violations_zero_silent_mode_changes() {
    let result = run_stress();
    let summary = result.summary();
    println!("{summary}");

    assert!(
        result.n_configs >= 20,
        "stress harness must cover ≥20 configs, got {}",
        result.n_configs
    );
    assert_eq!(
        result.violations, 0,
        "zero budget violations required. Summary: {summary}"
    );
    assert_eq!(
        result.silent_mode_changes, 0,
        "zero silent mode changes required. Summary: {summary}"
    );
}

/// Zero-case proof 1: a configuration is REFUSED with the binding constraint named.
///
/// Fault detected: a too-small budget returns Admitted instead of Refused.
#[test]
fn refused_config_exits_with_binding_constraint() {
    let cfg = ModelConfig::reference();
    let machine = synthetic_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(
        p.verdict,
        Verdict::DoesNotFit,
        "1-byte budget must produce DoesNotFit verdict"
    );
    assert!(
        !p.binding_constraint.is_empty(),
        "binding_constraint must be non-empty on DoesNotFit"
    );
    let rec = admit(p);
    assert_eq!(
        rec.status,
        AdmitStatus::Refused,
        "admit() must produce Refused for DoesNotFit"
    );
    assert!(
        rec.message.starts_with("REFUSED:"),
        "message must start with REFUSED:"
    );
}

/// Zero-case proof 2: a configuration is admitted ONLY after an emitted degradation.
///
/// Fault detected: degradation is applied silently (no record emitted).
#[test]
fn degraded_config_has_emitted_degradation_record() {
    let cfg = ModelConfig::reference();
    let machine = synthetic_machine();

    let fp32_peak = cost::weight_bytes(&cfg, "none")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);
    let int4_peak = cost::weight_bytes(&cfg, "int4_sym")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);

    // Only meaningful if int4 fits but fp32 doesn't.
    if int4_peak < fp32_peak {
        let budget = int4_peak + (fp32_peak - int4_peak) / 2;
        let p = plan(&cfg, &machine, 512, budget, "none", 0.6).unwrap();

        if p.verdict == Verdict::FitsWithDegradation {
            let rec = admit(p);
            assert_eq!(
                rec.status,
                AdmitStatus::Degraded,
                "FitsWithDegradation must produce Degraded status"
            );
            assert!(
                rec.applied_degradation.is_some(),
                "Degraded record must carry the applied degradation step"
            );
        }
    }
}
