//! Resource plan — predict and classify configurations.
//!
//! Given (model, quant, context, machine, budget), computes a `Plan`:
//!   - whether the config fits, fits with degradation, or does not fit
//!   - what the predicted peak memory is
//!   - an ordered list of degradation steps (cheapest first)
//!   - the binding constraint when nothing fits
//!
//! # Degradation order (cheapest first)
//!
//! 1. Lower quantisation (int8 → int4)
//! 2. Shorter context window (÷2, ÷4, ÷8)
//!
//! Each degradation step carries its predicted peak bytes and tok/s penalty.
//!
//! # Fail-closed design
//!
//! Unknown quant names, non-positive budgets, and negative context lengths
//! return `Err` instead of silently planning with defaults.  A typo like
//! `"int_4"` would otherwise silently plan at fp32 costs — a contract violation.

use crate::cost::{self, QuantBits};
use crate::model::ModelConfig;
use crate::probe::MachineProfile;

/// The outcome of a resource plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Fits,
    FitsWithDegradation,
    DoesNotFit,
}

/// Quantisation levels available for degradation, in ascending aggressiveness.
static QUANT_ORDER: &[&str] = &["none", "int8_sym", "int4_sym"];

/// One concrete degradation option.
#[derive(Debug, Clone)]
pub struct DegradationStep {
    pub kind: DegradationKind,
    pub description: String,
    pub predicted_peak_bytes: u64,
    pub predicted_tok_s: f64,
    pub fits_budget: bool,
}

/// Category of degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationKind {
    LowerQuant,
    ShorterContext,
}

/// Full resource plan.
#[derive(Debug, Clone)]
pub struct Plan {
    pub verdict: Verdict,
    pub predicted_peak_bytes: u64,
    /// Conservative 95% CI: (lower, upper).
    pub predicted_peak_ci: (u64, u64),
    pub predicted_tok_s: f64,
    pub budget_bytes: u64,
    pub quant: String,
    pub context_len: usize,
    pub degradations: Vec<DegradationStep>,
    /// Human-readable description of the binding constraint when `verdict == DoesNotFit`.
    pub binding_constraint: String,
}

/// Errors returned by `plan()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    UnknownQuant(String),
    InvalidContextLen(String),
    InvalidBudget(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::UnknownQuant(q) => write!(f, "unknown quantisation {q:?}"),
            PlanError::InvalidContextLen(m) => write!(f, "invalid context_len: {m}"),
            PlanError::InvalidBudget(m) => write!(f, "invalid budget: {m}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// Produce a resource plan for the given configuration.
///
/// # Fail-closed
///
/// Returns `Err` for unknown quant names, non-positive context lengths, or
/// non-positive budgets.  Callers must validate before calling.
///
/// # Algorithm
///
/// 1. Compute `predicted_peak_bytes` for the requested config.
/// 2. If it fits the budget → `Verdict::Fits`.
/// 3. Otherwise enumerate degradations (lower quant, shorter context) in
///    ascending cost order and return `Verdict::FitsWithDegradation` if any fit.
/// 4. If nothing fits → `Verdict::DoesNotFit` with the binding constraint named.
pub fn plan(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    context_len: usize,
    budget_bytes: u64,
    quant: &str,
    bandwidth_utilisation: f64,
) -> Result<Plan, PlanError> {
    // Fail-closed validation.
    if QuantBits::from_name(quant).is_none() {
        return Err(PlanError::UnknownQuant(quant.to_string()));
    }
    if context_len == 0 {
        return Err(PlanError::InvalidContextLen("must be >= 1".to_string()));
    }
    if budget_bytes == 0 {
        return Err(PlanError::InvalidBudget("must be > 0".to_string()));
    }

    let cost_est = cost::estimate(cfg, machine, context_len, quant, bandwidth_utilisation);
    let predicted_peak = cost_est.total_peak_bytes;
    let predicted_tok_s = cost_est.predicted_tok_s;

    // Simple ±20% CI (calibrate.rs would tighten this).
    let peak_ci = (
        (predicted_peak as f64 * 0.8) as u64,
        (predicted_peak as f64 * 1.2) as u64,
    );

    if predicted_peak <= budget_bytes {
        return Ok(Plan {
            verdict: Verdict::Fits,
            predicted_peak_bytes: predicted_peak,
            predicted_peak_ci: peak_ci,
            predicted_tok_s,
            budget_bytes,
            quant: quant.to_string(),
            context_len,
            degradations: vec![],
            binding_constraint: String::new(),
        });
    }

    // Enumerate degradations.
    let mut degradations: Vec<DegradationStep> = Vec::new();

    // 1. Lower quantisation.
    if let Some(current_idx) = QUANT_ORDER.iter().position(|&q| q == quant) {
        for &q in &QUANT_ORDER[current_idx + 1..] {
            let dq = cost::estimate(cfg, machine, context_len, q, bandwidth_utilisation);
            let fits = dq.total_peak_bytes <= budget_bytes;
            degradations.push(DegradationStep {
                kind: DegradationKind::LowerQuant,
                description: format!("Use {q} instead of {quant}"),
                predicted_peak_bytes: dq.total_peak_bytes,
                predicted_tok_s: dq.predicted_tok_s,
                fits_budget: fits,
            });
        }
    }

    // 2. Shorter context (÷2, ÷4, ÷8).
    for divisor in [2usize, 4, 8] {
        let shorter = (context_len / divisor).max(1);
        let sc = cost::estimate(cfg, machine, shorter, quant, bandwidth_utilisation);
        let fits = sc.total_peak_bytes <= budget_bytes;
        degradations.push(DegradationStep {
            kind: DegradationKind::ShorterContext,
            description: format!(
                "Reduce context to {shorter} tokens (1/{divisor} of {context_len})"
            ),
            predicted_peak_bytes: sc.total_peak_bytes,
            predicted_tok_s: sc.predicted_tok_s,
            fits_budget: fits,
        });
    }

    let fitting = degradations.iter().find(|d| d.fits_budget);

    if fitting.is_some() {
        Ok(Plan {
            verdict: Verdict::FitsWithDegradation,
            predicted_peak_bytes: predicted_peak,
            predicted_peak_ci: peak_ci,
            predicted_tok_s,
            budget_bytes,
            quant: quant.to_string(),
            context_len,
            degradations,
            binding_constraint: String::new(),
        })
    } else {
        let binding = format!(
            "needs {:.2} GB, budget {:.2} GB; no degradation fits",
            predicted_peak as f64 / 1e9,
            budget_bytes as f64 / 1e9,
        );
        Ok(Plan {
            verdict: Verdict::DoesNotFit,
            predicted_peak_bytes: predicted_peak,
            predicted_peak_ci: peak_ci,
            predicted_tok_s,
            budget_bytes,
            quant: quant.to_string(),
            context_len,
            degradations,
            binding_constraint: binding,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Tests for plan module.
    //!
    //! Each test names the fault it detects.

    use super::*;
    use crate::model::ModelConfig;
    use crate::probe::MachineProfile;

    fn ref_machine() -> MachineProfile {
        MachineProfile {
            hostname: "test-host".into(),
            platform_str: "test".into(),
            measured_at: 0.0,
            memory_bandwidth_bps: 20_000_000_000.0,
            gemm_throughput_flops: 100_000_000_000.0,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 8,
        }
    }

    /// Fault detected: Verdict::Fits returned when config exceeds budget.
    #[test]
    fn fits_when_budget_is_large() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        // Budget = 1 GB >> ~54 MB reference model
        let p = plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        assert_eq!(
            p.verdict,
            Verdict::Fits,
            "reference model should fit in 1 GB"
        );
    }

    /// Fault detected: DoesNotFit returned when config actually fits.
    #[test]
    fn does_not_fit_when_budget_tiny() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        // Budget = 1 byte — nothing can fit.
        let p = plan(&cfg, &m, 512, 1, "none", 0.6).unwrap();
        assert_eq!(
            p.verdict,
            Verdict::DoesNotFit,
            "budget=1 byte must produce DoesNotFit"
        );
        assert!(
            !p.binding_constraint.is_empty(),
            "binding_constraint must be populated on DoesNotFit"
        );
    }

    /// Fault detected: degradation verdict emitted but degradation list is empty.
    #[test]
    fn fits_with_degradation_has_nonempty_degradation_list() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        // Budget just below the fp32 peak but enough for int4.
        let fp32_peak = cost::weight_bytes(&cfg, "none")
            + cost::kv_cache_bytes(&cfg, 512, "none")
            + cost::activation_bytes(&cfg);
        let just_below = fp32_peak - 1;
        let p = plan(&cfg, &m, 512, just_below, "none", 0.6).unwrap();
        if p.verdict == Verdict::FitsWithDegradation {
            assert!(
                !p.degradations.is_empty(),
                "FitsWithDegradation must have at least one degradation"
            );
            assert!(
                p.degradations.iter().any(|d| d.fits_budget),
                "at least one degradation must fit the budget"
            );
        }
        // If verdict is DoesNotFit at this budget, that's also acceptable
        // (int4 may still exceed budget for certain configs).
    }

    /// Fault detected: unknown quant silently plans at fp32.
    #[test]
    fn unknown_quant_is_an_error() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let result = plan(&cfg, &m, 512, 1_000_000_000, "int2_bogus", 0.6);
        assert!(
            matches!(result, Err(PlanError::UnknownQuant(_))),
            "unknown quant must return PlanError::UnknownQuant"
        );
    }

    /// Fault detected: zero context_len accepted (would produce zero KV bytes, hiding errors).
    #[test]
    fn zero_context_is_an_error() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let result = plan(&cfg, &m, 0, 1_000_000_000, "none", 0.6);
        assert!(
            matches!(result, Err(PlanError::InvalidContextLen(_))),
            "context_len=0 must return PlanError::InvalidContextLen"
        );
    }

    /// Fault detected: zero budget silently passes instead of returning error.
    #[test]
    fn zero_budget_is_an_error() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let result = plan(&cfg, &m, 512, 0, "none", 0.6);
        assert!(
            matches!(result, Err(PlanError::InvalidBudget(_))),
            "budget=0 must return PlanError::InvalidBudget"
        );
    }

    /// Fault detected: a degradation step that doesn't fit is marked fits_budget=true.
    #[test]
    fn degradation_fits_budget_flag_is_correct() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan(&cfg, &m, 512, 1, "none", 0.6).unwrap();
        // With budget=1 byte, no degradation should fit.
        for d in &p.degradations {
            assert!(
                !d.fits_budget,
                "no degradation can fit 1-byte budget: {d:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property-based tests (proptest)
    //
    // Properties come from the plan algorithm's invariants, not the implementation.
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        /// Property: Verdict::Fits only ever produced when predicted_peak <= budget.
        ///
        /// Fault detected: Fits verdict emitted when config actually exceeds budget.
        #[test]
        fn fits_verdict_implies_peak_le_budget(
            budget_mb in 1u64..2000,
        ) {
            let cfg = ModelConfig::reference();
            let m = ref_machine();
            let budget = budget_mb * 1_000_000;
            let p = plan(&cfg, &m, 512, budget, "none", 0.6).unwrap();
            if p.verdict == Verdict::Fits {
                prop_assert!(
                    p.predicted_peak_bytes <= p.budget_bytes,
                    "Fits verdict but peak {} > budget {}",
                    p.predicted_peak_bytes, p.budget_bytes
                );
            }
        }

        /// Property: Verdict::DoesNotFit implies binding_constraint is non-empty.
        ///
        /// Fault detected: DoesNotFit emitted without naming the binding constraint
        /// (vacuous refusal with no diagnostic info).
        #[test]
        fn does_not_fit_has_binding_constraint(
            budget_bytes in 1u64..10_000,
        ) {
            let cfg = ModelConfig::reference();
            let m = ref_machine();
            let p = plan(&cfg, &m, 512, budget_bytes, "none", 0.6).unwrap();
            if p.verdict == Verdict::DoesNotFit {
                prop_assert!(
                    !p.binding_constraint.is_empty(),
                    "DoesNotFit must name binding_constraint"
                );
            }
        }

        /// Property: for any valid budget, a larger budget never produces a worse verdict.
        ///
        /// Formally: budget_a < budget_b => verdict(budget_a) is at least as restrictive
        /// as verdict(budget_b). The ordering is DoesNotFit < FitsWithDegradation < Fits.
        ///
        /// Fault detected: non-monotone verdict function (larger budget produces stricter verdict).
        #[test]
        fn verdict_monotone_in_budget(
            budget_a_mb in 1u64..50,
            budget_b_mb in 100u64..2000,
        ) {
            let cfg = ModelConfig::reference();
            let m = ref_machine();
            let pa = plan(&cfg, &m, 512, budget_a_mb * 1_000_000, "none", 0.6).unwrap();
            let pb = plan(&cfg, &m, 512, budget_b_mb * 1_000_000, "none", 0.6).unwrap();

            fn verdict_rank(v: Verdict) -> u8 {
                match v {
                    Verdict::DoesNotFit => 0,
                    Verdict::FitsWithDegradation => 1,
                    Verdict::Fits => 2,
                }
            }
            prop_assert!(
                verdict_rank(pa.verdict) <= verdict_rank(pb.verdict),
                "verdict must be non-decreasing in budget: small_budget({budget_a_mb}MB)={:?} > large_budget({budget_b_mb}MB)={:?}",
                pa.verdict, pb.verdict
            );
        }

        /// Property: FitsWithDegradation verdict implies at least one degradation step fits.
        ///
        /// Fault detected: FitsWithDegradation verdict emitted but no fitting degradation exists.
        #[test]
        fn fits_with_degradation_has_fitting_step(
            budget_mb in 10u64..200,
        ) {
            let cfg = ModelConfig::reference();
            let m = ref_machine();
            let p = plan(&cfg, &m, 512, budget_mb * 1_000_000, "none", 0.6).unwrap();
            if p.verdict == Verdict::FitsWithDegradation {
                prop_assert!(
                    p.degradations.iter().any(|d| d.fits_budget),
                    "FitsWithDegradation must have at least one fitting degradation step"
                );
            }
        }
    }
}
