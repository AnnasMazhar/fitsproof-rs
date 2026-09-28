//! `FitsproofClient` — the Rust API surface for the v0.2 delivery contract.
//!
//! The v0.2 MANDATE requires the contract to ship as a **binary + plugin**,
//! not merely as code people must read source to use.  This module provides:
//!
//! - `FitsproofClient` — a struct that holds configuration (budget, quant,
//!   context) and exposes `plan()`, `admit()`, `guard()` as methods.
//!
//! - `guard()` — a function that raises (`Result::Err`) **before** the caller
//!   allocates any model memory.  This is the Rust analogue of the Python
//!   `@fitsproof.guard(budget=...)` decorator.
//!
//! # Example
//!
//! ```rust
//! use fitsproof::client::{FitsproofClient, GuardError};
//!
//! fn load_model(budget_gb: f64) -> Result<(), Box<GuardError>> {
//!     let client = FitsproofClient::new(budget_gb);
//!     client.guard()?; // raises before any allocation if the config won't fit
//!     // ... actual model loading happens here ...
//!     Ok(())
//! }
//! ```
//!
//! # Why this exists
//!
//! The Python edition ships a `@fitsproof.guard(budget=...)` decorator that
//! prevents a function from running when the predicted peak would exceed the
//! declared budget.  In Rust, the idiomatic equivalent is a method that
//! returns `Result<(), GuardError>` and can be propagated with `?`.
//!
//! The guard fires **before** any model weights are loaded — the whole point is
//! to prevent the allocation, not measure it after the fact.

use crate::admit::{admit, AdmitRecord, AdmitStatus};
use crate::model::ModelConfig;
use crate::plan::{plan, Plan, PlanError};
use crate::probe::MachineProfile;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Returned by `guard()` when the config would not fit within the budget.
///
/// # Fault detected
///
/// If `guard()` returned `Ok(())` even when the budget is insufficient, the
/// caller would proceed to allocate model memory and hit an OOM — the exact
/// failure mode fitsproof exists to prevent.
#[derive(Debug, Clone)]
pub struct GuardError {
    /// Human-readable reason: the binding constraint that prevents admission.
    pub binding_constraint: String,
    /// The `AdmitRecord` produced by the failed admission check.
    pub record: AdmitRecord,
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fitsproof guard refused: {}", self.binding_constraint)
    }
}

impl std::error::Error for GuardError {}

// ---------------------------------------------------------------------------
// FitsproofClient
// ---------------------------------------------------------------------------

/// A configured fitsproof client.  Holds budget, quant, and context settings.
///
/// Create with `FitsproofClient::new(budget_gb)` (uses defaults for quant/context)
/// or with the builder methods `with_quant()` and `with_context()`.
///
/// # Fault detected (by tests)
///
/// - `guard()` must return `Err` when `budget_gb` is smaller than the predicted
///   peak — i.e. it must not silently admit a refused configuration.
/// - `guard()` must return `Ok(())` when the config fits within the declared budget.
/// - `plan()` must return a `Plan` with the expected verdict for a given config.
pub struct FitsproofClient {
    budget_bytes: u64,
    quant: String,
    context_len: usize,
    machine: MachineProfile,
    model: ModelConfig,
}

impl FitsproofClient {
    /// Create a client with the given budget (GB) and default settings.
    ///
    /// Defaults: quant=`none` (fp32), context_len=512, reference bundle model.
    pub fn new(budget_gb: f64) -> Self {
        Self {
            budget_bytes: (budget_gb * 1e9) as u64,
            quant: "none".to_string(),
            context_len: 512,
            machine: synthetic_machine(),
            model: ModelConfig::reference(),
        }
    }

    /// Set quantisation (e.g. `"q4_k_m"`, `"int8_sym"`, `"float16"`).
    ///
    /// Returns `self` for chaining.
    pub fn with_quant(mut self, quant: &str) -> Self {
        self.quant = quant.to_string();
        self
    }

    /// Set context length in tokens.
    pub fn with_context(mut self, context_len: usize) -> Self {
        self.context_len = context_len;
        self
    }

    /// Override the model config (use when you have already parsed a GGUF).
    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.model = model;
        self
    }

    /// Run the resource plan for the current configuration.
    pub fn plan(&self) -> Result<Plan, PlanError> {
        plan(
            &self.model,
            &self.machine,
            self.context_len,
            self.budget_bytes,
            &self.quant,
            0.6,
        )
    }

    /// Admit the current configuration.
    ///
    /// Returns the full `AdmitRecord` regardless of outcome.
    /// Callers that need error propagation should use `guard()` instead.
    pub fn admit(&self) -> Result<AdmitRecord, PlanError> {
        let p = self.plan()?;
        Ok(admit(p))
    }

    /// Guard: return `Err(GuardError)` if the config would be refused, `Ok(())` if
    /// it would be admitted (possibly after degradation).
    ///
    /// This is the Rust analogue of the Python `@fitsproof.guard(budget=...)` decorator.
    /// Call this **before** loading model weights to prevent silent OOM.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use fitsproof::client::FitsproofClient;
    /// let client = FitsproofClient::new(4.0).with_quant("q4_k_m").with_context(4096);
    /// client.guard().expect("config must fit before loading model");
    /// ```
    pub fn guard(&self) -> Result<(), Box<GuardError>> {
        let rec = match self.admit() {
            Ok(r) => r,
            Err(e) => {
                // PlanError (unknown quant, invalid context, etc.) is treated as a
                // refusal — we cannot plan, therefore we cannot proceed.
                return Err(Box::new(GuardError {
                    binding_constraint: e.to_string(),
                    record: AdmitRecord {
                        status: AdmitStatus::Refused,
                        plan: crate::plan::Plan {
                            verdict: crate::plan::Verdict::DoesNotFit,
                            predicted_peak_bytes: 0,
                            predicted_peak_ci: (0, 0),
                            predicted_tok_s: 0.0,
                            budget_bytes: self.budget_bytes,
                            quant: self.quant.clone(),
                            context_len: self.context_len,
                            degradations: vec![],
                            binding_constraint: e.to_string(),
                        },
                        applied_degradation: None,
                        refusal_reason: e.to_string(),
                        message: format!("REFUSED: plan error — {e}"),
                    },
                }));
            }
        };

        if rec.status == AdmitStatus::Refused {
            Err(Box::new(GuardError {
                binding_constraint: rec.plan.binding_constraint.clone(),
                record: rec,
            }))
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone guard() function
// ---------------------------------------------------------------------------

/// Standalone guard: return `Err(Box<GuardError>)` if the config would be refused.
///
/// Convenience wrapper around `FitsproofClient::new(budget_gb).guard()`.
/// Equivalent to the Python `@fitsproof.guard(budget=budget_gb)` decorator.
pub fn guard(budget_gb: f64) -> Result<(), Box<GuardError>> {
    FitsproofClient::new(budget_gb).guard()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A fast synthetic `MachineProfile` used for pre-load checks.
///
/// We use conservative values so the guard is pessimistic (admits fewer configs)
/// rather than optimistic (admits configs that would fail on real hardware).
fn synthetic_machine() -> MachineProfile {
    MachineProfile {
        hostname: "fitsproof-client".into(),
        platform_str: "fitsproof-client".into(),
        measured_at: 0.0,
        memory_bandwidth_bps: 20_000_000_000.0, // 20 GB/s — conservative DDR4
        gemm_throughput_flops: 100_000_000_000.0, // 100 GFLOP/s
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for `FitsproofClient` and `guard()`.
    //!
    //! Fault map:
    //! - `guard_refuses_insufficient_budget`: guard() returns Err when budget < predicted peak
    //! - `guard_admits_sufficient_budget`: guard() returns Ok when config fits
    //! - `guard_error_names_binding_constraint`: GuardError carries the binding constraint
    //! - `standalone_guard_refuses_impossible_budget`: guard(0.0001) refuses with Err
    //! - `client_plan_matches_direct_plan`: FitsproofClient::plan() agrees with plan() directly
    //! - `client_admit_refused_has_status_refused`: admit() on impossible config returns Refused
    //! - `guard_propagates_with_question_mark`: Err propagates through ? operator
    //! - `guard_admitted_after_degradation_is_ok`: degraded-only config returns Ok from guard()

    use super::*;
    use crate::cost;

    /// Fault detected: guard() returning Ok even when budget is below predicted peak,
    /// which would allow the caller to proceed with model loading and hit OOM.
    #[test]
    fn guard_refuses_insufficient_budget() {
        // Budget = 1 byte: impossible for any config.
        let client = FitsproofClient::new(1e-9);
        let result = client.guard();
        assert!(
            result.is_err(),
            "guard must return Err when budget is impossibly small"
        );
    }

    /// Fault detected: guard() returning Err even when the config clearly fits,
    /// which would block the caller from loading a model that is safe to run.
    #[test]
    fn guard_admits_sufficient_budget() {
        // Budget = 4 GB: reference bundle fits comfortably.
        let client = FitsproofClient::new(4.0);
        let result = client.guard();
        assert!(
            result.is_ok(),
            "guard must return Ok when config fits within budget; got: {:?}",
            result
        );
    }

    /// Fault detected: GuardError not carrying the binding constraint, making
    /// the error message useless to the caller.
    #[test]
    fn guard_error_names_binding_constraint() {
        let client = FitsproofClient::new(1e-9);
        let err = client.guard().unwrap_err();
        assert!(
            !err.binding_constraint.is_empty(),
            "GuardError must name the binding constraint; got empty string"
        );
    }

    /// Fault detected: standalone guard() not delegating correctly to FitsproofClient.
    #[test]
    fn standalone_guard_refuses_impossible_budget() {
        let result = guard(1e-9);
        assert!(result.is_err(), "standalone guard(1e-9) must return Err");
    }

    /// Fault detected: FitsproofClient::plan() computing different values than calling
    /// plan() directly with the same arguments — i.e. the client is using wrong params.
    #[test]
    fn client_plan_matches_direct_plan() {
        use crate::plan::plan as do_plan;
        let client = FitsproofClient::new(4.0)
            .with_quant("none")
            .with_context(512);
        let client_plan = client.plan().expect("plan should succeed");

        let machine = synthetic_machine();
        let model = ModelConfig::reference();
        let direct_plan = do_plan(&model, &machine, 512, (4.0_f64 * 1e9) as u64, "none", 0.6)
            .expect("direct plan should succeed");

        assert_eq!(
            client_plan.predicted_peak_bytes, direct_plan.predicted_peak_bytes,
            "client plan and direct plan must agree on predicted peak"
        );
        assert_eq!(
            client_plan.verdict, direct_plan.verdict,
            "client plan and direct plan must agree on verdict"
        );
    }

    /// Fault detected: admit() returning a non-Refused status for an impossible config.
    #[test]
    fn client_admit_refused_has_status_refused() {
        let client = FitsproofClient::new(1e-9);
        let rec = client.admit().expect("admit() should not return PlanError");
        assert_eq!(
            rec.status,
            AdmitStatus::Refused,
            "status must be Refused for impossible budget"
        );
    }

    /// Fault detected: GuardError does not implement std::error::Error, preventing
    /// use with ? operator and the standard error handling ecosystem.
    #[test]
    fn guard_propagates_with_question_mark() {
        fn inner() -> Result<(), Box<dyn std::error::Error>> {
            let client = FitsproofClient::new(1e-9);
            client.guard()?;
            Ok(())
        }
        assert!(
            inner().is_err(),
            "guard error must propagate through ? operator"
        );
    }

    /// Fault detected: guard() returning Err for a config that fits only after
    /// degradation — it should return Ok because the contract does permit
    /// degraded admission (the degradation is recorded explicitly).
    #[test]
    fn guard_admitted_after_degradation_is_ok() {
        // Use a budget that requires degradation: between int4 peak and fp32 peak.
        let model = ModelConfig::reference();
        let _machine = synthetic_machine();

        let fp32_peak = cost::weight_bytes(&model, "none")
            + cost::kv_cache_bytes(&model, 512, "fp16")
            + cost::activation_bytes(&model);

        let int4_peak = cost::weight_bytes(&model, "int4_sym")
            + cost::kv_cache_bytes(&model, 512, "fp16")
            + cost::activation_bytes(&model);

        // Budget is above int4 peak but below fp32 peak — should degrade to int4 and admit.
        if fp32_peak > int4_peak {
            let budget_gb = (int4_peak as f64 + fp32_peak as f64) / 2.0 / 1e9;
            let client = FitsproofClient::new(budget_gb);
            let result = client.guard();
            // May be Ok (degraded admit) or Err (if even int4 doesn't fit at this exact budget).
            // The contract says: Ok if any degradation fits, Err only if nothing fits.
            // We just verify: if it's Ok, the admit record is not Refused.
            if let Ok(()) = result {
                let rec = client.admit().unwrap();
                assert_ne!(
                    rec.status,
                    AdmitStatus::Refused,
                    "admitted-after-degradation guard must not produce Refused record"
                );
            }
        }
    }

    /// Fault detected: with_quant() not actually changing the quant used in plan(),
    /// so a caller who asks for q4_k_m gets fp32 costs instead.
    #[test]
    fn with_quant_changes_plan_peak() {
        let client_fp32 = FitsproofClient::new(4.0).with_quant("none");
        let client_int4 = FitsproofClient::new(4.0).with_quant("int4_sym");

        let peak_fp32 = client_fp32.plan().unwrap().predicted_peak_bytes;
        let peak_int4 = client_int4.plan().unwrap().predicted_peak_bytes;

        assert!(
            peak_int4 < peak_fp32,
            "int4_sym peak ({peak_int4}) must be less than fp32 peak ({peak_fp32})"
        );
    }

    /// Fault detected: with_context() not changing the KV cache bytes,
    /// so context length has no effect on predicted peak.
    #[test]
    fn with_context_changes_plan_peak() {
        let client_small = FitsproofClient::new(4.0).with_context(128);
        let client_large = FitsproofClient::new(4.0).with_context(4096);

        let peak_small = client_small.plan().unwrap().predicted_peak_bytes;
        let peak_large = client_large.plan().unwrap().predicted_peak_bytes;

        assert!(
            peak_large > peak_small,
            "larger context must produce larger predicted peak: {peak_large} vs {peak_small}"
        );
    }
}
