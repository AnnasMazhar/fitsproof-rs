//! The enforcement point — `admit()`.
//!
//! Given a `Plan`, either:
//!   (a) admits it unchanged (`Verdict::Fits`),
//!   (b) applies the cheapest fitting degradation and emits a `DegradedRecord`, or
//!   (c) refuses with an explicit message naming the binding constraint.
//!
//! **INVARIANT**: there is no code path that changes execution mode without
//! emitting an `AdmitRecord`.  This is the central guarantee of fitsproof.
//!
//! Every mode change must go through `admit()`; callers must not bypass it.
//! A REFUSED record must NOT be discarded silently — calling code must inspect
//! the `status` field and act accordingly.

use crate::plan::{DegradationStep, Plan, Verdict};

/// The outcome of an admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitStatus {
    Admitted,
    Degraded,
    Refused,
}

/// Record of an admission decision. Always emitted; never suppressed.
///
/// # Fault detected
///
/// If `admit()` can return without producing a record, a silent mode change
/// can occur. Tests inject a plan that triggers each branch and assert the
/// returned record is correctly populated.
#[derive(Debug, Clone)]
pub struct AdmitRecord {
    pub status: AdmitStatus,
    /// The plan this record was produced from.
    pub plan: Plan,
    /// The degradation step that was applied, if any.
    pub applied_degradation: Option<DegradationStep>,
    /// Non-empty when `status == Refused`.
    pub refusal_reason: String,
    /// Human-readable one-line summary of the decision.
    pub message: String,
}

/// The typed error returned when a configuration is refused.
///
/// Carries the binding constraint so callers can surface a useful message
/// without having to format it themselves.
#[derive(Debug, Clone)]
pub struct DoesNotFitPlan {
    pub binding_constraint: String,
    pub record: AdmitRecord,
}

impl std::fmt::Display for DoesNotFitPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "REFUSED: {}", self.binding_constraint)
    }
}

impl std::error::Error for DoesNotFitPlan {}

/// Enforce the resource contract defined by `plan`.
///
/// Returns an `AdmitRecord` in ALL cases.  The record must be inspected by
/// the caller; silently discarding a `Refused` record is a contract violation.
///
/// # Logic
///
/// - `Verdict::Fits`                → `AdmitStatus::Admitted`, no degradation
/// - `Verdict::FitsWithDegradation` → `AdmitStatus::Degraded`, cheapest fitting degradation
/// - `Verdict::DoesNotFit`          → `AdmitStatus::Refused`, binding constraint named
///
/// # Fault detected
///
/// The `DoesNotFit` branch must emit `Refused`, not `Degraded`.  A bug that
/// emits `Degraded` with `None` degradation would silently pass a refused config.
pub fn admit(plan: Plan) -> AdmitRecord {
    match plan.verdict {
        Verdict::Fits => {
            let msg = format!(
                "ADMITTED: {:.3} GB predicted peak <= {:.3} GB budget (margin: {:.1} MB)",
                plan.predicted_peak_bytes as f64 / 1e9,
                plan.budget_bytes as f64 / 1e9,
                (plan.budget_bytes as f64 - plan.predicted_peak_bytes as f64) / 1e6,
            );
            AdmitRecord {
                status: AdmitStatus::Admitted,
                plan,
                applied_degradation: None,
                refusal_reason: String::new(),
                message: msg,
            }
        }

        Verdict::FitsWithDegradation => {
            // Find cheapest fitting degradation (first in list that fits).
            let fitting = plan.degradations.iter().find(|d| d.fits_budget).cloned();

            match fitting {
                Some(step) => {
                    let msg = format!(
                        "DEGRADED: base config needs {:.3} GB > budget {:.3} GB. \
                         Applying: {}. New predicted peak: {:.3} GB.",
                        plan.predicted_peak_bytes as f64 / 1e9,
                        plan.budget_bytes as f64 / 1e9,
                        step.description,
                        step.predicted_peak_bytes as f64 / 1e9,
                    );
                    AdmitRecord {
                        status: AdmitStatus::Degraded,
                        plan,
                        applied_degradation: Some(step),
                        refusal_reason: String::new(),
                        message: msg,
                    }
                }
                None => {
                    // Defensive: verdict said degradation is available, but none fit.
                    // Refuse for safety — never silently proceed.
                    let msg = format!(
                        "REFUSED (internal inconsistency): verdict=FitsWithDegradation \
                         but no degradation fits. budget={:.3} GB, predicted={:.3} GB.",
                        plan.budget_bytes as f64 / 1e9,
                        plan.predicted_peak_bytes as f64 / 1e9,
                    );
                    let reason = "Internal inconsistency: no degradation fits".to_string();
                    AdmitRecord {
                        status: AdmitStatus::Refused,
                        plan,
                        applied_degradation: None,
                        refusal_reason: reason.clone(),
                        message: msg,
                    }
                }
            }
        }

        Verdict::DoesNotFit => {
            let msg = format!("REFUSED: {}", plan.binding_constraint);
            let reason = plan.binding_constraint.clone();
            AdmitRecord {
                status: AdmitStatus::Refused,
                plan,
                applied_degradation: None,
                refusal_reason: reason,
                message: msg,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the admit module.
    //!
    //! Each test names the fault it detects (QUALITY-CONTRACT §1).

    use super::*;
    use crate::model::ModelConfig;
    use crate::plan;
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

    /// Fault detected: admit() returns Refused for a config that fits.
    #[test]
    fn admitted_when_fits() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        assert_eq!(p.verdict, Verdict::Fits);
        let rec = admit(p);
        assert_eq!(
            rec.status,
            AdmitStatus::Admitted,
            "fits plan must produce Admitted record"
        );
        assert!(
            rec.applied_degradation.is_none(),
            "no degradation should be applied for a Fits plan"
        );
        assert!(
            rec.message.starts_with("ADMITTED:"),
            "message must start with ADMITTED:"
        );
    }

    /// Fault detected: admit() returns Admitted for a refused config (contract bypass).
    #[test]
    fn refused_when_does_not_fit() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        // Budget = 1 byte — nothing can fit.
        let p = plan::plan(&cfg, &m, 512, 1, "none", 0.6).unwrap();
        assert_eq!(p.verdict, Verdict::DoesNotFit);
        let rec = admit(p);
        assert_eq!(
            rec.status,
            AdmitStatus::Refused,
            "DoesNotFit plan must produce Refused record"
        );
        assert!(
            !rec.refusal_reason.is_empty(),
            "refusal_reason must be populated on Refused"
        );
        assert!(
            rec.message.starts_with("REFUSED:"),
            "message must start with REFUSED:"
        );
    }

    /// Fault detected: Degraded record has no applied_degradation (silent mode change).
    #[test]
    fn degraded_record_has_applied_degradation() {
        // Find a budget just below fp32 but enough for int4.
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        use crate::cost;
        let fp32_peak = cost::weight_bytes(&cfg, "none")
            + cost::kv_cache_bytes(&cfg, 512, "none")
            + cost::activation_bytes(&cfg);
        let int4_peak = cost::weight_bytes(&cfg, "int4_sym")
            + cost::kv_cache_bytes(&cfg, 512, "int4_sym")
            + cost::activation_bytes(&cfg);

        // Budget: between int4 peak and fp32 peak.
        if int4_peak < fp32_peak {
            let budget = int4_peak + (fp32_peak - int4_peak) / 2;
            let p = plan::plan(&cfg, &m, 512, budget, "none", 0.6).unwrap();
            if p.verdict == Verdict::FitsWithDegradation {
                let rec = admit(p);
                assert_eq!(rec.status, AdmitStatus::Degraded);
                assert!(
                    rec.applied_degradation.is_some(),
                    "Degraded record must carry the applied degradation step"
                );
                assert!(
                    rec.message.starts_with("DEGRADED:"),
                    "message must start with DEGRADED:"
                );
            }
        }
    }

    /// Fault detected: DoesNotFit branch emits Degraded instead of Refused.
    #[test]
    fn does_not_fit_verdict_always_produces_refused_status() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        // Force DoesNotFit by using a 1-byte budget.
        let p = plan::plan(&cfg, &m, 512, 1, "none", 0.6).unwrap();
        let rec = admit(p);
        assert_ne!(
            rec.status,
            AdmitStatus::Degraded,
            "DoesNotFit must not produce Degraded status"
        );
        assert_eq!(rec.status, AdmitStatus::Refused);
    }

    /// Fault detected: admitted record has non-empty refusal_reason (pollutes logs).
    #[test]
    fn admitted_record_has_empty_refusal_reason() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);
        assert!(
            rec.refusal_reason.is_empty(),
            "admitted record must have empty refusal_reason, got {:?}",
            rec.refusal_reason
        );
    }
}
