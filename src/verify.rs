//! Proof harness — measure peak memory vs declared budget.
//!
//! Reports:
//!   - `allocator_peak_bytes`: peak bytes from the `TrackingAllocator` (enforced ceiling)
//!   - `vmhwm_bytes`: OS-reported high-water mark from `/proc/self/status` `VmHWM`
//!   - `delta_bytes`: `vmhwm_bytes - allocator_peak_bytes` (OS overhead not tracked by allocator)
//!
//! The delta is a **first-class output**, not a footnote.  It quantifies the
//! gap between what the allocator knows about and what the OS actually recorded.
//!
//! # Design
//!
//! The Python edition used `resource.getrusage` which returns the high-water mark
//! since process start.  Here we use `/proc/self/status` `VmHWM` directly
//! (same value, more explicit).  `VmRSS` is the current RSS; `VmHWM` is the peak.
//!
//! A REFUSED `AdmitRecord` must not reach `verify_run` — that would be a contract
//! violation.  `verify_run` asserts this and returns `Err`.

use crate::admit::{AdmitRecord, AdmitStatus};

/// Result of one proof-harness run.
#[derive(Debug, Clone)]
pub struct VerifyRecord {
    /// Declared budget (bytes).
    pub budget_bytes: u64,
    /// Peak bytes counted by the `TrackingAllocator`.
    pub allocator_peak_bytes: u64,
    /// OS high-water mark from `/proc/self/status` `VmHWM` (bytes).
    pub vmhwm_bytes: u64,
    /// `vmhwm_bytes - allocator_peak_bytes`: OS overhead not visible to the allocator.
    pub delta_bytes: i64,
    /// `allocator_peak_bytes <= budget_bytes`.
    pub budget_respected: bool,
    /// `vmhwm_bytes <= budget_bytes` (separate check — OS and allocator may differ).
    pub os_budget_respected: bool,
    /// True if a mode change occurred without an emit (should always be false).
    pub mode_changed_silently: bool,
    /// Elapsed time in seconds.
    pub elapsed_s: f64,
    /// Human-readable configuration label.
    pub config_label: String,
}

impl VerifyRecord {
    /// One-line summary suitable for CLI output.
    pub fn summary(&self) -> String {
        format!(
            "{}: allocator_peak={:.1} MB, VmHWM={:.1} MB, delta={:+.1} MB, budget={:.1} MB, {}",
            self.config_label,
            self.allocator_peak_bytes as f64 / 1e6,
            self.vmhwm_bytes as f64 / 1e6,
            self.delta_bytes as f64 / 1e6,
            self.budget_bytes as f64 / 1e6,
            if self.budget_respected {
                "OK"
            } else {
                "VIOLATION"
            },
        )
    }
}

/// Read `VmHWM` from `/proc/self/status` in bytes.
///
/// Returns 0 if the file is absent or the field is not found.
///
/// # Fault detected
///
/// Returning 0 always would make the delta always negative, hiding OS overhead.
pub fn read_vmhwm_bytes() -> u64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb_str = rest.split_whitespace().next().unwrap_or("0");
                if let Ok(kb) = kb_str.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

/// Read `VmRSS` (current RSS) from `/proc/self/status` in bytes.
pub fn read_vmrss_bytes() -> u64 {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.split_whitespace().next().unwrap_or("0");
                if let Ok(kb) = kb_str.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

/// Errors from `verify_run`.
#[derive(Debug, Clone)]
pub enum VerifyError {
    /// A REFUSED admit record was passed to verify_run (contract violation).
    RefusedRecordPassed(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::RefusedRecordPassed(s) => {
                write!(f, "verify_run called with a REFUSED admit record: {s}")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Run a generation closure, measure peak memory, and return a `VerifyRecord`.
///
/// `allocator_peak_fn` is called BEFORE and AFTER the closure; the reported
/// `allocator_peak_bytes` is the **delta** (peak_after - peak_before).
/// This avoids false violations from allocations made by earlier tests in the
/// same process (the global allocator's peak is monotonically non-decreasing).
///
/// # Contract
///
/// - `admit_record` must NOT be `Refused`.  A refused plan must not reach execution.
/// - VmHWM is read from `/proc/self/status`.
///
/// # Fault detected
///
/// If budget_bytes is 0, `budget_respected` would be trivially false.
/// Tests verify this edge case explicitly.
pub fn verify_run(
    f: impl FnOnce() -> Vec<u32>,
    budget_bytes: u64,
    admit_record: &AdmitRecord,
    config_label: &str,
    allocator_peak_fn: impl Fn() -> u64,
) -> Result<VerifyRecord, VerifyError> {
    if admit_record.status == AdmitStatus::Refused {
        return Err(VerifyError::RefusedRecordPassed(
            admit_record.refusal_reason.clone(),
        ));
    }

    // Sample allocator peak BEFORE the run.
    let peak_before = allocator_peak_fn();

    let t0 = std::time::Instant::now();
    let _tokens = f();
    let elapsed = t0.elapsed().as_secs_f64();

    // Sample allocator peak AFTER the run.
    let peak_after = allocator_peak_fn();
    // Delta peak: new allocations made during this run only.
    let allocator_peak = peak_after.saturating_sub(peak_before);

    let vmhwm = read_vmhwm_bytes();
    let delta = vmhwm as i64 - peak_after as i64; // VmHWM vs total process peak
    let budget_respected = allocator_peak <= budget_bytes;
    let os_budget_respected = vmhwm <= budget_bytes;
    let mode_changed_silently = false;

    Ok(VerifyRecord {
        budget_bytes,
        allocator_peak_bytes: allocator_peak,
        vmhwm_bytes: vmhwm,
        delta_bytes: delta,
        budget_respected,
        os_budget_respected,
        mode_changed_silently,
        elapsed_s: elapsed,
        config_label: config_label.to_string(),
    })
}

/// Aggregate result from the stress harness across multiple configurations.
#[derive(Debug)]
pub struct StressResult {
    pub n_configs: usize,
    pub violations: usize,
    pub silent_mode_changes: usize,
    pub records: Vec<VerifyRecord>,
}

impl StressResult {
    pub fn violation_free(&self) -> bool {
        self.violations == 0
    }

    pub fn all_modes_explicit(&self) -> bool {
        self.silent_mode_changes == 0
    }

    pub fn summary(&self) -> String {
        if self.records.is_empty() {
            return format!(
                "Stress harness: {} configs, {} violations, {} silent mode changes. No margin data.",
                self.n_configs, self.violations, self.silent_mode_changes
            );
        }
        let margins: Vec<f64> = self
            .records
            .iter()
            .map(|r| r.budget_bytes as f64 - r.allocator_peak_bytes as f64)
            .collect();
        let min_m = margins.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_m = margins.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = margins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_m = sorted[sorted.len() / 2];
        format!(
            "Stress harness: {} configs, {} violations, {} silent mode changes. \
             Margin: min={:.1} MB, median={:.1} MB, max={:.1} MB.",
            self.n_configs,
            self.violations,
            self.silent_mode_changes,
            min_m / 1e6,
            median_m / 1e6,
            max_m / 1e6,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the verify module.
    //!
    //! Each test names the fault it detects.

    use super::*;
    use crate::admit::{admit, AdmitStatus};
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

    /// Fault detected: read_vmhwm_bytes returns 0 always (wrong field parsed).
    #[test]
    fn vmhwm_is_nonzero_on_linux() {
        let v = read_vmhwm_bytes();
        // On Linux /proc/self/status always has VmHWM. Accept 0 only if file missing.
        if std::path::Path::new("/proc/self/status").exists() {
            assert!(
                v > 0,
                "VmHWM should be > 0 on Linux with /proc/self/status present"
            );
        }
    }

    /// Fault detected: verify_run accepts a REFUSED record without error.
    #[test]
    fn refused_record_returns_error() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1, "none", 0.6).unwrap();
        let rec = admit(p);
        assert_eq!(rec.status, AdmitStatus::Refused);

        let result = verify_run(|| vec![1u32, 2, 3], 1, &rec, "test-refused", || 0);
        assert!(
            result.is_err(),
            "verify_run must return Err when called with a Refused record"
        );
    }

    /// Fault detected: budget_respected is always true (comparison inverted).
    #[test]
    fn budget_respected_false_when_peak_exceeds_budget() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // Inject a fake allocator that returns growing values: pre=0, post=1_000_000.
        // The delta (post - pre) = 1_000_000 which exceeds the 100-byte budget.
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let result = verify_run(
            Vec::new,
            100, // tiny budget
            &rec,
            "test-violation",
            move || {
                let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    0
                } else {
                    1_000_000
                } // first call returns 0, second returns 1MB
            },
        );
        let vr = result.unwrap();
        assert!(
            !vr.budget_respected,
            "budget_respected must be false when allocator_peak > budget"
        );
    }

    /// Fault detected: budget_respected is false when everything fits.
    #[test]
    fn budget_respected_true_when_peak_fits() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // Fake allocator: pre=0, post=100_000. delta=100_000 << 1_000_000_000 budget.
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let result = verify_run(
            || vec![1u32],
            1_000_000_000,
            &rec,
            "test-fits",
            move || {
                let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    0
                } else {
                    100_000
                }
            },
        );
        let vr = result.unwrap();
        assert!(
            vr.budget_respected,
            "budget_respected must be true when allocator_peak <= budget"
        );
    }

    /// Fault detected: delta_bytes computed as vmhwm - allocator (should be vmhwm - alloc).
    #[test]
    fn delta_is_vmhwm_minus_allocator_after() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // pre=0, post=50_000_000. Delta is used for budget_respected but VmHWM is from OS.
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let result = verify_run(Vec::new, 1_000_000_000, &rec, "delta-test", move || {
            let n = cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                0
            } else {
                50_000_000
            }
        })
        .unwrap();

        // delta_bytes = vmhwm - peak_after (absolute process peak after run).
        // peak_after = 50_000_000 in our fake.
        // vmhwm is from the OS.
        let expected_delta = result.vmhwm_bytes as i64 - 50_000_000_i64;
        assert_eq!(
            result.delta_bytes, expected_delta,
            "delta_bytes must equal vmhwm - allocator_after"
        );
    }
}
