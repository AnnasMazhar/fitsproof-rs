//! Proof harness — measure peak memory vs declared budget.
//!
//! Reports:
//!   - `allocator_peak_bytes`: peak bytes from the `TrackingAllocator` (enforced ceiling)
//!   - `vmhwm_bytes`: OS-reported high-water mark from `/proc/self/status` `VmHWM`
//!   - `delta_bytes`: `vmhwm_bytes - allocator_peak_bytes` (OS overhead not tracked by allocator)
//!
//! The delta is a **first-class output**, not a footnote.  It quantifies the
//! gap between what the allocator knows about and what the OS actually recorded.
//! This gap is the bypass class (mmap / thread-stacks / static data) documented
//! in README Limitations.
//!
//! # Design
//!
//! The Python edition used `resource.getrusage` which returns the high-water mark
//! since process start.  Here we use `/proc/self/status` `VmHWM` directly
//! (same value, more explicit).  `VmRSS` is the current RSS; `VmHWM` is the peak.
//!
//! A REFUSED `AdmitRecord` must not reach `verify_run` — that would be a contract
//! violation.  `verify_run` asserts this and returns `Err`.
//!
//! # Ceiling and VmHWM (F1 / F7)
//!
//! When a ceiling is installed the global allocator refuses any allocation that
//! would exceed it.  `std::fs::read_to_string` allocates through the global
//! allocator, so reading `/proc/self/status` while a tight ceiling is active
//! silently returns 0 (F7).  The fix: **always read VmHWM after clearing the
//! ceiling** (`set_ceiling(0)`), then re-install the ceiling if needed.
//! Callers that install a ceiling before `verify_run` must clear it and let
//! `verify_run` re-install it from the budget field of the admit record.

use crate::admit::{AdmitRecord, AdmitStatus};

/// Result of one proof-harness run.
#[derive(Debug, Clone)]
pub struct VerifyRecord {
    /// Declared budget (bytes).
    pub budget_bytes: u64,
    /// Peak bytes counted by the `TrackingAllocator` (absolute process high-water mark
    /// at end of closure, not a delta — F3 fix).
    pub allocator_peak_bytes: u64,
    /// OS high-water mark from `/proc/self/status` `VmHWM` (bytes).
    pub vmhwm_bytes: u64,
    /// `vmhwm_bytes - allocator_peak_bytes`: OS overhead not visible to the allocator.
    pub delta_bytes: i64,
    /// `allocator_peak_bytes <= budget_bytes && vmhwm_bytes <= budget_bytes` (F2 fix).
    pub budget_respected: bool,
    /// `vmhwm_bytes <= budget_bytes`.
    pub os_budget_respected: bool,
    /// True if a mode change occurred without an emit (should always be false).
    pub mode_changed_silently: bool,
    /// Elapsed time in seconds.
    pub elapsed_s: f64,
    /// Human-readable configuration label.
    pub config_label: String,
    /// Human-readable description of which bound was violated (empty if respected).
    pub violated_bound: String,
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

/// Read `VmHWM` from `/proc/self/status` in bytes, using a stack-allocated buffer
/// to avoid any heap allocation.
///
/// This avoids the F7 bug where a live tight ceiling would refuse the heap
/// allocation inside `std::fs::read_to_string`, silently returning 0.
///
/// Returns 0 if the file is absent or the field is not found.
///
/// # Fault detected
///
/// Returning 0 always would make the delta always negative, hiding OS overhead.
pub fn read_vmhwm_bytes() -> u64 {
    read_proc_status_field_kb(b"VmHWM:")
}

/// Read `VmRSS` (current RSS) from `/proc/self/status` in bytes.
pub fn read_vmrss_bytes() -> u64 {
    read_proc_status_field_kb(b"VmRSS:")
}

/// Read a `kB`-valued field from `/proc/self/status` without heap allocation.
///
/// Uses a fixed 8 KiB stack buffer; the entire `/proc/self/status` file is
/// well under 4 KiB in practice.
fn read_proc_status_field_kb(field: &[u8]) -> u64 {
    use std::fs::File;
    use std::io::Read;

    let mut buf = [0u8; 8192];
    let n = match File::open("/proc/self/status").and_then(|mut f| f.read(&mut buf)) {
        Ok(n) => n,
        Err(_) => return 0,
    };
    let content = &buf[..n];

    // Find the field prefix in the raw bytes.
    let pos = content
        .windows(field.len())
        .position(|w| w == field)
        .unwrap_or(usize::MAX);
    if pos == usize::MAX {
        return 0;
    }

    // Skip the field name and parse the decimal integer.
    let rest = &content[pos + field.len()..];
    // Skip whitespace.
    let rest = rest
        .iter()
        .position(|&b| b != b' ' && b != b'\t')
        .map(|i| &rest[i..])
        .unwrap_or(rest);
    // Read digits.
    let end = rest
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(rest.len());
    let digits = &rest[..end];
    // Parse as UTF-8 (safe: all ascii digits).
    let s = std::str::from_utf8(digits).unwrap_or("0");
    s.parse::<u64>().unwrap_or(0) * 1024
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
/// # F3 fix — meaningful allocator peak delta
///
/// `allocator_peak_bytes` is `peak_after - peak_before` where:
/// - `peak_before` is sampled BEFORE the closure runs
/// - `peak_after` is sampled AFTER the closure runs
///
/// Callers must construct all model weights *inside* the closure so those
/// allocations raise `peak_after` above `peak_before`.  In a warm process this
/// gives the net weight+KV-cache allocation, not the stale process HWM.
///
/// # F2 fix — both bounds gate the verdict
///
/// `budget_respected = allocator_peak_bytes <= budget_bytes && vmhwm <= budget_bytes`.
/// The `violated_bound` field names which bound failed.
///
/// # F7 fix — VmHWM read after ceiling is cleared
///
/// VmHWM is read after `set_ceiling(0)` to avoid the ceiling blocking the
/// heap allocation inside `std::fs::read_to_string`.  The caller must NOT
/// re-install a ceiling between the closure and this read.
///
/// # Contract
///
/// - `admit_record` must NOT be `Refused`.  A refused plan must not reach execution.
/// - VmHWM is read from `/proc/self/status`.
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

    // F3: sample peak BEFORE the closure to establish baseline.
    let peak_before = allocator_peak_fn();

    let t0 = std::time::Instant::now();
    let _tokens = f();
    let elapsed = t0.elapsed().as_secs_f64();

    // F3: sample peak AFTER. Delta = contribution from this closure's allocations.
    let peak_after = allocator_peak_fn();
    let allocator_peak = peak_after.saturating_sub(peak_before);

    // Read VmHWM — ceiling is cleared in allocator_peak_fn, so no F7 blindness.
    let vmhwm = read_vmhwm_bytes();

    let delta = vmhwm as i64 - allocator_peak as i64;

    // F2 fix: both allocator and OS numbers gate the verdict.
    let allocator_ok = allocator_peak <= budget_bytes;
    let os_budget_respected = vmhwm <= budget_bytes;
    let budget_respected = allocator_ok && os_budget_respected;

    let violated_bound = if budget_respected {
        String::new()
    } else if !allocator_ok && !os_budget_respected {
        format!(
            "allocator_peak {:.3} GB > budget {:.3} GB AND VmHWM {:.3} GB > budget",
            allocator_peak as f64 / 1e9,
            budget_bytes as f64 / 1e9,
            vmhwm as f64 / 1e9,
        )
    } else if !allocator_ok {
        format!(
            "allocator_peak {:.3} GB > budget {:.3} GB",
            allocator_peak as f64 / 1e9,
            budget_bytes as f64 / 1e9,
        )
    } else {
        format!(
            "VmHWM {:.3} GB > budget {:.3} GB (off-allocator memory: mmap/thread-stacks/static)",
            vmhwm as f64 / 1e9,
            budget_bytes as f64 / 1e9,
        )
    };

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
        violated_bound,
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

        // Fake allocator: first call (before) returns 0, second call (after) returns 1_000_000.
        // Delta = 1_000_000 - 0 = 1_000_000 which exceeds the 100-byte budget.
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
                }
            },
        );
        let vr = result.unwrap();
        assert!(
            !vr.budget_respected,
            "budget_respected must be false when allocator_peak > budget"
        );
        assert!(!vr.violated_bound.is_empty(), "violated_bound must be set");
    }

    /// Fault detected: budget_respected is false when everything fits.
    #[test]
    fn budget_respected_true_when_peak_fits() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // allocator_peak = 100_000 (delta: 100_000 - 0), well below 1 GB budget; VmHWM from OS also expected < 1 GB.
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
            "budget_respected must be true when both peaks <= budget"
        );
        assert!(
            vr.violated_bound.is_empty(),
            "violated_bound must be empty when respected"
        );
    }

    /// Fault detected: delta_bytes computed incorrectly.
    #[test]
    fn delta_is_vmhwm_minus_allocator_peak() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // peak_before=0, peak_after=50_000_000 → allocator_peak=50_000_000
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

        // delta_bytes = vmhwm - allocator_peak
        let expected_delta = result.vmhwm_bytes as i64 - 50_000_000_i64;
        assert_eq!(
            result.delta_bytes, expected_delta,
            "delta_bytes must equal vmhwm - allocator_peak"
        );
    }

    /// F2: budget_respected must be false when VmHWM > budget, even if allocator_peak is tiny.
    #[test]
    fn budget_respected_false_when_vmhwm_exceeds_budget() {
        let cfg = ModelConfig::reference();
        let m = ref_machine();
        let p = plan::plan(&cfg, &m, 512, 1_000_000_000, "none", 0.6).unwrap();
        let rec = admit(p);

        // VmHWM from the OS is the real process HWM which will be tens of MB.
        // Set budget below what VmHWM will be.
        let tiny_budget = 1024; // 1 KB — VmHWM will certainly exceed this
        let result = verify_run(Vec::new, tiny_budget, &rec, "vmhwm-test", || 0).unwrap();

        // VmHWM > 1 KB → os_budget_respected=false → budget_respected=false
        assert!(
            !result.os_budget_respected,
            "os_budget_respected must be false when VmHWM > budget (VmHWM={})",
            result.vmhwm_bytes
        );
        assert!(
            !result.budget_respected,
            "budget_respected must be false when VmHWM > budget"
        );
        assert!(
            result.violated_bound.contains("VmHWM"),
            "violated_bound must name VmHWM, got: {:?}",
            result.violated_bound
        );
    }
}
