//! Smoke test: the binary exists, runs, and reports its version.
//!
//! This is the seed baseline only. Real contract tests (allocator ceiling, plan/admit/verify,
//! stress harness) belong in `tests/` and `src/**` per specs/fitsproof-rs.md.

use std::process::Command;

#[test]
fn binary_reports_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_fitsproof"))
        .arg("--version")
        .output()
        .expect("binary should run");
    assert!(out.status.success(), "exit status: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("fitsproof "),
        "unexpected version output: {stdout:?}"
    );
}

#[test]
fn unknown_subcommand_exits_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_fitsproof"))
        .arg("definitely-not-a-command")
        .output()
        .expect("binary should run");
    assert_eq!(out.status.code(), Some(2));
}

/// F1: over-budget verify produces a loud refusal (exit 134 SIGABRT or exit 2), not exit 0.
///
/// The reference model weights are ~57 MB fp32. A budget of 0.0567 GB (~56.7 MB) is
/// just at/below the actual peak, so either:
///   (a) the allocator ceiling trips during weight construction → SIGABRT (exit 134), or
///   (b) the VmHWM gate catches it after the run → exit 2.
///
/// In both cases the outcome is a loud, observable failure rather than silent pass (exit 0).
/// The subprocess test confirms the enforcement is live (not bypassed as in the pre-fix code
/// where no ceiling was installed and exit 0 was guaranteed).
///
/// This test runs the binary in a subprocess so that any SIGABRT terminates only the child.
#[test]
fn over_budget_verify_refuses_loudly() {
    // Budget ≈ predicted peak (56.7 MB); actual peak including OS overhead exceeds this.
    let out = Command::new(env!("CARGO_BIN_EXE_fitsproof"))
        .args(["verify", "--budget-gb", "0.0567"])
        .output()
        .expect("binary should run");

    let code = out.status.code();
    let signal = {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            out.status.signal()
        }
        #[cfg(not(unix))]
        {
            None::<i32>
        }
    };

    // Must NOT exit 0 (that would mean the ceiling is bypassed / silent over-budget).
    // Accept: exit 2 (VmHWM gate), exit 134 (SIGABRT from allocator ceiling), or signal 6.
    let is_loud_refusal = code == Some(2) || code == Some(134) || signal == Some(6);
    assert!(
        is_loud_refusal,
        "over-budget verify must refuse loudly (exit 2 or SIGABRT exit 134), \
         got code={code:?} signal={signal:?}\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // Specifically must NOT be exit 0 (silent pass).
    assert_ne!(
        code,
        Some(0),
        "over-budget verify must not silently pass (exit 0); got code={code:?}"
    );
}
