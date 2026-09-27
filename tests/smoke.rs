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
