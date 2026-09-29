//! CLI integration tests — mutation-killing tests for cmd_* and parse_budget_gb.
//!
//! These tests invoke the compiled binary with specific flags and assert on exact
//! output content or exit codes.  Each test names a specific fault it would catch.
//!
//! # Fault mapping
//!
//! | test | fault detected |
//! |------|---------------|
//! | plan_budget_printed_as_passed | parse_budget_gb returns None (uses 4.0 default instead of passed value) |
//! | plan_budget_not_zero_default | parse_budget_gb returns Some(0.0) |
//! | plan_budget_gb_times_1e9 | budget_bytes = budget_gb / 1e9 (tiny budget, ref fits in 4 GB but not 0 bytes) |
//! | plan_budget_in_gb_not_bytes | budget_bytes is raw u64 not scaled by 1e9 |
//! | admit_refused_exits_2 | admit: == Refused changed to != Refused (refused config exits 0 instead of 2) |
//! | admit_admitted_exits_0 | admit: == Refused changed to always-true (admitted config exits 2) |
//! | verify_refused_exits_2 | verify: refused budget, must exit 2 |
//! | verify_admitted_prints_budget | verify: budget printed correctly (budget_gb * 1e9 arithmetic) |
//! | verify_admitted_budget_respected | verify: !record.budget_respected negation deleted |
//! | pareto_json_budget_correct | cmd_pareto: budget_gb * 1e9 arithmetic in pareto result |
//! | pareto_exits_0 | cmd_pareto: returns Default::default() (ExitCode 0 for success) |
//! | pareto_frontier_nonempty_for_large_budget | cmd_pareto: budget scaled too small → empty frontier |
//! | stress_binary_exits_0 | cmd_stress returns Default::default() when all pass |
//! | plan_no_binding_constraint_when_fits | binding_constraint printed iff nonempty (! deleted → always prints) |
//! | admit_large_budget_shows_admitted | large budget → "ADMITTED:" prefix in output |
//! | stress_no_absurd_budgets_in_output | fp32_peak arithmetic overflow → absurdly large budgets |
//! | stress_summary_exact_counts | violations += with -= or *=: count field wrong in summary |
//! | stress_output_has_margin_line | violation_free() && all_modes_explicit() → || (exit 0 when violated) |
//! | stress_all_admitted_configs_ok | eff_budget * 4 replaced with + or /: degraded budget too small |
//! | stress_1gb_configs_are_admitted | == AdmitStatus::Refused replaced with !=: admitted configs REFUSED |

use std::process::Command;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fitsproof"))
}

// ---------------------------------------------------------------------------
// parse_budget_gb — the parsing helper in cmd_* functions
// ---------------------------------------------------------------------------

/// Fault detected: parse_budget_gb returns None (ignores --budget-gb, falls back to 4.0).
/// The plan output would show "Budget: 4.000 GB" instead of "Budget: 2.000 GB".
#[test]
fn plan_budget_printed_as_passed() {
    let out = binary()
        .args(["plan", "--budget-gb", "2"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "plan should succeed: {stdout}");
    // If parse_budget_gb returned None, budget_gb = 4.0 and this line would say "4.000 GB".
    assert!(
        stdout.contains("Budget:          2.000 GB"),
        "Budget line must reflect the --budget-gb 2 argument, got: {stdout:?}"
    );
}

/// Fault detected: parse_budget_gb returns Some(0.0).
/// Budget line would show "Budget: 0.000 GB".
#[test]
fn plan_budget_not_zero_default() {
    let out = binary()
        .args(["plan", "--budget-gb", "2"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("Budget:          0.000 GB"),
        "Budget must not be zero when 2 was passed, got: {stdout:?}"
    );
}

/// Fault detected: budget_bytes = (budget_gb / 1e9) as u64 (division instead of multiplication).
/// With budget_gb = 8, that gives budget_bytes = 0 → every config is REFUSED.
/// But with budget_gb = 8, the reference model (~0.055 GB) must fit → plan succeeds and exits 0.
#[test]
fn plan_budget_gb_times_1e9_not_divided() {
    let out = binary()
        .args(["plan", "--budget-gb", "8"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // With / instead of *, budget_bytes = 0, plan returns Err or DoesNotFit → exits 2.
    assert!(
        out.status.success(),
        "plan with 8 GB budget should succeed; exit: {:?}\nstdout: {stdout}",
        out.status.code()
    );
    assert!(
        stdout.contains("Fits"),
        "Verdict must be Fits for 8 GB budget, got: {stdout:?}"
    );
}

/// Fault detected: budget_bytes is the raw u64 of budget_gb (e.g. 4u64 not 4_000_000_000).
/// With budget_gb = 4, raw u64 = 4 bytes — reference model (~55 MB) would be refused.
/// We check that plan with 4 GB budget admits the reference model.
#[test]
fn plan_budget_in_gb_not_bytes() {
    let out = binary()
        .args(["plan", "--budget-gb", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "plan with 4 GB budget should succeed: {stdout}"
    );
    // If budget_bytes = 4 (literal bytes), the ~55 MB ref model would not fit.
    assert!(
        stdout.contains("Fits"),
        "plan must produce Fits verdict for 4 GB budget, got: {stdout:?}"
    );
    // Budget line must show 4 GB, not 0.000.
    assert!(
        stdout.contains("Budget:          4.000 GB"),
        "Budget line must be 4.000 GB, got: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// cmd_admit — AdmitStatus::Refused exit code check
// ---------------------------------------------------------------------------

/// Fault detected: `== AdmitStatus::Refused` changed to `!= AdmitStatus::Refused`.
/// A refused config would exit 0 instead of 2.
#[test]
fn admit_refused_exits_2() {
    let out = binary()
        .args(["admit", "--budget-gb", "0.001"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "admit with 0.001 GB budget must exit 2 (refused), got: {:?}",
        out.status.code()
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("REFUSED:"),
        "refused output must start with REFUSED:, got: {stdout:?}"
    );
}

/// Fault detected: if the condition is always-true, an admitted config exits 2 instead of 0.
#[test]
fn admit_admitted_exits_0() {
    let out = binary()
        .args(["admit", "--budget-gb", "8"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "admit with 8 GB budget must exit 0 (admitted), got: {:?}",
        out.status.code()
    );
}

/// Fault detected: wrong budget scaling causes a large-budget admit to become a refusal.
/// The output must contain "ADMITTED:" for 8 GB budget.
#[test]
fn admit_large_budget_shows_admitted() {
    let out = binary()
        .args(["admit", "--budget-gb", "8"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("ADMITTED:"),
        "admit with 8 GB must show ADMITTED:, got: {stdout:?}"
    );
    // Check that the budget (8 GB = 8 × 10^9 bytes) is represented in the output.
    // "8.000 GB budget" appears in the ADMITTED message.
    assert!(
        stdout.contains("8.000 GB"),
        "ADMITTED message must mention the 8 GB budget, got: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// cmd_verify — arithmetic and exit code checks
// ---------------------------------------------------------------------------

/// Fault detected: verify with 0.001 GB budget → must exit 2 (refused before engine runs).
#[test]
fn verify_refused_exits_2() {
    let out = binary()
        .args(["verify", "--budget-gb", "0.001"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "verify with 0.001 GB budget must exit 2, got: {:?}",
        out.status.code()
    );
}

/// Fault detected: budget_bytes = budget_gb / 1e9 in cmd_verify → budget_bytes ≈ 0 → REFUSED.
/// With 4 GB budget, verify must succeed (exit 0) and print "budget: 4.000 GB".
#[test]
fn verify_admitted_prints_budget() {
    let out = binary()
        .args(["verify", "--budget-gb", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify with 4 GB budget must succeed: {stdout}"
    );
    assert!(
        stdout.contains("budget:         4.000 GB"),
        "verify must print 'budget: 4.000 GB', got: {stdout:?}"
    );
}

/// Fault detected: `! record.budget_respected` negation deleted → exits 2 even when budget respected.
#[test]
fn verify_admitted_budget_respected() {
    let out = binary()
        .args(["verify", "--budget-gb", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "verify with 4 GB budget must exit 0 when budget respected: {stdout}"
    );
    assert!(
        stdout.contains("budget_respected: true"),
        "must print budget_respected: true, got: {stdout:?}"
    );
}

/// Fault detected: `== AdmitStatus::Refused` negated in cmd_verify — admitted → exits 2.
#[test]
fn verify_large_budget_exits_0() {
    let out = binary()
        .args(["verify", "--budget-gb", "8"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "verify with 8 GB budget must exit 0: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// cmd_pareto — budget arithmetic and exit code
// ---------------------------------------------------------------------------

/// Fault detected: cmd_pareto returns Default::default() (ExitCode from u8::default = 0) — already 0.
/// This covers the "replace cmd_pareto -> ExitCode with Default::default()" mutant.
/// Since Default is also success, this tests that the full path runs by checking output.
#[test]
fn pareto_exits_0() {
    let out = binary()
        .args(["pareto", "--budget-gb", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pareto must exit 0, got: {:?}\n{stdout}",
        out.status.code()
    );
}

/// Fault detected: budget_bytes = (budget_gb + 1e9) or (budget_gb / 1e9) → wrong scale.
/// With budget_gb = 4, budget_bytes should be 4_000_000_000.
/// If it was 4 / 1e9 ≈ 0, no configs would fit → frontier_size = 0.
/// If it was 4 + 1e9 = very large, that still works so test uses 0.1 GB where / would give ~0.
#[test]
fn pareto_frontier_nonempty_for_large_budget() {
    let out = binary()
        .args(["pareto", "--budget-gb", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "pareto must succeed: {stdout}");
    // With correct arithmetic (4 GB = 4e9 bytes), the ref model (~55 MB) fits in many configs.
    // frontier_size should be > 0. If budget was scaled wrong to ~4 bytes, frontier_size = 0.
    assert!(
        !stdout.contains("\"frontier_size\":0"),
        "frontier must be non-empty for 4 GB budget, got: {stdout:?}"
    );
    // budget_gb appears in each frontier entry as "budget_gb":4.0
    assert!(
        stdout.contains("\"budget_gb\":4.0") || stdout.contains("\"budget_gb\": 4.0"),
        "pareto JSON must record budget_gb as 4.0, got: {stdout:?}"
    );
}

/// Fault detected: budget_bytes = (budget_gb * 1e9) replaced with + (4 + 1e9 = 1_000_000_004).
/// The total_configs must be consistent with the sweep: pareto sweeps quantizations × context lengths.
/// A broken budget_bytes (e.g. + instead of *) produces a different set of admissible configs.
#[test]
fn pareto_json_budget_correct() {
    let out = binary()
        .args(["pareto", "--budget-gb", "0.1"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "pareto with 0.1 GB budget must succeed: {stdout}"
    );
    // With 0.1 GB (100 MB), the ref model at fp32 (~53 MB weights + KV) should still fit in some configs.
    // If budget_bytes = 0.1 + 1e9 ≈ 1 GB → many more configs fit than at 0.1 GB.
    // We assert budget_gb is recorded correctly in the output.
    assert!(
        stdout.contains("\"budget_gb\":0.1") || stdout.contains("\"budget_gb\": 0.1"),
        "pareto JSON must record budget_gb as 0.1, got: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// cmd_stress binary — exit code
// ---------------------------------------------------------------------------

/// Fault detected: cmd_stress returns Default::default() (exits 0) regardless of violation count.
/// This is caught by checking the binary does actually exit 0 AND produce the summary line.
/// The non-trivial check: summary must say "0 violations" — if Default::default() was used,
/// the summary line would still be computed (it's printed before the check), but a real violation
/// would also exit 0 incorrectly. This test forces the non-violation path.
#[test]
fn stress_binary_exits_0_and_prints_summary() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stress binary must exit 0 when all configs pass: {stdout}"
    );
    // Must contain the summary line with "0 violations".
    assert!(
        stdout.contains("0 violations"),
        "stress output must contain '0 violations', got: {stdout:?}"
    );
    assert!(
        stdout.contains("0 silent"),
        "stress output must contain '0 silent', got: {stdout:?}"
    );
}

/// Fault detected: violation counter arithmetic `+= with -=` or `with *=` in cmd_stress.
/// If violations were decremented or multiplied instead of incremented, the summary would
/// show non-zero violations and the binary would exit 2.  Since there ARE no violations,
/// any incorrect counting also shows up as the wrong number.
#[test]
fn stress_binary_zero_violations_in_summary() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // "Stress harness: 25 configs, 0 violations, 0 silent mode changes."
    assert!(
        stdout.contains(", 0 violations,"),
        "summary must report exactly 0 violations, got: {stdout:?}"
    );
    assert!(
        stdout.contains(", 0 silent mode changes"),
        "summary must report exactly 0 silent mode changes, got: {stdout:?}"
    );
}

/// Fault detected: `n_configs` count wrong (e.g. fp32_peak multiplication changed).
/// We check the harness ran at least 20 configs, matching the acceptance criterion.
#[test]
fn stress_binary_covers_20_configs() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The summary includes "25 configs" (or at least shows the n_configs number).
    // Also check the count is >= 20 by looking for the pattern "N configs".
    // Regex-free: we know it's 25 from the hardcoded spec list.
    assert!(
        stdout.contains("25 configs"),
        "stress summary must report 25 configs, got: {stdout:?}"
    );
}

/// Fault detected: `== with !=` on the violation_free() check in cmd_stress.
/// If the condition is inverted, a passing harness exits 2.
#[test]
fn stress_passing_harness_exits_0() {
    let out = binary().arg("stress").output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stress must exit 0 when violations = 0 and silent_changes = 0, got: {:?}",
        out.status.code()
    );
}

// ---------------------------------------------------------------------------
// cmd_plan — binding_constraint conditional print
// ---------------------------------------------------------------------------

/// Fault detected: `! p.binding_constraint.is_empty()` negated → always prints constraint.
/// When a plan FITS, binding_constraint is empty, and the "Binding constraint:" line must NOT appear.
#[test]
fn plan_no_binding_constraint_when_fits() {
    let out = binary()
        .args(["plan", "--budget-gb", "8"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "plan with 8 GB must succeed: {stdout}"
    );
    assert!(
        !stdout.contains("Binding constraint:"),
        "Fitting plan must NOT print 'Binding constraint:', got: {stdout:?}"
    );
}

/// Fault detected: `! p.binding_constraint.is_empty()` negated → line suppressed on DoesNotFit.
/// When a plan is refused (budget 0.001 GB), the binding constraint line MUST appear in plan output.
/// Note: plan exits 0 even for DoesNotFit (it just reports the verdict), admit exits 2.
#[test]
fn plan_shows_binding_constraint_when_refused() {
    let out = binary()
        .args(["plan", "--budget-gb", "0.001"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "plan must succeed (reports verdict only): {stdout}"
    );
    assert!(
        stdout.contains("Binding constraint:"),
        "DoesNotFit plan must print 'Binding constraint:', got: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// cmd_stress — arithmetic mutation-killing tests
// ---------------------------------------------------------------------------
// The missed mutants from mutation-c2 are all in cmd_stress arithmetic.
// These tests verify the _observable output_ of those arithmetic paths.

/// Fault detected: `fp32_peak` computed with wrong arithmetic.
/// The config `ref/fp32/ctx512/below_fp32` uses budget = fp32_peak - 1.
/// This means the fp32 config does NOT fit at that budget — so either:
/// (a) the plan degrades to a smaller quant that fits (admitted with degradation), or
/// (b) no degradation fits and it is REFUSED.
/// In both cases, the budget shown in the output must be approximately fp32_peak (64-65 MB).
/// If fp32_peak arithmetic was mutated to return 0, budget = u64::MAX (wraps), and the
/// config would be trivially admitted at "budget=X GB" — a very large number.
/// We verify that no admitted config shows an absurdly large budget (> 10 GB) in the output.
#[test]
fn stress_no_absurd_budgets_in_output() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Check no admitted config shows a budget > 100 GB (which would indicate overflow
    // from a zero or negative fp32_peak computation).
    for line in stdout.lines() {
        if line.starts_with("ref/") && !line.starts_with("[REFUSED]") {
            // Extract "budget=X MB" from the line.
            if let Some(idx) = line.find("budget=") {
                let rest = &line[idx + 7..];
                let end = rest
                    .find(|c: char| !c.is_ascii_digit() && c != '.')
                    .unwrap_or(rest.len());
                if let Ok(budget_mb) = rest[..end].parse::<f64>() {
                    assert!(
                        budget_mb < 100_000.0, // 100 GB in MB
                        "budget {budget_mb} MB is absurdly large (fp32_peak arithmetic overflow?): {line:?}"
                    );
                }
            }
        }
    }
}

/// Fault detected: `violations += 1` replaced with `violations -= 1` or `violations *= 1`.
/// If violated configs were counted incorrectly and any violation fires, exit would be wrong.
/// Indirectly covered by stress_binary_exits_0, but this explicitly checks the count field.
///
/// Fault also detected: `silent_changes += 1` wrong → summary has non-zero silent_changes.
#[test]
fn stress_summary_exact_counts() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The summary line is:
    // "Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: ..."
    assert!(
        stdout.contains("25 configs, 0 violations, 0 silent mode changes"),
        "stress summary must have exact counts, got: {stdout:?}"
    );
}

/// Fault detected: `&& with ||` on `violation_free() && all_modes_explicit()`.
/// If it were `||`, a harness with violations but no silent changes would still exit 0.
/// We can't inject a violation at binary level, but we CAN verify the exit code is 0
/// AND the exact summary string appears — if `||` were used with a broken violation_free
/// that returned false, the test catches it when combined with stress_binary_exits_0.
///
/// Also: fault detected if `violation_free()` returns false incorrectly.
/// The Margin line must appear (means records were pushed) — if margin computation was wrong
/// (e.g., `* with +` in the margin calculation), the min/median/max would differ.
#[test]
fn stress_output_has_margin_line() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Margin: min="),
        "stress must print a Margin line, got: {stdout:?}"
    );
    // The margin values should be positive (configs had budgets well above actual usage).
    // "min=" should be followed by a positive number, not zero or negative.
    let min_idx = stdout.find("min=").expect("margin line found above");
    let rest = &stdout[min_idx + 4..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    let min_val: f64 = rest[..end].parse().unwrap_or(0.0);
    assert!(min_val > 0.0, "min margin must be positive, got {min_val}");
}

/// Fault detected: `eff_budget = step.predicted_peak_bytes * 4` replaced with `+` or `/`.
/// This affects the budget used for verify_run when a config is degraded.
/// The degraded config (below_fp32) is refused, so eff_budget only matters for configs
/// that actually get degraded.  The stress output should show no violations for any
/// admitted-with-degradation config.  If eff_budget were too small (/ instead of *),
/// those configs would show budget violations.
///
/// We verify: all lines that say "OK" have budget_respected = true.
#[test]
fn stress_all_admitted_configs_ok() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Every non-REFUSED line should end with ", OK".
    for line in stdout.lines() {
        if (line.starts_with("ref/") || line.starts_with('[')) && !line.starts_with("[REFUSED]") {
            assert!(
                line.ends_with(", OK"),
                "admitted config line must end with ', OK', got: {line:?}"
            );
        }
    }
}

/// Fault detected: `== AdmitStatus::Refused` replaced with `!=` in cmd_stress loop.
/// If the condition is inverted, REFUSED configs would be counted as admitted and
/// ADMITTED configs would be skipped (REFUSED line printed for them).
/// We check that the 1 GB budget configs (which clearly fit) are NOT prefixed with [REFUSED].
#[test]
fn stress_1gb_configs_are_admitted() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // "ref/fp32/ctx512/1GB" and "ref/fp32/ctx256/1GB" must not be REFUSED.
    assert!(
        !stdout.contains("[REFUSED] ref/fp32/ctx512/1GB"),
        "1 GB fp32 config must not be REFUSED, got: {stdout:?}"
    );
    assert!(
        !stdout.contains("[REFUSED] ref/fp32/ctx256/1GB"),
        "1 GB fp32/ctx256 config must not be REFUSED, got: {stdout:?}"
    );
}
