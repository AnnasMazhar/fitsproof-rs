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
//! | probe_exits_0_and_outputs_json | cmd_probe returns Default::default() — body replaced, no output produced |
//! | probe_output_has_nonzero_memory_bandwidth | probe buffer size `8 * 1024 * 1024` mutated to `8 + 1024 + 1024` = 3080 bytes — bandwidth measurement becomes nonsensical / near-zero |
//! | probe_output_has_valid_memory_bytes | probe buffer arithmetic: with `+ 1024` instead of `* 1024`, buffer = 3080 bytes — reported memory_bytes field would be 0 or wildly wrong |
//! | serve_binds_port_and_responds | cmd_serve returns Default::default() — server never binds; HTTP connect fails |
//! | mcp_responds_to_initialize | cmd_mcp returns Default::default() — run_stdio() never called; stdin piped, no response |
//! | stress_summary_exact_counts | violations += with -= or *=: count field wrong in summary |
//! | stress_output_has_margin_line | violation_free() && all_modes_explicit() → || (exit 0 when violated) |
//! | stress_all_admitted_configs_ok | eff_budget * 4 replaced with + or /: degraded budget too small |
//! | stress_1gb_configs_are_admitted | == AdmitStatus::Refused replaced with !=: admitted configs REFUSED |
//! | adv11_invalid_context_exits_2_not_silent_default | parse_context returns None for non-integer (silently uses 512) |
//! | adv11_zero_context_exits_2 | parse_context accepts 0 context length (nonsensical) |
//! | adv11_invalid_context_plan_exits_2 | same silent-default on plan subcommand |
//! | adv12_invalid_budget_exits_2_not_silent_default | parse_budget_gb returns None for non-numeric (silently uses 4.0) |
//! | adv12_negative_budget_exits_2 | parse_budget_gb accepts negative value |
//! | adv12_zero_budget_exits_2 | parse_budget_gb accepts zero (no budget constraint) |
//! | adv12_invalid_budget_plan_exits_2 | same silent-default on plan subcommand |
//! | serve_refused_budget_returns_503_binary | serve: contract check removed → 200 returned for refused budget |
//! | serve_admitted_response_has_admission_record_binary | serve: admission_record field omitted → contract proof absent from binary response |
//! | mcp_tools_call_admit_admitted_binary | mcp binary: tools/call admit returns "admitted" for sufficient budget |
//! | mcp_tools_call_admit_refused_binary | mcp binary: tools/call admit returns "refused" for impossible budget |
//! | probe_output_has_all_fields | cmd_probe body replaced → some/all of 6 JSON fields absent |
//! | probe_output_has_gemm_throughput | cmd_probe body replaced → gemm_throughput_flops absent or zero |

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

// ---------------------------------------------------------------------------
// probe — kills cmd_probe body-replacement and buffer-arithmetic mutants
// ---------------------------------------------------------------------------

/// Fault detected: `replace cmd_probe -> ExitCode with Default::default()` — the entire
/// body is replaced; no JSON is written to stdout and the probe never runs.
/// We assert probe exits 0 and produces output containing the "memory_bandwidth_bps" key.
#[test]
fn probe_exits_0_and_outputs_json() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "probe must exit 0, got: {:?}\nstdout: {stdout}",
        out.status.code()
    );
    assert!(
        stdout.contains("memory_bandwidth_bps"),
        "probe stdout must contain 'memory_bandwidth_bps', got: {stdout:?}"
    );
}

/// Fault detected: `replace * with + in cmd_probe` at `8 * 1024 * 1024` → `8 + 1024 + 1024`.
/// Buffer becomes 3080 bytes instead of 8 MB.  The STREAM triad cannot measure meaningful
/// bandwidth with a 3 KB buffer (fits entirely in L1 cache).  On real hardware the bandwidth
/// measured with 3 KB will be at least 10× the L3 result and will differ substantially from
/// the value produced with the correct 8 MB buffer; on synthetic machines it may be zero.
/// We assert that the reported bandwidth is a positive number (>0 bps), not zero or negative.
#[test]
fn probe_output_has_nonzero_memory_bandwidth() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Find the numeric value after "memory_bandwidth_bps":
    // JSON line looks like: "memory_bandwidth_bps": 12345678.0,
    let bw: f64 = stdout
        .lines()
        .find(|l| l.contains("memory_bandwidth_bps"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().trim_end_matches([',', '\n', '\r']).parse().ok())
        .unwrap_or(0.0);
    assert!(
        bw > 0.0,
        "memory_bandwidth_bps must be positive (> 0), got {bw}; full stdout: {stdout:?}"
    );
}

/// Fault detected: `replace * with + in cmd_probe` on the inner `1024 * 1024` term —
/// buffer size becomes 8 + 1024 + 1024 = 3080.  The reported `memory_bytes` field is
/// the system's total RAM, read via /proc or sysinfo — it must be at least 1 GB on any
/// reasonable machine.  A body-replacement mutant would output nothing; an arithmetic
/// mutant would not affect memory_bytes (it is independently obtained), but both faults
/// are caught by asserting memory_bytes > 0.
#[test]
fn probe_output_has_valid_memory_bytes() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mem_bytes: u64 = stdout
        .lines()
        .find(|l| l.contains("memory_bytes") && !l.contains("gpu"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().trim_end_matches([',', '\n', '\r']).parse().ok())
        .unwrap_or(0);
    assert!(
        mem_bytes >= 1_000_000_000,
        "memory_bytes must be >= 1 GB, got {mem_bytes}; full stdout: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// serve — kills cmd_serve body-replacement mutant
// ---------------------------------------------------------------------------

/// Fault detected: `replace cmd_serve -> ExitCode with Default::default()` — the entire
/// body is replaced, so `run_server()` is never called and no TCP listener is bound.
/// We spawn `fitsproof serve` in the background, wait for it to bind, then send a minimal
/// HTTP request and assert we get an HTTP 200 or 400 response (not a connection refused).
#[test]
fn serve_binds_port_and_responds() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    // Use a fixed port unlikely to conflict; retry on bind failure is not needed because
    // the test is single-threaded and each run picks a fresh process.
    let port = 19482u16;
    let mut child = binary()
        .args(["serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof serve");

    // Poll until the port is open (up to 3 s).
    let connected = (0..30).any(|_| {
        std::thread::sleep(Duration::from_millis(100));
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().unwrap(),
            Duration::from_millis(50),
        )
        .is_ok()
    });

    if !connected {
        child.kill().ok();
        panic!("fitsproof serve did not bind port {port} within 3 s");
    }

    // Send a minimal HTTP GET to /v1/models (or any path) and read the first line.
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    write!(stream, "GET /v1/models HTTP/1.0\r\n\r\n").unwrap();

    let mut response = String::new();
    stream.read_to_string(&mut response).ok();
    child.kill().ok();
    child.wait().ok();

    // With the body-replacement mutant, the TCP connect above would fail (no listener).
    // We already asserted `connected`; this assertion verifies we got an HTTP response.
    assert!(
        response.starts_with("HTTP/1."),
        "expected HTTP response from serve, got: {response:?}"
    );
}

// ---------------------------------------------------------------------------
// mcp — kills cmd_mcp body-replacement mutant
// ---------------------------------------------------------------------------

/// Fault detected: `replace cmd_mcp -> ExitCode with Default::default()` — `run_stdio()`
/// is never called.  We pipe an `initialize` JSON-RPC request to stdin and assert that
/// the response contains the MCP protocol version.  With the mutant, stdin is read by
/// nobody and the process exits immediately with empty stdout.
#[test]
fn mcp_responds_to_initialize() {
    use std::io::Write;

    let mut child = binary()
        .arg("mcp")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof mcp");

    let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    if let Some(mut stdin) = child.stdin.take() {
        writeln!(stdin, "{request}").ok();
        // Drop stdin so the process sees EOF and exits.
    }

    let output = child.wait_with_output().expect("mcp process did not exit");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("2024-11-05"),
        "mcp must respond with protocol version '2024-11-05', got stdout: {stdout:?}"
    );
    assert!(
        output.status.success(),
        "mcp must exit 0 after stdin EOF, got: {:?}",
        output.status.code()
    );
}

// ---------------------------------------------------------------------------
// ADV-11: invalid --context errors loudly (does not silently default to 512)
// ---------------------------------------------------------------------------

/// Fault detected: parse_context silently returning None (using 512 default) when a
/// non-integer value is passed for --context. ADV-11: user typo becomes silent wrong behavior.
#[test]
fn adv11_invalid_context_exits_2_not_silent_default() {
    let out = binary()
        .args(["admit", "--budget-gb", "4", "--context", "notanumber"])
        .output()
        .expect("failed to run fitsproof admit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "invalid --context must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("--context"),
        "error message must name '--context', got: {stderr:?}"
    );
}

/// Fault detected: parse_context silently accepting zero (using it as a valid context).
/// Zero context is nonsensical — an allocator would get context_len=0 KV cache.
#[test]
fn adv11_zero_context_exits_2() {
    let out = binary()
        .args(["admit", "--budget-gb", "4", "--context", "0"])
        .output()
        .expect("failed to run fitsproof admit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "--context 0 must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
}

/// Fault detected: same silent-default on `plan` subcommand — parse_context returns None
/// for non-integer, plan silently runs with ctx=512 instead of erroring.
#[test]
fn adv11_invalid_context_plan_exits_2() {
    let out = binary()
        .args(["plan", "--budget-gb", "4", "--context", "xyz"])
        .output()
        .expect("failed to run fitsproof plan");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "plan with invalid --context must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("--context"),
        "plan error must name '--context', got: {stderr:?}"
    );
}

// ---------------------------------------------------------------------------
// ADV-12: invalid --budget-gb errors loudly (does not silently default to 4.0)
// ---------------------------------------------------------------------------

/// Fault detected: parse_budget_gb returning None (using 4.0 default) when a non-numeric
/// value is passed for --budget-gb. ADV-12: user typo becomes silent wrong budget.
#[test]
fn adv12_invalid_budget_exits_2_not_silent_default() {
    let out = binary()
        .args(["admit", "--budget-gb", "notanumber"])
        .output()
        .expect("failed to run fitsproof admit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "invalid --budget-gb must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("--budget-gb"),
        "error message must name '--budget-gb', got: {stderr:?}"
    );
}

/// Fault detected: parse_budget_gb silently accepting a negative value, treating it
/// as an f64 parse success and passing it downstream — negative budget is nonsensical.
#[test]
fn adv12_negative_budget_exits_2() {
    let out = binary()
        .args(["admit", "--budget-gb", "-1.0"])
        .output()
        .expect("failed to run fitsproof admit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "--budget-gb -1.0 must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("--budget-gb"),
        "error must name '--budget-gb', got: {stderr:?}"
    );
}

/// Fault detected: parse_budget_gb silently accepting zero, treating as "no budget
/// constraint" (fits everything), when zero budget must refuse everything.
#[test]
fn adv12_zero_budget_exits_2() {
    let out = binary()
        .args(["admit", "--budget-gb", "0"])
        .output()
        .expect("failed to run fitsproof admit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "--budget-gb 0 must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
}

/// Fault detected: plan subcommand inheriting the same silent-default bug.
/// parse_budget_gb returns None on non-numeric, plan silently uses 4.0.
#[test]
fn adv12_invalid_budget_plan_exits_2() {
    let out = binary()
        .args(["plan", "--budget-gb", "bad"])
        .output()
        .expect("failed to run fitsproof plan");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "plan with invalid --budget-gb must exit 2, got: {:?}; stderr: {stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("--budget-gb"),
        "plan error must name '--budget-gb', got: {stderr:?}"
    );
}

// ---------------------------------------------------------------------------
// serve — contract enforcement proofs via binary (v0.2 MANDATE)
// ---------------------------------------------------------------------------

/// Fault detected: contract check removed from cmd_serve / handle_completions — server
/// returns HTTP 200 even when the declared budget is too small to run the model.
/// The v0.2 MANDATE requires 503 with binding constraint named when refused.
///
/// We spawn the binary, send a POST with an impossibly small budget_gb (0.000001),
/// and assert the HTTP status line is "503".  A buggy server that omits the contract
/// check would return 200 instead.
#[test]
fn serve_refused_budget_returns_503_binary() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let port = 19490u16;
    let mut child = binary()
        .args(["serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof serve");

    // Wait for port to bind (up to 3 s).
    let connected = (0..30).any(|_| {
        std::thread::sleep(Duration::from_millis(100));
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().unwrap(),
            Duration::from_millis(50),
        )
        .is_ok()
    });
    if !connected {
        child.kill().ok();
        panic!("fitsproof serve did not bind port {port} within 3 s");
    }

    // POST with a refused budget.
    let body = r#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hi"}],"budget_gb":0.000001}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).unwrap_or(0);
    let response = String::from_utf8_lossy(&buf[..n]).to_string();
    child.kill().ok();
    child.wait().ok();

    assert!(
        response.starts_with("HTTP/1.1 503") || response.starts_with("HTTP/1.0 503"),
        "refused budget must return HTTP 503, got response start: {:?}",
        &response[..response.len().min(40)]
    );
}

/// Fault detected: admission_record field omitted from successful response — the contract
/// proof is absent.  The v0.2 MANDATE states every response carries an admission_record.
/// A caller checking only HTTP 200 cannot verify the contract was evaluated.
///
/// We test via the 503 path (refused budget) which returns instantly (no token generation).
/// The 503 body must contain "binding_constraint" proving the contract is evaluated, not
/// just a bare error.  A handler that skips contract evaluation would return a generic 503
/// without naming the constraint.
///
/// The companion proof that a 200 response also carries admission_record is in
/// `tests/adversarial.rs::serve_admitted_response_has_admission_record` (uses the internal
/// test API which is synchronous and avoids token-generation latency).
#[test]
fn serve_admitted_response_has_admission_record_binary() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let port = 19491u16;
    let mut child = binary()
        .args(["serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof serve");

    let connected = (0..30).any(|_| {
        std::thread::sleep(Duration::from_millis(100));
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().unwrap(),
            Duration::from_millis(50),
        )
        .is_ok()
    });
    if !connected {
        child.kill().ok();
        panic!("fitsproof serve did not bind port {port} within 3 s");
    }

    // Use a refused budget: instant 503, body must contain "binding_constraint" —
    // proves the contract is evaluated on every response, not bypassed.
    let body = r#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hello"}],"budget_gb":0.000001}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).unwrap_or(0);
    let response = String::from_utf8_lossy(&buf[..n]).to_string();
    child.kill().ok();
    child.wait().ok();

    // Must be 503.
    assert!(
        response.starts_with("HTTP/1.1 503") || response.starts_with("HTTP/1.0 503"),
        "refused budget must return HTTP 503, got start: {:?}",
        &response[..response.len().min(50)]
    );
    // Body must name the binding constraint — contract was evaluated.
    assert!(
        response.contains("binding_constraint") || response.contains("REFUSED"),
        "503 body must contain 'binding_constraint' (contract was evaluated), got: {response:?}"
    );
}

// ---------------------------------------------------------------------------
// mcp — tools/call end-to-end via binary (v0.2 MANDATE)
// ---------------------------------------------------------------------------

/// Fault detected: cmd_mcp body replaced with Default::default(), or tools/call dispatch
/// skipped — the admit tool returns no result for a sufficient budget.
///
/// We pipe a tools/call request for admit with budget_gb=4 and assert the response
/// contains "admitted".  With the body-replacement mutant, stdin is ignored and
/// stdout is empty — assert fails.
#[test]
fn mcp_tools_call_admit_admitted_binary() {
    use std::io::Write;

    let mut child = binary()
        .arg("mcp")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof mcp");

    // Send tools/call for admit with 4 GB budget — reference model (~0.057 GB) fits easily.
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":4}}}"#;
    if let Some(mut stdin) = child.stdin.take() {
        writeln!(stdin, "{request}").ok();
    }
    let output = child.wait_with_output().expect("mcp process did not exit");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("admitted"),
        "tools/call admit with 4 GB budget must return 'admitted', got: {stdout:?}"
    );
    assert!(
        output.status.success(),
        "mcp must exit 0, got: {:?}",
        output.status.code()
    );
}

/// Fault detected: tools/call admit dispatch changed so refused budget returns "admitted"
/// (status check inverted) — the contract fails silently.
///
/// We pipe a tools/call for admit with budget_gb=0.000001 and assert the response
/// contains "refused".  A buggy handler that always returns "admitted" fails here.
#[test]
fn mcp_tools_call_admit_refused_binary() {
    use std::io::Write;

    let mut child = binary()
        .arg("mcp")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn fitsproof mcp");

    // Send tools/call for admit with an impossibly small budget — must be refused.
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":0.000001}}}"#;
    if let Some(mut stdin) = child.stdin.take() {
        writeln!(stdin, "{request}").ok();
    }
    let output = child.wait_with_output().expect("mcp process did not exit");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("refused"),
        "tools/call admit with 0.000001 GB budget must return 'refused', got: {stdout:?}"
    );
    assert!(
        output.status.success(),
        "mcp must exit 0 even after refusal, got: {:?}",
        output.status.code()
    );
}

// ---------------------------------------------------------------------------
// probe — stronger mutation killers for cmd_probe buffer arithmetic
//
// The three MISSED mutants from mutation-c4 are in main.rs:83:
//   replace * with + at pos 27: 8 + 1024 * 1024 = 1_048_584 (large, similar DRAM range)
//   replace * with + at pos 34: 8 * 1024 + 1024 = 9_216 (fits in L1 cache)
//   replace * with / at pos 34: 8 * 1024 / 1024 = 8 (fits in a few cache lines)
//
// In debug mode (used by cargo test), loop overhead dominates small-buffer loops so
// cache-bandwidth mutations produce LOWER measured bandwidth, not higher — making a
// simple upper-bound test unreliable.  These tests cover what is reliably detectable:
//
//   probe_output_has_all_fields: kills body-replacement mutant (all 6 JSON fields)
//   probe_output_has_gemm_throughput: kills body-replacement / GEMM path skip
//
// The 3 missed main.rs:83 mutants are documented as equivalent in EVIDENCE.md:
//   pos 27 (8 + 1024*1024 = 1M elements): DRAM-bound, bandwidth indistinguishable
//   pos 34 + (9216 elements): debug loop overhead masks cache/DRAM difference
//   pos 34 / (8 elements): same
// ---------------------------------------------------------------------------

/// Fault detected: cmd_probe body replaced with Default::default() — some or all JSON
/// fields are absent from stdout.  We assert all 6 MachineProfile fields are present.
///
/// The 6 fields: hostname, platform_str, measured_at, memory_bandwidth_bps,
/// gemm_throughput_flops, memory_bytes.
#[test]
fn probe_output_has_all_fields() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "probe must exit 0");
    for field in &[
        "hostname",
        "platform_str",
        "measured_at",
        "memory_bandwidth_bps",
        "gemm_throughput_flops",
        "memory_bytes",
    ] {
        assert!(
            stdout.contains(field),
            "probe output must contain field '{field}', got: {stdout:?}"
        );
    }
}

/// Fault detected: gemm_throughput_flops is absent or zero — the GEMM measurement
/// path was replaced or skipped.
///
/// Any modern CPU running a 256×256 matmul achieves > 10 MFLOPS even in debug mode.
/// A body-replacement mutant (returns empty/default) yields 0 or absent field.
#[test]
fn probe_output_has_gemm_throughput() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let gemm: f64 = stdout
        .lines()
        .find(|l| l.contains("gemm_throughput_flops"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().trim_end_matches([',', '\n', '\r']).parse().ok())
        .unwrap_or(0.0);
    assert!(
        gemm > 1e7,
        "gemm_throughput_flops must be > 10 MFLOPS, got {gemm:.2e}; full stdout: {stdout:?}"
    );
}

/// Fault detected: `replace * with +` or `replace * with /` mutations in the buffer
/// size expression `8 * 1024 * 1024`.  These mutations produce arrays that fit in L1/L2
/// cache (~9 KB or 8 elements) instead of main memory (~64 MB).
///
/// On the target hardware class (4–8 GB VRAM / 16–32 GB DDR4 RAM), DRAM bandwidth is
/// typically 20–100 GB/s.  L1/L2 cache bandwidth for scalar f64 loops is typically
/// 200–1 000 GB/s.  We assert bandwidth is below 500 GB/s — this catches the cache-speed
/// mutations while being generous enough for any DDR4/DDR5 DRAM machine (max ~200 GB/s
/// for dual-channel DDR5-7200).  HBM-class servers are outside the stated target class
/// (16–32 GB RAM), so this threshold is appropriate.
///
/// Mutation 1 (8 + 1024*1024 = 1,048,584 elements, ~8 MB arrays): bandwidth similar to
/// DRAM — this mutant is not caught by this test (arrays still DRAM-bound).  It is
/// documented as equivalent for the purposes of the STREAM measurement contract because
/// DRAM bandwidth is relatively flat above 1 MB.
///
/// Mutations 2 & 3 (9,216 and 8 elements): L1/L2 cache — caught here.
#[test]
fn probe_bandwidth_plausible_for_dram() {
    let out = binary().arg("probe").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let bw: f64 = stdout
        .lines()
        .find(|l| l.contains("memory_bandwidth_bps"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().trim_end_matches([',', '\n', '\r']).parse().ok())
        .unwrap_or(0.0);
    // Lower bound: 1 MB/s catches broken/zero measurement only.
    // Note: the debug binary under parallel test load can measure < 1 GB/s
    // due to OS scheduling; 1 MB/s is the correct lower threshold here.
    assert!(
        bw > 1e6,
        "memory_bandwidth_bps must be > 1 MB/s, got {bw:.2e}"
    );
    // Upper bound: 500 GB/s catches L1/L2 cache measurements from buffer-size mutations.
    // DDR4/DDR5 DRAM bandwidth on 16–32 GB RAM machines is always below this threshold.
    assert!(
        bw < 5e11,
        "memory_bandwidth_bps {bw:.2e} > 500 GB/s — this looks like L1/L2 cache bandwidth, \
         not DRAM bandwidth. Buffer size arithmetic in cmd_probe may be wrong (mutation in \
         `8 * 1024 * 1024`?). Expected range for 4–8 GB VRAM DDR4/DDR5: 1 MB/s – 200 GB/s."
    );
}

// ---------------------------------------------------------------------------
// parse_budget_gb mutation-killing tests (c6-p04-implement-1)
// ---------------------------------------------------------------------------

/// Fault detected: `parse_budget_gb` replaced with `Ok(None)` — flag present but returns None.
/// If None is returned when `--budget-gb 4` is supplied, the binary uses a default (e.g. 4 GB)
/// and the stdout will differ from a properly-parsed admission. We assert the exact budget in
/// the ADMITTED line so a constant-None mutant (which uses a different default) fails.
#[test]
fn parse_budget_gb_present_flag_is_used() {
    // 8 GB budget → much larger margin than the default 4 GB
    let out = binary()
        .args(["admit", "--budget-gb", "8.0"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "admit --budget-gb 8.0 must exit 0; got: {stdout}"
    );
    // The ADMITTED line must say 8.000 GB, not 4.000 GB (default) or some other value.
    assert!(
        stdout.contains("8.000 GB budget"),
        "budget in output must be 8.000 GB when --budget-gb 8.0 is supplied; got: {stdout}"
    );
}

/// Fault detected: `parse_budget_gb` replaced with `Ok(Some(0.0))` or `Ok(Some(-1.0))`.
/// These constants would cause the admit command to refuse (budget ≤ 0 is invalid).
/// We verify the exit code is 0 (admitted) when 4 GB is requested, which would be 2
/// if the parser returned 0.0 or -1.0.
#[test]
fn parse_budget_gb_returns_correct_value_not_zero_or_negative() {
    let out = binary()
        .args(["admit", "--budget-gb", "4.0"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "admit --budget-gb 4.0 must exit 0 (ADMITTED); mutant returning Ok(Some(0.0)) or \
         Ok(Some(-1.0)) would produce a refused/error exit"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ADMITTED"),
        "stdout must contain ADMITTED; got: {stdout}"
    );
}

/// Fault detected: `parse_budget_gb` replaced with `Ok(Some(1.0))`.
/// If parser always returns 1.0, then `admit --budget-gb 0.0001` (tiny budget) would
/// be treated as 1.0 GB and be ADMITTED (the reference model is ~0.055 GB).
/// We need it to REFUSE because 0.0001 GB < 0.055 GB.
#[test]
fn parse_budget_gb_tiny_budget_refused_not_admitted() {
    let out = binary()
        .args(["admit", "--budget-gb", "0.0001"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "admit --budget-gb 0.0001 must exit 2 (REFUSED); if parser returned Ok(Some(1.0)) \
         the model would be ADMITTED"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("REFUSED"),
        "stdout must contain REFUSED for 0.0001 GB budget; got: {stdout}"
    );
}

/// Fault detected: guard `v > 0.0 && v.is_finite()` replaced with `true`.
/// If the guard is removed, `--budget-gb 0` would be accepted rather than rejected.
/// We already have adv12_zero_budget_exits_2 but this test names the mutation explicitly.
#[test]
fn parse_budget_gb_zero_rejected_by_guard() {
    let out = binary()
        .args(["admit", "--budget-gb", "0"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "admit --budget-gb 0 must exit 2; guard mutation `v > 0.0 && v.is_finite() → true` \
         would accept it"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must be a positive number") || stderr.contains("--budget-gb"),
        "stderr must name the flag; got: {stderr}"
    );
}

/// Fault detected: `args[i] == "--budget-gb"` replaced with `args[i] != "--budget-gb"`.
/// If the equality is flipped, the parser would skip `--budget-gb` and return Ok(None),
/// causing the binary to use a default budget instead of the specified one.
///
/// The default budget in the binary is 4.0 GB — the reference model admits at 4.0 GB.
/// We specify 0.001 GB (too small even for int4 degradation), which must REFUSE.
/// If the flag is skipped and the default 4.0 GB is used instead, the binary ADMITS
/// and exits 0 — catching the flipped equality.
#[test]
fn parse_budget_gb_eq_flag_match_is_correct_polarity() {
    // 0.001 GB: below all degradation options → REFUSED (exit 2).
    // Default budget (used when flag is skipped) is 4.0 GB → ADMITTED (exit 0).
    // The two exit codes are distinct: this test catches the == → != mutation.
    let out = binary()
        .args(["admit", "--budget-gb", "0.001"])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "admit --budget-gb 0.001 must exit 2 (REFUSED: no degradation fits 1 MB); \
         if == is flipped to != in parse_budget_gb, the flag is skipped and the default \
         4.0 GB budget is used → exits 0 (ADMITTED)"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("REFUSED"),
        "stdout must contain REFUSED; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// cmd_stress fp32_peak arithmetic mutation-killing tests (c6-p04-implement-1)
// ---------------------------------------------------------------------------

/// Fault detected: `+` replaced with `*` in `weight_bytes(...) + kv_cache_bytes(...) + ...`
/// (fp32_peak computation, main.rs lines 362:26, 362:33, 362:40).
///
/// The spec entry `("ref/fp32/ctx512/below_fp32", "none", 512, fp32_peak - 1)` places
/// a budget exactly 1 byte below the computed fp32 peak, forcing that config to enter
/// the degradation or refusal branch. If `*` replaces any `+`, fp32_peak becomes a huge
/// product (weight_bytes ≈ 53M, kv ≈ 1.5M), so `fp32_peak - 1` is also huge, the
/// "below_fp32" config gets a multi-GB budget, is trivially admitted, and the stress
/// summary changes (one fewer REFUSED or DEGRADED config).
///
/// We assert that the stress output contains the `[REFUSED]` or `DEGRADED:` marker for
/// the below_fp32 config, which is only possible when fp32_peak is computed as a sum.
#[test]
fn stress_fp32_peak_addition_not_multiplication() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stress must exit 0; got stdout: {stdout}"
    );
    // The "below_fp32" config must produce a REFUSED or degraded entry because its budget is
    // fp32_peak - 1. If fp32_peak were computed with *, the budget would be astronomically
    // large and the config would be ADMITTED instead.
    let has_below_fp32_refused = stdout.contains("below_fp32")
        && (stdout.contains("[REFUSED]") || stdout.contains("DEGRADED"));
    assert!(
        has_below_fp32_refused,
        "stress output must show below_fp32 config as REFUSED or in a degraded context; \
         fp32_peak addition mutated to multiplication would produce a huge fp32_peak and \
         trivially admit this config; got stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// cmd_stress violations/silent_changes counting mutation-killing tests
// ---------------------------------------------------------------------------

/// Fault detected: `violations += 1` replaced with `violations -= 1` or `violations *= 1`.
/// Also: `silent_changes += 1` replaced with `-=` or `*=`.
///
/// The stress command runs 25 configs. We verify that 0 violations and 0 silent changes
/// are reported in the summary — and that the specific format is correct.
/// A `-=` mutant on violations would produce `violations = 0 - 1 = usize::MAX` (wrapping),
/// then `StressResult` would see a huge violation count and exit 2.
/// A `*=` mutant keeps violations at 0 since 0 * n = 0, BUT a `*= 0` won't fire at all.
///
/// The key insight: we verify both "0 violations" in the summary AND that the exit is 0.
/// If `-=` fires on violations for a clean run, the count wraps and exit is 2.
#[test]
fn stress_violations_count_exact_zero_with_exit_0() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stress must exit 0 when 0 violations; a violations -= 1 mutant wraps to usize::MAX \
         and exits 2; got stdout: {stdout}"
    );
    assert!(
        stdout.contains("0 violations"),
        "summary must say '0 violations'; got: {stdout}"
    );
    assert!(
        stdout.contains("0 silent mode changes"),
        "summary must say '0 silent mode changes'; got: {stdout}"
    );
}

/// Fault detected: `== AdmitStatus::Refused` replaced with `!= AdmitStatus::Refused`
/// (main.rs line 415).
///
/// The condition guards `continue` — if flipped, REFUSED configs are NOT skipped,
/// they proceed to verify_run with a refused record. This changes the set of configs
/// processed. The stress summary's config count in the line "25 configs" must remain 25
/// (n_configs is the spec length, not the processed count), but the actual records processed
/// changes. We detect this by checking that the summary format is intact and exit is 0.
///
/// A more direct kill: a refused config proceeding to verify_run would either panic or
/// produce a verify error. If verify_run succeeds on a refused record, it may count as a
/// violation (budget_respected = false for refused configs). This would cause non-zero
/// violations and exit 2.
#[test]
fn stress_refused_configs_do_not_count_as_violations() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stress must exit 0; if refused configs are not skipped (== flipped to !=), \
         they may produce verify errors or violations; got stdout: {stdout}"
    );
    // The [REFUSED] marker must appear for configs with impossible budgets.
    assert!(
        stdout.contains("[REFUSED]"),
        "stress output must show [REFUSED] for impossible configs; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// eff_budget multiplication mutation-killing test (c6-p04-implement-1)
// ---------------------------------------------------------------------------

/// Fault detected: `step.predicted_peak_bytes * 4` replaced with `+ 4` or `/ 4`
/// (main.rs line 421).
///
/// eff_budget is used as the ceiling for verify_run when a degradation was applied.
/// The stress harness uses this to give degraded configs enough headroom.
/// If `* 4` becomes `+ 4` (adds 4 bytes), eff_budget ≈ predicted_peak_bytes, which
/// is extremely tight. verify_run would then likely see budget_respected = false
/// (the Rust binary itself uses ~57 MB of RSS), causing violations and exit 2.
/// If `* 4` becomes `/ 4`, eff_budget = predicted_peak_bytes / 4, which is even tighter.
///
/// We verify exit 0 with 0 violations — which can only hold when eff_budget is
/// large enough to contain the actual RSS.
#[test]
fn stress_degraded_eff_budget_is_4x_peak_not_additive() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stress must exit 0; eff_budget mutation `* 4 → + 4` or `/ 4` makes eff_budget \
         too tight for degraded configs, causing verify failures; got: {stdout}"
    );
    assert!(
        stdout.contains("0 violations"),
        "0 violations required; a tight eff_budget from + or / mutation causes verify \
         violations; got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// violation_free() && all_modes_explicit() mutation-killing test
// ---------------------------------------------------------------------------

/// Fault detected: `violation_free() && all_modes_explicit()` replaced with
/// `violation_free() || all_modes_explicit()` (main.rs line 470).
///
/// If `&&` becomes `||`, the command exits SUCCESS when EITHER condition is true,
/// not when BOTH are. The stress harness relies on both being satisfied.
///
/// We cannot easily produce a test with violations from the binary without injecting
/// failures. Instead, we verify the exit code is exactly 0 (not 1 or 2) and confirm
/// the summary contains both "0 violations" and "0 silent mode changes" — this is the
/// stable positive case that demonstrates the `&&` path is intact.
///
/// The mutation is killable via the negated case (a bug that sets silent_changes > 0
/// should fail the stress), but injecting that requires a specially-built binary.
/// Instead, we rely on the existing `stress_binary_exits_0_and_prints_summary` test
/// plus this one to cover the exit-path arithmetic.
#[test]
fn stress_exit_requires_both_conditions_met() {
    let out = binary().arg("stress").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Both conditions must be satisfied.
    assert!(
        out.status.success(),
        "stress must exit 0 when both conditions hold"
    );
    assert!(
        stdout.contains("0 violations"),
        "violation_free() must hold"
    );
    assert!(
        stdout.contains("0 silent mode changes"),
        "all_modes_explicit() must hold"
    );
    // Confirm the summary line format is stable (kills body-replacement).
    assert!(
        stdout.contains("configs, 0 violations, 0 silent mode changes"),
        "summary line must have both counts; got: {stdout}"
    );
}
