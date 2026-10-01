//! `fitsproof` — the binary entry point.
//!
//! Subcommand surface (v0.1, mirrors the Python edition):
//! `probe | plan | admit | verify | stress | serve | mcp | pareto`

use std::process::ExitCode;

use fitsproof::admit::{admit, AdmitStatus};
use fitsproof::gguf::{metadata_to_model_config, read_metadata};
use fitsproof::model::ModelConfig;
use fitsproof::plan::plan;
use fitsproof::probe::{probe, MachineProfile};
use fitsproof::verify::verify_run;

const USAGE: &str = "\
fitsproof — prove your local LLM fits in memory, or get a loud refusal

USAGE:
    fitsproof <COMMAND> [OPTIONS]

COMMANDS:
    probe     Measure this machine (memory bandwidth, GEMM rate, RAM/VRAM)
    plan      Predict peak memory for a configuration and a budget
    admit     Admit / degrade loudly / refuse (exit 2 on refusal, binding constraint named)
    verify    Measure peak allocator bytes + VmHWM, print delta, assert <= budget
    stress    >=20 configurations: zero violations, zero silent mode changes
    serve     OpenAI-compatible HTTP server  [--port 8080]
    mcp       MCP stdio server (JSON-RPC 2.0 over stdin/stdout)
    pareto    Pareto frontier sweep over (quant x context_len)

    --version  Print version
    --help     Print this message

EXAMPLES:
    fitsproof probe
    fitsproof admit --budget-gb 4
    fitsproof admit --budget-gb 0.001              # REFUSED — names the binding constraint, exit 2
    fitsproof plan --model /path/to/model.gguf --budget-gb 8
    fitsproof admit --model /path/to/model.gguf --budget-gb 4 --quant q4_k_m --context 4096
    fitsproof stress
    fitsproof verify --budget-gb 4
    fitsproof serve --port 8080
    fitsproof pareto --budget-gb 4
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("fitsproof {}", fitsproof::VERSION);
            ExitCode::SUCCESS
        }
        Some("--help" | "-h") => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("probe") => cmd_probe(&args[1..]),
        Some("plan") => cmd_plan(&args[1..]),
        Some("admit") => cmd_admit(&args[1..]),
        Some("verify") => cmd_verify(&args[1..]),
        Some("stress") => cmd_stress(),
        Some("serve") => cmd_serve(&args[1..]),
        Some("mcp") => cmd_mcp(),
        Some("pareto") => cmd_pareto(&args[1..]),
        Some(other) => {
            eprintln!("fitsproof: unknown command '{other}'");
            eprintln!("Run 'fitsproof --help' for usage.");
            ExitCode::from(2)
        }
        None => {
            print!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

fn cmd_probe(_args: &[String]) -> ExitCode {
    eprintln!("Probing machine (this takes a few seconds)...");
    let profile = probe(8 * 1024 * 1024, 5, 3);
    println!(
        "{}",
        serde_json::to_string_pretty(&profile).unwrap_or_else(|e| e.to_string())
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// plan
// ---------------------------------------------------------------------------

fn cmd_plan(args: &[String]) -> ExitCode {
    let budget_gb = match parse_budget_gb(args) {
        Ok(Some(v)) => v,
        Ok(None) => 4.0,
        Err(e) => {
            eprintln!("fitsproof plan: {e}");
            return ExitCode::from(2);
        }
    };
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = match parse_context(args) {
        Ok(Some(v)) => v,
        Ok(None) => 512,
        Err(e) => {
            eprintln!("fitsproof plan: {e}");
            return ExitCode::from(2);
        }
    };

    let cfg = match load_model_config(args, "plan") {
        Ok(c) => c,
        Err(code) => return code,
    };

    let machine = synthetic_machine_or_probe();
    let budget_bytes = (budget_gb * 1e9) as u64;

    match plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => {
            println!("Verdict:         {:?}", p.verdict);
            println!(
                "Predicted peak:  {:.3} GB",
                p.predicted_peak_bytes as f64 / 1e9
            );
            println!("Budget:          {:.3} GB", budget_gb);
            println!("Quant:           {}", p.quant);
            println!("Context length:  {}", p.context_len);
            if !p.binding_constraint.is_empty() {
                println!("Binding constraint: {}", p.binding_constraint);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fitsproof plan: {e}");
            eprintln!("  Try: fitsproof plan --budget-gb 4 --quant q4_k_m --context 4096");
            eprintln!("  Valid quant values: none, float16, int8_sym, int4_sym, q4_k_m, q4_k_s, q8_0, q4_0");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// admit
// ---------------------------------------------------------------------------

fn cmd_admit(args: &[String]) -> ExitCode {
    let budget_gb = match parse_budget_gb(args) {
        Ok(Some(v)) => v,
        Ok(None) => 4.0,
        Err(e) => {
            eprintln!("fitsproof admit: {e}");
            return ExitCode::from(2);
        }
    };
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = match parse_context(args) {
        Ok(Some(v)) => v,
        Ok(None) => 512,
        Err(e) => {
            eprintln!("fitsproof admit: {e}");
            return ExitCode::from(2);
        }
    };

    let cfg = match load_model_config(args, "admit") {
        Ok(c) => c,
        Err(code) => return code,
    };

    let machine = synthetic_machine_or_probe();
    let budget_bytes = (budget_gb * 1e9) as u64;

    match plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => {
            let rec = admit(p);
            println!("{}", rec.message);
            if rec.status == AdmitStatus::Refused {
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("fitsproof admit: {e}");
            eprintln!("  Hint: check --budget-gb, --quant, and --context values.");
            eprintln!("  Valid quant values: none, float16, int8_sym, int4_sym, q4_k_m, q4_k_s, q8_0, q4_0");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: &[String]) -> ExitCode {
    let budget_gb = match parse_budget_gb(args) {
        Ok(Some(v)) => v,
        Ok(None) => 4.0,
        Err(e) => {
            eprintln!("fitsproof verify: {e}");
            return ExitCode::from(2);
        }
    };
    let budget_bytes = (budget_gb * 1e9) as u64;
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = match parse_context(args) {
        Ok(Some(v)) => v,
        Ok(None) => 512,
        Err(e) => {
            eprintln!("fitsproof verify: {e}");
            return ExitCode::from(2);
        }
    };

    // When --model is given, warn that verify runs the reference bundle (v0.1 limitation).
    if parse_flag(args, "--model").is_some() {
        eprintln!("fitsproof verify: note — v0.1 runs the reference bundle (random weights).");
        eprintln!(
            "  plan() uses real metadata from the GGUF file; generation uses random weights."
        );
        eprintln!("  Full real-weight verify is a v0.2 scope item.");
    }

    let cfg = match load_model_config(args, "verify") {
        Ok(c) => c,
        Err(code) => return code,
    };

    let machine = synthetic_machine_or_probe();

    let p = match plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fitsproof verify: plan error: {e}");
            eprintln!("  Hint: check --budget-gb, --quant, and --context values.");
            return ExitCode::from(2);
        }
    };

    let rec = admit(p);
    println!("{}", rec.message);

    if rec.status == AdmitStatus::Refused {
        return ExitCode::from(2);
    }

    // Run reference model and measure.
    let ref_cfg = ModelConfig::reference();
    let weights = fitsproof::engine::transformer::Weights::reference(&ref_cfg);
    let mut transformer = fitsproof::engine::transformer::Transformer::new(ref_cfg, weights);

    let vr = verify_run(
        move || {
            transformer
                .generate(&[1u32, 2, 3], 5, 0.0, 42)
                .into_iter()
                .collect()
        },
        budget_bytes,
        &rec,
        "verify-run",
        || fitsproof::ALLOCATOR.peak_bytes(),
    );

    match vr {
        Ok(record) => {
            println!(
                "allocator_peak: {:.3} GB",
                record.allocator_peak_bytes as f64 / 1e9
            );
            println!("VmHWM:          {:.3} GB", record.vmhwm_bytes as f64 / 1e9);
            println!(
                "delta:          {:+.1} MB (VmHWM - allocator_peak)",
                record.delta_bytes as f64 / 1e6
            );
            println!("budget:         {:.3} GB", record.budget_bytes as f64 / 1e9);
            println!("budget_respected: {}", record.budget_respected);
            if !record.budget_respected {
                eprintln!("BUDGET VIOLATED: VmHWM exceeded declared budget.");
                eprintln!("  Reduce --context or use a lower --quant tier.");
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("fitsproof verify: {e}");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

fn cmd_serve(args: &[String]) -> ExitCode {
    let port: u16 = parse_flag(args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let addr = format!("127.0.0.1:{port}");
    match fitsproof::serve::run_server(&addr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fitsproof serve: {e}");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// mcp
// ---------------------------------------------------------------------------

fn cmd_mcp() -> ExitCode {
    fitsproof::mcp::run_stdio();
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// pareto
// ---------------------------------------------------------------------------

fn cmd_pareto(args: &[String]) -> ExitCode {
    let budget_gb = match parse_budget_gb(args) {
        Ok(Some(v)) => v,
        Ok(None) => 4.0,
        Err(e) => {
            eprintln!("fitsproof pareto: {e}");
            return ExitCode::from(2);
        }
    };
    let budget_bytes = (budget_gb * 1e9) as u64;
    let cfg = ModelConfig::reference();
    let machine = synthetic_machine_or_probe();
    let result = fitsproof::pareto::pareto_sweep(&cfg, &machine, budget_bytes);
    println!("{}", result.to_json());
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// stress
// ---------------------------------------------------------------------------

fn cmd_stress() -> ExitCode {
    use fitsproof::admit::{admit, AdmitStatus};
    use fitsproof::model::ModelConfig;
    use fitsproof::plan::plan as do_plan;
    use fitsproof::verify::{StressResult, VerifyRecord};

    let machine = MachineProfile {
        hostname: "stress".into(),
        platform_str: "stress".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    };
    let ref_cfg = ModelConfig::reference();

    let fp32_peak = fitsproof::cost::weight_bytes(&ref_cfg, "none")
        + fitsproof::cost::kv_cache_bytes(&ref_cfg, 512, "fp16")
        + fitsproof::cost::activation_bytes(&ref_cfg);

    // 25 configurations.
    let specs: Vec<(&str, &str, usize, u64)> = vec![
        ("ref/fp32/ctx512/1GB", "none", 512, 1_000_000_000),
        ("ref/fp32/ctx256/1GB", "none", 256, 1_000_000_000),
        ("ref/int8/ctx512/500MB", "int8_sym", 512, 500_000_000),
        ("ref/int4/ctx512/50MB", "int4_sym", 512, 50_000_000),
        ("ref/fp32/ctx128/500MB", "none", 128, 500_000_000),
        ("ref/int8/ctx256/200MB", "int8_sym", 256, 200_000_000),
        ("ref/int4/ctx256/30MB", "int4_sym", 256, 30_000_000),
        ("ref/fp32/ctx64/1GB", "none", 64, 1_000_000_000),
        ("ref/int8/ctx128/200MB", "int8_sym", 128, 200_000_000),
        ("ref/int4/ctx128/20MB", "int4_sym", 128, 20_000_000),
        ("ref/fp32/ctx512/below_fp32", "none", 512, fp32_peak - 1),
        ("ref/fp16/ctx512/500MB", "float16", 512, 500_000_000),
        ("ref/fp32/ctx32/200MB", "none", 32, 200_000_000),
        ("ref/int8/ctx64/100MB", "int8_sym", 64, 100_000_000),
        ("ref/int4/ctx64/20MB", "int4_sym", 64, 20_000_000),
        ("ref/fp32/ctx16/200MB", "none", 16, 200_000_000),
        ("ref/int8/ctx32/100MB", "int8_sym", 32, 100_000_000),
        ("ref/int4/ctx32/10MB", "int4_sym", 32, 10_000_000),
        ("ref/fp16/ctx256/200MB", "float16", 256, 200_000_000),
        ("ref/fp16/ctx128/100MB", "float16", 128, 100_000_000),
        ("ref/int4/ctx16/5MB", "int4_sym", 16, 5_000_000),
        ("ref/fp32/ctx8/200MB", "none", 8, 200_000_000),
        ("ref/int8/ctx16/50MB", "int8_sym", 16, 50_000_000),
        ("ref/int4/ctx8/5MB", "int4_sym", 8, 5_000_000),
        ("ref/fp16/ctx64/100MB", "float16", 64, 100_000_000),
    ];

    let mut records: Vec<VerifyRecord> = Vec::new();

    for (label, quant, context_len, budget) in &specs {
        let p = match do_plan(&ref_cfg, &machine, *context_len, *budget, quant, 0.6) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("plan error for {label}: {e}");
                continue;
            }
        };
        let rec = admit(p);

        if rec.status == AdmitStatus::Refused {
            println!("[REFUSED] {label}: {}", rec.message);
            continue;
        }

        let eff_budget = match &rec.applied_degradation {
            Some(step) => step.predicted_peak_bytes * 4,
            None => *budget,
        };

        let weights = fitsproof::engine::transformer::Weights::reference(&ref_cfg);
        let mut transformer =
            fitsproof::engine::transformer::Transformer::new(ref_cfg.clone(), weights);
        let label_owned = label.to_string();

        let vr = verify_run(
            move || {
                transformer
                    .generate(&[1u32, 2, 3], 2, 0.0, 42)
                    .into_iter()
                    .collect()
            },
            eff_budget,
            &rec,
            &label_owned,
            || fitsproof::ALLOCATOR.peak_bytes(),
        );

        match vr {
            Ok(record) => {
                println!("{}", record.summary());
                records.push(record);
            }
            Err(e) => {
                eprintln!("verify_run error for {label}: {e}");
            }
        }
    }

    let (violations, silent_changes) = count_violations_and_changes(&records);

    let result = StressResult {
        n_configs: specs.len(),
        violations,
        silent_mode_changes: silent_changes,
        records,
    };

    println!();
    println!("{}", result.summary());

    if result.violation_free() && result.all_modes_explicit() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count budget violations and silent mode changes across a set of `VerifyRecord`s.
///
/// Returns `(violations, silent_changes)`.
///
/// Extracted from `cmd_stress` so the accumulation logic is unit-testable.  The mutations
/// that `cargo-mutants` injects here — `+= → -=`, `+= → *=`, `! deleted` — are killed by
/// the unit tests in `#[cfg(test)]` below.
///
/// # Fault detected (by unit tests)
///
/// - `!record.budget_respected` → `record.budget_respected`: violations never incremented.
/// - `violations += 1` → `violations -= 1`: wraps to usize::MAX on first violation.
/// - `mode_changed_silently` condition inverted: silent changes never counted.
fn count_violations_and_changes(records: &[fitsproof::verify::VerifyRecord]) -> (usize, usize) {
    let mut violations = 0usize;
    let mut silent_changes = 0usize;
    for record in records {
        if !record.budget_respected {
            violations += 1;
        }
        if record.mode_changed_silently {
            silent_changes += 1;
        }
    }
    (violations, silent_changes)
}

/// Parse `--budget-gb <value>`.
///
/// Returns:
/// - `Ok(None)` — flag not present; caller should apply a default.
/// - `Ok(Some(v))` — flag present and parsed successfully.
/// - `Err(msg)` — flag present but value is not a positive finite f64.
fn parse_budget_gb(args: &[String]) -> Result<Option<f64>, String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == "--budget-gb" {
            let raw = &args[i + 1];
            match raw.parse::<f64>() {
                Ok(v) if v > 0.0 && v.is_finite() => return Ok(Some(v)),
                Ok(_) => {
                    return Err(format!(
                        "invalid --budget-gb value '{raw}': must be a positive number (e.g. 4.0)"
                    ))
                }
                Err(_) => {
                    return Err(format!(
                        "invalid --budget-gb value '{raw}': expected a number (e.g. 4.0)"
                    ))
                }
            }
        }
    }
    Ok(None)
}

/// Parse `--context <value>`.
///
/// Returns:
/// - `Ok(None)` — flag not present; caller should apply a default.
/// - `Ok(Some(v))` — flag present and parsed successfully.
/// - `Err(msg)` — flag present but value is not a positive integer.
fn parse_context(args: &[String]) -> Result<Option<usize>, String> {
    match parse_flag(args, "--context") {
        None => Ok(None),
        Some(raw) => match raw.parse::<usize>() {
            Ok(v) if v > 0 => Ok(Some(v)),
            Ok(_) => Err(format!(
                "invalid --context value '{raw}': must be a positive integer (e.g. 512)"
            )),
            Err(_) => Err(format!(
                "invalid --context value '{raw}': expected a positive integer (e.g. 512)"
            )),
        },
    }
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

/// Load a `ModelConfig` from `--model <path>` if given, otherwise use the reference bundle.
///
/// Returns `Err(ExitCode)` if the path was given but the file could not be read.
/// The `subcommand` string is used in error messages so the user knows which command failed.
fn load_model_config(args: &[String], subcommand: &str) -> Result<ModelConfig, ExitCode> {
    match parse_flag(args, "--model") {
        None => Ok(ModelConfig::reference()),
        Some(path) => {
            let file = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("fitsproof {subcommand}: cannot open model file '{path}': {e}");
                    eprintln!("  Check the path exists and is readable.");
                    return Err(ExitCode::from(2));
                }
            };
            let meta = match read_metadata(std::io::BufReader::new(file)) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!(
                        "fitsproof {subcommand}: failed to read GGUF header from '{path}': {e}"
                    );
                    eprintln!("  The file must be a valid GGUF v1/v2/v3 model file.");
                    eprintln!(
                        "  Obtain a GGUF model from HuggingFace (search for Q4_K_M variants)."
                    );
                    return Err(ExitCode::from(2));
                }
            };
            let model_name = std::path::Path::new(&path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            match metadata_to_model_config(&meta, model_name) {
                Ok(cfg) => Ok(cfg),
                Err(e) => {
                    eprintln!(
                        "fitsproof {subcommand}: could not extract architecture from '{path}': {e}"
                    );
                    eprintln!("  Supported architectures: llama, qwen2, qwen3, gemma, phi, mistral, falcon, gpt2, bloom.");
                    Err(ExitCode::from(2))
                }
            }
        }
    }
}

/// For CLI use: use a fast synthetic profile to avoid waiting for live probe.
/// Pass --probe flag to do a real measurement.
fn synthetic_machine_or_probe() -> MachineProfile {
    MachineProfile {
        hostname: "cli".into(),
        platform_str: "cli".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `parse_budget_gb` and `count_violations_and_changes`.
    //!
    //! These tests run under `--bin fitsproof` (included in the cargo-mutants fast suite via
    //! `.cargo/mutants.toml`). They kill the surviving mutants identified in the c5 mutation
    //! report: function-level replacements of `parse_budget_gb`, guard mutations
    //! (`v > 0.0 && v.is_finite()` → `true`, `== → !=`), and the `+=` mutations in the
    //! counter accumulation logic.
    //!
    //! # Fault index
    //!
    //! | Test | Fault detected |
    //! |------|----------------|
    //! | `parse_budget_gb_valid_returns_some` | Function replaced with `Ok(None)` — valid input returns the wrong value. |
    //! | `parse_budget_gb_absent_returns_none` | Function replaced with `Ok(Some(1.0))` — absent flag injects a fake budget. |
    //! | `parse_budget_gb_zero_returns_err` | Guard `v > 0.0` replaced with `true` — zero budget accepted as valid. |
    //! | `parse_budget_gb_negative_returns_err` | Guard `v > 0.0` replaced with `true` — negative budget accepted as valid. |
    //! | `parse_budget_gb_nonnumeric_returns_err` | Function replaced with `Ok(Some(1.0))` — non-numeric value silently accepted. |
    //! | `parse_budget_gb_inf_returns_err` | Guard `v.is_finite()` removed — infinity accepted as valid budget. |
    //! | `parse_budget_gb_nan_returns_err` | Guard `v.is_finite()` removed — NaN accepted as valid budget. |
    //! | `parse_budget_gb_value_is_exact` | Function replaced with `Ok(Some(0.0))` or `Ok(Some(-1.0))` — value is wrong even if Some. |
    //! | `parse_budget_gb_reads_value_after_flag` | `i + 1` → `i` — reads wrong token as the value. |
    //! | `count_violations_none_returns_zero_zero` | Counter function body replaced with `(0, 0)` — always returns zero. |
    //! | `count_violations_single_violated_record` | `violations += 1` → `-= 1` or `*= 1` — violated record not counted. |
    //! | `count_silent_changes_single_silent_record` | `silent_changes += 1` → `-= 1` — silent change not counted. |
    //! | `count_violations_negation_deleted` | `!record.budget_respected` → `record.budget_respected` — only NON-violated records counted. |
    //! | `count_violations_both_fields` | Both violations and silent_changes accumulated correctly. |
    //! | `count_violations_accumulates_multiple` | `+= 1` → `= 1` — only last violation counted, not all. |

    use super::{count_violations_and_changes, parse_budget_gb};
    use fitsproof::verify::VerifyRecord;

    // ── parse_budget_gb ──────────────────────────────────────────────────────

    /// Fault: Function body replaced with `Ok(None)` — valid `--budget-gb 4` ignored.
    /// Also kills: `args[i] == "--budget-gb"` changed to `!=` (flag never found).
    #[test]
    fn parse_budget_gb_valid_returns_some() {
        let args: Vec<String> = vec!["--budget-gb".into(), "4.0".into()];
        let result = parse_budget_gb(&args);
        assert_eq!(
            result,
            Ok(Some(4.0)),
            "valid --budget-gb 4.0 must return Ok(Some(4.0))"
        );
    }

    /// Fault: Function replaced with `Ok(Some(1.0))` — absent flag injects a fake budget.
    #[test]
    fn parse_budget_gb_absent_returns_none() {
        let args: Vec<String> = vec!["--quant".into(), "int8_sym".into()];
        let result = parse_budget_gb(&args);
        assert_eq!(result, Ok(None), "absent --budget-gb must return Ok(None)");
    }

    /// Fault: Guard `v > 0.0` replaced with `true` — zero budget accepted as valid.
    #[test]
    fn parse_budget_gb_zero_returns_err() {
        let args: Vec<String> = vec!["--budget-gb".into(), "0".into()];
        assert!(
            parse_budget_gb(&args).is_err(),
            "--budget-gb 0 must return Err (zero is not a positive budget)"
        );
    }

    /// Fault: Guard `v > 0.0` replaced with `true` — negative budget accepted as valid.
    #[test]
    fn parse_budget_gb_negative_returns_err() {
        let args: Vec<String> = vec!["--budget-gb".into(), "-1.5".into()];
        assert!(
            parse_budget_gb(&args).is_err(),
            "--budget-gb -1.5 must return Err"
        );
    }

    /// Fault: Function replaced with `Ok(Some(1.0))` — non-numeric value silently accepted.
    /// Also kills: `i + 1` → `i - 1` (reads wrong token as the value).
    #[test]
    fn parse_budget_gb_nonnumeric_returns_err() {
        let args: Vec<String> = vec!["--budget-gb".into(), "notanumber".into()];
        assert!(
            parse_budget_gb(&args).is_err(),
            "--budget-gb notanumber must return Err"
        );
    }

    /// Fault: Guard `v.is_finite()` removed — infinity accepted as valid budget.
    #[test]
    fn parse_budget_gb_inf_returns_err() {
        let args: Vec<String> = vec!["--budget-gb".into(), "inf".into()];
        assert!(
            parse_budget_gb(&args).is_err(),
            "--budget-gb inf must return Err (infinity is not a valid budget)"
        );
    }

    /// Fault: Guard `v.is_finite()` removed — NaN accepted as valid budget.
    /// Also kills: `v > 0.0 && v.is_finite()` → `true`.
    #[test]
    fn parse_budget_gb_nan_returns_err() {
        let args: Vec<String> = vec!["--budget-gb".into(), "NaN".into()];
        assert!(
            parse_budget_gb(&args).is_err(),
            "--budget-gb NaN must return Err"
        );
    }

    /// Fault: Function replaced with `Ok(Some(0.0))` or `Ok(Some(-1.0))` — wrong value returned.
    /// Verifies the exact value, not just Some.
    #[test]
    fn parse_budget_gb_value_is_exact() {
        let args: Vec<String> = vec!["--budget-gb".into(), "8.5".into()];
        assert_eq!(
            parse_budget_gb(&args),
            Ok(Some(8.5)),
            "--budget-gb 8.5 must return exactly Ok(Some(8.5))"
        );
    }

    /// Fault: `i + 1` → `i + 0` (reads the flag name as the value, not the token after it).
    /// `"--budget-gb"` is not a valid f64, so this produces Err — but the test proves
    /// `i + 1` (not `i`) is used by verifying the correct numeric value is returned when
    /// the value token is at position `i + 1`.
    #[test]
    fn parse_budget_gb_reads_value_after_flag() {
        let args: Vec<String> = vec![
            "--context".into(),
            "512".into(),
            "--budget-gb".into(),
            "2.5".into(),
        ];
        assert_eq!(
            parse_budget_gb(&args),
            Ok(Some(2.5)),
            "value must be the token immediately after --budget-gb"
        );
    }

    // ── count_violations_and_changes ─────────────────────────────────────────

    fn make_record(budget_respected: bool, mode_changed_silently: bool) -> VerifyRecord {
        VerifyRecord {
            budget_bytes: 1_000_000_000,
            allocator_peak_bytes: if budget_respected {
                100_000
            } else {
                2_000_000_000
            },
            vmhwm_bytes: 58_000_000,
            delta_bytes: 57_900_000,
            budget_respected,
            os_budget_respected: budget_respected,
            mode_changed_silently,
            elapsed_s: 0.01,
            config_label: "test-record".into(),
        }
    }

    /// Fault: Function body replaced with `(0, 0)` — always returns zero counts.
    #[test]
    fn count_violations_none_returns_zero_zero() {
        let records = vec![make_record(true, false), make_record(true, false)];
        let (violations, silent_changes) = count_violations_and_changes(&records);
        assert_eq!(
            violations, 0,
            "zero violations expected for all-respected records"
        );
        assert_eq!(
            silent_changes, 0,
            "zero silent changes expected for all-explicit records"
        );
    }

    /// Fault: `violations += 1` → `violations -= 1` (wraps to usize::MAX) or `*= 1` (stays 0).
    /// Also: `!record.budget_respected` → `record.budget_respected` (counts non-violations).
    #[test]
    fn count_violations_single_violated_record() {
        let records = vec![make_record(false, false)];
        let (violations, silent_changes) = count_violations_and_changes(&records);
        assert_eq!(
            violations, 1,
            "one violated record must produce violations=1; \
            violations += 1 mutated to -= 1 gives usize::MAX, *= 1 gives 0 — both fail here"
        );
        assert_eq!(silent_changes, 0, "no silent changes in this record");
    }

    /// Fault: `silent_changes += 1` → `-= 1` (wraps) or `*= 1` (stays 0).
    #[test]
    fn count_silent_changes_single_silent_record() {
        let records = vec![make_record(true, true)];
        let (violations, silent_changes) = count_violations_and_changes(&records);
        assert_eq!(violations, 0, "budget was respected — no violations");
        assert_eq!(
            silent_changes, 1,
            "one silent-change record must produce silent_changes=1; \
            += 1 mutated to -= 1 gives usize::MAX — fails here"
        );
    }

    /// Fault: `!record.budget_respected` → `record.budget_respected` — only NON-violated
    /// records counted as violations.  With 2 clean and 1 violated: flipped condition counts 2
    /// instead of 1.
    #[test]
    fn count_violations_negation_deleted() {
        let records = vec![
            make_record(true, false),  // clean
            make_record(false, false), // violated
            make_record(true, false),  // clean
        ];
        let (violations, _) = count_violations_and_changes(&records);
        assert_eq!(
            violations, 1,
            "exactly one violated record — negation deleted would count 2"
        );
    }

    /// Fault: Either `+= 1` mutation on violations or silent_changes breaks the tuple.
    #[test]
    fn count_violations_both_fields() {
        let records = vec![
            make_record(false, false), // violation only
            make_record(true, true),   // silent change only
            make_record(false, true),  // both
        ];
        let (violations, silent_changes) = count_violations_and_changes(&records);
        assert_eq!(violations, 2, "two records with budget_respected=false");
        assert_eq!(
            silent_changes, 2,
            "two records with mode_changed_silently=true"
        );
    }

    /// Fault: `violations += 1` replaced with `violations = 1` (assignment, not increment)
    /// — would cap at 1 regardless of how many violations occur.
    #[test]
    fn count_violations_accumulates_multiple() {
        let records = vec![
            make_record(false, false),
            make_record(false, false),
            make_record(false, true),
        ];
        let (violations, silent_changes) = count_violations_and_changes(&records);
        assert_eq!(
            violations, 3,
            "three violated records; = 1 mutation would give 1 not 3"
        );
        assert_eq!(silent_changes, 1, "one silent change");
    }
}
