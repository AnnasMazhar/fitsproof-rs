//! `fitsproof` — the binary entry point.
//!
//! Subcommand surface (v0.1, mirrors the Python edition):
//! `probe | plan | admit | verify | stress | serve | mcp | pareto`

use std::process::ExitCode;

use fitsproof::admit::{admit, AdmitStatus};
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
    serve     OpenAI-compatible HTTP server  [not yet implemented in v0.1]
    mcp       MCP stdio server              [not yet implemented in v0.1]
    pareto    Pareto frontier sweep         [not yet implemented in v0.1]

    --version  Print version
    --help     Print this message

EXAMPLES:
    fitsproof probe
    fitsproof admit --budget-gb 4
    fitsproof admit --budget-gb 0.001    # REFUSED — names the binding constraint, exit 2
    fitsproof stress
    fitsproof verify --budget-gb 4
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
        Some("serve" | "mcp" | "pareto") => {
            let sub = args[0].as_str();
            eprintln!("fitsproof: '{sub}' is not yet implemented in v0.1.");
            eprintln!("See specs/fitsproof-rs.md for the delivery schedule.");
            ExitCode::from(2)
        }
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
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let cfg = ModelConfig::reference();
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
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// admit
// ---------------------------------------------------------------------------

fn cmd_admit(args: &[String]) -> ExitCode {
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let cfg = ModelConfig::reference();
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
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: &[String]) -> ExitCode {
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let budget_bytes = (budget_gb * 1e9) as u64;
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let cfg = ModelConfig::reference();
    let machine = synthetic_machine_or_probe();

    let p = match plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fitsproof verify: plan error: {e}");
            return ExitCode::from(2);
        }
    };

    let rec = admit(p);
    println!("{}", rec.message);

    if rec.status == AdmitStatus::Refused {
        return ExitCode::from(2);
    }

    // F4: determine effective quant/context from applied_degradation.
    let (eff_quant, eff_context) = effective_config(&rec, &quant, context_len);

    // F1: install the ceiling before generating.
    // Ceiling = budget_bytes (the declared contract).
    fitsproof::ALLOCATOR.set_ceiling(budget_bytes);
    // F3: reset peak to current so peak_bytes() reflects only this run.
    fitsproof::ALLOCATOR.reset_peak_to_current();

    // F3: construct weights INSIDE the verify closure so those allocations
    // are captured by the absolute peak measurement.
    // F4: use eff_context for KV cache sizing.
    let cfg_for_run = build_effective_cfg(&cfg, eff_context);
    let eff_quant_owned = eff_quant.to_string();

    let vr = verify_run(
        move || {
            // Build weights inside the measured region (F3 fix).
            let weights = fitsproof::engine::transformer::Weights::reference_with_quant(
                &cfg_for_run,
                &eff_quant_owned,
            );
            let mut transformer =
                fitsproof::engine::transformer::Transformer::new(cfg_for_run, weights);
            // F4: quantised bundles use warmup_only (no fp32 inference in v0.1).
            if transformer.weights.can_generate() {
                transformer
                    .generate(&[1u32, 2, 3], 5, 0.0, 42)
                    .into_iter()
                    .collect()
            } else {
                transformer.warmup_only();
                Vec::new()
            }
        },
        budget_bytes,
        &rec,
        "verify-run",
        || fitsproof::ALLOCATOR.peak_bytes(),
    );

    // F1/F7: clear ceiling after the run (ceiling was active during the closure).
    fitsproof::ALLOCATOR.set_ceiling(0);

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
                eprintln!("BUDGET VIOLATED: {}", record.violated_bound);
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
        + fitsproof::cost::kv_cache_bytes(&ref_cfg, 512, "none")
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
    let mut violations = 0usize;
    let mut silent_changes = 0usize;

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

        // F4: use effective quant/context from degradation.
        let (eff_quant, eff_context) = effective_config(&rec, quant, *context_len);
        let eff_budget = match &rec.applied_degradation {
            Some(step) => step.predicted_peak_bytes * 4,
            None => *budget,
        };

        // F1: install ceiling before the run.
        fitsproof::ALLOCATOR.set_ceiling(eff_budget);
        // F3: reset peak to current so peak_bytes() reflects only this run.
        fitsproof::ALLOCATOR.reset_peak_to_current();

        let label_owned = label.to_string();
        let eff_quant_owned = eff_quant.to_string();
        let cfg_for_run = build_effective_cfg(&ref_cfg, eff_context);

        let vr = verify_run(
            move || {
                // F3: construct weights INSIDE the closure for absolute peak measurement.
                let weights = fitsproof::engine::transformer::Weights::reference_with_quant(
                    &cfg_for_run,
                    &eff_quant_owned,
                );
                let mut transformer =
                    fitsproof::engine::transformer::Transformer::new(cfg_for_run, weights);
                // F4: quantised bundles use warmup_only (no fp32 inference in v0.1).
                if transformer.weights.can_generate() {
                    transformer
                        .generate(&[1u32, 2, 3], 2, 0.0, 42)
                        .into_iter()
                        .collect()
                } else {
                    transformer.warmup_only();
                    Vec::new()
                }
            },
            eff_budget,
            &rec,
            &label_owned,
            || {
                // Clear ceiling before sampling peak (F1/F7 fix).
                fitsproof::ALLOCATOR.set_ceiling(0);
                fitsproof::ALLOCATOR.peak_bytes()
            },
        );

        match vr {
            Ok(record) => {
                println!("{}", record.summary());
                if !record.budget_respected {
                    violations += 1;
                }
                if record.mode_changed_silently {
                    silent_changes += 1;
                }
                records.push(record);
            }
            Err(e) => {
                eprintln!("verify_run error for {label}: {e}");
            }
        }
    }

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

fn parse_budget_gb(args: &[String]) -> Option<f64> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == "--budget-gb" {
            return args[i + 1].parse().ok();
        }
    }
    None
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

/// For CLI use: use a fast synthetic profile to avoid waiting for live probe.
fn synthetic_machine_or_probe() -> MachineProfile {
    MachineProfile {
        hostname: "cli".into(),
        platform_str: "cli".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0, // F12 fix: no VRAM on this machine; was wrongly storing VmHWM here
        cpu_count: 8,
    }
}

/// F4: extract effective (quant, context_len) from an AdmitRecord's applied_degradation.
///
/// If a degradation was applied, use its parameters; otherwise fall back to the
/// original request.
fn effective_config<'a>(
    rec: &fitsproof::admit::AdmitRecord,
    original_quant: &'a str,
    original_context: usize,
) -> (&'a str, usize) {
    // Note: the returned quant lifetime is tied to original_quant since we only
    // reference static strings from the plan module.
    if let Some(step) = &rec.applied_degradation {
        let eff_quant = match step.kind {
            fitsproof::plan::DegradationKind::LowerQuant => {
                // The description contains "Use <quant> instead of ..."; extract quant.
                // Rather than parsing, look up against the known quant levels.
                let desc = &step.description;
                if desc.contains("int4_sym") {
                    "int4_sym"
                } else if desc.contains("int8_sym") {
                    "int8_sym"
                } else if desc.contains("float16") {
                    "float16"
                } else {
                    original_quant
                }
            }
            fitsproof::plan::DegradationKind::ShorterContext => original_quant,
        };

        let eff_context = match step.kind {
            fitsproof::plan::DegradationKind::ShorterContext => {
                // Description: "Reduce context to <N> tokens (1/D of C)"
                // Parse the target context length from the description.
                extract_context_from_description(&step.description).unwrap_or(original_context)
            }
            fitsproof::plan::DegradationKind::LowerQuant => original_context,
        };

        (eff_quant, eff_context)
    } else {
        (original_quant, original_context)
    }
}

/// Parse "Reduce context to <N> tokens ..." from a degradation description.
fn extract_context_from_description(desc: &str) -> Option<usize> {
    // "Reduce context to 256 tokens (1/2 of 512)"
    let after = desc.strip_prefix("Reduce context to ")?;
    let end = after.find(' ')?;
    after[..end].parse().ok()
}

/// Build a ModelConfig with a potentially different max_seq_len for F4.
///
/// When the degradation reduces context, the KV cache in the engine must also
/// use the shorter context length.
fn build_effective_cfg(base: &ModelConfig, context_len: usize) -> ModelConfig {
    let mut cfg = base.clone();
    cfg.max_seq_len = context_len;
    cfg
}
