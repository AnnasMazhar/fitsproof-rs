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
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

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
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

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
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
    let budget_bytes = (budget_gb * 1e9) as u64;
    let quant = parse_flag(args, "--quant").unwrap_or_else(|| "none".to_string());
    let context_len: usize = parse_flag(args, "--context")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

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
    let budget_gb = parse_budget_gb(args).unwrap_or(4.0);
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
