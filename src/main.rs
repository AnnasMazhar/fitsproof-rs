//! `fitsproof` — the binary entry point.
//!
//! Subcommand surface (v0.1, mirrors the Python edition so the contract is what users learn):
//! `probe | plan | admit | verify | stress | serve | mcp | pareto`
//!
//! This file is the CLI shell only. Contract logic belongs in the library modules.

use std::process::ExitCode;

const USAGE: &str = "\
fitsproof — prove your local LLM fits in memory, or get a loud refusal

USAGE:
    fitsproof <COMMAND> [OPTIONS]

COMMANDS:
    probe     Measure this machine (memory bandwidth, GEMM rate, RAM/VRAM)
    plan      Predict peak memory for a configuration and a budget
    admit     Admit / degrade loudly / refuse (exit 2 on refusal)
    verify    Measure peak RSS during generation, assert <= budget
    stress    >=20 configurations: zero violations, zero silent mode changes
    serve     OpenAI-compatible HTTP server
    mcp       MCP stdio server (probe / plan / admit)
    pareto    Measured Pareto frontier over (quantization, context length)

    --version  Print version
    --help     Print this message
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
        Some(other) => {
            eprintln!("fitsproof: '{other}' is not implemented yet in the compiled edition.");
            eprintln!("See specs/fitsproof-rs.md for the v0.1 scope.");
            ExitCode::from(2)
        }
        None => {
            print!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
