//! MCP stdio server — exposes `probe`, `plan`, and `admit` as MCP tools.
//!
//! Protocol: JSON-RPC 2.0 over stdin/stdout.
//! Spec: https://spec.modelcontextprotocol.io/specification/2024-11-05/
//!
//! # Supported MCP methods
//!
//! - `initialize` — returns server capabilities and tool list
//! - `tools/list`  — lists probe / plan / admit
//! - `tools/call`  — dispatches to the named tool
//! - `ping`        — liveness check
//!
//! # Tool signatures
//!
//! `probe` — no arguments. Returns machine bandwidth, GEMM rate, RAM.
//!
//! `plan` — arguments: `budget_gb` (number), `quant` (string, optional),
//!          `context_len` (integer, optional). Returns verdict + predicted peak.
//!
//! `admit` — same arguments as plan. Returns admitted/degraded/refused with
//!           the binding constraint when refused.

use std::io::{BufRead, BufReader, Write};

use crate::admit::{admit as do_admit, AdmitStatus};
use crate::model::ModelConfig;
use crate::plan::plan as do_plan;
use crate::probe::{probe, MachineProfile};

// ---------------------------------------------------------------------------
// Minimal JSON-RPC 2.0 helpers
// ---------------------------------------------------------------------------

fn rpc_result(id: &str, result: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{id},"result":{result}}}"#)
}

fn rpc_error(id: &str, code: i32, message: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{id},"error":{{"code":{code},"message":"{message}"}}}}"#)
}

/// Extract a string field from a flat JSON object.
fn json_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let after = json[pos + needle.len()..].trim_start_matches([' ', ':']);
    if let Some(after) = after.strip_prefix('"') {
        let end = after.find('"')?;
        Some(&after[..end])
    } else {
        None
    }
}

/// Extract a numeric field from a flat JSON object.
fn json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let after = json[pos + needle.len()..].trim_start_matches([' ', ':']);
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

/// Extract the RPC id (string or number) as a JSON token.
fn extract_id(json: &str) -> String {
    let needle = "\"id\"";
    if let Some(pos) = json.find(needle) {
        let after = json[pos + needle.len()..].trim_start_matches([' ', ':']);
        // String id.
        if let Some(after) = after.strip_prefix('"') {
            if let Some(end) = after.find('"') {
                return format!("\"{}\"", &after[..end]);
            }
        }
        // Numeric id.
        let end = after
            .find(|c: char| !c.is_ascii_digit() && c != '-')
            .unwrap_or(after.len());
        if !after[..end].is_empty() {
            return after[..end].to_string();
        }
    }
    "null".to_string()
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

fn tool_probe() -> String {
    let profile = probe(8 * 1024 * 1024, 3, 3);
    format!(
        r#"{{"bandwidth_gbps":{:.2},"gemm_tflops":{:.2},"ram_gb":{:.1},"hostname":"{}"}}"#,
        profile.memory_bandwidth_bps / 1e9,
        profile.gemm_throughput_flops / 1e12,
        profile.memory_bytes as f64 / 1e9,
        profile.hostname,
    )
}

fn tool_plan(args: &str) -> Result<String, String> {
    let budget_gb = json_f64(args, "budget_gb").unwrap_or(4.0);
    let quant = json_str(args, "quant").unwrap_or("none").to_string();
    let context_len = json_f64(args, "context_len")
        .map(|v| v as usize)
        .unwrap_or(512);

    let cfg = ModelConfig::reference();
    let machine = synthetic_machine();
    let budget_bytes = (budget_gb * 1e9) as u64;

    match do_plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => Ok(format!(
            r#"{{"verdict":"{:?}","predicted_peak_gb":{:.3},"budget_gb":{:.3},"quant":"{}","context_len":{},"binding_constraint":"{}"}}"#,
            p.verdict,
            p.predicted_peak_bytes as f64 / 1e9,
            budget_gb,
            p.quant,
            p.context_len,
            p.binding_constraint,
        )),
        Err(e) => Err(e.to_string()),
    }
}

fn tool_admit(args: &str) -> Result<String, String> {
    let budget_gb = json_f64(args, "budget_gb").unwrap_or(4.0);
    let quant = json_str(args, "quant").unwrap_or("none").to_string();
    let context_len = json_f64(args, "context_len")
        .map(|v| v as usize)
        .unwrap_or(512);

    let cfg = ModelConfig::reference();
    let machine = synthetic_machine();
    let budget_bytes = (budget_gb * 1e9) as u64;

    match do_plan(&cfg, &machine, context_len, budget_bytes, &quant, 0.6) {
        Ok(p) => {
            let rec = do_admit(p);
            let status_str = match rec.status {
                AdmitStatus::Admitted => "admitted",
                AdmitStatus::Degraded => "degraded",
                AdmitStatus::Refused => "refused",
            };
            Ok(format!(
                r#"{{"status":"{}","message":"{}","binding_constraint":"{}"}}"#,
                status_str,
                rec.message.replace('"', "\\\""),
                rec.plan.binding_constraint.replace('"', "\\\""),
            ))
        }
        Err(e) => Err(e.to_string()),
    }
}

fn synthetic_machine() -> MachineProfile {
    MachineProfile {
        hostname: "mcp".into(),
        platform_str: "mcp".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

// ---------------------------------------------------------------------------
// Tool list (MCP schema)
// ---------------------------------------------------------------------------

fn tools_list_json() -> &'static str {
    r#"{"tools":[
  {"name":"probe","description":"Measure this machine: memory bandwidth, GEMM rate, RAM.",
   "inputSchema":{"type":"object","properties":{}}},
  {"name":"plan","description":"Predict peak memory for a configuration and budget.",
   "inputSchema":{"type":"object","properties":{
     "budget_gb":{"type":"number","description":"Budget in GB"},
     "quant":{"type":"string","description":"Quantisation: none|float16|int8_sym|int4_sym|q4_k_m"},
     "context_len":{"type":"integer","description":"Context length in tokens"}
   },"required":["budget_gb"]}},
  {"name":"admit","description":"Admit/degrade/refuse with named binding constraint.",
   "inputSchema":{"type":"object","properties":{
     "budget_gb":{"type":"number","description":"Budget in GB"},
     "quant":{"type":"string","description":"Quantisation: none|float16|int8_sym|int4_sym|q4_k_m"},
     "context_len":{"type":"integer","description":"Context length in tokens"}
   },"required":["budget_gb"]}}
]}"#
}

// ---------------------------------------------------------------------------
// Initialize response
// ---------------------------------------------------------------------------

fn initialize_result() -> String {
    format!(
        r#"{{"protocolVersion":"2024-11-05","capabilities":{{"tools":{{}}}},"serverInfo":{{"name":"fitsproof","version":"{}"}}}}"#,
        crate::VERSION
    )
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

fn handle_rpc(line: &str) -> String {
    let id = extract_id(line);

    let method = match json_str(line, "method") {
        Some(m) => m.to_string(),
        None => return rpc_error(&id, -32600, "missing method"),
    };

    match method.as_str() {
        "initialize" => rpc_result(&id, &initialize_result()),
        "ping" => rpc_result(&id, r#"{"pong":true}"#),
        "tools/list" | "mcp/listTools" => rpc_result(&id, tools_list_json()),
        "tools/call" | "mcp/callTool" => {
            // Extract tool name and arguments from the params object.
            // params shape: {"name":"<tool>","arguments":{...}}
            let tool_name = json_str(line, "name").unwrap_or("").to_string();
            // For arguments, pass the full line (our json_ helpers search the full string).
            let args_portion = if let Some(pos) = line.find("\"arguments\"") {
                &line[pos..]
            } else {
                "{}"
            };

            let result = match tool_name.as_str() {
                "probe" => Ok(tool_probe()),
                "plan" => tool_plan(args_portion),
                "admit" => tool_admit(args_portion),
                other => Err(format!("unknown tool: {other}")),
            };

            match result {
                Ok(content) => rpc_result(
                    &id,
                    &format!(r#"{{"content":[{{"type":"text","text":{content}}}]}}"#),
                ),
                Err(e) => rpc_error(&id, -32603, &e.replace('"', "'")),
            }
        }
        // Notifications (no id required, no response needed but we return empty).
        "notifications/initialized" => String::new(),
        other => rpc_error(&id, -32601, &format!("method not found: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the MCP stdio server.
///
/// Reads JSON-RPC 2.0 requests from stdin, one per line, and writes responses
/// to stdout.  Runs until stdin is closed.
pub fn run_stdio() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in BufReader::new(stdin.lock()).lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let response = handle_rpc(line);
        if !response.is_empty() {
            writeln!(out, "{response}").ok();
            out.flush().ok();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Fault detected: initialize returns wrong protocol version.
    #[test]
    fn initialize_returns_protocol_version() {
        let resp = handle_rpc(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        assert!(
            resp.contains("2024-11-05"),
            "initialize must include protocol version, got: {resp}"
        );
    }

    /// Fault detected: tools/list returns no tools.
    #[test]
    fn tools_list_returns_all_three_tools() {
        let resp = handle_rpc(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#);
        assert!(resp.contains("\"probe\""), "must list probe tool");
        assert!(resp.contains("\"plan\""), "must list plan tool");
        assert!(resp.contains("\"admit\""), "must list admit tool");
    }

    /// Fault detected: ping returns error instead of pong.
    #[test]
    fn ping_returns_pong() {
        let resp = handle_rpc(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}"#);
        assert!(resp.contains("pong"), "ping must return pong, got: {resp}");
    }

    /// Fault detected: tools/call with admit doesn't return status field.
    #[test]
    fn admit_tool_returns_status() {
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":4.0}}}"#;
        let resp = handle_rpc(req);
        assert!(
            resp.contains("admitted") || resp.contains("degraded") || resp.contains("refused"),
            "admit tool must return a status, got: {resp}"
        );
    }

    /// Fault detected: admit tool doesn't refuse tiny budget.
    #[test]
    fn admit_tool_refuses_tiny_budget() {
        let req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":0.00001}}}"#;
        let resp = handle_rpc(req);
        assert!(
            resp.contains("refused"),
            "admit with tiny budget must return refused, got: {resp}"
        );
    }

    /// Fault detected: plan tool doesn't return predicted_peak_gb.
    #[test]
    fn plan_tool_returns_predicted_peak() {
        let req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"plan","arguments":{"budget_gb":4.0}}}"#;
        let resp = handle_rpc(req);
        assert!(
            resp.contains("predicted_peak_gb"),
            "plan tool must return predicted_peak_gb, got: {resp}"
        );
    }

    /// Fault detected: unknown tool doesn't return error.
    #[test]
    fn unknown_tool_returns_error() {
        let req = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nonexistent","arguments":{}}}"#;
        let resp = handle_rpc(req);
        assert!(
            resp.contains("error"),
            "unknown tool must return error, got: {resp}"
        );
    }

    /// Fault detected: missing method doesn't return -32600.
    #[test]
    fn missing_method_returns_parse_error() {
        let resp = handle_rpc(r#"{"jsonrpc":"2.0","id":7}"#);
        assert!(
            resp.contains("error"),
            "missing method must return error, got: {resp}"
        );
    }

    /// Fault detected: extract_id fails on numeric ids.
    #[test]
    fn extract_id_handles_numeric() {
        assert_eq!(extract_id(r#"{"id":42}"#), "42");
    }

    /// Fault detected: extract_id fails on string ids.
    #[test]
    fn extract_id_handles_string() {
        assert_eq!(extract_id(r#"{"id":"abc"}"#), "\"abc\"");
    }
}
