//! OpenAI-compatible HTTP server.
//!
//! Implements a minimal `/v1/chat/completions` endpoint using only the Rust
//! standard library (no external HTTP framework).  Every response carries an
//! `admission_record` field with the contract outcome.  When the declared
//! budget is violated, the server returns 503 with the binding constraint.
//!
//! # Integration
//!
//! Swap `base_url` in any OpenAI client:
//! ```bash
//! OPENAI_BASE_URL=http://localhost:8080/v1
//! ```
//!
//! # Request shape (subset of OpenAI ChatCompletions)
//! ```json
//! {
//!   "model": "fitsproof/ref",
//!   "messages": [{"role":"user","content":"hello"}],
//!   "max_tokens": 32,
//!   "temperature": 0.0,
//!   "budget_gb": 4.0   // fitsproof extension
//! }
//! ```
//!
//! # Response shape
//! Standard OpenAI ChatCompletion with an extra `admission_record` field.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

use crate::admit::{admit, AdmitRecord, AdmitStatus};
use crate::engine::transformer::{Transformer, Weights};
use crate::model::ModelConfig;
use crate::plan::plan;
use crate::probe::MachineProfile;

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 parser
// ---------------------------------------------------------------------------

/// A parsed HTTP request (only what we need).
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> Option<HttpRequest> {
    let mut reader = BufReader::new(stream);

    // Read request line.
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    // Read headers until blank line.
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some(rest) = line.to_lowercase().strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    // Read body.
    let mut body = vec![0u8; content_length.min(1_048_576)]; // cap at 1 MB
    if content_length > 0 {
        use std::io::Read;
        reader.read_exact(&mut body).ok()?;
    }

    Some(HttpRequest { method, path, body })
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (only what we need from the request)
// ---------------------------------------------------------------------------

/// Extract a string value from a flat JSON object: `"key": "value"`.
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

/// Extract a numeric value from a flat JSON object: `"key": 3.5`.
fn json_f64(json: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let pos = json.find(&needle)?;
    let after = json[pos + needle.len()..].trim_start_matches([' ', ':']);
    // Find end of number (first non-numeric char).
    let end = after
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e')
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

/// Extract the content of the last "content" field in the messages array.
fn extract_last_user_content(json: &str) -> Option<String> {
    // Find all "content": "..." occurrences and return the last one.
    let mut last = None;
    let mut search = json;
    while let Some(pos) = search.find("\"content\"") {
        let rest = search[pos + 9..].trim_start_matches([' ', ':']);
        if let Some(rest) = rest.strip_prefix('"') {
            if let Some(end) = rest.find('"') {
                last = Some(rest[..end].to_string());
            }
        }
        search = &search[pos + 9..];
    }
    last
}

// ---------------------------------------------------------------------------
// HTTP response helpers
// ---------------------------------------------------------------------------

fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn write_options_response(stream: &mut TcpStream) {
    let response = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n";
    let _ = stream.write_all(response.as_bytes());
}

// ---------------------------------------------------------------------------
// Chat completions handler
// ---------------------------------------------------------------------------

fn escape_json(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            c => vec![c],
        })
        .collect()
}

fn handle_completions(body: &[u8], machine: &MachineProfile) -> (u16, String) {
    let json = String::from_utf8_lossy(body);

    // Parse request fields.
    let budget_gb = json_f64(&json, "budget_gb").unwrap_or(4.0);
    let max_tokens = json_f64(&json, "max_tokens").unwrap_or(16.0) as usize;
    let temperature = json_f64(&json, "temperature").unwrap_or(0.0) as f32;
    let prompt = extract_last_user_content(&json).unwrap_or_else(|| "hello".to_string());
    let model_name = json_str(&json, "model")
        .unwrap_or("fitsproof/ref")
        .to_string();

    let budget_bytes = (budget_gb * 1e9) as u64;
    let cfg = ModelConfig::reference();

    // Contract: plan + admit.
    let p = match plan(&cfg, machine, 512, budget_bytes, "none", 0.6) {
        Ok(p) => p,
        Err(e) => {
            return (
                503,
                format!(
                    r#"{{"error":{{"message":"plan error: {}","type":"fitsproof_error"}}}}"#,
                    escape_json(&e.to_string())
                ),
            );
        }
    };

    let rec: AdmitRecord = admit(p);

    if rec.status == AdmitStatus::Refused {
        return (
            503,
            format!(
                r#"{{"error":{{"message":"{}","type":"fitsproof_refused","admission_record":{{"status":"refused","binding_constraint":"{}"}}}}}}"#,
                escape_json(&rec.message),
                escape_json(&rec.plan.binding_constraint)
            ),
        );
    }

    // Generate tokens using the reference bundle.
    // In v0.2, this will use real loaded weights from the model file.
    let weights = Weights::reference(&cfg);
    let mut transformer = Transformer::new(cfg.clone(), weights);

    // Convert prompt to simple token ids (byte values mod vocab_size).
    let prompt_ids: Vec<u32> = prompt
        .bytes()
        .take(32)
        .map(|b| (b as u32) % cfg.vocab_size as u32)
        .collect();

    let token_ids = transformer.generate(
        if prompt_ids.is_empty() {
            &[1, 2, 3]
        } else {
            &prompt_ids
        },
        max_tokens.min(64),
        temperature,
        42,
    );

    // Decode tokens back to text (reference bundle: just show token ids as text).
    let generated_text: String = token_ids
        .iter()
        .map(|&id| (id % 128) as u8 as char)
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .collect();
    let generated_text = if generated_text.is_empty() {
        format!("[{} tokens generated]", token_ids.len())
    } else {
        generated_text
    };

    // Build OpenAI-compatible response.
    let admission_json = match rec.status {
        AdmitStatus::Admitted => format!(
            r#"{{"status":"admitted","message":"{}"}}"#,
            escape_json(&rec.message)
        ),
        AdmitStatus::Degraded => format!(
            r#"{{"status":"degraded","message":"{}"}}"#,
            escape_json(&rec.message)
        ),
        AdmitStatus::Refused => unreachable!(),
    };

    let response_body = format!(
        r#"{{
  "id": "fitsproof-1",
  "object": "chat.completion",
  "model": "{model_name}",
  "choices": [{{
    "index": 0,
    "message": {{
      "role": "assistant",
      "content": "{content}"
    }},
    "finish_reason": "stop"
  }}],
  "usage": {{
    "prompt_tokens": {prompt_tokens},
    "completion_tokens": {completion_tokens},
    "total_tokens": {total_tokens}
  }},
  "admission_record": {admission_json}
}}"#,
        model_name = escape_json(&model_name),
        content = escape_json(&generated_text),
        prompt_tokens = prompt_ids.len(),
        completion_tokens = token_ids.len(),
        total_tokens = prompt_ids.len() + token_ids.len(),
        admission_json = admission_json,
    );

    (200, response_body)
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------

fn handle_models() -> String {
    r#"{"object":"list","data":[{"id":"fitsproof/ref","object":"model","created":1700000000,"owned_by":"fitsproof"}]}"#.to_string()
}

// ---------------------------------------------------------------------------
// Test helper (exposed for integration tests in tests/adversarial.rs)
// ---------------------------------------------------------------------------

/// Dispatch a single HTTP request by path and body string.
///
/// Returns (status_code, response_body). Exposed for integration tests — not
/// part of the stable public API. Uses a synthetic machine profile.
#[doc(hidden)]
pub fn handle_request_for_test(path: &str, body: &str) -> (u16, String) {
    let machine = MachineProfile {
        hostname: "test".into(),
        platform_str: "test".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    };
    match path {
        "/v1/chat/completions" | "/v1/chat/completions/" => {
            handle_completions(body.as_bytes(), &machine)
        }
        "/v1/models" | "/v1/models/" => (200, handle_models()),
        "/health" => (200, r#"{"status":"ok"}"#.to_string()),
        _ => (
            404,
            r#"{"error":{"message":"Not found","type":"invalid_request_error"}}"#.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Main server loop
// ---------------------------------------------------------------------------

/// Run the OpenAI-compatible HTTP server.
///
/// Binds to `addr` (e.g. `"127.0.0.1:8080"`) and handles one request at a
/// time.  Returns on error.
pub fn run_server(addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("fitsproof serve: listening on http://{addr}");
    eprintln!("  POST /v1/chat/completions  — generate with contract enforcement");
    eprintln!("  GET  /v1/models            — list available models");
    eprintln!("  Press Ctrl-C to stop.");

    let machine = MachineProfile {
        hostname: "serve".into(),
        platform_str: "serve".into(),
        measured_at: 1_000_000.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    };

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        let req = match read_request(&stream) {
            Some(r) => r,
            None => continue,
        };

        // CORS preflight.
        if req.method == "OPTIONS" {
            write_options_response(&mut stream);
            continue;
        }

        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") | ("GET", "/v1/models/") => {
                write_response(&mut stream, 200, &handle_models());
            }
            ("GET", "/health") => {
                write_response(&mut stream, 200, r#"{"status":"ok"}"#);
            }
            ("POST", "/v1/chat/completions") | ("POST", "/v1/chat/completions/") => {
                let (status, body) = handle_completions(&req.body, &machine);
                write_response(&mut stream, status, &body);
            }
            _ => {
                write_response(
                    &mut stream,
                    404,
                    r#"{"error":{"message":"Not found","type":"invalid_request_error"}}"#,
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Fault detected: json_str returns None for a present key.
    #[test]
    fn json_str_extracts_value() {
        let json = r#"{"model": "gpt-4", "temperature": 0.5}"#;
        assert_eq!(json_str(json, "model"), Some("gpt-4"));
    }

    /// Fault detected: json_f64 returns None for a present key.
    #[test]
    fn json_f64_extracts_value() {
        let json = r#"{"budget_gb": 4.0, "max_tokens": 32}"#;
        assert!((json_f64(json, "budget_gb").unwrap() - 4.0).abs() < 1e-9);
        assert!((json_f64(json, "max_tokens").unwrap() - 32.0).abs() < 1e-9);
    }

    /// Fault detected: json_f64 returns wrong value for a missing key.
    #[test]
    fn json_f64_missing_key_returns_none() {
        let json = r#"{"model": "x"}"#;
        assert_eq!(json_f64(json, "budget_gb"), None);
    }

    /// Fault detected: escape_json doesn't escape double quotes.
    #[test]
    fn escape_json_escapes_quotes() {
        assert_eq!(escape_json(r#"say "hi""#), r#"say \"hi\""#);
    }

    /// Fault detected: escape_json doesn't escape newlines.
    #[test]
    fn escape_json_escapes_newline() {
        assert_eq!(escape_json("a\nb"), r"a\nb");
    }

    /// Fault detected: handle_completions returns wrong status for tiny budget.
    #[test]
    fn handle_completions_tiny_budget_returns_503() {
        let machine = crate::probe::MachineProfile {
            hostname: "test".into(),
            platform_str: "test".into(),
            measured_at: 1.0,
            memory_bandwidth_bps: 20_000_000_000.0,
            gemm_throughput_flops: 100_000_000_000.0,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 4,
        };
        let body = br#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hi"}],"budget_gb":0.0001}"#;
        let (status, resp) = handle_completions(body, &machine);
        assert_eq!(
            status, 503,
            "tiny budget must return 503 REFUSED, got: {resp}"
        );
    }

    /// Fault detected: handle_completions returns error status for valid budget.
    #[test]
    fn handle_completions_valid_budget_returns_200() {
        let machine = crate::probe::MachineProfile {
            hostname: "test".into(),
            platform_str: "test".into(),
            measured_at: 1.0,
            memory_bandwidth_bps: 20_000_000_000.0,
            gemm_throughput_flops: 100_000_000_000.0,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 4,
        };
        let body = br#"{"model":"fitsproof/ref","messages":[{"role":"user","content":"hello"}],"budget_gb":4.0,"max_tokens":4}"#;
        let (status, resp) = handle_completions(body, &machine);
        assert_eq!(status, 200, "valid budget must return 200, got: {resp}");
        assert!(
            resp.contains("admission_record"),
            "response must include admission_record"
        );
    }

    /// Fault detected: response doesn't include admission_record on successful request.
    #[test]
    fn handle_completions_response_includes_admission_record() {
        let machine = crate::probe::MachineProfile {
            hostname: "test".into(),
            platform_str: "test".into(),
            measured_at: 1.0,
            memory_bandwidth_bps: 20_000_000_000.0,
            gemm_throughput_flops: 100_000_000_000.0,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 4,
        };
        let body =
            br#"{"messages":[{"role":"user","content":"test"}],"budget_gb":4.0,"max_tokens":2}"#;
        let (_status, resp) = handle_completions(body, &machine);
        assert!(
            resp.contains("\"admission_record\""),
            "response must have admission_record field"
        );
    }

    /// Fault detected: extract_last_user_content returns None when content is present.
    #[test]
    fn extract_last_user_content_finds_content() {
        let json = r#"{"messages":[{"role":"user","content":"hello world"}]}"#;
        let content = extract_last_user_content(json);
        assert_eq!(content, Some("hello world".to_string()));
    }
}
