//! Pareto frontier sweep.
//!
//! Sweeps (quantisation, context_length) pairs over a range of budget levels
//! and computes the Pareto frontier: configurations where no other configuration
//! is strictly better on both axes (smaller peak memory AND longer context).
//!
//! The two objectives:
//! 1. Minimise predicted peak memory (lower is better — fits cheaper hardware).
//! 2. Maximise context length (longer is better — more useful).
//!
//! A configuration is **Pareto-dominated** if there exists another with
//! strictly smaller peak AND strictly larger context.  The frontier is the
//! non-dominated set.
//!
//! # Output
//!
//! JSON array of Pareto-optimal configs, sorted by context length descending.

use crate::model::ModelConfig;
use crate::plan::{plan as do_plan, Verdict};
use crate::probe::MachineProfile;

/// A single configuration in the sweep.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub quant: String,
    pub context_len: usize,
    pub predicted_peak_bytes: u64,
    pub budget_bytes: u64,
    pub verdict: Verdict,
}

/// The Pareto frontier result.
#[derive(Debug)]
pub struct ParetoResult {
    pub total_configs: usize,
    pub admitted_configs: usize,
    pub frontier: Vec<SweepConfig>,
}

impl ParetoResult {
    /// Format as JSON for CLI output.
    pub fn to_json(&self) -> String {
        let mut frontier_json = String::from("[\n");
        for (i, cfg) in self.frontier.iter().enumerate() {
            frontier_json.push_str(&format!(
                r#"  {{"quant":"{}","context_len":{},"predicted_peak_gb":{:.3},"budget_gb":{:.3},"verdict":"{:?}"}}"#,
                cfg.quant,
                cfg.context_len,
                cfg.predicted_peak_bytes as f64 / 1e9,
                cfg.budget_bytes as f64 / 1e9,
                cfg.verdict,
            ));
            if i + 1 < self.frontier.len() {
                frontier_json.push(',');
            }
            frontier_json.push('\n');
        }
        frontier_json.push(']');

        format!(
            r#"{{"total_configs":{},"admitted_configs":{},"frontier_size":{},"frontier":{}}}"#,
            self.total_configs,
            self.admitted_configs,
            self.frontier.len(),
            frontier_json,
        )
    }
}

/// Compute the Pareto frontier for (quant × context_len) at a given budget.
///
/// Returns Pareto-optimal configurations sorted by context_len descending.
pub fn pareto_sweep(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    budget_bytes: u64,
) -> ParetoResult {
    let quants = ["none", "float16", "int8_sym", "int4_sym", "q4_k_m"];
    let context_lens = [128usize, 256, 512, 1024, 2048, 4096, 8192];

    let mut all_configs: Vec<SweepConfig> = Vec::new();

    for &quant in &quants {
        for &ctx in &context_lens {
            let p = match do_plan(cfg, machine, ctx, budget_bytes, quant, 0.6) {
                Ok(p) => p,
                Err(_) => continue,
            };
            all_configs.push(SweepConfig {
                quant: quant.to_string(),
                context_len: ctx,
                predicted_peak_bytes: p.predicted_peak_bytes,
                budget_bytes,
                verdict: p.verdict,
            });
        }
    }

    let total_configs = all_configs.len();

    // Keep only admitted (or degraded) configs for the frontier.
    let admitted: Vec<&SweepConfig> = all_configs
        .iter()
        .filter(|c| !matches!(c.verdict, Verdict::DoesNotFit))
        .collect();
    let admitted_configs = admitted.len();

    // Compute Pareto frontier: keep configs not dominated by any other.
    // Objective 1: minimise peak (lower = better)
    // Objective 2: maximise context (higher = better)
    let mut frontier: Vec<SweepConfig> = Vec::new();
    for candidate in &admitted {
        let dominated = admitted.iter().any(|other| {
            other.predicted_peak_bytes < candidate.predicted_peak_bytes
                && other.context_len > candidate.context_len
        });
        if !dominated {
            frontier.push((*candidate).clone());
        }
    }

    // Sort by context_len descending (most capable first).
    frontier.sort_by(|a, b| {
        b.context_len
            .cmp(&a.context_len)
            .then(a.predicted_peak_bytes.cmp(&b.predicted_peak_bytes))
    });

    // Deduplicate: keep only the best (lowest peak) for each context length.
    let mut deduped: Vec<SweepConfig> = Vec::new();
    let mut seen_ctx = std::collections::HashSet::new();
    for c in frontier {
        if seen_ctx.insert(c.context_len) {
            deduped.push(c);
        }
    }

    ParetoResult {
        total_configs,
        admitted_configs,
        frontier: deduped,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_machine() -> MachineProfile {
        MachineProfile {
            hostname: "test".into(),
            platform_str: "test".into(),
            measured_at: 1.0,
            memory_bandwidth_bps: 20_000_000_000.0,
            gemm_throughput_flops: 100_000_000_000.0,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 8,
        }
    }

    /// Fault detected: pareto_sweep returns empty frontier on a generous budget.
    #[test]
    fn pareto_sweep_returns_non_empty_frontier_for_generous_budget() {
        let cfg = ModelConfig::reference();
        let machine = test_machine();
        let result = pareto_sweep(&cfg, &machine, 4_000_000_000);
        assert!(
            !result.frontier.is_empty(),
            "frontier must be non-empty for a generous budget, got: {:?}",
            result.frontier
        );
    }

    /// Fault detected: frontier contains dominated configs.
    #[test]
    fn frontier_configs_are_not_dominated() {
        let cfg = ModelConfig::reference();
        let machine = test_machine();
        let result = pareto_sweep(&cfg, &machine, 4_000_000_000);

        // For each pair (a, b) in frontier, neither should dominate the other.
        for (i, a) in result.frontier.iter().enumerate() {
            for (j, b) in result.frontier.iter().enumerate() {
                if i == j {
                    continue;
                }
                let b_dominates_a = b.predicted_peak_bytes < a.predicted_peak_bytes
                    && b.context_len > a.context_len;
                assert!(
                    !b_dominates_a,
                    "config {j} dominates config {i}: ({}, {}) dominates ({}, {})",
                    b.predicted_peak_bytes, b.context_len, a.predicted_peak_bytes, a.context_len,
                );
            }
        }
    }

    /// Fault detected: frontier is not sorted by context_len descending.
    #[test]
    fn frontier_is_sorted_by_context_len_descending() {
        let cfg = ModelConfig::reference();
        let machine = test_machine();
        let result = pareto_sweep(&cfg, &machine, 4_000_000_000);
        for w in result.frontier.windows(2) {
            assert!(
                w[0].context_len >= w[1].context_len,
                "frontier must be sorted by context_len descending: {} < {}",
                w[0].context_len,
                w[1].context_len
            );
        }
    }

    /// Fault detected: tiny budget returns admitted configs that don't fit.
    #[test]
    fn tiny_budget_returns_empty_or_refused_frontier() {
        let cfg = ModelConfig::reference();
        let machine = test_machine();
        // 1 KB — nothing should fit.
        let result = pareto_sweep(&cfg, &machine, 1_024);
        assert_eq!(
            result.admitted_configs, 0,
            "no configs should be admitted for a 1 KB budget"
        );
        assert!(
            result.frontier.is_empty(),
            "frontier must be empty for a 1 KB budget"
        );
    }

    /// Fault detected: to_json produces invalid JSON (missing braces/brackets).
    #[test]
    fn pareto_result_to_json_is_valid_shape() {
        let cfg = ModelConfig::reference();
        let machine = test_machine();
        let result = pareto_sweep(&cfg, &machine, 4_000_000_000);
        let json = result.to_json();
        assert!(json.starts_with('{'), "JSON must start with {{");
        assert!(json.ends_with('}'), "JSON must end with }}");
        assert!(
            json.contains("\"frontier\""),
            "JSON must contain frontier key"
        );
        assert!(
            json.contains("\"total_configs\""),
            "JSON must contain total_configs"
        );
    }
}
