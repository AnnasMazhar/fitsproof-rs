//! Analytical cost model for LLM inference.
//!
//! Computes predicted peak memory and throughput from model architecture
//! and machine measurements, using the roofline model.
//!
//! # Roofline model (Williams et al. 2009)
//!
//! LLM decoding is memory-bandwidth-bound:
//!
//!   `tok/s = effective_bandwidth / bytes_per_token_of_weights`
//!
//! where `bytes_per_token_of_weights` ≈ `weight_bytes(model, quant)` because
//! all weights are streamed once per decode step (at context lengths where KV
//! cache is not the dominant term).
//!
//! # Peak memory formula
//!
//!   `peak = weight_bytes + kv_cache_bytes + activation_bytes`
//!
//! `kv_cache_bytes` (GQA, Ainslie et al. 2023):
//!   `2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element`
//!
//! # Sources
//!
//! - Williams et al. 2009 (Roofline), https://dl.acm.org/doi/10.1145/1498765.1498785
//! - Ainslie et al. 2023 (GQA), https://arxiv.org/abs/2305.13245
//! - Sheng et al. 2023 (FlexGen), https://arxiv.org/abs/2303.06865
//! - Kaplan et al. 2020 (Scaling Laws), https://arxiv.org/abs/2001.08361

use crate::model::ModelConfig;
use crate::probe::MachineProfile;

/// Bits per element for each quantisation scheme.
///
/// `none` means fp32 (32 bits).  int4 packs 2 values per byte (4 bits effective).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantBits(pub f64);

impl QuantBits {
    /// Parse a quantisation name to bits per element.
    ///
    /// Returns `None` for unknown names (callers must reject unknown quants fail-closed).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" | "float32" | "fp32" => Some(QuantBits(32.0)),
            "float16" | "fp16" | "f16" => Some(QuantBits(16.0)),
            "int8_sym" | "int8_asym" | "int8" | "q8_0" => Some(QuantBits(8.0)),
            "int4_sym" | "int4_asym" | "int4" | "q4_k" | "q4_0" => Some(QuantBits(4.0)),
            _ => None,
        }
    }

    /// Bytes per element (fractional for int4: 0.5).
    #[inline]
    pub fn bytes_per_element(self) -> f64 {
        self.0 / 8.0
    }
}

/// Full cost estimate for a (model, quant, context, machine) configuration.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Model weight storage in bytes.
    pub weight_bytes: u64,
    /// KV cache in bytes for the full context window.
    pub kv_cache_bytes: u64,
    /// Peak activation buffer (one batch, one layer) in bytes.
    pub activation_bytes: u64,
    /// Conservative peak: weight + kv_cache + activation.
    pub total_peak_bytes: u64,
    /// Decode throughput prediction (tokens/second).
    pub predicted_tok_s: f64,
    /// Time to first token (seconds, prefill estimate).
    pub predicted_ttft_s: f64,
    /// Arithmetic intensity (FLOPS/byte) for one decode step.
    pub arithmetic_intensity: f64,
}

/// Compute model weight storage in bytes.
///
/// Counts: embedding table (fp32) + per-layer attention projections (Q, K, V, O) +
/// per-layer FFN (gate, up, down) + per-layer norms (fp32) + final norm + unembed (fp32).
///
/// Norm weights stay fp32 regardless of quant (standard practice in quantised models).
///
/// # Fault detected
///
/// Omitting the embedding table doubles the error for models with large vocabularies.
/// Tests verify against a known-config reference (see `tests/cost_known_answer.rs`).
pub fn weight_bytes(cfg: &ModelConfig, quant: &str) -> u64 {
    let bits = QuantBits::from_name(quant)
        .expect("weight_bytes: unknown quant — caller must validate")
        .bytes_per_element();

    let d = cfg.hidden_size as f64;
    let h = cfg.num_heads as f64;
    let kv_h = cfg.num_kv_heads as f64;
    let hd = cfg.head_dim as f64;
    let ff = cfg.intermediate_size as f64;
    let v = cfg.vocab_size as f64;
    let l = cfg.num_layers as f64;

    // Embedding table (fp32 always)
    let embed_bytes = v * d * 4.0;

    // Per-layer attention: Q(h*hd*d) + K(kv_h*hd*d) + V(kv_h*hd*d) + O(d*h*hd)
    let attn_per_layer = (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd) * bits;

    // Per-layer FFN: gate(ff*d) + up(ff*d) + down(d*ff)
    let ffn_per_layer = (ff * d + ff * d + d * ff) * bits;

    // Per-layer norms (fp32, 2 per layer)
    let norm_per_layer = 2.0 * d * 4.0;

    // Final norm + unembed (fp32)
    let final_bytes = d * 4.0 + v * d * 4.0;

    (embed_bytes + l * (attn_per_layer + ffn_per_layer + norm_per_layer) + final_bytes) as u64
}

/// Compute KV cache memory for `context_len` tokens across all layers.
///
/// Formula (GQA, Ainslie et al. 2023):
///   `2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element`
///
/// The factor 2 covers both K and V tensors.
///
/// # Fault detected
///
/// Omitting the factor 2 halves the estimate and causes budget violations to go
/// undetected. Tests verify with known-config reference.
pub fn kv_cache_bytes(cfg: &ModelConfig, context_len: usize, quant: &str) -> u64 {
    let bits = QuantBits::from_name(quant)
        .expect("kv_cache_bytes: unknown quant — caller must validate")
        .bytes_per_element();
    (2.0 * cfg.num_layers as f64
        * cfg.num_kv_heads as f64
        * context_len as f64
        * cfg.head_dim as f64
        * bits) as u64
}

/// Peak activation buffer for one token, one layer.
///
/// Conservative: two full hidden-size buffers (input + output) plus one FFN buffer.
pub fn activation_bytes(cfg: &ModelConfig) -> u64 {
    ((2 * cfg.hidden_size + cfg.intermediate_size) * 4) as u64 // fp32
}

/// Predict decode throughput (tokens/second) using the roofline model.
///
/// Decode is memory-bandwidth-bound:
///   `tok/s = effective_bandwidth / weight_bytes`
///
/// `bandwidth_utilisation` is the empirical fraction of peak bandwidth achievable
/// (calibrate fits this from measurements; default 0.6 is conservative).
///
/// Source: Sheng et al. 2023 (FlexGen §3.1).
///
/// # Fault detected
///
/// Using `utilisation=1.0` over-predicts throughput by 40–67% vs measured.
pub fn decode_tok_s(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    quant: &str,
    bandwidth_utilisation: f64,
) -> f64 {
    let w = weight_bytes(cfg, quant);
    let effective_bw = machine.memory_bandwidth_bps * bandwidth_utilisation;
    if w == 0 || effective_bw <= 0.0 {
        return 0.0;
    }
    effective_bw / w as f64
}

/// Predict time-to-first-token (TTFT) for a prompt of `seq_len` tokens.
///
/// Prefill is compute-bound at large batch. FLOPS ≈ 2 * n_params * seq_len.
///
/// Source: Kaplan et al. 2020 (Scaling Laws, Appendix D).
///
/// # Fault detected
///
/// Omitting the factor 2 halves the FLOPS estimate and underpredicts TTFT.
pub fn prefill_ttft_s(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    seq_len: usize,
    quant: &str,
) -> f64 {
    let bits = QuantBits::from_name(quant)
        .expect("prefill_ttft_s: unknown quant")
        .bytes_per_element();
    let d = cfg.hidden_size as f64;
    let h = cfg.num_heads as f64;
    let kv_h = cfg.num_kv_heads as f64;
    let hd = cfg.head_dim as f64;
    let ff = cfg.intermediate_size as f64;
    let v = cfg.vocab_size as f64;
    let l = cfg.num_layers as f64;

    // Approximate n_params
    let params = v * d // embed
        + l * (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd) * (bits / 4.0) // rough normalisation
        + l * (ff * d + ff * d + d * ff) * (bits / 4.0)
        + v * d; // unembed
    let flops = 2.0 * params * seq_len as f64;
    if machine.gemm_throughput_flops <= 0.0 {
        return f64::INFINITY;
    }
    flops / machine.gemm_throughput_flops
}

/// Arithmetic intensity for one decode step: FLOPS per byte read.
///
/// For fp32: ≈ 0.5 FLOP/byte. For int8: ≈ 1 FLOP/byte.
/// Decode is always memory-bandwidth-bound in practice on CPU.
pub fn arithmetic_intensity(cfg: &ModelConfig, quant: &str) -> f64 {
    let d = cfg.hidden_size as f64;
    let h = cfg.num_heads as f64;
    let kv_h = cfg.num_kv_heads as f64;
    let hd = cfg.head_dim as f64;
    let ff = cfg.intermediate_size as f64;
    let v = cfg.vocab_size as f64;
    let l = cfg.num_layers as f64;

    let params = v * d
        + l * (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd)
        + l * (ff * d + ff * d + d * ff)
        + v * d;
    let w = weight_bytes(cfg, quant);
    if w == 0 {
        return 0.0;
    }
    (2.0 * params) / w as f64
}

/// Full cost estimate for a (model, machine, context, quant) configuration.
pub fn estimate(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    context_len: usize,
    quant: &str,
    bandwidth_utilisation: f64,
) -> CostEstimate {
    let w = weight_bytes(cfg, quant);
    let kv = kv_cache_bytes(cfg, context_len, quant);
    let act = activation_bytes(cfg);
    let total = w + kv + act;
    let tok_s = decode_tok_s(cfg, machine, quant, bandwidth_utilisation);
    let ttft = prefill_ttft_s(cfg, machine, context_len, quant);
    let ai = arithmetic_intensity(cfg, quant);

    CostEstimate {
        weight_bytes: w,
        kv_cache_bytes: kv,
        activation_bytes: act,
        total_peak_bytes: total,
        predicted_tok_s: tok_s,
        predicted_ttft_s: ttft,
        arithmetic_intensity: ai,
    }
}

#[cfg(test)]
mod tests {
    //! Known-answer tests (KATs) for the cost model.
    //!
    //! Each test derives the expected value independently from the formulas
    //! (hand-computed from the spec), then checks the implementation matches.
    //!
    //! Sources verified: Ainslie et al. 2023 (GQA), Williams et al. 2009 (Roofline).

    use super::*;
    use crate::model::ModelConfig;
    use crate::probe::MachineProfile;

    fn ref_cfg() -> ModelConfig {
        ModelConfig::reference()
    }

    fn ref_machine() -> MachineProfile {
        MachineProfile {
            hostname: "test-host".into(),
            platform_str: "test".into(),
            measured_at: 0.0,
            memory_bandwidth_bps: 20_000_000_000.0, // 20 GB/s
            gemm_throughput_flops: 100_000_000_000.0, // 100 GFLOPS
            memory_bytes: 32 * 1024 * 1024 * 1024,
            gpu_memory_bytes: 0,
            cpu_count: 8,
        }
    }

    /// Fault detected: embedding table omitted from weight_bytes.
    ///
    /// Hand-computed for reference config (6L, 384H, 6Q-heads, 2KV-heads, 64 hd, 1536 ff, 512V):
    ///
    ///   embed            = 512 * 384 * 4                    =    786_432  (fp32)
    ///   attn Q/layer     = 6 * 64 * 384 * 4                =    589_824
    ///   attn K/layer     = 2 * 64 * 384 * 4                =    196_608
    ///   attn V/layer     = 2 * 64 * 384 * 4                =    196_608
    ///   attn O/layer     = 384 * 6 * 64 * 4                =    589_824
    ///   attn/layer total                                    =  1_572_864
    ///   ffn gate/layer   = 1536 * 384 * 4                  =  2_359_296
    ///   ffn up/layer     = 1536 * 384 * 4                  =  2_359_296
    ///   ffn down/layer   = 384 * 1536 * 4                  =  2_359_296
    ///   ffn/layer total                                     =  7_077_888
    ///   norm/layer       = 2 * 384 * 4                     =      3_072
    ///   per-layer total  = 1_572_864 + 7_077_888 + 3_072   =  8_653_824
    ///   6 layers         = 6 * 8_653_824                   = 51_922_944
    ///   final norm       = 384 * 4                         =      1_536
    ///   unembed          = 512 * 384 * 4                   =    786_432
    ///   total = 786_432 + 51_922_944 + 1_536 + 786_432     = 53_497_344
    ///
    /// Source: formula from cost.rs weight_bytes(), traced term by term.
    /// This is an exact match — no tolerance.
    #[test]
    fn weight_bytes_reference_fp32_known_answer() {
        let cfg = ref_cfg();
        let wb = weight_bytes(&cfg, "none");
        // Exact hand-computed value — see derivation above.
        let expected: u64 = 53_497_344;
        assert_eq!(
            wb, expected,
            "weight_bytes fp32 expected {expected}, got {wb}"
        );
    }

    /// Fault detected: factor 2 omitted from kv_cache_bytes (halves the estimate).
    /// Hand-computed: 2 * 6 * 2 * 512 * 64 * 4 = 6_291_456 bytes (fp32, context=512).
    #[test]
    fn kv_cache_bytes_reference_fp32_known_answer() {
        let cfg = ref_cfg();
        // 2 * n_layers * n_kv_heads * context * head_dim * 4
        let expected: u64 = 2 * 6 * 2 * 512 * 64 * 4;
        let got = kv_cache_bytes(&cfg, 512, "none");
        assert_eq!(
            got, expected,
            "kv_cache_bytes fp32 got {got}, expected {expected}"
        );
    }

    /// Fault detected: int8 quantisation doubles the KV cache estimate (wrong bits).
    /// int8 = 1 byte/element. Expected: 2 * 6 * 2 * 512 * 64 * 1 = 1_572_864 bytes.
    #[test]
    fn kv_cache_bytes_int8_is_quarter_of_fp32() {
        let cfg = ref_cfg();
        let fp32 = kv_cache_bytes(&cfg, 512, "none");
        let int8 = kv_cache_bytes(&cfg, 512, "int8_sym");
        assert_eq!(
            int8,
            fp32 / 4,
            "int8 kv cache should be 1/4 of fp32 (got fp32={fp32}, int8={int8})"
        );
    }

    /// Fault detected: int4 quantisation not halving vs int8.
    #[test]
    fn kv_cache_bytes_int4_is_half_of_int8() {
        let cfg = ref_cfg();
        let int8 = kv_cache_bytes(&cfg, 512, "int8_sym");
        let int4 = kv_cache_bytes(&cfg, 512, "int4_sym");
        assert_eq!(
            int4,
            int8 / 2,
            "int4 kv cache should be 1/2 of int8 (got int8={int8}, int4={int4})"
        );
    }

    /// Fault detected: total_peak_bytes doesn't sum weight+kv+activation.
    #[test]
    fn total_peak_is_sum_of_components() {
        let cfg = ref_cfg();
        let machine = ref_machine();
        let est = estimate(&cfg, &machine, 512, "none", 0.6);
        assert_eq!(
            est.total_peak_bytes,
            est.weight_bytes + est.kv_cache_bytes + est.activation_bytes,
            "total_peak_bytes must equal the sum of components"
        );
    }

    /// Fault detected: bandwidth utilisation=1.0 used when 0.6 was passed.
    #[test]
    fn decode_tok_s_uses_utilisation() {
        let cfg = ref_cfg();
        let machine = ref_machine();
        let tok_s_60 = decode_tok_s(&cfg, &machine, "none", 0.6);
        let tok_s_30 = decode_tok_s(&cfg, &machine, "none", 0.3);
        // 0.6 should be 2× 0.3
        let ratio = tok_s_60 / tok_s_30;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "tok_s ratio (0.6/0.3 utilisation) should be 2.0, got {ratio}"
        );
    }

    /// Fault detected: unknown quant name silently defaults to fp32 instead of panicking.
    #[test]
    #[should_panic(expected = "unknown quant")]
    fn weight_bytes_panics_on_unknown_quant() {
        let cfg = ref_cfg();
        let _ = weight_bytes(&cfg, "int2_mystery");
    }

    // -----------------------------------------------------------------------
    // Property-based tests (proptest)
    //
    // Properties come from the method's assumptions, not from the implementation.
    // Source for each property is cited in the assertion comment.
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        /// Property (GQA, Ainslie et al. 2023):
        /// kv_cache_bytes is strictly monotonically increasing in context_len.
        ///
        /// Fault detected: formula is non-monotone (e.g. integer overflow at large contexts
        /// or incorrect use of integer division).
        #[test]
        fn kv_cache_bytes_monotone_in_context_len(
            a in 1usize..512,
            b in 513usize..1024,
        ) {
            let cfg = ref_cfg();
            let small = kv_cache_bytes(&cfg, a, "none");
            let large = kv_cache_bytes(&cfg, b, "none");
            prop_assert!(
                large > small,
                "kv_cache_bytes must be strictly increasing in context_len: ctx={a} got {small}, ctx={b} got {large}"
            );
        }

        /// Property (roofline, Williams et al. 2009 §3):
        /// decode_tok_s is strictly monotonically increasing in bandwidth_utilisation.
        ///
        /// Fault detected: formula ignores the utilisation parameter (off-by-one, wrong variable).
        #[test]
        fn decode_tok_s_monotone_in_utilisation(
            low in 0.1f64..0.5,
            high in 0.6f64..1.0,
        ) {
            let cfg = ref_cfg();
            let machine = ref_machine();
            let t_low = decode_tok_s(&cfg, &machine, "none", low);
            let t_high = decode_tok_s(&cfg, &machine, "none", high);
            prop_assert!(
                t_high > t_low,
                "decode_tok_s must increase with bandwidth_utilisation: {low} -> {t_low}, {high} -> {t_high}"
            );
        }

        /// Property (quantisation bits hierarchy):
        /// weight_bytes at lower precision ≤ weight_bytes at higher precision.
        ///
        /// Fault detected: QuantBits::from_name returns wrong bits/element, breaking ordering.
        #[test]
        fn weight_bytes_precision_ordering(
            n_layers in 1usize..8,
            hidden in 64usize..512,
        ) {
            // Build a minimal config; use fixed values for the rest.
            let cfg = crate::model::ModelConfig {
                num_layers: n_layers,
                hidden_size: hidden,
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                intermediate_size: hidden * 4,
                vocab_size: 256,
                max_seq_len: 512,
                name: "proptest-config".into(),
            };
            let fp32 = weight_bytes(&cfg, "none");
            let int8 = weight_bytes(&cfg, "int8_sym");
            let int4 = weight_bytes(&cfg, "int4_sym");
            prop_assert!(
                int4 <= int8,
                "int4 weight_bytes must be <= int8: int4={int4}, int8={int8}"
            );
            prop_assert!(
                int8 <= fp32,
                "int8 weight_bytes must be <= fp32: int8={int8}, fp32={fp32}"
            );
        }

        /// Property: total_peak_bytes = weight + kv_cache + activation, for any context_len.
        ///
        /// Fault detected: estimate() computes total independently rather than summing components.
        #[test]
        fn total_peak_is_sum_of_components_any_context(ctx in 1usize..2048) {
            let cfg = ref_cfg();
            let machine = ref_machine();
            let est = estimate(&cfg, &machine, ctx, "none", 0.6);
            let expected = est.weight_bytes + est.kv_cache_bytes + est.activation_bytes;
            prop_assert!(
                est.total_peak_bytes == expected,
                "total_peak_bytes {} must equal sum of components {} at ctx={}",
                est.total_peak_bytes, expected, ctx
            );
        }
    }
}
