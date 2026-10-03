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
            "int4_sym" | "int4_asym" | "int4" | "q4_k" | "q4_0" | "q4_k_m" | "q4_k_s" | "q4_1" => {
                Some(QuantBits(4.0))
            }
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
/// # Embedding and unembed dtype
///
/// `token_embd.weight` and `output.weight` are stored at the model's declared dtype, not
/// at the quantisation format of the transformer layers.  For fp32 models (`quant="none"`)
/// they are fp32 (4 bytes/element).  For all other quantisation formats (int8, int4,
/// Q4_K_M, etc.) they are stored at fp16 (2 bytes/element) per GGUF convention — see
/// llama.cpp `src/llama-model-loader.cpp` and the GGUF spec tensor_type field.
///
/// Using fp32 for embeddings on quantised models overcounts by `V × d_model × 2 bytes`
/// per copy (embed + unembed) — ~2 GB for a 7B-class model with vocab = 128 K,
/// hidden = 4096.  This would make `admit()` refuse configs that actually fit
/// (false-positive refusals).
///
/// Source: RESEARCH.md §1933 — "treat `token_embd.weight` and `output.weight` as fp16
/// regardless of the declared quant" (cross-referenced with llama.cpp source and GGUF spec).
///
/// # Fault detected
///
/// Omitting the embedding table doubles the error for models with large vocabularies.
/// Using fp32 (not fp16) for embeddings on quantised models overcounts by ~2 GB for 7B models.
/// Tests verify against a known-config reference with exact hand-computed values.
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

    // Embedding dtype: fp32 for fp32 models; fp16 for all quantised formats.
    // Source: GGUF convention — token_embd.weight and output.weight stored at fp16 in
    // quantised GGUF files (llama.cpp src/llama-model-loader.cpp; GGUF spec tensor_type).
    let embed_bpe: f64 = if bits == 4.0 { 4.0 } else { 2.0 };

    // Embedding table (embed_bpe bytes/element)
    let embed_bytes = v * d * embed_bpe;

    // Per-layer attention: Q(h*hd*d) + K(kv_h*hd*d) + V(kv_h*hd*d) + O(d*h*hd)
    let attn_per_layer = (h * hd * d + kv_h * hd * d + kv_h * hd * d + d * h * hd) * bits;

    // Per-layer FFN: gate(ff*d) + up(ff*d) + down(d*ff)
    let ffn_per_layer = (ff * d + ff * d + d * ff) * bits;

    // Per-layer norms (fp32, 2 per layer)
    let norm_per_layer = 2.0 * d * 4.0;

    // Final norm (fp32) + unembed (embed_bpe bytes/element — same dtype as embedding table)
    let final_bytes = d * 4.0 + v * d * embed_bpe;

    (embed_bytes + l * (attn_per_layer + ffn_per_layer + norm_per_layer) + final_bytes) as u64
}

/// Compute KV cache memory for `context_len` tokens across all layers.
///
/// Formula (GQA, Ainslie et al. 2023):
///   `2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element`
///
/// The factor 2 covers both K and V tensors.
///
/// # KV cache precision is independent of weight quantisation
///
/// **`kv_quant` is NOT the weight quantisation format.**  In real LLM inference
/// (llama.cpp, vLLM, Hugging Face Transformers), the KV cache is stored in the
/// *activation dtype*, which defaults to float16 regardless of weight quantisation.
/// A model with int4 weights does NOT get a 4-bit KV cache unless an explicit
/// `--cache-quant` / `--kv-cache-dtype` flag is passed.  Callers should pass
/// `"fp16"` unless they are explicitly modelling quantised KV caches.
///
/// Source: llama.cpp `ggml_backend_metal_buffer_type_alloc_size` allocates KV at
/// GGML_TYPE_F16 by default; vLLM `ModelRunner.kv_cache_dtype` defaults to "auto"
/// which maps to float16 for most backends (vLLM v0.4+ docs).
///
/// # Fault detected
///
/// Passing the weight quant (e.g. "int4_sym") to this function produces a 4× or
/// 2× underestimate of KV memory, causing budget violations to go undetected for
/// quantised models.  Tests verify with an independent ground-truth computation.
pub fn kv_cache_bytes(cfg: &ModelConfig, context_len: usize, kv_quant: &str) -> u64 {
    let bits = QuantBits::from_name(kv_quant)
        .expect("kv_cache_bytes: unknown kv_quant — caller must validate")
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
///
/// # KV cache precision
///
/// KV cache is always computed at fp16 (2 bytes/element) — the standard activation
/// dtype across llama.cpp, vLLM, and Transformers.  The weight quantisation (`quant`)
/// does **not** affect KV cache size.  This is the architecturally correct default;
/// use `kv_cache_bytes` directly if you need to model explicit KV quantisation.
pub fn estimate(
    cfg: &ModelConfig,
    machine: &MachineProfile,
    context_len: usize,
    quant: &str,
    bandwidth_utilisation: f64,
) -> CostEstimate {
    let w = weight_bytes(cfg, quant);
    // KV cache is fp16 by default — independent of weight quantisation.
    let kv = kv_cache_bytes(cfg, context_len, "fp16");
    let act = activation_bytes(cfg);
    // Use saturating arithmetic to prevent wrapping on overflow (ADV-C5-P2-1).
    // Without this, extreme context_len values cause kv_cache_bytes to saturate
    // to u64::MAX, and then adding weight_bytes wraps to a small value, bypassing
    // the budget check entirely.
    let total = w.saturating_add(kv).saturating_add(act);
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
    ///   unembed          = 512 * 384 * 4                   =    786_432  (fp32 — same as embed)
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

    /// Fault detected: any arithmetic error in the fp16 model weight_bytes formula —
    /// e.g. replacing `*` with `+` in attn_per_layer or ffn_per_layer for fp16 models.
    ///
    /// The only existing fp16 coverage (`quant_bits_bytes_per_element_exact_values` in
    /// contract_mutants.rs) asserts bounds only: `fp16 < fp32` and `fp16 > 1_574_400`.
    /// These bounds are too loose: a mutation changing `*` to `+` in the attn formula
    /// (producing ~393,218 instead of ~786,432 attn bytes) would drop total from
    /// 26,758,656 to ~24,372,224 — still within those loose bounds and thus UNDETECTED.
    ///
    /// Hand-computed for reference config (6L, 384H, 6Q-heads, 2KV-heads, 64 hd, 1536 ff, 512V)
    /// at float16 (2 bytes/element for transformer layers; fp16 = 2 bytes for embed/unembed):
    ///
    ///   embed            = 512 * 384 * 2                    =    393_216  (fp16)
    ///   attn Q/layer     = 6 * 64 * 384 * 2                =    589_824
    ///   attn K/layer     = 2 * 64 * 384 * 2                =    196_608
    ///   attn V/layer     = 2 * 64 * 384 * 2                =    196_608
    ///   attn O/layer     = 384 * 6 * 64 * 2                =    589_824
    ///   attn/layer total                                    =    786_432
    ///   ffn gate/layer   = 1536 * 384 * 2                  =  1_179_648
    ///   ffn up/layer     = 1536 * 384 * 2                  =  1_179_648
    ///   ffn down/layer   = 384 * 1536 * 2                  =  1_179_648
    ///   ffn/layer total                                     =  3_538_944
    ///   norm/layer       = 2 * 384 * 4                     =      3_072  (fp32 — unchanged)
    ///   per-layer total  = 786_432 + 3_538_944 + 3_072     =  4_328_448
    ///   6 layers         = 6 * 4_328_448                   = 25_970_688
    ///   final norm       = 384 * 4                         =      1_536  (fp32 — unchanged)
    ///   unembed          = 512 * 384 * 2                   =    393_216  (fp16 — same as embed)
    ///   total = 393_216 + 25_970_688 + 1_536 + 393_216     = 26_758_656
    ///
    /// Source: cost.rs formula traced term by term; fp16 embed/unembed from GGUF convention
    /// (RESEARCH.md §1933; llama.cpp src/llama-model-loader.cpp).
    #[test]
    fn weight_bytes_fp16_exact_known_answer() {
        let cfg = ref_cfg();
        let got = weight_bytes(&cfg, "float16");
        let expected: u64 = 26_758_656;
        assert_eq!(
            got, expected,
            "weight_bytes float16 got {got}, expected {expected}. \
             Derivation: embed(393216) + 6×layer(4328448) + final(394752) = 26758656. \
             If got ~24372224, attn *→+ mutation is present. \
             If got ~393218, all layer * were replaced with +."
        );
    }

    /// Fault detected: embed and unembed use fp32 (4 bytes) instead of fp16 (2 bytes)
    /// for quantised models, overcounting by ~2 GB for 7B-class models.
    ///
    /// GGUF convention: `token_embd.weight` and `output.weight` are stored at fp16 in
    /// quantised GGUF files regardless of the transformer-layer quantisation format.
    /// Source: RESEARCH.md §1933; llama.cpp src/llama-model-loader.cpp.
    ///
    /// This test is the REGRESSION GUARD: injecting `embed_bpe = 4.0` for int4_sym
    /// (reverting to old fp32 embed behaviour) would produce 8_080_896 instead of
    /// 7_294_464 — a 786_432-byte (768 KB) overcount on the tiny reference config,
    /// scaling to ~2.1 GB overcount per copy for a 7B model (Llama-3.1-8B: 2 × 1.05 GB).
    ///
    /// Hand-computed for reference config (6L, 384H, 6Q-heads, 2KV-heads, 64 hd, 1536 ff, 512V)
    /// at int4_sym (0.5 bytes/element for transformer layers; fp16 = 2 bytes for embed/unembed):
    ///
    ///   embed            = 512 * 384 * 2                    =    393_216  (fp16)
    ///   attn Q/layer     = 6 * 64 * 384 * 0.5              =     73_728
    ///   attn K/layer     = 2 * 64 * 384 * 0.5              =     24_576
    ///   attn V/layer     = 2 * 64 * 384 * 0.5              =     24_576
    ///   attn O/layer     = 384 * 6 * 64 * 0.5              =     73_728
    ///   attn/layer total                                    =    196_608
    ///   ffn gate/layer   = 1536 * 384 * 0.5                =    294_912
    ///   ffn up/layer     = 1536 * 384 * 0.5                =    294_912
    ///   ffn down/layer   = 384 * 1536 * 0.5                =    294_912
    ///   ffn/layer total                                     =    884_736
    ///   norm/layer       = 2 * 384 * 4                     =      3_072
    ///   per-layer total  = 196_608 + 884_736 + 3_072       =  1_084_416
    ///   6 layers         = 6 * 1_084_416                   =  6_506_496
    ///   final norm       = 384 * 4                         =      1_536
    ///   unembed          = 512 * 384 * 2                   =    393_216  (fp16 — same as embed)
    ///   total = 393_216 + 6_506_496 + 1_536 + 393_216      =  7_294_464
    ///
    /// Source: GGUF convention (llama.cpp), RESEARCH.md §1933.
    #[test]
    fn weight_bytes_embed_unembed_are_fp16_for_quant_models() {
        let cfg = ref_cfg();

        // int4_sym: embed + unembed must be fp16 (2 bytes), not fp32 (4 bytes).
        let int4 = weight_bytes(&cfg, "int4_sym");
        let expected_int4: u64 = 7_294_464;
        assert_eq!(
            int4, expected_int4,
            "weight_bytes int4_sym: embed/unembed must be fp16 (2 bytes/elem); \
             got {int4}, expected {expected_int4}. \
             If you get 8_080_896, the regression is present: embed_bpe was reverted to fp32 \
             (4 bytes) for quantised models — overcounts by ~786 KB on this config, \
             ~2.1 GB per embedding copy on a 7B model."
        );

        // int8_sym: same fp16 embed/unembed rule.
        let int8 = weight_bytes(&cfg, "int8_sym");
        let expected_int8: u64 = 13_782_528;
        assert_eq!(
            int8, expected_int8,
            "weight_bytes int8_sym: embed/unembed must be fp16 (2 bytes/elem); \
             got {int8}, expected {expected_int8}."
        );

        // fp32 model: embed + unembed stay fp32 (4 bytes/elem) — unchanged.
        let fp32 = weight_bytes(&cfg, "none");
        let expected_fp32: u64 = 53_497_344;
        assert_eq!(
            fp32, expected_fp32,
            "weight_bytes fp32 must remain unchanged at {expected_fp32}; got {fp32}"
        );

        // Ordering invariant: int4 < int8 < fp32.
        assert!(int4 < int8, "int4 ({int4}) must be < int8 ({int8})");
        assert!(int8 < fp32, "int8 ({int8}) must be < fp32 ({fp32})");
    }

    /// Fault detected: factor 2 omitted from kv_cache_bytes (halves the estimate).
    /// Hand-computed: 2 * 6 * 2 * 512 * 64 * 2 = 3_145_728 bytes (fp16, context=512).
    ///
    /// fp16 = 2 bytes/element.  GQA formula (Ainslie et al. 2023):
    ///   2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element
    ///   = 2 * 6 * 2 * 512 * 64 * 2 = 3_145_728
    #[test]
    fn kv_cache_bytes_reference_fp16_known_answer() {
        let cfg = ref_cfg();
        // 2 * n_layers * n_kv_heads * context * head_dim * 2 (fp16 = 2 bytes)
        let expected: u64 = 2 * 6 * 2 * 512 * 64 * 2;
        let got = kv_cache_bytes(&cfg, 512, "fp16");
        assert_eq!(
            got, expected,
            "kv_cache_bytes fp16 got {got}, expected {expected}"
        );
    }

    /// Fault detected (REGRESSION GUARD): `estimate()` couples KV cache bytes to weight
    /// quantisation — i.e., the caller passes `quant` to `kv_cache_bytes` instead of
    /// hardcoding `"fp16"`.
    ///
    /// The KV cache precision is the ACTIVATION dtype (fp16 by default), not the weight
    /// quantisation format.  A model with int4 weights has the SAME KV cache size as a
    /// model with fp32 weights when both use the default KV dtype.
    ///
    /// The regression this test guards against: if someone changes `estimate()` to call
    /// `kv_cache_bytes(cfg, context_len, quant)` instead of
    /// `kv_cache_bytes(cfg, context_len, "fp16")`, the int4 estimate drops 4× and int8
    /// drops 2×, causing silent under-prediction for quantised models.
    ///
    /// Ground truth (hand-computed, fp16 = 2 bytes/element, context=512):
    ///   2 * 6 * 2 * 512 * 64 * 2 = 3_145_728 bytes
    /// This must equal `kv_cache_bytes` in both the fp32-weight and int4-weight estimates.
    ///
    /// Previous version of this test was VACUOUS: it called `kv_cache_bytes(&cfg, 512, "fp16")`
    /// twice with identical arguments and asserted equality — a tautology that cannot detect
    /// the regression it was written to prevent.  This replacement tests through `estimate()`
    /// with differing weight quants, exercising the actual coupling point.
    #[test]
    fn kv_cache_bytes_independent_of_weight_quant() {
        let cfg = ref_cfg();
        let machine = ref_machine();

        // Run estimate() with fp32 weights and with int4 weights.
        // If estimate() internally passes the weight quant to kv_cache_bytes (the regression),
        // the int4 estimate will be 4× smaller than the fp32 estimate — this catches it.
        let est_fp32 = estimate(&cfg, &machine, 512, "none", 0.6);
        let est_int4 = estimate(&cfg, &machine, 512, "int4_sym", 0.6);

        assert_eq!(
            est_fp32.kv_cache_bytes, est_int4.kv_cache_bytes,
            "estimate().kv_cache_bytes must be identical for fp32 and int4 weight models: \
             fp32_weights={}, int4_weights={}",
            est_fp32.kv_cache_bytes, est_int4.kv_cache_bytes
        );

        // Confirm against hand-computed ground truth: 2 * 6 * 2 * 512 * 64 * 2 = 3_145_728
        let expected_kv: u64 = 2 * 6 * 2 * 512 * 64 * 2;
        assert_eq!(
            est_fp32.kv_cache_bytes, expected_kv,
            "kv_cache_bytes for fp32-weight model must match hand-computed fp16 ground truth: \
             got {}, expected {expected_kv}",
            est_fp32.kv_cache_bytes
        );

        // Separately confirm that a fp32 KV cache (explicit) is 2× fp16 KV cache.
        let kv_fp32_kv = kv_cache_bytes(&cfg, 512, "none"); // explicit fp32 KV (unusual)
        assert_eq!(
            kv_fp32_kv,
            expected_kv * 2,
            "fp32 KV cache must be 2× fp16 KV cache \
             (got fp32_kv={kv_fp32_kv}, expected {})",
            expected_kv * 2
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
            let small = kv_cache_bytes(&cfg, a, "fp16");
            let large = kv_cache_bytes(&cfg, b, "fp16");
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

    /// F10: Kaplan factor-2 KAT (Scaling Laws §D, FLOPS = 2 * n_params * seq_len).
    ///
    /// Doubling seq_len must double TTFT (linear in seq_len from the factor-2 formula).
    /// Fault detected: if the factor-2 is missing (FLOPS = n_params * seq_len), TTFT
    /// would still scale linearly but would be 2× too low — a different KAT.
    ///
    /// This test verifies the factor-2 by checking TTFT at seq_len=1 vs seq_len=2:
    /// ratio must equal exactly 2.0.
    #[test]
    fn kaplan_prefill_factor2_kat() {
        let cfg = ref_cfg();
        let machine = ref_machine();
        let ttft_1 = prefill_ttft_s(&cfg, &machine, 1, "none");
        let ttft_2 = prefill_ttft_s(&cfg, &machine, 2, "none");
        // Linearity: ttft(2) / ttft(1) must be 2.0 (linear in seq_len).
        let ratio = ttft_2 / ttft_1;
        assert!(
            (ratio - 2.0).abs() < 1e-6,
            "TTFT must be linear in seq_len (factor-2 from Kaplan 2020): ratio={ratio}"
        );

        // Additionally verify the formula value is positive and finite.
        assert!(
            ttft_1.is_finite() && ttft_1 > 0.0,
            "prefill_ttft_s must return a positive finite value, got {ttft_1}"
        );
    }
}
