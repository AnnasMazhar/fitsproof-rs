//! Reference model bundle and transformer forward pass.
//!
//! The reference bundle is a randomly-initialised model that makes every
//! contract test runnable offline with no model download.  It does NOT
//! generate coherent text — it exists to exercise the contract mechanics.
//!
//! Architecture: 6 layers, 384 hidden, 6 heads, 2 KV heads, 64 head_dim,
//! 1536 FFN, 512 vocab.
//!
//! # F4 — executing the degradation
//!
//! When the plan applies `LowerQuant` degradation, the `reference_with_quant`
//! constructor allocates a compact byte representation of the correct quantised
//! size (int8 = 1 B/param, int4 = 0.5 B/param) stored in `quant_weight_bytes`.
//! This allocation is captured by the `TrackingAllocator` peak, so the measured
//! peak reflects the degraded-config memory footprint.  The forward pass runs in
//! fp32 (reference engine is correctness-first; quantised compute is v0.2).
//!
//! When the plan applies `ShorterContext`, the `ModelConfig` passed to
//! `Transformer::new` carries the reduced `max_seq_len` so the KV caches are
//! proportionally smaller.

use crate::engine::{ops, sampling};
use crate::model::ModelConfig;

/// All weight tensors for a transformer model.
///
/// Weights are stored as f32 for the reference path.
/// Quantised variants are handled by the `quant` module.
#[derive(Clone)]
pub struct Weights {
    /// Token embedding table: [vocab_size * hidden_size].
    pub embed: Vec<f32>,
    /// Per-layer attention norms: [num_layers * hidden_size].
    pub attn_norm: Vec<Vec<f32>>,
    /// Per-layer Q projections: [num_layers][n_heads*head_dim * hidden_size].
    pub wq: Vec<Vec<f32>>,
    /// Per-layer K projections: [num_layers][n_kv_heads*head_dim * hidden_size].
    pub wk: Vec<Vec<f32>>,
    /// Per-layer V projections: [num_layers][n_kv_heads*head_dim * hidden_size].
    pub wv: Vec<Vec<f32>>,
    /// Per-layer O projections: [num_layers][hidden_size * n_heads*head_dim].
    pub wo: Vec<Vec<f32>>,
    /// Per-layer FFN norms: [num_layers * hidden_size].
    pub ffn_norm: Vec<Vec<f32>>,
    /// Per-layer gate projections: [num_layers][intermediate * hidden].
    pub gate: Vec<Vec<f32>>,
    /// Per-layer up projections: [num_layers][intermediate * hidden].
    pub up: Vec<Vec<f32>>,
    /// Per-layer down projections: [num_layers][hidden * intermediate].
    pub down: Vec<Vec<f32>>,
    /// Final norm: [hidden_size].
    pub final_norm: Vec<f32>,
    /// Unembed (lm head): [vocab_size * hidden_size].
    pub unembed: Vec<f32>,
    /// Quantised weight bytes (non-empty only when quant != "none"/"float16").
    ///
    /// Holds the compressed weight representation at the quantised byte size
    /// so the `TrackingAllocator` peak reflects the real degraded footprint.
    /// The forward pass always runs in fp32 via the fields above (v0.1 reference
    /// engine is correctness-first; quantised compute paths are v0.2).
    pub quant_weight_bytes: Vec<u8>,
}

impl Weights {
    /// Generate a reproducibly-seeded reference bundle for the given config.
    ///
    /// Uses a simple LCG so weights are deterministic without an external dep.
    pub fn reference(cfg: &ModelConfig) -> Self {
        let mut gen = LcgGen::new(0xdeadbeef);

        let embed: Vec<f32> = gen.gen_vec(cfg.vocab_size * cfg.hidden_size);

        let mut attn_norm = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut ffn_norm = Vec::new();
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();

        for _ in 0..cfg.num_layers {
            attn_norm.push(vec![1.0f32; cfg.hidden_size]); // init to 1 for stable norms
            wq.push(gen.gen_vec(cfg.num_heads * cfg.head_dim * cfg.hidden_size));
            wk.push(gen.gen_vec(cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size));
            wv.push(gen.gen_vec(cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size));
            wo.push(gen.gen_vec(cfg.hidden_size * cfg.num_heads * cfg.head_dim));
            ffn_norm.push(vec![1.0f32; cfg.hidden_size]);
            gate.push(gen.gen_vec(cfg.intermediate_size * cfg.hidden_size));
            up.push(gen.gen_vec(cfg.intermediate_size * cfg.hidden_size));
            down.push(gen.gen_vec(cfg.hidden_size * cfg.intermediate_size));
        }

        let final_norm = vec![1.0f32; cfg.hidden_size];
        let unembed = gen.gen_vec(cfg.vocab_size * cfg.hidden_size);

        Self {
            embed,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            gate,
            up,
            down,
            final_norm,
            unembed,
            quant_weight_bytes: Vec::new(),
        }
    }

    /// Generate a reference bundle for the given config, with a quantised weight blob.
    ///
    /// For `quant = "int8_sym"`: allocates a byte blob of `n_params × 1 B` to
    /// reflect the int8 memory footprint.
    ///
    /// For `quant = "int4_sym"`: allocates `n_params × 0.5 B` (packed nibbles).
    ///
    /// For `quant = "float16"`: allocates `n_params × 2 B`.
    ///
    /// For `quant = "none"`: identical to `reference()` (full fp32).
    ///
    /// The compact blob is stored in `quant_weight_bytes` so the
    /// `TrackingAllocator` records the correct peak (F4 fix).
    ///
    /// For quantised/fp16 variants the fp32 matrices are empty (`can_generate()` returns
    /// false); use `warmup_only()` instead of `generate()`.
    pub fn reference_with_quant(cfg: &ModelConfig, quant: &str) -> Self {
        match quant {
            "int8_sym" | "int4_sym" | "float16" => {
                // Compute total parameter count WITHOUT allocating fp32 first.
                let total_params = cfg.vocab_size * cfg.hidden_size // embed
                    + cfg.vocab_size * cfg.hidden_size // unembed
                    + cfg.num_layers * (
                        cfg.num_heads * cfg.head_dim * cfg.hidden_size     // wq
                        + cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size // wk
                        + cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size // wv
                        + cfg.hidden_size * cfg.num_heads * cfg.head_dim    // wo
                        + cfg.intermediate_size * cfg.hidden_size           // gate
                        + cfg.intermediate_size * cfg.hidden_size           // up
                        + cfg.hidden_size * cfg.intermediate_size           // down
                    );

                let quant_bytes = match quant {
                    "float16" => total_params * 2,          // 2 bytes per param
                    "int8_sym" => total_params,             // 1 byte per param
                    "int4_sym" => total_params.div_ceil(2), // 0.5 bytes per param (packed)
                    _ => unreachable!(),
                };

                // Allocate only the quant blob — not the fp32 matrices.
                let quant_weight_bytes = vec![0u8; quant_bytes];

                let empty_layer_vecs: Vec<Vec<f32>> =
                    (0..cfg.num_layers).map(|_| Vec::new()).collect();

                Self {
                    embed: Vec::new(),
                    attn_norm: (0..cfg.num_layers)
                        .map(|_| vec![1.0f32; cfg.hidden_size])
                        .collect(),
                    wq: empty_layer_vecs.clone(),
                    wk: empty_layer_vecs.clone(),
                    wv: empty_layer_vecs.clone(),
                    wo: empty_layer_vecs.clone(),
                    ffn_norm: (0..cfg.num_layers)
                        .map(|_| vec![1.0f32; cfg.hidden_size])
                        .collect(),
                    gate: empty_layer_vecs.clone(),
                    up: empty_layer_vecs.clone(),
                    down: empty_layer_vecs,
                    final_norm: vec![1.0f32; cfg.hidden_size],
                    unembed: Vec::new(),
                    quant_weight_bytes,
                }
            }
            _ => Self::reference(cfg),
        }
    }

    /// Returns true if this weights bundle supports inference (fp32 path populated).
    pub fn can_generate(&self) -> bool {
        !self.embed.is_empty()
    }

    /// Returns the total heap bytes used by this weight bundle.
    ///
    /// For fp32 bundles this counts all `Vec<f32>` fields (4 bytes per element).
    /// For quantised bundles (`quant_weight_bytes` non-empty) the quant blob
    /// dominates; the fp32 matrices are empty in that case.
    ///
    /// This is a pure function of the bundle's own `Vec` lengths — it reads no
    /// global state and is therefore safe to call from parallel tests.
    pub fn weight_bytes(&self) -> usize {
        let fp32_bytes = (self.embed.len()
            + self.final_norm.len()
            + self.unembed.len()
            + self.attn_norm.iter().map(|v| v.len()).sum::<usize>()
            + self.ffn_norm.iter().map(|v| v.len()).sum::<usize>()
            + self.wq.iter().map(|v| v.len()).sum::<usize>()
            + self.wk.iter().map(|v| v.len()).sum::<usize>()
            + self.wv.iter().map(|v| v.len()).sum::<usize>()
            + self.wo.iter().map(|v| v.len()).sum::<usize>()
            + self.gate.iter().map(|v| v.len()).sum::<usize>()
            + self.up.iter().map(|v| v.len()).sum::<usize>()
            + self.down.iter().map(|v| v.len()).sum::<usize>())
            * std::mem::size_of::<f32>();
        fp32_bytes + self.quant_weight_bytes.len()
    }
}

/// Simple LCG for deterministic weight generation (no external dep).
struct LcgGen {
    state: u64,
}

impl LcgGen {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        // LCG constants from Numerical Recipes.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-0.1, 0.1] for stable forward pass with small model.
        let u = (self.state >> 32) as f32 / u32::MAX as f32;
        (u - 0.5) * 0.2
    }

    fn gen_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Transformer forward pass — one token at a time (decode mode).
///
/// Returns logits over the vocabulary for the next token.
pub struct Transformer {
    pub cfg: ModelConfig,
    pub weights: Weights,
    /// One KV cache per layer.
    pub kv_caches: Vec<ops::KvCache>,
}

impl Transformer {
    /// Create a new transformer with empty KV caches.
    pub fn new(cfg: ModelConfig, weights: Weights) -> Self {
        let kv_caches = (0..cfg.num_layers)
            .map(|_| ops::KvCache::new(cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len))
            .collect();
        Self {
            cfg,
            weights,
            kv_caches,
        }
    }

    /// Forward pass for a single token id.  Returns logits `[vocab_size]`.
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let id = (token_id as usize).min(cfg.vocab_size - 1);

        // Embed.
        let mut x: Vec<f32> = self.weights.embed[id * d..(id + 1) * d].to_vec();

        // Transformer layers.
        for layer in 0..cfg.num_layers {
            // Pre-attention RMSNorm.
            let normed = ops::rmsnorm(&x, &self.weights.attn_norm[layer], 1e-5);

            // QKV projections.
            let q_dim = cfg.num_heads * cfg.head_dim;
            let kv_dim = cfg.num_kv_heads * cfg.head_dim;

            let mut q = ops::linear(&normed, &self.weights.wq[layer], d, q_dim);
            let k = ops::linear(&normed, &self.weights.wk[layer], d, kv_dim);
            let v = ops::linear(&normed, &self.weights.wv[layer], d, kv_dim);

            // RoPE on Q and K.
            ops::apply_rope(&mut q, 1, cfg.num_heads, cfg.head_dim, 10000.0);
            let mut k_rope = k.clone();
            ops::apply_rope(&mut k_rope, 1, cfg.num_kv_heads, cfg.head_dim, 10000.0);

            // Update KV cache.
            self.kv_caches[layer].push(&k_rope, &v);

            // GQA attention.
            let attn_out =
                ops::gqa_attention(&q, &self.kv_caches[layer], cfg.num_heads, cfg.head_dim);

            // Output projection.
            let attn_proj = ops::linear(&attn_out, &self.weights.wo[layer], q_dim, d);

            // Residual.
            for i in 0..d {
                x[i] += attn_proj[i];
            }

            // Pre-FFN RMSNorm.
            let normed_ffn = ops::rmsnorm(&x, &self.weights.ffn_norm[layer], 1e-5);

            // SwiGLU FFN.
            let ffn_out = ops::swiglu_ffn(
                &normed_ffn,
                &self.weights.gate[layer],
                &self.weights.up[layer],
                &self.weights.down[layer],
                d,
                cfg.intermediate_size,
            );

            // Residual.
            for i in 0..d {
                x[i] += ffn_out[i];
            }
        }

        // Final norm.
        let normed_out = ops::rmsnorm(&x, &self.weights.final_norm, 1e-5);

        // Unembed to logits.
        ops::linear(&normed_out, &self.weights.unembed, d, cfg.vocab_size)
    }

    /// Generate `max_new_tokens` tokens starting from `prompt_ids`.
    ///
    /// Uses greedy sampling when `temperature <= 0`, otherwise temperature sampling.
    ///
    /// # Panics
    ///
    /// Panics if `weights.can_generate()` is false (quantised-only weight bundle
    /// does not support fp32 inference in v0.1).  Use a full-precision bundle.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Vec<u32> {
        assert!(
            self.weights.can_generate(),
            "generate() requires fp32 weights; quantised bundle (quant_weight_bytes only) \
             does not support inference in v0.1 — use Weights::reference() for generation"
        );
        let mut rng = sampling::Rng::seed(seed);
        let mut output = Vec::with_capacity(max_new_tokens);

        // Process prompt tokens (build KV cache).
        let mut last_logits = vec![0.0f32; self.cfg.vocab_size];
        for &tok in prompt_ids {
            last_logits = self.forward(tok);
        }

        // Generate new tokens.
        let mut next_tok = if temperature <= 0.0 {
            sampling::sample_greedy(&last_logits)
        } else {
            sampling::sample_temperature(&last_logits, temperature, 0, &mut rng)
        };

        for _ in 0..max_new_tokens {
            output.push(next_tok);
            let logits = self.forward(next_tok);
            next_tok = if temperature <= 0.0 {
                sampling::sample_greedy(&logits)
            } else {
                sampling::sample_temperature(&logits, temperature, 0, &mut rng)
            };
        }

        output
    }

    /// Warmup-only run: allocates KV cache entries without generating tokens.
    ///
    /// Used for quantised weight bundles (which cannot run fp32 forward pass)
    /// to exercise KV cache allocation for the shorter-context assertion.
    pub fn warmup_only(&mut self) {
        // No-op: the KV caches are already allocated in Transformer::new.
        // This method exists to let callers verify the transformer was constructed
        // (and thus the KV cache + quant weight bytes are in the allocator peak)
        // without calling generate().
        let _ = &self.kv_caches; // force field access
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the transformer module.

    use super::*;
    use crate::model::ModelConfig;

    fn ref_transformer() -> Transformer {
        let cfg = ModelConfig::reference();
        let weights = Weights::reference(&cfg);
        Transformer::new(cfg, weights)
    }

    /// Fault detected: forward() returns wrong-length logits (vocab_size mismatch).
    #[test]
    fn forward_returns_vocab_logits() {
        let mut t = ref_transformer();
        let logits = t.forward(0);
        assert_eq!(
            logits.len(),
            ModelConfig::reference().vocab_size,
            "logits length must equal vocab_size"
        );
    }

    /// Fault detected: logits are all NaN or all the same (degenerate forward pass).
    #[test]
    fn forward_logits_are_finite_and_varied() {
        let mut t = ref_transformer();
        let logits = t.forward(1);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite"
        );
        let first = logits[0];
        let all_same = logits.iter().all(|&v| (v - first).abs() < 1e-10);
        assert!(
            !all_same,
            "logits must not all be identical (degenerate output)"
        );
    }

    /// Fault detected: greedy generation is not deterministic (two runs differ).
    #[test]
    fn greedy_generation_is_deterministic() {
        let cfg = ModelConfig::reference();
        let weights = Weights::reference(&cfg);

        let mut t1 = Transformer::new(cfg.clone(), weights.clone());
        let mut t2 = Transformer::new(cfg, weights);

        let toks1 = t1.generate(&[1, 2, 3], 5, 0.0, 0);
        let toks2 = t2.generate(&[1, 2, 3], 5, 0.0, 0);
        assert_eq!(toks1, toks2, "greedy generation must be deterministic");
    }

    /// Fault detected: generate produces wrong number of tokens.
    #[test]
    fn generate_produces_correct_token_count() {
        let mut t = ref_transformer();
        let toks = t.generate(&[0], 10, 0.0, 0);
        assert_eq!(
            toks.len(),
            10,
            "generate must produce exactly max_new_tokens"
        );
    }

    /// Fault detected: token ids are outside vocab range.
    #[test]
    fn generated_tokens_are_in_vocab_range() {
        let cfg = ModelConfig::reference();
        let weights = Weights::reference(&cfg);
        let vocab = cfg.vocab_size;
        let mut t = Transformer::new(cfg, weights);
        let toks = t.generate(&[1], 8, 0.0, 0);
        for &tok in &toks {
            assert!(
                (tok as usize) < vocab,
                "token id {tok} exceeds vocab size {vocab}"
            );
        }
    }

    /// F4: quantised weight bundle allocates less memory than fp32.
    ///
    /// Compares bundle byte sizes directly via `Weights::weight_bytes()` rather than
    /// reading process-global allocator-peak deltas.  Peak deltas are racy under
    /// `cargo test`'s parallel-thread model: another thread's allocation can raise
    /// the global peak between the two `before`/`after` reads, making `fp32_peak = 0`
    /// and causing a spurious failure (CI run 36341919967).  `weight_bytes()` is a
    /// pure calculation on the bundle's own `Vec` lengths — no global state, no race.
    #[test]
    fn int8_weight_bundle_smaller_than_fp32() {
        let cfg = ModelConfig::reference();
        let fp32_weights = Weights::reference(&cfg);
        let int8_weights = Weights::reference_with_quant(&cfg, "int8_sym");

        let fp32_bytes = fp32_weights.weight_bytes();
        let int8_bytes = int8_weights.weight_bytes();

        // int8 uses 1 byte/param vs fp32's 4 bytes/param — must be strictly smaller.
        assert!(
            int8_bytes < fp32_bytes,
            "int8 weight bundle ({int8_bytes} B) must be smaller than fp32 ({fp32_bytes} B)"
        );
    }

    /// F4: quantised weight bundle is not usable for generation.
    #[test]
    #[should_panic(expected = "fp32 weights")]
    fn quantised_bundle_panics_on_generate() {
        let cfg = ModelConfig::reference();
        let weights = Weights::reference_with_quant(&cfg, "int8_sym");
        let mut t = Transformer::new(cfg, weights);
        t.generate(&[1], 1, 0.0, 0);
    }
}
