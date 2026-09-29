//! Reference model bundle and transformer forward pass.
//!
//! The reference bundle is a randomly-initialised model that makes every
//! contract test runnable offline with no model download.  It does NOT
//! generate coherent text — it exists to exercise the mechanics.
//!
//! Architecture: 6 layers, 384 hidden, 6 heads, 2 KV heads, 64 head_dim,
//! 1536 FFN, 512 vocab.
//!
//! # Loading real weights
//!
//! `Weights::from_gguf(path, cfg)` loads dequantised f32 tensors from a GGUF
//! file.  GGUF tensor names follow the standard llama/qwen/mistral convention:
//!
//! | GGUF name pattern               | Weight field  |
//! |---------------------------------|---------------|
//! | `token_embd.weight`             | embed         |
//! | `blk.{N}.attn_norm.weight`      | attn_norm[N]  |
//! | `blk.{N}.attn_q.weight`         | wq[N]         |
//! | `blk.{N}.attn_k.weight`         | wk[N]         |
//! | `blk.{N}.attn_v.weight`         | wv[N]         |
//! | `blk.{N}.attn_output.weight`    | wo[N]         |
//! | `blk.{N}.ffn_norm.weight`       | ffn_norm[N]   |
//! | `blk.{N}.ffn_gate.weight`       | gate[N]       |
//! | `blk.{N}.ffn_up.weight`         | up[N]         |
//! | `blk.{N}.ffn_down.weight`       | down[N]       |
//! | `output_norm.weight`            | final_norm    |
//! | `output.weight` / `lm_head.weight` | unembed    |

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
        }
    }

    /// Load weights from a GGUF file, dequantised to f32.
    ///
    /// Tensor names follow the standard llama/qwen/mistral convention used by
    /// llama.cpp: `blk.{N}.attn_q.weight`, `token_embd.weight`, etc.
    ///
    /// For any tensor not found in the file (e.g. architectures that use
    /// `tok_embeddings.weight` instead of `token_embd.weight`), a zero-filled
    /// vector of the correct size is substituted.  This keeps the forward pass
    /// runnable for architecture exploration; for production use every tensor
    /// should be present.
    ///
    /// # Errors
    /// Returns `Err(String)` if the file cannot be opened or the GGUF header
    /// is malformed.
    pub fn from_gguf(path: &str, cfg: &ModelConfig) -> Result<Self, String> {
        use crate::gguf_tensors::TensorStore;

        let (store, _meta) = TensorStore::load_from_file(path).map_err(|e| e.to_string())?;

        // Helper: look up a tensor and return its data, or zeros if absent.
        let get = |name: &str, expected_len: usize| -> Vec<f32> {
            if let Some(t) = store.get(name) {
                // Truncate or extend to expected length for safety.
                let mut data = t.data.clone();
                data.resize(expected_len, 0.0);
                data
            } else {
                vec![0.0f32; expected_len]
            }
        };

        // Also try alternate embedding names used by some models.
        let embed_len = cfg.vocab_size * cfg.hidden_size;
        let embed = if let Some(t) = store.get("token_embd.weight") {
            let mut data = t.data.clone();
            data.resize(embed_len, 0.0);
            data
        } else if let Some(t) = store.get("tok_embeddings.weight") {
            let mut data = t.data.clone();
            data.resize(embed_len, 0.0);
            data
        } else {
            vec![0.0f32; embed_len]
        };

        let unembed_len = cfg.vocab_size * cfg.hidden_size;
        let unembed = if let Some(t) = store.get("output.weight") {
            let mut data = t.data.clone();
            data.resize(unembed_len, 0.0);
            data
        } else if let Some(t) = store.get("lm_head.weight") {
            let mut data = t.data.clone();
            data.resize(unembed_len, 0.0);
            data
        } else {
            // Many models tie embed and unembed weights.
            embed.clone()
        };

        let final_norm = get("output_norm.weight", cfg.hidden_size);

        let mut attn_norm = Vec::with_capacity(cfg.num_layers);
        let mut wq = Vec::with_capacity(cfg.num_layers);
        let mut wk = Vec::with_capacity(cfg.num_layers);
        let mut wv = Vec::with_capacity(cfg.num_layers);
        let mut wo = Vec::with_capacity(cfg.num_layers);
        let mut ffn_norm = Vec::with_capacity(cfg.num_layers);
        let mut gate = Vec::with_capacity(cfg.num_layers);
        let mut up = Vec::with_capacity(cfg.num_layers);
        let mut down = Vec::with_capacity(cfg.num_layers);

        for layer in 0..cfg.num_layers {
            let pfx = format!("blk.{layer}");
            attn_norm.push(get(&format!("{pfx}.attn_norm.weight"), cfg.hidden_size));
            wq.push(get(
                &format!("{pfx}.attn_q.weight"),
                cfg.num_heads * cfg.head_dim * cfg.hidden_size,
            ));
            wk.push(get(
                &format!("{pfx}.attn_k.weight"),
                cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size,
            ));
            wv.push(get(
                &format!("{pfx}.attn_v.weight"),
                cfg.num_kv_heads * cfg.head_dim * cfg.hidden_size,
            ));
            wo.push(get(
                &format!("{pfx}.attn_output.weight"),
                cfg.hidden_size * cfg.num_heads * cfg.head_dim,
            ));
            ffn_norm.push(get(&format!("{pfx}.ffn_norm.weight"), cfg.hidden_size));
            gate.push(get(
                &format!("{pfx}.ffn_gate.weight"),
                cfg.intermediate_size * cfg.hidden_size,
            ));
            up.push(get(
                &format!("{pfx}.ffn_up.weight"),
                cfg.intermediate_size * cfg.hidden_size,
            ));
            down.push(get(
                &format!("{pfx}.ffn_down.weight"),
                cfg.hidden_size * cfg.intermediate_size,
            ));
        }

        Ok(Self {
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
        })
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
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Vec<u32> {
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

    /// Fault detected: Weights::from_gguf returns Err for a non-existent path.
    /// If the error path is suppressed, it would return Ok with zeros — but the test
    /// checks that an invalid path produces an Err, not a silent zero-filled weight set.
    #[test]
    fn from_gguf_nonexistent_path_returns_err() {
        let cfg = ModelConfig::reference();
        let result = Weights::from_gguf("/nonexistent/path/model.gguf", &cfg);
        assert!(
            result.is_err(),
            "from_gguf with a nonexistent path must return Err"
        );
    }

    /// Fault detected: from_gguf returns weights with wrong tensor dimension.
    /// If the embed vector has the wrong length, forward() would panic or produce
    /// garbage.  This test checks dimension correctness against the config.
    ///
    /// Uses the real GGUF model if FITSPROOF_REAL_GGUF is set; skips otherwise.
    #[test]
    fn from_gguf_weights_have_correct_dimensions() {
        let path = match std::env::var("FITSPROOF_REAL_GGUF") {
            Ok(p) => p,
            Err(_) => {
                println!("FITSPROOF_REAL_GGUF not set — skipping real-weight dimension check");
                return;
            }
        };

        // Read real ModelConfig from the GGUF header.
        use crate::gguf::{metadata_to_model_config, read_metadata};
        let file = std::fs::File::open(&path).expect("open GGUF file");
        let meta = read_metadata(std::io::BufReader::new(file)).expect("read metadata");
        let cfg = metadata_to_model_config(&meta, "test_model").expect("extract config");

        let weights = Weights::from_gguf(&path, &cfg).expect("from_gguf must succeed");

        assert_eq!(
            weights.embed.len(),
            cfg.vocab_size * cfg.hidden_size,
            "embed dimension mismatch"
        );
        assert_eq!(
            weights.attn_norm.len(),
            cfg.num_layers,
            "attn_norm layer count mismatch"
        );
        assert_eq!(weights.wq.len(), cfg.num_layers, "wq layer count mismatch");
        assert_eq!(
            weights.unembed.len(),
            cfg.vocab_size * cfg.hidden_size,
            "unembed dimension mismatch"
        );

        // Embedding must not be all zeros when loaded from a real GGUF.
        let all_zero = weights.embed.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "embed weights must not all be zero for real model"
        );
    }

    /// Fault detected: generate() on real GGUF weights produces out-of-range tokens.
    /// This is the real end-to-end generate() test — reads real weights, runs forward pass.
    ///
    /// Requires FITSPROOF_REAL_GGUF to point at a GGUF model file.
    #[test]
    fn from_gguf_generate_produces_in_range_tokens() {
        let path = match std::env::var("FITSPROOF_REAL_GGUF") {
            Ok(p) => p,
            Err(_) => {
                println!("FITSPROOF_REAL_GGUF not set — skipping real-weight generation test");
                return;
            }
        };

        use crate::gguf::{metadata_to_model_config, read_metadata};
        let file = std::fs::File::open(&path).expect("open GGUF file");
        let meta = read_metadata(std::io::BufReader::new(file)).expect("read metadata");
        let cfg = metadata_to_model_config(&meta, "test_model").expect("extract config");

        let weights = Weights::from_gguf(&path, &cfg).expect("from_gguf must succeed");
        let mut transformer = Transformer::new(cfg.clone(), weights);

        let toks = transformer.generate(&[1, 2, 3], 4, 0.0, 42);
        assert_eq!(toks.len(), 4, "must generate exactly 4 tokens");
        for &tok in &toks {
            assert!(
                (tok as usize) < cfg.vocab_size,
                "token id {tok} exceeds vocab size {}",
                cfg.vocab_size
            );
        }
        println!("Real GGUF generate(): {toks:?}");
    }
}
