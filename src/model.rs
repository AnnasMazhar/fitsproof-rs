//! Model configuration — shape parameters shared across contract and engine.
//!
//! `ModelConfig` is the authoritative description of a model's architecture.
//! It carries enough information to compute weight bytes, KV cache bytes,
//! and activation estimates without loading actual weights.

use serde::{Deserialize, Serialize};

/// Architecture parameters for a transformer model.
///
/// Field names follow the HuggingFace / GGUF conventions where applicable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Total number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension (d_model).
    pub hidden_size: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (= num_heads for MHA; < num_heads for GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension. Typically hidden_size / num_heads.
    pub head_dim: usize,
    /// FFN intermediate dimension (SwiGLU uses gate+up projections of this width).
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Human-readable name (e.g. "tinystories-reference").
    pub name: String,
}

impl ModelConfig {
    /// An in-repo, randomly-initialised reference configuration.
    ///
    /// Used by all contract tests and stress tests so they run offline with
    /// no model download. This config does NOT generate coherent text;
    /// it exists to exercise the contract mechanics.
    ///
    /// Architecture: 6 layers, 384 hidden, 6 heads, 2 KV heads, 64 head_dim,
    /// 1536 FFN, 512 vocab — ~38 MB at fp32.
    pub fn reference() -> Self {
        Self {
            num_layers: 6,
            hidden_size: 384,
            num_heads: 6,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 1536,
            vocab_size: 512,
            max_seq_len: 512,
            name: "reference-6L-384H".to_string(),
        }
    }

    /// A "small" 3B-class config for stress-test coverage.
    ///
    /// Approximates a 3B model (Llama-style GQA).
    pub fn small_3b() -> Self {
        Self {
            num_layers: 28,
            hidden_size: 3072,
            num_heads: 24,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 8192,
            vocab_size: 32000,
            max_seq_len: 4096,
            name: "small-3b-class".to_string(),
        }
    }
}
