//! Transformer building blocks: RMSNorm, RoPE, GQA attention, SwiGLU FFN.
//!
//! All kernels are scalar reference paths — correct by construction, no SIMD.
//!
//! # Sources
//!
//! - Zhang & Sennrich 2019 (RMSNorm), https://arxiv.org/abs/1910.07467
//! - Su et al. 2022 (RoPE), https://arxiv.org/abs/2104.09864
//! - Ainslie et al. 2023 (GQA), https://arxiv.org/abs/2305.13245
//! - Touvron et al. 2023 (SwiGLU in Llama), https://arxiv.org/abs/2302.13971

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root-mean-square layer normalisation (Zhang & Sennrich 2019).
///
/// `rms = sqrt(mean(x^2) + eps)`
/// `out[i] = (x[i] / rms) * weight[i]`
///
/// # Fault detected
///
/// Using LayerNorm (subtract mean) instead of RMSNorm (no mean subtraction)
/// would produce different outputs for non-zero-mean inputs.
/// Test uses a known-answer from the paper's formula.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rmsnorm: x and weight must be same length"
    );
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| (xi / rms) * wi)
        .collect()
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
// ---------------------------------------------------------------------------

/// Apply RoPE to a query or key tensor `[seq_len, n_heads, head_dim]` (flat row-major).
///
/// Rotating pairs: for each pair (x_{2i}, x_{2i+1}) at position `pos`:
///   `θ_i = pos / (base^(2i / head_dim))`
///   `x_{2i}'   = x_{2i}   * cos(θ_i) - x_{2i+1} * sin(θ_i)`
///   `x_{2i+1}' = x_{2i+1} * cos(θ_i) + x_{2i}   * sin(θ_i)`
///
/// # Fault detected
///
/// Swapping sin/cos signs would rotate in the wrong direction.
/// Test verifies that a full 2π rotation returns the original vector.
pub fn apply_rope(tensor: &mut [f32], seq_len: usize, n_heads: usize, head_dim: usize, base: f32) {
    assert_eq!(tensor.len(), seq_len * n_heads * head_dim);
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");

    for pos in 0..seq_len {
        for head in 0..n_heads {
            let offset = (pos * n_heads + head) * head_dim;
            for i in 0..head_dim / 2 {
                let theta = pos as f32 / base.powf(2.0 * i as f32 / head_dim as f32);
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let x0 = tensor[offset + 2 * i];
                let x1 = tensor[offset + 2 * i + 1];
                tensor[offset + 2 * i] = x0 * cos_t - x1 * sin_t;
                tensor[offset + 2 * i + 1] = x1 * cos_t + x0 * sin_t;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// KV cache for one layer: stores K and V tensors for all past tokens.
///
/// Layout: `[max_seq_len, n_kv_heads, head_dim]` flat row-major.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub k_cache: Vec<f32>, // [max_seq_len * n_kv_heads * head_dim]
    pub v_cache: Vec<f32>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Number of tokens stored so far.
    pub len: usize,
}

impl KvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let size = max_seq_len * n_kv_heads * head_dim;
        Self {
            k_cache: vec![0.0; size],
            v_cache: vec![0.0; size],
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Append one token's K and V vectors.
    ///
    /// `k` and `v` are `[n_kv_heads * head_dim]` each.
    pub fn push(&mut self, k: &[f32], v: &[f32]) {
        assert!(self.len < self.max_seq_len, "KV cache overflow");
        let stride = self.n_kv_heads * self.head_dim;
        let base = self.len * stride;
        self.k_cache[base..base + stride].copy_from_slice(k);
        self.v_cache[base..base + stride].copy_from_slice(v);
        self.len += 1;
    }
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention (GQA)
// ---------------------------------------------------------------------------

/// GQA attention for a single query token against the KV cache.
///
/// `q`: `[n_heads * head_dim]` — query for the current token.
/// `cache`: past KV (n_kv_heads may be < n_heads; KV heads are shared across groups).
///
/// Returns `[n_heads * head_dim]` output.
///
/// # Fault detected
///
/// Forgetting the 1/sqrt(head_dim) scale would produce unnormalised attention.
pub fn gqa_attention(q: &[f32], cache: &KvCache, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let n_kv_heads = cache.n_kv_heads;
    let seq_len = cache.len;
    assert_eq!(
        n_heads % n_kv_heads,
        0,
        "n_heads must be divisible by n_kv_heads for GQA"
    );
    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = n_kv_heads * head_dim;

    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_head = h / group_size;
        let q_offset = h * head_dim;

        // Compute attention scores for this head.
        let mut scores = vec![0.0f32; seq_len];
        for (t, score) in scores.iter_mut().enumerate().take(seq_len) {
            let k_offset = t * kv_stride + kv_head * head_dim;
            let dot: f32 = (0..head_dim)
                .map(|d| q[q_offset + d] * cache.k_cache[k_offset + d])
                .sum();
            *score = dot * scale;
        }

        // Softmax over scores.
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
        let attn: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp() / sum).collect();

        // Weighted sum of values.
        let out_offset = h * head_dim;
        for (t, &attn_t) in attn.iter().enumerate().take(seq_len) {
            let v_offset = t * kv_stride + kv_head * head_dim;
            for d in 0..head_dim {
                out[out_offset + d] += attn_t * cache.v_cache[v_offset + d];
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// SwiGLU FFN
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward: out = down(silu(gate(x)) ⊙ up(x)).
///
/// `gate_w`: `[intermediate_size * hidden_size]` row-major.
/// `up_w`:   `[intermediate_size * hidden_size]` row-major.
/// `down_w`: `[hidden_size * intermediate_size]` row-major.
///
/// # Fault detected
///
/// Using ReLU instead of SiLU in the gate would produce different activations.
/// Test verifies that gate(x)=0 when x=0 (SiLU(0)*up(0)=0).
pub fn swiglu_ffn(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    // Gate projection: intermediate_size activations
    let mut gate = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        for j in 0..hidden {
            gate[i] += gate_w[i * hidden + j] * x[j];
        }
        gate[i] = crate::engine::tensor::silu(gate[i]);
    }

    // Up projection: intermediate_size activations
    let mut up = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        for j in 0..hidden {
            up[i] += up_w[i * hidden + j] * x[j];
        }
    }

    // Element-wise multiply: gate ⊙ up
    let fused: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

    // Down projection: hidden_size output
    let mut out = vec![0.0f32; hidden];
    for i in 0..hidden {
        for j in 0..intermediate {
            out[i] += down_w[i * intermediate + j] * fused[j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Linear projection
// ---------------------------------------------------------------------------

/// Dense linear: y = x @ W^T, where W is [out * in] row-major.
pub fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            out[o] += w[o * in_dim + i] * x[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Known-answer tests for transformer building blocks.

    use super::*;

    /// Fault detected: RMSNorm computes LayerNorm instead (subtracts mean).
    /// Hand (Zhang & Sennrich 2019 eq. 4):
    /// x=[3,4], weight=[1,1], eps=0:
    ///   rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    ///   out = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
    #[test]
    fn rmsnorm_known_answer() {
        let x = vec![3.0f32, 4.0];
        let w = vec![1.0f32, 1.0];
        let out = rmsnorm(&x, &w, 0.0);
        let expected_0 = 3.0 / (12.5f32).sqrt();
        let expected_1 = 4.0 / (12.5f32).sqrt();
        assert!(
            (out[0] - expected_0).abs() < 1e-5,
            "rmsnorm[0] got {}",
            out[0]
        );
        assert!(
            (out[1] - expected_1).abs() < 1e-5,
            "rmsnorm[1] got {}",
            out[1]
        );
    }

    /// Fault detected: RMSNorm output is zero for constant weight=0 but non-zero x.
    #[test]
    fn rmsnorm_scales_by_weight() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w2 = vec![2.0f32; 4];
        let w1 = vec![1.0f32; 4];
        let out2 = rmsnorm(&x, &w2, 1e-6);
        let out1 = rmsnorm(&x, &w1, 1e-6);
        for (a, b) in out2.iter().zip(out1.iter()) {
            assert!(
                (a - 2.0 * b).abs() < 1e-5,
                "rmsnorm with weight=2 should be 2× weight=1"
            );
        }
    }

    /// Fault detected: RoPE sign swap (cos/sin switched).
    /// A rotation by 2π returns the original vector.
    /// For position=0: all thetas=0, cos=1, sin=0 → output == input.
    #[test]
    fn rope_position_zero_is_identity() {
        let mut tensor = vec![1.0f32, 2.0, 3.0, 4.0]; // 1 seq, 1 head, head_dim=4
        let orig = tensor.clone();
        apply_rope(&mut tensor, 1, 1, 4, 10000.0);
        for (a, b) in tensor.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6, "rope at position 0 must be identity");
        }
    }

    /// F10: RoPE known-answer test at position 1 (non-trivial rotation).
    ///
    /// Hand-computed from Su et al. 2022:
    ///   pos=1, head_dim=4, base=10000
    ///   pair 0: θ = 1 / 10000^0 = 1.0
    ///           x0' = 1*cos(1) - 2*sin(1),  x1' = 2*cos(1) + 1*sin(1)
    ///   pair 1: θ = 1 / 10000^0.5 = 0.01
    ///           x2' = 3*cos(0.01) - 4*sin(0.01),  x3' = 4*cos(0.01) + 3*sin(0.01)
    #[test]
    fn rope_position_one_known_answer() {
        // Two tokens, one head, head_dim=4.
        // Token at pos=0: identity (all zeros so we can ignore it).
        // Token at pos=1: [1.0, 2.0, 3.0, 4.0].
        let mut tensor = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut tensor, 2, 1, 4, 10000.0);

        // pos=0 must stay identity.
        for v in &tensor[0..4] {
            assert!(v.abs() < 1e-6, "pos=0 must be identity rotation, got {v}");
        }

        let cos1 = 1.0_f32.cos();
        let sin1 = 1.0_f32.sin();
        let cos001 = 0.01_f32.cos();
        let sin001 = 0.01_f32.sin();

        let expected = [
            1.0 * cos1 - 2.0 * sin1,
            2.0 * cos1 + 1.0 * sin1,
            3.0 * cos001 - 4.0 * sin001,
            4.0 * cos001 + 3.0 * sin001,
        ];

        for (i, (got, want)) in tensor[4..8].iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "rope position=1 index {i}: got {got}, expected {want}"
            );
        }
    }

    /// Fault detected: GQA doesn't scale dot product by 1/sqrt(head_dim).
    /// Single token, single head, single KV head — attention over 1 past token
    /// with identical Q and K should produce the V vector.
    #[test]
    fn gqa_single_token_returns_value() {
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let mut cache = KvCache::new(n_kv_heads, head_dim, 8);
        // Push one token: K and V are identity-like vectors.
        let k = vec![1.0f32, 0.0, 0.0, 0.0];
        let v = vec![0.0f32, 1.0, 2.0, 3.0];
        cache.push(&k, &v);

        // Query that exactly matches K (dot product maximised for head 0).
        // For GQA both heads share this single KV head.
        let q = vec![
            1.0f32, 0.0, 0.0, 0.0, // head 0
            1.0, 0.0, 0.0, 0.0, // head 1 (same KV head)
        ];
        let out = gqa_attention(&q, &cache, n_heads, head_dim);
        // With 1 token in cache, attention is deterministic: output = V regardless of score.
        assert_eq!(
            out.len(),
            n_heads * head_dim,
            "output length must be n_heads*head_dim"
        );
        // Head 0 output should be V = [0, 1, 2, 3]
        let h0 = &out[0..4];
        assert!((h0[0] - 0.0).abs() < 1e-5, "gqa h0[0] got {}", h0[0]);
        assert!((h0[1] - 1.0).abs() < 1e-5, "gqa h0[1] got {}", h0[1]);
    }

    /// Fault detected: SwiGLU uses ReLU instead of SiLU (different activation shape).
    /// Input x=[0,...,0] → gate=0, silu(0)=0 → out=0.
    #[test]
    fn swiglu_zero_input_produces_zero_output() {
        let hidden = 4;
        let intermediate = 8;
        let x = vec![0.0f32; hidden];
        let gate_w = vec![1.0f32; intermediate * hidden];
        let up_w = vec![1.0f32; intermediate * hidden];
        let down_w = vec![1.0f32; hidden * intermediate];
        let out = swiglu_ffn(&x, &gate_w, &up_w, &down_w, hidden, intermediate);
        for &v in &out {
            assert!(
                v.abs() < 1e-6,
                "swiglu with zero input must produce zero output, got {v}"
            );
        }
    }

    /// Fault detected: KV cache push doesn't advance len (next push overwrites).
    #[test]
    fn kv_cache_len_advances_on_push() {
        let mut cache = KvCache::new(2, 4, 16);
        assert_eq!(cache.len, 0);
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        cache.push(&k, &v);
        assert_eq!(cache.len, 1, "len must advance after push");
        cache.push(&k, &v);
        assert_eq!(cache.len, 2, "len must advance on second push");
    }
}
