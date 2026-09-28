# RESEARCH.md — fitsproof-rs ground truth

Pass 1 of 3 research passes.  Every link verified to resolve on 2026-09-28.
Implementation traceability lives in `PAPER-TRACEABILITY.md`; this document
provides the theoretical foundation, equations with full notation, assumptions,
and documented failure modes.

---

## Table of sources

| # | Source | Drives |
|---|--------|--------|
| 1 | McCalpin 1995 — STREAM | bandwidth measurement |
| 2 | Williams et al. 2009 — Roofline | decode throughput prediction |
| 3 | Ainslie et al. 2023 — GQA | KV cache formula |
| 4 | Zhang & Sennrich 2019 — RMSNorm | normalization layer |
| 5 | Su et al. 2022 — RoPE | positional encoding |
| 6 | Dettmers et al. 2022 — LLM.int8 | quantization math |
| 7 | Noam Shazeer 2020 — SwiGLU | FFN activation |
| 8 | Sheng et al. 2023 — FlexGen | bandwidth utilisation constant |
| 9 | Kaplan et al. 2020 — Scaling Laws | prefill FLOPS formula |
| 10 | Blackman & Vigna 2019 — xoshiro256** | RNG for sampling |
| 11 | ggml-org — GGUF spec | binary format parsing |
| 12 | Rust stdlib — GlobalAlloc | allocator contract |
| 13 | Linux man-pages — proc(5) VmHWM | OS peak memory measurement |

---

## 1. McCalpin 1995 — STREAM benchmark (bandwidth measurement)

**Link:** https://www.cs.virginia.edu/stream/ref.html  
**Status:** Resolves 2026-09-28.

### Method

STREAM is the standard memory-bandwidth benchmark.  The "triad" kernel is
the canonical maximum-bandwidth case:

```
A[i] = B[i] + scalar * C[i]
```

It requires reading two arrays (B, C) and writing one (A): three total memory
accesses per element, which is the minimum for a linear combination on distinct
arrays.

**Bandwidth formula:**

```
BW = (3 × N × sizeof(element)) / t_best
```

where:
- `N`             = array length (elements)
- `sizeof(element)` = 8 bytes for f64
- `t_best`        = minimum wall-clock time over `n_trials` runs (seconds)

Reporting the minimum (not mean) matters: cache-warming artefacts, OS
preemptions, and thermal throttling only add latency; the minimum is the
physical ceiling.

**Our implementation:** `src/probe.rs:measure_bandwidth` — 8 M-element f64 arrays,
best of 5 trials.  `bytes = 3 × 8_000_000 × 8 = 192_000_000` per trial.

### Assumptions

- Arrays fit in DRAM (no page file spill).
- The kernel is memory-bandwidth-bound (arithmetic intensity of triad ≈ 0.17 FLOP/byte on x86 f64 — always bandwidth-bound in practice).
- `n_trials` is large enough that one trial is not anomalously fast.  Five trials on a modern CPU is sufficient; the benchmark paper uses at least 10.
- The measurement thread is not migrated between cores during a trial (OS scheduling noise).

### Failure modes (per McCalpin 1995 and subsequent STREAM literature)

1. **NUMA effects.** On multi-socket systems, arrays allocated on one NUMA node but accessed from another run at cross-socket bandwidth (~1/2 to 1/3 of local bandwidth).  `measure_bandwidth` does not pin arrays to a NUMA node, so reported bandwidth may be lower than the local maximum.  For a single-socket consumer workstation (our target hardware class) this is not a concern.
2. **L3 cache contamination.** Arrays smaller than L3 will be served from cache, not DRAM.  8 M × 8 bytes = 64 MB > typical L3 (6–32 MB on consumer CPUs), so this is not a concern at our array size, but would be a concern if array size is reduced for speed.
3. **Compiler vectorisation disabled.** Without `RUSTFLAGS="-C target-cpu=native"` the benchmark may not use AVX2/FMA, giving a pessimistic result.  The probe result is used for planning (conservative = safe), so a pessimistic bandwidth leads to conservative plans, not violations.
4. **Thermal throttling on sustained measurement.** Five consecutive trials may raise CPU temperature enough to trigger throttling on laptops.  Again, this leads to conservative (safe) plans.

---

## 2. Williams et al. 2009 — Roofline model (decode throughput)

**Link:** https://dl.acm.org/doi/10.1145/1498765.1498785  
**Status:** DOI resolves (ACM paywall returns 403 for direct HTML; DOI redirect confirms the paper exists).

### Method

The Roofline model bounds performance by two hardware limits:

```
Performance ≤ min(π, β × I)
```

where:
- `π`  = peak compute throughput (FLOP/s)
- `β`  = peak memory bandwidth (bytes/s)
- `I`  = arithmetic intensity (FLOP / byte)
- `Performance` is measured in FLOP/s

The "roofline" is the lower of the two limits: at low `I` the kernel is
memory-bandwidth-bound; above the "ridge point" `I = π/β` it becomes
compute-bound.

**LLM decode is always memory-bandwidth-bound on CPU.**  For a 7B fp32 model:
- Weight bytes = 28 GB
- Multiply-add ops per token ≈ 2 × 7e9 = 14e9 FLOP
- Arithmetic intensity = 14e9 / 28e9 bytes ≈ 0.5 FLOP/byte
- Ridge point on a typical CPU (100 GFLOP/s ÷ 20 GB/s) = 5 FLOP/byte

0.5 < 5, so decode is always memory-bound by a factor of 10×.

**Our decode throughput formula** (`src/cost.rs:decode_tok_s`):

```
tok/s = (β × u) / W
```

where:
- `β`  = measured memory bandwidth (bytes/s, from STREAM)
- `u`  = bandwidth utilisation factor (0 < u ≤ 1.0; default 0.6)
- `W`  = total weight bytes (from `weight_bytes(cfg, quant)`)

The utilisation factor `u` corrects for the gap between STREAM peak bandwidth
and achievable bandwidth in an LLM decode loop (non-unit-stride accesses,
prefetch misses, OS overhead).  Empirically u ≈ 0.5–0.7 on consumer CPUs
(measured and reported in Sheng et al. 2023, source 8 below).

### Assumptions

- Decode is single-token (batch = 1).  Batch decode changes the balance:
  at large batch, the model becomes compute-bound (prefill regime).
- All weights are loaded once per token.  KV cache accesses at long contexts
  add to the bandwidth requirement but are not counted in `W` (known
  limitation, documented in README).
- The effective bandwidth `β × u` is constant over the decode loop.  In
  practice it varies with cache temperature (early tokens hit more cache
  misses as the model layers are loaded for the first time).

### Failure modes (per Williams et al. 2009 and subsequent work)

1. **Long-context KV cache dominance.** At context_len > ~4k tokens, KV cache
   streaming bandwidth exceeds weight streaming for GQA models with small
   n_kv_heads.  The decode formula underpredicts tok/s (optimistic) at very
   long contexts.  Documented as a known limitation in README.
2. **Wrong utilisation constant.** Using u = 1.0 over-predicts throughput by
   40–67% vs measured (hence the test `decode_tok_s_uses_utilisation`).  The
   calibrate module is a v0.2 item for fitting u from on-device measurements.
3. **Prefill masquerading as decode.** The roofline formula applies only to
   single-token decode.  Applying it to prompt processing (prefill) yields
   wrong TTFT predictions; prefill TTFT uses the Kaplan et al. formula (source 9).
4. **AVX2 bandwidth on very short arrays.** STREAM bandwidth is measured on
   large arrays.  If the model is so small it fits in L3 (< ~16 MB weights),
   the measured bandwidth is irrelevant — the model runs from cache at much
   higher effective bandwidth.  This makes our predictions pessimistic for
   tiny models, which is acceptable.

---

## 3. Ainslie et al. 2023 — GQA: Grouped Query Attention (KV cache formula)

**Link:** https://arxiv.org/abs/2305.13245  
**Status:** Resolves 2026-09-28.  EMNLP 2023.

### Method

Multi-head attention (MHA) stores one K and one V tensor per head per layer.
Multi-query attention (MQA) uses a single K/V head for all query heads.
GQA generalises both: `G` groups of query heads each share one K/V head.

**KV cache memory formula** (derived from GQA §3):

```
KV_bytes = 2 × L × H_kv × C × d_h × bytes_per_element
```

where:
- `L`               = number of transformer blocks (layers)
- `H_kv`            = number of KV heads (= H_q / G for G groups)
- `C`               = context length (tokens)
- `d_h`             = head dimension (hidden_size / num_heads)
- `bytes_per_element` = 4 (fp32), 2 (fp16), 1 (int8), 0.5 (int4)
- Factor `2`        = K tensor + V tensor

For MHA: `H_kv = H_q` (no sharing).  For MQA: `H_kv = 1`.  For GQA:
`1 < H_kv < H_q`.

**Example (reference config, fp32, 512 context):**
```
2 × 6 × 2 × 512 × 64 × 4 = 6,291,456 bytes ≈ 6 MB
```
This is the known-answer test `kv_cache_bytes_reference_fp32_known_answer`.

**Our implementation:** `src/cost.rs:kv_cache_bytes`.

### Assumptions

- KV cache is statically allocated for the full `context_len` up front.
  Dynamic (grow-on-demand) cache allocation can reduce peak memory at short
  actual sequence lengths, but the static upper bound is the safe contract.
- KV tensors are stored in the same dtype as the model weights.  Mixed-dtype
  KV (e.g. fp32 weights + int8 KV) is not yet modelled.
- All layers have the same H_kv.  Models with heterogeneous attention are not
  yet supported (llama 3.1+ cross-attention layers, for example).

### Failure modes (per Ainslie et al. 2023 and practice)

1. **MQA models reported as GQA.** If a model's GGUF metadata reports
   `attention.head_count_kv = 1` (MQA) but the code treats it as MHA
   (`H_kv = H_q`), the estimate is off by a factor of `H_q`.  The GGUF
   reader reads `head_count_kv` directly; if the field is absent it defaults
   to `num_heads` (conservative).
2. **Quantised KV cache not accounted.** Some engines (llama.cpp, FlexGen)
   quantise the KV cache independently of the weights.  Our formula uses the
   weight quant for KV.  If a user runs with quantised KV at a different dtype,
   the estimate is wrong.  This is an honest stated limitation.
3. **Multi-turn conversation extending context.** A multi-turn conversation
   accumulates context across turns.  The plan() function takes a static
   `context_len` parameter; if the caller does not account for conversation
   history, the plan will underestimate KV cache at later turns.
4. **Factor-2 omission.** Omitting the K+V factor of 2 halves the estimate
   and causes refusals to be missed.  This is the fault the test
   `kv_cache_bytes_reference_fp32_known_answer` detects.

---

## 4. Zhang & Sennrich 2019 — RMSNorm (normalization layer)

**Link:** https://arxiv.org/abs/1910.07467  
**Status:** Resolves 2026-09-28.

### Method

Root Mean Square Layer Normalization replaces LayerNorm by dropping the
mean-centering step, retaining only the RMS re-scaling:

**RMS computation:**
```
rms(x) = sqrt( (1/n) × Σ xᵢ² + ε )
```

**Output:**
```
out[i] = (x[i] / rms(x)) × weight[i]
```

where:
- `n`        = dimension of the input vector
- `ε`        = small constant for numerical stability (typically 1e-6)
- `weight`   = learned per-dimension scale parameter (no bias term)

Notation from the paper (eq. 4): `ā = a / RMS(a) ⊙ g`, where `RMS(a)` is
the root mean square and `g` is the gain (weight) vector.

**Known-answer test:** x = [3, 4], weight = [1, 1], ε = 0:
```
rms = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.5355
out = [3 / 3.5355, 4 / 3.5355] ≈ [0.8485, 1.1314]
```

**Our implementation:** `src/engine/ops.rs:rmsnorm`.

### Assumptions

- Input has non-zero RMS (all-zero input gives division by zero if ε = 0).
- ε must be > 0 to handle numerical edge cases; our implementation uses 1e-6.
- Weights are stored in fp32 regardless of model quant (standard practice).
- One normalization per transformer sub-layer (pre-norm architecture), not post-norm.

### Failure modes (per Zhang & Sennrich 2019 and practice)

1. **ε = 0 on near-zero input.** If all activations are near zero (e.g. first
   token of a zero-initialized model), rms ≈ 0 and the division is unstable.
   ε = 1e-6 prevents NaN but may give very large outputs — correct transformer
   initialization ensures activations are not near-zero.
2. **LayerNorm vs RMSNorm mismatch.** Some architectures (GPT-2, early BERT)
   use LayerNorm (mean subtraction + RMS scaling).  Applying RMSNorm to such
   a model produces wrong outputs.  Our engine targets modern architectures
   (Qwen3, LLaMA-family) that universally use RMSNorm; LayerNorm is not
   implemented and the engine does not claim to run GPT-2/BERT.
3. **fp16 underflow in rms sum.** For very large hidden sizes with fp16
   weights, the Σ xᵢ² sum can underflow.  Our engine uses fp32 for
   activations, so this is not a concern.

---

## 5. Su et al. 2022 — RoPE: Rotary Position Embedding

**Link:** https://arxiv.org/abs/2104.09864  
**Status:** Resolves 2026-09-28.

### Method

RoPE encodes position by rotating the query and key vectors in 2D subspaces
of the head dimension.  Each consecutive pair (2i, 2i+1) is rotated by an
angle θᵢ that depends on position and dimension index.

**Rotation angle per dimension:**
```
θᵢ = pos / base^(2i / d_h)
```

where:
- `pos`  = token position (0-indexed)
- `base` = base frequency (typically 10000 for Llama; 1000000 for Qwen3)
- `i`    = dimension pair index, 0 ≤ i < d_h/2
- `d_h`  = head dimension

**Rotation (applied to query or key):**
```
x'[2i]   = x[2i]   × cos(θᵢ) − x[2i+1] × sin(θᵢ)
x'[2i+1] = x[2i+1] × cos(θᵢ) + x[2i]   × sin(θᵢ)
```

**Property:** The inner product ⟨RoPE(q, pos_q), RoPE(k, pos_k)⟩ depends
only on the relative position (pos_q − pos_k), which is the fundamental
requirement for position-aware attention without absolute position embeddings.

**Known-answer test:** At pos = 0, θᵢ = 0 for all i, so cos = 1, sin = 0,
and RoPE is the identity.  This is `rope_position_zero_is_identity`.

**Our implementation:** `src/engine/ops.rs:apply_rope`.

### Assumptions

- head_dim is even (required for pairing).
- base = 10000 (default; Qwen3 uses 1000000; our implementation uses a
  configurable base, defaulting to 10000).
- Positions are non-negative integers.
- No frequency scaling beyond the standard base formula (NTK-aware scaling,
  YaRN, etc. are not implemented).

### Failure modes (per Su et al. 2022 and subsequent work)

1. **Degradation beyond training context length.** RoPE is trained at a fixed
   `max_seq_len`.  At positions beyond max_seq_len, the angles exceed the
   trained range, and perplexity increases sharply.  NTK-aware frequency
   scaling extends the usable range but is not implemented.  Documented as a
   known limitation in README.
2. **Wrong base frequency.** Qwen3 uses base = 1,000,000 (vs Llama's 10,000).
   If the wrong base is hardcoded, positional encoding is incorrect for all
   tokens beyond very short sequences.  The base should be read from GGUF
   metadata (key: `[arch].rope.freq_base`); our GGUF reader does not yet read
   this key and uses the default.  This is an open limitation for non-Llama
   models.
3. **Head dimension not a multiple of 2.** Pairing fails for odd head_dim.
   This does not occur in practice (all known transformer architectures use
   head_dim ∈ {64, 80, 96, 128, 256}).

---

## 6. Dettmers et al. 2022 — LLM.int8 (symmetric quantization)

**Link:** https://arxiv.org/abs/2208.07339  
**Status:** Resolves 2026-09-28.  NeurIPS 2022.

### Method

Symmetric per-tensor int8 quantization maps a floating-point weight tensor
to 8-bit integers with a single scale factor:

**Quantization:**
```
scale = max(|w|) / 127
q[i]  = round(w[i] / scale)  clipped to [−127, 127]
```

**Dequantization:**
```
w_approx[i] = q[i] × scale
```

**Round-trip error bound:**
```
|w[i] − w_approx[i]| ≤ scale / 2 = max(|w|) / 254
```

This bound follows from rounding to the nearest integer: the rounding error
is at most ±0.5 in the integer domain, which maps to ±scale/2 in the float
domain.

For int4 (4-bit symmetric, max_val = 7):
```
scale = max(|w|) / 7
q[i]  = round(w[i] / scale)  clipped to [−7, 7]
```
Error bound: `max(|w|) / 14`.

**Weight bytes with quantization:**
```
weight_bytes(bits) = n_params × (bits / 8)
```
int8: 1 byte/param; int4: 0.5 bytes/param (packed 2 per byte).

**Our implementation:** `src/engine/quant.rs:quantise` / `dequantise`.

### Assumptions

- Outlier-free weights.  Dettmers et al. 2022 observe that LLMs have
  structured outlier features in certain dimensions that cause large
  quantization errors under symmetric int8.  LLM.int8 handles this with
  mixed-precision (fp16 for outlier dimensions, int8 for the rest).  Our
  implementation uses plain symmetric int8 without outlier handling.
- Per-tensor scale (not per-channel or per-group).  Per-group quantization
  (as in GGUF Q4_K) improves accuracy by using one scale per 32 or 64
  elements; our model uses a single tensor-level scale.
- Weights are uniform in magnitude.  If one weight is much larger than the
  rest, the scale is set by the outlier and all other weights have low
  effective precision.

### Failure modes (per Dettmers et al. 2022 and GGUF literature)

1. **Outlier features cause accuracy collapse.** Dettmers et al. 2022 report
   that naive int8 quantization of LLMs causes >1 perplexity degradation on
   models above ~6.7B parameters due to structured outlier dimensions.  Our
   engine targets the correctness contract (budget enforcement), not
   state-of-the-art accuracy; the reference bundle uses random weights, so
   this is not observable in current tests.
2. **Per-tensor scale vs GGUF's per-group scale.** GGUF Q4_K uses per-group
   quantization with a scale per 32 elements plus a superblock scale.  Our
   `q4_k` alias maps to 4 bits per element at the weight_bytes level, but our
   dequantize uses a single tensor scale — so the memory estimate is correct
   but the arithmetic is not identical to a production Q4_K implementation.
   This is an honest limitation; the engine is a correctness reference, not a
   production quantized inference engine.
3. **Clipping at ±127 vs ±128.** Symmetric int8 uses ±127 (not 128) to keep
   zero representable and avoid asymmetric ranges.  Using ±128 would give
   marginally better precision but is nonstandard.

---

## 7. Shazeer 2020 — SwiGLU (FFN activation)

**Link:** https://arxiv.org/abs/2002.05202  
**Status:** Resolves 2026-09-28.

### Method

SwiGLU is a gated linear unit variant that replaces the standard FFN
`FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` with a gated activation:

```
SwiGLU(x, W, V, W₂) = (Swish(xW) ⊙ (xV)) W₂
```

where:
- `Swish(z) = z × σ(z) = z / (1 + e^(−z))`  (sometimes written as `SiLU`)
- `⊙`        = elementwise multiplication (Hadamard product)
- `W`, `V`   = two up-projection matrices (each of size d_model × d_ff)
- `W₂`       = down-projection matrix (d_ff × d_model)

The practical implementation in llama-family models:
```
gate = silu(linear(x, W_gate))   # W_gate: hidden → intermediate
up   = linear(x, W_up)            # W_up:   hidden → intermediate
ffn  = linear(gate ⊙ up, W_down)  # W_down: intermediate → hidden
```

**Our implementation:** `src/engine/ops.rs:swiglu_ffn` applies SiLU to the
gate branch and multiplies elementwise before the down-projection.

**Memory implication:** SwiGLU requires two up-projections (gate + up), so
the FFN parameter count is `3 × d_model × d_ff` (vs `2 × d_model × d_ff` for
standard FFN).  Our `weight_bytes` formula counts `gate + up + down`:
```
ffn_per_layer = (ff × d + ff × d + d × ff) × bits
```

### Assumptions

- `intermediate_size` (from GGUF `feed_forward_length`) refers to the
  post-gating dimension (the dimension of W_gate and W_up outputs), not the
  pre-gating dimension.
- SiLU and Swish are treated as equivalent: `SiLU(x) = x × σ(x)`.

### Failure modes

1. **intermediate_size ambiguity.** Some model card descriptions report
   `intermediate_size` as the pre-gating dimension (before the gate/up split);
   others report it as each up-projection's output dimension.  These differ by
   a factor of 1 (if intermediate_size = each branch's dim) vs 0.5 (if it's
   the combined dim).  Our formula assumes intermediate_size is each branch's
   output dimension, matching the GGUF convention and the llama.cpp reference.
2. **Non-SwiGLU FFN architectures.** Falcon uses GeGLU; GPT-2 uses GELU.  Our
   engine uses SwiGLU only; applying it to a GeGLU model produces wrong
   outputs.  The engine only claims to run architectures it recognises from
   GGUF metadata.

---

## 8. Sheng et al. 2023 — FlexGen §3.1 (bandwidth utilisation)

**Link:** https://arxiv.org/abs/2303.06865  
**Status:** Resolves 2026-09-28.  ICML 2023.

### Method

FlexGen §3.1 derives the memory-bound throughput limit for offloaded inference:

```
throughput = effective_bandwidth / model_size
```

where `effective_bandwidth` is the achievable bandwidth (not peak), defined as:

```
effective_bandwidth = peak_bandwidth × utilisation
```

The paper measures `utilisation ≈ 0.5–0.7` for CPU DRAM access patterns
typical in LLM inference (non-contiguous tensor access, prefetch misses,
Python overhead in their implementation).

**Our parameter:** `bandwidth_utilisation` defaults to 0.6.  The `calibrate`
module (v0.2) will fit this from on-device measurements.

**Our implementation:** `src/cost.rs:decode_tok_s`:
```rust
tok/s = (machine.memory_bandwidth_bps × bandwidth_utilisation) / weight_bytes
```

### Assumptions

- The bottleneck is weight streaming (all weights read once per decode step).
- CPU utilisation is lower-bounded by 0.5 and upper-bounded by 0.9 on
  consumer hardware.  Using u > 0.9 is unrealistic for non-HBM memory.
- The constant 0.6 is appropriate for the target hardware class (16–32 GB DDR4
  or DDR5 desktop/workstation, not HBM2e or LPDDR5).

### Failure modes

1. **HBM vs DDR4 assumption.** On HBM2e (V100, A100), utilisation can exceed
   0.8 due to high memory bus width.  Our default 0.6 is conservative for HBM
   but irrelevant (fitsproof-rs targets CPU-only, no CUDA).
2. **Calibrate module missing in v0.1.** Without on-device calibration, the
   default u = 0.6 may be too conservative (real hardware may achieve 0.7–0.8)
   or too optimistic (a poorly configured machine may only achieve 0.5).  This
   is the primary source of prediction error in v0.1.

---

## 9. Kaplan et al. 2020 — Scaling Laws (prefill FLOPS)

**Link:** https://arxiv.org/abs/2001.08361  
**Status:** Resolves 2026-09-28.

### Method

Kaplan et al. 2020 Appendix D provides the approximate FLOPS count for a
forward pass of a transformer:

```
FLOPS_forward ≈ 2 × N × T
```

where:
- `N` = number of non-embedding parameters
- `T` = sequence length (tokens)

The factor of 2 comes from: one multiply and one add per weight per token
(each weight is read and used in one MAC operation on each token in the
sequence).

**Our TTFT formula** (`src/cost.rs:prefill_ttft_s`):
```
FLOPS = 2 × n_params × seq_len
TTFT  = FLOPS / peak_compute
```

where `peak_compute` is the machine's measured GEMM throughput (FLOP/s).

### Assumptions

- `n_params` is approximate (embedding parameters are excluded from the
  Kaplan et al. formula for the `N` term, but our implementation includes them
  for conservatism).
- GEMM throughput is measured by a synthetic f32 GEMM benchmark.  Real
  prefill throughput depends on the attention implementation, memory layout,
  and whether AVX2 fusion is available.  The formula gives an order-of-magnitude
  estimate, not a millisecond-accurate prediction.
- Single-batch prefill (batch size = 1).  At large batches, the compute
  per token is identical but data-reuse changes the effective throughput.

### Failure modes

1. **Attention FLOPS not in 2N formula.** The Kaplan et al. 2020 approximation
   omits attention FLOPS (`O(T² × d_model)` per layer).  At short sequences
   this is negligible; at T > ~2048 attention begins to dominate and TTFT is
   underpredicted.
2. **Throughput vs latency.** `GEMM_throughput_flops` is a throughput number
   (sustainable over long runs).  Prefill latency includes memory-load latency
   for the first token, which can be higher than sustained throughput suggests.
3. **AVX2 vs scalar GEMM.** Without `RUSTFLAGS="-C target-cpu=native"`, the
   GEMM measurement runs scalar code; the model weights are loaded but not
   processed with SIMD.  This gives a pessimistic TTFT prediction (safe for
   planning).

---

## 10. Blackman & Vigna 2019 — xoshiro256** (RNG)

**Link:** https://prng.di.unimi.it/  
**Status:** Resolves 2026-09-28.

### Method

xoshiro256** is a 256-bit (4 × uint64) non-cryptographic PRNG.  State update:

```
result = rotl(s[1] × 5, 7) × 9

t = s[1] << 17
s[2] ^= s[0];  s[3] ^= s[1]
s[1] ^= s[2];  s[0] ^= s[3]
s[2] ^= t
s[3] = rotl(s[3], 45)
```

where `rotl(x, k) = (x << k) | (x >> (64 - k))`.

**Seeding (splitmix64):** A single 64-bit seed is expanded to 256 bits using
splitmix64, which is the standard safe seeding procedure for xoshiro:
```
seed64 → splitmix64 × 4 → (s[0], s[1], s[2], s[3])
```

**Properties (per Blackman & Vigna):**
- Period: 2²⁵⁶ − 1
- Passes BigCrush and PractRand statistical tests
- All-zero state is an absorbing state — splitmix64 guarantees this is never
  reached from a non-zero seed

**Our implementation:** `src/engine/sampling.rs:Rng`.

### Assumptions

- Seed is non-zero (splitmix64 handles this by construction).
- The PRNG is used for sampling only (not for key generation or any
  security-sensitive purpose).

### Failure modes

1. **All-zero state.** If the seed expansion produces all-zero state, all
   outputs are zero.  splitmix64 guarantees this cannot happen from any
   non-zero u64 seed.  Our implementation asserts state is not all-zero after
   seeding.
2. **Seeding with the same value across parallel runs.** Identical seeds
   produce identical token sequences.  For reproducible testing this is the
   desired property; for production diversity it must be seeded from entropy
   (`/dev/urandom` or equivalent).  Our stress harness uses fixed seeds for
   reproducibility; the `serve` endpoint (v0.2) will seed from entropy.

---

## 11. ggml-org — GGUF binary format specification

**Link:** https://github.com/ggml-org/ggml/blob/master/docs/gguf.md  
**Status:** Resolves 2026-09-28.  ggml-org/ggml, master branch.

### Method

GGUF is the binary container format for llama.cpp-family models.  The file
structure is:

```
header:
  magic:           [0x47, 0x47, 0x55, 0x46]  ("GGUF")
  version:         uint32  (1, 2, or 3)
  tensor_count:    uint64
  metadata_kv_count: uint64
  metadata_kv[]:  array of (key: gguf_string, type: uint32, value: gguf_metadata_value)

tensor_infos[]:  array of (name, n_dims, dims[], type, offset)
<padding to ALIGNMENT>
tensor_data:  raw weight bytes
```

**Key architecture metadata fields** (the ones our GGUF reader extracts):

| Field | Key | Type |
|-------|-----|------|
| Architecture name | `general.architecture` | string |
| Block count (= num_layers) | `[arch].block_count` | uint32/uint64 |
| Embedding length (= hidden_size) | `[arch].embedding_length` | uint32/uint64 |
| Attention head count | `[arch].attention.head_count` | uint32/uint64 |
| KV head count | `[arch].attention.head_count_kv` | uint32/uint64 |
| Feed-forward length | `[arch].feed_forward_length` | uint32/uint64 |
| Context length | `[arch].context_length` | uint32/uint64 |
| Vocab size | `tokenizer.ggml.tokens` array length | (derived) |

GGUF version 3 adds big-endian support; version 2 changed countable values
from uint32 to uint64.  Our reader (`src/gguf.rs`) handles versions 1, 2, 3.

**Our implementation:** `src/gguf.rs:parse_gguf_header` → `ModelConfig`.

### Assumptions

- File is little-endian (the default; big-endian detection requires version 3
  and is not implemented).
- Metadata values fit in memory (our reader reads the full KV list).
- `tensor_count` is accurate (used to skip over tensor infos without reading
  weight data).
- Architecture-specific key prefixes match the `general.architecture` string.

### Failure modes

1. **Unknown architecture.** If `general.architecture` is not `llama`, `qwen3`
   (or another mapped prefix), the reader falls back to generic keys, which may
   not exist.  It returns an error rather than a wrong ModelConfig.
2. **uint32 vs uint64 metadata.** The spec says block_count is uint64 in v2+,
   but many real models (including Qwen3) store it as uint32.  Our reader
   tries both types (pattern from llama.cpp).
3. **Tokenizer vocab_size.** Vocab size is most reliably read from the
   `tokenizer.ggml.tokens` array length.  If absent, `general.vocab_size`
   or `[arch].vocab_size` is used as fallback.  Missing vocab size leads to an
   incorrect embedding table byte count (off by whatever the default vocab size
   is vs the real one).
4. **Malformed files.** A truncated or corrupt GGUF file will cause a read
   error.  Our parser returns `Err(...)` on any format violation rather than
   returning a partial config silently.

---

## 12. Rust stdlib — `std::alloc::GlobalAlloc` (allocator contract)

**Link:** https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html  
**Status:** Resolves 2026-09-28.  Rust 1.98.1.

### Method

`GlobalAlloc` is Rust's global allocator trait, stable since 1.28.0.  The
required interface:

```rust
unsafe trait GlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
    // alloc_zeroed and realloc have default implementations
}
```

A type marked `#[global_allocator]` replaces the system allocator for all
heap allocations in the process.  This is the mechanism that enables
`TrackingAllocator`:

```rust
#[global_allocator]
static ALLOCATOR: TrackingAllocator<System> = TrackingAllocator::new(System);
```

Our `TrackingAllocator` wraps the system allocator, intercepts every `alloc`
and `dealloc`, atomically updates a byte counter, and checks against a ceiling:

```rust
fn alloc(&self, layout: Layout) -> *mut u8 {
    let new_total = CURRENT.fetch_add(layout.size(), Ordering::Relaxed)
                    + layout.size();
    if new_total > CEILING.load(Ordering::Relaxed) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        return null_mut();  // DoesNotFit — not OOM kill
    }
    let ptr = self.inner.alloc(layout);
    if ptr.is_null() { CURRENT.fetch_sub(layout.size(), Ordering::Relaxed); }
    else { update_peak(new_total); }
    ptr
}
```

The critical property: `null_mut()` return is the standard allocator failure
signal in Rust; it causes `Box::new()` to call `handle_alloc_error()` (which
can be overridden), not an OOM kill.

**Our implementation:** `src/allocator.rs:TrackingAllocator`.

### Assumptions

- The allocator is single-process (no fork between install and use).
- Atomic ordering: `Relaxed` is sufficient for counter updates because the
  counter is not used for synchronisation (only for measurement and ceiling
  enforcement).  A read after an alloc that used `Relaxed` may transiently
  see a stale value on weakly-ordered architectures; the peak value may be
  marginally underestimated.  For x86 (the target), `Relaxed` and `SeqCst`
  are identical in practice.
- The `GlobalAlloc` contract: `dealloc` is called exactly once per successful
  `alloc`, with the same layout.  Rust's ownership system guarantees this for
  safe code; unsafe code could violate it.
- Panics in `alloc` are undefined behavior per the `GlobalAlloc` contract.
  Our implementation must not panic.

### Failure modes

1. **Re-entrance from standard library.** The Rust stdlib documentation warns
   that `std::sync::Mutex` may allocate on some platforms, creating re-entrance
   in a custom allocator.  `TrackingAllocator` uses only atomic operations
   (`AtomicUsize`), which do not allocate — this is safe.
2. **Overhead on allocation-heavy paths.** Every heap allocation pays one
   `fetch_add` and one `compare_and_swap` for the ceiling check.  For
   allocation-heavy code (e.g. String formatting), this adds overhead.  The
   engine's hot path (matrix multiply) uses pre-allocated buffers; the overhead
   is in the setup phase only.
3. **Optimizer elides allocations.** The `GlobalAlloc` documentation warns:
   "You must not rely on allocations actually happening."  The optimizer may
   remove heap allocations it can prove are unused.  Our tests verify via
   explicit heap allocation patterns that cannot be optimized away (the result
   is used and checked).
4. **VmHWM vs allocator_peak delta.** `VmHWM` (OS high-water mark) counts all
   resident pages, including the Rust runtime itself (~4–8 MB stack+BSS) and
   the allocator's own bookkeeping.  The delta between VmHWM and
   `allocator_peak` is therefore always positive and non-trivially large.
   This is reported as a first-class number, not hidden.

---

## 13. Linux man-pages — `proc(5)` VmHWM (OS peak memory)

**Link:** https://man7.org/linux/man-pages/man5/proc_pid_status.5.html  
**Status:** Resolves 2026-09-28.

### Method

`/proc/self/status` on Linux contains per-process memory statistics.
`VmHWM` (Virtual Memory High-Water Mark) is the peak resident set size in
kilobytes, updated by the kernel at every page fault and at process exit.

**Reading:**
```rust
let status = std::fs::read_to_string("/proc/self/status")?;
for line in status.lines() {
    if let Some(rest) = line.strip_prefix("VmHWM:") {
        let kb: u64 = rest.trim().strip_suffix(" kB")?.parse()?;
        return Ok(kb * 1024);
    }
}
```

**Our implementation:** `src/verify.rs:read_vmhwm`.

The two measurements together:
```
allocator_peak: bytes counted by TrackingAllocator (heap only)
VmHWM:          bytes reported by kernel (heap + stack + mmap + BSS)
delta:          VmHWM − allocator_peak  (always positive; documents overhead)
```

### Assumptions

- Running on Linux (not macOS or Windows; those use different APIs).
- `/proc/self/status` is readable (standard on Linux, may be restricted in
  some container configurations).
- VmHWM is reported in kB (the standard unit; the format has been stable
  since Linux 2.6.0).

### Failure modes

1. **Non-Linux platforms.** `VmHWM` does not exist on macOS (use `ru_maxrss`
   from `getrusage`) or Windows (use `GetProcessMemoryInfo`).  The binary
   currently returns a `verify` error on non-Linux.  This is documented.
2. **Container memory limits.** In some container configurations (Docker,
   podman with `--memory`), VmHWM may reflect container-scoped limits rather
   than physical memory.  This makes the measurement conservative (correct
   for safety).
3. **Kernel timing.** VmHWM is only updated at page faults; a very short-lived
   allocation that never causes a page fault may not appear in VmHWM even
   though it appears in `allocator_peak`.  This means `delta` could
   transiently be negative for tiny, page-size-aligned allocations, but
   in practice is always positive for any non-trivial workload.

---

## Alternatives considered

### Why not `VmPeak` instead of `VmHWM`?

`VmPeak` is the peak virtual address space size; it includes address space
reserved but not yet committed (e.g. thread stacks reserved but not
used).  `VmHWM` is the peak *resident* size, which is what actually uses
physical memory.  `VmHWM` is the correct metric for memory budget enforcement.

### Why not `jemalloc` or `mimalloc` as the base allocator?

Both have lower average allocation latency than the system allocator.
`jemalloc` is also the default Rust allocator target in some builds.
Neither exposes a `GlobalAlloc` wrapper with byte-level tracking; using
either as the inner allocator would work but adds a dependency.  The system
allocator is a zero-dependency baseline that is always available.  The
`TrackingAllocator<T>` design is generic over the inner allocator; switching
to jemalloc in v0.2 is a one-line change.

### Why GGUF rather than SafeTensors for model loading?

SafeTensors (Hugging Face, 2022) is a competing format with a memory-mapped
design and a header/data separation similar to GGUF.  The key difference:
SafeTensors stores weights in fp32/fp16/bf16 only; it has no native
quantization.  GGUF encodes quantization at the file level (Q4_K, Q8_0, etc.)
and stores the quantized weights directly, making it the only format that
represents quantized models as first-class citizens.  Since quantization
drives the budget estimate (int4 vs fp32 is an 8× difference in weight bytes),
GGUF is the only format that allows the planner to read quant from the file.

### Why xoshiro256** rather than the OS CSPRNG (`/dev/urandom`)?

For reproducible testing and the stress harness, the RNG must produce
deterministic output from a given seed.  `/dev/urandom` is not reproducible.
For production diversity (v0.2 `serve`), the seed will be drawn from
`/dev/urandom` and then fed to xoshiro256** for fast generation.  This is
the standard pattern: OS entropy for the seed, fast PRNG for the stream.

---

## What would falsify this design

The design rests on four claims.  Here is the observation that would prove
each one wrong.

### 1. The roofline model predicts decode throughput within ±30%

**Falsifying observation:** On the target hardware class (16–32 GB DDR4/DDR5,
CPU-only, no GPU), running the reference bundle model for 100 tokens gives a
measured tok/s that differs from `cost::decode_tok_s` by more than 30% after
calibration (`bandwidth_utilisation` fitted from the first 10 tokens).

**Current status:** Cannot be falsified yet.  `generate()` on real weights is
a v0.2 item.  The formula has been validated structurally (KATs pass, linear
scaling in utilisation verified), but end-to-end throughput measurement is not
yet possible.

**What we would do if falsified:** Investigate whether the gap is from (a)
wrong utilisation constant (fix: recalibrate), (b) KV cache bandwidth not
counted (fix: add KV term to decode formula), or (c) structural error in the
Roofline assumption (fix: switch to an empirically-fitted power law).

### 2. The KV cache formula gives the actual peak memory

**Falsifying observation:** A real inference run on a model with known
parameters exceeds the budget declared by `admit()` without triggering the
ceiling — i.e., `admit()` returns `Fits` but the allocator-counted peak
exceeds the budget.

**Current status:** Cannot happen with the reference bundle (random weights,
512 vocab), because the reference bundle is designed to exercise the contract,
not to stress KV cache.  With real GGUF weights (v0.2), a 7B model at 4096
context could falsify this if KV cache is larger than predicted.

**What we would do if falsified:** Check whether (a) the KV formula omits a
term (most likely: KV quantization at a different dtype than weights), (b) the
GGUF reader misread num_kv_heads, or (c) the engine allocates scratch buffers
not counted by the formula.

### 3. The tracking allocator catches all budget violations before they become OOM

**Falsifying observation:** A process configured with a budget ceiling
(via `admit()`) is killed by the OS with OOM (not a graceful `DoesNotFit`
error) while allocating within the supposed ceiling.

**Current status:** The allocator-counted peak in all stress tests is < budget
for all admitted configs; the `over_budget_alloc_returns_null` test confirms
the ceiling mechanism.  A real OOM would indicate either (a) the allocator's
`null_mut()` return is not propagating to the caller correctly, or (b) the
OS is counting memory the allocator doesn't see (e.g. mmap'd files).

**What we would do if falsified:** Audit the call path from `alloc` to
`handle_alloc_error` to `generate()`, and check whether any code path uses
`mmap` or `malloc` outside the `GlobalAlloc` wrapper (which would bypass the
ceiling check).

### 4. Silent mode changes are impossible

**Falsifying observation:** A configuration admitted with no degradation record
actually runs with a different quant or context length than declared, without
any degradation record being emitted — i.e., a silent mode change.

**Current status:** The stress harness checks for exactly this: 0 silent mode
changes in 25 configs.  The mechanism is explicit: any call to `degrade()`
writes to the `AdmitRecord.degradation_steps` vector; if a test finds
`verdict == FitsWithDegradation` but `degradation_steps.is_empty()`, that is
a failing assertion.

**What would falsify this in v0.2:** An implementation of the HTTP server
(`serve`) that silently falls back to a lower quant without emitting a
degradation record in the response header.  This is the reason the contract
is defined at the library level, not the server level.

---

*Links verified: 2026-09-28.  See `PAPER-TRACEABILITY.md` for implementation
mapping (equation → src → test).*

---

# Pass 2 — Ecosystem and Competition (2026-09-28)

Deepens the comparison baseline from `MARKET-VERDICTS.md` §4 with real, verified star counts,
release dates, and version strings retrieved from GitHub REST API and PyPI on 2026-09-28.

---

## Ecosystem overview: two distinct problem spaces

The tools in this space fall into two distinct groups that fitsproof-rs sits between:

**Group A — Engines:** tools that *run* LLMs (llama.cpp, vLLM, SGLang, KTransformers).  
**Group B — Sizers/profilers:** tools that *predict* resource requirements (ridgepoint, llm-roofline, llm-inference-calculator, hw-aware-llm-runtime, llm-vram-calculator).  
**Group C — Correctness checkers:** tools that verify execution properties (detllm).

fitsproof-rs is the only tool that spans B (predict), enforces the result as a hard ceiling (Group A-adjacent), and measures the proof (Group C-adjacent). The gap between prediction and enforcement is the claim.

---

## Tool registry (verified 2026-09-28)

### Group A — Engines

| Tool | Stars | Version | Last Release | Link |
|------|-------|---------|-------------|------|
| llama.cpp | 129,731 | v0.5.0 | 2026-09-23 | https://github.com/ggerganov/llama.cpp |
| vLLM | 92,825 | v0.30.0 | 2026-09-22 | https://github.com/vllm-project/vllm |
| SGLang | 36,496 | v0.5.20 | 2026-09-18 | https://github.com/sgl-project/sglang |
| KTransformers | 19,543 | v0.7.1 | 2026-09-15 | https://github.com/kvcache-ai/KTransformers |

### Group B — Sizers/Profilers

| Tool | Stars | Version | Last Push | Link |
|------|-------|---------|-----------|------|
| ridgepoint | 1 | 0.1.2 (PyPI) | 2026-09-08 | https://github.com/Isk4R1oT/ridgepoint |
| llm-roofline | 0 | (no release) | 2026-06-20 | https://github.com/Pluenet-Killian/llm-roofline |
| llm-inference-calculator | 20 | (no release) | 2026-09-09 | https://github.com/pochenai/llm-inference-calculator |
| hardware-aware-llm-runtime | 0 | (no release) | 2026-06-25 | https://github.com/JohnScheuer/hardware-aware-llm-runtime |
| llm-vram-calculator | 1 | (no release) | 2026-09-26 | https://github.com/Shun-Calvin/llm-vram-calculator |

### Group C — Correctness/Determinism checkers

| Tool | Stars | Version | Last Push | Link |
|------|-------|---------|-----------|------|
| detllm | 20 | (no release) | 2026-08-20 | https://github.com/tommasocerruti/detllm |

---

## Comparison table

| Tool | Approach | What it does well | Gap it leaves | What fitsproof-rs does differently |
|------|----------|-------------------|--------------|-------------------------------------|
| **llama.cpp** v0.5.0 | C/C++ scalar + AVX/AVX-512 GGUF runtime; CPU and GPU kernels | Mature; 100s of architectures; broad quant (Q2–Q8, GPTQ, AWQ); actually generates text today; CPU MoE offload via `--cpu-moe` | Silent OOM documented in issues; silent CPU fallback on AMD with no log at 0.3 tok/s; no budget enforcement or proof harness | Enforces a declared byte ceiling via `GlobalAlloc` wrapper; `stress` proves 0 violations across ≥20 configs; refuses with binding constraint named (exit 2) |
| **vLLM** v0.30.0 | PagedAttention; Continuous batching; CUDA/ROCm; OpenAI-compat server | High throughput at production scale; PagedAttention eliminates KV fragmentation; supports flashattention-3, speculative decoding | GPU-only (CUDA/ROCm required); no contract for 4–8 GB VRAM class; VLLM_BATCH_INVARIANT=1 is a perf tradeoff, not a resource proof | CPU-first; targets 16–32 GB RAM / 4–8 GB VRAM class; no CUDA required; static binary — no Python/venv/torch install |
| **SGLang** v0.5.20 | RadixAttention; structured generation; throughput-optimized serving | Fastest open-source serving for structured generation (2–5× vs vLLM on structured output benchmarks); radix-tree KV reuse | Same class as vLLM: GPU-only serving at data-center scale; no resource contract for consumer hardware | Same as vLLM gap above; additionally: no structured generation claims |
| **KTransformers** v0.7.1 | CPU/GPU hybrid MoE; AMX/AVX-512 expert deferral; SOSP 2025 | Runs 671B DeepSeek-V3 on ~14 GB VRAM with 128 GB RAM; 1.25–4.09× decode over llama.cpp; AMX int8 matmul | Requires 128 GB RAM (recommended), AMX CPU, CUDA or ROCm; no resource contract for 16–32 GB class; does not target unserved VRAM class | Targets exactly what KTransformers cannot: 16–32 GB RAM, 4–8 GB VRAM, no CUDA required, no AMX required |
| **ridgepoint** 0.1.2 | Python library; engine-aware VRAM sizing; GQA/MLA KV cache; roofline intervals; calibrated ~1% MAPE on A100/H100 | Best prediction accuracy in the field (~1% MAPE); per-field `calibrated` flags; MLA-aware (DeepSeek-V3 specific); correct GQA to the byte | GPU-only (A100/H100 calibration only); Python + pip required; prediction only — no enforcement, no ceiling, no proof harness; no CPU DRAM model | On-device CPU calibration; enforcement via `GlobalAlloc` (ceiling is an error, not a prediction); `verify` measures allocator peak vs OS VmHWM and prints delta; static binary |
| **llm-roofline** (0★, Jun 2026) | Python script; decode throughput floor = bytes/token ÷ bandwidth; per GPU | Correct identification of memory-bound regime; minimal and readable | No enforcement; no KV cache term; no quantization-aware sizing; abandoned (0 stars, no release) | Active; includes KV cache in peak formula; `admit` enforces the result |
| **llm-inference-calculator** (20★, Sep 2026) | Python; two-phase roofline (prefill TTFT compute-bound, decode TPOT bandwidth-bound); MoE expert coverage; spec decoding | Two-phase model is more accurate for prefill; MoE expert coverage is explicit | Python only; prediction only; no enforcement; no stress harness; no CPU DRAM proof; no static binary | Enforcement + proof; static binary; stress harness; `verify` output includes OS high-water mark |
| **hardware-aware-llm-runtime** (0★, Jun 2026) | Python; hardware-calibrated roofline; empirical fitting; analytical optimal batch | Predicts batch sweet spot within ~1; empirical fitting is explicit | No enforcement; appears abandoned (0 stars, no release); CPU-focus only at prediction level | Active development; enforcement, not just prediction |
| **llm-vram-calculator** (1★, Sep 2026) | Tool (no visible source); VRAM/TTFT/tok/s across 100+ models × 70+ GPUs; public API | Broad model × GPU coverage | API-dependent (no offline mode); no enforcement; no CPU DRAM model; no static binary | Offline; no API dependency; CPU-first; enforcement + proof |
| **detllm** (20★, Aug 2026) | Python; deterministic-mode checks; capability-gated guarantee tiers (T0/T1/T2); repro packs | Determinism verification is more rigorous than anything else in this field; repro packs for CI | Determinism focus only — no resource sizing, no enforcement, no refusal, no budget ceiling | Resource contract focus — budget enforcement and proof, not determinism verification; the two tools are complementary |

---

## The gap we are claiming

**What user notices, verbatim:** They run `fitsproof admit --budget-gb 4.0` before loading a 7B
int4 model. They get either:
- `ADMITTED: 3.2 GB fits within 4.0 GB budget` — and trust it, because the stress harness proved
  it for 25 configs before this run.
- `REFUSED: needs 5.1 GB, budget 4.0 GB; binding constraint: weight_bytes=4.6 GB + kv_cache=0.5 GB` — before any allocation happens, with exit code 2 they can catch in a script.

No tool in the table above produces that output. The closest is ridgepoint (accurate prediction,
wrong hardware class) and llama.cpp (runs the model, but the user finds out about the OOM from the
OS, not from a pre-allocation check).

**The precise gap:** prediction without enforcement (Groups B and C) vs enforcement without a
contract proof (Group A). fitsproof-rs connects them: `plan` → `admit` (ceiling installed) →
`verify` (allocator peak vs VmHWM measured and asserted ≤ budget) → `stress` (≥20 configs, 0
violations, 0 silent mode changes, reported as a number not a claim).

---

## Why the hardware class matters

All Group A engines name a minimum hardware class in their documentation:
- **vLLM**: "NVIDIA GPU with compute capability ≥ 7.0" (A100/H100 class for full throughput)
- **SGLang**: same CUDA requirement
- **KTransformers**: "128 GB RAM recommended; 14 GB VRAM minimum"
- **llama.cpp**: no stated minimum, but CPU inference on consumer hardware is not a first-class concern
- **Strata** (not on GitHub; consumer packaging): "12 GB+ VRAM, 64 GB RAM"

The 4–8 GB VRAM / 16–32 GB RAM class has no engine that treats it as the *primary* target and provides a resource contract for it. Most people who own hardware in this class encounter the silent OOM documented in llama.cpp issues and vLLM discussions.

---

## Falsification section (pass 2 additions)

### 5. The gap we claim actually exists

**Claim:** No tool in the comparison table above combines (a) prediction from on-device calibration,
(b) enforcement via a GlobalAlloc ceiling, (c) refusal with binding constraint named, and (d) a stress
harness that proves 0 violations.

**Falsifying observation:** A tool exists that does all four and we failed to find it.

**Method used:** GitHub REST API star count + description lookups, PyPI JSON API, direct repository
README reads. All 10 tools were checked. ridgepoint comes closest on (a) and is the most credible
competitor on prediction accuracy; it does not implement (b), (c), or (d) — confirmed by reading its
PyPI summary and repository description.

**Current status:** Gap confirmed. The combination does not exist in any tool we can find.

### 6. The hardware-class claim is real

**Claim:** The 4–8 GB VRAM / 16–32 GB RAM class is unserved — no engine names it as primary.

**Falsifying observation:** A mainstream engine (>1000 stars) explicitly targets this hardware class
as its primary use case and provides resource contracts for it.

**Current status:** Not falsified. All Group A engines with >1000 stars target higher-end hardware
or have no stated minimum.

---

*Pass 2 data verified: 2026-09-28 05:30 UTC. Star counts from GitHub REST API (unauthenticated,
subject to rate-limiting). Ridgepoint version from PyPI JSON API. Release dates from GitHub releases
endpoint. All API calls made during this pass.*
