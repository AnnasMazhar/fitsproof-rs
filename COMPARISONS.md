# COMPARISONS.md

fitsproof-rs's claim is the **resource contract**: predict + enforce a byte ceiling +
measure proof + degrade explicitly, in a single static binary, for the **4–8 GB VRAM /
16–32 GB RAM** hardware class that every major engine treats as secondary.

It does not claim speed or model coverage superiority — `COMPARISONS.md` names each
tool and states where it beats us.

Star counts and versions verified 2026-09-28T14:00 UTC from GitHub REST API and PyPI.

---

## Engines (Group A) — they run models; we prove the contract

| Tool | Stars | Version | What it does better than fitsproof-rs | Where fitsproof-rs differs |
|------|-------|---------|---------------------------------------|----------------------------|
| **llama.cpp** | 129,765 | v0.5.0 (2026-09-23) | Mature; hundreds of architectures; fast AVX/AVX-512/AMX CPU kernels; broad quant (Q2–Q8, GPTQ, AWQ, GGUF); actually generates text on real models today | Silent OOM documented in issues; no budget enforcement or proof; no `admit` that refuses before allocation |
| **vLLM** | 92,862 | v0.30.0 (2026-09-22) | GPU serving at production scale; PagedAttention; continuous batching; speculative decoding; high throughput | GPU-only (CUDA/ROCm); no resource contract for 4–8 GB VRAM class; requires Python + venv + CUDA toolkit |
| **SGLang** | 36,527 | v0.5.20 (2026-09-18) | Fastest structured generation (RadixAttention, 2–5× vs vLLM on structured output); multi-modal | Same as vLLM: GPU-only, no contract for consumer hardware |
| **KTransformers** | 19,544 | v0.7.1 (2026-09-15) | CPU/GPU hybrid MoE; AMX int8 kernels; runs 671B on ~14 GB VRAM; 1.25–4× over llama.cpp | Requires 128 GB RAM (recommended), AMX CPU, CUDA/ROCm; primary target is 14–80 GB VRAM class |
| **Grevix/aura** | 4 | no release (2026-09-03) | Rust-native; OS-level enforcement via cgroup v2 / Win32 Job Objects (kernel-kills child if limit exceeded); wraps llama-server for real model generation; auto-tunes context window; Windows support | Enforces at runtime (post-spawn, not pre-flight); no typed pre-flight refusal (exit 2 + named constraint); silent auto-tune has no typed degradation record; requires llama-server at runtime; no stress harness that generalizes offline |

---

## Sizers / Profilers (Group B) — they predict; we predict *and* enforce

| Tool | Stars | Version | What it does better | Where fitsproof-rs differs |
|------|-------|---------|---------------------|----------------------------|
| **ridgepoint** | 1 (GitHub) | 0.1.2 PyPI (2026-09-08) | Best prediction accuracy (~1% MAPE on A100/H100); engine-aware VRAM (GQA/MLA correct to the byte); per-field `calibrated` flags | GPU-only (A100/H100); Python + pip; prediction only — no enforcement, no ceiling, no stress harness |
| **llm-inference-calculator** | 21 | no release (2026-09-09) | Two-phase roofline (prefill TTFT compute-bound, decode TPOT bandwidth-bound); MoE expert coverage | Prediction only; Python; no enforcement; no static binary |
| **llm-roofline** | 0 | no release (2026-06-20) | Minimal, readable decode throughput floor | Abandoned (0 stars); no enforcement; no KV term; no quantization-aware sizing |
| **hardware-aware-llm-runtime** | 0 | no release (2026-06-25) | Hardware-calibrated roofline; analytical optimal batch prediction | Abandoned (0 stars); prediction only |
| **llm-vram-calculator** | 1 | no release (2026-09-26) | Broad coverage (100+ models × 70+ GPUs) via public API | API-dependent; no offline mode; no enforcement; no CPU DRAM model |

---

## Correctness checkers (Group C) — orthogonal, complementary

| Tool | Stars | What it does better | Where fitsproof-rs differs |
|------|-------|---------------------|----------------------------|
| **detllm** | 20 (2026-08-20) | Determinism verification: capability-gated guarantee tiers (T0/T1/T2), run/batch variance measurement, repro packs | Focuses on determinism, not resource contracts; no predict/admit/verify/stress pipeline |

---

## What none of the above do

No tool in this table (including aura):

1. **Enforces a declared budget** with a `GlobalAlloc` wrapper — a ceiling is an error path, not
   an advisory prediction. (aura enforces via cgroup v2, which kills the child process; fitsproof-rs
   returns a typed `DoesNotFit` error before any allocation happens.)
2. **Refuses pre-flight** — `fitsproof admit` exits 2 with the binding constraint named before any
   subprocess is spawned, any weight is loaded, and any allocation occurs.  aura's enforcement
   requires the engine to start.
3. **Runs a portable stress harness** (≥20 configs, 0 violations, 0 silent mode changes) that runs
   offline, on any machine, with no GPU, no engine, and no subprocess.
4. **Prints both** allocator-counted peak and OS `VmHWM` plus the delta — so the overhead of
   the runtime itself is a first-class, visible number.
5. **Emits typed degradation records** — `FitsWithDegradation` carries a structured
   `degradation_steps` vector; a missing record when the verdict is degraded is a failing test.
   aura auto-tunes silently (this is framed as a feature, not a contract).

That combination on the **4–8 GB VRAM / 16–32 GB RAM** hardware class is the claim.

---

*Data source: GitHub REST API (unauthenticated) + PyPI JSON API, 2026-09-28T14:00 UTC.*
