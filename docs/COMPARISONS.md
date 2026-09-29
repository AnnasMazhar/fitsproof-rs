# COMPARISONS.md — fitsproof-rs vs the ecosystem

Last updated: 2026-09-29 (cycle 4, pass 9).  
Star counts from GitHub REST API, 2026-09-29T10:30 UTC.  
Source data: `docs/RESEARCH.md` cycles 2–4 ecosystem passes.

---

## The positioning in one sentence

fitsproof-rs is not an engine.  It is the contract layer that sits in front of any engine:
predict peak memory from a GGUF file, enforce a declared byte ceiling before any allocation,
refuse loudly with the binding constraint named, and prove compliance by measuring allocator peak
vs OS VmHWM.  No tool in the table below provides this combination for the 4–8 GB VRAM /
16–32 GB RAM hardware class.

---

## Where each tool beats us

| Tool | Stars | What it does better than fitsproof-rs |
|------|-------|---------------------------------------|
| **llama.cpp** | 129,845 | Mature; hundreds of architectures; fast CPU+GPU kernels; broad quant support; actually generates text; `--cpu-moe` MoE offload |
| **vLLM** | 92,913 | GPU production serving; PagedAttention eliminates KV fragmentation; speculative decoding; high throughput at scale |
| **SGLang** | 36,569 | Fastest structured generation (RadixAttention, 2–5× vs vLLM on structured output benchmarks) |
| **KTransformers** | 19,544 | Runs 671B DeepSeek-V3 on ~14 GB VRAM with 128 GB RAM; AMX int8 expert kernels; 1.25–4.09× decode over llama.cpp |
| **mistral.rs** | 7,722 | Production Rust inference; CUDA + Metal + CPU; Python bindings; broad model support; speculative decoding |
| **coderredlab/runNburn** | 28 | Runs models far exceeding RAM (222 GiB model on 32 GiB budget); CPU/CUDA/Metal/Vulkan; OpenAI-compat server |
| **signerless/llm-checker** | 2,998 | Largest model catalog (33k artifacts); MCP server (`hw_detect`, `ollama_plan`); calibrated bytes/param table |
| **Grevix/aura** | 4 | OS-level enforcement via cgroup v2 / Win32 Job Objects; catches mmap'd allocations that GlobalAlloc misses |
| **ridgepoint** | 1 | Best prediction accuracy (~1% MAPE on A100/H100); MLA-aware; per-field `calibrated` flags; GQA correct to the byte |
| **detllm** | 20 | Determinism verification; capability-gated guarantee tiers (T0/T1/T2); repro packs |

---

## Full comparison table (18 tools)

### Group A — Engines (run models; fitsproof-rs proves the contract before they run)

| Tool | Stars | Version | Last push | Gap fitsproof-rs fills |
|------|-------|---------|-----------|------------------------|
| **llama.cpp** | 129,845 | v0.5.0 | 2026-09-29 | Silent OOM in issues; no pre-flight admit; no typed refusal with exit 2 and named binding constraint |
| **vLLM** | 92,913 | v0.30.0 | 2026-09-29 | GPU-only; no contract for 4–8 GB VRAM class; Python + CUDA required |
| **SGLang** | 36,569 | v0.5.20 | 2026-09-29 | GPU-only; same class as vLLM |
| **KTransformers** | 19,544 | v0.7.1 | 2026-09-28 | 128 GB RAM recommended; CUDA/ROCm required; not for 16–32 GB class |
| **EricLBuehler/mistral.rs** | 7,722 | active | 2026-09-29 | No budget enforcement; OOM-kills (documented CVE: unbounded media fetch → OOM-kill); primarily GPU-focused |
| **Grevix/aura** | 4 | no release | 2026-09-03 | Runtime enforcement (kills child), not pre-flight; no typed degradation record; requires llama-server |
| **coderredlab/runNburn** | 28 | r17/v0.13.0 | 2026-09-28 | Runtime mmap-residency budget, not pre-flight typed refusal; no stress harness; no allocator_peak vs VmHWM delta |
| **SimonWaldherr/RustyLLM** | 7 | active | 2026-09-19 | No memory budget enforcement; MCP tools are inference-only (`generate`, `chat`, `embed`), not contract tools |

### Group B — Sizers / Profilers (predict; fitsproof-rs predicts *and* enforces)

| Tool | Stars | Version | Last push | Gap fitsproof-rs fills |
|------|-------|---------|-----------|------------------------|
| **ridgepoint** | 1 | 0.1.2 PyPI | 2026-09-08 | GPU-only (A100/H100); Python; prediction only — no enforcement, ceiling, or stress harness |
| **llm-inference-calculator** | 21 | no release | 2026-09-09 | Prediction only; Python; no enforcement |
| **llm-roofline** | 0 | no release | 2026-06-20 | Abandoned; prediction only; no enforcement |
| **hardware-aware-llm-runtime** | 0 | no release | 2026-06-25 | Abandoned; prediction only |
| **llm-vram-calculator** | 1 | no release | 2026-08-03 | API-dependent; no offline mode; no enforcement |
| **signerless/llm-checker** | 2,998 | v3.7.0 | 2026-09-29 | Node.js; prediction+selection only; no enforcement; no exit 2 on budget refusal |
| **kkpkishan/llm-infra-planner** | 11 | no release | 2026-09-24 | Web app only; no CLI; no enforcement; no CI integration |
| **09Catho/VRAMancer** | 1 | v1.2 | 2026-06-08 | Prediction only; no typed exit-2 refusal; early-stage (9 commits) |
| **Sheikyon/LLM-X** | 4 | PyPI | 2026-01-27 | Python; SafeTensors only (no GGUF); prediction only |

### Group C — Correctness / Determinism (orthogonal; complementary to fitsproof-rs)

| Tool | Stars | What it does better | Relationship |
|------|-------|---------------------|--------------|
| **detllm** | 20 | Determinism verification; capability-gated tiers (T0/T1/T2); repro packs | Complementary — use detllm to verify output determinism; use fitsproof-rs to verify memory budget compliance |

---

## The five properties no single tool combines

After surveying all 18 tools, these five properties are absent from every competitor:

1. **Pre-flight typed refusal with named binding constraint** — `fitsproof admit --budget-gb N`
   exits 2 before any allocation, subprocess, or engine load, naming `weight_bytes`, `kv_cache`,
   or `activation` as the binding constraint.  aura enforces at runtime (kills child); runNburn
   enforces residency at runtime; all sizers are prediction-only with no machine-readable exit code.

2. **Typed degradation records** — `FitsWithDegradation` carries a structured `degradation_steps`
   vector; a missing record fails the stress test.  Every engine auto-tunes silently.

3. **Portable offline stress harness** — `fitsproof stress` runs ≥20 configs with 0 violations
   and 0 silent mode changes, offline, on any machine, with no GPU, no engine, no subprocess.

4. **allocator_peak + VmHWM + delta** — `verify` prints both the Rust heap peak and the OS
   high-water mark plus their difference.  The delta documents mmap overhead that the allocator
   cannot see.  No tool in the table exposes this measurement.

5. **Target hardware class: 4–8 GB VRAM / 16–32 GB RAM as primary** — mistral.rs, runNburn, and
   RustyLLM work on consumer hardware but treat GPU as primary.  KTransformers requires 128 GB RAM.
   vLLM/SGLang require CUDA.

---

## Integration pattern: use fitsproof-rs alongside an engine

fitsproof-rs is not a replacement for any engine above.  It is a pre-flight gate:

```bash
# fitsproof pre-flight: typed refusal if weight_bytes + kv exceeds budget (< 50 ms)
fitsproof admit \
  --model "$MODEL" \
  --quant q4_k_m \
  --context 4096 \
  --budget-gb 4

# your engine (llama.cpp, mistral.rs, runNburn, etc.) — only reached if pre-flight passes
llama-cli -m "$MODEL" -c 4096 -n 200 -p "Explain GQA in one paragraph"
```

For runtime OS-level backstop on mmap'd memory (not covered by `TrackingAllocator`), combine with
AURA's cgroup v2 enforcement:

```bash
fitsproof admit --model "$MODEL" --quant q4_k_m --context 4096 --budget-gb 4 || exit 2
aura run --model "$MODEL" --budget 4gb
```

Step 1 catches misconfigurations before any subprocess starts and names the binding constraint.
Step 2 adds a kernel-level backstop for mmap'd tensor pages that `GlobalAlloc` does not see.
They defend different attack surfaces; running both costs ~1 s total.

---

## Honest limitations

fitsproof-rs v0.1 cannot do what the engines above do:

- **Does not generate text on real model weights.** `verify` runs the reference bundle (random
  weights). Full real-weight generation is v0.2 scope (weight tensor loader required).
- **No GPU support.** CPU-first. No CUDA kernels. Speed comparisons vs llama.cpp/vLLM are not
  made and would be lost if made.
- **No CUDA.** Built on hardware with no CUDA toolkit; compute 5.2 GPU (M2000) cannot run most
  CUDA kernels anyway.
- **`serve` / `mcp` / `pareto` exit 2 in v0.1.** Implemented in tests; CLI wiring is v0.2.
- **Q4_K bpw uses 4.0 bpw** (actual Q4_K is ~4.4375 bpw including superblock metadata; mixed
  quant with fp16 embeddings not yet modelled).

See `docs/ADOPTION.md §5` for the single most likely adoption blocker.

---

*Data sourced from `docs/RESEARCH.md` cycles 2–4.  Star counts verified 2026-09-29T10:30 UTC via GitHub REST API (unauthenticated).*
