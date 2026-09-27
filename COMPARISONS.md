# COMPARISONS.md

fitsproof-rs's claim is the **resource contract**: predict + enforce a byte ceiling +
measure proof + degrade explicitly, in a single static binary.  It does not claim
speed or model coverage superiority.

| Tool | What it does better than fitsproof-rs | Where fitsproof-rs differs |
|---|---|---|
| **llama.cpp** | Mature, broad model support (hundreds of architectures), fast CPU kernels (GGML, AVX2/AVX-512), broad quant support (GPTQ, AWQ, GGUF), actually generates text on real models today | No resource contract: silent OOM or silent CPU fallback documented; no enforce+prove mode |
| **vLLM** | GPU serving, PagedAttention, high throughput (production scale), broad model support | GPU-only (CUDA/ROCm required); no resource contract for the 4–8 GB VRAM class |
| **KTransformers** | CPU/GPU hybrid MoE inference, AMX kernels, runs 671B on ~14 GB VRAM, 1.25–4× over llama.cpp | Requires 128 GB RAM recommended, AMX CPU, CUDA/ROCm; no enforce+prove contract |
| **ridgepoint** (PyPI) | Calibrated VRAM prediction for GPU (A100/H100, ~1% MAPE), KV cache for GQA/MLA correct to the byte, per-field calibration flags | GPU-only; no CPU DRAM contract; no enforce mode (predict only) |
| **Strata** | Consumer packaging, one-click install, GUI | Requires 12 GB+ VRAM and 64 GB RAM; no resource contract |
| **Python fitsproof** (oracle) | Complete CLI, OpenAI server, MCP server, guard decorator, calibrate module | Advisory only — cannot enforce a byte ceiling (NumPy allocates outside contract control) |

## What none of the above do

No tool in this list:

1. Enforces a declared budget with a `GlobalAlloc` wrapper (a ceiling is an error path, not advisory)
2. Runs a stress harness (≥20 configs, 0 violations, 0 silent mode changes) to prove the contract
3. Prints both allocator-counted peak and OS `VmHWM` plus the delta
4. Degrades explicitly with an emitted record — no silent CPU fallback, no silent quant downgrade

That combination on the **4–8 GB VRAM / 16–32 GB RAM** hardware class is the claim.
