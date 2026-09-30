# fitsproof-rs

**An LLM memory contract enforcer: predict peak memory, enforce a byte ceiling, refuse loudly
when the budget is violated — as a single static binary with no Python, no venv, no CUDA.**

**Who it's for:** anyone running LLMs on 4–8 GB VRAM / 16–32 GB RAM hardware — the class every
mainstream engine treats as secondary. If you've ever hit a silent OOM or a silent CPU fallback,
this is for you.

> "Your engine tells you it fits. This one proves it — and refuses, loudly, when it doesn't."

## Quickstart (works on a clean Linux machine)

```bash
# Build (requires Rust ≥ 1.75)
cargo build --release
alias fitsproof=./target/release/fitsproof

# What fits in 4 GB?
fitsproof admit --budget-gb 4

# What fits in 1 GB? (will refuse and name why)
fitsproof admit --budget-gb 1
echo "exit $?"   # → exit 2

# Stress harness: 25 configs, 0 violations
fitsproof stress

# Measure this machine
fitsproof probe

# Plan a specific config
fitsproof plan --budget-gb 8 --quant int8 --context 4096

# Verify peak RSS against a declared budget
fitsproof verify --budget-gb 4

# OpenAI-compatible server — every response carries an admission_record
fitsproof serve --port 8080 &
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fitsproof/ref","messages":[{"role":"user","content":"hi"}],"budget_gb":4}'
# → {"admission_record":{"status":"admitted", ...}, ...}

# MCP server — any MCP client can call probe/plan/admit
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":4}}}' \
  | fitsproof mcp
# → {"result":{"content":[{"type":"text","text":{"status":"admitted",...}}]}}
```

⭐ If this saves you a silent OOM, a star helps others find it.

Static musl binary — no dynamic libraries, no runtime deps:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
ldd target/x86_64-unknown-linux-musl/release/fitsproof
# → "not a dynamic executable"
```

## Headline evidence

`fitsproof stress` runs 25 configurations against a declared budget and fails the build on any
budget violation or undocumented mode change:

```
$ fitsproof stress
...
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=10.0 MB, median=200.0 MB, max=1000.0 MB.
```

Refused configs name the binding constraint. They exit 2 so your CI can gate on it:

```
$ fitsproof admit --budget-gb 0.001
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
$ echo $?
2
```

A real GGUF model (Qwen3 1.7B Q4) is read and planned against a budget.
The test searches standard model cache paths (`$HOME/.cache/`) or uses `FITSPROOF_REAL_GGUF`.
On a machine without a cached model it prints a skip message and passes — the limitation is
recorded as PARTIAL in `docs/EVIDENCE.md`. To reproduce the full run:

```bash
# Point at any GGUF you have (Q4_K_M of any 1–7B model works):
FITSPROOF_REAL_GGUF=/path/to/model.gguf cargo test --test real_model -- --nocapture
```

Output on a machine with a Qwen3 1.7B Q4 model cached:

```
$ FITSPROOF_REAL_GGUF=~/.cache/qmd/models/hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf \
    cargo test --test real_model -- --nocapture
Reading GGUF: ~/.cache/qmd/models/hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf
  GGUF version: 3 | Tensor count: 311 | Architecture: qwen3
  Predicted peak: 2.009 GB | Budget: 4.000 GB | Verdict: Fits
test real_gguf_model_plan_succeeds ... ok
```

**What this proves:** `plan()` reads real GGUF architecture metadata and returns a correct
prediction. It does *not* load tensor weights or run generation — that is a v0.2 scope item.
`EVIDENCE.md §5` documents this as PARTIAL and explains what the v0.2 weight loader enables.

`fitsproof verify` prints both the allocator-counted peak and the OS high-water mark (`VmHWM`),
plus the delta — so the overhead of the runtime is a visible number, not a footnote:

```
$ fitsproof verify --budget-gb 4
ADMITTED: 0.055 GB predicted peak <= 4.000 GB budget (margin: 3944.9 MB)
allocator_peak: 0.000 GB
VmHWM:          0.057 GB
delta:          +0.1 MB (VmHWM - allocator_peak)
budget:         4.000 GB
budget_respected: true
```

Full evidence register with raw terminal output: `docs/EVIDENCE.md`.

## How to plug it in

**In CI:** run `fitsproof admit --budget-gb <N>` before loading any model. It exits 2 on refusal;
your pipeline stops before the OOM.

**In a script:**
```bash
fitsproof admit --budget-gb 6 || { echo "won't fit — refusing to load"; exit 1; }
```

**Read a real GGUF before loading it:**
```bash
fitsproof plan --model /path/to/model.gguf --budget-gb 8
```

**Full llama.cpp integration — the complete pattern (no silent OOM):**
```bash
#!/usr/bin/env bash
# pre-flight check before every llama-cli invocation
set -euo pipefail

MODEL=/path/to/Qwen3-1.7B-Q4_K_M.gguf
BUDGET_GB=4.0
CTX=4096

# Step 1: fitsproof pre-flight (< 50 ms, no engine required)
# Exit 2 names the binding constraint (weight/kv/activation) and stops the script.
fitsproof admit --model "$MODEL" --quant q4_k_m --context $CTX --budget-gb $BUDGET_GB

# Step 2: launch llama.cpp only if pre-flight passes
llama-cli -m "$MODEL" -c $CTX -n 200 -p "Explain GQA in one paragraph"
```

If fitsproof refuses (e.g. tight budget on a 1.7B model):
```
REFUSED: needs 2.420 GB (weight=1.950 GB, kv=0.470 GB, activation=0.000 GB), budget 2.000 GB; no degradation fits
```
`kv=0.470 GB` contributes 0.5 GB at 4096 context → fix: reduce `CTX=2048`.

Full recipe with failure modes: `docs/ADOPTION.md §2`.

**Stress your configuration space:**
```bash
fitsproof stress  # fails the build on any violation; safe to run in CI
```

**In a Makefile (the `Makefile` in this repo is a working example):**
```makefile
preflight:
	fitsproof admit \
	    --model "$(MODEL)" \
	    --budget-gb $(BUDGET_GB) \
	    --quant $(QUANT) \
	    --context $(CTX)
```
```bash
make preflight MODEL=/path/to/llama-7b-q4.gguf BUDGET_GB=4 QUANT=q4_k_m CTX=4096
# → ADMITTED or REFUSED with named binding constraint; exit 0 or 2
```

**In GitHub Actions:**
```yaml
- name: Memory contract pre-flight
  run: |
    make preflight MODEL=${{ env.MODEL_PATH }} BUDGET_GB=4.0 QUANT=q4_k_m CTX=4096
    # exit 2 fails the job and names the constraint (weight/kv/activation)
```

## What this is NOT

- **Not a new CUDA kernel.** CPU-first, correctness-first scalar path. No GPU kernels. Speed
  comparisons vs llama.cpp/vLLM/KTransformers are irrelevant and intentionally omitted.
- **Not faster than llama.cpp or vLLM.** Those are faster, more mature, cover more hardware and
  models, and actually generate text on real weights today. `docs/COMPARISONS.md` says exactly where
  each one beats us.
- **Not a training tool.** Inference memory planning only.
- **No CUDA.** This codebase was built on a machine with no CUDA toolkit. Not a limitation of the
  design — CPU-first is the point. The hardware class being served (4–8 GB VRAM / 16–32 GB RAM)
  often has compute 5.2 GPUs that cannot run most CUDA kernels anyway.

## What the Python `fitsproof` is (the oracle)

The Python edition (`fitsproof`) is the **reference oracle**. It is not a runtime dependency —
the shipped binary requires no Python. The oracle relationship is dev-time only: contract
semantics were ported from Python to Rust, and the Rust versions were checked against Python
outputs on shared test inputs.

Key differences that justify the Rust port:

| Property | Python edition | This repo |
|---|---|---|
| Budget enforcement | predicted + measured after the fact | enforced by `TrackingAllocator` — a ceiling is an error, not an OOM |
| Install | Python + venv + NumPy | one static binary |
| Mutation testing | pytest + mutmut on NumPy plumbing | `cargo-mutants` on contract logic |

## Architecture

```
src/
  allocator.rs       TrackingAllocator (GlobalAlloc wrapper, atomic CAS ceiling), DoesNotFit error
  model.rs           ModelConfig (architecture parameters, reference bundle config)
  probe.rs           MachineProfile, STREAM-triad bandwidth + GEMM measurement
  cost.rs            Analytical roofline cost model (weight/KV/activation bytes, tok/s)
  plan.rs            Plan, Verdict (fits / fits_with_degradation / does_not_fit)
  admit.rs           AdmitRecord, admit() — the enforcement point
  verify.rs          VerifyRecord, verify_run() — allocator_peak + VmHWM + delta
  gguf.rs            Minimal GGUF version 1, 2, 3 header reader → ModelConfig
  client.rs          FitsproofClient, guard() — Rust API surface (v0.2 mandate: binary + plugin)
  serve.rs           OpenAI-compatible HTTP server (admission_record on every response)
  mcp.rs             MCP stdio server (probe / plan / admit tools, JSON-RPC 2.0)
  pareto.rs          Pareto frontier sweep (quant × context_len)
  engine/
    ops.rs           RMSNorm, RoPE, GQA attention, SwiGLU FFN, KV cache, linear
    quant.rs         int8_sym + int4_sym symmetric quantisation
    sampling.rs      greedy, temperature/top-k sampling, xoshiro256** RNG
    tensor.rs        softmax, silu
    transformer.rs   Weights, Transformer (forward pass + generate), reference bundle
tests/
  smoke.rs           Binary smoke tests (version, unknown command)
  stress.rs          25-config stress harness (acceptance criteria)
  adversarial.rs     48 byzantine/edge-case tests (overflow, malformed input, boundary faults,
                     race condition close, FitsproofClient API attacks)
  cmd_integration.rs 44 CLI integration tests (subcommand flags, error messages, exit codes)
  contract_mutants.rs  33 mutation-killing tests targeting cost/plan/admit arithmetic
  real_model.rs      Real GGUF model test (plan against real weights)
  value/
    test_incumbent_gap.rs  The two mandatory zero-case proofs (refused + degraded)
```

## CLI reference

```
fitsproof probe    # measure this machine (memory bandwidth, GEMM rate, RAM/VRAM)
fitsproof plan     # predict peak memory for a configuration and a budget
fitsproof admit    # admit / degrade loudly / refuse (exit 2 on refusal, binding constraint named)
fitsproof verify   # measure peak RSS + VmHWM, print both + delta, assert <= budget
fitsproof stress   # >=20 configs: zero violations, zero silent mode changes
fitsproof serve    # OpenAI-compatible HTTP server
fitsproof mcp      # MCP stdio server (JSON-RPC 2.0 over stdin/stdout)
fitsproof pareto   # Pareto frontier sweep (quant × context_len)

fitsproof --version
fitsproof --help
```

## Build

```bash
cargo build --release
./target/release/fitsproof --help
```

Static musl binary (no dynamic libraries):

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
ldd target/x86_64-unknown-linux-musl/release/fitsproof  # "not a dynamic executable"
```

## Comparisons

See `docs/COMPARISONS.md` for the full table with current star counts and release dates. Short version:

| Tool | What it does better than fitsproof-rs |
|---|---|
| llama.cpp | Mature, broad model support, fast CPU kernels, broad quant support, actually generates text |
| vLLM | GPU serving, PagedAttention, high throughput |
| KTransformers | CPU/GPU hybrid MoE, AMX, runs 671B on 14 GB VRAM |
| mistral.rs | Production Rust inference, GPU/CPU, Python bindings, broad model support |
| ridgepoint | Calibrated VRAM/roofline for GPU (A100/H100, ~1% MAPE) |

fitsproof-rs's position: none of the above couples a resource contract (predict + enforce a
byte ceiling + measure proof + degrade explicitly) into a single binary, for the unserved
4–8 GB VRAM / 16–32 GB RAM hardware class.

## Limitations

These are honest. A repo with no stated limitations is not credible.

- **No CUDA.** CPU-first. No GPU kernels. Speed comparisons vs llama.cpp/vLLM are irrelevant.
- **Reference engine is scalar.** The forward pass is a correctness-first scalar path.
  It is slower than hand-tuned C++. That is intentional; the product is the contract, not speed.
- **Real-model generation (v0.2).** The GGUF reader extracts architecture metadata and produces
  `plan()` predictions for real models. Loading tensor weights and running `generate()` on real
  weights requires a full weight tensor loader — a v0.2 scope item.
- **Real-weight generation via `serve` / `mcp`.** The HTTP server and MCP server run the reference bundle (randomly-initialised weights). Serving real GGUF weights requires the full weight tensor loader — use `fitsproof plan` or `admit` with `--model` for real-model contract checks.
- **KV cache bandwidth not in decode formula.** The decode formula counts weight streaming;
  KV cache access adds bandwidth at long contexts (known limitation, documented in the Python
  oracle too).

## Licence

MIT.
