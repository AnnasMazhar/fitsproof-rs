# fitsproof-rs

**The compiled edition of [fitsproof](https://github.com/AnnasMazhar/fitsproof): prove your local
LLM fits in memory — or get a loud refusal instead of a silent OOM.**

The Python edition established the contract. This repo makes it *enforceable*: a single static
binary, no Python, no venv, no torch — with a byte-counting allocator that turns the memory
budget from a prediction into a hard ceiling, and a proof harness that measures what the process
actually used against what was declared.

> Your engine tells you it fits. This one proves it — and refuses, loudly, when it doesn't.

Status: **v0.1 — allocator + contract layer + reference engine working; real-model generation
(not just planning) is a v0.2 scope item.** See `docs/EVIDENCE.md` for the claim register.

## Headline evidence

`fitsproof stress` runs 25 configurations against a declared budget and fails the build on any
violation or undocumented mode change. Real output:

```
$ fitsproof stress
...
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=10.0 MB, median=200.0 MB, max=1000.0 MB.
```

Refused configs name the binding constraint. Real output:

```
$ fitsproof admit --budget-gb 0.001
REFUSED: needs 0.06 GB, budget 0.00 GB; no degradation fits
$ echo $?
2
```

Real model (Qwen3 1.7B, GGUF on disk) reads correctly and plans:

```
$ cargo test --test real_model -- --nocapture
  GGUF version: 3 | Tensor count: 311 | Architecture: qwen3
  Predicted peak: 3.209 GB | Budget: 4.000 GB | Verdict: Fits
test real_gguf_model_plan_succeeds ... ok
```

## What this is NOT

fitsproof-rs is not a new CUDA kernel. It does not claim speed superiority over
llama.cpp, vLLM, or KTransformers. Those engines are faster, more mature, cover
more hardware, and support more models. `COMPARISONS.md` names each one and states
where it beats us.

The engine exists to make the contract real and testable: a scalar, correctness-first
Rust runtime that runs offline with no GPU and no CUDA toolkit. **No CUDA** — this
machine has no CUDA toolkit.

## Relationship to the Python `fitsproof` (the oracle)

The Python edition (`fitsproof`) is the **reference oracle**. It is not a runtime
dependency — the shipped binary requires no Python. The oracle relationship is
dev-time only: contract semantics were ported from Python to Rust, and the Rust
contract logic was reviewed for consistency with the Python edition's semantics.
A differential oracle test comparing Rust and Python outputs on shared test inputs
is a planned v0.2 item (see `docs/PAPER-TRACEABILITY.md` §9).

Key differences that justify the Rust port:

| Property | Python edition | This repo |
|---|---|---|
| Budget enforcement | predicted + measured after the fact | enforced by `TrackingAllocator` — a ceiling is an error, not an OOM |
| Install | Python + venv + NumPy | one static binary |
| Mutation testing | pytest + mutmut on NumPy plumbing | `cargo-mutants` on contract logic |

**Ceiling enforcement semantics.** When `verify`/`stress` install a ceiling via
`ALLOCATOR.set_ceiling(budget_bytes)`, any allocation that would push the process
past the budget returns null.  Rust's `handle_alloc_error` turns that into an
immediate process abort (exit 134, SIGABRT) rather than a silent over-budget run.
This is a loud, observable signal — not a typed `DoesNotFit` error (which requires
callers to use the advisory `check()` path before allocating).  The abort
semantics are the enforcement story for v0.1; a typed `Err(DoesNotFit)` choke-point
through `try_reserve` is a v0.2 item.

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

## CLI

```
fitsproof probe    # measure this machine (memory bandwidth, GEMM rate, RAM/VRAM)
fitsproof plan     # predict peak memory for a configuration and a budget
fitsproof admit    # admit / degrade loudly / refuse (exit 2 on refusal, binding constraint named)
fitsproof verify   # measure peak RSS + VmHWM, print both + delta, assert <= budget
fitsproof stress   # >=20 configs: zero violations, zero silent mode changes
fitsproof serve    # OpenAI-compatible HTTP server [v0.2]
fitsproof mcp      # MCP stdio server [v0.2]
fitsproof pareto   # Pareto frontier sweep [v0.2]

fitsproof --version
fitsproof --help
```

## Architecture

```
src/
  allocator.rs       TrackingAllocator (GlobalAlloc wrapper), DoesNotFit error
  model.rs           ModelConfig (architecture parameters, reference bundle config)
  probe.rs           MachineProfile, STREAM-triad bandwidth + GEMM measurement
  cost.rs            Analytical roofline cost model (weight/KV/activation bytes, tok/s)
  plan.rs            Plan, Verdict (fits / fits_with_degradation / does_not_fit)
  admit.rs           AdmitRecord, admit() — the enforcement point
  verify.rs          VerifyRecord, verify_run() — allocator_peak + VmHWM + delta
  gguf.rs            Minimal GGUF version 1, 2, 3 header reader → ModelConfig
  engine/
    ops.rs           RMSNorm, RoPE, GQA attention, SwiGLU FFN, KV cache, linear
    quant.rs         int8_sym + int4_sym symmetric quantisation
    sampling.rs      greedy, temperature/top-k sampling, xoshiro256** RNG
    tensor.rs        softmax, silu
    transformer.rs   Weights, Transformer (forward pass + generate), reference bundle
tests/
  smoke.rs           Binary smoke tests (version, unknown command)
  stress.rs          25-config stress harness (acceptance criteria)
  real_model.rs      Real GGUF model test (plan against real weights)
```

## Limitations

These are honest. A repo with no stated limitations is not credible.

- **No CUDA.** CPU-first. No GPU kernels. Speed comparisons vs llama.cpp/vLLM are irrelevant.
- **Reference engine is scalar.** The forward pass is a correctness-first scalar path.
  It is slower than hand-tuned C++. That is intentional; the product is the contract, not speed.
- **Real-model generation (v0.2).** The GGUF reader extracts architecture metadata and
  produces `plan()` predictions for real models. Loading tensor weights and running
  `generate()` on real weights requires a full weight tensor loader — a v0.2 item.
  See `docs/EVIDENCE.md` §5 (PARTIAL).
- **serve / mcp / pareto not implemented.** Exit 2 with message in v0.1.
- **KV cache bandwidth not in decode formula.** The decode formula counts weight streaming;
  KV cache access adds to bandwidth at long contexts (known limitation, documented in the
  Python oracle too).
- **Calibrate module.** The `±20%` CI is a placeholder; a fitted calibration (from on-device
  measurements) is a v0.2 item.
- **Off-allocator bypass class.** `TrackingAllocator` counts only allocations routed through
  Rust's `GlobalAlloc`. Three classes bypass it:
  - **`std::alloc::System` / direct `mmap`**: 64 MiB via `System.alloc_zeroed` produces
    `VmHWM ≈ 68 MB` while the allocator records `tracking_current ≈ 554 B` — a ~67 MB gap.
  - **Thread stacks**: each thread maps 8 MiB via the OS (Linux default); not counted by
    the global allocator. A 32 MiB stack-local buffer produces `VmHWM ≈ 35 MB` vs
    `tracking_current ≈ 665 B`.
  - **Static data** (`.bss` / `.data` / `mmap`-ed DSOs): counted by the OS but not the
    allocator.
  The **backstop** for this class is the `VmHWM <= budget` gate in `verify` (F2 fix):
  any off-allocator memory that causes a real over-budget condition will be caught there.
  `tests/attack_harness.rs` measures and asserts the gap is ≥ 60 MB, so the bypass is
  visible and documented rather than hidden.
  Numbers from: independent review REVIEW-muse-spark.md F5 (2026-09-27, ThinkStation P500).

## Comparisons

See `COMPARISONS.md` for the full table. Short version:

| Tool | What it does better than fitsproof-rs |
|---|---|
| llama.cpp | Mature, broad model support, fast CPU kernels, broad quant support, actually generates text |
| vLLM | GPU serving, PagedAttention, high throughput |
| KTransformers | CPU/GPU hybrid MoE, AMX, runs 671B on 14 GB VRAM |
| ridgepoint | Calibrated VRAM/roofline for GPU (A100/H100, ~1% MAPE) |
| Strata | Consumer packaging, one-click install |

fitsproof-rs's position: none of the above couples a resource contract (predict + enforce a
byte ceiling + measure proof + degrade explicitly) into a single binary, on the unserved
4–8 GB VRAM / 16–32 GB RAM hardware class.

## Licence

MIT.
