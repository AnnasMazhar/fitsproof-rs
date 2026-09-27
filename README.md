# fitsproof-rs

**The compiled edition of [fitsproof](https://github.com/AnnasMazhar/fitsproof): prove your local
LLM fits in memory — or get a loud refusal instead of a silent OOM.**

The Python edition established the contract. This repo makes it *enforceable*: a single static
binary, no Python, no venv, no torch — with a byte-counting allocator that turns the memory
budget from a prediction into a hard ceiling, and a proof harness that measures what the process
actually used against what was declared.

> Your engine tells you it fits. This one proves it — and refuses, loudly, when it doesn't.

Status: **under construction.** The v0.1 scope, the language decision and the definition of done
are in the campaign spec (`specs/fitsproof-rs.md`). This README will not claim anything the repo
cannot prove; the claim register is `docs/EVIDENCE.md`.

## Why a compiled edition

| | Python edition | this repo |
|---|---|---|
| Budget | predicted, then measured after the fact | **enforced** by the allocator — over-budget allocation is an error, never an OOM |
| Install | Python + venv + NumPy | one static binary |
| Calibration probe | minutes | seconds |
| Mutation testing | pytest + mutmut on array plumbing | `cargo-mutants` on the contract logic |

## What this is not

Not a kernel. Not faster than llama.cpp, vLLM or KTransformers — those are faster, more mature
and cover more hardware, and `COMPARISONS.md` will say so. No CUDA: this machine has no CUDA
toolkit, and the target is the CPU-first 4–8 GB VRAM / 16–32 GB RAM class that mainstream engines
ignore or silently fall back from.

## Build

```bash
cargo build --release
./target/release/fitsproof --help
```

## Licence

MIT.
