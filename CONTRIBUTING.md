# Contributing to fitsproof-rs

This repo is a single static binary that enforces LLM memory contracts. The bar is
correctness — every change must preserve the guarantee that the contract never degrades
silently.

## Before you start

1. Read `COMPARISONS.md` to understand what this is and is not trying to be.
2. Read `docs/EVIDENCE.md` — the evidence register is the definition of done.
3. Read `docs/PAPER-TRACEABILITY.md` — every core formula must map to a source.

## Development setup

```bash
# Requires Rust ≥ 1.75
git clone https://github.com/AnnasMazhar/fitsproof-rs
cd fitsproof-rs
cargo build
cargo test --all-targets       # must be green before and after your change
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

No other dependencies. No Python. No CUDA toolkit.

## What makes a good PR

**Green contract:** `cargo test` must pass. The stress harness (25 configs, 0 violations,
0 silent mode changes) is a hard gate — do not weaken it.

**Named fault:** every test must carry a comment naming the fault it detects. If you cannot
name the fault a test would catch, delete it. The QUALITY-CONTRACT is at
`/portfolio/specs/QUALITY-CONTRACT.md`.

**Known-answer validation:** any change to a numerical formula must include a hand-computed
known-answer test. Show your arithmetic in the test comment, or cite the equation from the paper.

**No silent degradations:** a mode change (quant downgrade, context truncation) must always
produce a `DegradationStep` record. The test `fits_with_degradation_applied_degradation_is_some`
in `tests/adversarial.rs` guards this.

**Evidence updated:** if you change a claim in README, update `docs/EVIDENCE.md` with the
new raw output.

## What we will not merge

- Speed benchmarks against llama.cpp, vLLM, or KTransformers. We explicitly disclaim speed
  superiority — see `COMPARISONS.md`.
- CUDA or GPU kernels. CPU-first is the design choice, not a gap to fill.
- New CLI flags that are not tested end-to-end.
- Stubs or placeholder functions. Depth over breadth — omit rather than stub.

## Good first issues

Look for issues labelled `good first issue`. They are typically:

- Adding a known-answer test for a formula that only has property-based tests.
- Extending the GGUF reader to handle a new key-value type.
- Adding a hardware entry to the compatibility notes in `docs/ADOPTION.md`.
- Improving error messages to name the binding constraint more precisely.

## Code style

- `cargo fmt` before committing. The CI runs `cargo fmt --check`.
- `cargo clippy -D warnings` must be clean.
- No `unwrap()` in library code. Use typed errors and the `?` operator.
- No bare `expect("msg")` on paths that are reachable from user input.
- No AI attribution in commits (`Co-Authored-By: ...`).
- Conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.

## Filing an issue

Please include:
- The exact command you ran.
- The raw output (not a summary).
- Your hardware: RAM, GPU VRAM (if any), OS.

Budget violation reports (the tool admitted something that then OOMed) are the highest priority.
Include the `fitsproof probe` output alongside the `admit` command.
