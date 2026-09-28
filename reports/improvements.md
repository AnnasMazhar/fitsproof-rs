# Improvement Log — fitsproof-rs

## c2-p08-improve-1 (cycle 2, pass 8) — 2026-09-28

### Findings fixed

Both open findings from the cycle 1 adversarial review (ADV-1 and ADV-2) are addressed in this
pass. ADV-1 is the primary fix; ADV-2 is the test-quality hardening.

---

### ADV-1 — Binding constraint omits component breakdown

**Severity:** Minor (adversarial review c1-p10/p11 rating)

**Root cause:** `src/plan.rs:plan()` formatted the binding constraint string as:
```
needs 0.055 GB, budget 0.001 GB; no degradation fits
```
The README claim is "exit 2 on refusal, **binding constraint named**". "Named" implies the user
can determine *why* the config does not fit — which component (weight, kv, activation) is the
bottleneck. The old message told you the total, not the breakdown. A user with a 4 GB budget
refusing a 5 GB model could not tell from the message whether the problem is 4.8 GB of weights
(fix: lower quant) or 200 MB of KV cache blowing up (fix: shorter context).

**Fix applied:**

`src/plan.rs` — `plan()` DoesNotFit branch now calls `cost_est.weight_bytes`, `.kv_cache_bytes`,
and `.activation_bytes` (already computed) to include a per-component breakdown:

```
needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
```

Raw terminal verification:
```
$ ./target/debug/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
$ echo $?
2
```

**Test added:**

`src/plan.rs::tests::binding_constraint_includes_component_breakdown`

Fault detected: binding_constraint omits "weight=", "kv=", or "activation=" fields.

Inject the fault: remove any of the three `{component}=…` expansions from the format string.
The test asserts all three `contains("weight=")`, `contains("kv=")`, `contains("activation=")`.

This test would have caught ADV-1 immediately: the original format string would fail all three assertions.

---

### ADV-2 — Boundary test weak: `!= DoesNotFit` misses `<= → <` fault

**Severity:** Minor (test-quality hardening, not a production bug)

**Root cause:** `tests/adversarial.rs::budget_exactly_at_predicted_peak_admits` asserted:
```rust
assert_ne!(p.verdict, Verdict::DoesNotFit, …);
```

The fault ADV-2 describes is: changing `<=` to `<` in plan.rs makes `budget == predicted_peak`
fall through to `FitsWithDegradation` instead of `Fits`. The `!= DoesNotFit` assertion passes
for *both* `Fits` and `FitsWithDegradation`, so the test does not catch the fault.

**Fix applied:**

`tests/adversarial.rs::budget_exactly_at_predicted_peak_admits` — assertion changed from:
```rust
assert_ne!(p.verdict, Verdict::DoesNotFit, …)
```
to:
```rust
assert_eq!(p.verdict, Verdict::Fits, …)
```

The updated doc comment explains the specific fault it now catches.

**Fault injection verification:**

Inject the fault: change `predicted_peak <= budget_bytes` to `predicted_peak < budget_bytes`
in `src/plan.rs`. With the old test the suite passes (FitsWithDegradation != DoesNotFit).
With the new test:
```
assertion `left == right` failed: plan with budget == predicted peak must return Verdict::Fits …
  left:  FitsWithDegradation
  right: Fits
```

### Before/after metrics

| Metric | Before (c2-p07 eval) | After (c2-p08-improve-1) | Delta |
|--------|---------------------|--------------------------|-------|
| Tests run | 187 | 188 | +1 (new `binding_constraint_includes_component_breakdown`) |
| Test failures | 0 | 0 | 0 |
| ADV-1 status | open | **fixed** | resolved |
| ADV-2 status | open | **fixed** | resolved |
| Refusal message itemizes components | No (total only) | Yes (weight/kv/activation) | +UX diagnostic |
| Boundary fault coverage (`<= vs <`) | Not caught by `!= DoesNotFit` | Caught by `== Fits` | +1 fault covered |
| `cargo clippy -D warnings` | PASS | PASS | — |
| `cargo fmt --check` | PASS | PASS | — |
| Open adversarial findings (minor) | 2 | **0** | −2 |

### Raw terminal output

```
$ ~/.cargo/bin/cargo test --all-targets 2>&1 | grep -E "test result:|running [0-9]"
running 125 tests
test result: ok. 125 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 51.03s
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 28 tests
test result: ok. 28 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
running 23 tests
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.94s
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 3 tests
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 70.39s
running 6 tests
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

```
$ ~/.cargo/bin/cargo clippy --all-targets -- -D warnings 2>&1
    Checking fitsproof-rs v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.98s
(exit 0 — clean)
```

```
$ ./target/debug/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
$ echo $?
2
```

---



## c1-p08-improve-1 (cycle 1, pass 8) — 2026-09-28

### Finding fixed

**Severity:** Major correctness bug (underestimates KV cache by 4× for int4 models, 2× for int8 models).

**Root cause:** `kv_cache_bytes()` was called with the weight quantisation string (e.g. `"int4_sym"`)
as its precision argument.  In real LLM inference (llama.cpp, vLLM, Hugging Face Transformers), the
KV cache is stored at the *activation dtype* — float16 by default — regardless of weight quantisation.
A model with int4 weights does NOT get a 4-bit KV cache unless `--cache-quant` / `--kv-cache-dtype`
is explicitly passed.  The old code was misrepresenting this as the default.

**Impact on predictions (reference config: 6L, 2KV-heads, 64 head-dim):**

| Context | Weight quant | Old KV estimate (wrong) | New KV estimate (correct) | Error factor |
|---------|-------------|------------------------|---------------------------|--------------|
| 512     | fp32        | 6,291,456 bytes (fp32) | 3,145,728 bytes (fp16)    | 2× over      |
| 512     | int8        | 1,572,864 bytes (int8) | 3,145,728 bytes (fp16)    | 2× under     |
| 512     | int4        |   786,432 bytes (int4) | 3,145,728 bytes (fp16)    | 4× under     |

Note: fp32 KV was 2× over because fp32 KV caches are unusual; int4/int8 were under because
quantised KV is not the default.

**Real-model impact (hypothetical 7B int4 at 4096 context: 32L, 8KV-heads, 128 head-dim):**
- Old code KV estimate: 2 × 32 × 8 × 4096 × 128 × 0.5 bytes = **1,073 MB**
- Correct KV estimate: 2 × 32 × 8 × 4096 × 128 × 2 bytes = **4,295 MB**
- Difference: **4× underestimate** — old code would claim "fits in 5 GB" when actual need is ~8 GB.

**Tests that previously validated the wrong behaviour (now corrected):**
- `kv_cache_bytes_reference_fp32_known_answer` — was testing fp32 KV, not the real default
- `kv_cache_bytes_int8_is_quarter_of_fp32` — asserted int8 KV = ¼ fp32 KV (both wrong defaults)
- `kv_cache_bytes_int4_is_half_of_int8` — asserted int4 KV = ½ int8 KV (reinforcing wrong model)

### Fix applied

1. **`src/cost.rs:kv_cache_bytes`** — parameter renamed from `quant` to `kv_quant`, doc block
   updated to explain the fp16-by-default behaviour with citations (llama.cpp default, vLLM docs).

2. **`src/cost.rs:estimate`** — changed `kv_cache_bytes(cfg, context_len, quant)` to
   `kv_cache_bytes(cfg, context_len, "fp16")`, decoupling weight quant from KV precision.
   Added doc comment explaining the architectural default.

3. **All call sites updated** — 7 sites across `tests/value/test_incumbent_gap.rs`,
   `tests/stress.rs`, `src/admit.rs`, `src/main.rs`, `src/plan.rs` changed to pass
   `"fp16"` for KV cache precision, not the weight quant.

4. **`docs/PAPER-TRACEABILITY.md`** — updated to reference new test names and describe the fix.

### New tests added

`src/cost.rs::tests::kv_cache_bytes_reference_fp16_known_answer`
- Fault detected: factor 2 omitted from kv_cache_bytes, or wrong precision assumed.
- Ground truth: 2 × 6 × 2 × 512 × 64 × 2 (fp16=2 bytes) = **3,145,728 bytes** — hand-computed.
- This is the KAT that would have caught the original bug (fp32 was used as "known answer" before).

`src/cost.rs::tests::kv_cache_bytes_independent_of_weight_quant`
- Fault detected: kv_cache_bytes coupled to weight quant (the original bug).
- Proves: `kv_cache_bytes(cfg, 512, "fp16")` is identical whether the model uses fp32 or int4 weights.
- Also proves: explicit fp32 KV cache is 2× fp16 KV cache (correct ratio).
- **This is the regression test** — if anyone couples weight quant back to KV precision, this fails.

### Before/after metrics

| Metric | Before (c1-p07) | After (c1-p08-improve-1) | Delta |
|--------|----------------|--------------------------|-------|
| Tests run | 110 | 109 | −1 (3 wrong tests removed, 2 correct added) |
| Test failures | 0 | 0 | 0 |
| KV cache correctness (int4 model) | **4× underestimate** | Correct (fp16 default) | +4× accuracy |
| KV cache correctness (int8 model) | **2× underestimate** | Correct (fp16 default) | +2× accuracy |
| KV cache correctness (fp32 model) | **2× overestimate** | Correct (fp16 default) | +2× accuracy |
| Regression test for this bug | None | `kv_cache_bytes_independent_of_weight_quant` | +1 guard |
| `cargo clippy -D warnings` | PASS | PASS | — |
| `cargo fmt --check` | PASS | PASS | — |

The test count decrease (110 → 109) reflects removal of 3 tests that validated incorrect behaviour
and addition of 2 tests that validate correct behaviour.  The net test count is irrelevant; the
quality gate that matters is: can the tests detect the specific bug they claim to detect?

### Raw terminal output

```
$ ~/.cargo/bin/cargo test --all-targets 2>&1 | grep -E "test result:|running [0-9]"

running 78 tests
test result: ok. 78 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 28.54s
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 19 tests
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.67s
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 3 tests
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 45.49s
running 6 tests
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

```
$ ~/.cargo/bin/cargo clippy --all-targets -- -D warnings 2>&1
    Checking fitsproof-rs v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.44s
(exit 0 — clean)
```

---

## c2-p09-improve-2 (cycle 2, pass 9) — 2026-09-28

### Findings fixed

Four adoption-readiness issues found and fixed in this pass.

---

### IMP-1 — README real_model evidence block misleads on reproducibility

**Severity:** Major credibility gap (a skeptical reviewer's first finding)

**Root cause:** The README `## Headline evidence` section showed `cargo test --test real_model --
--nocapture` producing output with `Predicted peak: 3.209 GB | Verdict: Fits` as if the test
always runs. The test file actually checks `FITSPROOF_REAL_GGUF` env var first, then searches
`$HOME/.cache/` paths — and **skips** (passes) when no model is found. A stranger doing a fresh
clone would see `SKIP: no real GGUF found` rather than the output shown, making the evidence block
non-reproducible as presented.

Additionally, the refusal message shown in README was the old single-total format from before
ADV-1 was fixed:
```
REFUSED: needs 0.06 GB, budget 0.00 GB; no degradation fits
```
But the actual binary (post ADV-1 fix) now emits:
```
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
```

**Fix applied:**

README `## Headline evidence` section updated:
- `real_model` block now shows the `FITSPROOF_REAL_GGUF=...` invocation explicitly, states what
  the test proves (metadata → plan, not weight generation), and links to `EVIDENCE.md §5` for the
  PARTIAL label.
- Refusal message example updated to show the component breakdown format that the binary actually
  emits post-ADV-1 fix.

Raw terminal verification (refusal message format):
```
$ ./target/release/fitsproof admit --budget-gb 0.001; echo "EXIT:$?"
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
EXIT:2
```

---

### IMP-2 — `plan` command did not name valid quant values on error

**Severity:** Minor UX gap (actionability parity with `admit`)

**Root cause:** `cmd_admit()` named the valid quant values on failure:
```
  Valid quant values: none, float16, int8_sym, int4_sym, q4_k_m, q4_k_s, q8_0, q4_0
```
But `cmd_plan()` only printed a generic "try" hint with no valid-values list. A stranger who
mistyped a quant name would get an actionable message from `admit` but not from `plan`.

**Fix applied:**

`src/main.rs: cmd_plan()` — added the same valid-values hint line as `cmd_admit()`.

Raw terminal verification:
```
$ ./target/release/fitsproof plan --budget-gb 4 --quant badquant; echo "EXIT:$?"
fitsproof plan: unknown quantisation "badquant"
  Try: fitsproof plan --budget-gb 4 --quant q4_k_m --context 4096
  Valid quant values: none, float16, int8_sym, int4_sym, q4_k_m, q4_k_s, q8_0, q4_0
EXIT:2
```

---

### IMP-3 — No Makefile / CI integration example existed

**Severity:** Minor adoption gap (README had only inline bash; no real CI target)

**Root cause:** ADOPTION.md §2 had a GitHub Actions snippet but no Makefile target. README's
"How to plug it in" section had only bare bash. Teams using Makefile-driven CI (most teams) had
no copy-paste target.

**Fix applied:**

`Makefile` created with four targets:
- `preflight` — runs `fitsproof admit` with MODEL/BUDGET_GB/QUANT/CTX variables; no MODEL set → reference bundle check
- `stress` — runs `fitsproof stress`; 0 violations required
- `test` — runs `cargo test --all-targets`
- `musl` — builds static binary and verifies `ldd` reports not-a-dynamic-executable

README `## How to plug it in` section updated with Makefile example and GitHub Actions snippet.

Raw terminal verification:
```
$ PATH="$HOME/.cargo/bin:$PATH" make preflight
cargo build --release
    Finished `release` profile ...
fitsproof preflight: no MODEL set — running reference bundle check
ADMITTED: 0.021 GB predicted peak <= 4.000 GB budget (margin: 3979.3 MB)
```

```
$ PATH="$HOME/.cargo/bin:$PATH" make stress 2>&1 | tail -3
...
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=10.0 MB, median=200.0 MB, max=1000.0 MB.
```

---

### IMP-4 — `synthetic_machine_or_probe()` used VmHWM as gpu_memory_bytes

**Severity:** Minor code smell (wrong semantic; no functional impact on tests)

**Root cause:** `synthetic_machine_or_probe()` in `src/main.rs` set:
```rust
gpu_memory_bytes: read_vmhwm_bytes(), // reuse proc read as a sanity check
```
`read_vmhwm_bytes()` reads `/proc/self/status VmHWM` — the CLI process's own peak RSS — and was
being used as the GPU memory field. This is semantically wrong: the CLI process RSS at startup
is ~60 MB, not a GPU VRAM capacity. The field is not used in any budget calculation in v0.1
(it's recorded in `MachineProfile` for informational output from `probe`), so there was no
functional impact, but any reviewer reading the code would find it alarming.

**Fix applied:**

`src/main.rs: synthetic_machine_or_probe()` — `gpu_memory_bytes` set to `0` (correct for
CPU-only environment with no GPU, matching the stress harness and reference machine configs).
Unused `read_vmhwm_bytes` import removed.

---

### Before/after metrics

| Metric | Before (c2-p08) | After (c2-p09-improve-2) | Delta |
|--------|----------------|--------------------------|-------|
| Tests run | 188 | 188 | 0 |
| Test failures | 0 | 0 | 0 |
| README real_model block reproducible on clean clone | No (silent skip → confusing) | Yes (env-var instructions shown) | Fixed |
| Refusal message in README matches binary output | No (old format, pre-ADV-1) | Yes (component breakdown shown) | Fixed |
| `plan` names valid quant values on error | No | Yes | Fixed |
| CI Makefile integration example | None | `make preflight MODEL=… BUDGET_GB=4` | Added |
| `gpu_memory_bytes` in synthetic profile | VmHWM of CLI process (wrong) | 0 (correct for CPU-only) | Fixed |
| `cargo clippy -D warnings` | PASS | PASS | — |
| `cargo fmt --check` | PASS | PASS | — |

### Raw terminal output

```
$ ~/.cargo/bin/cargo test --all-targets 2>&1 | grep -E "test result:|running [0-9]"
running 125 tests
test result: ok. 125 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 54.87s
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 28 tests
test result: ok. 28 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
running 23 tests
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.89s
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
running 3 tests
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 45.00s
running 6 tests
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

```
$ ~/.cargo/bin/cargo clippy --all-targets -- -D warnings 2>&1
    Checking fitsproof-rs v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
(exit 0 — clean)
```

```
$ ./target/release/fitsproof admit --budget-gb 0.001; echo "EXIT:$?"
REFUSED: needs 0.055 GB (weight=0.053 GB, kv=0.002 GB, activation=0.000 GB), budget 0.001 GB; no degradation fits
EXIT:2
```

```
$ ./target/release/fitsproof plan --budget-gb 4 --quant badquant; echo "EXIT:$?"
fitsproof plan: unknown quantisation "badquant"
  Try: fitsproof plan --budget-gb 4 --quant q4_k_m --context 4096
  Valid quant values: none, float16, int8_sym, int4_sym, q4_k_m, q4_k_s, q8_0, q4_0
EXIT:2
```



### Finding fixed

**Severity:** Major credibility gap (fictional integration recipe — ADOPTION.md commands did not work).

**Root cause:** `cmd_plan()`, `cmd_admit()`, and `cmd_verify()` in `src/main.rs` ignored the `--model`
flag entirely and always called `ModelConfig::reference()`.  The ADOPTION.md integration recipe
showed `fitsproof plan --model /path/to/model.gguf` and `fitsproof admit --model ...` as if they
worked — they did not.  A stranger following the recipe would see the reference bundle prediction (a
tiny synthetic model), not a prediction for their model.

**Secondary bug:** `QuantBits::from_name()` did not recognise `q4_k_m` — the most common GGUF
quantisation format.  The ADOPTION.md recipe used `--quant q4_k_m` throughout, which would have
failed with "unknown quantisation" even once `--model` was fixed.

**Error messages:** Were generic (`fitsproof admit: unknown quantisation "badquant"`) with no
indication of valid values or how to fix the problem.

### Fixes applied

1. **`src/main.rs: load_model_config()`** — new helper that parses `--model <path>`, opens the file,
   calls `read_metadata()` + `metadata_to_model_config()` from `src/gguf.rs`, and returns a
   `ModelConfig` built from real GGUF architecture metadata.  Falls back to `ModelConfig::reference()`
   when `--model` is absent.  Actionable error messages at every failure point (file not found, invalid
   GGUF, unsupported architecture).

2. **`cmd_plan()`, `cmd_admit()`, `cmd_verify()`** — all three now call `load_model_config()` instead
   of hardcoding `ModelConfig::reference()`.

3. **`cmd_verify()` honesty note** — when `--model` is given, the binary now emits a note that v0.1
   runs the reference bundle for the allocation measurement (real-weight verify is v0.2), so the user
   is not surprised by the delta.

4. **`src/cost.rs: QuantBits::from_name()`** — added `q4_k_m`, `q4_k_s`, `q4_1` as recognised
   names mapping to 4-bit precision.  These are the standard llama.cpp Q4 variant names.

5. **Error messages** — `cmd_admit` and `cmd_plan` now name the valid quant values on failure.
   `cmd_verify` names the cause and suggests `--context` or `--quant` reduction on budget violation.
   `load_model_config` names the supported architectures on parse failure.

6. **`docs/ADOPTION.md`** — updated verify step to show the actual v0.1 output (with the honest
   reference-bundle note), fixing the fictional output block.

### Evidence

Real GGUF file: `/home/openclaw/.cache/fitsproof/gguf/Qwen3-1.7B-Q4_K_M.gguf`

- `fitsproof plan --model <path> --budget-gb 8 --quant q4_k_m --context 4096` → `Predicted peak: 3.664 GB, Verdict: Fits`
- `fitsproof admit --model <path> --budget-gb 8 --quant q4_k_m --context 4096` → `ADMITTED (margin: 4335.8 MB), exit 0`
- `fitsproof admit --model <path> --budget-gb 0.5 --quant q4_k_m --context 4096` → `REFUSED, exit 2`
- `fitsproof admit --budget-gb 4 --quant badquant` → actionable error naming valid quant values, exit 2

Full terminal output in EVIDENCE.md §21–24.

### Before/after metrics

| Metric | Before (c1-p08) | After (c1-p09-improve-2) | Delta |
|--------|----------------|--------------------------|-------|
| Tests run | 109 | 109 | 0 (no tests removed) |
| Test failures | 0 | 0 | 0 |
| `--model` flag works in CLI | No (silently ignored) | Yes (reads real GGUF) | Fixed |
| `--quant q4_k_m` accepted | No (unknown quant error) | Yes | Fixed |
| ADOPTION.md recipe runnable | No (fictional) | Yes (end-to-end verified) | Fixed |
| Error messages actionable | Partial | Yes (names valid values, hints on fix) | Improved |
| `cargo clippy -D warnings` | PASS | PASS | — |
| `cargo fmt --check` | PASS | PASS | — |
