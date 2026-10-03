# Independent review — fitsproof-rs v0.1 (memory contract + prove step)

| Field | Value |
|---|---|
| Reviewer model | `opencode/mimo-v2.6-flash-free` |
| Scope | `8d5217f..8b95487` (3 commits) on branch `review/mimo` |
| Repo | `fitsproof-rs` v0.1.0, lib target `fitsproof` |
| Date | 2026-09-27 |
| Specs read | `fitsproof-rs.md`, `REAL-WORLD-PROOF.md`, `PROOF-AND-RELEASE.md` (in `~/portfolio/specs/`) |
| Method | full command reproduction, 6 adversarial attack binaries, mutation run, evidence cross-check |
| Files touched by reviewer | `docs/REVIEW-mimo-v2.6-flash-free.md` only. `src/`, `tests/`, `Cargo.toml` untouched. Attack harness lives outside the repo at `/tmp/opencode/fitsproof-attack`. Nothing pushed. |

**Verdict: NO-SHIP as a proof of the memory contract. SHIP as a foundation.**
Full paragraph in §7.

---

## 1. What I ran (reproduction of the documented claims)

| Command | Result | Matches docs? |
|---|---|---|
| `cargo fmt --all -- --check` | exit 0 | yes |
| `cargo clippy --all-targets -- -D warnings` | exit 0 | yes |
| `cargo test` | 67 lib + 1 real_model + 2 smoke + 3 stress = 73, 1 doc-test ignored, 0 failures | yes (EVIDENCE rows 1–4) |
| `cargo build --release` | ok | yes |
| `fitsproof --help` / `--version` | exit 0, 0.1.0 | yes |
| `fitsproof probe` | bandwidth/GEMM/RAM printed | yes (row 8) |
| `fitsproof plan` | predicted peak + budget verdict | yes (row 4) |
| `fitsproof admit --budget-gb 4` | ADMITTED, 0.057 GB, exit 0 | yes (row 4) |
| `fitsproof admit --budget-gb 0.001` | REFUSED, exit 2, binding constraint named | yes (row 1) |
| `fitsproof verify` | prints allocator_peak / VmHWM / delta / budget / budget_respected | yes (row 6) |
| `fitsproof stress` | 25 configs, 0 violations, 0 silent mode changes, min margin 10.0 MB | yes (row 3) |
| `fitsproof serve` / `bogus` | exit 2 | yes |
| `cargo test --test real_model -- --nocapture` | real GGUF read: 311 tensors, qwen3, predicted 3.209 GB / budget 4 GB | **row 5 path not reproducible** (see F12) |
| `scripts/check_no_internal_refs.sh` | CLEAN, exit 0 | yes |
| `cargo mutants -f src/admit.rs` | run (cargo-mutants **27.1.0 installed**) | **EVIDENCE open item says "not installed"** (F12) |

Predicted-vs-measured (single process, byte-accurate, attack bin `measure`):

| config | predicted total_peak | measured VmHWM | measured/predicted |
|---|---|---|---|
| ctx=8 | 53,555,712 B | 56,913,920 B | **1.063** |
| ctx=512 (default) | 56,652,288 B | 58,310,656 B | **1.029** |
| ctx=4096 | 78,672,384 B | 58,314,752 B | **0.741** |

Python sibling comparison: ~39 MB predicted vs ~300 MB measured (≈7.7×). The Rust
accounting is far tighter — but the spread across `--context` (0.74–1.06) is a
contract bug, not noise: at ctx=8 the prediction is **below** measurement (F11).
Absolute allocator peak was 56,679,680 B (= weights 53,503,104 + KV-at-max_seq_len
3,145,728 + slack), i.e. the engine ignores `--context` and allocates KV at
`max_seq_len` regardless.

---

## 2. Findings

Severity: **BLOCKER** = headline contract false or unprovable; **HIGH** = spec promise
broken; **MEDIUM** = evidence/hygiene; **LOW** = hardening.

| ID | Sev | Finding | Evidence | Status |
|---|---|---|---|---|
| F1 | BLOCKER | **The ceiling is never installed.** `set_ceiling()` and `check()` have **zero product callers** — only allocator unit tests. `admit`, `verify`, `stress`, `main` never install a ceiling, so README's headline ("turns the memory budget from a prediction into a hard ceiling") and spec §3.1 ("`admit` installs a ceiling") are false in the shipped binary. | `grep -rn set_ceiling src/ tests/` → only `src/allocator.rs:126` (def) + 5 test sites; `.check(` → only `src/allocator.rs:382,397,419`. Live demo: a 20 MB declared budget allocates 54 MB with no refusal (F2). | CONFIRMED |
| F2 | BLOCKER | **`verify` certifies a budget the process demonstrably exceeds.** Windowed delta (`peak_after − peak_before`) is measured *after* weights are allocated, so it is structurally ≈0; `os_budget_respected` is computed and **never printed or counted**; `budget_respected: true` is printed on screen directly under `VmHWM: 0.057 GB` with `budget: 0.020 GB`, exit 0. Stress compounds it: CLI uses `eff_budget = declared` for clean configs and `predicted*4` for degraded ones; the test harness declares `peak*2` up front. | 20 MB run: `DEGRADED … New predicted peak: 0.015 GB` / `allocator_peak: 0.000 GB` / `VmHWM: 0.057 GB` / `budget_respected: true` / **exit 0** / `/usr/bin/time -v` max RSS **55,480 KiB**. 16 MB run: identical. ctx8 @ 0.055 GB: admit ADMITTED, verify exit 0 with RSS 55,432 KiB > 55 MB budget. `src/verify.rs:158-164`, `src/main.rs:306-309`, `tests/stress.rs:84-148`. | CONFIRMED |
| F3 | HIGH | **The degradation record describes work the engine does not do.** `admit` reports "Use int8_sym instead of none", `verify` then runs `Weights::reference` (**fp32**), allocating the full fp32 footprint while claiming a 15 MB degraded prediction. The record is fiction: spec DoD 4 ("admitted only after an emitted degradation") is satisfied in *record* only, never in *execution*. | 20 MB run above: degraded claim 0.015 GB, RSS 0.055 GB. Engine always instantiates fp32 reference weights; the quant module is never wired into `transformer`. | CONFIRMED |
| F4 | HIGH | **Over-ceiling allocation kills the process; it does not return a typed error.** With a 64 MB ceiling installed, an ordinary `Vec` growth aborts: `memory allocation of 1048576 bytes failed`, **SIGABRT, exit 134** — not `DoesNotFit`, not exit 2. Spec §3.1's "typed error … never an OOM kill" holds only for `check()`, which nothing calls. `check()` is also TOCTOU-unsafe (below). | Attack `ceiling_null`: `check(1MB)` → typed `DoesNotFit` (Ok); `raw_alloc` → null; `box` → **exit 134**. Attack `realloc_growth`: direct realloc correctly returns NULL, but `Vec` growth → **exit 134**. | CONFIRMED |
| F5 | HIGH | **Non-allocator memory defeats the ceiling entirely.** 16 threads × 7 MB default stacks = **117,784,576 B** of resident memory under a **64 MB** ceiling while the heap grew 3,380 B and `check(1 MB) = Ok`. The ceiling bounds live heap only; VmHWM is the only honest number and it is never gated (F2). | Attack `ceiling_threads`. | CONFIRMED |
| F6 | HIGH | **GGUF path fails open.** Missing optional keys silently default `vocab=32000, ctx=4096, ff=4×h, head_dim=64`; `read_metadata` returns `Ok` after breaking on the first malformed KV (partial parse → defaults). A 1.7B-class model plans at **1.464 GB vs real 3.194 GB → 1,715,462,144 B (2.15×) under-prediction**, emitted as a confident plan with no warning — `admit` would answer FITS on a budget it does not fit. | Attack `gguf_defaults`: `parsed config (no error)`, `UNDER-PREDICTION: 1715462144 bytes (2.15x too small) reported as a confident plan, no warning`. `src/gguf.rs:216-226, 274-295`. | CONFIRMED |
| F7 | MEDIUM | **Proof-of-record gaps (spec DoD).** No `reports/` (mutation score ≥70% required); no `docs/ADVERSARIAL_REVIEW.md` (DoD 6); no tagged release / SHA256 / post-download smoke (PROOF-AND-RELEASE §1); no differential-oracle test though README §"Relationship to Python" asserts "checked against Python outputs on shared test inputs"; no traceability CI script (PROOF-AND-RELEASE §2 — only `check_no_internal_refs.sh`). | `ls reports/` → absent; `ls docs/` → EVIDENCE, PAPER-TRACEABILITY, check_no_internal_refs.sh only; `grep -rn "python" tests/` → no oracle test. | CONFIRMED |
| F8 | MEDIUM | **"Zero silent mode changes" is a constant.** `src/verify.rs:164` sets `mode_changed_silently = false` unconditionally, then stress asserts it. The acceptance metric can never fail — decorative control, and EVIDENCE row 3 / TRACEABILITY cite it as proof. | `src/verify.rs:164`; `tests/stress.rs` assertion on that field. | CONFIRMED |
| F9 | MEDIUM | **Known-answer test is tautological and the "hand-computed" figure is wrong.** `2*6*2*512*64*4 = 3,145,728`, not `6,291,456`. The wrong number is repeated in **three** places (cost.rs:321 comment, EVIDENCE:219, PAPER-TRACEABILITY:69), and the test computes `expected` from the *same expression* as the implementation, so it cannot detect a wrong-but-self-consistent formula. Attack bin confirms truth: `kv512=3145728`. | `src/cost.rs:321,330`; `docs/EVIDENCE.md:219`; `docs/PAPER-TRACEABILITY.md:69`. | CONFIRMED |
| F10 | MEDIUM | **RoPE is the identity everywhere in the forward pass.** All three call sites pass `seq_len = 1` → position 0 → θ=0 → no rotation. No test exercises a non-zero position, so nothing fails if RoPE is removed from `transformer`. TRACEABILITY row 5's buy ("attention respects token order" via positional encoding) is unbacked. | `src/engine/transformer.rs:168,170` (`apply_rope(&mut q, 1, …)`), `src/engine/ops.rs:301` (test also uses 1); only rope test is `rope_position_zero_is_identity`. | CONFIRMED |
| F11 | MEDIUM | **`plan --context` is disconnected from the engine.** KV cache is preallocated at `max_seq_len` regardless of the plan's context → predicted-vs-measured swings 0.74–1.06 (table §1); at ctx=8 the prediction is 6% **low**, which is the dangerous direction for a memory budget. | `measure` bin: ctx8 1.063 / ctx512 1.029 / ctx4096 0.741; allocator abs peak constant at 56,679,680 B across all three. | CONFIRMED |
| F12 | MEDIUM | **Evidence hygiene.** (a) EVIDENCE:275 claims "cargo-mutants not installed" — **27.1.0 is installed and was run**; (b) EVIDENCE row 5 raw output shows `/build/.cache/…` which does not exist here (actual `/build/.cache/…`) → not reproducible as written; (c) EVIDENCE row 6's three printed numbers don't reconcile (`0.000 / 0.057 / +0.1 MB`) because `allocator_peak` prints the window delta while `delta_bytes` uses the absolute peak; (d) README table claims musl static binary + `cargo-mutants on contract logic` + Python cross-check with **no EVIDENCE row** for any of them. | `cargo mutants --version` → 27.1.0; `docs/EVIDENCE.md:275` row 5, row 6; README:56-68, 78-84. | CONFIRMED |
| F13 | LOW | **`real_model` test passes vacuously** when no GGUF is present (prints SKIP, returns ok) → false green on a stranger's machine; PROOF-AND-RELEASE explicitly calls false-green worse than red. Should be `#[ignore]` + explicit run, or fail with a PARTIAL marker. | `tests/real_model.rs` skip path. | CONFIRMED |
| F14 | LOW | **Peak-counter race (by inspection, not reproduced).** Snapshot `fetch_add` then `load` lets a concurrent `free` between them under-report the true high-water. Concurrent 8×5000 run reported peak 833,193,350 B while the harness held up to ~1.28 GB live — suggestive, inconclusive. Fix: derive the after-value from `fetch_add`'s return (or `fetch_max`), never from a second load. | Attack `counter` (concurrent leg). | UNPROVEN |
| F15 | LOW | **Stress budgets below process baseline pass.** 10 MB / 20 MB / 50 MB declared budgets report OK while VmHWM is 58.3 MB — a direct consequence of F2, listed because it is the easiest regression to encode as a test. | stress output: `allocator_peak=0.0 MB`, `VmHWM 58.3 MB`, budgets 10/20/50 MB → OK, exit 0. | CONFIRMED |

### The one screen that decides the verdict

```
$ fitsproof verify --budget-gb 0.02 --quant none --context 512 ; echo EXIT:$?
DEGRADED: base config needs 0.057 GB > budget 0.020 GB. Applying: Use int8_sym instead of none. New predicted peak: 0.015 GB.
allocator_peak: 0.000 GB
VmHWM:          0.057 GB
delta:          +0.1 MB (VmHWM - allocator_peak)
budget:         0.020 GB
budget_respected: true
EXIT:0
   Maximum resident set size (kbytes): 55480
```

Declared 20 MB → record claims int8 @ 15 MB → engine runs fp32 → 54.2 MB resident →
tool prints `budget_respected: true` → exit 0. Spec §3.1 (`admit` installs a ceiling,
excess fails as typed error) and the Prove step (measure peak vs budget) both fail on
this one run.

---

## 3. Attacks that failed (do not repeat these)

| Attack | Setup | Outcome |
|---|---|---|
| Counter arithmetic overflow | push sizes toward `u64::MAX`, ceiling `u64::MAX` | Not reachable — `Layout ≤ isize::MAX`, ceiling add uses `saturating_add`; `u64::MAX` ceiling behaves as "disabled by design". |
| Free-then-realloc false trip | 10,000 alloc/free cycles across the ceiling boundary | Drift **0**; current returns to baseline; ceiling edges correct (`at ceiling` refuses, `1 B under` admits). |
| Refusal storm | 10,000 refused allocations | current/peak/count **unchanged** — refusal path does not corrupt counters. |
| Concurrent accounting drift | 8 threads × 5000 alloc/free | Live-heap delta **0**. Peak *value* may be under-reported (F14) — inconclusive. |
| Realloc accounting (direct) | `realloc` to cross the ceiling | Refused correctly (returns NULL) because the ceiling check is delta-based; next alloc correctly refused while `current == ceiling`. |
| Naive TOCTOU | 200,000 interleaved `check()`/`alloc()` iterations | Did not defeat the allocator — only a *deterministic* interleaving (other thread allocs in the window) reliably produced `check(1MB)=Ok` → `alloc(1MB)=NULL`. |
| Ceiling holds for pure heap growth | `check()` + `raw_alloc` under an installed ceiling | Typed `DoesNotFit` returned for the API that nothing calls (F4 — the *product* never exercises it). |

---

## 4. Ranked improvements

| # | Problem | Change | Test | Effort |
|---|---|---|---|---|
| P1 | F1 — ceiling never installed | `admit`/`verify`/`stress` call `set_ceiling(effective_budget)` before any allocation and refuse on the enforcement outcome | A run whose allocation exceeds the declared budget must fail with typed `DoesNotFit` / exit 2; grep test asserting `set_ceiling` has ≥1 product caller | S |
| P2 | F2 — verify certifies what it exceeds | Print **and gate** `os_budget_respected`; report absolute allocator peak alongside windowed delta (two named bases); make `budget_respected = allocator_abs.max(vmhwm) <= budget` | `verify --budget-gb 0.02` must exit 2; `budget_respected` must equal `VmHWM <= budget` | S |
| P3 | F2 — stress budgets inflated / baselined | Use declared budgets (drop `predicted*2` in tests, `predicted*4` for degraded in CLI); count `VmHWM > budget` as a violation | stress rows with 10 MB budgets must fail until accounting is honest | S |
| P4 | F8 — metric hardcoded false | Derive `mode_changed_silently` from emitted records vs mode transitions, or delete the field and the claim | Inject a mode change with no record → metric must be true | S |
| P5 | F4/F5 — over-ceiling = abort, non-heap invisible | Preflight `check()` at the allocation sites that can grow unboundedly (`Box`/`Vec` growth via `try_reserve`); report `peak_vmhwm` against budget as the outer bound; document that the ceiling covers live heap only | ceiling-crossing generation returns `Err(DoesNotFit)`, exit 2, never 134; thread-stack attack must show as a *violation*, not `Ok` | M |
| P6 | F6 — GGUF fails open | Error on missing `vocab/context/intermediate` keys instead of defaults; `read_metadata` returns `Err` on malformed KV instead of partial `Ok` | sparse-header GGUF → `Err`; patching `gguf_defaults` bin must print "no under-prediction" | S |
| P7 | F3 — degradation is fiction | Either execute the degraded quant (wire int8/int4 into `transformer`) or refuse to emit a degradation the engine cannot honour | verify under a degraded record must allocate ≤ degraded prediction | L |
| P8 | F11 — `--context` disconnected | Engine allocates KV at plan context (or `plan` stops offering `--context`); pick one source of truth | measured/predicted within e.g. 0.9–1.1 across ctx ∈ {8,512,4096} | M |
| P9 | F10 — RoPE identity | Pass the real position; add a test that fails if rotation at pos>0 is identity | remove `apply_rope` from forward → test must fail | S |
| P10 | F7/F9/F12/F13 — evidence | Fix `6,291,456` → `3,145,728` in all three files and make the KAT a genuinely independent hand-computed constant; re-record row 5 with the real path; remove/qualify README claims with no row (musl, mutants, Python oracle); mark `real_model` `#[ignore]`; run `cargo mutants` ≥70% on contract modules and commit `reports/`; add `docs/ADVERSARIAL_REVIEW.md`; add the traceability CI script | `grep -rn 6,291,456 src docs tests` → empty; every README headline claim has an EVIDENCE row or an explicit "not yet proven" | M |

Suggested order: **P1 → P2 → P3 → P4** (one PR, closes all three BLOCKERs plus the
decorative metric), then P6, P5, P7, P10.

---

## 5. Mutation sample (one file, honest scope)

`cargo mutants 27.1.0 -f src/admit.rs --timeout 600` (default out dir, copy-in-tmp;
baseline `18s build + 248s test`):

```
Found 2 mutants to test
MISSED   src/admit.rs:57:9: replace <impl Display for DoesNotFitPlan>::fmt -> Ok(Default::default())
2 mutants tested in 8m: 1 missed, 1 unviable
```

**1 of 1 viable mutant missed.** Tests assert `binding_constraint` is non-empty and
`message.starts_with("REFUSED:")` on the `AdmitRecord`, but nothing formats the
`DoesNotFitPlan` Display impl — so the *reported* binding-constraint text can be
deleted without a single test noticing. Sample is 1 viable mutant (cargo-mutants
finds little to mutate in this file), so read it as a coverage hole in the refusal
message, not as a project mutation score. No `reports/` directory exists to hold the
≥70% score the spec requires.

---

## 6. Evidence cross-check summary

| Documented claim | Verdict |
|---|---|
| EVIDENCE rows 1–4, 8 (fmt/clippy/test counts, probe, plan, admit, stress) | reproduced |
| EVIDENCE row 3 (25 configs, 0 violations, 0 silent mode changes) | reproduced — but the metric behind "0 silent mode changes" is hardcoded false (F8), and violations are window-delta-based (F2) |
| EVIDENCE row 5 (real GGUF) | reproduced with a **different path** than documented (F12b) |
| EVIDENCE row 6 (verify numbers) | reproduced; the three printed numbers are mutually inconsistent as an explanation (F12c) |
| EVIDENCE:275 "cargo-mutants not installed" | **false** — 27.1.0 present (F12a) |
| PAPER-TRACEABILITY rows 1,2,4,6,7 | tests exist and are meaningful |
| PAPER-TRACEABILITY row 3 (GQA KAT) | exists but wrong constant + self-referential (F9) |
| PAPER-TRACEABILITY row 5 (RoPE / token order) | test exists; the forward pass never uses a non-zero position (F10) |
| PAPER-TRACEABILITY row 8 (FlexGen) | duplicates row 2's test — decorative |
| PAPER-TRACEABILITY row 9 (Kaplan) | honestly marked indirect |
| "zero silent mode changes" as evidence | decorative constant (F8) |
| "budget asserted ≤ budget" in README | assertion is on a windowed delta that excludes the dominant term (F2) |
| README: "enforced by TrackingAllocator — a ceiling is an error, not an OOM" | **false in shipped code** (F1, F4) |
| README: "Rust versions checked against Python outputs" | no such test exists (F7) |
| README: static musl binary proof | not run; no EVIDENCE row (F12d) |
| TRACEABILITY row 3 arithmetic `6,291,456` | wrong (F9) |

---

## 7. Ship verdict

Not yet — and the gap is narrow but fundamental. The mechanics that exist are real:
accounting drifts 0, refusal semantics are correct, the cost model reproduces its
own figures to the byte, the reference-bundle predicted/measured ratio is an
excellent 1.03, clippy/fmt/tests are clean, the Limitations section is honest, and
the EVIDENCE + TRACEABILITY registers are exactly the right discipline for this kind
of work. But the product's central claim — *the budget becomes a hard ceiling, and
the prove step measures peak against it* — is currently aspirational: `set_ceiling`
is never called by anything that ships (F1), a 20 MB budget is certified
`budget_respected: true` while the process holds 54 MB and exits 0 (F2), the
degradation record describes an int8 run the engine never performs (F3), and the
failure mode for a genuine over-budget allocation is SIGABRT exit 134 rather than
the typed `DoesNotFit` / exit 2 the spec promises (F4). Three BLOCKERs-plus worth
of fixes (P1–P4) would make the headline true; until then this is a well-tested
*model* of memory proof, not a memory proof. Land P1–P4, re-run this exact review
(§1 commands, §2 F2 demo, §3 attacks), and I would ship on the follow-up.

---
*Attack harness: `/tmp/opencode/fitsproof-attack/src/bin/{ceiling_null, ceiling_threads, realloc_growth, toctou, counter, numbers, gguf_defaults, measure}`. Mutation log: `/tmp/opencode/mutants/admit.log`. All attack code is outside the reviewed tree.*
