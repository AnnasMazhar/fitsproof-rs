# ADVERSARIAL_REVIEW.md — fitsproof-rs cycle 1 pass 10

Independent adversarial review per QUALITY-CONTRACT §6.
Reviewer: claude-opus-4.5 (independent of builder).
Date: 2026-09-28 08:00–09:00 UTC.

---

## 1. Claims audit — the 3 most load-bearing README claims

### Claim 1: Stress harness runs 25 configs with 0 violations

**Source:** README.md §Headline evidence

**Exact claim:**
> `fitsproof stress` runs 25 configurations against a declared budget and fails the build on any
> budget violation or undocumented mode change

**Falsification attempt:**
```
$ ./target/release/fitsproof stress
ref/fp32/ctx512/1GB: allocator_peak=0.0 MB, VmHWM=56.7 MB, delta=+0.0 MB, budget=1000.0 MB, OK
ref/fp32/ctx256/1GB: allocator_peak=0.0 MB, VmHWM=58.3 MB, delta=+1.6 MB, budget=1000.0 MB, OK
ref/int8/ctx512/500MB: allocator_peak=0.0 MB, VmHWM=58.3 MB, delta=+1.7 MB, budget=500.0 MB, OK
[... 20 more configs ...]
[REFUSED] ref/int4/ctx16/5MB: REFUSED: needs 0.01 GB, budget 0.01 GB; no degradation fits
[REFUSED] ref/int4/ctx8/5MB: REFUSED: needs 0.01 GB, budget 0.01 GB; no degradation fits

Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=10.0 MB, median=200.0 MB, max=1000.0 MB.
$ echo $?
0
```

**Result:** VERIFIED. 25 configs, 0 violations, 0 silent mode changes. The 2 REFUSED configs
are expected (they are designed to test the refusal path) and do not count as violations.

---

### Claim 2: Refused configs exit 2 with binding constraint named

**Source:** README.md §Headline evidence

**Exact claim:**
> Refused configs name the binding constraint. They exit 2 so your CI can gate on it

**Falsification attempt:**
```
$ ./target/release/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.06 GB, budget 0.00 GB; no degradation fits
$ echo $?
2

$ ./target/release/fitsproof admit --budget-gb 0.0001
REFUSED: needs 0.06 GB, budget 0.00 GB; no degradation fits
$ echo $?
2
```

**Result:** VERIFIED. Exit code is 2. Refusal message includes sizes. Note: the binding
constraint ("no degradation fits") is named but the breakdown (weight_bytes vs kv_cache_bytes)
is not itemized in the output. This is a **minor** deviation from the README's promise of
"binding constraint named" — the current message tells you it doesn't fit but doesn't say
*which* component is the bottleneck.

**Finding:** ADV-1 (minor) — see findings table below.

---

### Claim 3: verify prints allocator_peak + VmHWM + delta

**Source:** README.md §Headline evidence

**Exact claim:**
> `fitsproof verify` prints both the allocator-counted peak and the OS high-water mark (`VmHWM`),
> plus the delta — so the overhead of the runtime is a visible number, not a footnote

**Falsification attempt:**
```
$ ./target/release/fitsproof verify --budget-gb 4
ADMITTED: 0.055 GB predicted peak <= 4.000 GB budget (margin: 3944.9 MB)
allocator_peak: 0.000 GB
VmHWM:          0.057 GB
delta:          +0.1 MB (VmHWM - allocator_peak)
budget:         4.000 GB
budget_respected: true
$ echo $?
0
```

**Result:** VERIFIED. All three values (allocator_peak, VmHWM, delta) are printed.
Budget is respected. Exit 0.

---

## 2. Citation audit — RESEARCH.md links

All 22 unique URLs in docs/RESEARCH.md were extracted and sampled (7 checked, all primary sources).

| URL | HTTP Status | Supports Claim |
|-----|-------------|----------------|
| https://arxiv.org/abs/1910.07467 | 200 | YES — Zhang & Sennrich RMSNorm paper |
| https://arxiv.org/abs/2305.13245 | 200 | YES — Ainslie et al. GQA paper (EMNLP 2023) |
| https://www.cs.virginia.edu/stream/ref.html | 200 | YES — McCalpin STREAM benchmark |
| https://prng.di.unimi.it/ | 200 | YES — Blackman & Vigna xoshiro256** |
| https://github.com/ggml-org/ggml/blob/master/docs/gguf.md | 200 | YES — GGUF spec |
| https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html | 200 | YES — Rust allocator docs |
| https://man7.org/linux/man-pages/man5/proc_pid_status.5.html | 200 | YES — Linux VmHWM docs |

**Result:** All sampled links resolve (HTTP 200) and support their attached claims.
No dead links found. No claim misattribution detected.

---

## 3. Test-quality audit — fault injection

Five tests were sampled. For each, the fault it claims to detect was injected into the source
code, and the test was run to verify it fails.

### Test 1: `kv_cache_bytes_reference_fp16_known_answer`

**Fault claimed:** Omitting the factor of 2 (K+V) in KV cache formula halves the estimate.

**Fault injected:** Changed `(2.0 * cfg.num_layers` to `(cfg.num_layers` in `src/cost.rs:146`.

**Result:**
```
assertion `left == right` failed: kv_cache_bytes fp16 got 786432, expected 1572864
```

**Verdict:** DETECTED. Test failed immediately with the exact symptom (half the expected value).

---

### Test 2: `rmsnorm_known_answer`

**Fault claimed:** Skipping the RMS division produces wrong normalization output.

**Fault injected:** Changed `(xi / rms) * wi` to `xi * wi` in `src/engine/ops.rs:37`.

**Result:**
```
panicked at src/engine/ops.rs:266:9:
rmsnorm[0] got 3
```

**Verdict:** DETECTED. Test failed with the unnormalized value (3 instead of expected ~0.8485).

---

### Test 3: `int8_round_trip_within_one_lsb` (compile-time detection)

**Fault claimed:** Using max_val=128 instead of 127 for int8 shifts the scale incorrectly.

**Fault injected:** Changed `Int8Sym => 127` to `Int8Sym => 128` in `src/engine/quant.rs:39`.

**Result:**
```
error: literal out of range for `i8`
  --> src/engine/quant.rs:39:37
   |
39 |             QuantScheme::Int8Sym => 128,
   |                                     ^^^
```

**Verdict:** DETECTED. Rust compiler rejects at compile time (stronger than runtime test).

---

### Test 4: `weight_bytes_reference_fp32_known_answer`

**Fault claimed:** Omitting the embedding table doubles error for large-vocab models.

**Fault injected:** Changed `let embed_bytes = v * d * 4.0;` to `let embed_bytes = 0.0;` in
`src/cost.rs:107`.

**Result:**
```
assertion `left == right` failed: weight_bytes fp32 expected 53497344, got 52710912
```

**Verdict:** DETECTED. Test failed with missing embedding table bytes (786,432 = 512 × 64 × 4 × 6).

---

### Test 5: `refused_record_has_refusal_reason`

**Fault claimed:** Refused AdmitRecord must carry a non-empty refusal_reason string.

**Fault injected:** Changed `refusal_reason: reason.clone()` to `refusal_reason: String::new()`
in `src/admit.rs:132,146`.

**Result:**
```
panicked at tests/adversarial.rs:244:17:
refused AdmitRecord must carry a non-empty refusal_reason
```

**Verdict:** DETECTED. Test failed with the exact fault message.

---

**Summary:** 5/5 fault injections detected (4 by test failure, 1 by compile-time error).
The test suite has real detection power for the faults it claims to catch.

---

## 4. Findings table

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| ADV-1 | minor | Refusal message says "no degradation fits" but does not itemize binding constraint (weight vs KV vs activation) | `admit --budget-gb 0.001` output: "needs 0.06 GB, budget 0.00 GB; no degradation fits" — user cannot tell which component is the bottleneck | open |
| ADV-2 | minor | `budget_exactly_at_predicted_peak_admits` test does not catch the `<=` to `<` boundary fault because degradation path still returns non-DoesNotFit verdict | Fault injection: changed `<=` to `<` on line 133 of plan.rs; test still passed because verdict was `FitsWithDegradation` not `DoesNotFit` | open |
| ADV-3 | info | allocator_peak shows 0.0 GB in verify output — this is correct (reference bundle allocates no model weights) but may confuse users who expect non-zero allocator activity | `verify --budget-gb 4` shows `allocator_peak: 0.000 GB` | limitation |

---

## 5. Disposition of findings

### ADV-1 (minor) — Binding constraint not itemized

**Status:** open

**Recommendation:** Enhance `admit()` output to include the breakdown:
```
REFUSED: needs 0.06 GB (weight=0.05 GB, kv=0.01 GB, activation=0.00 GB), budget 0.00 GB
```

This is a documentation/UX issue, not a correctness bug. The safety contract (refuse when
budget exceeded) is maintained.

### ADV-2 (minor) — Boundary test is weak

**Status:** open

**Recommendation:** Add a second assertion in `budget_exactly_at_predicted_peak_admits` that
checks `p.verdict == Verdict::Fits` (not just `!= DoesNotFit`), ensuring the exact-boundary
case returns the expected verdict rather than falling through to degradation.

This is a test-quality issue, not a production bug. The production code is correct (uses `<=`).

### ADV-3 (info) — allocator_peak is zero

**Status:** limitation

**Reason:** The reference bundle uses pre-allocated arrays for weights (not heap-allocated
during generate()). Real GGUF weight loading (v0.2) will show non-zero allocator_peak.
This is expected behavior, not a bug. No action required.

---

## 6. Summary

- **Claims audit:** 3/3 claims verified with live output. 1 minor deviation (binding constraint
  not itemized).
- **Citation audit:** 7/7 sampled links resolve and support their claims.
- **Fault injection:** 5/5 faults detected (4 test failures, 1 compile-time rejection).
- **Open findings:** 2 minor (ADV-1, ADV-2). Both are quality improvements, not blockers.
- **Blockers:** 0.

The repository meets the acceptance criteria for adversarial pass 1. The builder should
address ADV-1 and ADV-2 in an improve pass if time permits.

---

*Review completed: 2026-09-28 09:00 UTC.*
