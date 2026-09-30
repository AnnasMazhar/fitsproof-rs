# Adversarial Review — fitsproof-rs v0.1

**Pass:** c5-p11-adversarial-2 (independent verification)  
**Reviewer:** kiro:claude-opus-4.5 (independent of builder)  
**Date:** 2026-09-30  
**Mode:** ATTACK THE PROPERTY (pass 2)

---

## Summary

This pass directly attacked the core safety property: **predict peak memory, enforce a byte ceiling, refuse loudly when the budget is violated.**

**Result:** 1 CRITICAL bug found and fixed. The budget contract could be completely bypassed via integer overflow in the cost addition.

---

## Property Attack Results

### ATTACK 1-4: Input validation (negative budget, NaN, Infinity)

**Commands:**
```bash
./target/release/fitsproof admit --budget-gb -1.0
./target/release/fitsproof admit --budget-gb NaN
./target/release/fitsproof admit --budget-gb inf
./target/release/fitsproof admit --budget-gb Infinity
```

**Results:** All correctly rejected with exit 2 and actionable error messages.

**Verdict:** PASS — input validation is sound.

---

### ATTACK 5-8: Integer overflow budget bypass (ADV-C5-P2-1) — CRITICAL

**Attack:** Pass `context_len=18446744073709551615` (u64::MAX) to bypass the budget check.

**Mechanism:**
1. `kv_cache_bytes()` computes `2 * layers * kv_heads * context * head_dim * bytes`
2. With context=u64::MAX, this product overflows f64 precision and saturates to u64::MAX on cast
3. In `estimate()`, the total is computed as `w + kv + act`
4. `u64::MAX + 53_497_344 + 9_216` **wraps** to 53_506_559 (~0.054 GB) due to release-mode wrapping arithmetic
5. Configuration requiring exabytes is ADMITTED to a 4 GB budget

**Pre-fix command and output:**
```bash
./target/release/fitsproof admit --budget-gb 4.0 --context 18446744073709551615
```
```
ADMITTED: 0.054 GB predicted peak <= 4.000 GB budget (margin: 3946.5 MB)
EXIT:0
```

**This is a total contract bypass.** The model that was admitted would need ~18 exabytes of KV cache memory.

**Fix applied:**
```rust
// src/cost.rs: estimate()
// Before:
let total = w + kv + act;
// After:
let total = w.saturating_add(kv).saturating_add(act);
```

**Post-fix command and output:**
```bash
./target/release/fitsproof admit --budget-gb 4.0 --context 18446744073709551615
```
```
REFUSED: needs 18446744073.710 GB (weight=0.053 GB, kv=18446744073.710 GB, activation=0.000 GB), budget 4.000 GB; no degradation fits
EXIT:2
```

**Regression tests added:**
- `estimate_addition_overflow_does_not_wrap_to_small_value` — verifies total_peak_bytes(ctx=MAX) >= total_peak_bytes(ctx=512)
- `admit_refuses_extreme_context_len` — end-to-end verify REFUSED status

**Severity:** CRITICAL → **FIXED**

---

### ATTACK 9: Stress harness coverage gap

**Finding:** The stress harness (`fitsproof stress`) does not test overflow cases. All 25 configs use reasonable context lengths.

**Command:**
```bash
grep -c "18446744073709551615\|usize::MAX\|u64::MAX" tests/stress.rs
# Output: 0
```

**Assessment:** This is expected — the stress harness tests the happy path (admission contract under normal configs). The adversarial suite now covers overflow cases via the new tests.

**Severity:** Informational (documented)

---

### ATTACK 10: MCP server overflow bypass

**Command:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admit","arguments":{"budget_gb":4,"context_len":18446744073709551615}}}' | ./target/release/fitsproof mcp
```

**Pre-fix output:**
```json
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":{"status":"admitted",...}}]}}
```

**Post-fix output:**
```json
{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":{"status":"refused",...}}]}}
```

**Verdict:** PASS (after fix) — MCP path correctly propagates the fix.

---

### ATTACK 11: Serve endpoint (HTTP) overflow bypass

The serve endpoint does not accept `context_len` in the request body — it uses the reference bundle's default. This is **not a bypass** — the endpoint is designed for quick demos, not full configuration. The CLI/MCP paths are the configuration interfaces.

**Severity:** Informational (by design)

---

### ATTACK 12: vocab_size overflow via GGUF

**Analysis:** The `weight_bytes()` function uses the same f64 multiplication pattern:
```rust
let embed_bytes = (vocab_size as f64 * hidden_size as f64 * bytes_per_element) as u64;
```

With vocab_size=u64::MAX, this also saturates. However:
1. The vocab_size comes from GGUF metadata, not CLI input
2. GGUF files have 32-bit tensor count limits, so vocab_size > 4B is implausible from real models
3. The `saturating_add` fix in `estimate()` protects against the downstream wrap regardless

**Severity:** Informational (theoretical, protected by the fix)

---

## Findings Table

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| ADV-C5-P2-1 | CRITICAL | estimate() addition overflow wraps total_peak_bytes from u64::MAX to ~54 MB, bypassing budget contract | Pre-fix: `ADMITTED: 0.054 GB` for ctx=MAX; Post-fix: `REFUSED: 18446744073.710 GB` | **FIXED** |
| ADV-C5-P2-2 | CRITICAL | End-to-end budget bypass via CLI with extreme context_len | Same as above | **FIXED** (covered by ADV-C5-P2-1 fix) |
| ADV-C5-1 | Minor | vLLM docs link 404 (from Pass 1) | URL reorganised | Accepted (cosmetic) |

---

## Tests Added (ADV-C5-P2)

| Test | Fault Detected |
|------|----------------|
| `estimate_addition_overflow_does_not_wrap_to_small_value` | Verifies total_peak_bytes(ctx=MAX) >= total_peak_bytes(ctx=normal) — catches wrapping arithmetic |
| `admit_refuses_extreme_context_len` | End-to-end: context_len=MAX must produce AdmitStatus::Refused |

---

## Gate Check

- [x] Property attack attempted: bypass budget enforcement via overflow
- [x] Attack succeeded on unpatched code (CRITICAL finding)
- [x] Fix applied and verified
- [x] Regression tests added (2 new tests in adversarial.rs)
- [x] Full test suite passes (271 tests)
- [x] clippy + fmt clean
- [x] 0 open blockers

---

## Raw Evidence

### Pre-fix trace showing the overflow wrap

```bash
$ ./target/release/fitsproof plan --budget-gb 100 --context 18446744073709551615
Verdict:         Fits
Predicted peak:  0.054 GB     # ← WRONG: should be exabytes
Budget:          100.000 GB
Quant:           none
Context length:  18446744073709551615

$ ./target/release/fitsproof plan --budget-gb 100 --context 1000000
Verdict:         Fits
Predicted peak:  3.126 GB     # ← Correct: larger context → larger peak
Budget:          100.000 GB
Quant:           none
Context length:  1000000
```

The pre-fix prediction for context=MAX (0.054 GB) was **smaller** than context=1M (3.126 GB) due to wraparound.

### Post-fix verification

```bash
$ ./target/release/fitsproof admit --budget-gb 4.0 --context 18446744073709551615
REFUSED: needs 18446744073.710 GB (weight=0.053 GB, kv=18446744073.710 GB, activation=0.000 GB), budget 4.000 GB; no degradation fits
EXIT:2
```

### Test suite after fix

```
running 50 tests (adversarial.rs)
test result: ok. 50 passed; 0 failed

Total: 271 tests, 0 failures
```
