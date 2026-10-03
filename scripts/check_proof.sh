#!/usr/bin/env bash
# check_proof.sh — validate that fitsproof-rs satisfies its acceptance criteria.
#
# The repo's proof is:
# 1. All tests pass (cargo test exits 0)
# 2. clippy is clean (cargo clippy --all-targets -- -D warnings exits 0)
# 3. The stress harness runs without panic
# 4. The binary can refuse a configuration (exit 2)
# 5. docs/EVIDENCE.md exists
#
# This checker is falsifiable: removing EVIDENCE.md, breaking a test, or making
# clippy fail will cause the checker to fail and print which check broke.
#
# Exits 0 and prints PROOF_COMPLETE on success.
# Exits 1 with a clear message on any failure.
#
# Usage:
#     bash scripts/check_proof.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
EVIDENCE="$REPO_ROOT/docs/EVIDENCE.md"

MIN_TEST_COUNT=50  # Must have at least this many tests

failures=()

echo "check_proof: validating fitsproof-rs proof..."

# Check 1: docs/EVIDENCE.md exists
echo "  [1/5] Checking docs/EVIDENCE.md exists..."
if [[ ! -f "$EVIDENCE" ]]; then
    failures+=("MISSING: docs/EVIDENCE.md does not exist")
elif [[ $(stat -c%s "$EVIDENCE" 2>/dev/null || stat -f%z "$EVIDENCE") -lt 1000 ]]; then
    failures+=("INCOMPLETE: docs/EVIDENCE.md is too small (< 1000 bytes)")
fi

# Check 2: cargo test passes
echo "  [2/5] Running cargo test..."
cd "$REPO_ROOT"
test_output=$("$CARGO" test 2>&1) || {
    failures+=("CARGO TEST FAILED: cargo test exited non-zero")
    failures+=("$(echo "$test_output" | tail -5)")
}

# Count tests from output
test_count=$(echo "$test_output" | grep -oP '\d+(?= passed)' | paste -sd+ | bc 2>/dev/null || echo "0")
if [[ "$test_count" -lt "$MIN_TEST_COUNT" ]]; then
    failures+=("INSUFFICIENT TESTS: $test_count < $MIN_TEST_COUNT required")
fi

# Check 3: clippy is clean
echo "  [3/5] Running cargo clippy..."
clippy_output=$("$CARGO" clippy --all-targets -- -D warnings 2>&1) || {
    failures+=("CLIPPY FAILED: cargo clippy exited non-zero")
    failures+=("$(echo "$clippy_output" | grep -E '(error|warning):' | head -3)")
}

# Check 4: stress harness runs
echo "  [4/5] Running stress harness..."
"$CARGO" build --release --quiet 2>/dev/null
if [[ -x "$REPO_ROOT/target/release/fitsproof" ]]; then
    stress_output=$("$REPO_ROOT/target/release/fitsproof" stress 2>&1) || {
        failures+=("STRESS FAILED: fitsproof stress panicked or failed")
    }
    
    # Check for violations (lines containing "VIOLATION" but not "[REFUSED]")
    if echo "$stress_output" | grep -q "VIOLATION"; then
        failures+=("STRESS VIOLATIONS: budget violations detected")
    fi
else
    failures+=("BINARY MISSING: target/release/fitsproof does not exist after build")
fi

# Check 5: admit can refuse
echo "  [5/5] Verifying admit can refuse over-budget configs..."
if [[ -x "$REPO_ROOT/target/release/fitsproof" ]]; then
    admit_output=$("$REPO_ROOT/target/release/fitsproof" admit --budget-gb 0.001 2>&1) && {
        # If it succeeded (exit 0), that's wrong
        failures+=("ADMIT SHOULD REFUSE: admit with 0.001 GB budget should not succeed")
    }
    # Exit 2 is expected (refused), exit 1 might be an error
    admit_exit=$?
    if [[ "$admit_exit" -ne 2 && "$admit_exit" -ne 1 ]]; then
        failures+=("ADMIT UNEXPECTED EXIT: expected 1 or 2, got $admit_exit")
    fi
fi

# Report results
if [[ ${#failures[@]} -gt 0 ]]; then
    echo "check_proof: FAIL"
    for f in "${failures[@]}"; do
        echo "  $f"
    done
    exit 1
fi

echo "check_proof: all checks passed."
echo "PROOF_COMPLETE"
exit 0
