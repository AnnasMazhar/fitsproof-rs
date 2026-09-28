#!/usr/bin/env bash
# docs/demo.sh — asciinema demo script for fitsproof-rs
#
# Records a ~60-second terminal session showing the core contract:
#   1. Build the binary
#   2. Stress harness: 25 configs, 0 violations
#   3. Refuse a too-small budget (exit 2)
#   4. Admit a 4 GB budget with margin reported
#   5. Verify: allocator_peak + VmHWM + delta printed
#
# Usage:
#   # Install asciinema first: https://asciinema.org/docs/installation
#   # Then record:
#   asciinema rec docs/demo.cast --overwrite --command "bash docs/demo.sh"
#
#   # Convert to GIF (requires agg: https://github.com/asciinema/agg):
#   agg docs/demo.cast docs/demo.gif
#
#   # Or upload and embed the asciinema player in README:
#   asciinema upload docs/demo.cast
#
# The script is deterministic — run it twice to confirm identical output.
# If the outputs differ, the demo is not reproducible and the CI gate fails.

set -euo pipefail

# Slow down output so the recording is readable
DELAY=0.5

_print() {
    echo -e "\033[1;32m$\033[0m \033[1m$*\033[0m"
    sleep "$DELAY"
}

# ── 1. Build ────────────────────────────────────────────────────────────────
_print "cargo build --release"
cargo build --release 2>&1 | grep -E "Compiling|Finished"

FITSPROOF=./target/release/fitsproof

# ── 2. Stress harness ───────────────────────────────────────────────────────
echo ""
_print "$FITSPROOF stress"
$FITSPROOF stress 2>&1 | tail -3

# ── 3. Refuse a 1 MB budget ─────────────────────────────────────────────────
echo ""
_print "$FITSPROOF admit --budget-gb 0.001"
$FITSPROOF admit --budget-gb 0.001 || true
echo "exit code: $?"

# ── 4. Admit a 4 GB budget ──────────────────────────────────────────────────
echo ""
_print "$FITSPROOF admit --budget-gb 4"
$FITSPROOF admit --budget-gb 4

# ── 5. Verify: allocator + VmHWM + delta ────────────────────────────────────
echo ""
_print "$FITSPROOF verify --budget-gb 4"
$FITSPROOF verify --budget-gb 4

echo ""
echo "# fitsproof-rs: predict → enforce → prove. No silent OOM."
