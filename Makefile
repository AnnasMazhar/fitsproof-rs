# fitsproof-rs integration Makefile
#
# Provides a pre-flight memory contract check for any GGUF model + llama.cpp workflow.
# Usage:
#   make preflight MODEL=/path/to/model.gguf BUDGET_GB=4.0 QUANT=q4_k_m CTX=4096
#   make stress
#   make test
#
# In CI (GitHub Actions, etc.):
#   - name: Memory pre-flight
#     run: make preflight MODEL=$MODEL_PATH BUDGET_GB=4.0 QUANT=q4_k_m CTX=4096
#
# Exit codes:
#   0 — admitted (model fits, proceed to inference)
#   2 — refused (named binding constraint); the job fails before any OOM

FITSPROOF ?= ./target/release/fitsproof
MODEL     ?=
BUDGET_GB ?= 4.0
QUANT     ?= q4_k_m
CTX       ?= 4096

.PHONY: build preflight stress test clean

## Build the release binary.
build:
	cargo build --release

## Run the memory contract pre-flight check.
## Exits 2 and names the binding constraint (weight/kv/activation) if the model does not fit.
## Run this before any 'llama-cli' or 'llama-server' invocation.
##
## Example: make preflight MODEL=~/.cache/models/llama-7b-q4.gguf BUDGET_GB=6 QUANT=q4_k_m CTX=4096
preflight: build
	@if [ -z "$(MODEL)" ]; then \
	    echo "fitsproof preflight: no MODEL set — running reference bundle check"; \
	    $(FITSPROOF) admit --budget-gb $(BUDGET_GB) --quant $(QUANT) --context $(CTX); \
	else \
	    $(FITSPROOF) admit \
	        --model "$(MODEL)" \
	        --budget-gb $(BUDGET_GB) \
	        --quant $(QUANT) \
	        --context $(CTX); \
	fi

## Run the 25-config stress harness.
## Fails if any configuration violates its declared budget or produces a silent mode change.
## Safe to run in CI; uses the reference bundle (no model file required).
stress: build
	$(FITSPROOF) stress

## Run the full test suite.
test:
	cargo test --all-targets

## Build a portable static binary (no dynamic libraries, no libc).
## Requires: rustup target add x86_64-unknown-linux-musl
musl:
	cargo build --release --target x86_64-unknown-linux-musl
	ldd target/x86_64-unknown-linux-musl/release/fitsproof || true

clean:
	cargo clean
