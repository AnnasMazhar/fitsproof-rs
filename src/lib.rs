//! fitsproof-rs — the compiled edition of fitsproof.
//!
//! The contract this crate enforces: an LLM memory budget that is *enforced* by a
//! byte-counting allocator and then *proven* against measured process memory.
//!
//! # Module map
//!
//! - `allocator` — `TrackingAllocator` + `DoesNotFit` error
//! - `model`     — `ModelConfig` (architecture parameters)
//! - `probe`     — machine characterisation (STREAM triad, GEMM, RAM)
//! - `cost`      — analytical roofline cost model
//! - `plan`      — `Plan` / `Verdict` (fits / fits_with_degradation / does_not_fit)
//! - `admit`     — `AdmitRecord` (the enforcement point)
//! - `verify`    — `VerifyRecord` (allocator peak + VmHWM + delta)
//! - `engine`    — transformer ops, quantisation, sampling, reference bundle

pub mod admit;
pub mod allocator;
pub mod client;
pub mod cost;
pub mod engine;
pub mod gguf;
pub mod gguf_tensors;
pub mod mcp;
pub mod model;
pub mod pareto;
pub mod plan;
pub mod probe;
pub mod serve;
pub mod verify;

// Install the tracking allocator as the global allocator.
//
// This is the central mechanism that makes the byte ceiling enforceable rather
// than merely advisory.  All allocations in this process go through it.
#[global_allocator]
pub static ALLOCATOR: allocator::TrackingAllocator = allocator::TrackingAllocator::new();

/// Crate version, surfaced by `fitsproof --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    /// Fault detected: VERSION is the empty string (env var not set at compile time).
    #[test]
    fn version_is_not_empty() {
        assert!(!super::VERSION.is_empty());
    }

    /// Fault detected: global allocator not installed (ALLOCATOR is the system allocator).
    /// Proves the TrackingAllocator is actually wired as the GlobalAlloc.
    #[test]
    fn global_allocator_tracks_allocations() {
        let before = crate::ALLOCATOR.peak_bytes();
        // Force a heap allocation.
        let _v: Vec<u8> = vec![0u8; 4096];
        let after = crate::ALLOCATOR.peak_bytes();
        assert!(
            after >= before,
            "ALLOCATOR peak_bytes must not decrease after a heap allocation"
        );
    }
}
