//! Attack harness — documents the bypass class for off-allocator allocations.
//!
//! The `TrackingAllocator` counts only allocations routed through `GlobalAlloc`.
//! Three bypass classes exist that the allocator cannot see:
//!
//!   1. **System allocator** (`std::alloc::System`): direct `mmap`/`malloc` bypassing
//!      the global allocator.  Any library that calls `malloc` directly (not through
//!      Rust's allocator hook) falls into this class.
//!   2. **Thread stacks**: each thread's stack is `mmap`-ed by the OS outside the
//!      global allocator (default 8 MiB per thread on Linux).
//!   3. **Static data**: `.bss`/`.data` segments, read-only mappings (`mmap`-ed DSOs).
//!
//! The **contract backstop** for this class is `VmHWM` (the OS-reported process
//! high-water mark, `/proc/self/status`).  If off-allocator memory causes a budget
//! violation, the `VmHWM <= budget` gate in `verify_run` will catch it.
//!
//! This test **documents and measures** the gap — it does not hide it.
//! The gap must be >= 60 MB to confirm the bypass is real and the backstop is
//! necessary.  Measured on the review machine (ThinkStation P500):
//! `VmHWM = 67824 kB` vs `tracking_current = 554 B` after 64 MiB System alloc.
//!
//! # Reference
//!
//! REVIEW-muse-spark.md F5: "Off-allocator memory is invisible to the tracker."
//! Numbers from the reviewer's `alloc-attack` binary (commit 8b95487).

use std::alloc::{GlobalAlloc, Layout, System};

/// F5: System-allocator bypass creates a VmHWM gap >= 60 MB.
///
/// Fault detected: if the gap were 0, it would mean `std::alloc::System`
/// allocations are somehow tracked — which would falsely reassure users that
/// all memory is covered.
#[test]
fn system_alloc_bypass_creates_vmhwm_gap_ge_60_mb() {
    // Record allocator state before the bypass allocation.
    let peak_before = fitsproof::ALLOCATOR.peak_bytes();
    let tracking_before = fitsproof::ALLOCATOR.current_bytes();

    // Allocate 64 MiB through System (bypasses GlobalAlloc).
    const ALLOC_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
    let layout = Layout::from_size_align(ALLOC_BYTES, 4096).unwrap();
    let ptr = unsafe { System.alloc_zeroed(layout) };
    assert!(!ptr.is_null(), "System.alloc_zeroed must succeed");

    // Touch every page to ensure the OS accounts for the memory.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, ALLOC_BYTES) };
    // Write a pattern on every page boundary.
    for (i, chunk) in slice.chunks_mut(4096).enumerate() {
        chunk[0] = (i & 0xFF) as u8;
    }
    // Use std::hint::black_box to prevent dead-store elimination.
    let checksum: u64 = slice
        .chunks(4096)
        .enumerate()
        .map(|(i, chunk)| std::hint::black_box(chunk[0] as u64 + i as u64))
        .sum();
    // Just assert it's nonzero (it will be since i starts at 0 but other pages are nonzero).
    // Pages with i=0 write 0, so we have ALLOC_BYTES/4096 - 1 nonzero pages.
    assert!(
        std::hint::black_box(checksum) > 0,
        "checksum must be nonzero (pages were written)"
    );

    // Read VmHWM — should now reflect the 64 MiB allocation.
    let vmhwm = fitsproof::verify::read_vmhwm_bytes();
    let tracking_after = fitsproof::ALLOCATOR.current_bytes();
    let peak_after = fitsproof::ALLOCATOR.peak_bytes();

    // Allocator counters must not have changed (bypass is invisible to tracker).
    assert_eq!(
        tracking_after, tracking_before,
        "tracking_current must not change for System alloc (bypass is invisible)"
    );
    assert_eq!(
        peak_after, peak_before,
        "peak_bytes must not change for System alloc (bypass is invisible)"
    );

    // VmHWM must be at least 60 MB above the allocator-tracked peak.
    // This proves the bypass class is real and the VmHWM backstop is necessary.
    let gap = vmhwm as i64 - peak_after as i64;
    assert!(
        gap >= 60 * 1024 * 1024,
        "VmHWM − allocator_peak must be >= 60 MB after System alloc, got {:.1} MB gap \
         (vmhwm={:.1} MB, allocator_peak={:.1} MB)",
        gap as f64 / 1e6,
        vmhwm as f64 / 1e6,
        peak_after as f64 / 1e6,
    );

    // Cleanup.
    unsafe { System.dealloc(ptr, layout) };
}
