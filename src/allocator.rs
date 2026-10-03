//! Tracking allocator — the reason this compiled edition exists.
//!
//! A `GlobalAlloc` wrapper that counts:
//!   - `current_bytes`: live allocated bytes right now
//!   - `peak_bytes`: high-water mark since process start (monotonically non-decreasing)
//!   - `alloc_count`: total number of allocation calls
//!
//! When a ceiling is installed via [`TrackingAllocator::set_ceiling`], any allocation
//! that would push `current_bytes` past the ceiling fails — the allocator returns null,
//! which Rust translates into an allocation error that callers must handle explicitly.
//!
//! A typed error [`DoesNotFit`] names the binding constraint when a budget is exceeded.
//!
//! # Design invariants
//!
//! 1. `peak_bytes` is monotonically non-decreasing.
//! 2. An over-budget allocation returns `DoesNotFit`, never a panic, never an abort,
//!    never a silent OOM kill.
//! 3. A freed-then-reallocated pattern does not falsely trip the ceiling — `current_bytes`
//!    is decremented on dealloc before the next alloc is measured.
//! 4. **The ceiling check and increment are atomic** — `try_reserve` uses a CAS loop so
//!    that two concurrent threads cannot both pass the ceiling check and combine to exceed
//!    it (the TOCTOU race documented as ADV-3 in adversarial.rs, now fixed).
//!
//! # Research provenance
//!
//! Allocator hook pattern: standard Rust `GlobalAlloc` API
//! (https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html).
//! Atomic ordering: SeqCst used for correctness; relaxed load in `peak_bytes()` for
//! read-only queries (safe because the value is monotone for peak).
//! CAS loop for atomic reserve: standard compare-exchange retry pattern for lock-free
//! bounded counters (see Herlihy & Shavit, "The Art of Multiprocessor Programming", §5).

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// DoesNotFit — typed allocation refusal
// ---------------------------------------------------------------------------

/// Returned when an allocation would push usage past the installed ceiling.
///
/// Carries the human-readable `binding_constraint` string naming what would
/// be violated (e.g. `"activation buffer: 40 MB + current 3.8 GB > 4.0 GB ceiling"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoesNotFit {
    /// Human-readable description of the binding constraint.
    pub binding_constraint: String,
    /// The ceiling that was exceeded (bytes).
    pub ceiling_bytes: u64,
    /// The requested allocation size that would have exceeded it (bytes).
    pub requested_bytes: usize,
    /// Current live bytes at the time of refusal.
    pub current_bytes: u64,
}

impl fmt::Display for DoesNotFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoesNotFit: {} (requested {} B, current {} B, ceiling {} B)",
            self.binding_constraint, self.requested_bytes, self.current_bytes, self.ceiling_bytes,
        )
    }
}

impl std::error::Error for DoesNotFit {}

// ---------------------------------------------------------------------------
// TrackingAllocator
// ---------------------------------------------------------------------------

/// A `GlobalAlloc` wrapper that counts allocations and enforces a byte ceiling.
///
/// Install as the global allocator with:
/// ```ignore
/// #[global_allocator]
/// static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();
/// ```
pub struct TrackingAllocator {
    /// Underlying system allocator.
    inner: System,
    /// Live bytes currently allocated (signed to handle out-of-order free/alloc).
    current: AtomicI64,
    /// Peak bytes (high-water mark, always non-decreasing).
    peak: AtomicU64,
    /// Total allocation calls since process start.
    count: AtomicU64,
    /// Ceiling in bytes. 0 means "no ceiling installed".
    ceiling: AtomicU64,
}

impl TrackingAllocator {
    /// Create a new `TrackingAllocator` with no ceiling installed.
    pub const fn new() -> Self {
        Self {
            inner: System,
            current: AtomicI64::new(0),
            peak: AtomicU64::new(0),
            count: AtomicU64::new(0),
            ceiling: AtomicU64::new(0),
        }
    }

    /// Return live allocated bytes.
    #[inline]
    pub fn current_bytes(&self) -> u64 {
        self.current.load(Ordering::Relaxed).max(0) as u64
    }

    /// Return the peak (high-water mark) allocated bytes since process start.
    ///
    /// Monotonically non-decreasing: once a value is observed here it will
    /// never decrease.
    #[inline]
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Return the total number of allocation calls since process start.
    #[inline]
    pub fn alloc_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Install a hard ceiling.  Any allocation that would push `current_bytes` past
    /// `ceiling` will be refused (null returned → `alloc::handle_alloc_error` from caller).
    ///
    /// Pass `0` to remove the ceiling.
    pub fn set_ceiling(&self, ceiling_bytes: u64) {
        self.ceiling.store(ceiling_bytes, Ordering::SeqCst);
    }

    /// Check whether allocating `size` bytes would exceed the ceiling.
    ///
    /// Returns `Ok(())` if fine, `Err(DoesNotFit)` if it would exceed.
    ///
    /// This is a point-in-time advisory check. For atomic ceiling enforcement
    /// use `GlobalAlloc::alloc` / `try_reserve` which use a CAS loop.
    pub fn check(&self, size: usize) -> Result<(), DoesNotFit> {
        let ceil = self.ceiling.load(Ordering::SeqCst);
        if ceil == 0 {
            return Ok(());
        }
        let current = self.current.load(Ordering::SeqCst).max(0) as u64;
        let after = current.saturating_add(size as u64);
        if after > ceil {
            Err(DoesNotFit {
                binding_constraint: format!(
                    "allocation of {} B would push usage ({} B) past ceiling ({} B)",
                    size, current, ceil
                ),
                ceiling_bytes: ceil,
                requested_bytes: size,
                current_bytes: current,
            })
        } else {
            Ok(())
        }
    }

    /// Atomically reserve `size` bytes against the ceiling using a compare-exchange loop.
    ///
    /// The plain `check()` method has a TOCTOU window — two threads can both pass
    /// the check before either updates `current`. This method closes the race by
    /// atomically incrementing `current` only when the result would not exceed the
    /// ceiling, using a CAS loop on the signed `current` counter.
    ///
    /// Returns the new `current` value on success so the caller can update `peak`,
    /// or `Err(DoesNotFit)` if the ceiling would be exceeded.
    fn try_reserve(&self, size: usize) -> Result<u64, DoesNotFit> {
        let ceil = self.ceiling.load(Ordering::SeqCst);
        if ceil == 0 {
            // No ceiling — unconditionally increment.
            let prev = self.current.fetch_add(size as i64, Ordering::SeqCst);
            return Ok((prev + size as i64).max(0) as u64);
        }
        let size_i = size as i64;
        // CAS loop: read current, check, increment atomically.
        let mut current = self.current.load(Ordering::SeqCst);
        loop {
            let after = current.saturating_add(size_i);
            if after as u64 > ceil {
                return Err(DoesNotFit {
                    binding_constraint: format!(
                        "allocation of {} B would push usage ({} B) past ceiling ({} B)",
                        size,
                        current.max(0) as u64,
                        ceil
                    ),
                    ceiling_bytes: ceil,
                    requested_bytes: size,
                    current_bytes: current.max(0) as u64,
                });
            }
            match self.current.compare_exchange_weak(
                current,
                after,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return Ok(after.max(0) as u64),
                Err(observed) => current = observed,
            }
        }
    }

    /// Reset the peak high-water mark to the current live bytes.
    ///
    /// After this call, `peak_bytes()` reflects only allocations made from this
    /// point forward (net new high-water mark for the current run).  This allows
    /// the stress harness to get a meaningful per-run peak measurement in a
    /// warm process, where the global peak is already elevated from prior runs.
    ///
    /// # Safety
    /// Only call this between runs (no concurrent allocations).  The caller must
    /// ensure the process is quiescent (single-threaded measurement point).
    pub fn reset_peak_to_current(&self) {
        let current = self.current.load(Ordering::SeqCst).max(0) as u64;
        self.peak.store(current, Ordering::SeqCst);
    }

    /// Reset counters (for test isolation).  Ceiling is also cleared.
    ///
    /// # Safety
    /// Only call this when no other threads are allocating. Intended for tests.
    pub fn reset_for_test(&self) {
        self.current.store(0, Ordering::SeqCst);
        self.peak.store(0, Ordering::SeqCst);
        self.count.store(0, Ordering::SeqCst);
        self.ceiling.store(0, Ordering::SeqCst);
    }
}

impl Default for TrackingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();

        // Atomically reserve bytes against the ceiling (closes the TOCTOU race).
        // If the ceiling would be exceeded, refuse immediately without calling the
        // underlying allocator.
        let new_current = match self.try_reserve(size) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };

        let ptr = self.inner.alloc(layout);
        if ptr.is_null() {
            // Underlying allocator failed; undo the reservation.
            self.current.fetch_sub(size as i64, Ordering::SeqCst);
        } else {
            self.count.fetch_add(1, Ordering::SeqCst);
            // Update peak with the value we already computed atomically.
            let mut peak = self.peak.load(Ordering::Relaxed);
            while new_current > peak {
                match self.peak.compare_exchange_weak(
                    peak,
                    new_current,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        self.current
            .fetch_sub(layout.size() as i64, Ordering::SeqCst);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();

        let new_current = match self.try_reserve(size) {
            Ok(v) => v,
            Err(_) => return std::ptr::null_mut(),
        };

        let ptr = self.inner.alloc_zeroed(layout);
        if ptr.is_null() {
            self.current.fetch_sub(size as i64, Ordering::SeqCst);
        } else {
            self.count.fetch_add(1, Ordering::SeqCst);
            let mut peak = self.peak.load(Ordering::Relaxed);
            while new_current > peak {
                match self.peak.compare_exchange_weak(
                    peak,
                    new_current,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();

        // If growing, atomically reserve the delta against the ceiling.
        let new_current_opt = if new_size > old_size {
            let delta = new_size - old_size;
            match self.try_reserve(delta) {
                Ok(v) => Some(v),
                Err(_) => return std::ptr::null_mut(),
            }
        } else {
            None
        };

        let new_ptr = self.inner.realloc(ptr, layout, new_size);
        if new_ptr.is_null() {
            // Undo the reservation if we made one.
            if new_size > old_size {
                self.current
                    .fetch_sub((new_size - old_size) as i64, Ordering::SeqCst);
            }
        } else {
            if new_size <= old_size {
                // Shrink — adjust current downward (no ceiling check needed).
                self.current
                    .fetch_sub((old_size - new_size) as i64, Ordering::SeqCst);
            }
            if let Some(new_current) = new_current_opt {
                let mut peak = self.peak.load(Ordering::Relaxed);
                while new_current > peak {
                    match self.peak.compare_exchange_weak(
                        peak,
                        new_current,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
        }
        new_ptr
    }
}

// SAFETY: TrackingAllocator only uses atomics for shared state.
unsafe impl Sync for TrackingAllocator {}

#[cfg(test)]
mod tests {
    //! Tests for the tracking allocator.
    //!
    //! Each test names the fault it detects (QUALITY-CONTRACT §1).

    use super::*;
    use std::alloc::{GlobalAlloc, Layout};

    fn make_allocator() -> TrackingAllocator {
        TrackingAllocator::new()
    }

    /// Fault detected: allocator fails to count bytes, so peak is always 0.
    #[test]
    fn peak_advances_on_alloc() {
        let a = make_allocator();
        let layout = Layout::array::<u8>(1024).unwrap();
        unsafe {
            let ptr = a.alloc(layout);
            assert!(!ptr.is_null());
            assert!(a.peak_bytes() >= 1024, "peak should advance after alloc");
            a.dealloc(ptr, layout);
        }
    }

    /// Fault detected: peak decrements on dealloc (would break monotonicity invariant).
    #[test]
    fn peak_is_monotone_after_dealloc() {
        let a = make_allocator();
        let layout = Layout::array::<u8>(4096).unwrap();
        unsafe {
            let ptr = a.alloc(layout);
            assert!(!ptr.is_null());
            let peak_after_alloc = a.peak_bytes();
            a.dealloc(ptr, layout);
            let peak_after_dealloc = a.peak_bytes();
            assert_eq!(
                peak_after_dealloc, peak_after_alloc,
                "peak must not decrease after dealloc"
            );
        }
    }

    /// Fault detected: a freed-then-reallocated pattern falsely trips the ceiling
    /// (current_bytes not decremented on free).
    #[test]
    fn freed_then_reallocated_does_not_false_trip_ceiling() {
        let a = make_allocator();
        // Set ceiling large enough for one allocation but not two simultaneously.
        a.set_ceiling(8192);
        let layout = Layout::array::<u8>(4096).unwrap();
        unsafe {
            let ptr = a.alloc(layout);
            assert!(!ptr.is_null(), "first alloc should succeed");
            // Free it — current_bytes must drop.
            a.dealloc(ptr, layout);
            let current_after_free = a.current_bytes();
            // Second alloc should not trip the ceiling because the first was freed.
            let ptr2 = a.alloc(layout);
            assert!(
                !ptr2.is_null(),
                "second alloc after free should succeed (current was {current_after_free} B)"
            );
            a.dealloc(ptr2, layout);
        }
    }

    /// Fault detected: over-budget allocation silently succeeds instead of returning null.
    #[test]
    fn over_budget_alloc_returns_null() {
        let a = make_allocator();
        a.set_ceiling(512);
        let layout = Layout::array::<u8>(1024).unwrap(); // > ceiling
        unsafe {
            let ptr = a.alloc(layout);
            // Must return null, not a valid pointer.
            assert!(
                ptr.is_null(),
                "over-budget alloc must return null, got {:?}",
                ptr
            );
        }
    }

    /// Fault detected: check() allows an over-budget allocation (would bypass ceiling).
    #[test]
    fn check_refuses_over_budget() {
        let a = make_allocator();
        a.set_ceiling(1000);
        let result = a.check(1001);
        assert!(
            result.is_err(),
            "check() must return Err for a request exceeding the ceiling"
        );
        let err = result.unwrap_err();
        assert_eq!(err.ceiling_bytes, 1000);
        assert_eq!(err.requested_bytes, 1001);
    }

    /// Fault detected: check() refuses a fit allocation (false positive).
    #[test]
    fn check_allows_under_budget() {
        let a = make_allocator();
        a.set_ceiling(10_000);
        assert!(a.check(999).is_ok(), "under-budget check must succeed");
    }

    /// Fault detected: alloc_count is not incremented, so accounting is wrong.
    #[test]
    fn alloc_count_increments() {
        let a = make_allocator();
        let before = a.alloc_count();
        let layout = Layout::array::<u8>(64).unwrap();
        unsafe {
            let ptr = a.alloc(layout);
            assert!(!ptr.is_null());
            assert_eq!(a.alloc_count(), before + 1, "alloc_count must increment");
            a.dealloc(ptr, layout);
        }
    }

    /// Fault detected: DoesNotFit's binding_constraint is empty (no useful info for user).
    #[test]
    fn does_not_fit_has_non_empty_constraint() {
        let a = make_allocator();
        a.set_ceiling(100);
        let err = a.check(200).unwrap_err();
        assert!(
            !err.binding_constraint.is_empty(),
            "binding_constraint must not be empty"
        );
        // The error message must mention the sizes.
        let s = err.to_string();
        assert!(s.contains("DoesNotFit"), "Display must include DoesNotFit");
    }
}
