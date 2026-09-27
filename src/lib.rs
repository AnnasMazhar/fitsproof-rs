//! fitsproof-rs — the compiled edition of fitsproof.
//!
//! The contract this crate exists to enforce: an LLM memory budget that is *enforced* by a
//! byte-counting allocator and then *proven* against measured process memory.
//!
//! v0.1 scope, module map and the definition of done live in
//! `specs/fitsproof-rs.md` of the campaign repo (`~/portfolio/specs/fitsproof-rs.md`).

/// Crate version, surfaced by `fitsproof --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_not_empty() {
        assert!(!super::VERSION.is_empty());
    }
}
