//! Tensor utilities shared across engine modules.
//!
//! All ops are scalar-reference paths.  Correctness first; no SIMD here.

/// Softmax in-place on a slice.
///
/// Numerically stable: subtracts max before exp.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (v - max_val).exp()).sum();
    for v in x.iter_mut() {
        *v = ((*v - max_val).exp()) / sum;
    }
}

/// Element-wise sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU activation: x * sigmoid(x).
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fault detected: softmax outputs don't sum to 1 (wrong normalisation).
    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    /// Fault detected: softmax of equal values is not uniform.
    #[test]
    fn softmax_uniform_input() {
        let mut x = vec![2.0f32; 4];
        softmax(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6, "uniform softmax got {v}");
        }
    }

    /// Fault detected: SiLU(0) != 0 (activation wrong at origin).
    /// Hand: silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.
    #[test]
    fn silu_at_zero() {
        assert!((silu(0.0)).abs() < 1e-7, "silu(0) must be 0");
    }
}
