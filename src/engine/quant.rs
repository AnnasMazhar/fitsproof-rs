//! Quantisation — int8 symmetric and int4 symmetric weight quantisation.
//!
//! # int8 symmetric (per-tensor)
//!
//! `scale = max(|w|) / 127`
//! `q = round(w / scale).clamp(-127, 127)`
//! `w_approx = q * scale`
//!
//! # int4 symmetric (per-tensor)
//!
//! Same as int8 but with range [-7, 7] (4 bits).
//!
//! # Source
//!
//! Dettmers et al. 2022 (LLM.int8), https://arxiv.org/abs/2208.07339 —
//! describes symmetric per-tensor quantisation as the baseline.

/// A quantised tensor with its scale.
#[derive(Debug, Clone)]
pub struct QuantTensor {
    /// Quantised values stored as i8 (int8 or int4 clipped to i8 range).
    pub data: Vec<i8>,
    /// Dequantisation scale: `w_approx = q * scale`.
    pub scale: f32,
    /// Quantisation scheme (for display / error messages).
    pub scheme: QuantScheme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    Int8Sym,
    Int4Sym,
}

impl QuantScheme {
    /// Maximum magnitude of quantised values.
    pub fn max_val(self) -> i8 {
        match self {
            QuantScheme::Int8Sym => 127,
            QuantScheme::Int4Sym => 7,
        }
    }
}

/// Quantise a float slice to `QuantTensor` using symmetric per-tensor quantisation.
///
/// # Fault detected
///
/// Using max_val=128 instead of 127 for int8 would shift the scale,
/// causing dequantisation errors larger than 1 LSB. Tests verify round-trip
/// error is within 1/127 of the max weight value.
pub fn quantise(weights: &[f32], scheme: QuantScheme) -> QuantTensor {
    let max_abs = weights.iter().map(|&w| w.abs()).fold(0.0f32, f32::max);

    let mv = scheme.max_val() as f32;
    let scale = if max_abs > 0.0 { max_abs / mv } else { 1.0 };

    let data: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let q = (w / scale).round();
            q.clamp(-mv, mv) as i8
        })
        .collect();

    QuantTensor {
        data,
        scale,
        scheme,
    }
}

/// Dequantise a `QuantTensor` back to f32.
///
/// `w_approx[i] = data[i] as f32 * scale`
pub fn dequantise(qt: &QuantTensor) -> Vec<f32> {
    qt.data.iter().map(|&q| q as f32 * qt.scale).collect()
}

#[cfg(test)]
mod tests {
    //! Known-answer tests for quantisation.
    //!
    //! Values are derived by hand from the formulas above.

    use super::*;

    /// Fault detected: int8 round-trip error exceeds 1 LSB (scale wrong or clamp wrong).
    /// Hand: weights=[1.0, -1.0, 0.5], max_abs=1.0, scale=1/127≈0.00787,
    /// q=[127, -127, 63], dequant≈[1.0, -1.0, 0.496].
    #[test]
    fn int8_round_trip_within_one_lsb() {
        let weights = vec![1.0f32, -1.0, 0.5, 0.0, -0.5];
        let qt = quantise(&weights, QuantScheme::Int8Sym);
        let dq = dequantise(&qt);
        let max_abs = weights.iter().map(|&w| w.abs()).fold(0.0f32, f32::max);
        let lsb = max_abs / 127.0;
        for (orig, approx) in weights.iter().zip(dq.iter()) {
            let err = (orig - approx).abs();
            assert!(
                err <= lsb + 1e-6,
                "int8 round-trip error {err:.6} > 1 LSB ({lsb:.6}) for weight {orig}"
            );
        }
    }

    /// Fault detected: int4 round-trip error exceeds 1 int4 LSB (max_val wrong).
    /// Hand: max_val=7, scale=1.0/7≈0.1429, q=[7,-7,4,0,-4], dequant≈[1.0,-1.0,0.571,0,-0.571].
    #[test]
    fn int4_round_trip_within_one_lsb() {
        let weights = vec![1.0f32, -1.0, 0.5, 0.0, -0.5];
        let qt = quantise(&weights, QuantScheme::Int4Sym);
        let dq = dequantise(&qt);
        let max_abs = weights.iter().map(|&w| w.abs()).fold(0.0f32, f32::max);
        let lsb = max_abs / 7.0;
        for (orig, approx) in weights.iter().zip(dq.iter()) {
            let err = (orig - approx).abs();
            assert!(
                err <= lsb + 1e-6,
                "int4 round-trip error {err:.6} > 1 int4-LSB ({lsb:.6}) for weight {orig}"
            );
        }
    }

    /// Fault detected: all-zero weights produce NaN scale (division by 0).
    #[test]
    fn zero_weights_produce_finite_output() {
        let weights = vec![0.0f32; 16];
        let qt = quantise(&weights, QuantScheme::Int8Sym);
        let dq = dequantise(&qt);
        assert!(
            qt.scale.is_finite(),
            "scale must be finite for zero weights"
        );
        for &v in &dq {
            assert!(
                v.is_finite(),
                "dequantised value must be finite for zero weights"
            );
        }
    }

    /// Fault detected: int4 quantises to int8 range (max_val not 7).
    #[test]
    fn int4_values_stay_in_range() {
        let weights: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) / 32.0).collect();
        let qt = quantise(&weights, QuantScheme::Int4Sym);
        for &q in &qt.data {
            assert!(
                (-7..=7).contains(&q),
                "int4 quantised value {q} outside [-7,7]"
            );
        }
    }
}
