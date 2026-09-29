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

    /// Fault detected: int8 `max_val` is wrong (e.g. 126 instead of 127).
    ///
    /// Ground truth: for `weights = [1.0, -1.0, 0.5, -0.5]`, max_abs = 1.0.
    /// The algorithm specifies `scale = max_abs / max_val = 1.0 / 127 ≈ 0.007874`.
    /// Any other max_val produces a detectably different scale. This KAT pins the
    /// constant to the published value in Dettmers et al. 2022 §2 ("symmetric int8
    /// quantisation uses the range [-127, 127]").
    ///
    /// This test catches faults that `int8_round_trip_within_one_lsb` misses: changing
    /// max_val from 127 to 126 shifts the scale by < 1%, which is within the LSB
    /// tolerance but is still the wrong constant.
    #[test]
    fn int8_scale_is_exact_known_answer() {
        let weights = vec![1.0f32, -1.0, 0.5, -0.5];
        let qt = quantise(&weights, QuantScheme::Int8Sym);
        // max_abs = 1.0; scale = 1.0 / 127
        let expected_scale = 1.0f32 / 127.0;
        assert!(
            (qt.scale - expected_scale).abs() < 1e-7,
            "int8 scale must be max_abs/127 = {expected_scale:.8}, got {:.8}",
            qt.scale
        );
    }

    /// Fault detected: int4 `max_val` is wrong (e.g. 6 instead of 7).
    ///
    /// Ground truth: for `weights = [1.0, -1.0, 0.5, -0.5]`, max_abs = 1.0.
    /// `scale = 1.0 / 7 ≈ 0.142857`. Changing max_val to 6 gives scale = 1/6 ≈ 0.1667,
    /// which is detectably different. Pins the int4 constant to its published value.
    #[test]
    fn int4_scale_is_exact_known_answer() {
        let weights = vec![1.0f32, -1.0, 0.5, -0.5];
        let qt = quantise(&weights, QuantScheme::Int4Sym);
        let expected_scale = 1.0f32 / 7.0;
        assert!(
            (qt.scale - expected_scale).abs() < 1e-6,
            "int4 scale must be max_abs/7 = {expected_scale:.8}, got {:.8}",
            qt.scale
        );
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

    // -----------------------------------------------------------------------
    // Property-based tests (proptest)
    //
    // Properties come from the quantisation algorithm's invariants.
    // Source: Dettmers et al. 2022 (LLM.int8), §2 "Method".
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        /// Property: int8 symmetric quantised values always stay in [-127, 127].
        ///
        /// Fault detected: clamp uses wrong range (e.g. [-128, 128] or [-255, 255]).
        #[test]
        fn int8_values_stay_in_range(
            weights in prop::collection::vec(-10.0f32..10.0, 1..64),
        ) {
            let qt = quantise(&weights, QuantScheme::Int8Sym);
            for &q in &qt.data {
                prop_assert!(
                    (-127..=127).contains(&q),
                    "int8 quantised value {q} outside [-127, 127]"
                );
            }
        }

        /// Property: int4 symmetric quantised values always stay in [-7, 7].
        ///
        /// Fault detected: int4 uses int8 clamp range, so values exceed 7.
        #[test]
        fn int4_values_stay_in_range_proptest(
            weights in prop::collection::vec(-10.0f32..10.0, 1..64),
        ) {
            let qt = quantise(&weights, QuantScheme::Int4Sym);
            for &q in &qt.data {
                prop_assert!(
                    (-7..=7).contains(&q),
                    "int4 quantised value {q} outside [-7, 7]"
                );
            }
        }

        /// Property: dequantised values are within 1 LSB of originals.
        ///
        /// Fault detected: scale computation wrong — round-trip error exceeds 1 LSB.
        /// Derives LSB from the scheme's max_val, not from implementation.
        #[test]
        fn int8_round_trip_within_one_lsb_proptest(
            weights in prop::collection::vec(-1.0f32..1.0, 1..32),
        ) {
            let qt = quantise(&weights, QuantScheme::Int8Sym);
            let dq = dequantise(&qt);
            let max_abs = weights.iter().map(|&w| w.abs()).fold(0.0f32, f32::max);
            if max_abs > 1e-6 {
                let lsb = max_abs / 127.0;
                for (orig, approx) in weights.iter().zip(dq.iter()) {
                    let err = (orig - approx).abs();
                    prop_assert!(
                        err <= lsb + 1e-5,
                        "int8 round-trip error {err:.6} > 1 LSB ({lsb:.6})"
                    );
                }
            }
        }

        /// Property: scale is always positive and finite for any non-zero weight tensor.
        ///
        /// Fault detected: division by zero when max_abs=0 (should use fallback scale=1.0).
        #[test]
        fn scale_is_positive_and_finite(
            weights in prop::collection::vec(-100.0f32..100.0, 1..128),
        ) {
            let qt8 = quantise(&weights, QuantScheme::Int8Sym);
            let qt4 = quantise(&weights, QuantScheme::Int4Sym);
            prop_assert!(qt8.scale > 0.0 && qt8.scale.is_finite(), "int8 scale must be positive and finite");
            prop_assert!(qt4.scale > 0.0 && qt4.scale.is_finite(), "int4 scale must be positive and finite");
        }
    }
}
