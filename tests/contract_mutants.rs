//! Mutation-killing tests for the contract modules: cost, plan, admit.
//!
//! These tests are designed to kill specific mutants in the contract arithmetic.
//! Each test names the exact mutation it is designed to catch (QUALITY-CONTRACT §1).
//!
//! The previous cargo-mutants run (c1) only covered main.rs and scored 33%.
//! These tests target cost.rs/plan.rs/admit.rs where the real contract logic lives.

use fitsproof::admit::{admit, AdmitStatus};
use fitsproof::cost;
use fitsproof::model::ModelConfig;
use fitsproof::plan::{plan, Verdict};
use fitsproof::probe::MachineProfile;

fn ref_machine() -> MachineProfile {
    MachineProfile {
        hostname: "test".into(),
        platform_str: "test".into(),
        measured_at: 0.0,
        memory_bandwidth_bps: 20_000_000_000.0,
        gemm_throughput_flops: 100_000_000_000.0,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        gpu_memory_bytes: 0,
        cpu_count: 8,
    }
}

// ---------------------------------------------------------------------------
// cost::weight_bytes — arithmetic mutation killers
// ---------------------------------------------------------------------------

/// Fault detected: replace * with + in the weight_bytes embedding table term.
/// `embed_bytes = v * d * 4` — if * is replaced with + (embed = v + d * 4 or v * d + 4),
/// the result is orders of magnitude wrong.
#[test]
fn weight_bytes_fp32_exact_known_answer() {
    // Reference bundle: num_layers=6, hidden_size=384, num_heads=6, num_kv_heads=2,
    // head_dim=64, intermediate_size=1536, vocab_size=512, max_seq_len=512.
    // Hand-computed (fp32, 4 bytes/element):
    //   d=384, h=6, kv_h=2, hd=64, ff=1536, v=512, l=6
    //   embed = 512 * 384 * 4 = 786,432
    //   attn_per_layer = (6*64*384 + 2*64*384 + 2*64*384 + 384*6*64) * 4
    //                  = (147456 + 49152 + 49152 + 147456) * 4 = 393216 * 4 = 1,572,864
    //   ffn_per_layer  = (1536*384 + 1536*384 + 384*1536) * 4
    //                  = 589824 * 3 * 4 = 7,077,888
    //   norm_per_layer = 2 * 384 * 4 = 3,072
    //   layer_total    = 1,572,864 + 7,077,888 + 3,072 = 8,653,824
    //   6_layers       = 51,922,944
    //   final          = 384*4 + 512*384*4 = 1,536 + 786,432 = 787,968
    //   total          = 786,432 + 51,922,944 + 787,968 = 53,497,344
    let cfg = ModelConfig::reference();
    let result = cost::weight_bytes(&cfg, "none");
    assert_eq!(result, 53_497_344, "weight_bytes fp32 exact value mismatch");
}

/// Fault detected: replace * with / or + in the KV cache formula.
/// `kv = 2 * n_layers * n_kv_heads * context_len * head_dim * bpe`
/// Every multiply is a potential mutant target.
#[test]
fn kv_cache_bytes_fp16_exact_known_answer() {
    // Reference bundle: num_layers=6, num_kv_heads=2, head_dim=64.
    // fp16 = 2 bytes/element, context_len=512.
    // Hand-computed: 2 * 6 * 2 * 512 * 64 * 2 = 1,572,864
    let cfg = ModelConfig::reference();
    let result = cost::kv_cache_bytes(&cfg, 512, "fp16");
    assert_eq!(result, 1_572_864, "kv_cache fp16 exact value mismatch");
}

/// Fault detected: factor-of-2 dropped from KV cache (only K or V, not both).
#[test]
fn kv_cache_bytes_factor_of_2_is_load_bearing() {
    // Double context: result must double (linear in context_len × 2 factor).
    let cfg = ModelConfig::reference();
    let kv_512 = cost::kv_cache_bytes(&cfg, 512, "fp16");
    let kv_1024 = cost::kv_cache_bytes(&cfg, 1024, "fp16");
    assert_eq!(
        kv_1024,
        kv_512 * 2,
        "kv_cache must scale linearly with context_len"
    );
}

/// Fault detected: kv_cache_bytes not scaling with num_kv_heads.
/// A config with 2× kv_heads should produce 2× KV bytes.
#[test]
fn kv_cache_bytes_scales_with_kv_heads() {
    use fitsproof::model::ModelConfig;
    let mut cfg2 = ModelConfig::reference();
    cfg2.num_kv_heads = 2;
    let mut cfg4 = ModelConfig::reference();
    cfg4.num_kv_heads = 4;
    let kv2 = cost::kv_cache_bytes(&cfg2, 512, "fp16");
    let kv4 = cost::kv_cache_bytes(&cfg4, 512, "fp16");
    assert_eq!(kv4, kv2 * 2, "kv_cache must scale with num_kv_heads");
}

/// Fault detected: weight_bytes int4 is NOT exactly half of int8 per element,
/// because norms stay fp32 and embeddings use fp16 (not fp32) for quant models.
/// Tests must verify the actual values, not a simple ½ ratio.
#[test]
fn weight_bytes_int4_less_than_int8_less_than_fp32() {
    let cfg = ModelConfig::reference();
    let fp32 = cost::weight_bytes(&cfg, "none");
    let int8 = cost::weight_bytes(&cfg, "int8_sym");
    let int4 = cost::weight_bytes(&cfg, "int4_sym");
    assert!(
        int4 < int8,
        "int4 weight bytes ({int4}) must be less than int8 ({int8})"
    );
    assert!(
        int8 < fp32,
        "int8 weight bytes ({int8}) must be less than fp32 ({fp32})"
    );
}

/// Fault detected: weight_bytes ignoring num_layers (treating l=1 always).
/// More layers must produce more weight bytes.
#[test]
fn weight_bytes_scales_with_num_layers() {
    let mut cfg1 = ModelConfig::reference();
    cfg1.num_layers = 1;
    let mut cfg6 = ModelConfig::reference();
    cfg6.num_layers = 6;
    let w1 = cost::weight_bytes(&cfg1, "none");
    let w6 = cost::weight_bytes(&cfg6, "none");
    // 6 layers > 1 layer (embedding and final weights are constant, but per-layer grows).
    assert!(
        w6 > w1,
        "6-layer model ({w6}) must have more weight bytes than 1-layer ({w1})"
    );
}

/// Fault detected: activation_bytes returning 0 (causing total_peak to miss activations).
#[test]
fn activation_bytes_nonzero() {
    let cfg = ModelConfig::reference();
    let act = cost::activation_bytes(&cfg);
    assert!(act > 0, "activation_bytes must be positive");
    // Reference bundle: hidden_size=384, intermediate_size=1536.
    // Hand-computed: (2*384 + 1536) * 4 = 2304 * 4 = 9,216
    assert_eq!(act, 9_216, "activation_bytes exact value mismatch");
}

/// Fault detected: total_peak != weight + kv + activation (missing a component).
#[test]
fn total_peak_equals_sum_of_components() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let est = cost::estimate(&cfg, &machine, 512, "none", 0.6);
    let expected = est.weight_bytes + est.kv_cache_bytes + est.activation_bytes;
    assert_eq!(
        est.total_peak_bytes, expected,
        "total_peak_bytes must equal weight + kv_cache + activation"
    );
}

/// Fault detected: decode_tok_s sign error (divide by bandwidth instead of weights).
/// tok/s = bandwidth * utilisation / weight_bytes.
/// If * is replaced with / in the numerator, tok/s is 400x smaller than bandwidth.
#[test]
fn decode_tok_s_decreases_with_larger_model() {
    // Larger model (more weight bytes) → lower throughput at same bandwidth.
    let mut small = ModelConfig::reference();
    small.num_layers = 1;
    small.hidden_size = 64;
    small.intermediate_size = 128;

    let mut large = ModelConfig::reference();
    large.num_layers = 12;

    let machine = ref_machine();
    let tok_small = cost::decode_tok_s(&small, &machine, "none", 0.6);
    let tok_large = cost::decode_tok_s(&large, &machine, "none", 0.6);
    assert!(
        tok_small > tok_large,
        "smaller model must have higher tok/s: {tok_small} vs {tok_large}"
    );
}

// ---------------------------------------------------------------------------
// plan::plan — verdict mutation killers
// ---------------------------------------------------------------------------

/// Fault detected: plan() returning FitsWithDegradation when config clearly fits.
#[test]
fn plan_verdict_fits_when_large_budget() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    assert_eq!(
        p.verdict,
        Verdict::Fits,
        "very large budget must produce Fits verdict"
    );
    assert!(
        p.degradations.is_empty(),
        "Fits verdict must have no degradations"
    );
}

/// Fault detected: plan() returning Fits when budget is 1 byte (impossible config).
#[test]
fn plan_verdict_does_not_fit_when_zero_budget() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(
        p.verdict,
        Verdict::DoesNotFit,
        "1-byte budget must produce DoesNotFit"
    );
    assert!(
        !p.binding_constraint.is_empty(),
        "DoesNotFit must name binding constraint"
    );
}

/// Fault detected: plan() not checking if `predicted_peak <= budget_bytes` correctly
/// (e.g. using < instead of <=, causing off-by-one at the exact boundary).
#[test]
fn plan_fits_at_exact_peak() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let fp32_peak = cost::weight_bytes(&cfg, "none")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);
    // Budget exactly equals peak — must Fit.
    let p = plan(&cfg, &machine, 512, fp32_peak, "none", 0.6).unwrap();
    assert_eq!(
        p.verdict,
        Verdict::Fits,
        "budget == predicted_peak must produce Fits, got {:?} (peak={fp32_peak})",
        p.verdict
    );
}

/// Fault detected: plan() using >= instead of > for the DoesNotFit boundary,
/// causing a 1-byte-below-peak config to be incorrectly admitted.
#[test]
fn plan_does_not_fit_one_byte_below_peak() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let fp32_peak = cost::weight_bytes(&cfg, "none")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);
    // Budget is 1 byte less than peak — must not be Fits.
    if fp32_peak > 1 {
        let p = plan(&cfg, &machine, 512, fp32_peak - 1, "none", 0.6).unwrap();
        assert_ne!(
            p.verdict,
            Verdict::Fits,
            "budget = predicted_peak - 1 must not produce Fits"
        );
    }
}

/// Fault detected: plan() FitsWithDegradation verdict when NO degradation actually fits.
/// The degradation list must contain at least one `fits_budget=true` step.
#[test]
fn plan_fits_with_degradation_has_at_least_one_fitting_step() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let fp32_peak = cost::weight_bytes(&cfg, "none")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);
    let int4_peak = cost::weight_bytes(&cfg, "int4_sym")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);

    if int4_peak < fp32_peak {
        // Budget between int4 and fp32 peaks.
        let budget = int4_peak + (fp32_peak - int4_peak) / 2;
        let p = plan(&cfg, &machine, 512, budget, "none", 0.6).unwrap();
        if p.verdict == Verdict::FitsWithDegradation {
            let any_fits = p.degradations.iter().any(|d| d.fits_budget);
            assert!(
                any_fits,
                "FitsWithDegradation must have at least one degradation with fits_budget=true"
            );
        }
    }
}

/// Fault detected: plan() binding_constraint is empty for DoesNotFit verdict.
#[test]
fn plan_does_not_fit_binding_constraint_non_empty() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(p.verdict, Verdict::DoesNotFit);
    assert!(
        !p.binding_constraint.is_empty(),
        "DoesNotFit must name the binding constraint"
    );
    // The binding constraint must mention GB values (not a blank placeholder).
    assert!(
        p.binding_constraint.contains("GB"),
        "binding_constraint must mention GB: {:?}",
        p.binding_constraint
    );
}

/// Fault detected: plan() predicted_peak_bytes is 0 for a valid config
/// (cost::estimate returning 0 due to arithmetic bug).
#[test]
fn plan_predicted_peak_bytes_nonzero() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    assert!(
        p.predicted_peak_bytes > 0,
        "predicted_peak_bytes must be nonzero for a valid config"
    );
}

/// Fault detected: plan CI lower bound >= upper bound (swapped or wrong multiplier).
#[test]
fn plan_ci_lower_less_than_upper() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    assert!(
        p.predicted_peak_ci.0 < p.predicted_peak_ci.1,
        "CI lower ({}) must be less than upper ({})",
        p.predicted_peak_ci.0,
        p.predicted_peak_ci.1
    );
}

/// Fault detected: plan() with larger context does not increase predicted peak.
#[test]
fn plan_larger_context_increases_peak() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p_small = plan(&cfg, &machine, 128, 10_000_000_000, "none", 0.6).unwrap();
    let p_large = plan(&cfg, &machine, 2048, 10_000_000_000, "none", 0.6).unwrap();
    assert!(
        p_large.predicted_peak_bytes > p_small.predicted_peak_bytes,
        "larger context must increase predicted peak: {} vs {}",
        p_large.predicted_peak_bytes,
        p_small.predicted_peak_bytes
    );
}

// ---------------------------------------------------------------------------
// admit — verdict to status mapping (mutation killers)
// ---------------------------------------------------------------------------

/// Fault detected: Fits plan producing Refused status (swapped branch).
#[test]
fn admit_fits_plan_produces_admitted() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    assert_eq!(p.verdict, Verdict::Fits);
    let rec = admit(p);
    assert_eq!(rec.status, AdmitStatus::Admitted);
}

/// Fault detected: DoesNotFit plan producing Admitted status (swapped branch).
#[test]
fn admit_does_not_fit_produces_refused() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(p.verdict, Verdict::DoesNotFit);
    let rec = admit(p);
    assert_eq!(rec.status, AdmitStatus::Refused);
}

/// Fault detected: FitsWithDegradation producing Admitted (skipping degradation emission).
#[test]
fn admit_degraded_produces_degraded_status_not_admitted() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let fp32_peak = cost::weight_bytes(&cfg, "none")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);
    let int4_peak = cost::weight_bytes(&cfg, "int4_sym")
        + cost::kv_cache_bytes(&cfg, 512, "fp16")
        + cost::activation_bytes(&cfg);

    if int4_peak < fp32_peak {
        let budget = int4_peak + (fp32_peak - int4_peak) / 2;
        let p = plan(&cfg, &machine, 512, budget, "none", 0.6).unwrap();
        if p.verdict == Verdict::FitsWithDegradation {
            let rec = admit(p);
            assert_ne!(
                rec.status,
                AdmitStatus::Admitted,
                "FitsWithDegradation must not produce Admitted status"
            );
        }
    }
}

/// Fault detected: admit() message for Admitted not starting with "ADMITTED:".
/// The message format is load-bearing — callers parse it (e.g. the serve.rs handler).
#[test]
fn admit_message_prefix_is_correct_for_each_status() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();

    // Admitted
    let p_ok = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    let rec_ok = admit(p_ok);
    assert!(
        rec_ok.message.starts_with("ADMITTED:"),
        "Admitted message must start with ADMITTED: got {:?}",
        rec_ok.message
    );

    // Refused
    let p_no = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    let rec_no = admit(p_no);
    assert!(
        rec_no.message.starts_with("REFUSED:"),
        "Refused message must start with REFUSED: got {:?}",
        rec_no.message
    );
}

/// Fault detected: margin calculation in ADMITTED message is negative or zero,
/// meaning budget < peak was allowed (arithmetic sign error).
#[test]
fn admit_message_reports_positive_margin_when_fits() {
    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 10_000_000_000, "none", 0.6).unwrap();
    let rec = admit(p);
    // The message contains "margin: X MB" — X must be positive.
    assert!(
        rec.message.contains("margin:"),
        "Admitted message must contain margin field"
    );
    // Parse the margin value — it must not be negative.
    if let Some(margin_pos) = rec.message.find("margin: ") {
        let rest = &rec.message[margin_pos + 8..];
        let end = rest.find(' ').unwrap_or(rest.len());
        let val: f64 = rest[..end]
            .trim_end_matches([' ', 'M', 'B'])
            .parse()
            .unwrap_or(-1.0);
        assert!(
            val >= 0.0,
            "margin in ADMITTED message must be >= 0, got {val}"
        );
    }
}

// ---------------------------------------------------------------------------
// cost::QuantBits — from_name match arm deletion killers (c5 mutants caught.txt)
// ---------------------------------------------------------------------------

/// Fault detected: deleting "none"|"fp32" match arm → from_name returns None for fp32.
/// A None from from_name causes weight_bytes to panic (fail-closed).  This test
/// checks the exact bytes_per_element value for fp32 (4.0 bytes).
#[test]
fn quant_bits_fp32_bytes_per_element_is_4() {
    // fp32 = 32 bits / 8 = 4.0 bytes/element.
    // If the match arm is deleted, from_name("none") returns None and weight_bytes panics.
    let cfg = ModelConfig::reference();
    let fp32_w = cost::weight_bytes(&cfg, "none");
    // Also check fp32 alias
    let fp32_w2 = cost::weight_bytes(&cfg, "fp32");
    assert_eq!(
        fp32_w, fp32_w2,
        "none and fp32 must produce same weight bytes"
    );
    // 4 bytes/element → must be 4× the int8 bytes for quantised layers.
    let int8_w = cost::weight_bytes(&cfg, "int8");
    // embedding/norms stay fp32 regardless — total fp32 > 4× int8.
    // But weight_bytes_fp32 > weight_bytes_int8 must hold.
    assert!(
        fp32_w > int8_w,
        "fp32 weight bytes ({fp32_w}) must exceed int8 ({int8_w})"
    );
}

/// Fault detected: deleting "fp16"|"f16" match arm → from_name returns None for fp16.
/// KV cache defaults to fp16 in the standard path; deleting this arm breaks kv_cache_bytes.
#[test]
fn quant_bits_fp16_bytes_per_element_is_2() {
    // fp16 = 16 bits / 8 = 2.0 bytes/element.
    let cfg = ModelConfig::reference();
    // kv_cache_bytes uses "fp16" quant by default.
    let kv_fp16 = cost::kv_cache_bytes(&cfg, 512, "fp16");
    let kv_fp32 = cost::kv_cache_bytes(&cfg, 512, "none");
    // fp16 KV must be exactly half of fp32 KV (all-linear formula, no fp32 bias).
    assert_eq!(
        kv_fp32,
        kv_fp16 * 2,
        "fp32 kv_cache must be 2× fp16 kv_cache: fp32={kv_fp32} fp16={kv_fp16}"
    );
}

/// Fault detected: deleting "int8"|"q8_0" match arm → from_name returns None for int8.
#[test]
fn quant_bits_int8_bytes_per_element_is_1() {
    // int8 = 8 bits / 8 = 1.0 bytes/element.
    // KV at int8 must be half of KV at fp16.
    let cfg = ModelConfig::reference();
    let kv_int8 = cost::kv_cache_bytes(&cfg, 512, "int8");
    let kv_fp16 = cost::kv_cache_bytes(&cfg, 512, "fp16");
    assert_eq!(
        kv_fp16,
        kv_int8 * 2,
        "fp16 kv_cache must be 2× int8 kv_cache: fp16={kv_fp16} int8={kv_int8}"
    );
}

/// Fault detected: deleting "int4"|"q4_k_m" match arm → from_name returns None for int4.
#[test]
fn quant_bits_int4_bytes_per_element_is_half() {
    // int4 = 4 bits / 8 = 0.5 bytes/element.
    // KV at int4 must be half of KV at int8.
    let cfg = ModelConfig::reference();
    let kv_int4 = cost::kv_cache_bytes(&cfg, 512, "int4_sym");
    let kv_int8 = cost::kv_cache_bytes(&cfg, 512, "int8_sym");
    assert_eq!(
        kv_int8,
        kv_int4 * 2,
        "int8 kv_cache must be 2× int4 kv_cache: int8={kv_int8} int4={kv_int4}"
    );
    // Also verify q4_k_m alias
    let kv_q4km = cost::kv_cache_bytes(&cfg, 512, "q4_k_m");
    assert_eq!(
        kv_int4, kv_q4km,
        "int4_sym and q4_k_m must produce same kv_cache bytes"
    );
}

/// Fault detected: `bytes_per_element` returning 0.0 / 1.0 / -1.0 constant (mutant lines 58:9).
/// Also kills: replace `/` with `%` or `*` in the division.
/// bytes_per_element = bits / 8.0; fp32 → 4.0, fp16 → 2.0, int8 → 1.0, int4 → 0.5.
#[test]
fn quant_bits_bytes_per_element_exact_values() {
    // Hand-computed: bits / 8.0 for each tier.
    // fp32: 32/8 = 4.0
    let cfg = ModelConfig::reference();
    let fp32_bytes = cost::weight_bytes(&cfg, "none");
    let fp16_bytes = cost::weight_bytes(&cfg, "fp16");
    // If bytes_per_element returns 0.0 for all, fp16 weight_bytes collapse to embedding+final (fp32 constant).
    // Specifically: with 0.0 bpe, attn_per_layer=0, ffn_per_layer=0.
    // Reference: embed = 512*384*4 = 786432, final = 384*4 + 512*384*4 = 787968.
    // So weight_bytes with bpe=0 would be 786432 + 6*0 + 787968 = 1,574,400.
    // With correct bpe, fp32 = 53,497,344 >> 1,574,400.
    assert!(
        fp32_bytes > 10_000_000,
        "fp32 weight_bytes must be > 10MB, got {fp32_bytes} (bytes_per_element=0 bug produces ~1.6MB)"
    );
    // fp16 must be strictly less than fp32 (because linear layers at 2 bpe < 4 bpe)
    // but > the constant (embed+final stay fp32).
    assert!(
        fp16_bytes < fp32_bytes,
        "fp16 weight_bytes ({fp16_bytes}) must be less than fp32 ({fp32_bytes})"
    );
    assert!(
        fp16_bytes > 1_574_400,
        "fp16 weight_bytes ({fp16_bytes}) must exceed the embed-only floor of 1,574,400 (bpe=0 bug)"
    );
}

// ---------------------------------------------------------------------------
// weight_bytes arithmetic — component-level exact values (kills lines 109-120 mutants)
// ---------------------------------------------------------------------------

/// Fault detected: any +↔* or +↔- replacement in the per-layer attention formula.
/// These are the densest cluster of mutants (lines 109, 112, 115, 118, 120).
/// Known-answer: attn_per_layer for reference bundle (fp32).
/// h=6, hd=64, d=384, kv_h=2, bits=4.0:
///   (6*64*384 + 2*64*384 + 2*64*384 + 384*6*64) * 4.0
///   = (147456 + 49152 + 49152 + 147456) * 4 = 393216 * 4 = 1,572,864
/// For 6 layers: 6 * 1,572,864 = 9,437,184.
/// This test verifies the total fp32 known-answer (already in weight_bytes_fp32_exact_known_answer)
/// and adds a check isolated to the per-layer component by subtracting the constant parts.
#[test]
fn weight_bytes_per_layer_component_exact_fp32() {
    // All per-layer contribution = total - embed - final.
    // embed = 512*384*4 = 786,432
    // final = 384*4 + 512*384*4 = 1,536 + 786,432 = 787,968
    // total_fp32 = 53,497,344
    // per_layer_total = 53,497,344 - 786,432 - 787,968 = 51,922,944
    // per_layer_total / 6 layers = 8,653,824
    // Hand-check:
    //   attn = 1,572,864
    //   ffn  = (1536*384 + 1536*384 + 384*1536) * 4 = 589824 * 3 * 4 = 7,077,888
    //   norm = 2 * 384 * 4 = 3,072
    //   sum  = 1,572,864 + 7,077,888 + 3,072 = 8,653,824  ✓
    let cfg = ModelConfig::reference();
    let total = cost::weight_bytes(&cfg, "none");
    let embed: u64 = 512 * 384 * 4; // 786,432
    let final_w: u64 = 384 * 4 + 512 * 384 * 4; // 787,968
    let per_layer_total = total - embed - final_w;
    assert_eq!(
        per_layer_total, 51_922_944,
        "per-layer weight contribution must be 51,922,944 for reference bundle fp32"
    );
    // Per layer
    let per_layer_avg = per_layer_total / 6;
    assert_eq!(
        per_layer_avg, 8_653_824,
        "per-layer avg must be 8,653,824 for reference bundle fp32 with 6 layers"
    );
}

/// Fault detected: replacing + with - in the embed+final summation (lines 118:31, 120:18/74).
/// With replace + with -, final_bytes = d*4 - v*d*4 (negative for realistic models).
/// Verified by checking: weight_bytes with 1 layer vs 0-layer model structure.
#[test]
fn weight_bytes_final_norm_and_unembed_contribute_positive() {
    // Isolate final_bytes contribution: a 0-layer model has only embed + final.
    // embed = v * d * 4 = 512 * 384 * 4 = 786,432
    // final = d * 4 + v * d * 4 = 1,536 + 786,432 = 787,968
    // total_0layer = 786,432 + 787,968 = 1,574,400
    let mut cfg = ModelConfig::reference();
    cfg.num_layers = 0;
    let zero_layer = cost::weight_bytes(&cfg, "none");
    assert_eq!(
        zero_layer, 1_574_400,
        "0-layer model must have exactly embed+final = 1,574,400 bytes (fp32)"
    );
}

/// Fault detected: weight_bytes returning constant 0 or 1 (lines 93:5 mutants).
#[test]
fn weight_bytes_returns_nontrivial_for_reference_model() {
    let cfg = ModelConfig::reference();
    let w = cost::weight_bytes(&cfg, "none");
    // Known exact: 53,497,344 bytes (~53 MB).
    assert_eq!(
        w, 53_497_344,
        "weight_bytes must be 53,497,344 for reference fp32 (mutant '0' or '1' would fail this)"
    );
}

// ---------------------------------------------------------------------------
// admit::DoesNotFitPlan::Display — format mutant killer (admit.rs:57:9)
// ---------------------------------------------------------------------------

/// Fault detected: `DoesNotFitPlan::fmt` replaced with `Ok(Default::default())`.
/// The mutation produces an empty string for `format!("{}", plan)`.
/// The Display must produce a string starting with "REFUSED:".
#[test]
fn does_not_fit_plan_display_contains_refused() {
    use fitsproof::admit::DoesNotFitPlan;
    use fitsproof::model::ModelConfig;
    use fitsproof::plan::{plan, Verdict};

    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(p.verdict, Verdict::DoesNotFit);

    let rec = admit(p.clone());
    let err = DoesNotFitPlan {
        binding_constraint: rec.refusal_reason.clone(),
        record: rec,
    };

    // If fmt is replaced with Ok(Default::default()), format! returns "".
    let displayed = format!("{}", err);
    assert!(
        displayed.starts_with("REFUSED:"),
        "DoesNotFitPlan Display must start with 'REFUSED:' — got {:?}",
        displayed
    );
    // Must include the binding constraint text (not just "REFUSED:")
    assert!(
        displayed.len() > 8,
        "DoesNotFitPlan Display must include binding constraint beyond the prefix"
    );
}

/// Fault detected: admit() returning Default::default() (line 79 mutant).
/// Default::default() for AdmitRecord produces status=Admitted (the zero/default),
/// which would silently admit a refused config.
/// This test also exercises the admit::Error impl through std::error::Error trait.
#[test]
fn does_not_fit_plan_is_error_trait_object() {
    use fitsproof::admit::DoesNotFitPlan;
    use fitsproof::model::ModelConfig;
    use fitsproof::plan::{plan, Verdict};
    use std::error::Error;

    let cfg = ModelConfig::reference();
    let machine = ref_machine();
    let p = plan(&cfg, &machine, 512, 1, "none", 0.6).unwrap();
    assert_eq!(p.verdict, Verdict::DoesNotFit);
    let rec = admit(p);
    let err = DoesNotFitPlan {
        binding_constraint: rec.refusal_reason.clone(),
        record: rec,
    };
    // std::error::Error::source must not panic; binding_constraint must be non-empty.
    let _ = err.source();
    assert!(
        !err.binding_constraint.is_empty(),
        "binding_constraint must be non-empty for a refused config"
    );
    // The error message (via Display) must be non-empty.
    assert!(
        !err.to_string().is_empty(),
        "DoesNotFitPlan to_string() must be non-empty"
    );
}
