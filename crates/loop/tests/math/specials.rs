//! IEEE's special cases, and the boundaries of each program's range.

use ptxc_loop::math;

#[test]
fn exp_meets_ieee_special_cases() {
    let exp = |x| math::exp().eval(x);

    assert!(exp(f32::NAN).is_nan());
    assert_eq!(exp(f32::INFINITY), f32::INFINITY);
    assert_eq!(exp(f32::NEG_INFINITY), 0.0);
    assert_eq!(exp(0.0), 1.0);
    assert_eq!(exp(-0.0), 1.0);
    assert_eq!(exp(88.722_84), f32::INFINITY);
    assert!(exp(88.722_83).is_finite());
    assert_eq!(exp(-103.972_09), 0.0);
    assert_eq!(exp(-103.278_93), f32::from_bits(1));
}

#[test]
fn log_meets_ieee_special_cases() {
    let log = |x| math::log().eval(x);

    assert!(log(f32::NAN).is_nan());
    assert!(log(-1.0).is_nan());
    assert!(log(f32::NEG_INFINITY).is_nan());
    assert_eq!(log(f32::INFINITY), f32::INFINITY);
    assert_eq!(log(0.0), f32::NEG_INFINITY);
    assert_eq!(log(-0.0), f32::NEG_INFINITY);
    assert_eq!(log(1.0), 0.0);
    assert_eq!(log(f32::from_bits(1)), -103.278_93);
}

#[test]
fn expm1_meets_ieee_special_cases() {
    let expm1 = |x| math::expm1().eval(x);

    assert!(expm1(f32::NAN).is_nan());
    assert_eq!(expm1(f32::INFINITY), f32::INFINITY);
    assert_eq!(expm1(f32::NEG_INFINITY), -1.0);
    assert_eq!(expm1(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(expm1(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(expm1(f32::from_bits(1)), f32::from_bits(1));
    assert_eq!(expm1(-20.0), -1.0);
    assert_eq!(expm1(88.722_84), f32::INFINITY);
    assert!(expm1(88.722_83).is_finite());
}

#[test]
fn log1p_meets_ieee_special_cases() {
    let log1p = |x| math::log1p().eval(x);

    assert!(log1p(f32::NAN).is_nan());
    assert!(log1p(-2.0).is_nan());
    assert_eq!(log1p(-1.0), f32::NEG_INFINITY);
    assert_eq!(log1p(f32::INFINITY), f32::INFINITY);
    assert_eq!(log1p(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(log1p(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(log1p(f32::from_bits(1)), f32::from_bits(1));
    assert!(log1p(f32::MAX).is_finite());
}

#[test]
fn tanh_meets_ieee_special_cases() {
    let tanh = |x| math::tanh().eval(x);

    assert!(tanh(f32::NAN).is_nan());
    assert_eq!(tanh(f32::INFINITY), 1.0);
    assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
    assert_eq!(tanh(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(tanh(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(tanh(f32::from_bits(1)), f32::from_bits(1));
    assert_eq!(tanh(-9.5), -1.0);
}

#[test]
fn erf_meets_ieee_special_cases() {
    let erf = |x| math::erf().eval(x);

    assert!(erf(f32::NAN).is_nan());
    assert_eq!(erf(f32::INFINITY), 1.0);
    assert_eq!(erf(f32::NEG_INFINITY), -1.0);
    assert_eq!(erf(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(erf(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(erf(f32::from_bits(1)), f32::from_bits(1));
    assert_eq!(erf(3.95), 1.0);
    assert!(erf(3.9) < 1.0);
}

#[test]
fn sin_and_cos_meet_ieee_special_cases() {
    let sin = |x| math::sin().eval(x);
    let cos = |x| math::cos().eval(x);

    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(sin(x).is_nan() && cos(x).is_nan(), "{x}");
    }
    assert_eq!(sin(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(sin(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cos(0.0), 1.0);
    assert_eq!(cos(-0.0), 1.0);
    assert_eq!(sin(f32::from_bits(1)), f32::from_bits(1));
    assert_eq!(sin(-f32::MAX), -sin(f32::MAX));
    assert_eq!(cos(-f32::MAX), cos(f32::MAX));
}
