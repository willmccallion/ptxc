//! Activations, `Clip` and `PRelu`, composed from scalar operations in the
//! forms `ptxc-interp` evaluates.

use super::NodeContext;
use crate::compose::{add, at_least, div, f32, greater, less, max, min, mul, neg, select, unary};
use crate::error::LowerError;
use crate::precision::{demote, promote};
use crate::stage::{LoweredGroup, Operand};
use ptxc_graph::{Activation, GeluApproximation};
use ptxc_loop::{Scalar, TensorExpr, UnaryOp};
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

pub(super) fn activation(
    cx: &NodeContext<'_>,
    activation: Activation,
) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x] = cx.exact(operands)?;
        Ok(apply(activation, x))
    })
}

/// `Clip(x, min, max)` as `min` below `min`, then `max` above `max`, so `max`
/// wins when the bounds cross and NaN passes through.
pub(super) fn clip(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x, low, high] = cx.exact(operands)?;
        let (low, high) = (low.expr.clone(), high.expr.clone());
        let raised = select(less(x.expr.clone(), low.clone()), low, x.expr.clone());
        Ok(select(greater(raised.clone(), high.clone()), high, raised))
    })
}

/// `PRelu(x, slope)`: `slope * x` where `x` is negative, else `x`.
pub(super) fn prelu(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    cx.elementwise(|operands| {
        let [x, slope] = cx.exact(operands)?;
        let wide = |operand: &Operand| promote(operand.expr.clone(), operand.dtype);
        let scaled = demote(mul(wide(slope), wide(x)), x.dtype);
        Ok(select(less(x.expr.clone(), zero(x)), scaled, x.expr.clone()))
    })
}

const fn zero(x: &Operand) -> TensorExpr {
    TensorExpr::Const(Scalar::zero(x.dtype))
}

/// `max(x, 0)` in `x`'s own type, passing NaN through.
fn relu(x: &Operand) -> TensorExpr {
    select(less(x.expr.clone(), zero(x)), zero(x), x.expr.clone())
}

/// `activation` of `x`: `Relu` in `x`'s own type, which also takes signed
/// integers, and every other activation in f32.
fn apply(activation: Activation, input: &Operand) -> TensorExpr {
    let wide = promote(input.expr.clone(), input.dtype);
    let x = || wide.clone();
    let computed = match activation {
        Activation::Relu => return relu(input),
        Activation::Sigmoid => sigmoid(&x()),
        Activation::Tanh => unary(UnaryOp::Tanh, x()),
        Activation::Softplus => softplus(&x()),
        Activation::Softsign => div(x(), add(f32(1.0), unary(UnaryOp::Abs, x()))),
        Activation::HardSwish => mul(x(), clamp_unit(add(div(x(), f32(6.0)), f32(0.5)))),
        Activation::Mish => mul(x(), unary(UnaryOp::Tanh, softplus(&x()))),
        Activation::Gelu(approximation) => gelu(approximation, &x()),
        Activation::LeakyRelu { alpha } => {
            select(at_least(x(), f32(0.0)), x(), mul(f32(alpha.get()), x()))
        }
        Activation::Elu { alpha } => {
            select(at_least(x(), f32(0.0)), x(), scaled_expm1(alpha.get(), x()))
        }
        Activation::Celu { alpha } => {
            let alpha = alpha.get();
            let below = scaled_expm1(alpha, div(x(), f32(alpha)));
            add(max(x(), f32(0.0)), min(below, f32(0.0)))
        }
        Activation::ThresholdedRelu { alpha } => {
            select(greater(x(), f32(alpha.get())), x(), f32(0.0))
        }
        Activation::Selu { alpha, gamma } => {
            let inner = select(greater(x(), f32(0.0)), x(), scaled_expm1(alpha.get(), x()));
            mul(f32(gamma.get()), inner)
        }
        Activation::HardSigmoid { alpha, beta } => {
            clamp_unit(add(mul(f32(alpha.get()), x()), f32(beta.get())))
        }
    };
    demote(computed, input.dtype)
}

/// `1 / (1 + e^-x)` for `x >= 0`, else `e^x / (1 + e^x)`: neither overflows.
fn sigmoid(x: &TensorExpr) -> TensorExpr {
    let positive = div(f32(1.0), add(f32(1.0), unary(UnaryOp::Exp, neg(x.clone()))));
    let e = unary(UnaryOp::Exp, x.clone());
    let negative = div(e.clone(), add(f32(1.0), e));
    select(at_least(x.clone(), f32(0.0)), positive, negative)
}

/// `ln(1 + e^x)` as `max(x, 0) + ln(1 + e^-|x|)`, which cannot overflow.
fn softplus(x: &TensorExpr) -> TensorExpr {
    let tail = unary(UnaryOp::Exp, neg(unary(UnaryOp::Abs, x.clone())));
    add(max(x.clone(), f32(0.0)), unary(UnaryOp::Log1p, tail))
}

fn gelu(approximation: GeluApproximation, x: &TensorExpr) -> TensorExpr {
    let half_x = mul(f32(0.5), x.clone());
    let cdf = match approximation {
        GeluApproximation::None => unary(UnaryOp::Erf, mul(x.clone(), f32(FRAC_1_SQRT_2))),
        GeluApproximation::Tanh => {
            let cube = mul(mul(x.clone(), x.clone()), x.clone());
            let inner = add(mul(f32(0.044_715), cube), x.clone());
            unary(UnaryOp::Tanh, mul(f32(FRAC_2_SQRT_PI * FRAC_1_SQRT_2), inner))
        }
    };
    mul(half_x, add(f32(1.0), cdf))
}

/// `alpha * (e^x - 1)`.
fn scaled_expm1(alpha: f32, x: TensorExpr) -> TensorExpr {
    mul(f32(alpha), unary(UnaryOp::Expm1, x))
}

/// `x` limited to `[0, 1]`, NaN passing through.
fn clamp_unit(x: TensorExpr) -> TensorExpr {
    min(max(x, f32(0.0)), f32(1.0))
}
