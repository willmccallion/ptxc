//! Activations, `Clip` and `PRelu`, computed in f32 (D19).

use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::{Extents, broadcast_map};
use crate::numeric::{Elements, floats, store, store_floats, widen};
use ptxc_core::float::Overflow;
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{Activation, GeluApproximation, TensorType};
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

pub(super) fn activation(
    activation: Activation,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let values = floats(first(operands)?.data())?;
    let applied: Vec<f32> = values.into_iter().map(|x| apply(activation, x)).collect();
    store_floats(&applied, output.dtype(), Overflow::NonFinite)
}

/// One element, from each activation's definition.
fn apply(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Relu => relu(x),
        Activation::Sigmoid => sigmoid(x),
        Activation::Tanh => x.tanh(),
        Activation::Softplus => softplus(x),
        Activation::Softsign => x / (1.0 + x.abs()),
        Activation::HardSwish => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
        Activation::Mish => x * softplus(x).tanh(),
        Activation::Gelu(approximation) => gelu(approximation, x),
        Activation::LeakyRelu { alpha } => {
            if x >= 0.0 {
                x
            } else {
                alpha.get() * x
            }
        }
        Activation::Elu { alpha } => {
            if x >= 0.0 {
                x
            } else {
                alpha.get() * x.exp_m1()
            }
        }
        Activation::Celu { alpha } => celu(x, alpha.get()),
        Activation::ThresholdedRelu { alpha } => {
            if x > alpha.get() {
                x
            } else {
                0.0
            }
        }
        Activation::Selu { alpha, gamma } => selu(x, alpha.get(), gamma.get()),
        Activation::HardSigmoid { alpha, beta } => {
            alpha.get().mul_add(x, beta.get()).clamp(0.0, 1.0)
        }
    }
}

/// `max(x, 0) + min(alpha * (e^(x / alpha) - 1), 0)`, passing NaN through
/// as NumPy's `maximum` and `minimum` do, where `f32::max` would drop it.
fn celu(x: f32, alpha: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    x.max(0.0) + (alpha * (x / alpha).exp_m1()).min(0.0)
}

/// `max(x, 0)`, passing NaN through.
fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

/// `gamma * x` above zero, else `gamma * alpha * (e^x - 1)`.
fn selu(x: f32, alpha: f32, gamma: f32) -> f32 {
    gamma * if x > 0.0 { x } else { alpha * x.exp_m1() }
}

/// `1 / (1 + e^-x)`, in the form that cannot overflow for either sign.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + e^x)` as `max(x, 0) + ln(1 + e^-|x|)`, equal but overflow-free.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn gelu(approximation: GeluApproximation, x: f32) -> f32 {
    match approximation {
        GeluApproximation::None => 0.5 * x * (1.0 + libm::erff(x * FRAC_1_SQRT_2)),
        GeluApproximation::Tanh => {
            let sqrt_2_over_pi = FRAC_2_SQRT_PI * FRAC_1_SQRT_2;
            0.5 * x * (1.0 + (sqrt_2_over_pi * 0.044_715_f32.mul_add(x * x * x, x)).tanh())
        }
    }
}

/// `Clip(x, min, max)` as `min(max(x, min), max)`, so `max` wins when the
/// bounds cross; NaN passes through, as NumPy's `clip`.
pub(super) fn clip(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [x, low, high] = operands else { return Err(Malformed::Operands.into()) };
    let clipped = match (widen(x.data()), widen(low.data()), widen(high.data())) {
        (Elements::Float(x), Elements::Float(low), Elements::Float(high)) => {
            let (low, high) = scalar_pair(&low, &high)?;
            Elements::Float(
                x.into_iter()
                    .map(|v| {
                        let raised = if v < low { low } else { v };
                        if raised > high { high } else { raised }
                    })
                    .collect(),
            )
        }
        (Elements::Int(x), Elements::Int(low), Elements::Int(high)) => {
            let (low, high) = scalar_pair(&low, &high)?;
            Elements::Int(x.into_iter().map(|v| v.max(low).min(high)).collect())
        }
        _ => return Err(Malformed::DType(x.dtype()).into()),
    };
    store(clipped, output.dtype())
}

fn scalar_pair<T: Copy>(low: &[T], high: &[T]) -> Result<(T, T), InterpError> {
    match (low, high) {
        ([low], [high]) => Ok((*low, *high)),
        _ => Err(Malformed::Operands.into()),
    }
}

/// `PRelu(x, slope)`: `x` where it is not negative, else `slope * x`.
pub(super) fn prelu(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [x, slope] = operands else { return Err(Malformed::Operands.into()) };
    let out = Extents::of(output.shape())?;
    let (xe, se) = (Extents::of(x.shape())?, Extents::of(slope.shape())?);
    let product = match (widen(x.data()), widen(slope.data())) {
        (Elements::Float(xs), Elements::Float(ss)) => {
            Elements::Float(broadcast_map(&[(&xs, &xe), (&ss, &se)], &out, |a| {
                if a[0] < 0.0 { a[1] * a[0] } else { a[0] }
            }))
        }
        (Elements::Int(xs), Elements::Int(ss)) => {
            Elements::Int(broadcast_map(&[(&xs, &xe), (&ss, &se)], &out, |a| {
                if a[0] < 0 { a[1].wrapping_mul(a[0]) } else { a[0] }
            }))
        }
        _ => return Err(Malformed::DType(x.dtype()).into()),
    };
    store(product, output.dtype())
}
