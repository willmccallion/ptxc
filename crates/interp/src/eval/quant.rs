//! Quantization exactly as ONNX defines it (D19): `QuantizeLinear`,
//! `DequantizeLinear`, `QLinearMatMul` and `QLinearConv`.

use super::conv::Convolution;
use super::matmul::Batched;
use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::numeric::{
    Elements, floats, int_to_f32, int_to_f64, saturate_ints, store_floats, truncate, widen,
};
use ptxc_core::float::Overflow;
use ptxc_core::{Affine, DType, Granularity, HostTensor, QuantParams, TensorData};
use ptxc_graph::{DequantizeAttrs, QConvAttrs, QMatMulAttrs, QuantizeAttrs, TensorType};

/// The affine map of the element at `index`.
fn map_at(params: &QuantParams, index: &[usize]) -> Affine {
    match params.granularity() {
        Granularity::PerTensor(map) => *map,
        Granularity::PerAxis { axis, maps } => maps[index[axis.index()]],
    }
}

/// The single map of a per-tensor grid.
fn per_tensor(params: &QuantParams) -> Result<Affine, InterpError> {
    match params.granularity() {
        Granularity::PerTensor(map) => Ok(*map),
        Granularity::PerAxis { .. } => Err(Malformed::Granularity.into()),
    }
}

/// The map of each element of a tensor of `extents`, in row-major order.
fn maps(params: &QuantParams, extents: &Extents) -> Vec<Affine> {
    extents.indices().map(|index| map_at(params, &index)).collect()
}

const fn is_fp8(dtype: DType) -> bool {
    matches!(dtype, DType::F8E4M3Fn | DType::F8E5M2)
}

/// Integer results rounded half to even and saturated to the grid's type.
fn store_grid(values: &[f64], dtype: DType) -> Result<TensorData, InterpError> {
    saturate_ints(&values.iter().map(|&v| truncate(v.round_ties_even())).collect::<Vec<_>>(), dtype)
}

/// `QuantizeLinear`: `saturate(round(x / scale) + zero_point)`, the quotient
/// in f32 and rounded half to even. FP8 targets cast `x / scale`, as
/// `saturate` says.
pub(super) fn quantize(
    attrs: &QuantizeAttrs,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let values = floats(input.data())?;
    let maps = maps(attrs.params(), &Extents::of(input.shape())?);
    let dtype = attrs.params().dtype().dtype();
    let quotients: Vec<f32> =
        values.iter().zip(&maps).map(|(&x, map)| x / map.scale.get()).collect();
    if is_fp8(dtype) {
        return store_floats(&quotients, dtype, attrs.overflow());
    }
    let shifted: Vec<f64> = quotients
        .iter()
        .zip(&maps)
        .map(|(&q, map)| f64::from(q.round_ties_even()) + f64::from(map.zero_point.get()))
        .collect();
    store_grid(&shifted, dtype)
}

/// `DequantizeLinear`: `(x - zero_point) * scale`, the subtraction exact
/// and the product in f32.
pub(super) fn dequantize(
    attrs: &DequantizeAttrs,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let maps = maps(&attrs.params, &Extents::of(input.shape())?);
    let products: Vec<f32> = match widen(input.data()) {
        Elements::Int(values) => values
            .iter()
            .zip(&maps)
            .map(|(&x, map)| int_to_f32(x - i64::from(map.zero_point.get())) * map.scale.get())
            .collect(),
        Elements::Float(values) => {
            values.iter().zip(&maps).map(|(&x, map)| x * map.scale.get()).collect()
        }
        Elements::Bool(_) => return Err(Malformed::DType(input.dtype()).into()),
    };
    store_floats(&products, attrs.output, Overflow::NonFinite)
}

/// A quantized operand's elements minus their zero points, exactly.
fn centered(tensor: &HostTensor, params: &QuantParams) -> Result<Vec<i64>, InterpError> {
    let maps = maps(params, &Extents::of(tensor.shape())?);
    match widen(tensor.data()) {
        Elements::Int(values) => Ok(values
            .iter()
            .zip(&maps)
            .map(|(&x, map)| x - i64::from(map.zero_point.get()))
            .collect()),
        _ => Err(Malformed::DType(tensor.dtype()).into()),
    }
}

/// An exact integer sum rescaled onto the output grid, as ONNX's reference
/// computes it: the multiplier `a_scale * b_scale / y_scale` in f32, the
/// product in f64, then the zero point added after rounding.
fn requantize(sum: i64, a: Affine, b: Affine, y: Affine) -> f64 {
    let multiplier = a.scale.get() * b.scale.get() / y.scale.get();
    (int_to_f64(sum) * f64::from(multiplier)).round_ties_even() + f64::from(y.zero_point.get())
}

/// The map for each of `extent` indices along the one axis a grid may vary
/// along, which the op's type rule fixes.
fn maps_along(params: &QuantParams, extent: usize) -> Vec<Affine> {
    match params.granularity() {
        Granularity::PerTensor(map) => vec![*map; extent],
        Granularity::PerAxis { maps, .. } => maps.clone(),
    }
}

/// `QLinearMatMul`; `a`'s maps may vary by row and `b`'s by column.
pub(super) fn qmatmul(
    attrs: &QMatMulAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [lhs, rhs] = operands else { return Err(Malformed::Operands.into()) };
    let (lhs_extents, rhs_extents) = (Extents::of(lhs.shape())?, Extents::of(rhs.shape())?);
    let batched = Batched::of(&lhs_extents, &rhs_extents);
    let (row_count, column_count) = (batched.rows, batched.columns);
    let rows = maps_along(&attrs.a, row_count);
    let columns = maps_along(&attrs.b, column_count);
    let output_map = per_tensor(&attrs.output)?;
    let (left, right) = (centered(lhs, &attrs.a)?, centered(rhs, &attrs.b)?);
    let sums = batched.products(&left, &right, 0, |x, w, acc| x * w + acc);
    let scaled: Vec<f64> = sums
        .iter()
        .enumerate()
        .map(|(flat, &sum)| {
            let (row, column) = (flat / column_count % row_count, flat % column_count);
            requantize(sum, rows[row], columns[column], output_map)
        })
        .collect();
    store_grid(&scaled, output.dtype())
}

/// `QLinearConv`: the integer convolution plus the i32 bias, rescaled per
/// output channel.
pub(super) fn qconv(
    attrs: &QConvAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (x, w, bias) = match operands {
        [x, w] => (x, w, None),
        [x, w, b] => (x, w, Some(widen(b.data()))),
        _ => return Err(Malformed::Operands.into()),
    };
    let bias = match bias {
        Some(Elements::Int(values)) => Some(values),
        Some(_) => return Err(Malformed::DType(DType::I32).into()),
        None => None,
    };
    let convolution = Convolution::of(&attrs.conv.axes, attrs.conv.groups.get(), x, w, output)?;
    let sums = convolution.convolve(
        &centered(x, &attrs.input)?,
        &centered(w, &attrs.weights)?,
        0,
        |a, b, acc| a * b + acc,
    );
    let (channels, channel_count) =
        (convolution.output().strides()[1], convolution.output().dims()[1]);
    let (input_map, y) = (per_tensor(&attrs.input)?, per_tensor(&attrs.output)?);
    let weight_maps = maps_along(&attrs.weights, channel_count);
    let scaled: Vec<f64> = sums
        .iter()
        .enumerate()
        .map(|(flat, &sum)| {
            let channel = flat / channels % channel_count;
            let biased = sum + bias.as_ref().map_or(0, |b| b[channel]);
            requantize(biased, input_map, weight_maps[channel], y)
        })
        .collect();
    store_grid(&scaled, output.dtype())
}
