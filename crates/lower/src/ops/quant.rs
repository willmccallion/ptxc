//! `QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul` and `QLinearConv`,
//! as ONNX defines them (D19). Per-axis scales and zero points are tables
//! the stages read (D46).

use super::NodeContext;
use super::conv::{Products, convolved, geometry};
use super::matmul::product_reads;
use crate::builder::{GroupBuilder, Operand, StageBuilder};
use crate::compose::{add, div, f32, max, min, mul, sub, unary};
use crate::error::LowerError;
use crate::precision::convert;
use crate::stage::{Binding, LoweredGroup, TableId};
use ptxc_core::{DType, Granularity, HostTensor, QuantParams};
use ptxc_graph::{DequantizeAttrs, QConvAttrs, QMatMulAttrs, QuantizeAttrs, ValueId};
use ptxc_loop::{ExprId, ReduceOp, Scalar, TensorExpr, UnaryOp};

/// Where a grid's scale and zero point come from: one map, inlined as
/// constants, or a map per index along an axis, read from tables.
enum Maps {
    Single { scale: f32, zero_point: i64 },
    Tables { scale: TableId, zero_point: TableId, axis: usize },
}

/// One element's map: its f32 scale and `i64` zero point.
struct Map {
    scale: TensorExpr,
    zero_point: TensorExpr,
}

impl Maps {
    /// `params`' maps, as tables in `group` if they vary along an axis.
    fn of(group: &mut GroupBuilder<'_>, params: &QuantParams) -> Self {
        match params.granularity() {
            Granularity::PerTensor(map) => {
                Self::Single { scale: map.scale.get(), zero_point: i64::from(map.zero_point.get()) }
            }
            Granularity::PerAxis { axis, maps } => {
                let scales: Vec<f32> = maps.iter().map(|map| map.scale.get()).collect();
                let zero_points: Vec<i64> =
                    maps.iter().map(|map| i64::from(map.zero_point.get())).collect();
                Self::Tables {
                    scale: group.table(HostTensor::vector(scales.into())),
                    zero_point: group.table(HostTensor::vector(zero_points.into())),
                    axis: axis.index(),
                }
            }
        }
    }

    /// The map of the element whose index along the varying axis is
    /// `along(axis)`; a single map needs none.
    fn at(
        &self,
        stage: &mut StageBuilder<'_>,
        along: impl FnOnce(usize) -> Option<ExprId>,
    ) -> Option<Map> {
        Some(match *self {
            Self::Single { scale, zero_point } => {
                Map { scale: f32(scale), zero_point: TensorExpr::Const(Scalar::i64(zero_point)) }
            }
            Self::Tables { scale, zero_point, axis } => {
                let index = along(axis)?;
                Map {
                    scale: stage.load(Binding::Table(scale), vec![index]).expr,
                    zero_point: stage.load(Binding::Table(zero_point), vec![index]).expr,
                }
            }
        })
    }
}

/// `rounded + zero_point`, an integral f32 plus an `i64`, saturated to the
/// 8-bit grid type `dtype`. NaN becomes 0, as the reference's truncation of
/// NaN does.
fn on_grid(rounded: TensorExpr, zero_point: TensorExpr, dtype: DType) -> Option<TensorExpr> {
    let (lowest, highest) = match dtype {
        DType::I8 => (i64::from(i8::MIN), i64::from(i8::MAX)),
        DType::U8 => (0, i64::from(u8::MAX)),
        _ => return None,
    };
    // NOTE: the reference adds the zero point in f64; f32 is exact here,
    // as any sum past 2^24 saturates an 8-bit grid either way.
    let shifted = add(rounded, convert(zero_point, DType::I64, DType::F32));
    let integral = convert(shifted, DType::F32, DType::I64);
    let saturated = min(
        max(integral, TensorExpr::Const(Scalar::i64(lowest))),
        TensorExpr::Const(Scalar::i64(highest)),
    );
    Some(convert(saturated, DType::I64, dtype))
}

fn round(value: TensorExpr) -> TensorExpr {
    unary(UnaryOp::Round, value)
}

/// `QuantizeLinear`: `saturate(round(x / scale) + zero_point)`, the
/// quotient in f32 rounded half to even; an FP8 grid casts `x / scale`.
pub(super) fn quantize(
    cx: &NodeContext<'_>,
    attrs: &QuantizeAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let dtype = attrs.params().dtype().dtype();
    let mut group = cx.group();
    let maps = Maps::of(&mut group, attrs.params());
    let mut stage = group.stage(cx.name(), cx.graph.value(output).ty().shape());
    let x = stage.broadcast(input);
    let spatial = stage.spatial().to_vec();
    let map = maps
        .at(&mut stage, |axis| spatial.get(axis).copied())
        .ok_or(LowerError::Operands(cx.id))?;
    let quotient = div(convert(x.expr, x.dtype, DType::F32), map.scale);
    let result = if dtype.is_float() {
        TensorExpr::Cast { to: dtype, overflow: attrs.overflow(), value: Box::new(quotient) }
    } else {
        on_grid(round(quotient), map.zero_point, dtype).ok_or_else(|| cx.unsupported(dtype))?
    };
    stage.output(Binding::Value(output), result);
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(group.finish())
}

/// `DequantizeLinear`: `(x - zero_point) * scale`, the subtraction exact
/// and the product in f32; an FP8 grid has no zero point.
pub(super) fn dequantize(
    cx: &NodeContext<'_>,
    attrs: &DequantizeAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let mut group = cx.group();
    let maps = Maps::of(&mut group, &attrs.params);
    let mut stage = group.stage(cx.name(), cx.graph.value(output).ty().shape());
    let x = stage.broadcast(input);
    let spatial = stage.spatial().to_vec();
    let map = maps
        .at(&mut stage, |axis| spatial.get(axis).copied())
        .ok_or(LowerError::Operands(cx.id))?;
    let real = if x.dtype.is_float() {
        convert(x.expr, x.dtype, DType::F32)
    } else {
        convert(centered(&x, map.zero_point), DType::I64, DType::F32)
    };
    let product = mul(real, map.scale);
    stage.output(Binding::Value(output), convert(product, DType::F32, attrs.output));
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(group.finish())
}

/// `x - zero_point`, exactly, in `i64`.
fn centered(x: &Operand, zero_point: TensorExpr) -> TensorExpr {
    sub(convert(x.expr.clone(), x.dtype, DType::I64), zero_point)
}

/// An exact integer sum rescaled onto the output grid: the multiplier
/// `a_scale * b_scale / y_scale` in f32, then rounded and shifted.
fn requantized(sum: TensorExpr, [a, b]: [&Map; 2], y: &Map, dtype: DType) -> Option<TensorExpr> {
    // NOTE: the reference multiplies the sum by the multiplier in f64; ptxc
    // uses f32, as GPU epilogues do, so a rounding tie can land one step
    // apart once the sum passes 2^24.
    let multiplier = div(mul(a.scale.clone(), b.scale.clone()), y.scale.clone());
    let scaled = mul(convert(sum, DType::I64, DType::F32), multiplier);
    on_grid(round(scaled), y.zero_point.clone(), dtype)
}

/// The integer grids a quantized product's operands must be on; FP8
/// operands have no lowering yet.
fn integer_operands(cx: &NodeContext<'_>, values: &[ValueId]) -> Result<(), LowerError> {
    for &value in values {
        let dtype = cx.graph.value(value).ty().dtype();
        if dtype.is_float() {
            return Err(cx.unsupported(dtype));
        }
    }
    Ok(())
}

/// `QLinearMatMul`: the exact product of zero-centred operands, `a`'s map
/// varying by row and `b`'s by column, requantized onto the output grid.
pub(super) fn qmatmul(
    cx: &NodeContext<'_>,
    attrs: &QMatMulAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [lhs, rhs] = cx.values()?;
    integer_operands(cx, &[lhs, rhs])?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let dtype = cx.graph.value(output).ty().dtype();
    let mut group = cx.group();
    let (a_maps, b_maps, y_maps) = (
        Maps::of(&mut group, &attrs.a),
        Maps::of(&mut group, &attrs.b),
        Maps::of(&mut group, &attrs.output),
    );
    let mut stage = group.stage(cx.name(), cx.graph.value(output).ty().shape());
    let product = product_reads(cx, &mut stage, lhs, rhs)?;
    let missing = || LowerError::Operands(cx.id);
    let a = a_maps.at(&mut stage, |_| product.row).ok_or_else(missing)?;
    let b = b_maps.at(&mut stage, |_| product.column).ok_or_else(missing)?;
    let y = y_maps.at(&mut stage, |_| None).ok_or_else(missing)?;
    let terms = mul(
        centered(&product.lhs, a.zero_point.clone()),
        centered(&product.rhs, b.zero_point.clone()),
    );
    let sum =
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![product.inner], body: Box::new(terms) };
    let result = requantized(sum, [&a, &b], &y, dtype).ok_or_else(|| cx.unsupported(dtype))?;
    stage.output(Binding::Value(output), result);
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(group.finish())
}

/// `QLinearConv`: the exact convolution of zero-centred operands plus the
/// `i32` bias, requantized per output channel.
pub(super) fn qconv(cx: &NodeContext<'_>, attrs: &QConvAttrs) -> Result<LoweredGroup, LowerError> {
    let (input, weights, bias) = match cx.operands()[..] {
        [input, weights] => (input, weights, None),
        [input, weights, bias] => (input, weights, Some(bias)),
        _ => return Err(LowerError::Operands(cx.id)),
    };
    integer_operands(cx, &[input, weights])?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let dtype = cx.graph.value(output).ty().dtype();
    let geometry = geometry(cx, &attrs.conv, [input, weights, output])?;
    let mut group = cx.group();
    let (x_maps, w_maps, y_maps) = (
        Maps::of(&mut group, &attrs.input),
        Maps::of(&mut group, &attrs.weights),
        Maps::of(&mut group, &attrs.output),
    );
    let mut stage = group.stage(cx.name(), cx.graph.value(output).ty().shape());
    let spatial = stage.spatial().to_vec();
    let [sample, feature, outs @ ..] = &spatial[..] else {
        return Err(LowerError::Operands(cx.id));
    };
    let missing = || LowerError::Operands(cx.id);
    let x_map = x_maps.at(&mut stage, |_| None).ok_or_else(missing)?;
    let w_map = w_maps.at(&mut stage, |_| Some(*feature)).ok_or_else(missing)?;
    let y_map = y_maps.at(&mut stage, |_| None).ok_or_else(missing)?;
    let product = |value: Operand, weight: Operand| {
        mul(centered(&value, x_map.zero_point.clone()), centered(&weight, w_map.zero_point.clone()))
    };
    let zero = TensorExpr::Const(Scalar::i64(0));
    let products = Products { values: [input, weights], product: &product, zero };
    let sum = convolved(cx, &mut stage, &products, &geometry, (*sample, *feature, outs))?;
    let biased = match bias {
        Some(bias) => {
            let shift = stage.load(Binding::Value(bias), vec![*feature]);
            add(sum, convert(shift.expr, shift.dtype, DType::I64))
        }
        None => sum,
    };
    let result = requantized(biased, [&x_map, &w_map], &y_map, dtype)
        .ok_or_else(|| cx.unsupported(dtype))?;
    stage.output(Binding::Value(output), result);
    let stage = cx.finish(stage)?;
    group.push(stage);
    Ok(group.finish())
}
