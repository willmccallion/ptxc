//! Imports `QuantizeLinear`, `DequantizeLinear`, `QLinearMatMul` and
//! `QLinearConv`, with their scales and zero points as attributes (D32).

use super::conv::{check_kernel_shape, read_window};
use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind, Unsupported};
use crate::lowered::{Lowered, Operand};
use crate::tensor::dtype_from_onnx;
use ptxc_core::float::Overflow;
use ptxc_core::{
    Affine, DType, Granularity, HostTensor, QuantDType, QuantParams, Scale, TensorData, ZeroPoint,
};
use ptxc_graph::{
    ConvAttrs, DequantizeAttrs, Groups, Op, QConvAttrs, QMatMulAttrs, QMatMulOutput, QuantizeAttrs,
};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "QuantizeLinear" => quantize(context),
        "DequantizeLinear" => dequantize(context),
        "QLinearMatMul" => qmatmul(context),
        "QLinearConv" => qconv(context),
        _ => return None,
    })
}

fn invalid(problem: impl Into<String>) -> ImportError {
    ImportErrorKind::Invalid { field: "quantization", problem: problem.into() }.into()
}

fn quant_dtype(dtype: DType) -> Result<QuantDType, ImportError> {
    Ok(match dtype {
        DType::I8 => QuantDType::I8,
        DType::U8 => QuantDType::U8,
        DType::I32 => QuantDType::I32,
        DType::F8E4M3Fn => QuantDType::F8E4M3Fn,
        DType::F8E5M2 => QuantDType::F8E5M2,
        other => return Err(Unsupported::DType(format!("{other:?} as a quantized type")).into()),
    })
}

/// Rejects ONNX's blocked quantization (`block_size`, opset 21), which ptxc
/// does not support.
fn no_blocks(context: &mut NodeContext<'_>) -> Result<(), ImportError> {
    if context.attrs.int_or("block_size", 0)? == 0 {
        Ok(())
    } else {
        Err(context.attrs.unsupported("block_size"))
    }
}

/// Where per-axis parameters apply: an ONNX axis, possibly negative, of the
/// node's input `data`.
#[derive(Clone, Copy)]
struct PerAxis {
    axis: i64,
    data: usize,
}

/// Parameters from a scale input and an optional zero-point input: a scalar
/// scale, or a 1-D one with one element, is per tensor; a longer 1-D one is
/// per element of `per_axis`'s axis, which is only resolved then (ONNX's
/// default axis need not exist for a scalar).
fn params(
    context: &NodeContext<'_>,
    dtype: QuantDType,
    scale_input: usize,
    zero_input: usize,
    per_axis: Option<PerAxis>,
) -> Result<QuantParams, ImportError> {
    let scales = scales(context.constant(scale_input)?)?;
    let zeros = if context.has_input(zero_input) {
        zero_points(context.constant(zero_input)?)?
    } else {
        vec![0; scales.len()]
    };
    if zeros.len() != scales.len() {
        return Err(invalid("scale and zero point differ in length"));
    }
    let maps: Vec<Affine> = scales
        .iter()
        .zip(&zeros)
        .map(|(&scale, &zero)| Affine {
            scale: Scale::new(scale),
            zero_point: ZeroPoint::new(zero),
        })
        .collect();
    let scale_rank = context.ty(scale_input)?.shape().rank();
    let granularity = match (scale_rank.get(), per_axis) {
        (0, _) => Granularity::PerTensor(maps[0]),
        (1, _) if maps.len() == 1 => Granularity::PerTensor(maps[0]),
        (1, Some(PerAxis { axis, data })) => {
            let axis = context.axis(axis, context.ty(data)?.shape().rank())?;
            Granularity::PerAxis { axis, maps }
        }
        _ => {
            return Err(Unsupported::Attribute {
                op: context.op_type().to_owned(),
                attribute: "blocked scales".to_owned(),
            }
            .into());
        }
    };
    QuantParams::new(dtype, granularity).map_err(|error| invalid(error.to_string()))
}

/// Scales as `f32`; f16 and bf16 scales widen exactly.
fn scales(tensor: &HostTensor) -> Result<Vec<f32>, ImportError> {
    let values: Vec<f32> = match tensor.data() {
        TensorData::F32(values) => values.clone(),
        TensorData::F16(values) => values.iter().map(|value| value.to_f32()).collect(),
        TensorData::Bf16(values) => values.iter().map(|value| value.to_f32()).collect(),
        _ => return Err(Unsupported::DType(format!("{:?} scales", tensor.dtype())).into()),
    };
    if values.is_empty() { Err(invalid("no scales")) } else { Ok(values) }
}

/// Zero points as `i32`. FP8 zero points must be zero (see `QuantDType`).
fn zero_points(tensor: &HostTensor) -> Result<Vec<i32>, ImportError> {
    Ok(match tensor.data() {
        TensorData::I8(values) => values.iter().map(|&value| i32::from(value)).collect(),
        TensorData::U8(values) => values.iter().map(|&value| i32::from(value)).collect(),
        TensorData::I32(values) => values.clone(),
        TensorData::F8E4M3Fn(values) => values
            .iter()
            .map(|value| if value.to_f32() == 0.0 { Ok(0) } else { Err(()) })
            .collect::<Result<_, _>>()
            .map_err(|()| unsupported_fp8_zero())?,
        TensorData::F8E5M2(values) => values
            .iter()
            .map(|value| if value.to_f32() == 0.0 { Ok(0) } else { Err(()) })
            .collect::<Result<_, _>>()
            .map_err(|()| unsupported_fp8_zero())?,
        _ => return Err(Unsupported::DType(format!("{:?} zero points", tensor.dtype())).into()),
    })
}

fn unsupported_fp8_zero() -> ImportError {
    Unsupported::Attribute {
        op: "QuantizeLinear".to_owned(),
        attribute: "nonzero FP8 zero point".to_owned(),
    }
    .into()
}

/// The output's quantized type: its zero point's, or ONNX's default `u8`.
fn output_zero_dtype(context: &NodeContext<'_>) -> Result<DType, ImportError> {
    if context.has_input(7) { Ok(context.ty(7)?.dtype()) } else { Ok(DType::U8) }
}

/// `QuantizeLinear` and `DequantizeLinear`'s `axis` (default 1) of input 0.
fn axis_attr(context: &mut NodeContext<'_>) -> Result<PerAxis, ImportError> {
    Ok(PerAxis { axis: context.attrs.int_or("axis", 1)?, data: 0 })
}

/// `QuantizeLinear`. The quantized type is the zero point's, else
/// `output_dtype`'s, else ONNX's default `u8`. `precision` other than the
/// default (the scale's type) is not supported.
fn quantize(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    no_blocks(context)?;
    if context.attrs.int_or("precision", 0)? != 0 {
        return Err(context.attrs.unsupported("precision"));
    }
    let axis = axis_attr(context)?;
    let overflow = if context.attrs.int_or("saturate", 1)? == 0 {
        Overflow::NonFinite
    } else {
        Overflow::Saturate
    };
    let output_dtype = context.attrs.int_or("output_dtype", 0)?;
    let dtype = if context.has_input(2) {
        context.ty(2)?.dtype()
    } else if output_dtype != 0 {
        dtype_from_onnx(i32::try_from(output_dtype).map_err(|_| invalid("output_dtype"))?)?
    } else {
        DType::U8
    };
    let params = params(context, quant_dtype(dtype)?, 1, 2, Some(axis))?;
    Ok(Lowered::on_inputs(Op::Quantize(QuantizeAttrs::new(params, overflow)), 1))
}

/// `DequantizeLinear`: to `output_dtype` if set, else the scale's type.
fn dequantize(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    no_blocks(context)?;
    let axis = axis_attr(context)?;
    let output_dtype = context.attrs.int_or("output_dtype", 0)?;
    let output = if output_dtype == 0 {
        context.ty(1)?.dtype()
    } else {
        dtype_from_onnx(i32::try_from(output_dtype).map_err(|_| invalid("output_dtype"))?)?
    };
    let params = params(context, quant_dtype(context.ty(0)?.dtype())?, 1, 2, Some(axis))?;
    Ok(Lowered::on_inputs(Op::Dequantize(DequantizeAttrs { params, output }), 1))
}

/// `QLinearMatMul(a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero)`:
/// `a`'s parameters may be per row and `b`'s per column.
fn qmatmul(context: &NodeContext<'_>) -> Result<Lowered, ImportError> {
    let a_rank = context.ty(0)?.shape().rank();
    let b_rank = context.ty(3)?.shape().rank();
    let row = (a_rank.get() >= 2).then_some(PerAxis { axis: -2, data: 0 });
    let column = (b_rank.get() >= 1).then_some(PerAxis { axis: -1, data: 3 });
    let attrs = QMatMulAttrs {
        a: params(context, quant_dtype(context.ty(0)?.dtype())?, 1, 2, row)?,
        b: params(context, quant_dtype(context.ty(3)?.dtype())?, 4, 5, column)?,
        output: QMatMulOutput::Quantized(params(
            context,
            quant_dtype(output_zero_dtype(context)?)?,
            6,
            7,
            None,
        )?),
    };
    Ok(Lowered::Node {
        op: Op::QMatMul(attrs),
        operands: vec![Operand::Input(0), Operand::Input(3)],
    })
}

/// `QLinearConv(x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, B?)`:
/// the weights' parameters may be per output channel.
fn qconv(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let input = context.ty(0)?.shape().dims().to_vec();
    let kernel = context.ty(3)?.shape().dims().get(2..).unwrap_or_default().to_vec();
    check_kernel_shape(context, &kernel)?;
    let axes = read_window(context, input.get(2..).unwrap_or_default(), &kernel)?;
    let groups = u64::try_from(context.attrs.int_or("group", 1)?)
        .ok()
        .and_then(Groups::new)
        .ok_or_else(|| invalid("group must be positive"))?;
    let attrs = QConvAttrs {
        conv: ConvAttrs { axes, groups },
        input: params(context, quant_dtype(context.ty(0)?.dtype())?, 1, 2, None)?,
        weights: params(
            context,
            quant_dtype(context.ty(3)?.dtype())?,
            4,
            5,
            Some(PerAxis { axis: 0, data: 3 }),
        )?,
        output: params(context, quant_dtype(output_zero_dtype(context)?)?, 6, 7, None)?,
    };
    let mut operands = vec![Operand::Input(0), Operand::Input(3)];
    if context.has_input(8) {
        operands.push(Operand::Input(8));
    }
    Ok(Lowered::Node { op: Op::QConv(attrs), operands })
}
