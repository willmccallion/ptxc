//! Quantize, dequantize and quantized-compute ops.

use super::accepts::Accepts;
use super::conv::{ConvAttrs, check_bias};
use super::matmul::matmul_shape;
use super::{
    TypeError, TypeRule, check_axis, exact_operands, operands_with_optional, require_dtype,
};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::float::Overflow;
use ptxc_core::{Axis, DType, Granularity, QuantDType, QuantParams};

/// ONNX `QuantizeLinear`: a float tensor rounded onto `params`' grid. The
/// scales and zero points are constants at import, so they are attributes.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QuantizeAttrs {
    params: QuantParams,
    overflow: Overflow,
}

impl QuantizeAttrs {
    /// Quantization onto `params`. `overflow` is ONNX's `saturate`, which
    /// only affects FP8 targets; for integer targets it is normalised to
    /// `Saturate`, as integer quantization always saturates.
    pub const fn new(params: QuantParams, overflow: Overflow) -> Self {
        let overflow = match params.dtype() {
            QuantDType::F8E4M3Fn | QuantDType::F8E5M2 => overflow,
            QuantDType::I8 | QuantDType::U8 | QuantDType::I32 => Overflow::Saturate,
        };
        Self { params, overflow }
    }

    /// The target grid.
    pub const fn params(&self) -> &QuantParams {
        &self.params
    }

    /// What an out-of-range value becomes in an FP8 target.
    pub const fn overflow(&self) -> Overflow {
        self.overflow
    }
}

impl TypeRule for QuantizeAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Accepts::Float.check(input.dtype())?;
        let dtype = quantized_dtype(&self.params)?;
        check_params(&self.params, input)?;
        Ok(vec![TensorType::new(dtype, input.shape().clone(), Layout::RowMajor)])
    }
}

/// ONNX `DequantizeLinear`: a quantized tensor mapped back to `output`, a
/// float type. ONNX takes `output` from the scale tensor's type or
/// `output_dtype`; scales here are always f32, so it is recorded.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DequantizeAttrs {
    /// The grid the input is on. `I32` is allowed: QDQ models store biases
    /// that way.
    pub params: QuantParams,
    /// The float element type of the result.
    pub output: DType,
}

impl TypeRule for DequantizeAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        require_dtype(self.params.dtype().dtype(), input)?;
        Accepts::Float.check(self.output)?;
        check_params(&self.params, input)?;
        Ok(vec![TensorType::new(self.output, input.shape().clone(), Layout::RowMajor)])
    }
}

/// A matrix product on quantized operands with a quantized result, as ONNX
/// `QLinearMatMul` and the `Dequantize -> MatMul -> Quantize` chains QDQ
/// folding (M4) turns into it. Shapes follow `MatMul`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QMatMulAttrs {
    /// The left operand's grid.
    pub a: QuantParams,
    /// The right operand's grid.
    pub b: QuantParams,
    /// The result's grid.
    pub output: QuantParams,
}

impl TypeRule for QMatMulAttrs {
    /// As ONNX `QLinearMatMul`: `a`'s maps may vary by row and `b`'s by
    /// column, which the sum over the inner axis leaves constant; the
    /// result has one map.
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [a, b] = exact_operands(operands)?;
        for (operand, params) in [(a, &self.a), (b, &self.b)] {
            check_quantized_operand(operand, params)?;
        }
        let rank = |operand: &TensorType| operand.shape().rank().get();
        check_varies_only_along(&self.a, rank(a).checked_sub(2).map(Axis::new))?;
        check_varies_only_along(&self.b, (rank(b) >= 2).then(|| Axis::new(rank(b) - 1)))?;
        check_varies_only_along(&self.output, None)?;
        let dtype = quantized_dtype(&self.output)?;
        let result = TensorType::new(dtype, matmul_shape(a.shape(), b.shape())?, Layout::RowMajor);
        check_params(&self.output, &result)?;
        Ok(vec![result])
    }
}

/// A convolution on quantized operands with a quantized result, as ONNX
/// `QLinearConv` and the chains QDQ folding turns into it.
///
/// The optional bias is `i32`, on the grid of the input scale times the
/// weight scale with zero point 0. Shapes follow `Conv`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QConvAttrs {
    /// The convolution's geometry.
    pub conv: ConvAttrs,
    /// The input's grid.
    pub input: QuantParams,
    /// The weights' grid; per output channel is usual.
    pub weights: QuantParams,
    /// The result's grid.
    pub output: QuantParams,
}

impl TypeRule for QConvAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let ([input, weights], bias) = operands_with_optional(operands)?;
        for (operand, params) in [(input, &self.input), (weights, &self.weights)] {
            check_quantized_operand(operand, params)?;
        }
        check_varies_only_along(&self.input, None)?;
        check_varies_only_along(&self.weights, Some(Axis::new(0)))?;
        check_varies_only_along(&self.output, None)?;
        let shape = self.conv.output_shape(input.shape(), weights.shape())?;
        if let Some(bias) = bias {
            check_bias(bias, DType::I32, shape.dims()[1])?;
        }
        let result = TensorType::new(quantized_dtype(&self.output)?, shape, Layout::RowMajor);
        check_params(&self.output, &result)?;
        Ok(vec![result])
    }
}

/// The element type of a quantized result, which ONNX never makes `i32`.
const fn quantized_dtype(params: &QuantParams) -> Result<DType, TypeError> {
    match params.dtype() {
        QuantDType::I32 => Err(TypeError::UnsupportedDType(DType::I32)),
        dtype => Ok(dtype.dtype()),
    }
}

/// Checks that `operand` is on `params`' grid, which is not `i32`.
fn check_quantized_operand(operand: &TensorType, params: &QuantParams) -> Result<(), TypeError> {
    require_dtype(quantized_dtype(params)?, operand)?;
    check_params(params, operand)
}

/// Checks that `params` has one map, or maps varying only along `allowed`.
fn check_varies_only_along(params: &QuantParams, allowed: Option<Axis>) -> Result<(), TypeError> {
    match params.granularity() {
        Granularity::PerAxis { axis, .. } if Some(*axis) != allowed => {
            Err(TypeError::QuantAxis(*axis))
        }
        _ => Ok(()),
    }
}

/// Checks that per-axis parameters name an axis of `operand` and have one
/// map per element along it.
fn check_params(params: &QuantParams, operand: &TensorType) -> Result<(), TypeError> {
    let Granularity::PerAxis { axis, maps } = params.granularity() else {
        return Ok(());
    };
    check_axis(operand.shape(), *axis)?;
    let dim = operand.shape().dims()[axis.index()];
    if usize::try_from(dim.get()) == Ok(maps.len()) {
        Ok(())
    } else {
        Err(TypeError::QuantMapCount { axis: *axis, dim, maps: maps.len() })
    }
}
