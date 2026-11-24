//! Per-family evaluation, mirroring `ptxc_graph::ops`.

mod activation;
mod attention;
mod conv;
mod elementwise;
mod gather;
mod matmul;
mod norm;
mod pool;
mod quant;
mod reduce;
mod resize;
mod shape;
mod window;

use crate::error::{InterpError, Malformed};
use crate::numeric::store_wide;
use ptxc_core::float::Overflow;
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{Op, TensorType};

/// The results of `op` on `operands`, given the types its rule inferred.
pub(crate) fn evaluate(
    op: &Op,
    operands: &[&HostTensor],
    outputs: &[&TensorType],
) -> Result<Vec<HostTensor>, InterpError> {
    let [output, ..] = outputs else { return Ok(Vec::new()) };
    let single = |data: Result<TensorData, InterpError>| Ok(vec![result(output, data?)?]);
    match op {
        Op::Unary(op) => single(elementwise::unary(*op, operands, output)),
        Op::Binary(op) => single(elementwise::binary(*op, operands, output)),
        Op::Variadic(op) => single(elementwise::variadic(*op, operands, output)),
        Op::Where => single(elementwise::select(operands, output)),
        Op::Cast(attrs) => single(elementwise::cast(*attrs, operands)),
        Op::Activation(activation) => single(activation::activation(*activation, operands, output)),
        Op::Clip => single(activation::clip(operands, output)),
        Op::PRelu => single(activation::prelu(operands, output)),
        Op::MatMul => single(matmul::matmul(operands, output)),
        Op::Gemm(attrs) => single(matmul::gemm(attrs, operands, output)),
        Op::MatMulInteger(attrs) => single(matmul::matmul_integer(attrs, operands)),
        Op::Conv(attrs) => single(conv::conv(attrs, operands, output)),
        Op::Pool(attrs) => single(pool::pool(attrs, operands, output)),
        Op::GlobalPool(kind) => single(pool::global_pool(*kind, operands, output)),
        Op::Reduce(attrs) => single(reduce::reduce(attrs, operands, output)),
        Op::ArgReduce(attrs) => single(reduce::arg_reduce(attrs, operands, output)),
        Op::CumSum(attrs) => single(reduce::cum_sum(*attrs, operands, output)),
        Op::Softmax(attrs) => single(norm::softmax(attrs, operands, output)),
        Op::LayerNorm(attrs) => norm::layer_norm(attrs, operands, outputs),
        Op::BatchNorm(attrs) => single(norm::batch_norm(*attrs, operands, output)),
        Op::InstanceNorm(attrs) => single(norm::instance_norm(*attrs, operands, output)),
        Op::Reshape(_) => single(Ok(first(operands)?.data().clone())),
        Op::Expand(_) => single(shape::expand(operands, output)),
        Op::Transpose(attrs) => single(shape::transpose(attrs, operands, output)),
        Op::Tile(_) => single(shape::tile(operands, output)),
        Op::Concat(axis) => single(shape::concat(*axis, operands)),
        Op::Slice(attrs) => single(shape::slice(attrs, operands, output)),
        Op::Split(attrs) => shape::split(attrs, operands, outputs),
        Op::Pad(attrs) => single(shape::pad(attrs, operands, output)),
        Op::Trilu(attrs) => single(shape::trilu(attrs, operands)),
        Op::Gather(axis) => single(gather::gather(*axis, operands, output)),
        Op::GatherElements(axis) => single(gather::gather_elements(*axis, operands, output)),
        Op::GatherNd(batch_dims) => single(gather::gather_nd(*batch_dims, operands)),
        Op::Quantize(attrs) => single(quant::quantize(attrs, operands)),
        Op::Dequantize(attrs) => single(quant::dequantize(attrs, operands)),
        Op::QMatMul(attrs) => single(quant::qmatmul(attrs, operands, output)),
        Op::QConv(attrs) => single(quant::qconv(attrs, operands, output)),
        Op::Resize(attrs) => single(resize::resize(attrs, operands, output)),
        Op::Attention(attrs) => attention::attention(attrs, operands, outputs),
    }
}

/// A result with the shape its type rule inferred.
pub(super) fn result(ty: &TensorType, data: TensorData) -> Result<HostTensor, InterpError> {
    HostTensor::new(ty.shape().clone(), data).map_err(|error| Malformed::Result(error).into())
}

/// The first operand, which every op has.
pub(super) fn first<'t>(operands: &[&'t HostTensor]) -> Result<&'t HostTensor, InterpError> {
    operands.first().copied().ok_or_else(|| Malformed::Operands.into())
}

/// Accumulated floats rounded into the output's float type.
pub(super) fn round_to_output(values: &[f64], ty: &TensorType) -> Result<TensorData, InterpError> {
    store_wide(values, ty.dtype(), Overflow::NonFinite)
}
