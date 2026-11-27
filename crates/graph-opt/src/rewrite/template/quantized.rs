//! The quantized ops QDQ folding builds, from the quantization nodes a
//! pattern captured.

use super::producer_op;
use crate::rewrite::pattern::{Bindings, Capture};
use ptxc_core::{Axis, Granularity, QuantParams};
use ptxc_graph::{Graph, Op, Orientation, QConvAttrs, QMatMulAttrs, QMatMulOutput, ValueId};

/// A quantized op on the grids of the quantization nodes a pattern
/// captured.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum QuantizedTemplate {
    /// `QMatMul` of two dequantized operands, giving the captured float
    /// result's type (D72).
    Product {
        /// The left operand.
        a: ProductOperand,
        /// The right operand.
        b: ProductOperand,
        /// The float product.
        result: Capture,
    },
    /// The captured float-result `QMatMul`, requantized onto the grid of
    /// the captured quantization.
    Requantized {
        /// The float-result `QMatMul`'s output.
        product: Capture,
        /// The quantization's output.
        quantized: Capture,
    },
    /// `QConv` with the captured convolution's geometry, on the grids of
    /// its input's and weights' dequantizations and of the quantization of
    /// its result.
    Conv {
        /// The convolution's output.
        conv: Capture,
        /// The input's dequantization.
        input: Capture,
        /// The weights' dequantization.
        weights: Capture,
        /// The quantization's output.
        quantized: Capture,
    },
}

/// A dequantized matrix a quantized product reads, and whether it reads it
/// transposed, as `Gemm` may.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ProductOperand {
    /// The matrix's dequantization.
    pub(crate) dequantized: Capture,
    /// How the product reads it.
    pub(crate) orientation: Orientation,
}

impl QuantizedTemplate {
    /// The op, if the captured values come from the ops it reads.
    pub(super) fn resolve(&self, graph: &Graph, bindings: &Bindings) -> Option<Op> {
        let value = |capture: &Capture| bindings.get(capture).copied();
        let dequantized = |capture: &Capture| dequantized_params(graph, value(capture)?).cloned();
        let quantized = |capture: &Capture| quantized_params(graph, value(capture)?).cloned();
        let operand = |operand: &ProductOperand| {
            let params = dequantized(&operand.dequantized)?;
            match operand.orientation {
                Orientation::AsIs => Some(params),
                Orientation::Transposed => transposed(&params),
            }
        };
        match self {
            Self::Product { a, b, result } => {
                let dtype = graph.value(*bindings.get(result)?).ty().dtype();
                Some(Op::QMatMul(QMatMulAttrs {
                    a: operand(a)?,
                    b: operand(b)?,
                    output: QMatMulOutput::Float(dtype),
                }))
            }
            Self::Requantized { product, quantized: grid } => {
                let Op::QMatMul(QMatMulAttrs { a, b, output: QMatMulOutput::Float(_) }) =
                    producer_op(graph, bindings, product)?
                else {
                    return None;
                };
                let output = QMatMulOutput::Quantized(quantized(grid)?);
                Some(Op::QMatMul(QMatMulAttrs { a: a.clone(), b: b.clone(), output }))
            }
            Self::Conv { conv, input, weights, quantized: grid } => {
                let Op::Conv(geometry) = producer_op(graph, bindings, conv)? else { return None };
                Some(Op::QConv(QConvAttrs {
                    conv: geometry.clone(),
                    input: dequantized(input)?,
                    weights: dequantized(weights)?,
                    output: quantized(grid)?,
                }))
            }
        }
    }
}

/// The grid of the dequantization that gives `value`.
pub(crate) fn dequantized_params(graph: &Graph, value: ValueId) -> Option<&QuantParams> {
    match graph.node(graph.producer(value)?)?.op() {
        Op::Dequantize(attrs) => Some(&attrs.params),
        _ => None,
    }
}

/// The grid of the quantization that gives `value`.
fn quantized_params(graph: &Graph, value: ValueId) -> Option<&QuantParams> {
    match graph.node(graph.producer(value)?)?.op() {
        Op::Quantize(attrs) => Some(attrs.params()),
        _ => None,
    }
}

/// The grid of a matrix's transpose: maps varying along one axis vary
/// along the other.
fn transposed(params: &QuantParams) -> Option<QuantParams> {
    let Granularity::PerAxis { axis, maps } = params.granularity() else {
        return Some(params.clone());
    };
    let swapped = match axis.index() {
        0 => Axis::new(1),
        1 => Axis::new(0),
        _ => return None,
    };
    let granularity = Granularity::PerAxis { axis: swapped, maps: maps.clone() };
    QuantParams::new(params.dtype(), granularity).ok()
}
