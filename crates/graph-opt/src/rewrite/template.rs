//! `Template`: the right-hand side of a rule, resolved against a match.

mod quantized;
mod replacement;

use super::pattern::{Bindings, Capture};
use ptxc_core::float::{Bf16, F16};
use ptxc_core::{Axis, DType, Dim, HostTensor, Shape, TensorData};
use ptxc_graph::{Graph, Op, TensorType, TransposeAttrs};

pub(crate) use quantized::{ProductOperand, QuantizedTemplate, dequantized_params};
pub(crate) use replacement::Replacement;

/// What a match is replaced with, as MLIR's declarative rewrites build
/// their result ops.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Template {
    /// A value the pattern bound.
    Captured(Capture),
    /// A new node of `op` reading `operands`, its one output.
    Node {
        /// The op.
        op: OpTemplate,
        /// One template per operand.
        operands: Vec<Self>,
    },
    /// A scalar of `like`'s element type holding the epsilon of the batch
    /// normalisation that gives `of`.
    Epsilon {
        /// A value a batch normalisation gives.
        of: Capture,
        /// A value of the scalar's type.
        like: Capture,
    },
}

/// The op of a node a template builds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum OpTemplate {
    /// This op.
    Exact(Op),
    /// The op of the node that gives the captured value.
    Of(Capture),
    /// A reshape of a vector of `C` elements to `[C, 1, ..., 1]`, of the
    /// captured value's rank, so it scales that value along its first axis.
    AlongFirstAxisOf(Capture),
    /// The transpose doing `outer`'s transpose after `inner`'s.
    Composed {
        /// The value the first transpose gives.
        inner: Capture,
        /// The value the second gives.
        outer: Capture,
    },
    /// A quantized op on the grids of captured quantization nodes.
    Quantized(QuantizedTemplate),
}

impl Template {
    /// What replaces a match with `bindings` in `graph`; `None`, so the
    /// rule does not apply, if the template reads a capture or an attribute
    /// the match lacks, or builds a node whose type rule refuses it.
    pub(crate) fn resolve(&self, graph: &Graph, bindings: &Bindings) -> Option<Replacement> {
        match self {
            Self::Captured(capture) => bindings.get(capture).copied().map(Replacement::Existing),
            Self::Node { op, operands } => {
                let operands: Vec<Replacement> = operands
                    .iter()
                    .map(|operand| operand.resolve(graph, bindings))
                    .collect::<Option<_>>()?;
                let types: Vec<TensorType> =
                    operands.iter().map(|operand| operand.ty(graph)).collect();
                let op = op.resolve(graph, bindings, &types)?;
                Replacement::node(op, operands, &types)
            }
            Self::Epsilon { of, like } => {
                let Op::BatchNorm(attrs) = producer_op(graph, bindings, of)? else { return None };
                let dtype = graph.value(*bindings.get(like)?).ty().dtype();
                float_scalar(dtype, attrs.epsilon.get()).map(Replacement::Constant)
            }
        }
    }
}

impl OpTemplate {
    /// The op, for a node reading operands of `types`.
    fn resolve(&self, graph: &Graph, bindings: &Bindings, types: &[TensorType]) -> Option<Op> {
        match self {
            Self::Exact(op) => Some(op.clone()),
            Self::Of(capture) => producer_op(graph, bindings, capture).cloned(),
            Self::AlongFirstAxisOf(capture) => {
                let rank = graph.value(*bindings.get(capture)?).ty().shape().dims().len();
                let length = types.first()?.shape().element_count().get();
                let dims =
                    std::iter::once(length).chain(std::iter::repeat_n(1, rank.checked_sub(1)?));
                Some(Op::Reshape(Shape::new(dims.map(Dim::new).collect()).ok()?))
            }
            Self::Composed { inner, outer } => {
                let (Op::Transpose(first), Op::Transpose(second)) =
                    (producer_op(graph, bindings, inner)?, producer_op(graph, bindings, outer)?)
                else {
                    return None;
                };
                let permutation = second
                    .permutation
                    .iter()
                    .map(|axis| first.permutation.get(axis.index()).copied());
                Some(Op::Transpose(TransposeAttrs {
                    permutation: permutation.collect::<Option<Vec<Axis>>>()?,
                }))
            }
            Self::Quantized(quantized) => quantized.resolve(graph, bindings),
        }
    }
}

/// The op of the node that gives the captured value.
fn producer_op<'g>(graph: &'g Graph, bindings: &Bindings, capture: &Capture) -> Option<&'g Op> {
    let node = graph.producer(*bindings.get(capture)?)?;
    Some(graph.node(node)?.op())
}

/// `value` as a scalar of the float type `dtype`.
fn float_scalar(dtype: DType, value: f32) -> Option<HostTensor> {
    let data: TensorData = match dtype {
        DType::F32 => vec![value].into(),
        DType::F16 => vec![F16::from_f32(value)].into(),
        DType::Bf16 => vec![Bf16::from_f32(value)].into(),
        _ => return None,
    };
    HostTensor::new(Shape::scalar(), data).ok()
}
