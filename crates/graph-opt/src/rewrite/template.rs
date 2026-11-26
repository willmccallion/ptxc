//! `Template`: the right-hand side of a rule, instantiated from a match.

use super::pattern::{Bindings, Capture};
use crate::pass::PassError;
use ptxc_core::float::{Bf16, F16};
use ptxc_core::{Axis, DType, Dim, HostTensor, Shape, TensorData};
use ptxc_graph::{Graph, Op, Rewriter, TransposeAttrs, ValueId};

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
}

impl Template {
    /// The value replacing a match with `bindings`, building any nodes and
    /// constants through `rewriter`; `None` if it reads a capture or an
    /// attribute the match lacks.
    pub(crate) fn instantiate(
        &self,
        rewriter: &mut Rewriter<'_>,
        bindings: &Bindings,
    ) -> Result<Option<ValueId>, PassError> {
        match self {
            Self::Captured(capture) => Ok(bindings.get(capture).copied()),
            Self::Node { op, operands } => {
                let mut values = Vec::with_capacity(operands.len());
                for operand in operands {
                    let Some(value) = operand.instantiate(rewriter, bindings)? else {
                        return Ok(None);
                    };
                    values.push(value);
                }
                let Some(op) = op.resolve(rewriter.graph(), bindings, &values) else {
                    return Ok(None);
                };
                let outputs =
                    rewriter.add_node(op, values).map_err(ptxc_graph::RewriteError::from)?;
                Ok(outputs.first().copied())
            }
            Self::Epsilon { of, like } => {
                let graph = rewriter.graph();
                let epsilon = bindings.get(of).and_then(|&value| batch_norm_epsilon(graph, value));
                let dtype = bindings.get(like).map(|&value| graph.value(value).ty().dtype());
                let scalar =
                    epsilon.zip(dtype).and_then(|(epsilon, dtype)| float_scalar(dtype, epsilon));
                Ok(scalar.map(|tensor| rewriter.add_constant(tensor)))
            }
        }
    }
}

impl OpTemplate {
    /// The op, for a node reading `operands`.
    fn resolve(&self, graph: &Graph, bindings: &Bindings, operands: &[ValueId]) -> Option<Op> {
        let node_op = |capture: &Capture| {
            let node = graph.producer(*bindings.get(capture)?)?;
            Some(graph.node(node)?.op().clone())
        };
        match self {
            Self::Exact(op) => Some(op.clone()),
            Self::Of(capture) => node_op(capture),
            Self::AlongFirstAxisOf(capture) => {
                let rank = graph.value(*bindings.get(capture)?).ty().shape().dims().len();
                let length = graph.value(*operands.first()?).ty().shape().element_count().get();
                let dims =
                    std::iter::once(length).chain(std::iter::repeat_n(1, rank.checked_sub(1)?));
                Some(Op::Reshape(Shape::new(dims.map(Dim::new).collect()).ok()?))
            }
            Self::Composed { inner, outer } => {
                let (Op::Transpose(first), Op::Transpose(second)) =
                    (node_op(inner)?, node_op(outer)?)
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
        }
    }
}

/// The epsilon of the batch normalisation that gives `value`.
fn batch_norm_epsilon(graph: &Graph, value: ValueId) -> Option<f32> {
    match graph.node(graph.producer(value)?)?.op() {
        Op::BatchNorm(attrs) => Some(attrs.epsilon.get()),
        _ => None,
    }
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
