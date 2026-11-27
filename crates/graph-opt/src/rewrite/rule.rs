//! `RewriteRule`: a named pattern, template and guard, held as data.

use super::pattern::{Bindings, Capture, Pattern};
use super::template::{Template, dequantized_params};
use ptxc_core::float::{Bf16, F16};
use ptxc_core::{Affine, DType, Granularity, QuantParams, Scale, TensorData, ZeroPoint};
use ptxc_graph::{F32Attr, Graph, Op, Orientation, Use, ValueDef, ValueId};

/// A rewrite: where `lhs` matches and `guard` holds, the matched value's
/// readers read `rhs` instead.
///
/// Rules are data rather than closures, so another engine, such as an
/// e-graph's, can apply the same rule sets.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct RewriteRule {
    /// The rule's name, which the engine reports when it applies.
    pub(crate) name: &'static str,
    /// What it matches.
    pub(crate) lhs: Pattern,
    /// What the match is replaced with.
    pub(crate) rhs: Template,
    /// What must hold of the match besides its shape.
    pub(crate) guard: Guard,
}

/// A condition on a match's values, beyond the ops the pattern names.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Guard {
    /// The two values have the same type.
    SameType(Capture, Capture),
    /// The value is a transpose that leaves every axis in place.
    IdentityTranspose(Capture),
    /// The two values have the same element type.
    SameElementType(Capture, Capture),
    /// The value is read by one node, and is no graph output.
    SoleReader(Capture),
    /// The value is a constant whose every element is `Fill`'s.
    Filled(Capture, Fill),
    /// The value's elements are integers.
    IntegerElements(Capture),
    /// The value is a `Gemm` reading `A` and `B` in these orientations
    /// that scales neither its product nor `C`: alpha is 1, and so is beta
    /// if it reads `C`.
    UnscaledGemm {
        /// The `Gemm`'s output.
        gemm: Capture,
        /// How it reads `A`.
        a: Orientation,
        /// How it reads `B`.
        b: Orientation,
    },
    /// The dequantizations giving a convolution's bias, input and weights
    /// put the bias on the grid `QConv` adds it to the integer sum on.
    BiasOnGrid {
        /// The bias's dequantization.
        bias: Capture,
        /// The input's dequantization.
        input: Capture,
        /// The weights' dequantization.
        weights: Capture,
    },
    /// Every guard holds.
    All(Vec<Self>),
}

/// A constant's every element, for `Guard::Filled`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fill {
    /// Zero of either sign.
    Zero,
    /// One.
    One,
}

impl Guard {
    /// Whether the guard holds of `bindings` in `graph`. A capture the
    /// pattern did not bind fails it.
    pub(crate) fn holds(&self, graph: &Graph, bindings: &Bindings) -> bool {
        let value = |capture: &Capture| bindings.get(capture).copied();
        match self {
            Self::SameType(a, b) => match (value(a), value(b)) {
                (Some(a), Some(b)) => graph.value(a).ty() == graph.value(b).ty(),
                _ => false,
            },
            Self::IdentityTranspose(capture) => {
                value(capture).is_some_and(|transposed| is_identity_transpose(graph, transposed))
            }
            Self::SameElementType(a, b) => match (value(a), value(b)) {
                (Some(a), Some(b)) => graph.value(a).ty().dtype() == graph.value(b).ty().dtype(),
                _ => false,
            },
            Self::SoleReader(capture) => {
                value(capture).is_some_and(|read| matches!(graph.uses(read), [Use::Operand { .. }]))
            }
            Self::Filled(capture, fill) => value(capture).is_some_and(|constant| {
                let ValueDef::Constant(id) = graph.value(constant).def() else { return false };
                is_filled(graph.constant(id).data(), *fill)
            }),
            Self::IntegerElements(capture) => value(capture).is_some_and(|integers| {
                matches!(
                    graph.value(integers).ty().dtype(),
                    DType::I8 | DType::U8 | DType::I32 | DType::I64
                )
            }),
            Self::UnscaledGemm { gemm, a, b } => {
                value(gemm).is_some_and(|product| is_unscaled_gemm(graph, product, [*a, *b]))
            }
            Self::BiasOnGrid { bias, input, weights } => {
                let params = |capture: &Capture| dequantized_params(graph, value(capture)?);
                match (params(bias), params(input), params(weights)) {
                    (Some(bias), Some(input), Some(weights)) => {
                        is_bias_on_grid(bias, input, weights)
                    }
                    _ => false,
                }
            }
            Self::All(guards) => guards.iter().all(|guard| guard.holds(graph, bindings)),
        }
    }
}

/// Whether every element of `data` is `fill`. FP8 and bool, on which no
/// arithmetic is defined, never are.
fn is_filled(data: &TensorData, fill: Fill) -> bool {
    let float = |bits: u32, sign: u32, one: u32| match fill {
        Fill::Zero => bits & !sign == 0,
        Fill::One => bits == one,
    };
    let integer = |value: i64| value == i64::from(fill == Fill::One);
    match data {
        TensorData::F32(v) => v.iter().all(|x| float(x.to_bits(), 1 << 31, 1.0_f32.to_bits())),
        TensorData::F16(v) => {
            let one = u32::from(F16::from_f32(1.0).to_bits());
            v.iter().all(|x| float(u32::from(x.to_bits()), 1 << 15, one))
        }
        TensorData::Bf16(v) => {
            let one = u32::from(Bf16::from_f32(1.0).to_bits());
            v.iter().all(|x| float(u32::from(x.to_bits()), 1 << 15, one))
        }
        TensorData::I8(v) => v.iter().all(|&x| integer(i64::from(x))),
        TensorData::U8(v) => v.iter().all(|&x| integer(i64::from(x))),
        TensorData::I32(v) => v.iter().all(|&x| integer(i64::from(x))),
        TensorData::I64(v) => v.iter().all(|&x| integer(x)),
        TensorData::F8E4M3Fn(_) | TensorData::F8E5M2(_) | TensorData::Bool(_) => false,
    }
}

/// Whether `value` is the output of a transpose that keeps every axis.
fn is_identity_transpose(graph: &Graph, value: ValueId) -> bool {
    let node = graph.producer(value).and_then(|node| graph.node(node));
    node.is_some_and(|node| match node.op() {
        Op::Transpose(attrs) => {
            attrs.permutation.iter().enumerate().all(|(position, axis)| axis.index() == position)
        }
        _ => false,
    })
}

/// Whether `value` is the output of a `Gemm` reading its operands in
/// `orientations`, with alpha 1 and, if it reads `C`, beta 1.
fn is_unscaled_gemm(graph: &Graph, value: ValueId, orientations: [Orientation; 2]) -> bool {
    let Some(node) = graph.producer(value).and_then(|node| graph.node(node)) else { return false };
    let Op::Gemm(attrs) = node.op() else { return false };
    let one = F32Attr::new(1.0);
    let reads_c = node.operands().len() == 3;
    [attrs.a, attrs.b] == orientations && attrs.alpha == one && (!reads_c || attrs.beta == one)
}

/// Whether `bias` has zero point 0 and, for each output channel, the scale
/// of `input`'s one map times that channel's weight scale: the grid of the
/// integer sum `QConv` adds it to, as ONNX's `QLinearConv` defines its
/// bias. Its `i32` type is the type rule's to check.
fn is_bias_on_grid(bias: &QuantParams, input: &QuantParams, weights: &QuantParams) -> bool {
    let Granularity::PerTensor(input) = input.granularity() else { return false };
    let (Some(bias_maps), Some(weight_maps)) = (channel_maps(bias), channel_maps(weights)) else {
        return false;
    };
    let channels = bias_maps.len().max(weight_maps.len());
    (0..channels).all(|channel| {
        match (map_of_channel(bias_maps, channel), map_of_channel(weight_maps, channel)) {
            (Some(bias), Some(weights)) => {
                bias.zero_point == ZeroPoint::new(0)
                    && bias.scale == Scale::new(input.scale.get() * weights.scale.get())
            }
            _ => false,
        }
    })
}

/// The maps of a tensor whose first axis is the output channels: one for
/// all of them, or one per channel.
fn channel_maps(params: &QuantParams) -> Option<&[Affine]> {
    match params.granularity() {
        Granularity::PerTensor(map) => Some(std::slice::from_ref(map)),
        Granularity::PerAxis { axis, maps } => (axis.index() == 0).then_some(maps.as_slice()),
    }
}

/// The map of `channel`, from one map for all channels or one per channel.
fn map_of_channel(maps: &[Affine], channel: usize) -> Option<Affine> {
    match maps {
        [map] => Some(*map),
        maps => maps.get(channel).copied(),
    }
}
