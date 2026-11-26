//! `RewriteRule`: a named pattern, template and guard, held as data.

use super::pattern::{Bindings, Capture, Pattern};
use super::template::Template;
use ptxc_core::TensorData;
use ptxc_core::float::{Bf16, F16};
use ptxc_graph::{Graph, Op, Use, ValueDef, ValueId};

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
