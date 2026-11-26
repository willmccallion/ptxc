//! Algebraic simplifications.

use crate::rewrite::pattern::{Capture, Pattern};
use crate::rewrite::rule::{Fill, Guard, RewriteRule};
use crate::rewrite::template::{OpTemplate, Template};
use ptxc_graph::{BinaryOp, Op, UnaryOp};

const X: Capture = Capture("x");
const WEIGHTS: Capture = Capture("w");
const BIAS: Capture = Capture("b");
const CONV: Capture = Capture("conv");
const SCALE: Capture = Capture("scale");
const SHIFT: Capture = Capture("shift");
const MEAN: Capture = Capture("mean");
const VARIANCE: Capture = Capture("variance");
const CONSTANT: Capture = Capture("c");
const INNER: Capture = Capture("inner");
const OUTPUT: Capture = Capture("y");

/// The algebraic rules:
/// - a batch normalisation after a convolution only it reads folds into
///   the convolution's weights and bias, as every inference compiler folds
///   it, left for constant folding to compute;
/// - `x * 1` and `x + 0` are `x` where the constant does not broadcast `x`;
/// - a transpose of a transpose is one transpose, and a reshape of a
///   reshape one reshape.
pub(crate) fn algebraic() -> Vec<RewriteRule> {
    let mut rules = vec![
        batch_norm_into_conv("batch-norm-into-biased-conv", ConvBias::Present),
        batch_norm_into_conv("batch-norm-into-conv", ConvBias::Absent),
    ];
    // NOTE: IEEE addition gives `-0 + 0 = +0`; dropping `x + 0` keeps a
    // `-0` input, as XLA's simplifier does. Every comparison treats the
    // two zeros as equal.
    for (op, fill, names) in [
        ("mul", Fill::One, ["times-one", "one-times"]),
        ("add", Fill::Zero, ["plus-zero", "zero-plus"]),
    ] {
        rules.push(by_constant(names[0], op, [Pattern::Any(X), Pattern::Any(CONSTANT)], fill));
        rules.push(by_constant(names[1], op, [Pattern::Any(CONSTANT), Pattern::Any(X)], fill));
    }
    rules.push(merged(
        "transpose-of-transpose",
        "transpose",
        OpTemplate::Composed { inner: INNER, outer: OUTPUT },
    ));
    rules.push(merged("reshape-of-reshape", "reshape", OpTemplate::Of(OUTPUT)));
    rules
}

/// Whether a folded convolution has a bias of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ConvBias {
    Present,
    Absent,
}

/// The rule `name`, folding a batch normalisation into the convolution it
/// reads: with `inv = scale / sqrt(variance + epsilon)`, the weights become
/// `w * inv` along their output channels, and the bias `(b - mean) * inv +
/// shift`, `b` 0 if the convolution has none.
fn batch_norm_into_conv(name: &'static str, bias: ConvBias) -> RewriteRule {
    let mut conv_operands = vec![Pattern::Any(X), Pattern::Any(WEIGHTS)];
    if bias == ConvBias::Present {
        conv_operands.push(Pattern::Any(BIAS));
    }
    let conv = Pattern::Op { op: "conv", operands: conv_operands, capture: CONV };
    let parameters = [SCALE, SHIFT, MEAN, VARIANCE].map(Pattern::Any);
    let lhs = Pattern::Op {
        op: "batch_norm",
        operands: [vec![conv], parameters.to_vec()].concat(),
        capture: OUTPUT,
    };
    let epsilon = Template::Epsilon { of: OUTPUT, like: VARIANCE };
    let deviation = unary(UnaryOp::Sqrt, binary(BinaryOp::Add, captured(VARIANCE), epsilon));
    let inv = binary(BinaryOp::Div, captured(SCALE), deviation);
    let along_channels =
        Template::Node { op: OpTemplate::AlongFirstAxisOf(WEIGHTS), operands: vec![inv.clone()] };
    let weights = binary(BinaryOp::Mul, captured(WEIGHTS), along_channels);
    let centred = match bias {
        ConvBias::Present => binary(BinaryOp::Sub, captured(BIAS), captured(MEAN)),
        ConvBias::Absent => unary(UnaryOp::Neg, captured(MEAN)),
    };
    let shift = binary(BinaryOp::Add, binary(BinaryOp::Mul, centred, inv), captured(SHIFT));
    let rhs =
        Template::Node { op: OpTemplate::Of(CONV), operands: vec![captured(X), weights, shift] };
    let guard = Guard::All(vec![
        Guard::SoleReader(CONV),
        Guard::SameElementType(SCALE, WEIGHTS),
        Guard::SameElementType(MEAN, WEIGHTS),
    ]);
    RewriteRule { name, lhs, rhs, guard }
}

/// The rule `name`: `op` of `operands`, one `x` and one constant, is `x`
/// where the constant is all `fill` and the result has `x`'s type.
fn by_constant(
    name: &'static str,
    op: &'static str,
    operands: [Pattern; 2],
    fill: Fill,
) -> RewriteRule {
    let lhs = Pattern::Op { op, operands: operands.into(), capture: OUTPUT };
    let guard = Guard::All(vec![Guard::Filled(CONSTANT, fill), Guard::SameType(X, OUTPUT)]);
    RewriteRule { name, lhs, rhs: captured(X), guard }
}

/// The rule `name`: an `op` of an `op` of `x` is one `merged` op of `x`.
fn merged(name: &'static str, op: &'static str, merged: OpTemplate) -> RewriteRule {
    let inner = Pattern::Op { op, operands: vec![Pattern::Any(X)], capture: INNER };
    let lhs = Pattern::Op { op, operands: vec![inner], capture: OUTPUT };
    let rhs = Template::Node { op: merged, operands: vec![captured(X)] };
    RewriteRule { name, lhs, rhs, guard: Guard::All(Vec::new()) }
}

const fn captured(capture: Capture) -> Template {
    Template::Captured(capture)
}

fn unary(op: UnaryOp, operand: Template) -> Template {
    Template::Node { op: OpTemplate::Exact(Op::Unary(op)), operands: vec![operand] }
}

fn binary(op: BinaryOp, lhs: Template, rhs: Template) -> Template {
    Template::Node { op: OpTemplate::Exact(Op::Binary(op)), operands: vec![lhs, rhs] }
}
