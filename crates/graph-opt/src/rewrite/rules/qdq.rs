//! Folds dequantize, compute, quantize chains into quantized compute ops.

use crate::rewrite::pattern::{Capture, Pattern};
use crate::rewrite::rule::{Guard, RewriteRule};
use crate::rewrite::template::{OpTemplate, ProductOperand, QuantizedTemplate, Template};
use ptxc_core::Axis;
use ptxc_graph::{BinaryOp, Op, Orientation, TransposeAttrs};

const A: Capture = Capture("a");
const B: Capture = Capture("b");
const C: Capture = Capture("c");
const DEQUANTIZED_A: Capture = Capture("dequantized_a");
const DEQUANTIZED_B: Capture = Capture("dequantized_b");
const PRODUCT: Capture = Capture("product");
const X: Capture = Capture("x");
const WEIGHTS: Capture = Capture("w");
const BIAS: Capture = Capture("bias");
const DEQUANTIZED_X: Capture = Capture("dequantized_x");
const DEQUANTIZED_WEIGHTS: Capture = Capture("dequantized_w");
const DEQUANTIZED_BIAS: Capture = Capture("dequantized_bias");
const CONV: Capture = Capture("conv");
const OUTPUT: Capture = Capture("y");

/// The QDQ folding rules, as TensorRT's explicit quantization folds the
/// quantization nodes of a QDQ model into the ops they surround:
/// - a matmul of two dequantizations is a `QMatMul` with a float result,
///   and so is an unscaled `Gemm` of two, an operand it transposes read
///   through a `Transpose` and any `C` added after;
/// - a quantization of a float-result `QMatMul` only it reads is that
///   product requantized, so a whole dequantize, matmul, quantize chain
///   folds in two steps;
/// - a quantization of a convolution only it reads, of dequantized input
///   and weights and optionally a bias dequantized onto the grid of their
///   integer sum, is a `QConv`.
///
/// A rule whose op the type rule refuses does not apply: a requantized
/// product of FP8 operands, or a grid varying along an axis the quantized
/// op cannot rescale per index.
pub(crate) fn qdq() -> Vec<RewriteRule> {
    let mut rules = vec![dequantized_matmul()];
    for (orientations, names) in [
        ([Orientation::AsIs; 2], ["dequantized-gemm", "dequantized-gemm-plus-c"]),
        (
            [Orientation::AsIs, Orientation::Transposed],
            ["dequantized-gemm-transposed-b", "dequantized-gemm-transposed-b-plus-c"],
        ),
        (
            [Orientation::Transposed, Orientation::AsIs],
            ["dequantized-gemm-transposed-a", "dequantized-gemm-transposed-a-plus-c"],
        ),
        (
            [Orientation::Transposed; 2],
            ["dequantized-gemm-transposed-both", "dequantized-gemm-transposed-both-plus-c"],
        ),
    ] {
        rules.push(dequantized_gemm(names[0], orientations, Bias::Absent));
        rules.push(dequantized_gemm(names[1], orientations, Bias::Present));
    }
    rules.extend([
        requantized_product(),
        quantized_conv("quantized-biased-conv", Bias::Present),
        quantized_conv("quantized-conv", Bias::Absent),
    ]);
    rules
}

/// Whether a folded op has a bias operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bias {
    Present,
    Absent,
}

/// `matmul(dequantize(a), dequantize(b))` is `QMatMul(a, b)` with the
/// product's float type (D72).
fn dequantized_matmul() -> RewriteRule {
    let lhs = Pattern::Op {
        op: "matmul",
        operands: vec![dequantized(A, DEQUANTIZED_A), dequantized(B, DEQUANTIZED_B)],
        capture: OUTPUT,
    };
    let rhs = float_product([Orientation::AsIs; 2]);
    RewriteRule { name: "dequantized-matmul", lhs, rhs, guard: Guard::All(Vec::new()) }
}

/// The rule `name`: `gemm(dequantize(a), dequantize(b)[, c])`, reading
/// its operands in `orientations` and scaling nothing, is `QMatMul` of
/// `a` and `b`, each transposed first if the `Gemm` transposes it, with
/// `c` added after.
// NOTE: TensorRT's explicit quantization folds `C` into the quantized
// product's epilogue. `QMatMul` takes no bias, so `C` stays a float
// addition, which fusion puts in the product's epilogue in turn.
fn dequantized_gemm(name: &'static str, orientations: [Orientation; 2], bias: Bias) -> RewriteRule {
    let mut operands = vec![dequantized(A, DEQUANTIZED_A), dequantized(B, DEQUANTIZED_B)];
    let product = float_product(orientations);
    let rhs = match bias {
        Bias::Absent => product,
        Bias::Present => {
            operands.push(Pattern::Any(C));
            let add = OpTemplate::Exact(Op::Binary(BinaryOp::Add));
            Template::Node { op: add, operands: vec![product, Template::Captured(C)] }
        }
    };
    let lhs = Pattern::Op { op: "gemm", operands, capture: OUTPUT };
    let [a, b] = orientations;
    RewriteRule { name, lhs, rhs, guard: Guard::UnscaledGemm { gemm: OUTPUT, a, b } }
}

/// `QMatMul` of the quantized `a` and `b`, read in `orientations`, giving
/// the float type of the product it replaces.
fn float_product(orientations: [Orientation; 2]) -> Template {
    let [a, b] = orientations;
    let product = QuantizedTemplate::Product {
        a: ProductOperand { dequantized: DEQUANTIZED_A, orientation: a },
        b: ProductOperand { dequantized: DEQUANTIZED_B, orientation: b },
        result: OUTPUT,
    };
    let operands = [(A, a), (B, b)].map(|(quantized, orientation)| match orientation {
        Orientation::AsIs => Template::Captured(quantized),
        Orientation::Transposed => Template::Node {
            op: OpTemplate::Exact(Op::Transpose(TransposeAttrs {
                permutation: vec![Axis::new(1), Axis::new(0)],
            })),
            operands: vec![Template::Captured(quantized)],
        },
    });
    Template::Node { op: OpTemplate::Quantized(product), operands: operands.into() }
}

/// `quantize(QMatMul(a, b))`, the product's only reader, is `QMatMul(a,
/// b)` requantized onto the quantization's grid.
fn requantized_product() -> RewriteRule {
    let product = Pattern::Op {
        op: "qmatmul",
        operands: vec![Pattern::Any(A), Pattern::Any(B)],
        capture: PRODUCT,
    };
    let lhs = Pattern::Op { op: "quantize", operands: vec![product], capture: OUTPUT };
    let requantized = QuantizedTemplate::Requantized { product: PRODUCT, quantized: OUTPUT };
    let rhs = quantized_node(requantized, &[A, B]);
    RewriteRule { name: "requantized-product", lhs, rhs, guard: Guard::SoleReader(PRODUCT) }
}

/// The rule `name`: `quantize(conv(dequantize(x), dequantize(w)[,
/// dequantize(b)]))`, the convolution's only reader, is `QConv(x, w[, b])`
/// on the quantization's grid, where the bias is on the grid of the
/// integer sum.
fn quantized_conv(name: &'static str, bias: Bias) -> RewriteRule {
    let mut operands =
        vec![dequantized(X, DEQUANTIZED_X), dequantized(WEIGHTS, DEQUANTIZED_WEIGHTS)];
    let mut quantized = vec![X, WEIGHTS];
    // NOTE: TensorRT folds FP8 convolutions too. `QConv` takes FP8 grids,
    // but neither the interpreter nor the lowering computes them yet, so
    // an FP8 chain stays unfolded.
    let mut guards = vec![
        Guard::SoleReader(CONV),
        Guard::IntegerElements(X),
        Guard::IntegerElements(WEIGHTS),
        Guard::IntegerElements(OUTPUT),
    ];
    if bias == Bias::Present {
        operands.push(dequantized(BIAS, DEQUANTIZED_BIAS));
        quantized.push(BIAS);
        guards.push(Guard::BiasOnGrid {
            bias: DEQUANTIZED_BIAS,
            input: DEQUANTIZED_X,
            weights: DEQUANTIZED_WEIGHTS,
        });
    }
    let conv = Pattern::Op { op: "conv", operands, capture: CONV };
    let lhs = Pattern::Op { op: "quantize", operands: vec![conv], capture: OUTPUT };
    let template = QuantizedTemplate::Conv {
        conv: CONV,
        input: DEQUANTIZED_X,
        weights: DEQUANTIZED_WEIGHTS,
        quantized: OUTPUT,
    };
    let rhs = quantized_node(template, &quantized);
    RewriteRule { name, lhs, rhs, guard: Guard::All(guards) }
}

/// A dequantization of `quantized`, its output bound to `capture`.
fn dequantized(quantized: Capture, capture: Capture) -> Pattern {
    Pattern::Op { op: "dequantize", operands: vec![Pattern::Any(quantized)], capture }
}

/// A node of `op` reading the captured `operands`.
fn quantized_node(op: QuantizedTemplate, operands: &[Capture]) -> Template {
    Template::Node {
        op: OpTemplate::Quantized(op),
        operands: operands.iter().copied().map(Template::Captured).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::qdq;
    use crate::rewrite::{GreedyEngine, RewriteEngine};
    use ptxc_graph::text::parse;

    /// The rules that fold the graph `text`, in the order they apply.
    fn applied(text: &str) -> Vec<&'static str> {
        let mut graph = parse(text, &[]).unwrap();
        GreedyEngine.apply(&mut graph, &qdq()).unwrap()
    }

    #[test]
    fn each_chain_is_folded_by_the_rule_named_for_it() {
        let product = "\
graph(%a: u8[2, 4], %b: i8[4, 3]) {
  %0 = dequantize<params=[u8, [0.05, 128]], output=f32> %a : f32[2, 4]
  %1 = dequantize<params=[i8, [0.02, 0]], output=f32> %b : f32[4, 3]
  %2 = matmul %0, %1 : f32[2, 3]
  %3 = quantize<params=[u8, [0.1, 120]]> %2 : u8[2, 3]
  return %3
}
";
        let gemm = "\
graph(%a: u8[2, 4], %b: i8[3, 4], %c: f32[3]) {
  %0 = dequantize<params=[u8, [0.05, 128]], output=f32> %a : f32[2, 4]
  %1 = dequantize<params=[i8, [0.02, 0]], output=f32> %b : f32[3, 4]
  %2 = gemm<a=as_is, b=transposed, alpha=1.0, beta=1.0> %0, %1, %c : f32[2, 3]
  return %2
}
";
        let conv = |bias: &str, operand: &str| {
            format!(
                "\
graph(%x: u8[1, 1, 2, 2], %w: i8[1, 1, 1, 1], %b: i32[1]) {{
  %0 = dequantize<params=[u8, [0.5, 128]], output=f32> %x : f32[1, 1, 2, 2]
  %1 = dequantize<params=[i8, [0.25, 0]], output=f32> %w : f32[1, 1, 1, 1]
{bias}  %3 = conv<groups=1, axes=[[1, 1, 0, 0], [1, 1, 0, 0]]> %0, %1{operand} : f32[1, 1, 2, 2]
  %4 = quantize<params=[u8, [0.5, 128]]> %3 : u8[1, 1, 2, 2]
  return %4
}}
"
            )
        };
        let bias = "  %2 = dequantize<params=[i32, [0.125, 0]], output=f32> %b : f32[1]\n";

        assert_eq!(applied(product), ["dequantized-matmul", "requantized-product"]);
        assert_eq!(applied(gemm), ["dequantized-gemm-transposed-b-plus-c"]);
        assert_eq!(applied(&conv(bias, ", %2")), ["quantized-biased-conv"]);
        assert_eq!(applied(&conv("", "")), ["quantized-conv"]);
    }
}
