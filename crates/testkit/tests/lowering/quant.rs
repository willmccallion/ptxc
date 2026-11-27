//! Quantization onto every grid type, per tensor and per axis, and the
//! quantized products. A product of integer operands sums exactly and
//! compares within one step, its rescaling f32 where the reference's is
//! f64. One of FP8 operands sums in f32, so it is held to that sum's error
//! bound (D81).

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::float::{F8E4M3Fn, F8E5M2, Overflow};
use ptxc_core::{
    Affine, Axis, DType, Dim, Evaluate, Granularity, HostTensor, QuantDType, QuantParams, Scale,
    Shape, TensorData, ZeroPoint,
};
use ptxc_graph::{Op, QMatMulAttrs, QMatMulOutput};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{SumBound, Tolerance, compare_bounded};
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::quant::{dequantize_cases, qconv_cases, qmatmul_cases, quantize_cases};
use ptxc_testkit::families::values::absolute;
use ptxc_testkit::lowered::run_lowered;

fn check(
    (op, inputs): (Op, Vec<HostTensor>),
    tolerance: fn(DType) -> Tolerance,
) -> Result<(), TestCaseError> {
    let graph = single_node(op, &inputs);
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

/// A quantized product: within its sum's error bound if its operands are
/// FP8, within one step if they are integers.
fn check_product((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    match &op {
        Op::QMatMul(attrs) if attrs.a.dtype().dtype().is_float() => {
            check_fp8_product(attrs, &inputs).map_err(TestCaseError::fail)
        }
        _ => check((op, inputs), |_| Tolerance::QUANTIZED),
    }
}

/// Checks the lowering of an FP8 product of `[rows, inner]` by
/// `[inner, columns]` operands against the reference's within its sum's
/// error bound. The lowering sums `inner` exact products in f32, rounding
/// at most `inner - 1` times as its first addition is to 0, and scales the
/// sum twice (D72): `inner + 1` roundings on each term's way. The
/// reference's estimate and magnitudes, each rounded once into f32, make
/// `inner + 3`, and cover the reference's own f64 roundings too.
fn check_fp8_product(attrs: &QMatMulAttrs, inputs: &[HostTensor]) -> Result<(), String> {
    let estimate = in_f32(attrs, inputs);
    let magnitude = in_f32(attrs, &inputs.iter().map(absolute).collect::<Vec<_>>());
    let inner = inputs[0].shape().dims()[1].get();
    let roundings = u32::try_from(inner + 3).unwrap();
    let graph = single_node(Op::QMatMul(attrs.clone()), inputs);
    let text = ptxc_graph::text::print(&graph);
    let actual = run_lowered(graph, inputs, ptxc_loop::run).unwrap();
    let bound = SumBound { estimate: &estimate, magnitude: &magnitude, roundings };
    compare_bounded(&actual[0], bound)
        .map_err(|mismatch| format!("{mismatch}\n{text}\ninputs: {inputs:?}"))
}

/// The reference's product of `inputs` under `attrs`, given in f32.
fn in_f32(attrs: &QMatMulAttrs, inputs: &[HostTensor]) -> HostTensor {
    let attrs = QMatMulAttrs { output: QMatMulOutput::Float(DType::F32), ..attrs.clone() };
    let graph = single_node(Op::QMatMul(attrs), inputs);
    Interpreter::new(&graph).evaluate(inputs).unwrap().remove(0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn quantize_agrees_with_the_reference(case in quantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn dequantize_agrees_with_the_reference(case in dequantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn qmatmul_agrees_with_the_reference(case in qmatmul_cases()) {
        check_product(case)?;
    }

    #[test]
    fn qconv_agrees_with_the_reference(case in qconv_cases()) {
        check(case, |_| Tolerance::QUANTIZED)?;
    }
}

fn matrix(dims: [u64; 2], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(Shape::new(dims.map(Dim::new).to_vec()).unwrap(), data.into()).unwrap()
}

/// Proptest's shrunk case of #218. Row 1 sums `4.5 × -3.5`, `448 × 57344`
/// and `448 × -57344`: f32 has a spacing of 2 at 25690112, so the first
/// term rounds to -16 and the result lands 1.6% from the exact one, far
/// past one step of f16 but within the sum's bound.
#[test]
fn an_fp8_product_whose_terms_cancel_lies_within_its_sum_s_bound() {
    let e4m3 =
        |values: [f64; 6]| values.map(|v| F8E4M3Fn::from_f64(v, Overflow::Saturate)).to_vec();
    let e5m2 = |values: [f64; 6]| values.map(|v| F8E5M2::from_f64(v, Overflow::Saturate)).to_vec();
    let lhs = matrix([2, 3], e4m3([0.0, 0.0, 0.0, 4.5, 448.0, 448.0]));
    let rhs = matrix([3, 2], e5m2([-3.5, 0.0, 57344.0, 0.0, -57344.0, 0.0]));
    let map = Affine { scale: Scale::new(0.05), zero_point: ZeroPoint::new(0) };
    let per_row = Granularity::PerAxis { axis: Axis::new(0), maps: vec![map; 2] };
    let attrs = QMatMulAttrs {
        a: QuantParams::new(QuantDType::F8E4M3Fn, per_row).unwrap(),
        b: QuantParams::new(QuantDType::F8E5M2, Granularity::PerTensor(map)).unwrap(),
        output: QMatMulOutput::Float(DType::F16),
    };

    check_fp8_product(&attrs, &[lhs, rhs]).unwrap();
}

/// Integer results within one step, float ones within their type's
/// tolerance: folding moves a result's rounding onto the integer sum.
const fn folded_tolerance(dtype: DType) -> Tolerance {
    if dtype.is_float() { Tolerance::for_dtype(dtype) } else { Tolerance::QUANTIZED }
}

#[test]
fn qdq_chains_fold_and_lower_to_what_the_unfolded_graph_gives() {
    let product = "\
graph(%a: u8[2, 4], %b: i8[4, 3], %bt: i8[3, 4], %c: f32[3]) {
  %0 = dequantize<params=[u8, [0.05, 128]], output=f32> %a : f32[2, 4]
  %1 = dequantize<params=[i8, 1, [[0.02, 0], [0.03, 0], [0.01, 0]]], output=f32> %b : f32[4, 3]
  %2 = matmul %0, %1 : f32[2, 3]
  %3 = quantize<params=[u8, [0.1, 120]]> %2 : u8[2, 3]
  %4 = dequantize<params=[i8, 0, [[0.02, 0], [0.03, 0], [0.01, 0]]], output=f32> %bt : f32[3, 4]
  %5 = gemm<a=as_is, b=transposed, alpha=1.0, beta=1.0> %0, %4, %c : f32[2, 3]
  return %3, %5
}
";
    let conv = "\
graph(%x: u8[1, 2, 3, 3], %w: i8[2, 2, 2, 2], %b: i32[2]) {
  %0 = dequantize<params=[u8, [0.0625, 128]], output=f32> %x : f32[1, 2, 3, 3]
  %1 = dequantize<params=[i8, 0, [[0.03125, 0], [0.015625, 0]]], output=f32> %w : f32[2, 2, 2, 2]
  %2 = dequantize<params=[i32, 0, [[0.001953125, 0], [0.0009765625, 0]]], output=f32> %b : f32[2]
  %3 = conv<groups=1, axes=[[1, 1, 0, 0], [1, 1, 0, 0]]> %0, %1, %2 : f32[1, 2, 2, 2]
  %4 = quantize<params=[u8, [0.05, 100]]> %3 : u8[1, 2, 2, 2]
  return %4
}
";
    let tensor = |dims: &[u64], data: TensorData| {
        HostTensor::new(Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap(), data)
            .unwrap()
    };
    let activations = vec![130_u8, 200, 50, 128, 255, 0, 100, 140];
    let weights = vec![12_i8, -7, 33, -50, 18, 4, 91, -2, -64, 7, 25, -11];
    let image =
        vec![130_u8, 200, 50, 128, 255, 0, 100, 140, 90, 60, 170, 128, 30, 220, 110, 145, 75, 190];
    let filters = vec![3_i8, -5, 7, 2, -4, 6, 1, -8, 5, 2, -3, 9, -6, 4, 8, -1];
    let cases = [
        (
            product,
            vec![
                tensor(&[2, 4], activations.into()),
                tensor(&[4, 3], weights.clone().into()),
                tensor(&[3, 4], weights.into()),
                tensor(&[3], vec![0.5_f32, -1.0, 0.25].into()),
            ],
        ),
        (
            conv,
            vec![
                tensor(&[1, 2, 3, 3], image.into()),
                tensor(&[2, 2, 2, 2], filters.into()),
                tensor(&[2], vec![40_i32, -25].into()),
            ],
        ),
    ];

    for (text, inputs) in cases {
        let graph = ptxc_graph::text::parse(text, &[]).unwrap();
        check_within(graph, &inputs, folded_tolerance).unwrap();
    }
}
