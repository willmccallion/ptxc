//! Matrix products large enough to span several blocks, threads' tiles and
//! reduction steps of the SIMT tiling, each stage on the GPU matching the
//! loop interpreter exactly.

use crate::device::match_interpreter;
use ptxc_core::float::F16;
use ptxc_core::{Dim, HostTensor, Shape, TensorData};
use ptxc_graph::{F32Attr, GemmAttrs, Op, Orientation};
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::lowered::run_lowered;

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// Values in [-1, 1] that no two neighbouring elements share, offset by
/// `seed` so operands differ.
fn values(count: u64, seed: u16) -> Vec<f32> {
    (0..u16::try_from(count).unwrap())
        .map(|n| f32::from(n).mul_add(0.731, f32::from(seed)).sin())
        .collect()
}

fn floats(dims: &[u64], seed: u16) -> HostTensor {
    let count = dims.iter().product();
    HostTensor::new(shape(dims), values(count, seed).into()).unwrap()
}

fn halves(dims: &[u64], seed: u16) -> HostTensor {
    let count = dims.iter().product();
    let data: Vec<F16> = values(count, seed).into_iter().map(F16::from_f32).collect();
    HostTensor::new(shape(dims), data.into()).unwrap()
}

/// Integers in [-7, 7], whose products and sums stay exact.
fn integers(dims: &[u64], seed: u64) -> HostTensor {
    let count: u64 = dims.iter().product();
    let data: Vec<i32> =
        (0..count).map(|n| i32::try_from((n * 37 + seed) % 15).unwrap() - 7).collect();
    HostTensor::new(shape(dims), TensorData::I32(data)).unwrap()
}

/// Runs `op` on `inputs` stage by stage on the GPU, each matching the
/// interpreter.
fn matches_interpreter(op: Op, inputs: &[HostTensor]) {
    let graph = single_node(op, inputs);
    let context = ptxc_graph::text::print(&graph);
    let _outputs = run_lowered(graph, inputs, |kernel, arguments| {
        match_interpreter(kernel, arguments, &context)
    })
    .unwrap();
}

#[test]
fn a_batched_f32_matmul_matches_bit_for_bit() {
    matches_interpreter(Op::MatMul, &[floats(&[2, 70, 21], 1), floats(&[2, 21, 75], 2)]);
}

#[test]
fn an_f16_matmul_broadcasting_its_left_operand_matches_bit_for_bit() {
    matches_interpreter(Op::MatMul, &[halves(&[70, 21], 3), halves(&[3, 21, 45], 4)]);
}

#[test]
fn a_transposed_scaled_gemm_with_a_bias_matches_bit_for_bit() {
    let attrs = GemmAttrs {
        a: Orientation::Transposed,
        b: Orientation::Transposed,
        alpha: F32Attr::new(0.5),
        beta: F32Attr::new(-2.0),
    };
    let inputs = [floats(&[37, 130], 5), floats(&[90, 37], 6), floats(&[90], 7)];

    matches_interpreter(Op::Gemm(attrs), &inputs);
}

#[test]
fn an_integer_matmul_matches_exactly() {
    matches_interpreter(Op::MatMul, &[integers(&[66, 17], 8), integers(&[17, 70], 9)]);
}

/// A dot product whose terms cancel below f32's precision: fused, the
/// second term's product `1 - 2^-46` keeps its low bits and the sum is
/// `-2^-46`; a separate multiply would round it to 1 and give 0.
#[test]
fn a_summed_product_rounds_once_on_the_gpu_as_in_the_interpreter() {
    let epsilon = f32::EPSILON;
    let x = HostTensor::new(shape(&[1, 2]), vec![1.0, 1.0 + epsilon].into()).unwrap();
    let y = HostTensor::new(shape(&[2, 1]), vec![-1.0, 1.0 - epsilon].into()).unwrap();
    let graph = single_node(Op::MatMul, &[x.clone(), y.clone()]);

    let outputs = run_lowered(graph, &[x, y], |kernel, arguments| {
        match_interpreter(kernel, arguments, "a cancelling dot product")
    })
    .unwrap();

    let TensorData::F32(sum) = outputs[0].data() else { panic!("{:?}", outputs[0].dtype()) };
    assert_eq!(sum, &[-(epsilon * epsilon)]);
}
