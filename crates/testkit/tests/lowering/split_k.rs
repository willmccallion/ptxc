//! Skinny products, whose few output tiles leave the GPU idle, split along
//! their depth into a partial stage and a combine stage (D75), and match
//! the reference exactly on inputs whose sums are exact.

use ptxc_core::{DType, Dim, Evaluate, HostTensor, Shape, ZeroPoint};
use ptxc_graph::{F32Attr, GemmAttrs, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::lowered::run_lowered;
use ptxc_testkit::tensorized::exact_tensor;

fn matrix(dtype: DType, dims: [u64; 2], salt: u64) -> HostTensor {
    exact_tensor(dtype, Shape::new(dims.map(Dim::new).to_vec()).unwrap(), salt)
}

/// What running `op` on `inputs` lowered did: whether each stage it ran
/// multiplied on tensor cores, after checking its outputs equal the
/// reference's exactly.
fn run_exactly(op: Op, inputs: &[HostTensor]) -> Vec<bool> {
    let graph = single_node(op, inputs);
    let expected = Interpreter::new(&graph).evaluate(inputs).unwrap();
    let mut stages = Vec::new();

    let actual = run_lowered(graph, inputs, |kernel, arguments| {
        stages.push(kernel.multiplies_on_tensor_cores());
        ptxc_loop::run(kernel, arguments)
    })
    .unwrap();

    compare(&actual[0], &expected[0], Tolerance::Exact).unwrap();
    stages
}

#[test]
fn a_skinny_f16_matmul_sums_slices_of_its_depth_on_tensor_cores_then_the_slices() {
    let inputs = [matrix(DType::F16, [4, 3072], 1), matrix(DType::F16, [3072, 64], 2)];

    let stages = run_exactly(Op::MatMul, &inputs);

    assert_eq!(stages, [true, false]);
}

#[test]
fn a_skinny_f32_gemm_keeps_its_transpose_and_epilogue_through_the_split() {
    let inputs = [
        matrix(DType::F32, [1024, 4], 3),
        matrix(DType::F32, [64, 1024], 4),
        matrix(DType::F32, [1, 64], 5),
    ];
    let attrs = GemmAttrs {
        a: Orientation::Transposed,
        b: Orientation::Transposed,
        alpha: F32Attr::new(0.5),
        beta: F32Attr::new(2.0),
    };

    let stages = run_exactly(Op::Gemm(attrs), &inputs);

    assert_eq!(stages, [false, false]);
}

#[test]
fn a_skinny_integer_matmul_splits_exactly() {
    let inputs = [matrix(DType::I8, [4, 1024], 6), matrix(DType::I8, [1024, 64], 7)];
    let zero = ZeroPoints::PerTensor(ZeroPoint::new(0));
    let attrs = MatMulIntegerAttrs { a: zero.clone(), b: zero };

    let stages = run_exactly(Op::MatMulInteger(attrs), &inputs);

    assert_eq!(stages, [true, false]);
}

#[test]
fn a_product_too_shallow_to_split_stays_one_stage() {
    // Two slices would be 64 deep, shallower than a slice may be.
    let inputs = [matrix(DType::F16, [4, 128], 8), matrix(DType::F16, [128, 64], 9)];

    let stages = run_exactly(Op::MatMul, &inputs);

    assert_eq!(stages, [true]);
}
