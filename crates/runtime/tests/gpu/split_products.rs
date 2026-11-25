//! Skinny products split along their depth (D75): each stage on the GPU
//! matches the loop interpreter, and the compiled program, with its
//! partials in the arena, matches the reference, exactly on inputs whose
//! sums are exact.

use crate::device::match_interpreter_exactly;
use crate::programs::run_program;
use ptxc_core::{DType, Dim, Evaluate, HostTensor, Shape, ZeroPoint};
use ptxc_graph::{F32Attr, GemmAttrs, Graph, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::lowered::run_lowered;
use ptxc_testkit::tensorized::exact_tensor;

fn matrix(dtype: DType, dims: [u64; 2], salt: u64) -> HostTensor {
    exact_tensor(dtype, Shape::new(dims.map(Dim::new).to_vec()).unwrap(), salt)
}

/// A skinny product of each kind the default schedule splits: f16 on
/// tensor cores, f32 on CUDA cores with both operands transposed and an
/// epilogue reading `C`, and int8 on integer tensor cores.
fn skinny_products() -> Vec<(Graph, Vec<HostTensor>)> {
    let f16 = vec![matrix(DType::F16, [4, 3072], 1), matrix(DType::F16, [3072, 64], 2)];
    let f32 = vec![
        matrix(DType::F32, [1024, 1], 3),
        matrix(DType::F32, [200, 1024], 4),
        matrix(DType::F32, [1, 200], 5),
    ];
    let gemm = GemmAttrs {
        a: Orientation::Transposed,
        b: Orientation::Transposed,
        alpha: F32Attr::new(0.5),
        beta: F32Attr::new(2.0),
    };
    let i8 = vec![matrix(DType::I8, [16, 1024], 6), matrix(DType::I8, [1024, 96], 7)];
    let zero = ZeroPoints::PerTensor(ZeroPoint::new(0));
    let integer = MatMulIntegerAttrs { a: zero.clone(), b: zero };
    [(Op::MatMul, f16), (Op::Gemm(gemm), f32), (Op::MatMulInteger(integer), i8)]
        .into_iter()
        .map(|(op, inputs)| (single_node(op, &inputs), inputs))
        .collect()
}

#[test]
fn each_stage_of_a_split_product_matches_the_interpreter_exactly() {
    for (graph, inputs) in skinny_products() {
        let context = ptxc_graph::text::print(&graph);
        let mut stages = 0;

        let ran = run_lowered(graph, &inputs, |kernel, arguments| {
            stages += 1;
            match_interpreter_exactly(kernel, arguments, &context)
        });

        assert!(ran.is_ok(), "{context}");
        assert_eq!(stages, 2, "{context}");
    }
}

#[test]
fn a_split_product_compiles_to_a_program_that_matches_the_reference_exactly() {
    for (graph, inputs) in skinny_products() {
        let context = ptxc_graph::text::print(&graph);
        let expected = Interpreter::new(&graph).evaluate(&inputs).unwrap();

        let actual = run_program(graph, &inputs);

        compare(&actual[0], &expected[0], Tolerance::Exact).unwrap_or_else(|mismatch| {
            panic!("{mismatch}\n{context}");
        });
    }
}
