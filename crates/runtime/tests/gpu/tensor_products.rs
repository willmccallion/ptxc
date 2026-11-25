//! Matrix products under the default schedule on tensor cores, in every
//! type the target's `mma` table takes, at the sweep's kinds of shape made
//! small enough to interpret: square, uneven, skinny and transformer-like.
//! On inputs whose sums are exact they match the loop interpreter bit for
//! bit (D58).

use crate::device::{match_interpreter, match_interpreter_exactly, pipelined, target};
use ptxc_core::float::{F8E4M3Fn, Overflow};
use ptxc_core::{
    Affine, Axis, DType, Dim, Granularity, HostTensor, QuantDType, QuantParams, Scale, Shape,
    ZeroPoint,
};
use ptxc_graph::{Op, QMatMulAttrs, QMatMulOutput};
use ptxc_loop::TileOrder;
use ptxc_schedule::default_schedule;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::lowered::run_lowered;
use ptxc_testkit::tensorized::{Operands, exact_inputs, product};
use std::sync::Arc;

/// Rows, columns and depth: `[64, 50, 18]` copies f16 rows in 4-byte runs
/// and 8-bit ones element by element; the last is large enough for 128 by
/// 128 tiles.
const SHAPES: [[u64; 3]; 6] =
    [[64, 64, 64], [100, 100, 100], [1, 256, 96], [128, 96, 48], [64, 50, 18], [2048, 256, 40]];

#[test]
fn every_mma_type_matches_the_interpreter_exactly_at_the_sweeps_kinds_of_shape() {
    for &instruction in target().mma() {
        for extents in SHAPES {
            for b_order in [TileOrder::RowMajor, TileOrder::Transposed] {
                let compute = product(instruction, extents, Operands::matmul(extents, b_order));
                let schedule = default_schedule(Arc::clone(&compute), &target()).unwrap();
                let kernel = schedule.lower();
                let case = format!("{instruction:?}, {extents:?}, b {b_order:?}");

                assert!(kernel.multiplies_on_tensor_cores(), "{case}");
                let _outputs =
                    match_interpreter_exactly(&kernel, &exact_inputs(&compute), &case).unwrap();
            }
        }
    }
}

#[test]
fn every_mma_type_pipelined_over_two_to_four_stages_matches_the_interpreter_exactly() {
    for &instruction in target().mma() {
        for extents in [[100, 100, 100], [64, 50, 18]] {
            let compute =
                product(instruction, extents, Operands::matmul(extents, TileOrder::RowMajor));
            for stages in 2..=4 {
                let kernel = pipelined(&compute, stages);
                let case = format!("{instruction:?}, {extents:?}, {stages} stages");

                let _outputs =
                    match_interpreter_exactly(&kernel, &exact_inputs(&compute), &case).unwrap();
            }
        }
    }
}

/// E4M3 integers from -4 to 4, which FP8 holds and f32 sums exactly.
fn e4m3_integers(dims: [u64; 2], salt: u64) -> HostTensor {
    let count = dims[0] * dims[1];
    let values: Vec<F8E4M3Fn> = (0..count)
        .map(|n| {
            let value = i64::try_from((3 * n + 7 * salt) % 9).unwrap() - 4;
            F8E4M3Fn::from_i64(value, Overflow::Saturate)
        })
        .collect();
    let shape = Shape::new(dims.map(Dim::new).to_vec()).unwrap();
    HostTensor::new(shape, values.into()).unwrap()
}

/// An FP8 product dequantized to f32 (D72), `b`'s scale varying by column,
/// as a table the epilogue reads.
#[test]
fn an_fp8_qmatmul_with_a_float_result_multiplies_on_tensor_cores() {
    let map = |scale: f32| Affine { scale: Scale::new(scale), zero_point: ZeroPoint::new(0) };
    let columns = (0..96_u16).map(|j| map(0.25 * f32::from(1 + j % 3))).collect();
    let a = QuantParams::new(QuantDType::F8E4M3Fn, Granularity::PerTensor(map(0.5))).unwrap();
    let per_column = Granularity::PerAxis { axis: Axis::new(1), maps: columns };
    let b = QuantParams::new(QuantDType::F8E4M3Fn, per_column).unwrap();
    let attrs = QMatMulAttrs { a, b, output: QMatMulOutput::Float(DType::F32) };
    let inputs = [e4m3_integers([100, 64], 1), e4m3_integers([64, 96], 2)];
    let graph = single_node(Op::QMatMul(attrs), &inputs);
    let context = ptxc_graph::text::print(&graph);
    let mut on_tensor_cores = false;

    let ran = run_lowered(graph, &inputs, |kernel, arguments| {
        on_tensor_cores |= kernel.multiplies_on_tensor_cores();
        match_interpreter(kernel, arguments, &context)
    });

    assert!(ran.is_ok(), "{context}");
    assert!(on_tensor_cores, "{context}");
}
