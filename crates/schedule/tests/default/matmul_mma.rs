//! Matrix products tiled on tensor cores where the target has an `mma`
//! for their types: the kernel the rule builds, and that it computes
//! exactly what the serial schedule does on inputs whose sums are exact.

use crate::matmul::{Layout, PLAIN, product, tiled_and_serial};
use crate::{launch_dims, scheduled, shape};
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ReduceOp, TensorExpr, TensorName, TileOrder,
    VarName, print_kernel,
};
use ptxc_schedule::default_schedule;
use ptxc_target::{MmaShape, SmVersion, Target, preset};
use ptxc_testkit::golden;
use ptxc_testkit::tensorized::{self, Operands, exact_inputs};
use std::sync::Arc;

fn target() -> Target {
    preset(SmVersion::SM_89).unwrap()
}

fn on_tensor_cores(compute: Arc<ComputeDef>) -> bool {
    default_schedule(compute, &target()).unwrap().lower().multiplies_on_tensor_cores()
}

#[test]
fn an_f16_product_gives_blocks_of_four_warps_a_64_by_64_tile_each() {
    let compute = product(DType::F16, [2, 70, 45, 20], Layout { a_transposed: true, ..PLAIN });
    let kernel = default_schedule(Arc::clone(&compute), &target()).unwrap().lower();

    let (text, launch) = scheduled(compute);

    assert_eq!(launch_dims(launch), ([4, 1, 1], [32, 2, 2]));
    // Four stages of both 64 by 32 f16 slices.
    assert_eq!(kernel.smem().get(), 4 * 2 * 64 * 32 * 2);
    golden::check("matmul_mma.txt", &text).unwrap();
}

#[test]
fn a_product_with_a_grid_of_32_large_tiles_takes_them_and_a_smaller_one_does_not() {
    let large = product(DType::F16, [16, 4, 136, 16], PLAIN);
    let square = product(DType::F16, [1, 1024, 512, 16], PLAIN);
    let small = product(DType::F16, [1, 1024, 384, 32], PLAIN);
    let kernels = [&large, &small]
        .map(|compute| default_schedule(Arc::clone(compute), &target()).unwrap().lower());

    let [(_, large_launch), (_, square_launch), (_, small_launch)] =
        [large.clone(), square, small].map(scheduled);

    assert_eq!(launch_dims(large_launch), ([32, 1, 1], [32, 2, 2]));
    assert_eq!(launch_dims(square_launch), ([32, 1, 1], [32, 2, 2]));
    // The large tile has two stages of its 128 by 32 and 32 by 128 slices,
    // the small one four of its 64 by 32 and 32 by 64 ones.
    assert_eq!(kernels[0].smem().get(), 2 * 2 * 128 * 32 * 2);
    // Both swizzle their tiles' 16-byte units by their row's place among
    // 8: the large tile's `a` rows of 32 halves and `b` rows of 128, and
    // the small tile's of 32 and 64.
    let [large_text, small_text] = kernels.each_ref().map(print_kernel);
    assert!(large_text.contains("%a_tile: f16[8192] shared swizzle<3,4,3>"), "{large_text}");
    assert!(large_text.contains("%b_tile: f16[8192] shared swizzle<3,4,4>"), "{large_text}");
    assert!(small_text.contains("%a_tile: f16[8192] shared swizzle<3,4,3>"), "{small_text}");
    assert!(small_text.contains("%b_tile: f16[8192] shared swizzle<3,4,3>"), "{small_text}");
    assert_eq!(launch_dims(small_launch), ([96, 1, 1], [32, 2, 2]));
    assert_eq!(kernels[1].smem().get(), 4 * 2 * 64 * 32 * 2);
    let [tiled, serial] = tiled_and_serial(&large, &exact_inputs(&large));
    assert_eq!(tiled, serial);
}

/// An 8-bit integer product summed in `i32`, which the `mma` table takes.
fn integer_product(multiplicands: DType) -> Arc<ComputeDef> {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(32) };
    let instruction = target().mma_instruction(shape, multiplicands, DType::I32).unwrap();
    let extents = [20, 12, 40];
    tensorized::product(instruction, extents, Operands::matmul(extents, TileOrder::RowMajor))
}

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` with `a` f16 and `b`
/// bf16, both widened to f32, which no `mma` multiplies.
fn mixed_product() -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("mixed"));
    let lhs = compute.input(TensorName::new("a"), DType::F16, shape(&[16, 16]));
    let rhs = compute.input(TensorName::new("b"), DType::Bf16, shape(&[16, 8]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(16));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(8));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(16));
    let widened = |tensor, index| {
        let value = Box::new(TensorExpr::load(tensor, index));
        Box::new(TensorExpr::Cast { to: DType::F32, overflow: Overflow::Saturate, value })
    };
    let body = TensorExpr::Binary(
        BinaryOp::Mul,
        widened(lhs, vec![row, k]),
        widened(rhs, vec![k, column]),
    );
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_product_goes_to_tensor_cores_only_where_an_mma_takes_its_types() {
    let extents = [1, 20, 12, 40];
    let floats = [DType::F16, DType::Bf16, DType::F8E4M3Fn, DType::F8E5M2, DType::F32]
        .map(|dtype| on_tensor_cores(product(dtype, extents, PLAIN)));
    let integers = [DType::I8, DType::U8].map(|dtype| on_tensor_cores(integer_product(dtype)));

    assert_eq!(floats, [true, true, true, true, false]);
    assert_eq!(integers, [true, true]);
    assert!(!on_tensor_cores(mixed_product()));
}

#[test]
fn an_eight_bit_product_on_tensor_cores_matches_the_serial_schedule_exactly() {
    for compute in [integer_product(DType::I8), integer_product(DType::U8)] {
        let [tiled, serial] = tiled_and_serial(&compute, &exact_inputs(&compute));

        assert_eq!(tiled, serial);
    }
}

proptest! {
    #![proptest_config(crate::cases(32))]

    /// Every layout and overhang of an f16, bf16 or FP8 product gives the
    /// serial schedule's result on inputs whose sums are exact.
    #[test]
    fn a_product_on_tensor_cores_matches_the_serial_schedule_exactly(
        dtype in select(vec![DType::F16, DType::Bf16, DType::F8E4M3Fn]),
        extents in [1_u64..=2, 1_u64..=40, 1_u64..=40, 1_u64..=40],
        (a_transposed, b_transposed, a_broadcast) in any::<(bool, bool, bool)>(),
    ) {
        let compute = product(dtype, extents, Layout { a_transposed, b_transposed, a_broadcast });

        let [tiled, serial] = tiled_and_serial(&compute, &exact_inputs(&compute));

        prop_assert_eq!(tiled, serial);
    }
}
