//! Matrix products tiled on CUDA cores, f32 having no `mma`: the kernel
//! the rule builds, and that it computes exactly what the serial schedule
//! does.

use crate::{launch_dims, scheduled, shape};
use proptest::prelude::*;
use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim, HostTensor};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, ReduceOp, Schedule, TensorExpr,
    TensorName, VarName, run,
};
use ptxc_schedule::default_schedule;
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::golden;
use std::sync::Arc;

/// How a product's operands are stored and broadcast.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    /// `a` is stored `[k, m]` rather than `[m, k]`.
    pub(crate) a_transposed: bool,
    /// `b` is stored `[n, k]` rather than `[k, n]`.
    pub(crate) b_transposed: bool,
    /// `a` has no batch axis, and is read at the same matrix for every
    /// batch.
    pub(crate) a_broadcast: bool,
}

/// `out[batch, i, j] = sum over k of a[.., i, k] * b[batch, k, j]` on
/// `dtype` operands widened to f32, laid out as `layout` says.
pub(crate) fn product(
    dtype: DType,
    [batch, rows, columns, depth]: [u64; 4],
    layout: Layout,
) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let oriented = |transposed: bool, [rows, columns]: [u64; 2]| {
        if transposed { [columns, rows] } else { [rows, columns] }
    };
    let a_matrix = oriented(layout.a_transposed, [rows, depth]);
    let b_matrix = oriented(layout.b_transposed, [depth, columns]);
    let a_dims =
        if layout.a_broadcast { a_matrix.to_vec() } else { [&[batch], &a_matrix[..]].concat() };
    let b_dims = [&[batch], &b_matrix[..]].concat();
    let lhs_input = compute.input(TensorName::new("a"), dtype, shape(&a_dims));
    let rhs_input = compute.input(TensorName::new("b"), dtype, shape(&b_dims));
    let (_, batch_axis) = compute.spatial(VarName::new("batch"), Dim::new(batch));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let index = |batched: bool, transposed: bool, [row, column]: [ExprId; 2]| -> Vec<ExprId> {
        let matrix = if transposed { [column, row] } else { [row, column] };
        if batched { [&[batch_axis], &matrix[..]].concat() } else { matrix.to_vec() }
    };
    let widened = |tensor, index| TensorExpr::Cast {
        to: DType::F32,
        overflow: Overflow::Saturate,
        value: Box::new(TensorExpr::Load { tensor, index }),
    };
    let lhs = widened(lhs_input, index(!layout.a_broadcast, layout.a_transposed, [row, k]));
    let rhs = widened(rhs_input, index(true, layout.b_transposed, [k, column]));
    let body = TensorExpr::Binary(BinaryOp::Mul, Box::new(lhs), Box::new(rhs));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

pub(crate) const PLAIN: Layout =
    Layout { a_transposed: false, b_transposed: false, a_broadcast: false };

/// The inputs of `compute`, an f32 product, from `values` in turn.
fn inputs(compute: &ComputeDef, values: &[f32]) -> Vec<HostTensor> {
    let mut remaining = values.iter().copied().cycle();
    compute
        .inputs()
        .iter()
        .map(|input| {
            let count = usize::try_from(input.shape.element_count().get()).unwrap();
            let data: Vec<f32> = remaining.by_ref().take(count).collect();
            HostTensor::new(input.shape.clone(), data.into()).unwrap()
        })
        .collect()
}

/// The bits of `compute`'s outputs under the default schedule and the
/// serial one.
pub(crate) fn tiled_and_serial(
    compute: &Arc<ComputeDef>,
    inputs: &[HostTensor],
) -> [Vec<Vec<u8>>; 2] {
    let target = preset(SmVersion::SM_89).unwrap();
    let tiled = default_schedule(Arc::clone(compute), &target).unwrap().lower();
    let serial = Schedule::new(Arc::clone(compute)).validate(&target).unwrap().lower();
    let bits = |outputs: Vec<HostTensor>| -> Vec<Vec<u8>> {
        outputs.iter().map(|output| output.data().to_le_bytes()).collect()
    };
    [bits(run(&tiled, inputs).unwrap()), bits(run(&serial, inputs).unwrap())]
}

#[test]
fn a_product_gives_blocks_of_16_by_16_threads_a_64_by_64_tile_each() {
    let compute = product(DType::F32, [2, 70, 45, 20], Layout { a_transposed: true, ..PLAIN });
    let target = preset(SmVersion::SM_89).unwrap();
    let kernel = default_schedule(Arc::clone(&compute), &target).unwrap().lower();

    let (text, launch) = scheduled(compute);

    assert_eq!(launch_dims(launch), ([4, 1, 1], [16, 16, 1]));
    // Four stages of both 64 by 8 f32 slices.
    assert_eq!(kernel.smem().get(), 4 * 2 * 64 * 8 * 4);
    golden::check("matmul_simt.txt", &text).unwrap();
}

#[test]
fn a_product_over_several_blocks_and_steps_matches_the_serial_schedule() {
    let compute = product(DType::F32, [2, 70, 75, 21], Layout { b_transposed: true, ..PLAIN });
    let values: Vec<f32> = (0..1000_u16).map(|n| (f32::from(n) * 0.37).sin()).collect();
    let inputs = inputs(&compute, &values);

    let [tiled, serial] = tiled_and_serial(&compute, &inputs);

    assert_eq!(tiled, serial);
}

proptest! {
    #![proptest_config(crate::cases(64))]

    /// Every layout and overhang sums each output in the serial order, so
    /// the results match exactly.
    #[test]
    fn a_tiled_product_matches_the_serial_schedule_exactly(
        extents in [1_u64..=2, 1_u64..=20, 1_u64..=20, 1_u64..=20],
        (a_transposed, b_transposed, a_broadcast) in any::<(bool, bool, bool)>(),
        values in prop::collection::vec(-4.0_f32..4.0, 64),
    ) {
        let layout = Layout { a_transposed, b_transposed, a_broadcast };
        let compute = product(DType::F32, extents, layout);

        let [tiled, serial] = tiled_and_serial(&compute, &inputs(&compute, &values));

        prop_assert_eq!(tiled, serial);
    }
}
