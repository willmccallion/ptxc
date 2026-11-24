//! Tensorized products: each warp's tile computed by `mma` on fragments
//! matches the serial schedule exactly, on inputs whose sums are exact.

use crate::compute_build::shape;
use crate::interp::{floats, target};
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_core::float::{F16, Overflow};
use ptxc_core::{DType, Dim, HostTensor, Idx};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, LoopId, Open, Primitive, ReduceOp,
    Schedule, SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, TileOrder, VarName,
    print_kernel, print_schedule, run, unrolled,
};
use ptxc_target::{MmaInstruction, MmaShape};
use ptxc_testkit::golden;
use std::sync::Arc;

/// Which loads `out[i, j] = sum over k of x * y` multiplies: `x` and `y`
/// from the inputs `a` and `b`, given those and `i`, `j` and `k`.
pub(crate) type Operands = dyn FnOnce(
    &mut ComputeBuilder,
    [TensorParamId; 2],
    [ExprId; 3],
) -> [(TensorParamId, Vec<ExprId>); 2];

/// `out[i, j] = sum over k of x * y` over `extents`, the loads `operands`
/// chooses from f16 inputs of `shapes`, each widened to f32.
pub(crate) fn product_reading(
    shapes: [&[u64]; 2],
    [rows, columns, depth]: [u64; 3],
    operands: Box<Operands>,
) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let inputs = [("a", shapes[0]), ("b", shapes[1])]
        .map(|(name, dims)| compute.input(TensorName::new(name), DType::F16, shape(dims)));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let [x, y] = operands(&mut compute, inputs, [row, column, k]).map(|(tensor, index)| {
        Box::new(TensorExpr::Cast {
            to: DType::F32,
            overflow: Overflow::Saturate,
            value: Box::new(TensorExpr::Load { tensor, index }),
        })
    });
    let body = Box::new(TensorExpr::Binary(BinaryOp::Mul, x, y));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    Arc::new(compute.finish().unwrap())
}

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` over `extents`, `b`
/// holding its `k × n` matrix in `b_order`.
pub(crate) fn product(extents: [u64; 3], b_order: TileOrder) -> Arc<ComputeDef> {
    let [rows, columns, depth] = extents;
    let b_dims = match b_order {
        TileOrder::RowMajor => [depth, columns],
        TileOrder::Transposed => [columns, depth],
    };
    let operands =
        move |_: &mut ComputeBuilder, [a, b]: [TensorParamId; 2], [i, j, k]: [ExprId; 3]| {
            let b_index = match b_order {
                TileOrder::RowMajor => vec![k, j],
                TileOrder::Transposed => vec![j, k],
            };
            [(a, vec![i, k]), (b, b_index)]
        };
    product_reading([&[rows, depth], &b_dims], extents, Box::new(operands))
}

/// `sm_89`'s m16n8k16 on f16, accumulating in f32.
pub(crate) fn instruction() -> MmaInstruction {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    target().mma_instruction(shape, DType::F16, DType::F32).unwrap()
}

/// Whether a tensorized nest's block loops are bound to the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Blocks {
    Bound,
    Serial,
}

/// How a warp's work is laid out: tiles per warp along rows and columns,
/// warps per block along rows, and the block loops.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) tiles: [u64; 2],
    pub(crate) warps: u64,
    pub(crate) blocks: Blocks,
}

pub(crate) const ONE_TILE: Layout = Layout { tiles: [1, 1], warps: 1, blocks: Blocks::Bound };

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factors: &[u64]) -> Vec<LoopId> {
    let factors = factors.iter().map(|&factor| SplitFactor::new(factor).unwrap()).collect();
    schedule.split(loop_, factors).unwrap()
}

/// `compute` tensorized: rows split into blocks, warps, tiles and the
/// mma's 16; columns into blocks, tiles and 8; the reduction into steps of
/// 16 staging both operands; the tile's loops innermost.
pub(crate) fn tensorized(compute: &Arc<ComputeDef>, layout: Layout) -> Schedule<Open> {
    let (mut schedule, nest) = open(compute);
    let rows = split(&mut schedule, nest[0], &[layout.warps, layout.tiles[0], 16]);
    let columns = split(&mut schedule, nest[1], &[layout.tiles[1], 8]);
    let depth = split(&mut schedule, nest[2], &[16]);
    let order = vec![
        rows[0], columns[0], rows[1], depth[0], rows[2], columns[1], rows[3], columns[2], depth[1],
    ];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    if layout.blocks == Blocks::Bound {
        let blocks = [(rows[0], ThreadAxis::BlockY), (columns[0], ThreadAxis::BlockX)];
        for (loop_, axis) in blocks {
            schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
        }
    }
    schedule.apply(Primitive::Bind { loop_: rows[1], axis: ThreadAxis::ThreadY }).unwrap();
    for tensor in [0, 1].map(TensorParamId::new) {
        schedule.apply(Primitive::CacheRead { tensor, at: depth[0] }).unwrap();
    }
    let instruction = instruction();
    schedule
        .apply(Primitive::Tensorize { m: rows[3], n: columns[2], k: depth[1], instruction })
        .unwrap();
    schedule
}

/// The inputs of `compute`: small integers, every sum of whose products is
/// exact in f32 in any order.
fn inputs(compute: &ComputeDef) -> Vec<HostTensor> {
    compute
        .inputs()
        .iter()
        .enumerate()
        .map(|(salt, input)| {
            let count = input.shape.element_count().get();
            let values: Vec<F16> = (0..count)
                .map(|n| {
                    F16::from_i64(
                        i64::try_from((3 * n + 7 * u64::try_from(salt).unwrap()) % 9).unwrap() - 4,
                    )
                })
                .collect();
            HostTensor::new(input.shape.clone(), values.into()).unwrap()
        })
        .collect()
}

/// `compute`'s output under `schedule` and under the serial schedule.
fn tensorized_and_serial(compute: &Arc<ComputeDef>, schedule: Schedule<Open>) -> [Vec<f32>; 2] {
    let inputs = inputs(compute);
    let tensorized = run(&schedule.validate(&target()).unwrap().lower(), &inputs).unwrap();
    let serial =
        run(&Schedule::new(Arc::clone(compute)).validate(&target()).unwrap().lower(), &inputs)
            .unwrap();
    [floats(&tensorized[0]), floats(&serial[0])]
}

#[test]
fn a_warp_computing_one_tile_per_step_matches_the_serial_product() {
    let compute = product([32, 16, 32], TileOrder::RowMajor);

    let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, ONE_TILE));

    assert_eq!(tensorized, serial);
}

#[test]
fn a_tensorized_kernel_loads_staged_tiles_into_fragments_and_stores_element_by_element() {
    let compute = product([16, 8, 32], TileOrder::RowMajor);

    let kernel = tensorized(&compute, ONE_TILE).validate(&target()).unwrap().lower();

    golden::check("lower/tensorized_product.txt", &print_kernel(&kernel)).unwrap();
}

#[test]
fn warps_holding_arrays_of_tiles_over_overhanging_inputs_match_the_serial_product() {
    let compute = product([40, 21, 35], TileOrder::Transposed);
    let layout = Layout { tiles: [2, 2], warps: 2, blocks: Blocks::Bound };

    let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

    assert_eq!(tensorized, serial);
}

#[test]
fn unrolling_a_warps_array_of_tiles_keeps_each_tiles_place() {
    let compute = product([32, 16, 32], TileOrder::RowMajor);
    let layout = Layout { tiles: [2, 2], warps: 1, blocks: Blocks::Bound };
    let mut schedule = tensorized(&compute, layout);
    let nest = schedule.loops().nest().to_vec();
    let tile_loops = [nest[4], nest[5]];
    for loop_ in tile_loops {
        schedule.apply(Primitive::Unroll { loop_ }).unwrap();
    }
    let kernel = schedule.validate(&target()).unwrap().lower();

    let expanded = unrolled(&kernel);

    let inputs = inputs(&compute);
    let serial =
        run(&Schedule::new(Arc::clone(&compute)).validate(&target()).unwrap().lower(), &inputs);
    assert_eq!(floats(&run(&expanded, &inputs).unwrap()[0]), floats(&serial.unwrap()[0]));
}

#[test]
fn a_tile_whose_blocks_run_serially_in_one_block_matches_the_serial_product() {
    let compute = product([32, 24, 32], TileOrder::RowMajor);
    let layout = Layout { tiles: [1, 1], warps: 1, blocks: Blocks::Serial };

    let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

    assert_eq!(tensorized, serial);
}

#[test]
fn operands_in_either_order_and_at_any_affine_index_match_the_serial_product() {
    let reversed: Box<Operands> = Box::new(|compute, [a, b], [i, j, k]| {
        let zero = compute.exprs().constant(0);
        [(b, vec![k, j]), (a, vec![zero, i, k])]
    });
    let strided: Box<Operands> = Box::new(|compute, [a, b], [i, j, k]| {
        let two = compute.exprs().constant(2);
        [(a, vec![compute.exprs().mul(i, two), k]), (b, vec![k, j])]
    });
    let diagonal: Box<Operands> = Box::new(|compute, [a, b], [i, j, k]| {
        [(a, vec![compute.exprs().add(i, k)]), (b, vec![k, j])]
    });
    let products = [
        product_reading([&[1, 32, 32], &[32, 16]], [32, 16, 32], reversed),
        product_reading([&[63, 32], &[32, 16]], [32, 16, 32], strided),
        product_reading([&[63], &[32, 16]], [32, 16, 32], diagonal),
    ];

    for compute in products {
        let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, ONE_TILE));
        assert_eq!(tensorized, serial);
    }
}

#[test]
fn a_tensorized_schedule_prints_its_tile_and_replays() {
    let compute = product([32, 16, 32], TileOrder::RowMajor);
    let schedule = tensorized(&compute, ONE_TILE);

    let replayed = Schedule::replay(Arc::clone(&compute), schedule.trace()).unwrap();

    let printed = print_schedule(&schedule);
    assert!(printed.contains("  tensorize %i.3, %j.2, %k.1 mma.m16n8k16.f16.f32\n"), "{printed}");
    assert_eq!(print_schedule(&replayed), printed);
    assert_eq!(replayed.loops().tensorized(), schedule.loops().tensorized());
}

proptest! {
    #![proptest_config(crate::cases(32))]

    /// Any extents, tiles per warp, warps per block and storage of `b`
    /// give the serial schedule's result exactly.
    #[test]
    fn a_tensorized_product_matches_the_serial_product(
        extents in [1_u64..=40, 1_u64..=24, 1_u64..=48],
        tiles in [1_u64..=2, 1_u64..=2],
        warps in 1_u64..=2,
        b_order in prop_oneof![Just(TileOrder::RowMajor), Just(TileOrder::Transposed)],
    ) {
        let compute = product(extents, b_order);
        let layout = Layout { tiles, warps, blocks: Blocks::Bound };

        let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

        prop_assert_eq!(tensorized, serial);
    }
}
