//! Tensorized products: each warp's tile computed by `mma` on fragments
//! matches the serial schedule exactly, on inputs whose sums are exact.

use crate::interp::{floats, target};
use proptest::prelude::*;
use ptxc_core::{DType, Dim};
use ptxc_loop::{
    ComputeDef, Open, Schedule, TileOrder, print_kernel, print_schedule, run, unrolled,
};
use ptxc_target::{MmaInstruction, MmaShape};
use ptxc_testkit::golden;
use ptxc_testkit::tensorized::{
    self, Blocks, Layout, Loads, ONE_TILE, Operands, TileLoops, exact_inputs,
};
use std::sync::Arc;

/// `sm_89`'s m16n8k16 on f16, accumulating in f32.
pub(crate) fn instruction() -> MmaInstruction {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    target().mma_instruction(shape, DType::F16, DType::F32).unwrap()
}

/// The f16 product `out[i, j] = sum over k of a[i, k] * b[k, j]` over
/// `extents`, `b` holding its `k × n` matrix in `b_order`.
pub(crate) fn product(extents: [u64; 3], b_order: TileOrder) -> Arc<ComputeDef> {
    tensorized::product(instruction(), extents, Operands::matmul(extents, b_order))
}

/// The f16 product of the loads `loads` makes from `a` and `b` of
/// `shapes`, over `extents`.
pub(crate) fn product_reading(
    shapes: [&[u64]; 2],
    extents: [u64; 3],
    loads: Box<Loads>,
) -> Arc<ComputeDef> {
    let shapes = shapes.map(<[u64]>::to_vec);
    tensorized::product(instruction(), extents, Operands { shapes, loads })
}

/// `compute` tensorized with f16's m16n8k16, laid out as `layout`.
pub(crate) fn tensorized(compute: &Arc<ComputeDef>, layout: Layout) -> Schedule<Open> {
    tensorized::tensorized(compute, layout, instruction())
}

/// `compute`'s output under `schedule` and under the serial schedule.
fn tensorized_and_serial(compute: &Arc<ComputeDef>, schedule: Schedule<Open>) -> [Vec<f32>; 2] {
    let inputs = exact_inputs(compute);
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
fn a_staged_tile_fills_in_16_byte_runs_only_where_its_rows_are_whole_runs() {
    let [whole, ragged] = [32, 20].map(|depth| {
        let compute = product([16, 8, depth], TileOrder::RowMajor);
        print_kernel(&tensorized(&compute, ONE_TILE).validate(&target()).unwrap().lower())
    });

    let runs =
        |kernel: &str, tile: &str| kernel.contains(&format!("for %{tile}_run in 8 vectorize"));
    assert!(runs(&whole, "a") && runs(&whole, "b"));
    // `a`'s rows run along a depth of 20; `b`'s are its 8 columns.
    assert!(!runs(&ragged, "a") && runs(&ragged, "b"), "{ragged}");
}

#[test]
fn a_staged_tile_whose_origin_is_not_a_whole_run_from_its_rows_start_fills_by_element() {
    let shifted: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
        let four = compute.exprs().constant(4);
        [(a, vec![i, compute.exprs().add(k, four)]), (b, vec![k, j])]
    });
    let compute = product_reading([&[32, 40], &[32, 16]], [32, 16, 32], shifted);
    let schedule = tensorized(&compute, ONE_TILE);

    let kernel = print_kernel(&schedule.validate(&target()).unwrap().lower());

    assert!(!kernel.contains("%a_run") && kernel.contains("for %b_run in 8 vectorize"), "{kernel}");
    let [tiled, serial] = tensorized_and_serial(&compute, tensorized(&compute, ONE_TILE));
    assert_eq!(tiled, serial);
}

#[test]
fn only_a_tensorized_kernel_multiplies_on_tensor_cores() {
    let compute = product([32, 16, 32], TileOrder::RowMajor);

    let tiled = tensorized(&compute, ONE_TILE).validate(&target()).unwrap().lower();
    let serial = Schedule::new(Arc::clone(&compute)).validate(&target()).unwrap().lower();

    assert!(tiled.multiplies_on_tensor_cores());
    assert!(!serial.multiplies_on_tensor_cores());
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
    let layout = Layout { tiles: [2, 2], warps: 2, ..ONE_TILE };

    let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

    assert_eq!(tensorized, serial);
}

#[test]
fn unrolling_a_warps_array_of_tiles_keeps_each_tiles_place() {
    let compute = product([32, 16, 32], TileOrder::RowMajor);
    let layout = Layout { tiles: [2, 2], tile_loops: TileLoops::Unrolled, ..ONE_TILE };
    let kernel = tensorized(&compute, layout).validate(&target()).unwrap().lower();

    let expanded = unrolled(&kernel);

    let inputs = exact_inputs(&compute);
    let serial =
        run(&Schedule::new(Arc::clone(&compute)).validate(&target()).unwrap().lower(), &inputs);
    assert_eq!(floats(&run(&expanded, &inputs).unwrap()[0]), floats(&serial.unwrap()[0]));
}

#[test]
fn a_tile_whose_blocks_run_serially_in_one_block_matches_the_serial_product() {
    let compute = product([32, 24, 32], TileOrder::RowMajor);
    let layout = Layout { blocks: Blocks::Serial, ..ONE_TILE };

    let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

    assert_eq!(tensorized, serial);
}

#[test]
fn operands_in_either_order_and_at_any_affine_index_match_the_serial_product() {
    let reversed: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
        let zero = compute.exprs().constant(0);
        [(b, vec![k, j]), (a, vec![zero, i, k])]
    });
    let strided: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
        let two = compute.exprs().constant(2);
        [(a, vec![compute.exprs().mul(i, two), k]), (b, vec![k, j])]
    });
    let diagonal: Box<Loads> = Box::new(|compute, [a, b], [i, j, k]| {
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
        let layout = Layout { tiles, warps, ..ONE_TILE };

        let [tensorized, serial] = tensorized_and_serial(&compute, tensorized(&compute, layout));

        prop_assert_eq!(tensorized, serial);
    }
}
