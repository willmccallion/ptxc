//! Tensorized products on the GPU: every `mma` the target has, loading
//! its tiles by `ldmatrix` or element by element, matching the loop
//! interpreter bit for bit on inputs whose sums are exact in any order,
//! and an f16 product on random inputs within its tolerance.

use crate::device::{match_interpreter, match_interpreter_exactly, target};
use ptxc_core::float::F16;
use ptxc_core::{DType, Dim, HostTensor};
use ptxc_loop::{ComputeDef, LoweredKernel, TileOrder};
use ptxc_target::{MmaInstruction, MmaShape};
use ptxc_testkit::tensorized::{
    Blocks, Layout, Operands, TileLoops, exact_inputs, product, tensorized,
};
use std::sync::Arc;

/// A product overhanging `instruction`'s tiles on every axis, `b` stored
/// in `b_order`.
fn overhanging(instruction: MmaInstruction, b_order: TileOrder) -> Arc<ComputeDef> {
    let shape = instruction.shape;
    let extents = [shape.m.get() * 3 + 5, shape.n.get() * 4 + 3, shape.k.get() * 3 + 7];
    product(instruction, extents, Operands::matmul(extents, b_order))
}

fn lowered(
    compute: &Arc<ComputeDef>,
    layout: Layout,
    instruction: MmaInstruction,
) -> LoweredKernel {
    tensorized(compute, layout, instruction).validate(&target()).unwrap().lower()
}

#[test]
fn every_mma_matches_the_interpreter_exactly_with_either_storage_of_b() {
    for &instruction in target().mma() {
        for b_order in [TileOrder::RowMajor, TileOrder::Transposed] {
            for tile_loops in [TileLoops::Rolled, TileLoops::Unrolled] {
                let compute = overhanging(instruction, b_order);
                let layout = Layout { tiles: [2, 2], warps: 2, blocks: Blocks::Bound, tile_loops };
                let kernel = lowered(&compute, layout, instruction);
                let case = format!("{instruction:?}, b {b_order:?}, tile loops {tile_loops:?}");

                let _outputs =
                    match_interpreter_exactly(&kernel, &exact_inputs(&compute), &case).unwrap();
            }
        }
    }
}

#[test]
fn operands_read_at_strided_or_overlapping_rows_match_the_interpreter_exactly() {
    let instruction = target().mma_instruction(F16_SHAPE, DType::F16, DType::F32).unwrap();
    let strided = Operands {
        shapes: [vec![63, 32], vec![32, 16]],
        loads: Box::new(|compute, [a, b], [i, j, k]| {
            let two = compute.exprs().constant(2);
            [(a, vec![compute.exprs().mul(i, two), k]), (b, vec![k, j])]
        }),
    };
    let diagonal = Operands {
        shapes: [vec![63], vec![32, 16]],
        loads: Box::new(|compute, [a, b], [i, j, k]| {
            [(a, vec![compute.exprs().add(i, k)]), (b, vec![k, j])]
        }),
    };
    let layout =
        Layout { tiles: [1, 1], warps: 2, blocks: Blocks::Bound, tile_loops: TileLoops::Unrolled };

    for operands in [strided, diagonal] {
        let compute = product(instruction, [32, 16, 32], operands);
        let kernel = lowered(&compute, layout, instruction);

        let _outputs = match_interpreter_exactly(&kernel, &exact_inputs(&compute), "").unwrap();
    }
}

const F16_SHAPE: MmaShape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };

/// Values in [-1, 1] that neighbouring elements do not share, salted per
/// input.
fn random_halves(compute: &ComputeDef) -> Vec<HostTensor> {
    compute
        .inputs()
        .iter()
        .zip(1_u16..)
        .map(|(input, salt)| {
            let count = u16::try_from(input.shape.element_count().get()).unwrap();
            let values: Vec<F16> = (0..count)
                .map(|n| F16::from_f32(f32::from(n).mul_add(0.731, f32::from(salt)).sin()))
                .collect();
            HostTensor::new(input.shape.clone(), values.into()).unwrap()
        })
        .collect()
}

#[test]
fn an_f16_product_on_random_inputs_is_within_tolerance_of_the_interpreter() {
    let instruction = target().mma_instruction(F16_SHAPE, DType::F16, DType::F32).unwrap();
    let extents = [48, 40, 96];
    let compute = product(instruction, extents, Operands::matmul(extents, TileOrder::RowMajor));
    let layout =
        Layout { tiles: [1, 2], warps: 2, blocks: Blocks::Bound, tile_loops: TileLoops::Unrolled };
    let kernel = lowered(&compute, layout, instruction);

    let _outputs = match_interpreter(&kernel, &random_halves(&compute), "").unwrap();
}
