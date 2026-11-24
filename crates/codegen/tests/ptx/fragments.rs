//! Tensorized products: a multiplicand's fragment loaded by `ldmatrix`
//! where its tile's rows are aligned for it and element by element
//! otherwise, `mma.sync` for every variant the target has, and each lane's
//! accumulators stored element by element.

use crate::ptxas::assemble;
use crate::target;
use ptxc_codegen::{PtxModule, generate};
use ptxc_core::{ByteSize, DType, Dim};
use ptxc_loop::{ComputeDef, TileOrder};
use ptxc_target::{MmaInstruction, MmaShape};
use ptxc_testkit::golden;
use ptxc_testkit::tensorized::{Layout, ONE_TILE, Operands, TileLoops, product, tensorized};
use std::sync::Arc;

const UNROLLED: Layout = Layout { tile_loops: TileLoops::Unrolled, ..ONE_TILE };

fn f16_mma() -> MmaInstruction {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    target().mma_instruction(shape, DType::F16, DType::F32).unwrap()
}

/// A product two of `instruction`'s tiles wide and deep, `b` stored in
/// `b_order`.
fn matmul(instruction: MmaInstruction, b_order: TileOrder) -> Arc<ComputeDef> {
    let shape = instruction.shape;
    let extents = [shape.m.get() * 2, shape.n.get() * 2, shape.k.get() * 2];
    product(instruction, extents, Operands::matmul(extents, b_order))
}

fn emitted(compute: &Arc<ComputeDef>, layout: Layout, instruction: MmaInstruction) -> PtxModule {
    let kernel = tensorized(compute, layout, instruction).validate(&target()).unwrap().lower();
    generate(&kernel, &target()).unwrap()
}

fn count(module: &PtxModule, text: &str) -> usize {
    module.to_string().matches(text).count()
}

#[test]
fn every_mma_the_target_has_assembles_with_either_storage_of_b() {
    for &instruction in target().mma() {
        for b_order in [TileOrder::RowMajor, TileOrder::Transposed] {
            let module = emitted(&matmul(instruction, b_order), UNROLLED, instruction);

            assemble(&module);
            assert_eq!(count(&module, "mma.sync.aligned"), 1, "{instruction:?}");
        }
    }
}

#[test]
fn an_f16_product_loads_both_tiles_with_ldmatrix_and_multiplies_once_a_step() {
    let compute = matmul(f16_mma(), TileOrder::RowMajor);

    let module = emitted(&compute, UNROLLED, f16_mma());

    assert_eq!(count(&module, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"), 1);
    assert_eq!(count(&module, "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"), 1);
    golden::check("f16_product.ptx", &module.to_string()).unwrap();
}

#[test]
fn eight_bit_tiles_load_with_ldmatrix_only_where_their_runs_are_adjacent() {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(32) };
    let instruction = target().mma_instruction(shape, DType::I8, DType::I32).unwrap();
    let [k_major, n_major] = [TileOrder::Transposed, TileOrder::RowMajor]
        .map(|b_order| emitted(&matmul(instruction, b_order), UNROLLED, instruction));

    assert_eq!(count(&k_major, "ldmatrix"), 2);
    assert_eq!(count(&n_major, "ldmatrix"), 1);
    assert_eq!(count(&n_major, "ld.shared.s8"), 8);
}

#[test]
fn a_tile_whose_rows_are_not_16_byte_aligned_loads_element_by_element() {
    let instruction = f16_mma();
    let diagonal = Operands {
        shapes: [vec![63], vec![32, 16]],
        loads: Box::new(|compute, [a, b], [i, j, k]| {
            [(a, vec![compute.exprs().add(i, k)]), (b, vec![k, j])]
        }),
    };
    let compute = product(instruction, [32, 16, 32], diagonal);

    let module = emitted(&compute, UNROLLED, instruction);

    assert_eq!(count(&module, "ldmatrix"), 1);
    assemble(&module);
}

#[test]
fn a_warps_array_of_tiles_unrolled_needs_no_stack_frame() {
    let instruction = f16_mma();
    let layout = Layout { tiles: [2, 2], ..UNROLLED };
    let compute = matmul(instruction, TileOrder::RowMajor);

    let module = emitted(&compute, layout, instruction).to_string();

    let frame = ptxc_testkit::ptxas::stack_frame(&module, &target().sm().to_string()).unwrap();
    assert_eq!(frame, ByteSize::new(0));
}
