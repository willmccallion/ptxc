//! Staged tiles fill with `cp.async`: 16-byte copies cached only in L2,
//! 8- and 4-byte ones in L1 too, and a copy a bound can fail reads a
//! selected count of bytes, zero where the bound fails.

use crate::ptxas::assemble;
use crate::target;
use ptxc_codegen::generate;
use ptxc_core::{DType, Dim};
use ptxc_loop::TileOrder;
use ptxc_target::MmaShape;
use ptxc_testkit::tensorized::{ONE_TILE, Operands, product, tensorized};
use std::collections::BTreeSet;

/// The f16 product 16 by 8 by `depth` on one warp's tile, its copies as
/// (cache operator, bytes, whether a selected count bounds the bytes read).
fn copies(depth: u64) -> BTreeSet<(String, u64, bool)> {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    let instruction = target().mma_instruction(shape, DType::F16, DType::F32).unwrap();
    let extents = [16, 8, depth];
    let compute = product(instruction, extents, Operands::matmul(extents, TileOrder::RowMajor));
    let kernel = tensorized(&compute, ONE_TILE, instruction).validate(&target()).unwrap().lower();
    let module = generate(&kernel, &target()).unwrap();
    assemble(&module);
    module.to_string().lines().filter_map(copy).collect()
}

/// A `cp.async` line's cache operator, bytes and whether it has a count.
fn copy(line: &str) -> Option<(String, u64, bool)> {
    let rest = line.trim().strip_prefix("cp.async.")?;
    let (cache, operands) = rest.split_once(".shared.global ")?;
    let operands: Vec<&str> = operands.trim_end_matches(';').split(", ").collect();
    Some((cache.to_owned(), operands[2].parse().unwrap(), operands.len() == 4))
}

#[test]
fn a_tile_copies_in_the_cp_async_size_its_rows_allow() {
    // `a`'s rows run along the depth, `b`'s along its 8 columns; a depth
    // that overhangs the 16-deep step bounds both.
    let expected = [
        (32, vec![("cg", 16, false)]),
        (20, vec![("ca", 8, true), ("cg", 16, true)]),
        (18, vec![("ca", 4, true), ("cg", 16, true)]),
    ];

    for (depth, kinds) in expected {
        let kinds =
            kinds.into_iter().map(|(cache, bytes, bounded)| (cache.to_owned(), bytes, bounded));
        assert_eq!(copies(depth), kinds.collect(), "depth {depth}");
    }
}
