//! Matrix products under the default schedule on tensor cores, in every
//! type the target's `mma` table takes, at the sweep's kinds of shape made
//! small enough to interpret: square, uneven, skinny and transformer-like.
//! On inputs whose sums are exact they match the loop interpreter bit for
//! bit (D58).

use crate::device::{match_interpreter_exactly, target};
use ptxc_loop::TileOrder;
use ptxc_schedule::default_schedule;
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
