//! Swizzled tiles: the schedules a swizzle rejects, and that a swizzle
//! never changes a result.

use crate::cache::{a, agrees_with_default, b, matmul, operands, stage, tiled};
use crate::interp::target;
use crate::lower::lower;
use ptxc_core::Idx;
use ptxc_loop::{
    Primitive, ScheduleError, Swizzle, TensorName, TensorParamId, print_kernel, print_schedule,
};

/// The swizzle for 16-byte rows: 4 of them share a 128-byte bank line.
fn swizzle() -> Swizzle {
    Swizzle::for_rows(16).unwrap()
}

#[test]
fn a_staged_matmul_with_both_tiles_swizzled_computes_the_same_product() {
    // i32 tiles of 16 by 4 and 4 by 16: 16- and 64-byte rows, 256 bytes in all.
    let extents = [32, 32, 16];
    let compute = matmul(extents);
    let (mut schedule, [ko, _]) = tiled(&compute, [16, 4]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ko);
    for (tensor, row_bytes) in [(a(), 16), (b(), 64)] {
        let swizzle = Swizzle::for_rows(row_bytes).unwrap();
        schedule.apply(Primitive::Swizzle { tensor, swizzle }).unwrap();
    }

    let kernel = print_kernel(&lower(schedule.clone()));

    assert_eq!(kernel.matches("shared swizzle<3,4,3>").count(), 2, "{kernel}");
    assert!(print_schedule(&schedule).contains("swizzle a 3 4 3"));
    agrees_with_default(&compute, schedule, &operands(extents));
}

#[test]
fn only_a_staged_tile_holding_whole_blocks_swizzles_once() {
    let compute = matmul([32, 32, 16]);
    let (mut schedule, [ko, _]) = tiled(&compute, [16, 4]);
    let unstaged = schedule.apply(Primitive::Swizzle { tensor: a(), swizzle: swizzle() });
    let unknown =
        schedule.apply(Primitive::Swizzle { tensor: TensorParamId::new(5), swizzle: swizzle() });
    stage(&mut schedule, a(), ko);
    schedule.apply(Primitive::Swizzle { tensor: a(), swizzle: swizzle() }).unwrap();
    let again = schedule.apply(Primitive::Swizzle { tensor: a(), swizzle: swizzle() });
    // A 2 by 2 tile of `b` from a 4-row split holds 16 bytes, not 128.
    let (mut small, [small_ko, _]) = tiled(&matmul([4, 4, 4]), [2, 2]);
    stage(&mut small, b(), small_ko);
    small.apply(Primitive::Swizzle { tensor: b(), swizzle: swizzle() }).unwrap();

    assert_eq!(unstaged, Err(ScheduleError::SwizzleUnstaged(TensorName::new("a"))));
    assert_eq!(unknown, Err(ScheduleError::UnknownTensor(TensorParamId::new(5))));
    assert_eq!(again, Err(ScheduleError::AlreadySwizzled(TensorName::new("a"))));
    let rejected = small.validate(&target()).err();
    assert_eq!(rejected, Some(ScheduleError::SwizzleLayout(TensorName::new("b"))));
}

#[test]
fn a_staged_inputs_tile_extents_are_its_own() {
    let compute = matmul([32, 32, 16]);
    let (mut schedule, [ko, _]) = tiled(&compute, [16, 4]);
    stage(&mut schedule, a(), ko);

    let [staged, unstaged] = [a(), b()].map(|tensor| schedule.staged_extents(tensor).unwrap());

    assert_eq!(staged, Some(vec![ptxc_core::Dim::new(16), ptxc_core::Dim::new(4)]));
    assert_eq!(unstaged, None);
    stage(&mut schedule, b(), ko);
    let both = schedule.staged_extents(b()).unwrap();
    assert_eq!(both, Some(vec![ptxc_core::Dim::new(4), ptxc_core::Dim::new(16)]));
}
