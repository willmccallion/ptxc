//! Validating schedules against the `sm_89` target.

use crate::schedule_support::{open, reduction};
use ptxc_core::Dim;
use ptxc_loop::{
    LoopId, Open, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis, VarName,
};
use ptxc_target::{SmVersion, ThreadsPerBlock, preset};

fn validate(schedule: Schedule<Open>) -> Result<(), ScheduleError> {
    schedule.validate(&preset(SmVersion::SM_89).unwrap()).map(drop)
}

fn bind(schedule: &mut Schedule<Open>, loop_: LoopId, axis: ThreadAxis) {
    schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
}

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factor: u64) -> Vec<LoopId> {
    schedule.split(loop_, vec![SplitFactor::new(factor).unwrap()]).unwrap()
}

#[test]
fn a_tiled_schedule_within_the_limits_validates() {
    let compute = reduction([1000, 30, 64]);
    let (mut schedule, nest) = open(&compute);
    let rows = split(&mut schedule, nest[0], 256);
    bind(&mut schedule, rows[0], ThreadAxis::BlockX);
    bind(&mut schedule, rows[1], ThreadAxis::ThreadX);
    bind(&mut schedule, nest[1], ThreadAxis::BlockY);

    assert_eq!(validate(schedule), Ok(()));
}

#[test]
fn each_bound_axis_has_the_targets_limit() {
    let too_large = |extents: [u64; 3], axis: ThreadAxis| {
        let compute = reduction(extents);
        let (mut schedule, nest) = open(&compute);
        bind(&mut schedule, nest[0], axis);
        validate(schedule)
    };

    let error = |axis, extent, limit| {
        Err(ScheduleError::AxisTooLarge { axis, extent: Dim::new(extent), limit })
    };
    assert_eq!(
        too_large([2000, 1, 1], ThreadAxis::ThreadX),
        error(ThreadAxis::ThreadX, 2000, 1024)
    );
    assert_eq!(too_large([65, 1, 1], ThreadAxis::ThreadZ), error(ThreadAxis::ThreadZ, 65, 64));
    assert_eq!(
        too_large([70_000, 1, 1], ThreadAxis::BlockY),
        error(ThreadAxis::BlockY, 70_000, 65_535)
    );
    assert_eq!(too_large([64, 1, 1], ThreadAxis::ThreadZ), Ok(()));
    assert_eq!(too_large([70_000, 1, 1], ThreadAxis::BlockX), Ok(()));
}

#[test]
fn a_block_has_at_most_the_targets_threads() {
    let compute = reduction([64, 32, 1]);
    let (mut schedule, nest) = open(&compute);
    bind(&mut schedule, nest[0], ThreadAxis::ThreadX);
    bind(&mut schedule, nest[1], ThreadAxis::ThreadY);

    let limit = ThreadsPerBlock::new(1024);
    assert_eq!(validate(schedule), Err(ScheduleError::TooManyThreads { threads: 2048, limit }));

    let compute = reduction([32, 32, 1]);
    let (mut schedule, nest) = open(&compute);
    bind(&mut schedule, nest[0], ThreadAxis::ThreadX);
    bind(&mut schedule, nest[1], ThreadAxis::ThreadY);
    assert_eq!(validate(schedule), Ok(()));
}

#[test]
fn a_vectorized_loop_is_innermost_unguarded_and_two_or_four_wide() {
    let vectorized = |extents: [u64; 3], shape: fn(&mut Schedule<Open>, &[LoopId]) -> LoopId| {
        let compute = reduction(extents);
        let (mut schedule, nest) = open(&compute);
        let loop_ = shape(&mut schedule, &nest);
        schedule.apply(Primitive::Vectorize { loop_ }).unwrap();
        validate(schedule)
    };

    let innermost = vectorized([1, 4, 1], |schedule, nest| {
        schedule.apply(Primitive::Reorder { order: vec![nest[2], nest[1]] }).unwrap();
        nest[1]
    });
    assert_eq!(innermost, Ok(()));
    let width = Err(ScheduleError::VectorWidth { name: VarName::new("j"), extent: Dim::new(3) });
    assert_eq!(vectorized([1, 3, 1], |_, nest| nest[1]), width);
    assert_eq!(
        vectorized([1, 4, 2], |_, nest| nest[1]),
        Err(ScheduleError::VectorNotInnermost(VarName::new("j")))
    );
    let guarded = vectorized([1, 10, 1], |schedule, nest| {
        let parts = split(schedule, nest[1], 4);
        schedule.apply(Primitive::Reorder { order: vec![nest[2], parts[1]] }).unwrap();
        parts[1]
    });
    assert_eq!(guarded, Err(ScheduleError::VectorGuarded(VarName::new("j.1"))));
}

#[test]
fn a_reduction_shared_between_threads_has_one_axis_and_no_spatial_loop_inside() {
    let compute = reduction([4, 4, 64]);

    let (mut schedule, nest) = open(&compute);
    let parts = split(&mut schedule, nest[2], 32);
    bind(&mut schedule, parts[1], ThreadAxis::ThreadX);
    assert_eq!(validate(schedule), Ok(()));

    let (mut schedule, nest) = open(&compute);
    let parts = split(&mut schedule, nest[2], 8);
    bind(&mut schedule, parts[0], ThreadAxis::ThreadX);
    bind(&mut schedule, parts[1], ThreadAxis::ThreadY);
    assert_eq!(validate(schedule), Err(ScheduleError::CrossThreadAxes));

    let (mut schedule, nest) = open(&compute);
    let parts = split(&mut schedule, nest[2], 32);
    bind(&mut schedule, parts[1], ThreadAxis::ThreadX);
    schedule.apply(Primitive::Reorder { order: vec![parts[0], nest[1]] }).unwrap();
    assert_eq!(validate(schedule), Err(ScheduleError::SpatialInsideCrossThread(VarName::new("j"))));
}
