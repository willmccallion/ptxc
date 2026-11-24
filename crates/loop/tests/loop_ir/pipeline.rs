//! Pipelined loops: their tiles filled `stages - 1` iterations ahead, with
//! the same results as the default schedule for 2 to 4 stages, and the
//! schedules a pipeline rejects.

use crate::cache::{a, agrees_with_default, b, matmul, operands, stage, tiled};
use crate::interp::target;
use crate::lower::lower;
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_loop::{
    Open, Primitive, Schedule, ScheduleError, SplitFactor, StageCount, VarName, print_kernel,
    print_schedule,
};
use ptxc_target::SmemBytes;
use ptxc_testkit::golden;

const fn stages(count: u32) -> StageCount {
    StageCount::new(count).unwrap()
}

/// `matmul(extents)` in blocks of `tile` by `tile` threads, both operands
/// staged at the outer reduction loop of the reduction split by `depth`,
/// that loop pipelined over `count` stages.
fn pipelined(extents: [u64; 3], [tile, depth]: [u64; 2], count: u32) -> Schedule<Open> {
    let compute = matmul(extents);
    let (mut schedule, [ko, _]) = tiled(&compute, [tile, depth]);
    stage(&mut schedule, a(), ko);
    stage(&mut schedule, b(), ko);
    schedule.apply(Primitive::Pipeline { loop_: ko, stages: stages(count) }).unwrap();
    schedule
}

#[test]
fn a_pipelined_loop_starts_its_first_fills_ahead_and_refills_the_slot_just_read() {
    let schedule = pipelined([8, 8, 16], [4, 4], 3);

    let kernel = lower(schedule.clone());

    golden::check("lower/pipelined_matmul.txt", &print_kernel(&kernel)).unwrap();
    assert_eq!(kernel.smem().get(), 3 * 2 * 4 * 4 * 4);
    assert!(print_schedule(&schedule).contains("pipeline %k.0 3"), "{}", print_schedule(&schedule));
}

#[test]
fn pipelining_two_to_four_stages_never_changes_a_result() {
    let cases = [([8, 8, 16], [4, 4]), ([37, 45, 70], [8, 16]), ([5, 7, 3], [4, 8])];
    for (extents, tiling) in cases {
        for count in 2..=4 {
            let compute = matmul(extents);
            agrees_with_default(&compute, pipelined(extents, tiling, count), &operands(extents));
        }
    }
}

#[test]
fn a_loop_its_prologue_fills_entirely_refills_nothing() {
    // One reduction step for two stages, and two for three or four.
    let cases = [([1, 1, 1], [1, 1], 2, 1), ([4, 4, 8], [4, 4], 3, 2), ([4, 4, 8], [4, 4], 4, 2)];
    for (extents, tiling, count, steps) in cases {
        let schedule = pipelined(extents, tiling, count);

        let kernel = print_kernel(&lower(schedule.clone()));

        // Each step's two tiles are copied once, all before the loop.
        assert_eq!(kernel.matches("<- async").count(), 2 * steps, "{kernel}");
        agrees_with_default(&matmul(extents), schedule, &operands(extents));
    }
}

#[test]
fn only_the_loop_around_a_tiles_reads_fills_it_ahead() {
    // With `k` outermost, the accumulators' set-up and write-back also loop
    // over `i`, but read no tile, so only the update's `i` loop pipelines.
    let compute = matmul([8, 4, 8]);
    let (mut schedule, nest) = open(&compute);
    schedule.apply(Primitive::Reorder { order: vec![nest[2], nest[0], nest[1]] }).unwrap();
    stage(&mut schedule, a(), nest[0]);
    schedule.apply(Primitive::Pipeline { loop_: nest[0], stages: stages(2) }).unwrap();

    let kernel = print_kernel(&lower(schedule.clone()));

    assert_eq!(kernel.matches("async wait").count(), 1, "{kernel}");
    agrees_with_default(&compute, schedule, &operands([8, 4, 8]));
}

#[test]
fn a_stage_count_is_at_least_two() {
    assert_eq!([0, 1].map(StageCount::new), [None, None]);
    assert_eq!(StageCount::new(2).map(StageCount::get), Some(2));
}

#[test]
fn only_an_unannotated_loop_with_an_input_staged_at_it_pipelines() {
    let compute = matmul([8, 8, 16]);
    let (mut unstaged, [ko, ki]) = tiled(&compute, [4, 4]);
    unstaged.apply(Primitive::Pipeline { loop_: ko, stages: stages(2) }).unwrap();
    let mut unrolled = unstaged.clone();
    unrolled.apply(Primitive::Unroll { loop_: ki }).unwrap();
    let pipeline_inner = Primitive::Pipeline { loop_: ki, stages: stages(2) };

    let split = unstaged.split(ko, vec![SplitFactor::new(2).unwrap()]);
    let annotated = unrolled.apply(pipeline_inner);

    assert_eq!(split, Err(ScheduleError::Annotated(VarName::new("k.0"))));
    assert_eq!(annotated, Err(ScheduleError::Annotated(VarName::new("k.1"))));
    let unstaged_error = unstaged.validate(&target()).unwrap_err();
    assert_eq!(unstaged_error, ScheduleError::PipelineUnstaged(VarName::new("k.0")));
}

#[test]
fn each_stage_takes_its_own_copy_of_the_tiles_in_shared_memory() {
    // Two i32 tiles of 16 by 160 take 20 KiB, so two stages fit 48 KiB.
    let fits = pipelined([16, 16, 320], [16, 160], 2).validate(&target());
    let over = pipelined([16, 16, 320], [16, 160], 3).validate(&target());

    assert!(fits.is_ok(), "{:?}", fits.err());
    let (bytes, limit) = (ptxc_core::ByteSize::new(3 * 2 * 16 * 160 * 4), SmemBytes::new(49_152));
    assert_eq!(over.err(), Some(ScheduleError::SharedTooLarge { bytes, limit }));
}

proptest! {
    #![proptest_config(crate::cases(128))]

    /// Pipelining a tiled matmul's staged reduction loop over 2 to 4 stages
    /// never changes its result, however its extents divide.
    #[test]
    fn pipelining_a_staged_matmul_never_changes_its_result(
        extents in [1_u64..=9, 1_u64..=9, 1_u64..=12],
        tile in 1_u64..=4,
        depth in 1_u64..=4,
        count in 2_u32..=4,
    ) {
        let compute = matmul(extents);
        agrees_with_default(&compute, pipelined(extents, [tile, depth], count), &operands(extents));
    }
}
