//! Each primitive's effect on the loop nest, its errors, and replay.

use crate::schedule_support::{domain, open, reduction, visited};
use ptxc_core::Idx;
use ptxc_loop::{
    LoopAnnotation, LoopId, Primitive, Schedule, ScheduleError, SplitFactor, ThreadAxis, VarName,
    print_schedule,
};
use ptxc_testkit::golden;

fn factors(values: &[u64]) -> Vec<SplitFactor> {
    values.iter().map(|&value| SplitFactor::new(value).unwrap()).collect()
}

/// Visits every point of the domain once, in any order.
fn assert_covers(schedule: &Schedule<ptxc_loop::Open>, extents: &[u64]) {
    let mut points = visited(schedule);
    points.sort();
    assert_eq!(points, domain(extents), "{}", print_schedule(schedule));
}

#[test]
fn the_default_nest_puts_spatial_loops_outside_reductions() {
    let compute = reduction([2, 3, 4]);
    let (schedule, nest) = open(&compute);

    let names: Vec<&str> =
        nest.iter().map(|&id| schedule.loops().loops()[id].name.as_str()).collect();
    assert_eq!(names, ["i", "j", "k"]);
    assert_eq!(visited(&schedule), domain(&[2, 3, 4]));
}

#[test]
fn a_dividing_split_needs_no_guard() {
    let compute = reduction([8, 1, 1]);
    let (mut schedule, nest) = open(&compute);

    let parts = schedule.split(nest[0], factors(&[4])).unwrap();

    assert_eq!(parts.len(), 2);
    assert_eq!(schedule.loops().guards(), []);
    assert_covers(&schedule, &[8, 1, 1]);
}

#[test]
fn a_split_that_does_not_divide_rounds_up_and_guards() {
    let compute = reduction([10, 1, 1]);
    let (mut schedule, nest) = open(&compute);

    let parts = schedule.split(nest[0], factors(&[4])).unwrap();

    assert_eq!(schedule.loops().loops()[parts[0]].extent.get(), 3);
    assert_eq!(schedule.loops().guards().len(), 1);
    assert_covers(&schedule, &[10, 1, 1]);
}

#[test]
fn fusing_then_splitting_still_covers_the_domain() {
    let compute = reduction([6, 5, 3]);
    let (mut schedule, nest) = open(&compute);

    let fused = schedule.fuse(vec![nest[0], nest[1]]).unwrap();
    let parts = schedule.split(fused, factors(&[4, 2])).unwrap();
    schedule.apply(Primitive::Reorder { order: vec![nest[2], parts[0]] }).unwrap();

    assert_covers(&schedule, &[6, 5, 3]);
}

#[test]
fn a_tiled_reduction_prints_its_nest_indices_guards_and_trace() {
    let compute = reduction([64, 30, 32]);
    let (mut schedule, nest) = open(&compute);
    let i = schedule.split(nest[0], factors(&[16])).unwrap();
    let j = schedule.split(nest[1], factors(&[8])).unwrap();
    let k = schedule.split(nest[2], factors(&[4])).unwrap();
    for step in [
        Primitive::Reorder { order: vec![i[0], j[0], k[0], i[1], j[1], k[1]] },
        Primitive::Bind { loop_: i[0], axis: ThreadAxis::BlockX },
        Primitive::Bind { loop_: j[0], axis: ThreadAxis::BlockY },
        Primitive::Unroll { loop_: k[1] },
        Primitive::Vectorize { loop_: j[1] },
    ] {
        schedule.apply(step).unwrap();
    }

    assert_covers(&schedule, &[64, 30, 32]);
    golden::check("schedule/tiled_reduction.txt", &print_schedule(&schedule)).unwrap();
}

#[test]
fn replaying_the_trace_rebuilds_the_same_schedule() {
    let compute = reduction([12, 7, 5]);
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[1], factors(&[3])).unwrap();
    let fused = schedule.fuse(vec![nest[0], parts[0]]).unwrap();
    schedule.apply(Primitive::Bind { loop_: fused, axis: ThreadAxis::BlockX }).unwrap();

    let replayed = Schedule::replay(compute, schedule.trace()).unwrap();

    assert_eq!(print_schedule(&replayed), print_schedule(&schedule));
}

#[test]
fn a_rejected_primitive_changes_nothing() {
    let compute = reduction([4, 4, 4]);
    let (mut schedule, nest) = open(&compute);
    schedule.apply(Primitive::Bind { loop_: nest[0], axis: ThreadAxis::BlockX }).unwrap();
    let before = print_schedule(&schedule);

    let error = schedule.apply(Primitive::Bind { loop_: nest[1], axis: ThreadAxis::BlockX });

    assert_eq!(error, Err(ScheduleError::AxisTaken(ThreadAxis::BlockX)));
    assert_eq!(print_schedule(&schedule), before);
}

#[test]
fn split_and_fuse_reject_what_they_cannot_do() {
    let compute = reduction([4, 4, 4]);
    let (mut schedule, nest) = open(&compute);
    let name = |text: &str| VarName::new(text);

    assert_eq!(
        schedule.apply(Primitive::Split { loop_: nest[0], factors: Vec::new() }),
        Err(ScheduleError::EmptySplit)
    );
    assert_eq!(
        schedule.split(nest[0], factors(&[u64::MAX, 2])),
        Err(ScheduleError::ExtentOverflow)
    );
    assert_eq!(
        schedule.apply(Primitive::Fuse { loops: vec![nest[0]] }),
        Err(ScheduleError::FuseTooFew)
    );
    assert_eq!(
        schedule.apply(Primitive::Fuse { loops: vec![nest[1], nest[0]] }),
        Err(ScheduleError::NotAdjacent(name("i")))
    );
    assert_eq!(
        schedule.apply(Primitive::Fuse { loops: vec![nest[1], nest[2]] }),
        Err(ScheduleError::MixedKinds(name("k")))
    );
    let _parts = schedule.split(nest[0], factors(&[2])).unwrap();
    assert_eq!(schedule.split(nest[0], factors(&[2])), Err(ScheduleError::NotInNest(name("i"))));
    assert_eq!(
        schedule.apply(Primitive::Unroll { loop_: LoopId::new(99) }),
        Err(ScheduleError::UnknownLoop(LoopId::new(99)))
    );
}

#[test]
fn annotations_reject_reductions_where_they_cannot_run() {
    let compute = reduction([4, 4, 4]);
    let (mut schedule, nest) = open(&compute);
    let k = VarName::new("k");

    assert_eq!(
        schedule.apply(Primitive::Bind { loop_: nest[2], axis: ThreadAxis::BlockZ }),
        Err(ScheduleError::BlockBoundReduce(k.clone()))
    );
    assert_eq!(
        schedule.apply(Primitive::Vectorize { loop_: nest[2] }),
        Err(ScheduleError::VectorizedReduce(k.clone()))
    );
    schedule.apply(Primitive::Bind { loop_: nest[2], axis: ThreadAxis::ThreadX }).unwrap();
    assert_eq!(
        schedule.loops().loops()[nest[2]].annotation,
        LoopAnnotation::Bound(ThreadAxis::ThreadX)
    );
    assert_eq!(
        schedule.apply(Primitive::Unroll { loop_: nest[2] }),
        Err(ScheduleError::Annotated(k))
    );
}

#[test]
fn reorder_rejects_a_repeated_loop() {
    let compute = reduction([4, 4, 4]);
    let (mut schedule, nest) = open(&compute);

    let error = schedule.apply(Primitive::Reorder { order: vec![nest[1], nest[1]] });

    assert_eq!(error, Err(ScheduleError::RepeatedLoop(VarName::new("j"))));
}

#[test]
fn fusing_three_loops_inside_the_outermost_covers_the_domain() {
    let compute = reduction([3, 4, 2]);
    let (mut schedule, nest) = open(&compute);
    let parts = schedule.split(nest[0], factors(&[1])).unwrap();

    let fused = schedule.fuse(vec![parts[1], nest[1]]).unwrap();

    assert_eq!(schedule.loops().nest()[1], fused);
    assert_eq!(schedule.loops().loops()[fused].extent.get(), 4);
    assert_covers(&schedule, &[3, 4, 2]);
}

#[test]
fn a_split_of_a_fused_loop_by_the_inner_extent_gives_back_plain_indices() {
    let compute = reduction([6, 5, 1]);
    let (mut schedule, nest) = open(&compute);
    let fused = schedule.fuse(vec![nest[0], nest[1]]).unwrap();

    let _parts = schedule.split(fused, factors(&[5])).unwrap();

    let text = print_schedule(&schedule);
    assert!(text.contains("  %i = %i_j.0\n  %j = %i_j.1\n"), "{text}");
    assert_eq!(schedule.loops().guards(), []);
}

#[test]
fn an_empty_loop_cannot_be_bound() {
    let compute = reduction([4, 0, 3]);
    let (mut schedule, nest) = open(&compute);

    let error = schedule.apply(Primitive::Bind { loop_: nest[1], axis: ThreadAxis::ThreadX });

    assert_eq!(error, Err(ScheduleError::EmptyBound(VarName::new("j"))));
}

#[test]
fn an_axis_has_its_own_loop_until_a_split_combines_loops_for_it() {
    let compute = reduction([4, 6, 5]);
    let (mut schedule, nest) = open(&compute);
    let axes: Vec<_> = compute.iter_vars().ids().collect();
    let own = |schedule: &Schedule<ptxc_loop::Open>| -> Vec<Option<LoopId>> {
        axes.iter().map(|&axis| schedule.loops().loop_of(axis)).collect()
    };
    let before = own(&schedule);
    schedule.apply(Primitive::Reorder { order: vec![nest[2], nest[0]] }).unwrap();
    let reordered = own(&schedule);

    let _parts = schedule.split(nest[1], factors(&[4])).unwrap();

    assert_eq!(before, [Some(nest[0]), Some(nest[1]), Some(nest[2])]);
    assert_eq!(reordered, before);
    assert_eq!(own(&schedule), [Some(nest[0]), None, Some(nest[2])]);
}
