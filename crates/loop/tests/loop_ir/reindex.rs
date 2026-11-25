//! Tiles staged in the iteration space with `reindex`: a convolution's
//! input and weights staged as the matrices of an implicit GEMM, which
//! never changes a result, and the reindexes a schedule rejects.

use crate::compute_build::{load, shape};
use crate::interp::{ints, run_default, target, tensor};
use crate::lower::lower;
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_core::{DType, Dim, Idx};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeName, IterVarId, LoopId, Open, Primitive, ReduceOp,
    Schedule, ScheduleError, SplitFactor, StageCount, TensorExpr, TensorName, TensorParamId,
    ThreadAxis, VarName, print_kernel, print_schedule, run,
};
use ptxc_testkit::convolution::{Channels, Conv, axes, axis, conv, conv_inputs, w, x};
use ptxc_testkit::golden;
use std::sync::Arc;

/// A tiling of `conv`'s implicit GEMM: `rows` by `columns` threads, each
/// computing one output, over output positions `n·p·q` by channels `m`,
/// the reduction over taps and channels `r·s·c` in steps of `depth`.
#[derive(Clone, Copy, Debug)]
struct Tiling {
    rows: u64,
    columns: u64,
    depth: u64,
}

/// The output positions, all rows of the GEMM.
const POSITIONS: [&str; 3] = ["n", "p", "q"];

/// The taps and input channels, the GEMM's depth, input channels
/// innermost.
const TAPS: [&str; 3] = ["r", "s", "c"];

/// `conv`'s nest as an implicit GEMM, its step loop returned: both
/// operands reindexed as the GEMM's matrices, the input as output
/// positions by taps and the weights as channels by taps, then tiled by
/// `tiling` with every output position a row.
fn implicit_gemm(compute: &Arc<ComputeDef>, tiling: Tiling) -> (Schedule<Open>, LoopId) {
    let (mut schedule, _) = open(compute);
    let input = vec![axes(compute, &POSITIONS), axes(compute, &TAPS)];
    schedule.apply(Primitive::Reindex { tensor: x(), groups: input }).unwrap();
    let weights = vec![axes(compute, &["m"]), axes(compute, &TAPS)];
    schedule.apply(Primitive::Reindex { tensor: w(), groups: weights }).unwrap();
    let step = gemm_loops(&mut schedule, (tiling, &POSITIONS));
    (schedule, step)
}

/// `conv`'s default nest tiled as an implicit GEMM by `tiling`, its step
/// loop returned: the output positions `rows` fused and split onto
/// `blockIdx.y` and `threadIdx.y`, any other output position on
/// `blockIdx.z`, channels split onto `blockIdx.x` and `threadIdx.x`, and
/// taps and input channels fused and split into steps.
fn gemm_loops(schedule: &mut Schedule<Open>, (tiling, rows): (Tiling, &[&str])) -> LoopId {
    let own = |schedule: &Schedule<Open>, name: &str| {
        schedule.loops().loop_of(axis(schedule.compute(), name)).unwrap()
    };
    let owns = |schedule: &Schedule<Open>, names: &[&str]| -> Vec<LoopId> {
        names.iter().map(|name| own(schedule, name)).collect()
    };
    let batch: Vec<&str> = POSITIONS.into_iter().filter(|name| !rows.contains(name)).collect();
    let (batch, row_loops, column) =
        (owns(schedule, &batch), owns(schedule, rows), own(schedule, "m"));
    let taps = owns(schedule, &TAPS);
    let order = [&batch[..], &row_loops, &[column], &taps].concat();
    schedule.apply(Primitive::Reorder { order }).unwrap();
    let by = |factor| vec![SplitFactor::new(factor).unwrap()];
    let positions = schedule.fuse(row_loops).unwrap();
    let taps = schedule.fuse(taps).unwrap();
    let row_parts = schedule.split(positions, by(tiling.rows)).unwrap();
    let columns = schedule.split(column, by(tiling.columns)).unwrap();
    let steps = schedule.split(taps, by(tiling.depth)).unwrap();
    let order = vec![row_parts[0], columns[0], row_parts[1], columns[1], steps[0], steps[1]];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    let mut binds = vec![
        (row_parts[0], ThreadAxis::BlockY),
        (columns[0], ThreadAxis::BlockX),
        (row_parts[1], ThreadAxis::ThreadY),
        (columns[1], ThreadAxis::ThreadX),
    ];
    binds.extend(batch.into_iter().map(|loop_| (loop_, ThreadAxis::BlockZ)));
    for (loop_, axis) in binds {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    steps[0]
}

/// Both operands staged at `step`.
fn stage_matrices(schedule: &mut Schedule<Open>, step: LoopId) {
    for tensor in [x(), w()] {
        schedule.apply(Primitive::CacheRead { tensor, at: step }).unwrap();
    }
}

fn agrees_with_default(compute: &Arc<ComputeDef>, schedule: Schedule<Open>) {
    let inputs = conv_inputs(compute);
    let expected = run_default(compute, &inputs).unwrap();
    let actual = run(&lower(schedule), &inputs).unwrap();
    assert_eq!(ints(&actual[0]), ints(&expected[0]));
}

const PADDED: Conv = Conv {
    batch: 2,
    size: 4,
    channels: 4,
    features: 4,
    kernel: 3,
    stride: 1,
    dilation: 1,
    pad: 1,
    groups: 1,
};

const SQUARE: Tiling = Tiling { rows: 4, columns: 4, depth: 4 };

#[test]
fn an_implicit_gemm_stages_a_padded_convolution_as_two_matrices() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, step) = implicit_gemm(&compute, SQUARE);
    stage_matrices(&mut schedule, step);

    let kernel = lower(schedule.clone());

    golden::check("lower/reindexed_conv.txt", &print_kernel(&kernel)).unwrap();
    assert_eq!(kernel.smem().get(), 2 * 4 * 4 * 4);
    agrees_with_default(&compute, schedule);
}

#[test]
fn channels_innermost_copy_in_runs_and_channels_outside_by_element() {
    let run_lengths = |channels| {
        let compute = conv(PADDED, channels);
        let (mut schedule, step) = implicit_gemm(&compute, SQUARE);
        stage_matrices(&mut schedule, step);
        let printed = print_kernel(&lower(schedule.clone()));
        agrees_with_default(&compute, schedule);
        ["run 4", "run 1"].map(|run| printed.matches(run).count())
    };

    assert_eq!(run_lengths(Channels::Last), [2, 0]);
    assert_eq!(run_lengths(Channels::First), [0, 2]);
}

#[test]
fn a_pipelined_strided_dilated_convolution_agrees() {
    let shape = Conv {
        batch: 3,
        size: 9,
        channels: 8,
        features: 6,
        kernel: 3,
        stride: 2,
        dilation: 2,
        pad: 2,
        groups: 1,
    };
    for channels in [Channels::Last, Channels::First] {
        let compute = conv(shape, channels);
        let (mut schedule, step) =
            implicit_gemm(&compute, Tiling { rows: 8, columns: 4, depth: 8 });
        stage_matrices(&mut schedule, step);
        let stages = StageCount::new(3).unwrap();
        schedule.apply(Primitive::Pipeline { loop_: step, stages }).unwrap();

        agrees_with_default(&compute, schedule);
    }
}

#[test]
fn a_reindexed_schedule_prints_its_coordinates_and_replays_them() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, step) = implicit_gemm(&compute, SQUARE);
    stage_matrices(&mut schedule, step);

    let replayed = Schedule::replay(Arc::clone(&compute), schedule.trace()).unwrap();

    let printed = print_schedule(&schedule);
    assert!(printed.contains("  reindex x [n, p, q] [r, s, c]\n"), "{printed}");
    assert!(printed.contains("  reindex w [m] [r, s, c]\n"), "{printed}");
    assert_eq!(print_schedule(&replayed), printed);
    assert_eq!(replayed.loops().caches(), schedule.loops().caches());
}

#[test]
fn a_reindex_needs_a_known_input_once() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, _) = open(&compute);
    let groups = vec![axes(&compute, &POSITIONS), axes(&compute, &TAPS)];
    let reindex = |tensor| Primitive::Reindex { tensor, groups: groups.clone() };

    let unknown = schedule.apply(reindex(TensorParamId::new(5)));
    schedule.apply(reindex(x())).unwrap();
    let again = schedule.apply(reindex(x()));

    assert_eq!(unknown, Err(ScheduleError::UnknownTensor(TensorParamId::new(5))));
    assert_eq!(again, Err(ScheduleError::AlreadyReindexed(TensorName::new("x"))));
}

#[test]
fn a_reindexed_input_must_be_staged() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, step) = implicit_gemm(&compute, SQUARE);
    schedule.apply(Primitive::CacheRead { tensor: w(), at: step }).unwrap();

    let validated = schedule.validate(&target());

    assert_eq!(validated.err(), Some(ScheduleError::ReindexUnstaged(TensorName::new("x"))));
}

#[test]
fn groups_are_nonempty_and_of_distinct_nonempty_axes() {
    let compute = conv(PADDED, Channels::Last);
    let mut empty_axis = ComputeBuilder::new(ComputeName::new("empty"));
    let input = empty_axis.input(TensorName::new("x"), DType::I32, shape(&[1]));
    let _empty = empty_axis.spatial(VarName::new("i"), Dim::new(0));
    let zero = empty_axis.exprs().constant(0);
    empty_axis.output(TensorName::new("out"), load(input, &[zero]));
    let empty_axis = Arc::new(empty_axis.finish().unwrap());
    let rejected = |compute: &Arc<ComputeDef>, groups: Vec<Vec<IterVarId>>| {
        let (mut schedule, _) = open(compute);
        schedule.apply(Primitive::Reindex { tensor: x(), groups })
    };

    let [n, p] = [axis(&compute, "n"), axis(&compute, "p")];
    let cases = [
        rejected(&compute, Vec::new()),
        rejected(&compute, vec![vec![n], Vec::new()]),
        rejected(&compute, vec![vec![n, p], vec![p]]),
        rejected(&compute, vec![vec![n, IterVarId::new(7)]]),
        rejected(&empty_axis, vec![vec![IterVarId::new(0)]]),
    ];

    let bad = || Err(ScheduleError::ReindexAxes(TensorName::new("x")));
    assert_eq!(cases, [bad(), bad(), bad(), bad(), bad()]);
}

#[test]
fn a_tile_missing_an_axis_that_varies_across_it_is_rejected() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, _) = open(&compute);
    let groups = vec![axes(&compute, &POSITIONS), axes(&compute, &["r", "s"])];
    schedule.apply(Primitive::Reindex { tensor: x(), groups }).unwrap();
    let step = gemm_loops(&mut schedule, (SQUARE, &POSITIONS));
    schedule.apply(Primitive::CacheRead { tensor: x(), at: step }).unwrap();

    let validated = schedule.validate(&target());

    let expected =
        ScheduleError::ReindexUncovered { tensor: TensorName::new("x"), axis: VarName::new("c") };
    assert_eq!(validated.err(), Some(expected));
}

#[test]
fn an_axis_fixed_across_the_tile_needs_no_coordinate() {
    // Each image on its own block: the batch is fixed across a tile.
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, _) = open(&compute);
    let groups = vec![axes(&compute, &["p", "q"]), axes(&compute, &TAPS)];
    schedule.apply(Primitive::Reindex { tensor: x(), groups }).unwrap();
    let step = gemm_loops(&mut schedule, (SQUARE, &["p", "q"]));
    schedule.apply(Primitive::CacheRead { tensor: x(), at: step }).unwrap();

    agrees_with_default(&compute, schedule);
}

#[test]
fn a_run_stops_at_its_axis_where_the_input_row_runs_on() {
    // `out[i] = sum over r, k of a[i, r, k]`, `k` reading 6 of a row's 8
    // elements: a run of 4 from `k = 4` would copy `a[i, r, 6..8]` where
    // the tile holds `a[i, r + 1, 0..2]`.
    let mut compute = ComputeBuilder::new(ComputeName::new("partial_rows"));
    let rows = compute.input(TensorName::new("a"), DType::I32, shape(&[2, 3, 8]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(2));
    let (r_axis, r) = compute.reduce_axis(VarName::new("r"), Dim::new(3));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(6));
    let body = Box::new(load(rows, &[i, r, k]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![r_axis, k_axis], body };
    compute.output(TensorName::new("out"), sum);
    let compute = Arc::new(compute.finish().unwrap());
    let (mut schedule, nest) = open(&compute);
    let staged = TensorParamId::new(0);
    let groups = vec![axes(&compute, &["i"]), axes(&compute, &["r", "k"])];
    schedule.apply(Primitive::Reindex { tensor: staged, groups }).unwrap();
    let taps = schedule.fuse(vec![nest[1], nest[2]]).unwrap();
    let steps = schedule.split(taps, vec![SplitFactor::new(4).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: nest[0], axis: ThreadAxis::ThreadX }).unwrap();
    schedule.apply(Primitive::CacheRead { tensor: staged, at: steps[0] }).unwrap();

    let printed = print_kernel(&lower(schedule.clone()));

    assert!(printed.contains(" run 2"), "{printed}");
    let values: Vec<i32> = (0..48).collect();
    let inputs = [tensor(&[2, 3, 8], values)];
    let expected = run_default(&compute, &inputs).unwrap();
    assert_eq!(ints(&run(&lower(schedule), &inputs).unwrap()[0]), ints(&expected[0]));
}

fn conv_shape() -> impl Strategy<Value = Conv> {
    (1_u64..=2, 1_u64..=6, 1_u64..=6, 1_u64..=5, 1_u64..=3, 1_u64..=2, 1_u64..=2, 0_u64..=2)
        .prop_filter_map("a window the padded input holds", |shape| {
            let (batch, size, channels, features, kernel, stride, dilation, pad) = shape;
            let geometry =
                Conv { batch, size, channels, features, kernel, stride, dilation, pad, groups: 1 };
            (size + 2 * pad > dilation * (kernel - 1)).then_some(geometry)
        })
}

proptest! {
    #![proptest_config(crate::cases(64))]

    /// Staging a convolution's operands as an implicit GEMM's matrices, in
    /// either channel layout, pipelined or not, never changes its result.
    #[test]
    fn staging_a_convolution_as_an_implicit_gemm_never_changes_its_result(
        shape in conv_shape(),
        channels_last in any::<bool>(),
        tiling in (1_u64..=4, 1_u64..=4, 1_u64..=8),
        stages in 1_u32..=3,
    ) {
        let channels = if channels_last { Channels::Last } else { Channels::First };
        let compute = conv(shape, channels);
        let (rows, columns, depth) = tiling;
        let (mut schedule, step) = implicit_gemm(&compute, Tiling { rows, columns, depth });
        stage_matrices(&mut schedule, step);
        if let Some(stages) = StageCount::new(stages) {
            schedule.apply(Primitive::Pipeline { loop_: step, stages }).unwrap();
        }
        agrees_with_default(&compute, schedule);
    }
}
