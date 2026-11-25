//! Tiles staged in the iteration space with `reindex`: a convolution's
//! input and weights staged as the matrices of an implicit GEMM, which
//! never changes a result, and the reindexes a schedule rejects.

use crate::compute_build::{binary, load, shape};
use crate::interp::{ints, run_default, target, tensor};
use crate::lower::lower;
use crate::schedule_support::open;
use proptest::prelude::*;
use ptxc_core::{DType, Dim, HostTensor, Idx};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprArena, ExprId, IterVarId, LoopId, Open,
    Primitive, ReduceOp, Scalar, Schedule, ScheduleError, SplitFactor, StageCount, TensorExpr,
    TensorName, TensorParamId, ThreadAxis, VarName, print_kernel, print_schedule, run,
};
use ptxc_testkit::golden;
use std::sync::Arc;

/// A square 2-D convolution: its batch, input size and channels, output
/// channels, kernel size, stride, dilation and padding on every side.
#[derive(Clone, Copy, Debug)]
struct Conv {
    batch: u64,
    size: u64,
    channels: u64,
    features: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
    pad: u64,
}

/// Where a convolution's channels lie in its input and weights.
#[derive(Clone, Copy, Debug)]
enum Channels {
    /// Innermost: `x[n, h, w, c]` and `w[m, r, s, c]`.
    Last,
    /// After the batch or output channel: `x[n, c, h, w]` and
    /// `w[m, c, r, s]`.
    First,
}

impl Conv {
    /// The output size, at least one where the padded input holds a window.
    const fn output(self) -> u64 {
        let span = self.dilation * (self.kernel - 1) + 1;
        (self.size + 2 * self.pad - span) / self.stride + 1
    }

    fn signed(value: u64) -> i64 {
        i64::try_from(value).unwrap()
    }
}

/// The axes of `conv`: `n`, `p`, `q`, `m` spatial, `r`, `s`, `c` summed.
const N: usize = 0;
const P: usize = 1;
const Q: usize = 2;
const M: usize = 3;
const R: usize = 4;
const S: usize = 5;
const C: usize = 6;

fn axes(ids: &[usize]) -> Vec<IterVarId> {
    ids.iter().map(|&id| IterVarId::new(id)).collect()
}

fn x() -> TensorParamId {
    TensorParamId::new(0)
}

fn w() -> TensorParamId {
    TensorParamId::new(1)
}

/// Where tap `tap` of output `out` reads along one spatial axis, clamped
/// into the input, and whether it reads the input rather than padding,
/// as `ptxc-lower` builds a convolution's windows.
fn tap(
    exprs: &mut ExprArena,
    geometry: Conv,
    (out, tap): (ExprId, ExprId),
) -> (ExprId, TensorExpr) {
    let [stride, dilation] =
        [geometry.stride, geometry.dilation].map(|value| exprs.constant(Conv::signed(value)));
    let start = exprs.mul(out, stride);
    let offset = exprs.mul(tap, dilation);
    let padded = exprs.add(start, offset);
    let minus_pad = exprs.constant(-Conv::signed(geometry.pad));
    let position = exprs.add(padded, minus_pad);
    let (zero, last) = (exprs.constant(0), exprs.constant(Conv::signed(geometry.size) - 1));
    let above = exprs.max(position, zero);
    let clamped = exprs.min(above, last);
    let at = || TensorExpr::Index { index: position, dtype: DType::I64 };
    let bound = |value: u64| TensorExpr::Const(Scalar::i64(Conv::signed(value)));
    let after_start = binary(BinaryOp::GreaterOrEqual, at(), bound(0));
    let before_end = binary(BinaryOp::Less, at(), bound(geometry.size));
    (clamped, binary(BinaryOp::And, after_start, before_end))
}

/// `out[n, p, q, m] = sum over r, s, c of x * w` on `i32`, each tap
/// masked to zero where it reads padding.
fn conv(geometry: Conv, channels: Channels) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("conv"));
    let Conv { batch, size, channels: depth, features, kernel, .. } = geometry;
    let (x_shape, w_shape) = match channels {
        Channels::Last => ([batch, size, size, depth], [features, kernel, kernel, depth]),
        Channels::First => ([batch, depth, size, size], [features, depth, kernel, kernel]),
    };
    let image = compute.input(TensorName::new("x"), DType::I32, shape(&x_shape));
    let filter = compute.input(TensorName::new("w"), DType::I32, shape(&w_shape));
    let output = Dim::new(geometry.output());
    let (_, sample) = compute.spatial(VarName::new("n"), Dim::new(batch));
    let (_, row) = compute.spatial(VarName::new("p"), output);
    let (_, column) = compute.spatial(VarName::new("q"), output);
    let (_, feature) = compute.spatial(VarName::new("m"), Dim::new(features));
    let (tap_rows, tap_row) = compute.reduce_axis(VarName::new("r"), Dim::new(kernel));
    let (tap_columns, tap_column) = compute.reduce_axis(VarName::new("s"), Dim::new(kernel));
    let (channels_axis, channel) = compute.reduce_axis(VarName::new("c"), Dim::new(depth));
    let (row_read, row_inside) = tap(compute.exprs(), geometry, (row, tap_row));
    let (column_read, column_inside) = tap(compute.exprs(), geometry, (column, tap_column));
    let (x_index, w_index) = match channels {
        Channels::Last => {
            ([sample, row_read, column_read, channel], [feature, tap_row, tap_column, channel])
        }
        Channels::First => {
            ([sample, channel, row_read, column_read], [feature, channel, tap_row, tap_column])
        }
    };
    let product = binary(BinaryOp::Mul, load(image, &x_index), load(filter, &w_index));
    let inside = binary(BinaryOp::And, row_inside, column_inside);
    let zero = TensorExpr::Const(Scalar::zero(DType::I32));
    let body = TensorExpr::Select {
        condition: Box::new(inside),
        then: Box::new(product),
        otherwise: Box::new(zero),
    };
    let axes = vec![tap_rows, tap_columns, channels_axis];
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes, body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

/// Inputs of `conv` counting up from distinct starts, so a read of the
/// wrong element shows.
fn inputs(compute: &ComputeDef) -> Vec<HostTensor> {
    let tensors = compute.inputs();
    let values = |param: TensorParamId, start: i32| {
        let dims: Vec<u64> = tensors[param].shape.dims().iter().map(|dim| dim.get()).collect();
        let count = i32::try_from(dims.iter().product::<u64>()).unwrap();
        tensor(&dims, (0..count).map(|value| start + value % 23).collect::<Vec<i32>>())
    };
    vec![values(x(), 1), values(w(), -9)]
}

/// A tiling of `conv`'s implicit GEMM: `rows` by `columns` threads, each
/// computing one output, over output positions `n·p·q` by channels `m`,
/// the reduction over taps and channels `r·s·c` in steps of `depth`.
#[derive(Clone, Copy, Debug)]
struct Tiling {
    rows: u64,
    columns: u64,
    depth: u64,
}

/// `conv`'s nest as an implicit GEMM, its step loop returned: both
/// operands reindexed as the GEMM's matrices, the input as output
/// positions by taps and the weights as channels by taps, then tiled by
/// `tiling` with every output position a row.
fn implicit_gemm(compute: &Arc<ComputeDef>, tiling: Tiling) -> (Schedule<Open>, LoopId) {
    let (mut schedule, nest) = open(compute);
    let input = vec![axes(&[N, P, Q]), axes(&[R, S, C])];
    schedule.apply(Primitive::Reindex { tensor: x(), groups: input }).unwrap();
    let weights = vec![axes(&[M]), axes(&[R, S, C])];
    schedule.apply(Primitive::Reindex { tensor: w(), groups: weights }).unwrap();
    let step = gemm_loops(&mut schedule, &nest, (tiling, &[N, P, Q]));
    (schedule, step)
}

/// `conv`'s default nest `nest` tiled as an implicit GEMM by `tiling`, its
/// step loop returned: the axes of `rows` fused and split onto
/// `blockIdx.y` and `threadIdx.y`, any other output position axis on
/// `blockIdx.z`, channels split onto `blockIdx.x` and `threadIdx.x`, and
/// taps and input channels fused and split into steps.
fn gemm_loops(
    schedule: &mut Schedule<Open>,
    nest: &[LoopId],
    (tiling, rows): (Tiling, &[usize]),
) -> LoopId {
    let by = |factor| vec![SplitFactor::new(factor).unwrap()];
    let positions = schedule.fuse(rows.iter().map(|&axis| nest[axis]).collect()).unwrap();
    let taps = schedule.fuse(vec![nest[R], nest[S], nest[C]]).unwrap();
    let row_parts = schedule.split(positions, by(tiling.rows)).unwrap();
    let columns = schedule.split(nest[M], by(tiling.columns)).unwrap();
    let steps = schedule.split(taps, by(tiling.depth)).unwrap();
    let order = vec![row_parts[0], columns[0], row_parts[1], columns[1], steps[0], steps[1]];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    let mut binds = vec![
        (row_parts[0], ThreadAxis::BlockY),
        (columns[0], ThreadAxis::BlockX),
        (row_parts[1], ThreadAxis::ThreadY),
        (columns[1], ThreadAxis::ThreadX),
    ];
    let batch = [N, P, Q].into_iter().find(|axis| !rows.contains(axis));
    binds.extend(batch.map(|axis| (nest[axis], ThreadAxis::BlockZ)));
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
    let inputs = inputs(compute);
    let expected = run_default(compute, &inputs).unwrap();
    let actual = run(&lower(schedule), &inputs).unwrap();
    assert_eq!(ints(&actual[0]), ints(&expected[0]));
}

const PADDED: Conv =
    Conv { batch: 2, size: 4, channels: 4, features: 4, kernel: 3, stride: 1, dilation: 1, pad: 1 };

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
    let groups = vec![axes(&[N, P, Q]), axes(&[R, S, C])];
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

    let cases = [
        rejected(&compute, Vec::new()),
        rejected(&compute, vec![axes(&[N]), Vec::new()]),
        rejected(&compute, vec![axes(&[N, P]), axes(&[P])]),
        rejected(&compute, vec![axes(&[N, 7])]),
        rejected(&empty_axis, vec![axes(&[0])]),
    ];

    let bad = || Err(ScheduleError::ReindexAxes(TensorName::new("x")));
    assert_eq!(cases, [bad(), bad(), bad(), bad(), bad()]);
}

#[test]
fn a_tile_missing_an_axis_that_varies_across_it_is_rejected() {
    let compute = conv(PADDED, Channels::Last);
    let (mut schedule, nest) = open(&compute);
    let groups = vec![axes(&[N, P, Q]), axes(&[R, S])];
    schedule.apply(Primitive::Reindex { tensor: x(), groups }).unwrap();
    let step = gemm_loops(&mut schedule, &nest, (SQUARE, &[N, P, Q]));
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
    let (mut schedule, nest) = open(&compute);
    let groups = vec![axes(&[P, Q]), axes(&[R, S, C])];
    schedule.apply(Primitive::Reindex { tensor: x(), groups }).unwrap();
    let step = gemm_loops(&mut schedule, &nest, (SQUARE, &[P, Q]));
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
    let groups = vec![axes(&[0]), axes(&[1, 2])];
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
            let geometry = Conv { batch, size, channels, features, kernel, stride, dilation, pad };
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
