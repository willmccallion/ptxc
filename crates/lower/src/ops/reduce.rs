//! Reductions, arg-reductions and running sums, each reduction one stage
//! (D40). Floats accumulate in f32 (D43).

use super::NodeContext;
use crate::compose::{add, div, f32, mul, select, sub, unary};
use crate::error::LowerError;
use crate::precision::{arithmetic_dtype, convert, demote, promote};
use crate::stage::{Binding, LoweredGroup, Operand, StageBuilder};
use ptxc_core::{Axis, DType, Dim, ElementCount, IndexVec, Shape};
use ptxc_graph::{
    ArgKind, ArgReduceAttrs, CumSumAttrs, Inclusion, KeepDims, Layout, ReduceAttrs, ReduceOp,
    ScanDirection, TensorType, TieBreak, ValueId,
};
use ptxc_loop::{
    ArgKind as LoopArgKind, BinaryOp, ComputeError, ExprId, InfSign, IterVarId,
    ReduceOp as LoopReduceOp, Scalar, TensorExpr, TieBreak as LoopTieBreak, UnaryOp,
};

/// The reduction axes of one stage and how many elements each output
/// element combines.
struct Over {
    axes: Vec<IterVarId>,
    count: ElementCount,
}

impl Over {
    fn apply(&self, op: LoopReduceOp, body: TensorExpr) -> TensorExpr {
        TensorExpr::Reduce { op, axes: self.axes.clone(), body: Box::new(body) }
    }

    fn sum(&self, body: TensorExpr) -> TensorExpr {
        self.apply(LoopReduceOp::Sum, body)
    }
}

pub(super) fn reduce(
    cx: &NodeContext<'_>,
    attrs: &ReduceAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    if attrs.op == ReduceOp::LogSumExp {
        return log_sum_exp(cx, input, attrs);
    }
    cx.one_stage(|stage| {
        let (x, over) = load_reduced(cx, stage, input, attrs)?;
        if x.dtype.is_float() {
            let total = float_pass(attrs.op, promote(x.expr.clone(), x.dtype), &over);
            return Ok(demote(total, x.dtype));
        }
        integer_pass(cx, attrs.op, &x, &over)
    })
}

/// `x` read at each element its output element reduces, and the axes
/// reduced over. A reduction over no axes reduces each element alone,
/// through an axis of extent 1.
fn load_reduced(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    input: ValueId,
    attrs: &ReduceAttrs,
) -> Result<(Operand, Over), LowerError> {
    let shape = cx.graph.value(input).ty().shape();
    let (index, mut axes) = reduced_index(cx, stage, shape, &attrs.axes, attrs.keep_dims)?;
    if axes.is_empty() {
        axes.push(stage.reduce_axis(Dim::new(1)).0);
    }
    let x = stage.load(Binding::Value(input), index);
    Ok((x, Over { axes, count: reduced_count(shape, &attrs.axes) }))
}

/// The input index for one output element: a new reduction axis for each
/// reduced input axis and the matching spatial axis for every other; a
/// kept reduced axis is an output axis of extent 1 that nothing reads.
fn reduced_index(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    input: &Shape,
    reduced: &[Axis],
    keep: KeepDims,
) -> Result<(Vec<ExprId>, Vec<IterVarId>), LowerError> {
    let mut spatial = stage.spatial().to_vec().into_iter();
    let (mut index, mut axes) = (Vec::new(), Vec::new());
    for (axis, &extent) in input.rank().axes().zip(input.dims()) {
        if !reduced.contains(&axis) {
            index.push(spatial.next().ok_or(LowerError::Operands(cx.id))?);
            continue;
        }
        let (iter, var) = stage.reduce_axis(extent);
        index.push(var);
        axes.push(iter);
        if keep == KeepDims::Keep {
            let _unread = spatial.next();
        }
    }
    Ok((index, axes))
}

/// The number of elements one output element combines.
fn reduced_count(input: &Shape, reduced: &[Axis]) -> ElementCount {
    let count = input
        .rank()
        .axes()
        .zip(input.dims())
        .filter(|(axis, _)| reduced.contains(axis))
        .fold(1_u64, |count, (_, dim)| count.saturating_mul(dim.get()));
    ElementCount::new(count)
}

/// `op` over `w`, an f32 expression, in one pass. `LogSumExp` is the plain
/// formula; `log_sum_exp` applies it to shifted inputs.
fn float_pass(op: ReduceOp, w: TensorExpr, over: &Over) -> TensorExpr {
    let square = || mul(w.clone(), w.clone());
    match op {
        ReduceOp::Sum => over.sum(w),
        ReduceOp::Mean => div(over.sum(w), f32(over.count.to_f32())),
        ReduceOp::Prod => over.apply(LoopReduceOp::Prod, w),
        ReduceOp::Max => over.apply(LoopReduceOp::Max, w),
        ReduceOp::Min => over.apply(LoopReduceOp::Min, w),
        ReduceOp::SumSquare => over.sum(square()),
        ReduceOp::L1 => over.sum(unary(UnaryOp::Abs, w)),
        ReduceOp::L2 => unary(UnaryOp::Sqrt, over.sum(square())),
        ReduceOp::LogSum => unary(UnaryOp::Log, over.sum(w)),
        ReduceOp::LogSumExp => unary(UnaryOp::Log, over.sum(unary(UnaryOp::Exp, w))),
    }
}

/// `op` over integers or booleans: exact in their own type where ONNX's
/// result is, wrapping as the reference does; `Mean` divides the exact sum.
fn integer_pass(
    cx: &NodeContext<'_>,
    op: ReduceOp,
    x: &Operand,
    over: &Over,
) -> Result<TensorExpr, LowerError> {
    let value = || x.expr.clone();
    Ok(match op {
        ReduceOp::Sum => over.sum(value()),
        ReduceOp::Prod => over.apply(LoopReduceOp::Prod, value()),
        ReduceOp::Max => over.apply(LoopReduceOp::Max, value()),
        ReduceOp::Min => over.apply(LoopReduceOp::Min, value()),
        ReduceOp::SumSquare => over.sum(mul(value(), value())),
        ReduceOp::L1 => over.sum(unary(UnaryOp::Abs, value())),
        ReduceOp::Mean => integer_mean(cx, x, over)?,
        ReduceOp::L2 | ReduceOp::LogSum | ReduceOp::LogSumExp => {
            let w = convert(value(), x.dtype, DType::F32);
            truncated(float_pass(op, w, over), x.dtype)
        }
    })
}

/// The sum in `i64` divided by the count, truncating, as the reference's
/// f64 mean truncates. An empty reduction's sum is 0 and so is its mean,
/// as the reference's NaN truncates to.
fn integer_mean(cx: &NodeContext<'_>, x: &Operand, over: &Over) -> Result<TensorExpr, LowerError> {
    let too_large = || cx.ill_formed(ComputeError::ExtentTooLarge(Dim::new(over.count.get())));
    let count = i64::try_from(over.count.get().max(1)).map_err(|_| too_large())?;
    let divisor = Scalar::int(DType::I64, count).ok_or_else(too_large)?;
    let sum = over.sum(convert(x.expr.clone(), x.dtype, DType::I64));
    let mean = div(sum, TensorExpr::Const(divisor));
    Ok(convert(mean, DType::I64, x.dtype))
}

/// An f32 result truncated toward zero into the integer type `dtype`,
/// through `i64` as the reference's f64 result is.
fn truncated(result: TensorExpr, dtype: DType) -> TensorExpr {
    // NOTE: the reference computes integer `ReduceL2`, `ReduceLogSum` and
    // `ReduceLogSumExp` in f64; loop IR has no f64 (D42), so ptxc uses f32,
    // exact while the sums stay below 2^24.
    convert(convert(result, DType::F32, DType::I64), DType::I64, dtype)
}

/// `ReduceLogSumExp` in two stages: the maximum `m` of each output
/// element's inputs, then `m + ln(sum(e^(x - m)))`, which cannot overflow;
/// an infinite `m` is the result itself, and NaN anywhere gives NaN.
fn log_sum_exp(
    cx: &NodeContext<'_>,
    input: ValueId,
    attrs: &ReduceAttrs,
) -> Result<LoweredGroup, LowerError> {
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let ty = cx.graph.value(output).ty();
    let mut intermediates = IndexVec::new();
    let max = intermediates.push(TensorType::new(DType::F32, ty.shape().clone(), Layout::RowMajor));
    let wide = |x: &Operand| convert(x.expr.clone(), x.dtype, DType::F32);

    let mut first = StageBuilder::new(cx.graph, &intermediates, cx.named("max"), ty.shape());
    let (x, over) = load_reduced(cx, &mut first, input, attrs)?;
    first.output(Binding::Intermediate(max), over.apply(LoopReduceOp::Max, wide(&x)));

    let mut second = StageBuilder::new(cx.graph, &intermediates, cx.named("sum"), ty.shape());
    let (x, over) = load_reduced(cx, &mut second, input, attrs)?;
    let spatial = second.spatial().to_vec();
    let m = second.load(Binding::Intermediate(max), spatial).expr;
    let shifted = add(m.clone(), float_pass(ReduceOp::LogSumExp, sub(wide(&x), m.clone()), &over));
    let is_infinite = unary(UnaryOp::IsInf(InfSign::Either), m.clone());
    let result = select(is_infinite, m, shifted);
    let result =
        if x.dtype.is_float() { demote(result, x.dtype) } else { truncated(result, x.dtype) };
    second.output(Binding::Value(output), result);

    let stages = [first, second]
        .into_iter()
        .map(|stage| stage.finish().map_err(|source| cx.ill_formed(source)))
        .collect::<Result<_, _>>()?;
    Ok(LoweredGroup::new(intermediates, stages))
}

/// `ArgMax` or `ArgMin` along one axis, compared in the input's own type.
pub(super) fn arg_reduce(
    cx: &NodeContext<'_>,
    attrs: ArgReduceAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let shape = cx.graph.value(input).ty().shape();
        let (index, axes) = reduced_index(cx, stage, shape, &[attrs.axis], attrs.keep_dims)?;
        let [axis] = axes[..] else { return Err(LowerError::Operands(cx.id)) };
        let x = stage.load(Binding::Value(input), index);
        let kind = match attrs.kind {
            ArgKind::Max => LoopArgKind::Max,
            ArgKind::Min => LoopArgKind::Min,
        };
        let tie = match attrs.tie {
            TieBreak::First => LoopTieBreak::First,
            TieBreak::Last => LoopTieBreak::Last,
        };
        Ok(TensorExpr::ArgReduce { kind, tie, axis, body: Box::new(x.expr) })
    })
}

/// `CumSum` as, for each element, a sum along the axis masked to the
/// elements the scan has reached.
pub(super) fn cum_sum(
    cx: &NodeContext<'_>,
    attrs: CumSumAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [input] = cx.values()?;
    cx.one_stage(|stage| {
        let ty = cx.graph.value(input).ty();
        let position = attrs.axis.index();
        let extent = ty.shape().dim(attrs.axis).ok_or(LowerError::Operands(cx.id))?;
        let spatial = stage.spatial().to_vec();
        let here = *spatial.get(position).ok_or(LowerError::Operands(cx.id))?;
        let (k_axis, k) = stage.reduce_axis(extent);
        let index = spatial
            .iter()
            .enumerate()
            .map(|(p, &axis)| if p == position { k } else { axis })
            .collect();
        let x = stage.load(Binding::Value(input), index);
        let reached = TensorExpr::Binary(
            scan_comparison(attrs),
            Box::new(TensorExpr::Index { index: k, dtype: DType::I64 }),
            Box::new(TensorExpr::Index { index: here, dtype: DType::I64 }),
        );
        let wide = promote(x.expr.clone(), x.dtype);
        let zero = TensorExpr::Const(Scalar::zero(arithmetic_dtype(x.dtype)));
        let over = Over { axes: vec![k_axis], count: ElementCount::new(extent.get()) };
        let total = over.sum(select(reached, wide, zero));
        Ok(demote(total, x.dtype))
    })
}

/// How the summed position `k` compares with the output position for `k`
/// to be included.
const fn scan_comparison(attrs: CumSumAttrs) -> BinaryOp {
    match (attrs.direction, attrs.inclusion) {
        (ScanDirection::Forward, Inclusion::Inclusive) => BinaryOp::LessOrEqual,
        (ScanDirection::Forward, Inclusion::Exclusive) => BinaryOp::Less,
        (ScanDirection::Reverse, Inclusion::Inclusive) => BinaryOp::GreaterOrEqual,
        (ScanDirection::Reverse, Inclusion::Exclusive) => BinaryOp::Greater,
    }
}
