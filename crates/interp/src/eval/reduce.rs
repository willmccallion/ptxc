//! Reductions and arg-reductions. Floats reduce in f64 (D19).

use super::round_to_output;
use crate::error::{InterpError, Malformed};
use crate::eval::first;
use crate::extents::Extents;
use crate::numeric::{
    Elements, count_to_f64, int_range, int_to_f64, maximum, minimum, position_to_i64, store,
    store_ints, truncate, widen,
};
use ptxc_core::{Axis, HostTensor, TensorData};
use ptxc_graph::{ArgKind, ArgReduceAttrs, ReduceAttrs, ReduceOp, TensorType, TieBreak};

/// For each output position, the input offsets it reduces, in input order.
/// Keeping a reduced axis adds an extent-1 axis, which leaves the order of
/// output positions unchanged.
fn groups(input: &Extents, axes: &[Axis]) -> Vec<Vec<usize>> {
    let kept: Vec<usize> =
        (0..input.rank()).filter(|&axis| !axes.iter().any(|a| a.index() == axis)).collect();
    let out = Extents::from_dims(kept.iter().map(|&axis| input.dims()[axis]).collect());
    let mut groups = vec![Vec::new(); out.count()];
    for (flat, index) in input.indices().enumerate() {
        let position: usize =
            kept.iter().zip(out.strides()).map(|(&axis, stride)| index[axis] * stride).sum();
        groups[position].push(flat);
    }
    groups
}

pub(super) fn reduce(
    attrs: &ReduceAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let groups = groups(&Extents::of(input.shape())?, &attrs.axes);
    match widen(input.data()) {
        Elements::Float(values) => {
            let values: Vec<f64> = values.into_iter().map(f64::from).collect();
            let reduced: Vec<f64> = groups
                .iter()
                .map(|group| reduce_floats(attrs.op, group.iter().map(|&i| values[i])))
                .collect();
            round_to_output(&reduced, output)
        }
        Elements::Int(values) => {
            let (low, high) = int_range(output.dtype())?;
            let reduced: Vec<i64> = groups
                .iter()
                .map(|group| {
                    reduce_ints(
                        attrs.op,
                        &group.iter().map(|&i| values[i]).collect::<Vec<_>>(),
                        low,
                        high,
                    )
                })
                .collect();
            store_ints(&reduced, output.dtype())
        }
        Elements::Bool(values) => {
            let any = |group: &Vec<usize>| group.iter().any(|&i| values[i]);
            let all = |group: &Vec<usize>| group.iter().all(|&i| values[i]);
            let reduced = match attrs.op {
                ReduceOp::Max => groups.iter().map(any).collect(),
                ReduceOp::Min => groups.iter().map(all).collect(),
                _ => return Err(Malformed::DType(input.dtype()).into()),
            };
            store(Elements::Bool(reduced), output.dtype())
        }
    }
}

/// One group's float reduction, with ONNX's empty-set results: 0 for sums,
/// 1 for `Prod`, -inf for `Max`, `LogSum` and `LogSumExp`, +inf for `Min`,
/// and NaN for `Mean`.
fn reduce_floats(op: ReduceOp, values: impl Iterator<Item = f64> + Clone) -> f64 {
    match op {
        ReduceOp::Sum => values.sum(),
        ReduceOp::Mean => {
            let count = values.clone().count();
            values.sum::<f64>() / count_to_f64(count)
        }
        ReduceOp::Prod => values.product(),
        ReduceOp::Max => values.fold(f64::NEG_INFINITY, maximum),
        ReduceOp::Min => values.fold(f64::INFINITY, minimum),
        ReduceOp::SumSquare => values.map(|v| v * v).sum(),
        ReduceOp::L1 => values.map(f64::abs).sum(),
        ReduceOp::L2 => values.map(|v| v * v).sum::<f64>().sqrt(),
        ReduceOp::LogSum => values.sum::<f64>().ln(),
        ReduceOp::LogSumExp => log_sum_exp(values),
    }
}

/// `ln(sum(e^x))` shifted by the maximum, so large inputs cannot overflow.
fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// One group's integer reduction. Sums and products wrap, as in the input
/// type; `Max` and `Min` of nothing are the type's extremes; the rest go
/// through f64 and truncate, as ONNX's reference casts its float results.
fn reduce_ints(op: ReduceOp, values: &[i64], low: i64, high: i64) -> i64 {
    let wide = || values.iter().map(|&v| int_to_f64(v));
    match op {
        ReduceOp::Sum => values.iter().fold(0, |a, &b| a.wrapping_add(b)),
        ReduceOp::Prod => values.iter().fold(1, |a, &b| a.wrapping_mul(b)),
        ReduceOp::Max => values.iter().copied().max().unwrap_or(low),
        ReduceOp::Min => values.iter().copied().min().unwrap_or(high),
        ReduceOp::SumSquare => values.iter().fold(0, |a, &b| a.wrapping_add(b.wrapping_mul(b))),
        ReduceOp::L1 => values.iter().fold(0, |a, &b| a.wrapping_add(b.wrapping_abs())),
        ReduceOp::Mean | ReduceOp::L2 | ReduceOp::LogSum | ReduceOp::LogSumExp => {
            truncate(reduce_floats(op, wide().collect::<Vec<_>>().into_iter()))
        }
    }
}

/// `ArgMax` or `ArgMin`: the index along the axis of each group's extreme.
pub(super) fn arg_reduce(
    attrs: &ArgReduceAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let input = first(operands)?;
    let groups = groups(&Extents::of(input.shape())?, &[attrs.axis]);
    let indices: Vec<usize> = match widen(input.data()) {
        Elements::Float(values) => groups
            .iter()
            .map(|g| extreme(g.iter().map(|&i| values[i]), attrs, f32::is_nan))
            .collect(),
        Elements::Int(values) => {
            groups.iter().map(|g| extreme(g.iter().map(|&i| values[i]), attrs, |_| false)).collect()
        }
        Elements::Bool(_) => return Err(Malformed::DType(input.dtype()).into()),
    };
    let indices: Vec<i64> = indices.into_iter().map(position_to_i64).collect();
    store_ints(&indices, output.dtype())
}

/// The position of the extreme value; NaN wins, the first NaN found, as
/// NumPy's `argmax` and `argmin`.
fn extreme<T: PartialOrd + Copy>(
    values: impl Iterator<Item = T>,
    attrs: &ArgReduceAttrs,
    is_nan: fn(T) -> bool,
) -> usize {
    let mut best: Option<(usize, T)> = None;
    for (position, value) in values.enumerate() {
        let replaces = match best {
            None => true,
            Some((_, current)) if is_nan(current) => false,
            Some(_) if is_nan(value) => true,
            Some((_, current)) => match (attrs.kind, attrs.tie) {
                (ArgKind::Max, TieBreak::First) => value > current,
                (ArgKind::Max, TieBreak::Last) => value >= current,
                (ArgKind::Min, TieBreak::First) => value < current,
                (ArgKind::Min, TieBreak::Last) => value <= current,
            },
        };
        if replaces {
            best = Some((position, value));
        }
    }
    best.map_or(0, |(position, _)| position)
}
