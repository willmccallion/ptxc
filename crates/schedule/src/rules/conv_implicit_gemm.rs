//! Convolution as implicit GEMM, as cuDNN's and CUTLASS's kernels compute
//! it (D70): a sum over several axes of two loads multiplied, the input
//! read through its padding, whose axes group into a matrix product's. The
//! output positions only the left operand reads are its rows, the output
//! channels only the right reads its columns, and the taps and input
//! channels both read its depth. Each operand's tile is reindexed as its
//! matrix, each group fused into one loop, and the loops tiled as a matrix
//! product's.

use super::matmul_loops::{GemmLoops, own_loop};
use super::matmul_shape::{read_once, uncast};
use ptxc_loop::{
    BinaryOp, ComputeDef, ExprId, IndexVar, IterKind, IterVarId, LoopId, LoopKind, Open, Primitive,
    ReduceOp, Schedule, ScheduleError, TensorExpr, TensorParamId,
};

/// A sum of products whose axes group into a matrix product's.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ImplicitGemm {
    /// The spatial axes only the left operand reads, outermost first.
    pub(crate) rows: Vec<IterVarId>,
    /// The spatial axes only the right operand reads, outermost first.
    pub(crate) columns: Vec<IterVarId>,
    /// The summed axes, each read by both operands, the one the left
    /// operand's innermost index part reads last.
    pub(crate) depth: Vec<IterVarId>,
    /// The left operand and its index.
    pub(crate) lhs: (TensorParamId, Vec<ExprId>),
    /// The right operand and its index.
    pub(crate) rhs: (TensorParamId, Vec<ExprId>),
}

/// `compute` as an implicit GEMM: one output holding a sum of the product
/// of two loads, each read nowhere else; each summed axis read by both; at least one spatial axis read by each alone;
/// no gathers; and no empty axis, which the reduction rule keeps serial.
pub(crate) fn implicit_gemm(compute: &ComputeDef) -> Option<ImplicitGemm> {
    let [output] = compute.outputs() else { return None };
    let TensorExpr::Reduce { op: ReduceOp::Sum, axes, body } = output.expr.reduction()? else {
        return None;
    };
    let TensorExpr::Binary(BinaryOp::Mul, lhs, rhs) = uncast(body) else { return None };
    let (lhs, rhs) = (operand(uncast(lhs))?, operand(uncast(rhs))?);
    let (lhs_axes, rhs_axes) = (axes_read(compute, &lhs.1)?, axes_read(compute, &rhs.1)?);
    let only = |reads: &[IterVarId], other: &[IterVarId]| -> Vec<IterVarId> {
        let spatial = |axis: &IterVarId| compute.iter_vars()[*axis].kind == IterKind::Spatial;
        reads.iter().copied().filter(|axis| spatial(axis) && !other.contains(axis)).collect()
    };
    let (rows, columns) = (only(&lhs_axes, &rhs_axes), only(&rhs_axes, &lhs_axes));
    let summed_by_both = axes.iter().all(|axis| lhs_axes.contains(axis) && rhs_axes.contains(axis));
    let has_work = compute.iter_vars().iter().all(|axis| axis.extent.get() > 0);
    let read_once = [lhs.0, rhs.0].iter().all(|&tensor| read_once(&output.expr, body, tensor));
    let found = !rows.is_empty() && !columns.is_empty() && summed_by_both;
    if !(found && has_work && read_once && lhs.0 != rhs.0) {
        return None;
    }
    let depth = innermost_last(compute, &lhs.1, axes);
    Some(ImplicitGemm { rows, columns, depth, lhs, rhs })
}

/// The tensor `expr` loads and its index, if it is a load.
fn operand(expr: &TensorExpr) -> Option<(TensorParamId, Vec<ExprId>)> {
    let TensorExpr::Load { tensor, index, .. } = expr else { return None };
    Some((*tensor, index.clone()))
}

/// The axes `index` reads, in id order, if it reads no gathered index.
fn axes_read(compute: &ComputeDef, index: &[ExprId]) -> Option<Vec<IterVarId>> {
    let mut axes = Vec::new();
    for var in index.iter().flat_map(|&part| compute.exprs().vars(part)) {
        let IndexVar::Iter(axis) = compute.vars()[var] else { return None };
        if !axes.contains(&axis) {
            axes.push(axis);
        }
    }
    axes.sort_unstable();
    Some(axes)
}

/// `axes` ordered by the last part of `index` that reads each, so the axis
/// the innermost part reads is last and the depth steps along the input's
/// rows where it can, as channels do in NHWC.
fn innermost_last(compute: &ComputeDef, index: &[ExprId], axes: &[IterVarId]) -> Vec<IterVarId> {
    let position = |axis: IterVarId| {
        let var = compute.vars().iter_enumerated().find(|(_, var)| **var == IndexVar::Iter(axis));
        let reads =
            |part: ExprId| var.is_some_and(|(var, _)| compute.exprs().vars(part).contains(&var));
        index.iter().rposition(|&part| reads(part))
    };
    let mut ordered = axes.to_vec();
    ordered.sort_by_key(|&axis| (position(axis), axis));
    ordered
}

/// `gemm`'s operands reindexed as its matrices and each of its groups
/// fused into one loop of the default nest, the spatial loops besides
/// outside them.
pub(crate) fn gemm_loops(
    schedule: &mut Schedule<Open>,
    gemm: &ImplicitGemm,
) -> Result<GemmLoops, ScheduleError> {
    for ((tensor, index), others) in [(&gemm.lhs, &gemm.rows), (&gemm.rhs, &gemm.columns)] {
        let groups = matrix(schedule.compute(), index, others, &gemm.depth);
        schedule.apply(Primitive::Reindex { tensor: *tensor, groups })?;
    }
    let own = |schedule: &Schedule<Open>, axes: &[IterVarId]| {
        axes.iter().map(|&axis| own_loop(schedule, axis)).collect::<Result<Vec<_>, _>>()
    };
    let (rows, columns, depth) =
        (own(schedule, &gemm.rows)?, own(schedule, &gemm.columns)?, own(schedule, &gemm.depth)?);
    let tree = schedule.loops();
    let grouped = |id: &LoopId| rows.contains(id) || columns.contains(id);
    let batch: Vec<LoopId> = tree
        .nest()
        .iter()
        .copied()
        .filter(|id| tree.loops()[*id].kind == LoopKind::Spatial && !grouped(id))
        .collect();
    let order = [&batch[..], &rows, &columns, &depth].concat();
    schedule.apply(Primitive::Reorder { order })?;
    Ok(GemmLoops {
        row: one_loop(schedule, &rows)?,
        column: one_loop(schedule, &columns)?,
        depth: one_loop(schedule, &depth)?,
        batch,
        operands: [gemm.lhs.0, gemm.rhs.0],
    })
}

/// The groups that reindex an operand read at `index` as its matrix:
/// `others` by `depth`, or `depth` by `others` where its innermost index
/// part reads none of `depth`, so the tile's rows lie along the input's.
fn matrix(
    compute: &ComputeDef,
    index: &[ExprId],
    others: &[IterVarId],
    depth: &[IterVarId],
) -> Vec<Vec<IterVarId>> {
    let innermost = index.last().and_then(|&part| axes_read(compute, &[part])).unwrap_or_default();
    let along_depth = innermost.iter().any(|axis| depth.contains(axis));
    if along_depth {
        vec![others.to_vec(), depth.to_vec()]
    } else {
        vec![depth.to_vec(), others.to_vec()]
    }
}

/// `loops`, adjacent in the nest, as one loop.
fn one_loop(schedule: &mut Schedule<Open>, loops: &[LoopId]) -> Result<LoopId, ScheduleError> {
    match loops {
        [only] => Ok(*only),
        _ => schedule.fuse(loops.to_vec()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{ImplicitGemm, implicit_gemm};
    use ptxc_core::{DType, Dim, Shape};
    use ptxc_loop::{
        BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ReduceOp, TensorExpr, TensorName,
        VarName,
    };
    use ptxc_testkit::convolution::{Channels, Conv, axes, conv, w, x};

    const PADDED: Conv = Conv {
        batch: 2,
        size: 5,
        channels: 3,
        features: 4,
        kernel: 3,
        stride: 1,
        dilation: 1,
        pad: 1,
        groups: 1,
    };

    /// `conv`'s groups as `implicit_gemm` finds them, by axis name.
    fn found(geometry: Conv, channels: Channels) -> Option<[Vec<String>; 3]> {
        let compute = conv(geometry, channels);
        let names = |found: &[ptxc_loop::IterVarId]| -> Vec<String> {
            found.iter().map(|&axis| compute.iter_vars()[axis].name.to_string()).collect()
        };
        let ImplicitGemm { rows, columns, depth, lhs, rhs } = implicit_gemm(&compute)?;
        assert_eq!((lhs.0, rhs.0), (x(), w()));
        Some([names(&rows), names(&columns), names(&depth)])
    }

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|&name| name.to_owned()).collect()
    }

    #[test]
    fn a_convolution_groups_positions_channels_and_taps_innermost_read_last() {
        let positions = strings(&["n", "p", "q"]);

        let first = found(PADDED, Channels::First);
        let last = found(PADDED, Channels::Last);

        let expected = |depth: &[&str]| Some([positions.clone(), strings(&["m"]), strings(depth)]);
        assert_eq!(first, expected(&["c", "r", "s"]));
        assert_eq!(last, expected(&["r", "s", "c"]));
    }

    #[test]
    fn a_grouped_convolution_is_not_a_matrix_product() {
        let grouped = Conv { groups: 2, ..PADDED };

        assert_eq!(found(grouped, Channels::First), None);
    }

    #[test]
    fn an_empty_convolution_is_left_to_the_reduction_rule() {
        let compute = conv(PADDED, Channels::Last);
        let empty = Conv { batch: 0, ..PADDED };

        assert!(implicit_gemm(&compute).is_some());
        assert_eq!(found(empty, Channels::Last), None);
    }

    /// `out[i, j] = sum over k, l of a[i, k, l] * b[k, j, l]`, or with
    /// `b` reading `b[k, j, 0]` unless `both_read_l`.
    fn product(both_read_l: bool) -> ComputeDef {
        let mut compute = ComputeBuilder::new(ComputeName::new("product"));
        let shape = |dims: &[u64]| Shape::new(dims.iter().copied().map(Dim::new).collect());
        let left = compute.input(TensorName::new("a"), DType::I32, shape(&[2, 3, 2]).unwrap());
        let right = compute.input(TensorName::new("b"), DType::I32, shape(&[3, 2, 2]).unwrap());
        let (_, row) = compute.spatial(VarName::new("i"), Dim::new(2));
        let (_, column) = compute.spatial(VarName::new("j"), Dim::new(2));
        let (k_axis, depth) = compute.reduce_axis(VarName::new("k"), Dim::new(3));
        let (l_axis, inner) = compute.reduce_axis(VarName::new("l"), Dim::new(2));
        let zero = compute.exprs().constant(0);
        let right_inner = if both_read_l { inner } else { zero };
        let load = |tensor, index| Box::new(TensorExpr::load(tensor, index));
        let body = TensorExpr::Binary(
            BinaryOp::Mul,
            load(left, vec![row, depth, inner]),
            load(right, vec![depth, column, right_inner]),
        );
        let axes = vec![k_axis, l_axis];
        let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes, body: Box::new(body) };
        compute.output(TensorName::new("out"), sum);
        compute.finish().unwrap()
    }

    #[test]
    fn a_product_over_several_axes_both_operands_read_is_found() {
        let both = product(true);

        let found = implicit_gemm(&both).unwrap();

        assert_eq!(found.depth, axes(&both, &["k", "l"]));
    }

    #[test]
    fn every_summed_axis_is_read_by_both_operands() {
        assert!(implicit_gemm(&product(false)).is_none());
    }
}
