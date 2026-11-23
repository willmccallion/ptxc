//! Shared helpers for schedule tests: definitions to schedule, and the
//! iteration points a scheduled nest visits.

use crate::compute_build::{load, shape};
use ptxc_core::{DType, Dim, Idx};
use ptxc_loop::{
    ComputeBuilder, ComputeDef, ComputeName, LoopId, LoopTree, Open, ReduceOp, Schedule,
    TensorExpr, TensorName, VarName,
};
use std::collections::HashMap;
use std::sync::Arc;

/// `out[i, j] = sum[k](x[i, j, k])` over the given extents.
pub(crate) fn reduction(extents: [u64; 3]) -> Arc<ComputeDef> {
    let [i_extent, j_extent, k_extent] = extents;
    let mut compute = ComputeBuilder::new(ComputeName::new("reduction"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&extents));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(i_extent));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(j_extent));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(k_extent));
    let sum = TensorExpr::Reduce {
        op: ReduceOp::Sum,
        axes: vec![k_axis],
        body: Box::new(load(x, &[i, j, k])),
    };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

/// The default schedule of `compute`, and its loops in nest order.
pub(crate) fn open(compute: &Arc<ComputeDef>) -> (Schedule<Open>, Vec<LoopId>) {
    let schedule = Schedule::new(Arc::clone(compute));
    let loops = schedule.loops().nest().to_vec();
    (schedule, loops)
}

/// The axis-index tuples a scheduled nest visits, in visiting order,
/// skipping iterations a guard rejects.
pub(crate) fn visited(schedule: &Schedule<Open>) -> Vec<Vec<i64>> {
    let tree: &LoopTree = schedule.loops();
    let extents: Vec<u64> = tree.nest().iter().map(|&id| tree.loops()[id].extent.get()).collect();
    let mut points = Vec::new();
    let mut counters = vec![0_u64; extents.len()];
    if extents.contains(&0) {
        return points;
    }
    loop {
        let values: HashMap<usize, i64> = tree
            .nest()
            .iter()
            .zip(&counters)
            .map(|(id, &value)| (id.index(), i64::try_from(value).unwrap()))
            .collect();
        let lookup = |var: ptxc_loop::VarId| values.get(&var.index()).copied();
        let inside = tree.guards().iter().all(|guard| {
            tree.exprs().evaluate(guard.index, &lookup).unwrap()
                < i64::try_from(guard.extent.get()).unwrap()
        });
        if inside {
            let axes = schedule.compute().iter_vars().ids();
            points.push(
                axes.map(|axis| tree.exprs().evaluate(tree.axis_index(axis), &lookup).unwrap())
                    .collect(),
            );
        }
        if !advance(&mut counters, &extents) {
            return points;
        }
    }
}

/// Steps an odometer whose last digit turns fastest; false once it wraps.
fn advance(counters: &mut [u64], extents: &[u64]) -> bool {
    for position in (0..counters.len()).rev() {
        counters[position] += 1;
        if counters[position] < extents[position] {
            return true;
        }
        counters[position] = 0;
    }
    false
}

/// Every point of the iteration domain, once each, in row-major order.
pub(crate) fn domain(extents: &[u64]) -> Vec<Vec<i64>> {
    let mut points = Vec::new();
    let mut counters = vec![0_u64; extents.len()];
    if extents.contains(&0) {
        return points;
    }
    loop {
        points.push(counters.iter().map(|&value| i64::try_from(value).unwrap()).collect());
        if !advance(&mut counters, extents) {
            return points;
        }
    }
}
