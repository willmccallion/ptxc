//! Lowers a definition's reduction within each thread: accumulators over
//! the spatial loops inside the first reduction loop, initialised, updated
//! by the reduction loops, and written back, as TVM's
//! `decompose_reduction` does.

use super::combine::{Candidate, arg_start, combine, identity, position_start, replaces};
use super::context::Lowerer;
use crate::compute::TensorExpr;
use crate::expr::{ExprId, VarId};
use crate::schedule::{LoopId, LoopKind};
use crate::stmt::{BufferId, MemScope, Stmt, ValueExpr};
use ptxc_core::{DType, Dim, Idx};

/// Where a thread keeps a reduction's running result.
pub(super) struct Accumulator {
    /// The running value.
    pub(super) value: BufferId,
    /// The running position, for an arg-reduction.
    pub(super) position: Option<BufferId>,
    /// A one-element flag an arg-reduction's update computes once.
    pub(super) flag: Option<BufferId>,
    /// The element for the current spatial point.
    pub(super) index: ExprId,
}

impl Lowerer<'_> {
    /// The loops computing the one output, whose expression holds the
    /// reduction `reduction`.
    pub(super) fn reduction(&mut self, reduction: &TensorExpr) -> Stmt {
        let serial = self.serial_loops();
        let first = serial
            .iter()
            .position(|&id| self.tree.loops()[id].kind == LoopKind::Reduce)
            .unwrap_or(serial.len());
        let (outer, inner) = serial.split_at(first);
        let inner_spatial: Vec<LoopId> = inner
            .iter()
            .copied()
            .filter(|&id| self.tree.loops()[id].kind == LoopKind::Spatial)
            .collect();
        let accumulator = self.accumulator(reduction, &inner_spatial);
        let init = self.initialise(reduction, &accumulator);
        let init = self.wrap_loops(&inner_spatial, init);
        let update = self.update(reduction, &accumulator);
        let guards = self.tree.guards().to_vec();
        let update = self.wrap_guards(&guards, update);
        let update = self.wrap_loops(inner, update);
        let write_back = self.write_back(&accumulator);
        let write_back = self.wrap_loops(&inner_spatial, write_back);
        let body = Self::allocate(&accumulator, Stmt::Seq(vec![init, update, write_back]));
        self.wrap_loops(outer, body)
    }

    /// Local buffers with one element per point of `inner_spatial`.
    fn accumulator(&mut self, reduction: &TensorExpr, inner_spatial: &[LoopId]) -> Accumulator {
        let extents: Vec<Dim> =
            inner_spatial.iter().map(|&id| self.tree.loops()[id].extent).collect();
        let size = extents.iter().map(|dim| dim.get()).product::<u64>().max(1);
        let vars: Vec<ExprId> =
            inner_spatial.iter().map(|&id| self.exprs.var(VarId::new(id.index()))).collect();
        let index = self.linearize(&vars, &extents);
        let dtype = self.reduced_dtype(reduction);
        let value = self.scratch("acc", dtype, MemScope::Local, size);
        let (position, flag) = if matches!(reduction, TensorExpr::ArgReduce { .. }) {
            let position = self.scratch("acc_position", DType::I64, MemScope::Local, size);
            (Some(position), Some(self.scratch("replace", DType::Bool, MemScope::Local, 1)))
        } else {
            (None, None)
        };
        Accumulator { value, position, flag, index }
    }

    /// `body` with the accumulator's buffers allocated around it.
    pub(super) fn allocate(accumulator: &Accumulator, body: Stmt) -> Stmt {
        let buffers = [Some(accumulator.value), accumulator.position, accumulator.flag];
        buffers
            .into_iter()
            .flatten()
            .rev()
            .fold(body, |body, buffer| Stmt::Alloc { buffer, body: Box::new(body) })
    }

    /// The element type the reduction combines: its body's.
    pub(super) fn reduced_dtype(&self, reduction: &TensorExpr) -> DType {
        match reduction {
            TensorExpr::Reduce { body, .. } | TensorExpr::ArgReduce { body, .. } => {
                self.compute.dtype_of(body)
            }
            other => self.compute.dtype_of(other),
        }
    }

    /// Stores the reduction's starting value (and position).
    pub(super) fn initialise(&self, reduction: &TensorExpr, accumulator: &Accumulator) -> Stmt {
        let dtype = self.reduced_dtype(reduction);
        let (start, position) = match reduction {
            TensorExpr::Reduce { op, .. } => (identity(*op, dtype), None),
            TensorExpr::ArgReduce { kind, tie, .. } => {
                (arg_start(*kind, dtype), Some(position_start(*tie)))
            }
            _ => return Stmt::Seq(Vec::new()),
        };
        let mut stores = vec![Stmt::Store {
            buffer: accumulator.value,
            index: accumulator.index,
            value: ValueExpr::Const(start),
        }];
        if let (Some(buffer), Some(start)) = (accumulator.position, position) {
            stores.push(Stmt::Store {
                buffer,
                index: accumulator.index,
                value: ValueExpr::Const(start),
            });
        }
        Stmt::Seq(stores)
    }

    /// Folds the body's value at the current point into the accumulator.
    fn update(&mut self, reduction: &TensorExpr, accumulator: &Accumulator) -> Stmt {
        let unused = ValueExpr::Const(crate::compute::Scalar::bool(false));
        let current = ValueExpr::Load { buffer: accumulator.value, index: accumulator.index };
        let stores = match reduction {
            TensorExpr::Reduce { op, body, .. } => {
                let value = self.value(body, &unused);
                let dtype = value.dtype(&self.buffers);
                vec![Stmt::Store {
                    buffer: accumulator.value,
                    index: accumulator.index,
                    value: combine(*op, dtype, current, value),
                }]
            }
            TensorExpr::ArgReduce { kind, tie, axis, body } => {
                let value = self.value(body, &unused);
                let dtype = value.dtype(&self.buffers);
                let position =
                    ValueExpr::Index { index: self.tree.axis_index(*axis), dtype: DType::I64 };
                let challenger = Candidate { value, position };
                self.arg_update(*kind, *tie, dtype, &challenger, accumulator, accumulator.index)
            }
            _ => Vec::new(),
        };
        self.wrap_loaded(reduction, Stmt::Seq(stores))
    }

    /// Stores `challenger` into the accumulator element `index` where it
    /// replaces the best so far, deciding once through the flag.
    pub(super) fn arg_update(
        &mut self,
        kind: crate::compute::ArgKind,
        tie: crate::compute::TieBreak,
        dtype: DType,
        challenger: &Candidate,
        accumulator: &Accumulator,
        index: ExprId,
    ) -> Vec<Stmt> {
        let (Some(position_buffer), Some(flag)) = (accumulator.position, accumulator.flag) else {
            return Vec::new();
        };
        let zero = self.constant(0);
        let best = Candidate {
            value: ValueExpr::Load { buffer: accumulator.value, index },
            position: ValueExpr::Load { buffer: position_buffer, index },
        };
        let decided = ValueExpr::Load { buffer: flag, index: zero };
        let keep = |choice: &ValueExpr, current: &ValueExpr| ValueExpr::Select {
            condition: Box::new(decided.clone()),
            then: Box::new(choice.clone()),
            otherwise: Box::new(current.clone()),
        };
        vec![
            Stmt::Store {
                buffer: flag,
                index: zero,
                value: replaces(kind, tie, dtype, challenger, &best),
            },
            Stmt::Store {
                buffer: accumulator.value,
                index,
                value: keep(&challenger.value, &best.value),
            },
            Stmt::Store {
                buffer: position_buffer,
                index,
                value: keep(&challenger.position, &best.position),
            },
        ]
    }

    /// Stores the output from the finished accumulator, under the guards
    /// that do not read reduction loops.
    fn write_back(&mut self, accumulator: &Accumulator) -> Stmt {
        let result_buffer = accumulator.position.unwrap_or(accumulator.value);
        let result = ValueExpr::Load { buffer: result_buffer, index: accumulator.index };
        self.store_output(&result)
    }

    /// The output store with `result` for the reduction node, under the
    /// spatial guards and the gather indices its expression reads.
    pub(super) fn store_output(&mut self, result: &ValueExpr) -> Stmt {
        let output = self.compute.outputs()[0].clone();
        let value = self.value(&output.expr, result);
        let position = self.output_position();
        let store = Stmt::Store { buffer: self.outputs[0], index: position, value };
        let spatial_only = self.wrap_loaded_outside_reduction(&output.expr, store);
        let guards: Vec<_> = self
            .tree
            .guards()
            .iter()
            .copied()
            .filter(|guard| !self.reads_reduction(guard))
            .collect();
        self.wrap_guards(&guards, spatial_only)
    }
}
