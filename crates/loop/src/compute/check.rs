//! Type-checks a compute definition's outputs: element types, loads, index
//! variables, the scope of the one reduction axis set, and that every load
//! stays within its tensor over the iteration domain.

use super::compute_def::{IndexVar, TensorParam};
use super::error::ComputeError;
use super::iter_var::{IterKind, IterVar, IterVarId};
use super::tensor_expr::{TensorExpr, TensorParamId};
use super::types;
use crate::expr::{ExprArena, ExprId, Interval, VarId, VarRanges};
use ptxc_core::{Axis, DType, Dim, IndexVec, Rank};
use std::collections::HashSet;

/// Checks outputs one by one, remembering which reduction axes are taken.
pub(super) struct Checker<'a> {
    exprs: &'a ExprArena,
    vars: &'a IndexVec<VarId, IndexVar>,
    iter_vars: &'a IndexVec<IterVarId, IterVar>,
    inputs: &'a IndexVec<TensorParamId, TensorParam>,
    reduced: HashSet<IterVarId>,
    ranges: VarRanges,
}

impl<'a> Checker<'a> {
    pub(super) fn new(
        exprs: &'a ExprArena,
        vars: &'a IndexVec<VarId, IndexVar>,
        iter_vars: &'a IndexVec<IterVarId, IterVar>,
        inputs: &'a IndexVec<TensorParamId, TensorParam>,
    ) -> Self {
        let mut ranges = VarRanges::new();
        for (var, definition) in vars.iter_enumerated() {
            let IndexVar::Iter(axis) = definition else { continue };
            let last = iter_vars.get(*axis).and_then(|iter| iter.extent.get().checked_sub(1));
            if let Some(range) = last
                .and_then(|last| i64::try_from(last).ok())
                .and_then(|last| Interval::new(0, last))
            {
                ranges.insert(var, range);
            }
        }
        Self { exprs, vars, iter_vars, inputs, reduced: HashSet::new(), ranges }
    }

    /// Whether the outputs read a reduction.
    pub(super) fn has_reduction(&self) -> bool {
        !self.reduced.is_empty()
    }

    /// The element type of an output's expression.
    pub(super) fn output(&mut self, expr: &TensorExpr) -> Result<DType, ComputeError> {
        self.dtype(expr, &[])
    }

    /// Fails if a reduction axis was never reduced over.
    pub(super) fn every_reduce_axis_used(&self) -> Result<(), ComputeError> {
        let unused = self
            .iter_vars
            .iter_enumerated()
            .find(|(id, var)| var.kind == IterKind::Reduce && !self.reduced.contains(id));
        match unused {
            Some((_, var)) => Err(ComputeError::UnusedReduceAxis(var.name.clone())),
            None => Ok(()),
        }
    }

    /// `expr`'s element type, where `scope` lists the reduction axes the
    /// enclosing reductions bind.
    fn dtype(&mut self, expr: &TensorExpr, scope: &[IterVarId]) -> Result<DType, ComputeError> {
        match expr {
            TensorExpr::Load { tensor, index } => {
                self.check_index(*tensor, index, scope)?;
                Ok(self.input(*tensor)?.dtype)
            }
            TensorExpr::Const(scalar) => Ok(scalar.dtype()),
            TensorExpr::Unary(op, value) => types::unary(*op, self.dtype(value, scope)?),
            TensorExpr::Binary(op, lhs, rhs) => {
                types::binary(*op, self.dtype(lhs, scope)?, self.dtype(rhs, scope)?)
            }
            TensorExpr::Cast { to, value, .. } => self.dtype(value, scope).map(|_| *to),
            TensorExpr::Select { condition, then, otherwise } => {
                let condition = self.dtype(condition, scope)?;
                if condition != DType::Bool {
                    return Err(ComputeError::UnsupportedDType { op: "select", dtype: condition });
                }
                let (then, otherwise) = (self.dtype(then, scope)?, self.dtype(otherwise, scope)?);
                if then == otherwise {
                    Ok(then)
                } else {
                    Err(ComputeError::MixedDTypes { op: "select", lhs: then, rhs: otherwise })
                }
            }
            TensorExpr::Reduce { op, axes, body } => {
                let inner = self.bind_axes(axes, scope)?;
                types::reduce(*op, self.dtype(body, &inner)?)
            }
            TensorExpr::ArgReduce { kind, axis, body, .. } => {
                let inner = self.bind_axes(&[*axis], scope)?;
                let dtype = self.dtype(body, &inner)?;
                if types::arg_comparable(dtype) {
                    Ok(DType::I64)
                } else {
                    Err(ComputeError::UnsupportedDType { op: kind.name(), dtype })
                }
            }
        }
    }

    /// `scope` with `axes` added, after checking this is the definition's
    /// only reduction and each axis is a reduction axis, listed once.
    fn bind_axes(
        &mut self,
        axes: &[IterVarId],
        scope: &[IterVarId],
    ) -> Result<Vec<IterVarId>, ComputeError> {
        if axes.is_empty() {
            return Err(ComputeError::EmptyReduce);
        }
        if self.has_reduction() {
            return Err(ComputeError::MultipleReductions);
        }
        for &axis in axes {
            let var = self.iter_vars.get(axis).ok_or(ComputeError::UnknownAxis(axis))?;
            if var.kind != IterKind::Reduce {
                return Err(ComputeError::NotAReduceAxis(var.name.clone()));
            }
            if !self.reduced.insert(axis) {
                return Err(ComputeError::ReduceAxisReused(var.name.clone()));
            }
        }
        Ok(scope.iter().chain(axes).copied().collect())
    }

    fn input(&self, tensor: TensorParamId) -> Result<&'a TensorParam, ComputeError> {
        self.inputs.get(tensor).ok_or(ComputeError::UnknownTensor(tensor))
    }

    /// Checks a load of `tensor` at `index`: one index per axis, each
    /// reading only variables that are defined and in scope.
    fn check_index(
        &self,
        tensor: TensorParamId,
        index: &[ExprId],
        scope: &[IterVarId],
    ) -> Result<(), ComputeError> {
        let param = self.input(tensor)?;
        let actual = Rank::new(index.len());
        if param.shape.rank() != actual {
            return Err(ComputeError::IndexArity {
                tensor: param.name.clone(),
                expected: param.shape.rank(),
                actual,
            });
        }
        for (position, (&expr, &extent)) in index.iter().zip(param.shape.dims()).enumerate() {
            for var in self.exprs.vars(expr) {
                self.check_var(var, scope)?;
            }
            if !self.stays_within(expr, extent) {
                let axis = Axis::new(position);
                return Err(ComputeError::IndexOutOfBounds {
                    tensor: param.name.clone(),
                    axis,
                    extent,
                });
            }
        }
        Ok(())
    }

    /// Whether `expr` stays in `[0, extent)` over the iteration domain. An
    /// index read from a tensor is checked when it is read, not here; an
    /// empty axis makes the domain empty, so nothing is read at all.
    fn stays_within(&self, expr: ExprId, extent: Dim) -> bool {
        let read = self.exprs.vars(expr);
        let vacuous = read.iter().any(|&var| match self.vars.get(var) {
            Some(IndexVar::Loaded { .. }) => true,
            Some(IndexVar::Iter(axis)) => {
                self.iter_vars.get(*axis).is_some_and(|iter| iter.extent.get() == 0)
            }
            None => false,
        });
        if vacuous {
            return true;
        }
        let last = extent.get().checked_sub(1).and_then(|last| i64::try_from(last).ok());
        let bounds = self.exprs.bounds(expr, &self.ranges);
        bounds.zip(last).is_some_and(|(bounds, last)| bounds.min() >= 0 && bounds.max() <= last)
    }

    /// Checks that `var` may be read where `scope` is bound.
    fn check_var(&self, var: VarId, scope: &[IterVarId]) -> Result<(), ComputeError> {
        match self.vars.get(var).ok_or(ComputeError::UnknownVar(var))? {
            IndexVar::Iter(axis) => {
                let iter = self.iter_vars.get(*axis).ok_or(ComputeError::UnknownAxis(*axis))?;
                if iter.kind == IterKind::Reduce && !scope.contains(axis) {
                    return Err(ComputeError::ReduceAxisOutOfScope(iter.name.clone()));
                }
                Ok(())
            }
            IndexVar::Loaded { tensor, index, .. } => {
                let param = self.input(*tensor)?;
                if !matches!(param.dtype, DType::I32 | DType::I64) {
                    return Err(ComputeError::IndexNotInteger {
                        tensor: param.name.clone(),
                        dtype: param.dtype,
                    });
                }
                self.check_index(*tensor, index, scope)
            }
        }
    }
}
