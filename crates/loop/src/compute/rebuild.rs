//! Rebuilding one definition's expressions in another builder, as
//! `rfactor` and `inline` make definitions from others.

use super::error::ComputeError;
use super::iter_var::IterVarId;
use super::scalar::Scalar;
use super::tensor_expr::{TensorExpr, TensorParamId};
use crate::expr::ExprId;

/// Where a rebuild puts the source definition's index expressions, axes and
/// loads.
pub(super) trait Rebuild {
    /// `index`, an index expression of the source, in the new definition.
    fn index(&mut self, index: ExprId) -> Result<ExprId, ComputeError>;

    /// `axis`, an axis of the source, in the new definition.
    fn axis(&self, axis: IterVarId) -> Result<IterVarId, ComputeError>;

    /// What the source's read of `tensor` at `index`, `padding` outside it,
    /// becomes.
    fn load(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
        padding: Option<Scalar>,
    ) -> Result<TensorExpr, ComputeError>;

    /// What the source's `reduction` becomes, if not its own rebuild.
    fn reduction(&mut self, _reduction: &TensorExpr) -> Option<TensorExpr> {
        None
    }
}

/// `expr`, an expression of the source, rebuilt as `to` places it.
pub(super) fn rebuild(
    expr: &TensorExpr,
    to: &mut impl Rebuild,
) -> Result<TensorExpr, ComputeError> {
    Ok(match expr {
        TensorExpr::Load { tensor, index, padding } => return to.load(*tensor, index, *padding),
        TensorExpr::Const(scalar) => TensorExpr::Const(*scalar),
        TensorExpr::Index { index, dtype } => {
            TensorExpr::Index { index: to.index(*index)?, dtype: *dtype }
        }
        TensorExpr::Unary(op, value) => TensorExpr::Unary(*op, boxed(value, to)?),
        TensorExpr::Binary(op, lhs, rhs) => {
            TensorExpr::Binary(*op, boxed(lhs, to)?, boxed(rhs, to)?)
        }
        TensorExpr::Cast { to: dtype, overflow, value } => {
            TensorExpr::Cast { to: *dtype, overflow: *overflow, value: boxed(value, to)? }
        }
        TensorExpr::Select { condition, then, otherwise } => TensorExpr::Select {
            condition: boxed(condition, to)?,
            then: boxed(then, to)?,
            otherwise: boxed(otherwise, to)?,
        },
        TensorExpr::Reduce { op, axes, body } => {
            if let Some(reduction) = to.reduction(expr) {
                return Ok(reduction);
            }
            let axes = axes.iter().map(|&axis| to.axis(axis)).collect::<Result<_, _>>()?;
            TensorExpr::Reduce { op: *op, axes, body: boxed(body, to)? }
        }
        TensorExpr::ArgReduce { kind, tie, axis, body } => TensorExpr::ArgReduce {
            kind: *kind,
            tie: *tie,
            axis: to.axis(*axis)?,
            body: boxed(body, to)?,
        },
    })
}

fn boxed(value: &TensorExpr, to: &mut impl Rebuild) -> Result<Box<TensorExpr>, ComputeError> {
    rebuild(value, to).map(Box::new)
}
