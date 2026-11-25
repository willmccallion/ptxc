//! A padded load left unstaged: the load clamped into its tensor, and the
//! padding selected wherever the index lies outside it, as Halide lowers
//! `BoundaryConditions::constant_exterior`.

use super::context::Lowerer;
use crate::compute::{BinaryOp, Scalar, TensorParamId};
use crate::expr::{Expr, ExprId};
use crate::stmt::ValueExpr;
use ptxc_core::{DType, Dim};

impl Lowerer<'_> {
    /// `tensor` at compute-level `index`, or `padding` where the index lies
    /// outside it; only `padding` where the tensor is empty.
    pub(super) fn padded_load(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
        padding: Scalar,
    ) -> ValueExpr {
        let dims = self.compute.inputs()[tensor].shape.dims().to_vec();
        if dims.iter().any(|dim| dim.get() == 0) {
            return ValueExpr::Const(padding);
        }
        let (mut clamped, mut inside) = (Vec::new(), Vec::new());
        for (&part, &dim) in index.iter().zip(&dims) {
            let translated = self.translate(part);
            let part = self.simplify(translated);
            for condition in self.open_bounds(part, dim) {
                let [index, bound] = [condition.index, condition.bound].map(|id| self.integer(id));
                inside.push(ValueExpr::Binary(BinaryOp::Less, Box::new(index), Box::new(bound)));
            }
            clamped.push(self.clamped(part, dim));
        }
        let position = self.linearize(&clamped, &dims);
        let load = ValueExpr::Load { buffer: self.input_buffer(tensor), index: position };
        let both = |all, next| ValueExpr::Binary(BinaryOp::And, Box::new(all), Box::new(next));
        match inside.into_iter().reduce(both) {
            Some(condition) => ValueExpr::Select {
                condition: Box::new(condition),
                then: Box::new(load),
                otherwise: Box::new(ValueExpr::Const(padding)),
            },
            None => load,
        }
    }

    /// `part` clamped into `[0, dim)`, for a dim of at least 1, simplified.
    fn clamped(&mut self, part: ExprId, dim: Dim) -> ExprId {
        let zero = self.constant(0);
        let last = self.extent(Dim::new(dim.get() - 1));
        let above = self.exprs.max(part, zero);
        let clamped = self.exprs.min(above, last);
        self.simplify(clamped)
    }

    /// Index `id` as an `i64` value: a constant where it is one.
    fn integer(&self, id: ExprId) -> ValueExpr {
        match self.exprs.get(id) {
            Expr::Const(value) => ValueExpr::Const(Scalar::i64(value)),
            _ => ValueExpr::Index { index: id, dtype: DType::I64 },
        }
    }
}
