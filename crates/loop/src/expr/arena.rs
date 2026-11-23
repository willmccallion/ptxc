//! `ExprArena`: hash-consed index expressions addressed by `ExprId`.

use super::{Expr, ExprId, VarId};
use ptxc_core::IndexVec;
use std::collections::HashMap;

/// Every index expression of a kernel, each stored once, so structural
/// equality is id equality (D4).
#[derive(Clone, Debug, Default)]
pub struct ExprArena {
    exprs: IndexVec<ExprId, Expr>,
    ids: HashMap<Expr, ExprId>,
}

impl ExprArena {
    /// An empty arena.
    pub fn new() -> Self {
        Self::default()
    }

    /// The id of `expr`, added if it is new.
    pub fn intern(&mut self, expr: Expr) -> ExprId {
        if let Some(&id) = self.ids.get(&expr) {
            return id;
        }
        let id = self.exprs.push(expr);
        let _new: Option<ExprId> = self.ids.insert(expr, id);
        id
    }

    /// The expression `id` names.
    pub fn get(&self, id: ExprId) -> Expr {
        self.exprs[id]
    }

    /// `var`.
    pub fn var(&mut self, var: VarId) -> ExprId {
        self.intern(Expr::Var(var))
    }

    /// The constant `value`.
    pub fn constant(&mut self, value: i64) -> ExprId {
        self.intern(Expr::Const(value))
    }

    /// `a + b`.
    pub fn add(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Add(a, b))
    }

    /// `a * b`.
    pub fn mul(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Mul(a, b))
    }

    /// `floordiv(a, b)`.
    pub fn floor_div(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::FloorDiv(a, b))
    }

    /// `floormod(a, b)`.
    pub fn floor_mod(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::FloorMod(a, b))
    }

    /// `min(a, b)`.
    pub fn min(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Min(a, b))
    }

    /// `max(a, b)`.
    pub fn max(&mut self, a: ExprId, b: ExprId) -> ExprId {
        self.intern(Expr::Max(a, b))
    }
}
