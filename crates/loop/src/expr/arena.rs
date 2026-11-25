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

    /// The id of `expr`, if the arena holds it.
    pub(super) fn find(&self, expr: Expr) -> Option<ExprId> {
        self.ids.get(&expr).copied()
    }

    /// The variables `id` reads, each once, in first-read order.
    pub fn vars(&self, id: ExprId) -> Vec<VarId> {
        let mut found = Vec::new();
        let mut pending = vec![id];
        while let Some(next) = pending.pop() {
            match self.get(next) {
                Expr::Var(var) if !found.contains(&var) => found.push(var),
                Expr::Var(_) | Expr::Const(_) => {}
                Expr::Add(a, b)
                | Expr::Mul(a, b)
                | Expr::FloorDiv(a, b)
                | Expr::FloorMod(a, b)
                | Expr::Min(a, b)
                | Expr::Max(a, b) => pending.extend([b, a]),
            }
        }
        found
    }

    /// `id` with every read of `var` replaced by `replacement`.
    pub(crate) fn substitute(&mut self, id: ExprId, var: VarId, replacement: ExprId) -> ExprId {
        let sub = |arena: &mut Self, operand| arena.substitute(operand, var, replacement);
        match self.get(id) {
            Expr::Var(read) if read == var => replacement,
            Expr::Var(_) | Expr::Const(_) => id,
            Expr::Add(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.add(a, b)
            }
            Expr::Mul(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.mul(a, b)
            }
            Expr::FloorDiv(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.floor_div(a, b)
            }
            Expr::FloorMod(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.floor_mod(a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.min(a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = (sub(self, a), sub(self, b));
                self.max(a, b)
            }
        }
    }

    /// `id`, an expression of `from`, built in this arena with each variable
    /// it reads replaced by `var`'s expression for it; or the first variable
    /// `var` has none for.
    pub(crate) fn import(
        &mut self,
        from: &Self,
        id: ExprId,
        var: &impl Fn(VarId) -> Option<ExprId>,
    ) -> Result<ExprId, VarId> {
        let pair = |this: &mut Self, a, b| -> Result<_, VarId> {
            Ok((this.import(from, a, var)?, this.import(from, b, var)?))
        };
        Ok(match from.get(id) {
            Expr::Var(read) => var(read).ok_or(read)?,
            Expr::Const(value) => self.constant(value),
            Expr::Add(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.add(a, b)
            }
            Expr::Mul(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.mul(a, b)
            }
            Expr::FloorDiv(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.floor_div(a, b)
            }
            Expr::FloorMod(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.floor_mod(a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.min(a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = pair(self, a, b)?;
                self.max(a, b)
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ptxc_core::Idx;

    #[test]
    fn variables_are_listed_once_in_first_read_order() {
        let mut arena = ExprArena::new();
        let (x, y) = (arena.var(VarId::new(0)), arena.var(VarId::new(1)));
        let product = arena.mul(x, y);
        let sum = arena.add(x, product);

        assert_eq!(arena.vars(sum), [VarId::new(0), VarId::new(1)]);
    }

    #[test]
    fn an_imported_expression_reads_the_variables_it_is_given() {
        let mut from = ExprArena::new();
        let (x, y) = (from.var(VarId::new(0)), from.var(VarId::new(1)));
        let three = from.constant(3);
        let scaled = from.mul(x, three);
        let parts = [from.add(scaled, y), from.floor_div(x, y), from.floor_mod(x, y)];
        let parts = [parts[0], parts[1], parts[2], from.min(x, y), from.max(x, y)];
        let mut to = ExprArena::new();
        let (a, b) = (to.var(VarId::new(7)), to.var(VarId::new(8)));
        let renamed = |var: VarId| [a, b].get(var.index()).copied();

        let imported = parts.map(|part| to.import(&from, part, &renamed).unwrap());

        let three = to.constant(3);
        let scaled = to.mul(a, three);
        let expected = [to.add(scaled, b), to.floor_div(a, b), to.floor_mod(a, b)];
        assert_eq!(imported, [expected[0], expected[1], expected[2], to.min(a, b), to.max(a, b)]);
    }

    #[test]
    fn an_import_names_the_variable_it_has_no_expression_for() {
        let mut from = ExprArena::new();
        let (x, y) = (from.var(VarId::new(0)), from.var(VarId::new(1)));
        let sum = from.add(x, y);
        let mut to = ExprArena::new();
        let only_x = |var: VarId| (var == VarId::new(0)).then_some(ExprId::new(0));

        assert_eq!(to.import(&from, sum, &only_x), Err(VarId::new(1)));
    }
}
