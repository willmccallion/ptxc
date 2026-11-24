//! Expands the loops a schedule marks unrolled, as TVM's `UnrollLoop` pass
//! does: each becomes one copy of its body per value of its variable, with
//! the value substituted and every index simplified, so code generation
//! sees constant offsets where only the loop made them vary.

use crate::expr::{ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
use crate::stmt::{Condition, ForKind, FragmentTile, LoweredKernel, Stmt, TileAccess, ValueExpr};
use ptxc_core::Dim;

/// `kernel` with every unrolled loop expanded, and every guard that the
/// loops' and launch's ranges then decide resolved.
pub fn unrolled(kernel: &LoweredKernel) -> LoweredKernel {
    let mut unroller = Unroller {
        exprs: kernel.exprs.clone(),
        ranges: kernel.thread_ranges(),
        values: Vec::new(),
    };
    let body = unroller.stmt(&kernel.body);
    LoweredKernel { body, exprs: unroller.exprs, ..kernel.clone() }
}

struct Unroller {
    exprs: ExprArena,
    ranges: VarRanges,
    /// Each enclosing unrolled loop's variable and its value in this copy.
    values: Vec<(VarId, ExprId)>,
}

impl Unroller {
    fn stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Seq(stmts) => Stmt::Seq(stmts.iter().map(|stmt| self.stmt(stmt)).collect()),
            Stmt::For { var, extent, kind: ForKind::Unrolled, body } => {
                self.expand(*var, *extent, body)
            }
            Stmt::For { var, extent, kind, body } => {
                if let Some(range) = Interval::of_loop(*extent) {
                    self.ranges.insert(*var, range);
                }
                let body = Box::new(self.stmt(body));
                Stmt::For { var: *var, extent: *extent, kind: *kind, body }
            }
            Stmt::If { condition, then } => self.guarded(*condition, then),
            Stmt::Alloc { buffer, body } => {
                Stmt::Alloc { buffer: *buffer, body: Box::new(self.stmt(body)) }
            }
            Stmt::LoadIndex { var, buffer, index, body } => Stmt::LoadIndex {
                var: *var,
                buffer: *buffer,
                index: self.index(*index),
                body: Box::new(self.stmt(body)),
            },
            Stmt::Store { buffer, index, value } => {
                Stmt::Store { buffer: *buffer, index: self.index(*index), value: self.value(value) }
            }
            Stmt::Barrier => Stmt::Barrier,
            Stmt::AsyncCopy { target, target_index, source, source_index, run, bounds } => {
                Stmt::AsyncCopy {
                    target: *target,
                    target_index: self.index(*target_index),
                    source: *source,
                    source_index: self.index(*source_index),
                    run: *run,
                    bounds: bounds.iter().map(|&bound| self.condition(bound)).collect(),
                }
            }
            Stmt::AsyncCommit => Stmt::AsyncCommit,
            Stmt::AsyncWait { pending } => Stmt::AsyncWait { pending: *pending },
            Stmt::FragmentFill { fragment, value } => {
                Stmt::FragmentFill { fragment: self.tile(*fragment), value: *value }
            }
            Stmt::FragmentLoad { fragment, source, at } => {
                let at = TileAccess { origin: self.index(at.origin), ..*at };
                Stmt::FragmentLoad { fragment: self.tile(*fragment), source: *source, at }
            }
            Stmt::Mma { instruction, a, b, accumulator } => Stmt::Mma {
                instruction: *instruction,
                a: self.tile(*a),
                b: self.tile(*b),
                accumulator: self.tile(*accumulator),
            },
            Stmt::FragmentEach { fragment, row, column, body } => Stmt::FragmentEach {
                fragment: self.tile(*fragment),
                row: *row,
                column: *column,
                body: Box::new(self.stmt(body)),
            },
        }
    }

    /// One copy of `body` per value of `var` below `extent`, in order; none
    /// for an empty loop.
    fn expand(&mut self, var: VarId, extent: Dim, body: &Stmt) -> Stmt {
        let Some(values) = Interval::of_loop(extent) else { return Stmt::Seq(Vec::new()) };
        let mut copies = Vec::new();
        for value in values.min()..=values.max() {
            let constant = self.exprs.constant(value);
            self.values.push((var, constant));
            copies.push(self.stmt(body));
            let _copied = self.values.pop();
        }
        Stmt::Seq(copies)
    }

    /// `then` under `condition`, both with the current values substituted:
    /// `then` alone where the ranges show the condition always holds, and
    /// nothing where they show it never does.
    fn guarded(&mut self, condition: Condition, then: &Stmt) -> Stmt {
        let condition = self.condition(condition);
        let index = self.exprs.bounds(condition.index, &self.ranges);
        let bound = self.exprs.bounds(condition.bound, &self.ranges);
        if let (Some(index), Some(bound)) = (index, bound) {
            if index.max() < bound.min() {
                return self.stmt(then);
            }
            if index.min() >= bound.max() {
                return Stmt::Seq(Vec::new());
            }
        }
        Stmt::If { condition, then: Box::new(self.stmt(then)) }
    }

    /// `condition` with the current values substituted in both sides.
    fn condition(&mut self, condition: Condition) -> Condition {
        Condition { index: self.index(condition.index), bound: self.index(condition.bound) }
    }

    /// `fragment` with the current values substituted in its tile.
    fn tile(&mut self, fragment: FragmentTile) -> FragmentTile {
        FragmentTile { tile: self.index(fragment.tile), ..fragment }
    }

    /// `value` with the current values substituted in its indices.
    fn value(&mut self, value: &ValueExpr) -> ValueExpr {
        let mut sub = |operand: &ValueExpr| Box::new(self.value(operand));
        match value {
            ValueExpr::Load { buffer, index } => {
                ValueExpr::Load { buffer: *buffer, index: self.index(*index) }
            }
            ValueExpr::Const(scalar) => ValueExpr::Const(*scalar),
            ValueExpr::Element(fragment) => ValueExpr::Element(*fragment),
            ValueExpr::Index { index, dtype } => {
                ValueExpr::Index { index: self.index(*index), dtype: *dtype }
            }
            ValueExpr::Unary(op, operand) => ValueExpr::Unary(*op, sub(operand)),
            ValueExpr::Binary(op, lhs, rhs) => ValueExpr::Binary(*op, sub(lhs), sub(rhs)),
            ValueExpr::Cast { to, overflow, value } => {
                ValueExpr::Cast { to: *to, overflow: *overflow, value: sub(value) }
            }
            ValueExpr::Select { condition, then, otherwise } => ValueExpr::Select {
                condition: sub(condition),
                then: sub(then),
                otherwise: sub(otherwise),
            },
            ValueExpr::MulAdd { lhs, rhs, addend } => {
                ValueExpr::MulAdd { lhs: sub(lhs), rhs: sub(rhs), addend: sub(addend) }
            }
        }
    }

    /// `id` with the current values substituted and simplified; `id` itself
    /// if it reads none of them.
    fn index(&mut self, id: ExprId) -> ExprId {
        let reads = self.exprs.vars(id);
        let mut substituted = id;
        for &(var, value) in &self.values {
            if reads.contains(&var) {
                substituted = self.exprs.substitute(substituted, var, value);
            }
        }
        if substituted == id {
            return id;
        }
        Simplifier::new(&self.ranges).simplify(&mut self.exprs, substituted)
    }
}

#[cfg(test)]
mod tests {
    use super::unrolled;
    use crate::compute::TensorName;
    use crate::expr::Expr;
    use crate::stmt::{
        Buffer, ForKind, FragmentShape, FragmentTile, KernelVar, MemScope, Stmt, TileAccess,
        TileOrder,
    };
    use crate::test_kernels::warp_kernel;
    use ptxc_core::{DType, Dim, ElementCount};

    #[test]
    fn an_unrolled_loop_puts_each_copy_of_a_tile_load_at_its_own_origin_and_tile() {
        let mut kernel = warp_kernel();
        let mut buffer = |name: &str, scope| {
            let (name, size) = (TensorName::new(name), ElementCount::new(512));
            kernel.buffers.push(Buffer { name, dtype: DType::F16, scope, size })
        };
        let tile = buffer("tile", MemScope::Shared);
        let shape = FragmentShape { rows: Dim::new(16), columns: Dim::new(16) };
        let fragment = buffer("fragment", MemScope::Fragment(shape));
        let step = kernel.vars.push(KernelVar::Loop(crate::compute::VarName::new("step")));
        let (step_expr, tile_size) = (kernel.exprs.var(step), kernel.exprs.constant(256));
        let origin = kernel.exprs.mul(step_expr, tile_size);
        let at = TileAccess { origin, stride: Dim::new(16), order: TileOrder::RowMajor };
        let each_step = FragmentTile { buffer: fragment, tile: step_expr };
        let load = Stmt::FragmentLoad { fragment: each_step, source: tile, at };
        kernel.body = Stmt::For {
            var: step,
            extent: Dim::new(2),
            kind: ForKind::Unrolled,
            body: Box::new(load),
        };

        let expanded = unrolled(&kernel);

        let Stmt::Seq(copies) = expanded.body() else { panic!("{:?}", expanded.body()) };
        let places: Vec<[Expr; 2]> = copies
            .iter()
            .map(|copy| match copy {
                Stmt::FragmentLoad { fragment, at, .. } => {
                    [at.origin, fragment.tile].map(|id| expanded.exprs().get(id))
                }
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(places, [[Expr::Const(0); 2], [Expr::Const(256), Expr::Const(1)]]);
    }
}
