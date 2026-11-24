//! Expands the loops a schedule marks unrolled, as TVM's `UnrollLoop` pass
//! does: each becomes one copy of its body per value of its variable, with
//! the value substituted and every index simplified, so code generation
//! sees constant offsets where only the loop made them vary.

use crate::expr::{ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
use crate::stmt::{Condition, ForKind, KernelVar, LoweredKernel, Stmt, ValueExpr};
use ptxc_core::Dim;

/// `kernel` with every unrolled loop expanded, and every guard that the
/// loops' and launch's ranges then decide resolved.
pub fn unrolled(kernel: &LoweredKernel) -> LoweredKernel {
    let mut unroller =
        Unroller { exprs: kernel.exprs.clone(), ranges: thread_ranges(kernel), values: Vec::new() };
    let body = unroller.stmt(&kernel.body);
    LoweredKernel { body, exprs: unroller.exprs, ..kernel.clone() }
}

/// The range of each thread index, from the launch.
fn thread_ranges(kernel: &LoweredKernel) -> VarRanges {
    let mut ranges = VarRanges::new();
    for (var, definition) in kernel.vars.iter_enumerated() {
        if let KernelVar::Thread { axis, .. } = definition
            && let Some(range) = kernel.launch.range(*axis)
        {
            ranges.insert(var, range);
        }
    }
    ranges
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
        let condition =
            Condition { index: self.index(condition.index), bound: self.index(condition.bound) };
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

    /// `value` with the current values substituted in its indices.
    fn value(&mut self, value: &ValueExpr) -> ValueExpr {
        let mut sub = |operand: &ValueExpr| Box::new(self.value(operand));
        match value {
            ValueExpr::Load { buffer, index } => {
                ValueExpr::Load { buffer: *buffer, index: self.index(*index) }
            }
            ValueExpr::Const(scalar) => ValueExpr::Const(*scalar),
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
