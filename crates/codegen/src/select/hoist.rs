//! The index expressions a kernel computes once, at its start: every
//! largest one that divides and reads only grid and block indices, which
//! no loop or fragment element changes, such as the block's place in a
//! grid fused onto `blockIdx.x`. Value numbering then reuses their
//! registers everywhere, where each loop body or fragment element would
//! otherwise divide afresh. Cheaper arithmetic stays where it is used, so
//! it holds no register for the whole kernel.

use ptxc_loop::{Expr, ExprId, KernelVar, LoweredKernel, Stmt, TileAccess, ValueExpr};
use std::collections::{HashMap, HashSet};

/// Every largest index expression in `kernel`'s body that divides and
/// reads only grid and block indices, in the order the body first reads
/// them.
pub(super) fn invariant_indices(kernel: &LoweredKernel) -> Vec<ExprId> {
    let mut roots = Vec::new();
    statement_roots(kernel.body(), &mut roots);
    let mut finder = Finder {
        kernel,
        invariant: HashMap::new(),
        divides: HashMap::new(),
        seen: HashSet::new(),
        found: Vec::new(),
    };
    for root in roots {
        finder.largest(root);
    }
    finder.found
}

/// The index expressions `stmt` reads, directly or in its values.
fn statement_roots(stmt: &Stmt, roots: &mut Vec<ExprId>) {
    match stmt {
        Stmt::Seq(stmts) => stmts.iter().for_each(|stmt| statement_roots(stmt, roots)),
        Stmt::For { body, .. } | Stmt::Alloc { body, .. } => statement_roots(body, roots),
        Stmt::If { condition, then } => {
            roots.extend([condition.index, condition.bound]);
            statement_roots(then, roots);
        }
        Stmt::LoadIndex { index, body, .. } => {
            roots.push(*index);
            statement_roots(body, roots);
        }
        Stmt::Store { index, value, .. } => {
            roots.push(*index);
            value_roots(value, roots);
        }
        Stmt::Barrier => {}
        Stmt::FragmentFill { fragment, .. } => roots.push(fragment.tile),
        Stmt::FragmentLoad { fragment, at: TileAccess { origin, .. }, .. } => {
            roots.extend([fragment.tile, *origin]);
        }
        Stmt::Mma { a, b, accumulator, .. } => roots.extend([a.tile, b.tile, accumulator.tile]),
        Stmt::FragmentEach { fragment, body, .. } => {
            roots.push(fragment.tile);
            statement_roots(body, roots);
        }
    }
}

/// The index expressions `value` reads.
fn value_roots(value: &ValueExpr, roots: &mut Vec<ExprId>) {
    let operands: Vec<&ValueExpr> = match value {
        ValueExpr::Load { index, .. } | ValueExpr::Index { index, .. } => {
            roots.push(*index);
            return;
        }
        ValueExpr::Const(_) | ValueExpr::Element(_) => return,
        ValueExpr::Unary(_, operand) | ValueExpr::Cast { value: operand, .. } => vec![operand],
        ValueExpr::Binary(_, lhs, rhs) => vec![lhs, rhs],
        ValueExpr::Select { condition, then, otherwise } => vec![condition, then, otherwise],
        ValueExpr::MulAdd { lhs, rhs, addend } => vec![lhs, rhs, addend],
    };
    for operand in operands {
        value_roots(operand, roots);
    }
}

/// Finds the largest invariant expressions, remembering which it has seen.
struct Finder<'k> {
    kernel: &'k LoweredKernel,
    invariant: HashMap<ExprId, bool>,
    divides: HashMap<ExprId, bool>,
    seen: HashSet<ExprId>,
    found: Vec<ExprId>,
}

impl Finder<'_> {
    /// Records `id` if it is an invariant expression that divides, or else
    /// the largest such ones inside it.
    fn largest(&mut self, id: ExprId) {
        let expr = self.kernel.exprs().get(id);
        let Some([a, b]) = operands(expr) else { return };
        if self.is_invariant(id) {
            if self.divides(id) && self.seen.insert(id) {
                self.found.push(id);
            }
            return;
        }
        self.largest(a);
        self.largest(b);
    }

    /// Whether `id` holds a floored division or modulo.
    fn divides(&mut self, id: ExprId) -> bool {
        if let Some(&known) = self.divides.get(&id) {
            return known;
        }
        let divides = match self.kernel.exprs().get(id) {
            Expr::FloorDiv(..) | Expr::FloorMod(..) => true,
            compound => operands(compound).is_some_and(|[a, b]| self.divides(a) || self.divides(b)),
        };
        let _previous = self.divides.insert(id, divides);
        divides
    }

    /// Whether `id` reads only grid and block indices.
    fn is_invariant(&mut self, id: ExprId) -> bool {
        if let Some(&known) = self.invariant.get(&id) {
            return known;
        }
        let invariant = match self.kernel.exprs().get(id) {
            Expr::Const(_) => true,
            Expr::Var(var) => matches!(self.kernel.vars()[var], KernelVar::Thread { .. }),
            compound => operands(compound)
                .is_some_and(|[a, b]| self.is_invariant(a) && self.is_invariant(b)),
        };
        let _previous = self.invariant.insert(id, invariant);
        invariant
    }
}

/// The operands of a compound expression; a variable or constant has none.
const fn operands(expr: Expr) -> Option<[ExprId; 2]> {
    match expr {
        Expr::Var(_) | Expr::Const(_) => None,
        Expr::Add(a, b)
        | Expr::Mul(a, b)
        | Expr::Min(a, b)
        | Expr::Max(a, b)
        | Expr::FloorDiv(a, b)
        | Expr::FloorMod(a, b) => Some([a, b]),
    }
}
