//! Verifies lowered kernels: bounds, scopes, barrier placement and the
//! fragment statements.

use crate::compute::{VarName, types};
use crate::expr::{ExprId, Interval, VarId};
use crate::stmt::{Access, BufferId, KernelVar, LoweredKernel, MemScope, Stmt, ValueExpr};
pub use error::VerifyError;
use facts::Facts;
use ptxc_core::DType;
use std::collections::HashSet;

mod error;
mod facts;
mod fragment;

/// Checks the rules code generation relies on in `kernel`.
///
/// Every variable is bound where it is read; buffers are read and written as
/// their scope and access allow, scratch buffers only inside their
/// allocation and provably in bounds; no barrier is under a thread-dependent
/// condition; and values have the types they are used at. Global accesses
/// are in bounds by construction (D40) and not checked.
pub fn verify(kernel: &LoweredKernel) -> Result<(), VerifyError> {
    let mut scope = Scope {
        kernel,
        bound: HashSet::new(),
        facts: Facts::default(),
        allocated: HashSet::new(),
        divergent: false,
        elements: Vec::new(),
    };
    for (var, definition) in kernel.vars().iter_enumerated() {
        if let KernelVar::Thread { axis, .. } = definition {
            let _new = scope.bound.insert(var);
            if let Some(range) = kernel.launch().range(*axis) {
                scope.facts.insert_range(var, range);
            }
        }
    }
    scope.stmt(kernel.body())
}

/// What holds at a point of the kernel.
#[derive(Clone)]
struct Scope<'k> {
    kernel: &'k LoweredKernel,
    bound: HashSet<VarId>,
    facts: Facts,
    allocated: HashSet<BufferId>,
    divergent: bool,
    /// The fragments the enclosing `FragmentEach`es iterate.
    elements: Vec<BufferId>,
}

impl Scope<'_> {
    fn stmt(&mut self, stmt: &Stmt) -> Result<(), VerifyError> {
        match stmt {
            Stmt::Seq(stmts) => stmts.iter().try_for_each(|stmt| self.stmt(stmt)),
            Stmt::For { var, extent, body, .. } => {
                let mut inner = self.clone();
                let _new = inner.bound.insert(*var);
                if let Some(range) = Interval::of_loop(*extent) {
                    inner.facts.insert_range(*var, range);
                }
                inner.stmt(body)
            }
            Stmt::If { condition, then } => {
                self.index(condition.index)?;
                self.index(condition.bound)?;
                let mut inner = self.clone();
                inner.divergent |= self.depends_on_thread(condition.index)
                    || self.depends_on_thread(condition.bound);
                inner.facts.refine(self.kernel.exprs(), *condition);
                inner.stmt(then)
            }
            Stmt::Alloc { buffer, body } => self.alloc(*buffer, body),
            Stmt::LoadIndex { var, buffer, index, body } => {
                self.load(*buffer, *index)?;
                let dtype = self.kernel.buffers()[*buffer].dtype;
                if !matches!(dtype, DType::I32 | DType::I64) {
                    let place = "an index load".to_owned();
                    return Err(VerifyError::WrongType {
                        place,
                        expected: DType::I64,
                        actual: dtype,
                    });
                }
                let mut inner = self.clone();
                let _new = inner.bound.insert(*var);
                inner.stmt(body)
            }
            Stmt::Store { buffer, index, value } => self.store(*buffer, *index, value),
            Stmt::Barrier => {
                if self.divergent {
                    Err(VerifyError::DivergentBarrier)
                } else {
                    Ok(())
                }
            }
            Stmt::FragmentFill { fragment, value } => self.fill(*fragment, *value),
            Stmt::FragmentLoad { fragment, source, at } => self.tile_load(*fragment, *source, *at),
            Stmt::Mma { instruction, a, b, accumulator } => {
                self.mma(*instruction, [*a, *b, *accumulator])
            }
            Stmt::FragmentStore { fragment, target, at } => {
                self.tile_store(*fragment, *target, *at)
            }
            Stmt::FragmentEach { fragment, row, column, body } => {
                self.each(*fragment, [*row, *column], body)
            }
        }
    }

    /// `body` with `buffer` allocated: a scratch buffer, and a fragment
    /// with room for its tile.
    fn alloc(&self, buffer: BufferId, body: &Stmt) -> Result<(), VerifyError> {
        let found = &self.kernel.buffers()[buffer];
        if found.scope == MemScope::Global {
            return Err(VerifyError::GlobalAllocated(found.name.clone()));
        }
        if let MemScope::Fragment(shape) = found.scope
            && found.size.get() < shape.rows.get().saturating_mul(shape.columns.get())
        {
            return Err(VerifyError::FragmentSize(found.name.clone()));
        }
        let mut inner = self.clone();
        let _new = inner.allocated.insert(buffer);
        inner.stmt(body)
    }

    fn store(&self, buffer: BufferId, index: ExprId, value: &ValueExpr) -> Result<(), VerifyError> {
        let found = &self.kernel.buffers()[buffer];
        let input = self
            .kernel
            .params()
            .iter()
            .any(|param| param.buffer == buffer && param.access == Access::Read);
        if input {
            return Err(VerifyError::StoreToInput(found.name.clone()));
        }
        self.access(buffer, index)?;
        let actual = self.value(value)?;
        if actual == found.dtype {
            Ok(())
        } else {
            Err(VerifyError::WrongType {
                place: format!("a store to {}", found.name),
                expected: found.dtype,
                actual,
            })
        }
    }

    fn load(&self, buffer: BufferId, index: ExprId) -> Result<(), VerifyError> {
        let output = self
            .kernel
            .params()
            .iter()
            .any(|param| param.buffer == buffer && param.access == Access::Write);
        if output {
            return Err(VerifyError::LoadFromOutput(self.kernel.buffers()[buffer].name.clone()));
        }
        self.access(buffer, index)
    }

    /// Checks an access's index and, for a scratch buffer, that it is
    /// allocated here and the index stays within it.
    fn access(&self, buffer: BufferId, index: ExprId) -> Result<(), VerifyError> {
        self.index(index)?;
        let found = &self.kernel.buffers()[buffer];
        if found.scope == MemScope::Global {
            return Ok(());
        }
        if let MemScope::Fragment(_) = found.scope {
            return Err(VerifyError::FragmentAccess(found.name.clone()));
        }
        if !self.allocated.contains(&buffer) {
            return Err(VerifyError::NotAllocated(found.name.clone()));
        }
        let last = found.size.get().checked_sub(1).and_then(|last| i64::try_from(last).ok());
        let bounds = self.facts.bounds(self.kernel.exprs(), index);
        match bounds.zip(last) {
            Some((bounds, last)) if bounds.min() >= 0 && bounds.max() <= last => Ok(()),
            _ => Err(VerifyError::OutOfBounds(found.name.clone())),
        }
    }

    /// Checks that every variable `index` reads is bound.
    fn index(&self, index: ExprId) -> Result<(), VerifyError> {
        for var in self.kernel.exprs().vars(index) {
            if !self.bound.contains(&var) {
                let name = self
                    .kernel
                    .vars()
                    .get(var)
                    .map_or_else(|| VarName::new(format!("{var:?}")), |found| found.name().clone());
                return Err(VerifyError::UnboundVar(name));
            }
        }
        Ok(())
    }

    fn depends_on_thread(&self, index: ExprId) -> bool {
        self.kernel.exprs().vars(index).iter().any(|&var| {
            matches!(
                self.kernel.vars().get(var),
                Some(KernelVar::Thread { .. } | KernelVar::Loaded(_))
            )
        })
    }

    /// The type of `value`, after checking its loads, indices and operations.
    fn value(&self, value: &ValueExpr) -> Result<DType, VerifyError> {
        match value {
            ValueExpr::Load { buffer, index } => {
                self.load(*buffer, *index)?;
                Ok(self.kernel.buffers()[*buffer].dtype)
            }
            ValueExpr::Const(scalar) => Ok(scalar.dtype()),
            ValueExpr::Element(fragment) => self.element(*fragment),
            ValueExpr::Index { index, dtype } => self.index(*index).map(|()| *dtype),
            ValueExpr::Unary(op, operand) => Ok(types::unary(*op, self.value(operand)?)?),
            ValueExpr::Binary(op, lhs, rhs) => {
                Ok(types::binary(*op, self.value(lhs)?, self.value(rhs)?)?)
            }
            ValueExpr::Cast { to, value, .. } => self.value(value).map(|_| *to),
            ValueExpr::MulAdd { lhs, rhs, addend } => {
                for operand in [lhs, rhs, addend] {
                    let actual = self.value(operand)?;
                    if actual != DType::F32 {
                        let place = "a multiply-add".to_owned();
                        return Err(VerifyError::WrongType { place, expected: DType::F32, actual });
                    }
                }
                Ok(DType::F32)
            }
            ValueExpr::Select { condition, then, otherwise } => {
                let condition = self.value(condition)?;
                if condition != DType::Bool {
                    return Err(VerifyError::WrongType {
                        place: "a select's condition".to_owned(),
                        expected: DType::Bool,
                        actual: condition,
                    });
                }
                let (then, otherwise) = (self.value(then)?, self.value(otherwise)?);
                if then == otherwise {
                    Ok(then)
                } else {
                    Err(VerifyError::WrongType {
                        place: "a select's second branch".to_owned(),
                        expected: then,
                        actual: otherwise,
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod fragment_tests;
#[cfg(test)]
mod tests;
