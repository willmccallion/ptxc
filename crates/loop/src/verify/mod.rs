//! Verifies lowered kernels: bounds, scopes and barrier placement.

use crate::compute::{TensorName, VarName, types};
use crate::expr::{ExprId, Interval, VarId};
use crate::schedule::ThreadAxis;
use crate::stmt::{Access, BufferId, KernelVar, LoweredKernel, MemScope, Stmt, ValueExpr};
use facts::Facts;
use ptxc_core::DType;
use std::collections::HashSet;
use thiserror::Error;

mod facts;

/// A lowered kernel that breaks a rule codegen relies on.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// A variable read where nothing binds it.
    #[error("{0} is read where it is not bound")]
    UnboundVar(VarName),
    /// A store to a buffer the kernel only reads.
    #[error("{0} is an input, but is stored to")]
    StoreToInput(TensorName),
    /// A load from a buffer the kernel only writes.
    #[error("{0} is an output, but is loaded from")]
    LoadFromOutput(TensorName),
    /// A shared or local buffer used outside its `Alloc`.
    #[error("{0} is used outside its allocation")]
    NotAllocated(TensorName),
    /// An `Alloc` of a global buffer.
    #[error("{0} is global, so it is a parameter, not allocated")]
    GlobalAllocated(TensorName),
    /// A shared or local access not provably within the buffer.
    #[error("an access to {0} may leave the buffer")]
    OutOfBounds(TensorName),
    /// A barrier some threads of a block may not reach.
    #[error("a barrier is under a condition that depends on the thread")]
    DivergentBarrier,
    /// A value of the wrong type for where it goes.
    #[error("{place} takes {expected:?}, but is given {actual:?}")]
    WrongType {
        /// The store, index load or condition.
        place: String,
        /// The type it takes.
        expected: DType,
        /// The type given.
        actual: DType,
    },
    /// An operation on types it does not take.
    #[error(transparent)]
    Operation(#[from] crate::compute::ComputeError),
}

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
    };
    for (var, definition) in kernel.vars().iter_enumerated() {
        if let KernelVar::Thread { axis, .. } = definition {
            let _new = scope.bound.insert(var);
            if let Some(range) = launch_range(kernel, *axis) {
                scope.facts.insert_range(var, range);
            }
        }
    }
    scope.stmt(kernel.body())
}

fn launch_range(kernel: &LoweredKernel, axis: ThreadAxis) -> Option<Interval> {
    let launch = kernel.launch();
    let extent = match axis {
        ThreadAxis::BlockX => launch.grid[0].get(),
        ThreadAxis::BlockY => launch.grid[1].get(),
        ThreadAxis::BlockZ => launch.grid[2].get(),
        ThreadAxis::ThreadX => launch.block[0].get(),
        ThreadAxis::ThreadY => launch.block[1].get(),
        ThreadAxis::ThreadZ => launch.block[2].get(),
    };
    Interval::new(0, i64::from(extent) - 1)
}

/// What holds at a point of the kernel.
#[derive(Clone)]
struct Scope<'k> {
    kernel: &'k LoweredKernel,
    bound: HashSet<VarId>,
    facts: Facts,
    allocated: HashSet<BufferId>,
    divergent: bool,
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
            Stmt::Alloc { buffer, body } => {
                let found = &self.kernel.buffers()[*buffer];
                if found.scope == MemScope::Global {
                    return Err(VerifyError::GlobalAllocated(found.name.clone()));
                }
                let mut inner = self.clone();
                let _new = inner.allocated.insert(*buffer);
                inner.stmt(body)
            }
            Stmt::LoadIndex { var, buffer, index, body } => {
                self.load(*buffer, *index)?;
                let dtype = self.kernel.buffers()[*buffer].dtype;
                if !matches!(dtype, DType::I32 | DType::I64) {
                    return Err(VerifyError::WrongType {
                        place: "an index load".to_owned(),
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
        }
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
            ValueExpr::Index { index, dtype } => self.index(*index).map(|()| *dtype),
            ValueExpr::Unary(op, operand) => Ok(types::unary(*op, self.value(operand)?)?),
            ValueExpr::Binary(op, lhs, rhs) => {
                Ok(types::binary(*op, self.value(lhs)?, self.value(rhs)?)?)
            }
            ValueExpr::Cast { to, value, .. } => self.value(value).map(|_| *to),
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
mod tests;
