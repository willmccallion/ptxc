//! Selects loops, branches and barriers.

use super::{CodegenError, Selector};
use crate::ast::{B64, Compare, CvtMode, PtxType, Reg, Signedness, Special, Value};
use ptxc_loop::{KernelVar, Stmt, ThreadAxis};

const fn special(axis: ThreadAxis) -> Special {
    match axis {
        ThreadAxis::BlockX => Special::BlockX,
        ThreadAxis::BlockY => Special::BlockY,
        ThreadAxis::BlockZ => Special::BlockZ,
        ThreadAxis::ThreadX => Special::ThreadX,
        ThreadAxis::ThreadY => Special::ThreadY,
        ThreadAxis::ThreadZ => Special::ThreadZ,
    }
}

impl Selector<'_> {
    /// Binds every grid and block index the kernel reads, as `s64`.
    pub(super) fn thread_indices(&mut self) {
        let thread_vars: Vec<_> = self
            .kernel
            .vars()
            .iter_enumerated()
            .filter_map(|(var, definition)| match definition {
                KernelVar::Thread { axis, .. } => Some((var, *axis)),
                KernelVar::Loop(_) | KernelVar::Loaded(_) => None,
            })
            .collect();
        for (var, axis) in thread_vars {
            let position = self.builder.special(special(axis));
            let wide: Reg<B64> =
                self.builder.convert(CvtMode::Plain, (PtxType::S64, PtxType::U32), position.into());
            let _new = self.vars.insert(var, wide);
        }
    }

    /// Emits `stmt`.
    pub(super) fn statement(&mut self, stmt: &Stmt) -> Result<(), CodegenError> {
        match stmt {
            Stmt::Seq(stmts) => stmts.iter().try_for_each(|stmt| self.statement(stmt)),
            Stmt::For { var, extent, body, .. } => {
                // NOTE: unrolled and vectorized loops run as serial ones in
                // M2; ptxas unrolls short constant loops itself.
                let extent = i64::try_from(extent.get()).map_err(|_| CodegenError::TooLarge)?;
                let counter = self.builder.mov(PtxType::S64, Value::<B64>::int(0));
                let _previous = self.vars.insert(*var, counter);
                let (top, exit) = (self.builder.label(), self.builder.label());
                self.builder.place(top);
                let done = self.builder.compare_int(
                    Compare::Ge,
                    Signedness::Signed,
                    counter.into(),
                    Value::<B64>::int(extent),
                );
                self.builder.branch(exit, Some((done, true)));
                self.statement(body)?;
                self.builder.increment(counter);
                self.builder.branch(top, None);
                self.builder.place(exit);
                Ok(())
            }
            Stmt::If { condition, then } => {
                let index = self.index(condition.index)?;
                let bound = self.index_operand(condition.bound)?;
                let outside =
                    self.builder.compare_int(Compare::Ge, Signedness::Signed, index.into(), bound);
                let skip = self.builder.label();
                self.builder.branch(skip, Some((outside, true)));
                self.statement(then)?;
                self.builder.place(skip);
                Ok(())
            }
            Stmt::Alloc { buffer, body } => {
                self.declare(*buffer)?;
                self.statement(body)
            }
            Stmt::LoadIndex { var, buffer, index, body } => {
                let loaded = self.load_index(*buffer, *index)?;
                let _previous = self.vars.insert(*var, loaded);
                self.statement(body)
            }
            Stmt::Store { buffer, index, value } => {
                let value = self.value(value)?;
                self.store(*buffer, *index, value)
            }
            Stmt::Barrier => {
                self.builder.barrier();
                Ok(())
            }
        }
    }
}
