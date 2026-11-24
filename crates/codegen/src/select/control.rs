//! Selects loops, branches and barriers.

use super::{CodegenError, Selector};
use crate::ast::{B64, Compare, CvtMode, PtxType, Reg, Signedness, Special, Value};
use ptxc_core::Dim;
use ptxc_loop::{Condition, ForKind, Interval, KernelVar, Stmt, ThreadAxis, VarId};

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
            Stmt::For { var, extent, kind: ForKind::Vectorized, body }
                if self.vector_copy(*var, *extent, body)? =>
            {
                Ok(())
            }
            Stmt::For { var, extent, body, .. } => self.for_loop(*var, *extent, body),
            Stmt::If { condition, then } => self.guarded(*condition, then),
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
                self.store(*buffer, *index, value)?;
                self.known.forget_reads_of(*buffer);
                Ok(())
            }
            Stmt::Barrier => {
                self.builder.barrier();
                self.known.forget_values();
                Ok(())
            }
            Stmt::AsyncCopy { target, target_index, source, source_index, run, bounds } => {
                let ends = [(*target, *target_index), (*source, *source_index)];
                self.async_copy(ends, (*run, bounds))
            }
            Stmt::AsyncCommit => {
                self.builder.commit_copies();
                Ok(())
            }
            Stmt::AsyncWait { pending } => {
                self.wait_copies(*pending);
                Ok(())
            }
            Stmt::FragmentFill { fragment, value } => self.fill_fragment(*fragment, *value),
            Stmt::FragmentLoad { fragment, source, at } => {
                self.load_fragment(*fragment, (*source, *at))
            }
            Stmt::Mma { instruction, a, b, accumulator } => {
                self.mma(*instruction, [*a, *b, *accumulator])
            }
            Stmt::FragmentEach { fragment, row, column, body } => {
                self.each_element(*fragment, [*row, *column], body)
            }
        }
    }

    /// `then`, skipped where `condition` fails.
    fn guarded(&mut self, condition: Condition, then: &Stmt) -> Result<(), CodegenError> {
        let index = self.index(condition.index)?;
        let bound = self.index_operand(condition.bound)?;
        let outside =
            self.builder.compare_int(Compare::Ge, Signedness::Signed, index.into(), bound);
        let skip = self.builder.label();
        self.builder.branch(skip, Some((outside, true)));
        let outer = self.known.clone();
        self.statement(then)?;
        self.builder.place(skip);
        self.known.leave(outer);
        Ok(())
    }

    /// A counted loop over `body`, `var` from 0 to `extent - 1`.
    fn for_loop(&mut self, var: VarId, extent: Dim, body: &Stmt) -> Result<(), CodegenError> {
        // NOTE: a vectorized loop that is not a run copy runs as a serial
        // one; vector arithmetic is not selected yet. Unrolled loops never
        // reach here: `generate` expands them first.
        if let Some(range) = Interval::of_loop(extent) {
            self.ranges.insert(var, range);
        }
        let extent = i64::try_from(extent.get()).map_err(|_| CodegenError::TooLarge)?;
        let counter = self.builder.mov(PtxType::S64, Value::<B64>::int(0));
        let _previous = self.vars.insert(var, counter);
        let outer = self.known.clone();
        self.known.forget_values();
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
        self.known.leave(outer);
        Ok(())
    }
}
