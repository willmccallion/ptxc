//! Selects a vectorized loop that copies or clears a run of adjacent
//! elements as one vector access, `.v4.b32` for 16 bytes and `.v2.b32` for
//! 8, where every access the run makes starts aligned to its size, as a
//! staged tile's fill makes them.

use super::registers::WORD_BYTES;
use super::{CodegenError, Selector};
use crate::ast::{B32, B64, PtxType, Reg, Value};
use ptxc_core::Dim;
use ptxc_loop::{BufferId, Constant, ExprId, Stmt, ValueExpr, VarId};

impl Selector<'_> {
    /// Emits `body`, run for each `var` below `extent`, as one vector load
    /// and store, and says whether it could: `body` must store into `extent`
    /// adjacent elements, of 8 or 16 bytes in all, either each loaded from
    /// adjacent elements of a buffer, which verification makes its type, or
    /// zero.
    pub(super) fn vector_copy(
        &mut self,
        var: VarId,
        extent: Dim,
        body: &Stmt,
    ) -> Result<bool, CodegenError> {
        let Stmt::Store { buffer: target, index, value } = only(body) else { return Ok(false) };
        let dtype = self.kernel.buffers()[*target].dtype;
        let bytes = extent.get() * dtype.element_size().get();
        if !matches!(bytes, 8 | 16) {
            return Ok(false);
        }
        let source = match value {
            ValueExpr::Load { buffer, index } => Some((*buffer, *index)),
            ValueExpr::Const(scalar) if all_bits_zero(scalar.constant()) => None,
            _ => return Ok(false),
        };
        let element_twos = dtype.element_size().get().trailing_zeros();
        let indices = std::iter::once(*index).chain(source.map(|(_, from)| from));
        let usable = indices.collect::<Vec<_>>().into_iter().all(|at| {
            self.step(at, var) == Some(1)
                && self.twos(at, Some(var)) + element_twos >= bytes.trailing_zeros()
        });
        if !usable {
            return Ok(false);
        }
        self.copy_run(var, (source, (*target, *index)), bytes / WORD_BYTES)?;
        Ok(true)
    }

    /// The run's `words` from `source`'s element at `var` = 0, or zeros
    /// without one, stored to `target`'s.
    fn copy_run(
        &mut self,
        var: VarId,
        (source, (target, index)): (Option<(BufferId, ExprId)>, (BufferId, ExprId)),
        words: u64,
    ) -> Result<(), CodegenError> {
        let first = self.builder.mov(PtxType::S64, Value::<B64>::int(0));
        let _previous = self.vars.insert(var, first);
        let values: Vec<Reg<B32>> = if let Some((buffer, from)) = source {
            let (space, addr) = self.memory_element(buffer, from)?;
            self.builder.load_vector(space, addr, words)
        } else {
            let zero = self.builder.mov(PtxType::B32, Value::<B32>::int(0));
            (0..words).map(|_| zero).collect()
        };
        let (space, addr) = self.memory_element(target, index)?;
        self.builder.store_vector(space, addr, &values);
        self.known.forget_reads_of(target);
        Ok(())
    }
}

/// `stmt`, or the one statement a sequence of one holds.
fn only(stmt: &Stmt) -> &Stmt {
    match stmt {
        Stmt::Seq(stmts) if stmts.len() == 1 => only(&stmts[0]),
        other => other,
    }
}

/// Whether `constant` is stored as all zero bits: not `-0.0`.
const fn all_bits_zero(constant: Constant) -> bool {
    match constant {
        Constant::Float(value) => value.to_bits() == 0,
        Constant::Int(value) => value == 0,
        // NOTE: bool clears are never vectorized; no staged tile holds bools.
        Constant::Bool(_) => false,
    }
}
