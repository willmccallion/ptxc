//! Runs every thread of a block in lockstep between barriers.

use super::RunError;
use super::memory::Memory;
use super::ops;
use super::race::{GlobalWriters, PhaseLog};
use super::value::{Value, wrap};
use crate::expr::{ExprId, VarId};
use crate::schedule::ThreadAxis;
use crate::stmt::{BufferId, KernelVar, LoweredKernel, MemScope, Stmt, ValueExpr};
use std::collections::HashMap;

/// One thread's variables and local buffers.
#[derive(Default)]
struct Thread {
    vars: HashMap<VarId, i64>,
    locals: HashMap<BufferId, Memory>,
}

/// The global state every block shares.
pub(super) struct Grid<'k> {
    pub(super) kernel: &'k LoweredKernel,
    pub(super) globals: HashMap<BufferId, Memory>,
    pub(super) writers: GlobalWriters,
}

/// One block running: its threads, shared memory and current phase.
struct Block<'g, 'k> {
    grid: &'g mut Grid<'k>,
    threads: Vec<Thread>,
    shared: HashMap<BufferId, Memory>,
    phase: PhaseLog,
    first_thread: usize,
}

impl Grid<'_> {
    /// Runs block `block` of the grid, numbered x fastest.
    pub(super) fn run_block(&mut self, block: usize) -> Result<(), RunError> {
        let launch = self.kernel.launch();
        let dims =
            |extents: [u32; 3]| extents.map(|extent| usize::try_from(extent).unwrap_or(usize::MAX));
        let grid = dims(launch.grid.map(ptxc_target::BlockCount::get));
        let block_dims = dims(launch.block.map(ptxc_target::ThreadsPerBlock::get));
        let count = block_dims.iter().product::<usize>();
        let block_index = unflatten(block, grid);
        let threads = (0..count)
            .map(|thread| self.thread(block_index, unflatten(thread, block_dims)))
            .collect();
        let body = self.kernel.body();
        let mut running = Block {
            grid: self,
            threads,
            shared: HashMap::new(),
            phase: PhaseLog::default(),
            first_thread: block * count,
        };
        let all: Vec<usize> = (0..count).collect();
        running.exec(body, &all)
    }

    /// A thread whose grid and block indices are bound.
    fn thread(&self, block: [usize; 3], thread: [usize; 3]) -> Thread {
        let mut vars = HashMap::new();
        for (var, definition) in self.kernel.vars().iter_enumerated() {
            let KernelVar::Thread { axis, .. } = definition else { continue };
            let index = match axis {
                ThreadAxis::BlockX => block[0],
                ThreadAxis::BlockY => block[1],
                ThreadAxis::BlockZ => block[2],
                ThreadAxis::ThreadX => thread[0],
                ThreadAxis::ThreadY => thread[1],
                ThreadAxis::ThreadZ => thread[2],
            };
            let _new = vars.insert(var, i64::try_from(index).unwrap_or(i64::MAX));
        }
        Thread { vars, locals: HashMap::new() }
    }
}

/// The x, y, z position of flat position `flat` in `dims`, x fastest.
fn unflatten(flat: usize, dims: [usize; 3]) -> [usize; 3] {
    let x = flat % dims[0].max(1);
    let y = (flat / dims[0].max(1)) % dims[1].max(1);
    let z = flat / (dims[0].max(1) * dims[1].max(1));
    [x, y, z]
}

impl Block<'_, '_> {
    /// Runs `stmt` on the `active` threads, statement by statement.
    fn exec(&mut self, stmt: &Stmt, active: &[usize]) -> Result<(), RunError> {
        match stmt {
            Stmt::Seq(stmts) => stmts.iter().try_for_each(|stmt| self.exec(stmt, active)),
            Stmt::For { var, extent, body, .. } => {
                for value in 0..extent.get() {
                    for &thread in active {
                        let _previous = self.threads[thread]
                            .vars
                            .insert(*var, i64::try_from(value).unwrap_or(i64::MAX));
                    }
                    self.exec(body, active)?;
                }
                Ok(())
            }
            Stmt::If { condition, then } => {
                let mut taken = Vec::new();
                for &thread in active {
                    if self.index(condition.index, thread)? < self.index(condition.bound, thread)? {
                        taken.push(thread);
                    }
                }
                self.exec(then, &taken)
            }
            Stmt::Alloc { buffer, body } => self.alloc(*buffer, body, active),
            Stmt::LoadIndex { var, buffer, index, body } => {
                for &thread in active {
                    let position = self.index(*index, thread)?;
                    let loaded = self.load(*buffer, position, thread)?.int();
                    let _previous = self.threads[thread].vars.insert(*var, loaded);
                }
                self.exec(body, active)
            }
            Stmt::Store { buffer, index, value } => {
                for &thread in active {
                    let position = self.index(*index, thread)?;
                    let stored = self.value(value, thread)?;
                    self.store(*buffer, position, stored, thread)?;
                }
                Ok(())
            }
            Stmt::Barrier => {
                if active.len() != self.threads.len() {
                    return Err(RunError::DivergentBarrier);
                }
                self.phase.clear();
                Ok(())
            }
        }
    }

    /// Runs `body` with `buffer` allocated: once for the block if shared,
    /// once per active thread if local.
    fn alloc(&mut self, buffer: BufferId, body: &Stmt, active: &[usize]) -> Result<(), RunError> {
        let found = &self.grid.kernel.buffers()[buffer];
        let size = usize::try_from(found.size.get()).unwrap_or(usize::MAX);
        let fresh = Memory::unset(found.name.clone(), size);
        if found.scope == MemScope::Shared {
            let _previous = self.shared.insert(buffer, fresh);
            let result = self.exec(body, active);
            let _freed = self.shared.remove(&buffer);
            return result;
        }
        for &thread in active {
            let _previous = self.threads[thread].locals.insert(buffer, fresh.clone());
        }
        let result = self.exec(body, active);
        for &thread in active {
            let _freed = self.threads[thread].locals.remove(&buffer);
        }
        result
    }

    fn index(&self, index: ExprId, thread: usize) -> Result<i64, RunError> {
        let vars = &self.threads[thread].vars;
        self.grid
            .kernel
            .exprs()
            .evaluate(index, &|var| vars.get(&var).copied())
            .ok_or(RunError::Index)
    }

    fn memory(&mut self, buffer: BufferId, thread: usize) -> Result<&mut Memory, RunError> {
        let found = &self.grid.kernel.buffers()[buffer];
        let memory = match found.scope {
            MemScope::Global => self.grid.globals.get_mut(&buffer),
            MemScope::Shared => self.shared.get_mut(&buffer),
            MemScope::Local => self.threads[thread].locals.get_mut(&buffer),
        };
        memory.ok_or_else(|| RunError::NotAllocated(found.name.clone()))
    }

    fn load(&mut self, buffer: BufferId, index: i64, thread: usize) -> Result<Value, RunError> {
        let value = self.memory(buffer, thread)?.read(index)?;
        if self.grid.kernel.buffers()[buffer].scope == MemScope::Shared {
            let name = self.grid.kernel.buffers()[buffer].name.clone();
            self.phase.read(buffer, &name, index, thread)?;
        }
        Ok(value)
    }

    fn store(
        &mut self,
        buffer: BufferId,
        index: i64,
        value: Value,
        thread: usize,
    ) -> Result<(), RunError> {
        self.memory(buffer, thread)?.write(index, value)?;
        let found = &self.grid.kernel.buffers()[buffer];
        let name = found.name.clone();
        match found.scope {
            MemScope::Shared => self.phase.write(buffer, &name, index, thread),
            MemScope::Global => {
                self.grid.writers.write(buffer, &name, index, self.first_thread + thread)
            }
            MemScope::Local => Ok(()),
        }
    }

    /// `value` for one thread, each operation rounded to its type.
    fn value(&mut self, value: &ValueExpr, thread: usize) -> Result<Value, RunError> {
        let buffers = self.grid.kernel.buffers();
        Ok(match value {
            ValueExpr::Load { buffer, index } => {
                let position = self.index(*index, thread)?;
                self.load(*buffer, position, thread)?
            }
            ValueExpr::Const(scalar) => Value::of_scalar(*scalar),
            ValueExpr::Index { index, dtype } => {
                wrap(i128::from(self.index(*index, thread)?), *dtype)
            }
            ValueExpr::Unary(op, operand) => {
                let dtype = operand.dtype(buffers);
                ops::unary(*op, self.value(operand, thread)?, dtype)
            }
            ValueExpr::Binary(op, lhs, rhs) => {
                let dtype = lhs.dtype(buffers);
                let (lhs, rhs) = (self.value(lhs, thread)?, self.value(rhs, thread)?);
                ops::binary(*op, lhs, rhs, dtype)?
            }
            ValueExpr::Cast { to, overflow, value } => {
                let from = value.dtype(buffers);
                ops::cast(self.value(value, thread)?, from, *to, *overflow)
            }
            ValueExpr::Select { condition, then, otherwise } => {
                let condition = self.value(condition, thread)?.bool();
                let (then, otherwise) = (self.value(then, thread)?, self.value(otherwise, thread)?);
                if condition { then } else { otherwise }
            }
        })
    }
}
