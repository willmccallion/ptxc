//! The part of each address in a loop's body that the loop does not
//! change, computed once before the loop, as LLVM's LICM hoists it (D69).
//! ptxas does not hoist out of a branch the body runs only sometimes, such
//! as a pipelined loop's refill, so codegen does; value numbering then
//! hands the body the registers that hold these parts.

use super::position::Ranked;
use super::{CodegenError, Selector};
use ptxc_loop::{BufferId, ExprId, FragmentTile, Stmt, TileAccess, ValueExpr};

/// What a statement reads or writes in memory.
#[derive(Clone, Copy, Debug)]
enum Access {
    /// An element of a buffer, at an index.
    Element(BufferId, ExprId),
    /// A fragment's tile, from a shared buffer where a tile access places it.
    Tile(FragmentTile, BufferId, TileAccess),
}

impl Selector<'_> {
    /// Computes, before a loop `depth` deep running `body`, the part of
    /// each address `body` reads or writes that reads no variable of the
    /// loop or of a loop inside it, in each buffer allocated in memory
    /// before the loop.
    pub(super) fn hoist_addresses(&mut self, body: &Stmt, depth: u32) -> Result<(), CodegenError> {
        let mut found = Vec::new();
        accesses(body, &mut found);
        let outside = |rank: u32| rank < depth;
        for access in found {
            match access {
                Access::Element(buffer, _) | Access::Tile(_, buffer, _)
                    if !self.in_memory(buffer) => {}
                Access::Element(buffer, index) => {
                    let position = self.position_with(buffer, index, None, outside)?;
                    let _address = self.address_from(buffer, position)?;
                }
                Access::Tile(fragment, source, at) => {
                    for place in self.fragment_places(fragment, (source, at))? {
                        let within = Ranked { rank: 0, value: self.within_tile(at, place)? };
                        let position =
                            self.position_with(source, at.origin, Some(within), outside)?;
                        let _address = self.address_from(source, position)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether `buffer` is allocated already, in memory rather than
    /// registers.
    fn in_memory(&self, buffer: BufferId) -> bool {
        self.storage.contains_key(&buffer) && !self.registers.contains(&buffer)
    }
}

/// What `stmt` reads or writes in memory.
fn accesses(stmt: &Stmt, found: &mut Vec<Access>) {
    match stmt {
        Stmt::Seq(stmts) => stmts.iter().for_each(|stmt| accesses(stmt, found)),
        Stmt::For { body, .. }
        | Stmt::If { then: body, .. }
        | Stmt::Alloc { body, .. }
        | Stmt::FragmentEach { body, .. } => accesses(body, found),
        Stmt::LoadIndex { buffer, index, body, .. } => {
            found.push(Access::Element(*buffer, *index));
            accesses(body, found);
        }
        Stmt::Store { buffer, index, value } => {
            found.push(Access::Element(*buffer, *index));
            value_accesses(value, found);
        }
        Stmt::AsyncCopy { target, target_index, source, source_index, .. } => {
            found.extend([
                Access::Element(*target, *target_index),
                Access::Element(*source, *source_index),
            ]);
        }
        Stmt::FragmentLoad { fragment, source, at } => {
            found.push(Access::Tile(*fragment, *source, *at));
        }
        Stmt::Barrier
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::FragmentFill { .. }
        | Stmt::Mma { .. } => {}
    }
}

/// The elements `value` loads.
fn value_accesses(value: &ValueExpr, found: &mut Vec<Access>) {
    let operands: Vec<&ValueExpr> = match value {
        ValueExpr::Load { buffer, index } => {
            found.push(Access::Element(*buffer, *index));
            return;
        }
        ValueExpr::Const(_) | ValueExpr::Element(_) | ValueExpr::Index { .. } => return,
        ValueExpr::Unary(_, operand) | ValueExpr::Cast { value: operand, .. } => vec![operand],
        ValueExpr::Binary(_, lhs, rhs) => vec![lhs, rhs],
        ValueExpr::Select { condition, then, otherwise } => vec![condition, then, otherwise],
        ValueExpr::MulAdd { lhs, rhs, addend } => vec![lhs, rhs, addend],
    };
    for operand in operands {
        value_accesses(operand, found);
    }
}
