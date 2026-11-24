//! `FunctionBuilder`'s warp-level matrix instructions: `ldmatrix` and
//! `mma.sync`, which every lane of a warp executes together.

use super::function::FunctionBuilder;
use super::instr::{Instr, PtxType};
use super::operand::Address;
use super::reg::{B32, Class, Reg};
use ptxc_target::MmaShape;

/// What `ldmatrix` gives each group of four lanes: a row of each matrix,
/// or, transposed, a column.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MatrixOrder {
    Rows,
    Columns,
}

/// One `mma.sync` variant: its shape, and its accumulator's and
/// multiplicands' types.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MmaOp {
    pub(crate) shape: MmaShape,
    pub(crate) accumulator: PtxType,
    pub(crate) multiplicands: PtxType,
}

impl FunctionBuilder {
    /// `count` 8 by 8 matrices of 16-bit elements from shared memory, one
    /// 32-bit register each, lane `l` giving the address of row `l % 8` of
    /// matrix `l / 8`.
    pub(crate) fn ld_matrix(
        &mut self,
        order: MatrixOrder,
        count: u64,
        addr: Address,
    ) -> Vec<Reg<B32>> {
        let dst: Vec<Reg<B32>> = (0..count).map(|_| self.reg()).collect();
        let ids = dst.iter().map(|reg| reg.id()).collect();
        self.push(Instr::LdMatrix { order, dst: ids, addr });
        dst
    }

    /// `a · b + c` on one warp's fragments, as `op` computes it.
    pub(crate) fn mma<C: Class>(
        &mut self,
        op: MmaOp,
        [a, b]: [&[Reg<B32>]; 2],
        c: &[Reg<C>],
    ) -> Vec<Reg<C>> {
        let d: Vec<Reg<C>> = c.iter().map(|_| self.reg()).collect();
        let ids = |regs: &[Reg<B32>]| regs.iter().map(|reg| reg.id()).collect();
        self.push(Instr::Mma {
            op,
            d: d.iter().map(|reg| reg.id()).collect(),
            a: ids(a),
            b: ids(b),
            c: c.iter().map(|reg| reg.id()).collect(),
        });
        d
    }
}
