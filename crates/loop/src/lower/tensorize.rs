//! Lowers a tensorized product (D58). Its accumulators are a fragment
//! array over the tile loops left inside the reduction, filled with zero.
//! Each step loads both operands' fragments from their staged tiles and
//! multiplies them. The result is stored element by element through the
//! output's expression, the tile's row and column loops bound to each
//! element's position, so the output's guards and epilogue apply as they
//! do untensorized.

use super::context::Lowerer;
use crate::compute::{ComputeDef, Scalar};
use crate::expr::{Expr, ExprId, VarId};
use crate::schedule::{CacheId, LoopId, LoopKind, LoopTree, Operand, TensorizedProduct};
use crate::stmt::{
    BufferId, FragmentShape, FragmentTile, MemScope, Stmt, TileAccess, TileOrder, ValueExpr,
};
use ptxc_core::{Dim, Idx};
use ptxc_target::ThreadsPerBlock;

/// How the staged tile of `cache` holds an operand read at `index`, its
/// rows stepping with `rows` and its columns with `columns`: the order and
/// the distance between rows as stored. `None` if it reads `other`, the
/// tile's third loop, or if neither its rows nor its columns are adjacent.
/// A staged index is affine in the tile's loops, so one step of each from
/// the tile's origin gives the layout of the whole tile.
pub(crate) fn operand_layout(
    compute: &ComputeDef,
    tree: &LoopTree,
    (extents, threads): (&[Vec<Dim>], ThreadsPerBlock),
    (cache, index): (CacheId, &[ExprId]),
    [rows, columns, other]: [VarId; 3],
) -> Option<(TileOrder, Dim)> {
    let mut lowerer = Lowerer::new(compute, tree);
    lowerer.stage(extents, threads);
    let position = lowerer.tile_position(cache, index);
    if lowerer.exprs.vars(position).contains(&other) {
        return None;
    }
    let vars = [rows, columns];
    let origin = lowerer.at(position, vars, [0, 0]);
    let down = lowerer.at(position, vars, [1, 0]);
    let across = lowerer.at(position, vars, [0, 1]);
    let down = lowerer.difference(origin, down)?;
    let across = lowerer.difference(origin, across)?;
    let stride = |steps: i64| u64::try_from(steps).ok().map(Dim::new);
    match (down, across) {
        (rows, 1) => Some((TileOrder::RowMajor, stride(rows)?)),
        (1, columns) => Some((TileOrder::Transposed, stride(columns)?)),
        _ => None,
    }
}

impl Lowerer<'_> {
    /// The loops computing `product`, a tensorized reduction.
    pub(super) fn tensorized(&mut self, product: &TensorizedProduct) -> Stmt {
        let [m, n, k] = product.tile.loops;
        let serial = self.serial_loops();
        let first = serial
            .iter()
            .position(|&id| self.tree.loops()[id].kind == LoopKind::Reduce)
            .unwrap_or(serial.len());
        let (outer, inner) = serial.split_at(first);
        let steps: Vec<LoopId> =
            inner.iter().copied().filter(|id| ![m, n, k].contains(id)).collect();
        let tile_loops: Vec<LoopId> = steps
            .iter()
            .copied()
            .filter(|&id| self.tree.loops()[id].kind == LoopKind::Spatial)
            .collect();
        let fragments = self.fragments(product, &tile_loops);
        let init = Stmt::FragmentFill {
            fragment: fragments.accumulator,
            value: Scalar::zero(product.tile.instruction.accumulator),
        };
        let init = self.wrap_loops(&tile_loops, init);
        let update = self.mma_step(product, &fragments, [m, n, k]);
        let update = self.wrap_loops(&steps, update);
        let store = self.store_output(&ValueExpr::Element(fragments.accumulator.buffer));
        let write_back = Stmt::FragmentEach {
            fragment: fragments.accumulator,
            row: VarId::new(m.index()),
            column: VarId::new(n.index()),
            body: Box::new(store),
        };
        let write_back = self.wrap_loops(&tile_loops, write_back);
        let body = [fragments.accumulator.buffer, fragments.lhs.buffer, fragments.rhs.buffer]
            .into_iter()
            .rev()
            .fold(Stmt::Seq(vec![init, update, write_back]), |body, buffer| Stmt::Alloc {
                buffer,
                body: Box::new(body),
            });
        self.wrap_loops(outer, body)
    }

    /// One step: both operands' fragments loaded, then multiplied into the
    /// accumulator's tile.
    fn mma_step(
        &mut self,
        product: &TensorizedProduct,
        fragments: &Fragments,
        [m, n, k]: [LoopId; 3],
    ) -> Stmt {
        let var = |id: LoopId| VarId::new(id.index());
        let (lhs_tile, lhs_at) = self.operand_access(&product.lhs, [var(m), var(k)]);
        let (rhs_tile, rhs_at) = self.operand_access(&product.rhs, [var(k), var(n)]);
        Stmt::Seq(vec![
            Stmt::FragmentLoad { fragment: fragments.lhs, source: lhs_tile, at: lhs_at },
            Stmt::FragmentLoad { fragment: fragments.rhs, source: rhs_tile, at: rhs_at },
            Stmt::Mma {
                instruction: product.tile.instruction,
                a: fragments.lhs,
                b: fragments.rhs,
                accumulator: fragments.accumulator,
            },
        ])
    }

    /// The fragments of `product`: an accumulator tile for each point of
    /// `tile_loops`, and one tile of each operand.
    fn fragments(&mut self, product: &TensorizedProduct, tile_loops: &[LoopId]) -> Fragments {
        let instruction = product.tile.instruction;
        let shape = instruction.shape;
        let extents: Vec<Dim> = tile_loops.iter().map(|&id| self.tree.loops()[id].extent).collect();
        let count = extents.iter().map(|extent| extent.get()).product::<u64>();
        let vars: Vec<ExprId> =
            tile_loops.iter().map(|&id| self.exprs.var(VarId::new(id.index()))).collect();
        let tile = self.linearize(&vars, &extents);
        let only = self.constant(0);
        let mut fragment = |name, dtype, [rows, columns]: [Dim; 2], count: u64, tile| {
            let scope = MemScope::Fragment(FragmentShape { rows, columns });
            let size = rows.get() * columns.get() * count;
            FragmentTile { buffer: self.scratch(name, dtype, scope, size), tile }
        };
        let multiplicands = instruction.multiplicands;
        Fragments {
            accumulator: fragment("acc", instruction.accumulator, [shape.m, shape.n], count, tile),
            lhs: fragment("lhs", multiplicands, [shape.m, shape.k], 1, only),
            rhs: fragment("rhs", multiplicands, [shape.k, shape.n], 1, only),
        }
    }

    /// `operand`'s staged tile and where the fragment's tile starts in it,
    /// its rows stepping with `vars[0]` and its columns with `vars[1]`.
    fn operand_access(&mut self, operand: &Operand, vars: [VarId; 2]) -> (BufferId, TileAccess) {
        let position = self.tile_position(operand.cache, &operand.index);
        let origin = self.at(position, vars, [0, 0]);
        let buffer = self.tiles[operand.cache].buffer;
        (buffer, TileAccess { origin, stride: operand.stride, order: operand.order })
    }

    /// `position` with `vars` at `values`, simplified.
    fn at(&mut self, position: ExprId, vars: [VarId; 2], values: [i64; 2]) -> ExprId {
        let mut substituted = position;
        for (var, value) in vars.into_iter().zip(values) {
            let constant = self.constant(value);
            substituted = self.exprs.substitute(substituted, var, constant);
        }
        self.simplify(substituted)
    }

    /// `to - from`, if it is a constant.
    fn difference(&mut self, from: ExprId, to: ExprId) -> Option<i64> {
        let minus_one = self.constant(-1);
        let negated = self.exprs.mul(from, minus_one);
        let sum = self.exprs.add(to, negated);
        let simplified = self.simplify(sum);
        match self.exprs.get(simplified) {
            Expr::Const(value) => Some(value),
            _ => None,
        }
    }
}

/// A tensorized product's fragments.
struct Fragments {
    accumulator: FragmentTile,
    lhs: FragmentTile,
    rhs: FragmentTile,
}
