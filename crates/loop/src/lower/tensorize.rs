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
use ptxc_core::{DType, Dim, Idx};
use ptxc_target::{MmaShape, ThreadsPerBlock};

/// How the staged tile of `cache` holds the operand it stages, its rows
/// stepping with `rows` and its columns with `columns`: the order and the
/// distance between rows as stored. `None` if it reads `other`, the tile's
/// third loop, or if neither its rows nor its columns are adjacent. A
/// tile's coordinates are affine in the tile's loops, so one step of each
/// from the tile's origin gives the layout of the whole tile.
pub(crate) fn operand_layout(
    compute: &ComputeDef,
    tree: &LoopTree,
    (extents, threads): (&[Vec<Dim>], ThreadsPerBlock),
    cache: CacheId,
    [rows, columns, other]: [VarId; 3],
) -> Option<(TileOrder, Dim)> {
    let mut lowerer = Lowerer::new(compute, tree);
    lowerer.stage(extents, threads);
    let position = lowerer.tile_position(cache);
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
        let (outer, nest) = self.tile_nest(product);
        let fragments = self.fragments(product, &nest);
        let init = Stmt::FragmentFill {
            fragment: fragments.accumulator,
            value: Scalar::zero(product.tile.instruction.accumulator),
        };
        let init = self.wrap_loops(&nest.tiles, init);
        let update = self.mma_step(product, &fragments, &nest);
        let update = self.wrap_loops(&nest.steps, update);
        let store = self.store_output(&ValueExpr::Element(fragments.accumulator.buffer));
        let [m, n, _] = product.tile.loops;
        let write_back = Stmt::FragmentEach {
            fragment: fragments.accumulator,
            row: VarId::new(m.index()),
            column: VarId::new(n.index()),
            body: Box::new(store),
        };
        let write_back = self.wrap_loops(&nest.tiles, write_back);
        let body = [fragments.accumulator.buffer, fragments.lhs.buffer, fragments.rhs.buffer]
            .into_iter()
            .rev()
            .fold(Stmt::Seq(vec![init, update, write_back]), |body, buffer| Stmt::Alloc {
                buffer,
                body: Box::new(body),
            });
        self.wrap_loops(&outer, body)
    }

    /// The serial loops outside `product`'s first reduction loop, and the
    /// loops from it to the tile: its reduction steps, then the warp's
    /// tiles, which validation puts inside every step.
    fn tile_nest(&mut self, product: &TensorizedProduct) -> (Vec<LoopId>, TileNest) {
        let serial = self.serial_loops();
        let reduces = |id: &LoopId| self.tree.loops()[*id].kind == LoopKind::Reduce;
        let first = serial.iter().position(reduces).unwrap_or(serial.len());
        let (outer, inner) = serial.split_at(first);
        let inside: Vec<LoopId> =
            inner.iter().copied().filter(|id| !product.tile.loops.contains(id)).collect();
        let (steps, tiles): (Vec<LoopId>, Vec<LoopId>) = inside.iter().partition(|id| reduces(id));
        let lhs = self.operand_loops(&product.lhs, &tiles);
        let rhs = self.operand_loops(&product.rhs, &tiles);
        (outer.to_vec(), TileNest { steps, tiles, lhs, rhs })
    }

    /// The loops of `tiles` whose variables `operand`'s tile reads.
    fn operand_loops(&mut self, operand: &Operand, tiles: &[LoopId]) -> Vec<LoopId> {
        let position = self.tile_position(operand.cache);
        let read = self.exprs.vars(position);
        tiles.iter().copied().filter(|&id| read.contains(&VarId::new(id.index()))).collect()
    }

    /// One step: each operand's distinct fragments loaded, then every pair
    /// of the warp's tiles multiplied into its accumulator.
    fn mma_step(
        &mut self,
        product: &TensorizedProduct,
        fragments: &Fragments,
        nest: &TileNest,
    ) -> Stmt {
        let var = |id: LoopId| VarId::new(id.index());
        let [m, n, k] = product.tile.loops;
        let (lhs_tile, lhs_at) = self.operand_access(&product.lhs, [var(m), var(k)]);
        let (rhs_tile, rhs_at) = self.operand_access(&product.rhs, [var(k), var(n)]);
        let lhs = Stmt::FragmentLoad { fragment: fragments.lhs, source: lhs_tile, at: lhs_at };
        let rhs = Stmt::FragmentLoad { fragment: fragments.rhs, source: rhs_tile, at: rhs_at };
        let mma = Stmt::Mma {
            instruction: product.tile.instruction,
            a: fragments.lhs,
            b: fragments.rhs,
            accumulator: fragments.accumulator,
        };
        Stmt::Seq(vec![
            self.wrap_loops(&nest.lhs, lhs),
            self.wrap_loops(&nest.rhs, rhs),
            self.wrap_loops(&nest.tiles, mma),
        ])
    }

    /// The fragments of `product`: an accumulator tile for each of the
    /// warp's tiles, and a tile of each operand for each point of the tile
    /// loops it reads.
    fn fragments(&mut self, product: &TensorizedProduct, nest: &TileNest) -> Fragments {
        let instruction = product.tile.instruction;
        let MmaShape { m, n, k } = instruction.shape;
        let multiplicands = instruction.multiplicands;
        Fragments {
            accumulator: self.fragment_array(("acc", instruction.accumulator), [m, n], &nest.tiles),
            lhs: self.fragment_array(("lhs", multiplicands), [m, k], &nest.lhs),
            rhs: self.fragment_array(("rhs", multiplicands), [k, n], &nest.rhs),
        }
    }

    /// A fragment buffer named `name` of `dtype`, holding a `rows` by
    /// `columns` tile for each point of `loops`, at the point their
    /// variables give.
    fn fragment_array(
        &mut self,
        (name, dtype): (&str, DType),
        [rows, columns]: [Dim; 2],
        loops: &[LoopId],
    ) -> FragmentTile {
        let extents: Vec<Dim> = loops.iter().map(|&id| self.tree.loops()[id].extent).collect();
        let count = extents.iter().map(|extent| extent.get()).product::<u64>();
        let vars: Vec<ExprId> =
            loops.iter().map(|&id| self.exprs.var(VarId::new(id.index()))).collect();
        let tile = self.linearize(&vars, &extents);
        let scope = MemScope::Fragment(FragmentShape { rows, columns });
        let size = rows.get() * columns.get() * count;
        FragmentTile { buffer: self.scratch(name, dtype, scope, size), tile }
    }

    /// `operand`'s staged tile and where the fragment's tile starts in it,
    /// its rows stepping with `vars[0]` and its columns with `vars[1]`.
    fn operand_access(&mut self, operand: &Operand, vars: [VarId; 2]) -> (BufferId, TileAccess) {
        let position = self.tile_position(operand.cache);
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

/// The loops between a tensorized product's first reduction loop and its
/// tile: the reduction steps, outermost; the loops over the warp's tiles;
/// and those of them each operand's tile reads.
struct TileNest {
    steps: Vec<LoopId>,
    tiles: Vec<LoopId>,
    lhs: Vec<LoopId>,
    rhs: Vec<LoopId>,
}

/// A tensorized product's fragments.
struct Fragments {
    accumulator: FragmentTile,
    lhs: FragmentTile,
    rhs: FragmentTile,
}
