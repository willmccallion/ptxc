//! The block's threads filling a staged tile from its input, between two
//! barriers, element `e` by thread `e mod T` in round `e div T`.

use super::cache::{Tile, tile_size};
use super::context::Lowerer;
use crate::compute::{Scalar, TensorParamId, VarName};
use crate::expr::{ExprId, Interval, VarId};
use crate::schedule::{LoopAnnotation, ThreadAxis};
use crate::stmt::{Condition, ForKind, KernelVar, Stmt, ValueExpr};
use ptxc_core::{Dim, Idx};

impl Lowerer<'_> {
    /// `body` after a barrier, the filling of `tiles`, and another barrier.
    /// The first barrier keeps the fill from overwriting elements the
    /// previous iteration's body may still be reading.
    pub(super) fn fill_before(&mut self, tiles: &[Tile], body: Stmt) -> Stmt {
        if tiles.is_empty() {
            return body;
        }
        let mut steps = vec![Stmt::Barrier];
        for tile in tiles {
            steps.push(self.fill(tile));
        }
        steps.extend([Stmt::Barrier, body]);
        Stmt::Seq(steps)
    }

    /// The block's threads copying `tile` from its input: element `e` by
    /// thread `e mod T` in round `e div T`, for `T` threads. A tile no
    /// larger than the block takes one round, with no loop.
    fn fill(&mut self, tile: &Tile) -> Stmt {
        let size = tile_size(&tile.extents);
        // Validation's thread count is at least 1: an empty loop cannot be bound.
        let rounds = Dim::new(size.div_ceil(u64::from(tile.threads.get())));
        if rounds.get() == 1 {
            let only = self.constant(0);
            return self.fill_round(tile, only, rounds);
        }
        let (round_var, round) = self.fill_var(tile.tensor, rounds);
        let copy = self.fill_round(tile, round, rounds);
        Stmt::For { var: round_var, extent: rounds, kind: ForKind::Serial, body: Box::new(copy) }
    }

    /// Each thread's copy of its element of `tile` in `round` of `rounds`,
    /// under the bounds the loops' ranges do not already prove. In a
    /// tensorized nest the element is cleared first, so an `mma` over a
    /// tile overhanging its input multiplies zeros there.
    fn fill_round(&mut self, tile: &Tile, round: ExprId, rounds: Dim) -> Stmt {
        let element = self.tile_element(round, rounds);
        let coordinates = self.coordinates(element, tile);
        let dims = self.compute.inputs()[tile.tensor].shape.dims().to_vec();
        let position = self.linearize(&coordinates, &dims);
        let value = ValueExpr::Load { buffer: self.input_buffer(tile.tensor), index: position };
        let store = Stmt::Store { buffer: tile.buffer, index: element, value };
        let in_input = coordinates
            .iter()
            .zip(&dims)
            .rev()
            .fold(store, |body, (&index, &dim)| self.within(index, dim, body));
        let copy = if self.tree.tensorized().is_some() {
            let dtype = self.compute.inputs()[tile.tensor].dtype;
            let zero = ValueExpr::Const(Scalar::zero(dtype));
            let cleared = Stmt::Store { buffer: tile.buffer, index: element, value: zero };
            Stmt::Seq(vec![cleared, in_input])
        } else {
            in_input
        };
        self.within(element, Dim::new(tile_size(&tile.extents)), copy)
    }

    /// A new loop variable over `rounds`, for filling `tensor`'s tile, and
    /// its expression.
    fn fill_var(&mut self, tensor: TensorParamId, rounds: Dim) -> (VarId, ExprId) {
        let name = VarName::new(format!("{}_fill", self.compute.inputs()[tensor].name));
        let var = self.vars.push(KernelVar::Loop(name));
        if let Some(range) = Interval::of_loop(rounds) {
            self.insert_range(var, range);
        }
        (var, self.exprs.var(var))
    }

    /// The tile element the thread copies in `round` of `rounds`:
    /// `round * T + t`, for `T` threads and the thread's position `t` in
    /// its block, x fastest.
    fn tile_element(&mut self, round: ExprId, rounds: Dim) -> ExprId {
        let tree = self.tree;
        let axes = [ThreadAxis::ThreadZ, ThreadAxis::ThreadY, ThreadAxis::ThreadX];
        let mut indices = vec![round];
        let mut extents = vec![rounds];
        for axis in axes {
            indices.push(self.axis_thread(axis));
            extents.push(Dim::new(tree.axis_extent(axis)));
        }
        self.linearize(&indices, &extents)
    }

    /// The thread's index on `axis`: the variable of the loop bound to it,
    /// a tensorized nest's lane on `threadIdx.x`, or 0.
    fn axis_thread(&mut self, axis: ThreadAxis) -> ExprId {
        let tree = self.tree;
        let bound = tree
            .nest()
            .iter()
            .find(|&&id| tree.loops()[id].annotation == LoopAnnotation::Bound(axis));
        match (bound, self.lane) {
            (Some(&id), _) => self.exprs.var(VarId::new(id.index())),
            (None, Some(lane)) if axis == ThreadAxis::ThreadX => self.exprs.var(lane),
            _ => self.constant(0),
        }
    }

    /// The input index of tile element `element`, per axis: its row-major
    /// coordinates in the tile, offset by the tile's origin.
    fn coordinates(&mut self, element: ExprId, tile: &Tile) -> Vec<ExprId> {
        let mut stride = tile_size(&tile.extents);
        let mut coordinates = Vec::new();
        for (&origin, &extent) in tile.origin.iter().zip(&tile.extents) {
            // Tile extents are at least 1, so the stride stays exact.
            stride /= extent.get();
            let (stride_expr, extent_expr) = (self.extent(Dim::new(stride)), self.extent(extent));
            let quotient = self.exprs.floor_div(element, stride_expr);
            let local = self.exprs.floor_mod(quotient, extent_expr);
            let index = self.exprs.add(origin, local);
            coordinates.push(self.simplify(index));
        }
        coordinates
    }

    /// `body` under `0 <= index < bound`, checking only the sides the
    /// loops' ranges leave open.
    fn within(&mut self, index: ExprId, bound: Dim, body: Stmt) -> Stmt {
        let known = self.bounds(index);
        let mut guarded = body;
        if known.is_none_or(|known| i128::from(known.max()) >= i128::from(bound.get())) {
            let limit = self.extent(bound);
            let condition = Condition { index, bound: limit };
            guarded = Stmt::If { condition, then: Box::new(guarded) };
        }
        if known.is_none_or(|known| known.min() < 0) {
            // `0 <= index` as `-index < 1`: an `If` compares only one way.
            let negated = self.negated(index);
            let condition = Condition { index: self.simplify(negated), bound: self.constant(1) };
            guarded = Stmt::If { condition, then: Box::new(guarded) };
        }
        guarded
    }
}
