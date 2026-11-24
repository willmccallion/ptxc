//! The block's threads filling a staged tile from its input, between two
//! barriers: run `r` of the tile's adjacent elements by thread `r mod T` in
//! round `r div T`. A run is 16 bytes where the tile and its input allow,
//! copied by one vectorized loop, and one element otherwise.

use super::cache::{Tile, tile_size};
use super::context::Lowerer;
use crate::compute::{Scalar, TensorParamId, VarName};
use crate::expr::{ExprId, Interval, VarId};
use crate::schedule::{LoopAnnotation, ThreadAxis};
use crate::stmt::{BufferId, Condition, ForKind, KernelVar, Stmt, ValueExpr};
use ptxc_core::{Dim, Idx};

/// The bytes a thread copies at once where a tile allows: one 16-byte
/// vector access, as CUTLASS's tile iterators make.
const RUN_BYTES: u64 = 16;

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

    /// The block's threads copying `tile` from its input: run `r` by
    /// thread `r mod T` in round `r div T`, for `T` threads. A tile no
    /// larger than the block takes one round, with no loop.
    fn fill(&mut self, tile: &Tile) -> Stmt {
        let run = self.run_length(tile);
        let runs = Dim::new(tile_size(&tile.extents) / run);
        // Validation's thread count is at least 1: an empty loop cannot be bound.
        let rounds = Dim::new(runs.get().div_ceil(u64::from(tile.threads.get())));
        if rounds.get() == 1 {
            let only = self.constant(0);
            return self.fill_round(tile, (only, rounds), run);
        }
        let (round_var, round) = self.fill_var(tile.tensor, "fill", rounds);
        let copy = self.fill_round(tile, (round, rounds), run);
        Stmt::For { var: round_var, extent: rounds, kind: ForKind::Serial, body: Box::new(copy) }
    }

    /// The adjacent elements a thread copies at once: a 16-byte run where
    /// the tile's and its input's innermost extents and the tile's origin
    /// along that axis are multiples of it, so a run lies in one row of
    /// both, aligned; and one element otherwise.
    fn run_length(&self, tile: &Tile) -> u64 {
        let input = &self.compute.inputs()[tile.tensor];
        let run = RUN_BYTES / input.dtype.element_size().get();
        let innermost = (tile.extents.last(), input.shape.dims().last(), tile.origin.last());
        let (Some(tile_extent), Some(input_extent), Some(&origin)) = innermost else { return 1 };
        let fits = tile_extent.get().is_multiple_of(run)
            && input_extent.get().is_multiple_of(run)
            && self.exprs.is_multiple_of(origin, run);
        if fits { run } else { 1 }
    }

    /// Each thread's copy of its run of `tile` in `round` of `rounds`,
    /// under the bounds the loops' ranges do not already prove. A run
    /// starts at a multiple of its length in a row whose extent is one, so
    /// its first element's bounds are all of its elements'. In a
    /// tensorized nest the run is cleared first, so an `mma` over a tile
    /// overhanging its input multiplies zeros there.
    fn fill_round(&mut self, tile: &Tile, (round, rounds): (ExprId, Dim), run: u64) -> Stmt {
        let index = self.tile_element(round, rounds);
        let first = self.scaled(index, run);
        let coordinates = self.coordinates(first, tile);
        let dims = self.compute.inputs()[tile.tensor].shape.dims().to_vec();
        let position = self.linearize(&coordinates, &dims);
        let source = Some((self.input_buffer(tile.tensor), position));
        let copied = self.run_copy(tile, (first, source), run);
        let in_input = coordinates
            .iter()
            .zip(&dims)
            .rev()
            .fold(copied, |body, (&index, &dim)| self.within(index, dim, body));
        let copy = if self.tree.tensorized().is_some() {
            let cleared = self.run_copy(tile, (first, None), run);
            Stmt::Seq(vec![cleared, in_input])
        } else {
            in_input
        };
        self.within(index, Dim::new(tile_size(&tile.extents) / run), copy)
    }

    /// `id * factor`, left unsimplified for `id`'s guard to bound it.
    fn scaled(&mut self, id: ExprId, factor: u64) -> ExprId {
        if factor == 1 {
            return id;
        }
        let factor = self.extent(Dim::new(factor));
        self.exprs.mul(id, factor)
    }

    /// The `run` elements of `tile` from `first` set from `source`'s
    /// buffer at its position onward, or cleared without one: a vectorized
    /// loop, or one store for a run of one.
    fn run_copy(
        &mut self,
        tile: &Tile,
        (first, source): (ExprId, Option<(BufferId, ExprId)>),
        run: u64,
    ) -> Stmt {
        let (store_at, step) = if run == 1 {
            (first, None)
        } else {
            let role = if source.is_some() { "run" } else { "clear" };
            let (var, element) = self.fill_var(tile.tensor, role, Dim::new(run));
            // Unsimplified, so the guard on the run's index bounds the store.
            (self.exprs.add(first, element), Some((var, element)))
        };
        let value = match source {
            Some((buffer, position)) => {
                let at = step.map_or(position, |(_, element)| self.exprs.add(position, element));
                ValueExpr::Load { buffer, index: self.simplify(at) }
            }
            None => ValueExpr::Const(Scalar::zero(self.compute.inputs()[tile.tensor].dtype)),
        };
        let store = Stmt::Store { buffer: tile.buffer, index: store_at, value };
        match step {
            Some((var, _)) => Stmt::For {
                var,
                extent: Dim::new(run),
                kind: ForKind::Vectorized,
                body: Box::new(store),
            },
            None => store,
        }
    }

    /// A new loop variable over `extent`, named for filling `tensor`'s
    /// tile and its `role`, and its expression.
    fn fill_var(&mut self, tensor: TensorParamId, role: &str, extent: Dim) -> (VarId, ExprId) {
        let name = VarName::new(format!("{}_{role}", self.compute.inputs()[tensor].name));
        let var = self.vars.push(KernelVar::Loop(name));
        if let Some(range) = Interval::of_loop(extent) {
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
