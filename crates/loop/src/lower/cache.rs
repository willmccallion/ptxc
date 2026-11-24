//! Lowers the inputs staged in shared memory, as TVM's `cache_read` with
//! `compute_at`. A tile is a box per input axis: its origin is the staged
//! index at the first point of the loops it spans, its extent one more
//! than the index's growth over them. The loop it is staged at fills it
//! with the block's threads between two barriers, and the input's loads
//! read it instead.

use super::context::Lowerer;
use crate::compute::{ComputeDef, TensorName, TensorParamId, VarName};
use crate::expr::{Expr, ExprId, Interval, LinearSum, VarId};
use crate::schedule::{CacheRead, LoopAnnotation, LoopId, LoopTree, ScheduleError, ThreadAxis};
use crate::stmt::{BufferId, Condition, ForKind, KernelVar, MemScope, Stmt, ValueExpr};
use ptxc_core::{Dim, Idx};
use ptxc_target::ThreadsPerBlock;

/// A staged input's tile, as lowering fills and reads it.
#[derive(Clone, Debug)]
pub(super) struct Tile {
    tensor: TensorParamId,
    at: LoopId,
    buffer: BufferId,
    /// The input's index of the tile's first element, per axis.
    origin: Vec<ExprId>,
    extents: Vec<Dim>,
    /// The block's threads, which share the filling.
    threads: ThreadsPerBlock,
}

/// Each cache's tile extents, per axis of its input, if each staged index
/// grows affinely, with non-negative coefficients, over the loops its tile
/// spans.
pub(crate) fn tile_extents(
    compute: &ComputeDef,
    tree: &LoopTree,
) -> Result<Vec<Vec<Dim>>, ScheduleError> {
    let mut lowerer = Lowerer::new(compute, tree);
    tree.caches().iter().map(|cache| lowerer.tile_extents(cache)).collect()
}

/// `stmt` with `tiles` allocated around it.
pub(super) fn allocate_tiles(tiles: &[Tile], stmt: Stmt) -> Stmt {
    tiles
        .iter()
        .rev()
        .fold(stmt, |body, tile| Stmt::Alloc { buffer: tile.buffer, body: Box::new(body) })
}

/// The elements of a tile with `extents`. Validation checked that its
/// bytes fit shared memory, so the product does not overflow.
fn tile_size(extents: &[Dim]) -> u64 {
    extents.iter().map(|extent| extent.get()).product()
}

impl Lowerer<'_> {
    /// Makes each cache's tile buffer, with the `extents` validation found,
    /// to be filled by `threads` threads.
    pub(super) fn stage(&mut self, extents: &[Vec<Dim>], threads: ThreadsPerBlock) {
        let (compute, tree) = (self.compute, self.tree);
        for (cache, extents) in tree.caches().iter().zip(extents) {
            let spanned = self.spanned(cache.at);
            let index = self.kernel_index(cache);
            let origin = index.into_iter().map(|axis| self.first_point(axis, &spanned)).collect();
            let input = &compute.inputs()[cache.tensor];
            let name = format!("{}_tile", input.name);
            let buffer = self.scratch(&name, input.dtype, MemScope::Shared, tile_size(extents));
            let (tensor, at, extents) = (cache.tensor, cache.at, extents.clone());
            self.tiles.push(Tile { tensor, at, buffer, origin, extents, threads });
        }
    }

    /// The loops a tile at `at` spans, with their extents: the serial loops
    /// inside `at`, and every loop bound to a thread, since the block's
    /// threads share the tile. A loop bound to a block is fixed across it.
    fn spanned(&self, at: LoopId) -> Vec<(VarId, Dim)> {
        let tree = self.tree;
        let annotation = |id: LoopId| tree.loops()[id].annotation;
        let nest = tree.nest().iter().copied();
        let inside = nest.clone().skip_while(|&id| id != at).skip(1);
        let serial_inside =
            inside.filter(|&id| !matches!(annotation(id), LoopAnnotation::Bound(_)));
        let threads = nest.filter(
            |&id| matches!(annotation(id), LoopAnnotation::Bound(axis) if !axis.is_block()),
        );
        serial_inside
            .chain(threads)
            .map(|id| (VarId::new(id.index()), tree.loops()[id].extent))
            .collect()
    }

    /// `cache`'s index over the kernel's loops, simplified.
    fn kernel_index(&mut self, cache: &CacheRead) -> Vec<ExprId> {
        cache
            .index
            .iter()
            .map(|&id| {
                let translated = self.translate(id);
                self.simplify(translated)
            })
            .collect()
    }

    /// The tile's extent along each axis of `cache`'s input.
    fn tile_extents(&mut self, cache: &CacheRead) -> Result<Vec<Dim>, ScheduleError> {
        let compute = self.compute;
        let name = &compute.inputs()[cache.tensor].name;
        let spanned = self.spanned(cache.at);
        let index = self.kernel_index(cache);
        index.into_iter().map(|axis| self.tile_extent(axis, &spanned, name)).collect()
    }

    /// One more than `axis`'s growth over the `spanned` loops, each of which
    /// it must read alone in a term with a non-negative coefficient.
    fn tile_extent(
        &self,
        axis: ExprId,
        spanned: &[(VarId, Dim)],
        name: &TensorName,
    ) -> Result<Dim, ScheduleError> {
        let not_affine = || ScheduleError::CacheNotAffine(name.clone());
        let is_spanned = |var: VarId| spanned.iter().any(|&(spanned, _)| spanned == var);
        let sum = LinearSum::of(&self.exprs, axis).ok_or(ScheduleError::ExtentOverflow)?;
        let mut extent: u64 = 1;
        for (atom, coefficient) in sum.terms() {
            let spanned_loop = match self.exprs.get(atom) {
                Expr::Var(var) => spanned.iter().find(|&&(spanned, _)| spanned == var),
                _ => None,
            };
            let Some(&(_, loop_extent)) = spanned_loop else {
                if self.exprs.vars(atom).into_iter().any(is_spanned) {
                    return Err(not_affine());
                }
                continue;
            };
            let coefficient = u64::try_from(coefficient).map_err(|_| not_affine())?;
            let growth = coefficient.checked_mul(loop_extent.get().saturating_sub(1));
            extent = growth
                .and_then(|growth| extent.checked_add(growth))
                .ok_or(ScheduleError::ExtentOverflow)?;
        }
        Ok(Dim::new(extent))
    }

    /// `axis` at the first point of the `spanned` loops: each at 0.
    fn first_point(&mut self, axis: ExprId, spanned: &[(VarId, Dim)]) -> ExprId {
        let zero = self.constant(0);
        let first =
            spanned.iter().fold(axis, |expr, &(var, _)| self.exprs.substitute(expr, var, zero));
        self.simplify(first)
    }

    /// A load of `tensor` at compute-level `index` from its tile, if it is
    /// staged.
    pub(super) fn tile_load(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
    ) -> Option<ValueExpr> {
        let tile = self.tiles.iter().find(|tile| tile.tensor == tensor)?.clone();
        let mut offsets = Vec::new();
        for (&axis, &origin) in index.iter().zip(&tile.origin) {
            let translated = self.translate(axis);
            let negated = self.negated(origin);
            let offset = self.exprs.add(translated, negated);
            offsets.push(self.simplify(offset));
        }
        let position = self.linearize(&offsets, &tile.extents);
        Some(ValueExpr::Load { buffer: tile.buffer, index: position })
    }

    /// The tiles staged at `at` that `body` reads.
    pub(super) fn tiles_read_in(&self, at: LoopId, body: &Stmt) -> Vec<Tile> {
        self.tiles.iter().filter(|tile| tile.at == at && body.reads(tile.buffer)).cloned().collect()
    }

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
    /// under the bounds the loops' ranges do not already prove.
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
        self.within(element, Dim::new(tile_size(&tile.extents)), in_input)
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
            let bound = tree
                .nest()
                .iter()
                .find(|&&id| tree.loops()[id].annotation == LoopAnnotation::Bound(axis));
            indices.push(match bound {
                Some(&id) => self.exprs.var(VarId::new(id.index())),
                None => self.constant(0),
            });
            extents.push(Dim::new(tree.axis_extent(axis)));
        }
        self.linearize(&indices, &extents)
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

    /// `-id`.
    fn negated(&mut self, id: ExprId) -> ExprId {
        let minus_one = self.constant(-1);
        self.exprs.mul(id, minus_one)
    }
}
