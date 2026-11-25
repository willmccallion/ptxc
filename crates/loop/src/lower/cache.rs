//! Lowers the inputs staged in shared memory, as TVM's `cache_read` with
//! `compute_at`. A tile is a box over its coordinates: the input's index,
//! one per input axis, or, reindexed, the fused groups of axes, one per
//! group, as MetaSchedule's `ReIndex` stages a read. Its origin is the
//! coordinates at the first point of the loops it spans, its extent one
//! more than their growth over them. The loop it is staged at fills it with
//! the block's threads between two barriers, and the input's loads read it
//! instead.

use super::context::Lowerer;
use super::reindex::TileSpace;
use crate::compute::{ComputeDef, IterVarId, Scalar, TensorName, TensorParamId};
use crate::expr::{Expr, ExprId, LinearSum, VarId};
use crate::schedule::{CacheId, CacheRead, LoopAnnotation, LoopId, LoopTree, ScheduleError};
use crate::stmt::{BufferId, MemScope, Stmt, ValueExpr};
use ptxc_core::{Dim, Idx};
use ptxc_target::ThreadsPerBlock;

/// A staged input's tile, as lowering fills and reads it.
#[derive(Clone, Debug)]
pub(super) struct Tile {
    pub(super) tensor: TensorParamId,
    pub(super) at: LoopId,
    pub(super) buffer: BufferId,
    /// The space the tile is a box in.
    pub(super) space: TileSpace,
    /// What its loads read outside the input, which the tile holds there.
    pub(super) padding: Option<Scalar>,
    /// The coordinates of the element a load reads, over the kernel's
    /// loops.
    pub(super) coordinates: Vec<ExprId>,
    /// The coordinates of the tile's first element.
    pub(super) origin: Vec<ExprId>,
    pub(super) extents: Vec<Dim>,
    /// The iterations of `at` whose tiles the buffer holds at once, one
    /// after another: the pipeline's stages, or 1.
    pub(super) slots: Dim,
    /// The block's threads, which share the filling.
    pub(super) threads: ThreadsPerBlock,
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

/// The iterations' tiles the buffers of the tiles staged at `at` hold: the
/// stages of a pipeline there, or 1.
pub(crate) fn slots_at(tree: &LoopTree, at: LoopId) -> Dim {
    match tree.loops()[at].annotation {
        LoopAnnotation::Pipelined(stages) => Dim::new(u64::from(stages.get())),
        _ => Dim::new(1),
    }
}

/// The elements of a tile with `extents`. Validation checked that its
/// bytes fit shared memory, so the product does not overflow.
pub(super) fn tile_size(extents: &[Dim]) -> u64 {
    extents.iter().map(|extent| extent.get()).product()
}

impl Lowerer<'_> {
    /// Makes each cache's tile buffer, with the `extents` validation found,
    /// to be filled by `threads` threads.
    pub(super) fn stage(&mut self, extents: &[Vec<Dim>], threads: ThreadsPerBlock) {
        let (compute, tree) = (self.compute, self.tree);
        for (cache, extents) in tree.caches().iter().zip(extents) {
            let spanned = self.spanned(cache.at);
            let (space, coordinates) = self.tile_space(cache);
            let origin = coordinates
                .iter()
                .map(|&coordinate| self.first_point(coordinate, &spanned))
                .collect();
            let input = &compute.inputs()[cache.tensor];
            let name = format!("{}_tile", input.name);
            let slots = slots_at(tree, cache.at);
            let size = slots.get() * tile_size(extents);
            let scope = MemScope::Shared { swizzle: cache.swizzle };
            let buffer = self.scratch(&name, input.dtype, scope, size);
            let tile = Tile {
                tensor: cache.tensor,
                at: cache.at,
                buffer,
                space,
                padding: cache.padding,
                coordinates,
                origin,
                extents: extents.clone(),
                slots,
                threads,
            };
            let _stage: CacheId = self.tiles.push(tile);
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

    /// The space `cache`'s tile is a box in, and its coordinates over the
    /// kernel's loops: the input's index, simplified, or the reindexed
    /// groups' coordinates.
    fn tile_space(&mut self, cache: &CacheRead) -> (TileSpace, Vec<ExprId>) {
        let tree = self.tree;
        if let Some(found) = tree.reindexed().iter().find(|found| found.tensor == cache.tensor) {
            let space =
                TileSpace::Iteration { groups: found.groups.clone(), index: cache.index.clone() };
            return (space, found.coordinates.clone());
        }
        let index = cache
            .index
            .iter()
            .map(|&id| {
                let translated = self.translate(id);
                self.simplify(translated)
            })
            .collect();
        (TileSpace::Input, index)
    }

    /// The extents of `axes`.
    pub(super) fn axis_extents(&self, axes: &[IterVarId]) -> Vec<Dim> {
        axes.iter().map(|&axis| self.compute.iter_vars()[axis].extent).collect()
    }

    /// The tile's extent along each of `cache`'s coordinates.
    fn tile_extents(&mut self, cache: &CacheRead) -> Result<Vec<Dim>, ScheduleError> {
        let compute = self.compute;
        let name = &compute.inputs()[cache.tensor].name;
        let spanned = self.spanned(cache.at);
        let (space, coordinates) = self.tile_space(cache);
        self.check_covered(cache.tensor, &space, &spanned)?;
        coordinates.into_iter().map(|axis| self.tile_extent(axis, &spanned, name)).collect()
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

    /// A load of `tensor` from its tile, if it is staged.
    pub(super) fn tile_load(&mut self, tensor: TensorParamId) -> Option<ValueExpr> {
        let (cache, tile) = self.tiles.iter_enumerated().find(|(_, tile)| tile.tensor == tensor)?;
        let buffer = tile.buffer;
        Some(ValueExpr::Load { buffer, index: self.tile_position(cache) })
    }

    /// The position in stage `cache`'s tile of the element a load of its
    /// input reads.
    pub(super) fn tile_position(&mut self, cache: CacheId) -> ExprId {
        let tile = self.tiles[cache].clone();
        let mut offsets = Vec::new();
        for (&coordinate, &origin) in tile.coordinates.iter().zip(&tile.origin) {
            let negated = self.negated(origin);
            let offset = self.exprs.add(coordinate, negated);
            offsets.push(self.simplify(offset));
        }
        let position = self.linearize(&offsets, &tile.extents);
        let iteration = self.exprs.var(VarId::new(tile.at.index()));
        match self.slot_start(&tile, iteration) {
            Some(start) => self.exprs.add(start, position),
            None => position,
        }
    }

    /// The first element of the copy of `tile` for the iteration of its
    /// loop at `iteration`, in a buffer of several: the copy
    /// `iteration mod slots`. None where it is the buffer's first.
    pub(super) fn slot_start(&mut self, tile: &Tile, iteration: ExprId) -> Option<ExprId> {
        if tile.slots.get() == 1 {
            return None;
        }
        let slots = self.extent(tile.slots);
        let slot = self.exprs.floor_mod(iteration, slots);
        let size = self.extent(Dim::new(tile_size(&tile.extents)));
        let start = self.exprs.mul(slot, size);
        let start = self.simplify(start);
        (self.exprs.get(start) != Expr::Const(0)).then_some(start)
    }

    /// The tiles staged at `at` that `body` reads.
    pub(super) fn tiles_read_in(&self, at: LoopId, body: &Stmt) -> Vec<Tile> {
        self.tiles.iter().filter(|tile| tile.at == at && body.reads(tile.buffer)).cloned().collect()
    }

    /// `-id`.
    pub(super) fn negated(&mut self, id: ExprId) -> ExprId {
        let minus_one = self.constant(-1);
        self.exprs.mul(id, minus_one)
    }
}
