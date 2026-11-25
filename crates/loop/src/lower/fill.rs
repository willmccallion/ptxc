//! The block's threads filling a staged tile from its input, between two
//! barriers: run `r` of the tile's adjacent elements by thread `r mod T` in
//! round `r div T`. A run is 16, 8 or 4 bytes where the tile and its input
//! allow, started as one asynchronous copy that lands at a wait before the
//! second barrier, zero outside the input (D63); and otherwise one element,
//! loaded and stored.

use super::cache::{Tile, tile_size};
use super::context::Lowerer;
use crate::compute::{Scalar, TensorParamId, VarName};
use crate::expr::{ExprId, Interval, VarId};
use crate::schedule::{LoopAnnotation, ThreadAxis};
use crate::stmt::{BufferId, Condition, ForKind, KernelVar, Stmt, ValueExpr};
use ptxc_core::{Dim, Idx};

/// The runs a thread copies at once where a tile allows, longest first:
/// the sizes of `cp.async`, whose 16-byte copies CUTLASS's tile iterators
/// make.
const RUN_BYTES: [u64; 3] = [16, 8, 4];

/// How a thread copies its run: as one asynchronous copy, or, for one
/// element shorter than any of `RUN_BYTES`, with a load and a store.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunCopy {
    Async,
    Direct,
}

/// The adjacent elements a thread copies at once, and how.
#[derive(Clone, Copy)]
pub(super) struct Run {
    length: u64,
    copy: RunCopy,
}

impl Lowerer<'_> {
    /// `body` after a barrier, the filling of `tiles`, a wait for their
    /// asynchronous copies, and another barrier. The first barrier keeps
    /// the fill from overwriting elements the previous iteration's body may
    /// still be reading.
    pub(super) fn fill_before(&mut self, tiles: &[Tile], body: Stmt) -> Stmt {
        if tiles.is_empty() {
            return body;
        }
        let runs: Vec<Run> = tiles.iter().map(|tile| self.run(tile)).collect();
        let mut steps = vec![Stmt::Barrier];
        for (tile, &run) in tiles.iter().zip(&runs) {
            steps.push(self.fill(tile, run, None));
        }
        if runs.iter().any(|run| run.copy == RunCopy::Async) {
            steps.extend([Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }]);
        }
        steps.extend([Stmt::Barrier, body]);
        Stmt::Seq(steps)
    }

    /// The block's threads copying `tile` from its input in runs of `run`,
    /// into its buffer from `start`, or the first element without one: run
    /// `r` by thread `r mod T` in round `r div T`, for `T` threads. A tile
    /// no larger than the block takes one round, with no loop.
    pub(super) fn fill(&mut self, tile: &Tile, run: Run, start: Option<ExprId>) -> Stmt {
        let runs = Dim::new(tile_size(&tile.extents) / run.length);
        // Validation's thread count is at least 1: an empty loop cannot be bound.
        let rounds = Dim::new(runs.get().div_ceil(u64::from(tile.threads.get())));
        if rounds.get() == 1 {
            let only = self.constant(0);
            return self.fill_round(tile, (only, rounds), (run, start));
        }
        let (round_var, round) = self.fill_var(tile.tensor, "fill", rounds);
        let copy = self.fill_round(tile, (round, rounds), (run, start));
        Stmt::For { var: round_var, extent: rounds, kind: ForKind::Serial, body: Box::new(copy) }
    }

    /// How `tile` is copied: in the longest of `RUN_BYTES` whose length
    /// divides the tile's innermost extent and its origin's innermost
    /// coordinate, so a run lies in one row of the tile, aligned, and whose
    /// elements are adjacent in one row of the input, aligned; and one
    /// element where none is.
    pub(super) fn run(&self, tile: &Tile) -> Run {
        let element = self.compute.inputs()[tile.tensor].dtype.element_size().get();
        let length = match (tile.extents.last(), tile.origin.last()) {
            (Some(extent), Some(&origin)) => {
                let fits = |run: u64| {
                    extent.get().is_multiple_of(run)
                        && self.exprs.is_multiple_of(origin, run)
                        && self.runs_in_input(tile, run)
                };
                RUN_BYTES.iter().map(|bytes| bytes / element).find(|&run| fits(run)).unwrap_or(1)
            }
            _ => 1,
        };
        let copy =
            if RUN_BYTES.contains(&(length * element)) { RunCopy::Async } else { RunCopy::Direct };
        Run { length, copy }
    }

    /// Each thread's copy of its run of `tile` in `round` of `rounds`, into
    /// the buffer from `start`. A run starts at a multiple of its length in
    /// rows of the tile and the input whose extents are multiples of it, so
    /// its first element's bounds are all of its elements'.
    fn fill_round(
        &mut self,
        tile: &Tile,
        (round, rounds): (ExprId, Dim),
        (run, start): (Run, Option<ExprId>),
    ) -> Stmt {
        let index = self.tile_element(round, rounds);
        let first = self.scaled(index, run.length);
        let coordinates = self.coordinates(first, tile);
        let read = self.tile_read(tile, &coordinates);
        let dims = self.compute.inputs()[tile.tensor].shape.dims().to_vec();
        let position = self.linearize(&read, &dims);
        let source = (self.input_buffer(tile.tensor), self.simplify(position));
        let mut in_input = Vec::new();
        for (value, bound) in self.read_ranges(tile, &coordinates, &read).into_iter().rev() {
            in_input.extend(self.open_bounds(value, bound));
        }
        // Unsimplified, so the guard on the run's index bounds the copy.
        let target_index = start.map_or(first, |start| self.exprs.add(start, first));
        let copy = match run.copy {
            RunCopy::Async => Stmt::AsyncCopy {
                target: tile.buffer,
                target_index,
                source: source.0,
                source_index: source.1,
                run: Dim::new(run.length),
                bounds: in_input,
            },
            RunCopy::Direct => self.element_copy(tile, (target_index, source), in_input),
        };
        self.within(index, Dim::new(tile_size(&tile.extents) / run.length), copy)
    }

    /// `tile`'s buffer's element `first` stored from `source`'s, where every
    /// one of `in_input` holds. In a tensorized nest the element is cleared
    /// first, so an `mma` over a tile overhanging its input multiplies zeros
    /// there.
    fn element_copy(
        &self,
        tile: &Tile,
        (first, (buffer, position)): (ExprId, (BufferId, ExprId)),
        in_input: Vec<Condition>,
    ) -> Stmt {
        let value = ValueExpr::Load { buffer, index: position };
        let copied = Stmt::Store { buffer: tile.buffer, index: first, value };
        let guarded = guarded_by(in_input, copied);
        if self.tree.tensorized().is_none() {
            return guarded;
        }
        let zero = ValueExpr::Const(Scalar::zero(self.compute.inputs()[tile.tensor].dtype));
        let cleared = Stmt::Store { buffer: tile.buffer, index: first, value: zero };
        Stmt::Seq(vec![cleared, guarded])
    }

    /// `id * factor`, left unsimplified for `id`'s guard to bound it.
    fn scaled(&mut self, id: ExprId, factor: u64) -> ExprId {
        if factor == 1 {
            return id;
        }
        let factor = self.extent(Dim::new(factor));
        self.exprs.mul(id, factor)
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

    /// The coordinates of tile element `element`: its row-major place in
    /// the tile, offset by the tile's origin.
    fn coordinates(&mut self, element: ExprId, tile: &Tile) -> Vec<ExprId> {
        let locals = self.delinearize(element, &tile.extents);
        tile.origin
            .iter()
            .zip(locals)
            .map(|(&origin, local)| {
                let coordinate = self.exprs.add(origin, local);
                self.simplify(coordinate)
            })
            .collect()
    }

    /// `body` under `0 <= index < bound`, checking only the sides the
    /// loops' ranges leave open.
    pub(super) fn within(&mut self, index: ExprId, bound: Dim, body: Stmt) -> Stmt {
        let open = self.open_bounds(index, bound);
        guarded_by(open, body)
    }

    /// The conditions of `0 <= index < bound` the loops' ranges leave
    /// open, the upper bound first.
    fn open_bounds(&mut self, index: ExprId, bound: Dim) -> Vec<Condition> {
        let known = self.bounds(index);
        let mut open = Vec::new();
        if known.is_none_or(|known| i128::from(known.max()) >= i128::from(bound.get())) {
            open.push(Condition { index, bound: self.extent(bound) });
        }
        if known.is_none_or(|known| known.min() < 0) {
            // `0 <= index` as `-index < 1`: an `If` compares only one way.
            let negated = self.negated(index);
            open.push(Condition { index: self.simplify(negated), bound: self.constant(1) });
        }
        open
    }
}

/// `body` inside an `If` for each of `conditions`, the first innermost.
fn guarded_by(conditions: Vec<Condition>, body: Stmt) -> Stmt {
    conditions
        .into_iter()
        .fold(body, |then, condition| Stmt::If { condition, then: Box::new(then) })
}
