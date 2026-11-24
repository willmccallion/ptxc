//! The lowering context: the kernel being built, and how compute-level
//! indices, loads and values become kernel-level ones.

use super::cache::{Tile, allocate_tiles};
use crate::compute::{ComputeDef, IndexVar, TensorExpr, TensorParamId};
use crate::expr::{Expr, ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
use crate::schedule::{Guard, LoopAnnotation, LoopId, LoopKind, LoopTree};
use crate::stmt::{
    Access, Buffer, BufferId, Condition, ForKind, KernelParam, KernelVar, MemScope, Stmt, ValueExpr,
};
use ptxc_core::{DType, Dim, ElementCount, Idx, IndexVec, Shape};
use std::collections::{BTreeSet, HashMap};

/// Whether gather-index collection descends into a reduction's body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReductionBodies {
    Include,
    Skip,
}

/// The kernel under construction. Its arena starts as a copy of the loop
/// tree's, so axis indices and guards keep their ids; kernel variable `v`
/// is loop `v` for every loop the tree made.
pub(super) struct Lowerer<'s> {
    pub(super) compute: &'s ComputeDef,
    pub(super) tree: &'s LoopTree,
    pub(super) exprs: ExprArena,
    pub(super) vars: IndexVec<VarId, KernelVar>,
    pub(super) buffers: IndexVec<BufferId, Buffer>,
    pub(super) params: Vec<KernelParam>,
    inputs: IndexVec<TensorParamId, BufferId>,
    pub(super) outputs: Vec<BufferId>,
    loaded: HashMap<VarId, VarId>,
    ranges: VarRanges,
    /// The tiles of the inputs staged in shared memory.
    pub(super) tiles: Vec<Tile>,
}

impl<'s> Lowerer<'s> {
    pub(super) fn new(compute: &'s ComputeDef, tree: &'s LoopTree) -> Self {
        let mut vars = IndexVec::new();
        for found in tree.loops() {
            let var = match found.annotation {
                LoopAnnotation::Bound(axis) => KernelVar::Thread { name: found.name.clone(), axis },
                _ => KernelVar::Loop(found.name.clone()),
            };
            let _id: VarId = vars.push(var);
        }
        let mut lowerer = Self {
            compute,
            tree,
            exprs: tree.exprs().clone(),
            vars,
            buffers: IndexVec::new(),
            params: Vec::new(),
            inputs: IndexVec::new(),
            outputs: Vec::new(),
            loaded: HashMap::new(),
            ranges: tree.ranges(),
            tiles: Vec::new(),
        };
        lowerer.add_params();
        lowerer
    }

    fn add_params(&mut self) {
        for param in self.compute.inputs() {
            let buffer = self.global(param.name.clone(), param.dtype, &param.shape, Access::Read);
            let _input: TensorParamId = self.inputs.push(buffer);
        }
        let shape = self.compute.output_shape().clone();
        for output in self.compute.outputs() {
            let buffer = self.global(output.name.clone(), output.dtype, &shape, Access::Write);
            self.outputs.push(buffer);
        }
    }

    fn global(
        &mut self,
        name: crate::compute::TensorName,
        dtype: DType,
        shape: &Shape,
        access: Access,
    ) -> BufferId {
        let size = shape.element_count();
        let buffer = self.buffers.push(Buffer { name, dtype, scope: MemScope::Global, size });
        self.params.push(KernelParam { buffer, access });
        buffer
    }

    /// A new shared or local buffer.
    pub(super) fn scratch(
        &mut self,
        name: &str,
        dtype: DType,
        scope: MemScope,
        size: u64,
    ) -> BufferId {
        let name = crate::compute::TensorName::new(name);
        self.buffers.push(Buffer { name, dtype, scope, size: ElementCount::new(size) })
    }

    /// `tensor`'s global buffer.
    pub(super) fn input_buffer(&self, tensor: TensorParamId) -> BufferId {
        self.inputs[tensor]
    }

    /// Adds `var`'s range, for simplifying expressions that read it.
    pub(super) fn insert_range(&mut self, var: VarId, range: Interval) {
        self.ranges.insert(var, range);
    }

    /// Bounds of `id`'s value over the variables' ranges, if they have them.
    pub(super) fn bounds(&self, id: ExprId) -> Option<Interval> {
        self.exprs.bounds(id, &self.ranges)
    }

    /// `expr`, a constant, in the kernel arena.
    pub(super) fn constant(&mut self, value: i64) -> ExprId {
        self.exprs.constant(value)
    }

    /// `id` simplified over the loops' ranges.
    pub(super) fn simplify(&mut self, id: ExprId) -> ExprId {
        Simplifier::new(&self.ranges).simplify(&mut self.exprs, id)
    }

    /// The kernel expression for compute index expression `id`: axes become
    /// their indices over the loops, and loaded indices kernel variables.
    pub(super) fn translate(&mut self, id: ExprId) -> ExprId {
        let pair = |this: &mut Self, a, b| (this.translate(a), this.translate(b));
        match self.compute.exprs().get(id) {
            Expr::Var(var) => match &self.compute.vars()[var] {
                IndexVar::Iter(axis) => self.tree.axis_index(*axis),
                IndexVar::Loaded { .. } => {
                    let kernel_var = self.loaded_var(var);
                    self.exprs.var(kernel_var)
                }
            },
            Expr::Const(value) => self.exprs.constant(value),
            Expr::Add(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.add(a, b)
            }
            Expr::Mul(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.mul(a, b)
            }
            Expr::FloorDiv(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.floor_div(a, b)
            }
            Expr::FloorMod(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.floor_mod(a, b)
            }
            Expr::Min(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.min(a, b)
            }
            Expr::Max(a, b) => {
                let (a, b) = pair(self, a, b);
                self.exprs.max(a, b)
            }
        }
    }

    /// The kernel variable for compute gather index `var`, made once.
    fn loaded_var(&mut self, var: VarId) -> VarId {
        if let Some(&kernel_var) = self.loaded.get(&var) {
            return kernel_var;
        }
        let kernel_var = self.vars.push(KernelVar::Loaded(self.compute.var_name(var).clone()));
        let _new: Option<VarId> = self.loaded.insert(var, kernel_var);
        kernel_var
    }

    /// `dim` as an index constant. It fits: `ComputeBuilder::finish`
    /// rejects extents and tensor sizes past `i64::MAX`.
    pub(super) fn extent(&mut self, dim: Dim) -> ExprId {
        self.exprs.constant(i64::try_from(dim.get()).unwrap_or(i64::MAX))
    }

    /// The row-major flat position of `indices` in `dims`, simplified.
    pub(super) fn linearize(&mut self, indices: &[ExprId], dims: &[Dim]) -> ExprId {
        let mut flat = self.constant(0);
        for (&index, &dim) in indices.iter().zip(dims) {
            let extent = self.extent(dim);
            let scaled = self.exprs.mul(flat, extent);
            flat = self.exprs.add(scaled, index);
        }
        self.simplify(flat)
    }

    /// The flat position in `tensor` of the compute-level `index`.
    fn input_position(&mut self, tensor: TensorParamId, index: &[ExprId]) -> ExprId {
        let translated: Vec<ExprId> = index.iter().map(|&id| self.translate(id)).collect();
        let dims = self.compute.inputs()[tensor].shape.dims().to_vec();
        self.linearize(&translated, &dims)
    }

    /// The flat output position of the current spatial point.
    pub(super) fn output_position(&mut self) -> ExprId {
        let indices: Vec<ExprId> = self
            .compute
            .iter_vars()
            .iter_enumerated()
            .filter(|(_, iter)| iter.kind == crate::compute::IterKind::Spatial)
            .map(|(axis, _)| self.tree.axis_index(axis))
            .collect();
        let dims = self.compute.output_shape().dims().to_vec();
        self.linearize(&indices, &dims)
    }

    /// `expr` as a kernel value. `reduced` stands in for the definition's
    /// one reduction node where `expr` holds it; a reduction's body and an
    /// elementwise output hold none (D40), so there it is never read.
    pub(super) fn value(&mut self, expr: &TensorExpr, reduced: &ValueExpr) -> ValueExpr {
        let sub = |this: &mut Self, operand: &TensorExpr| Box::new(this.value(operand, reduced));
        match expr {
            TensorExpr::Load { tensor, index } => {
                if let Some(tile) = self.tile_load(*tensor, index) {
                    return tile;
                }
                let index = self.input_position(*tensor, index);
                ValueExpr::Load { buffer: self.inputs[*tensor], index }
            }
            TensorExpr::Const(scalar) => ValueExpr::Const(*scalar),
            TensorExpr::Index { index, dtype } => {
                ValueExpr::Index { index: self.translate(*index), dtype: *dtype }
            }
            TensorExpr::Unary(op, operand) => ValueExpr::Unary(*op, sub(self, operand)),
            TensorExpr::Binary(op, lhs, rhs) => {
                ValueExpr::Binary(*op, sub(self, lhs), sub(self, rhs))
            }
            TensorExpr::Cast { to, overflow, value } => {
                ValueExpr::Cast { to: *to, overflow: *overflow, value: sub(self, value) }
            }
            TensorExpr::Select { condition, then, otherwise } => ValueExpr::Select {
                condition: sub(self, condition),
                then: sub(self, then),
                otherwise: sub(self, otherwise),
            },
            TensorExpr::Reduce { .. } | TensorExpr::ArgReduce { .. } => reduced.clone(),
        }
    }

    /// `body` inside the loops of `loops`, outermost first, each loop that
    /// a tile is staged at filling it first.
    pub(super) fn wrap_loops(&mut self, loops: &[LoopId], body: Stmt) -> Stmt {
        loops.iter().rev().fold(body, |body, &id| {
            let found = &self.tree.loops()[id];
            let kind = match found.annotation {
                LoopAnnotation::Unrolled => ForKind::Unrolled,
                LoopAnnotation::Vectorized => ForKind::Vectorized,
                LoopAnnotation::Serial | LoopAnnotation::Bound(_) => ForKind::Serial,
            };
            let extent = found.extent;
            let tiles = self.tiles_read_in(id, &body);
            let body = self.fill_before(&tiles, body);
            let looped =
                Stmt::For { var: VarId::new(id.index()), extent, kind, body: Box::new(body) };
            allocate_tiles(&tiles, looped)
        })
    }

    /// `body` inside an `If` per guard.
    pub(super) fn wrap_guards<'g>(
        &mut self,
        guards: impl IntoIterator<Item = &'g Guard>,
        body: Stmt,
    ) -> Stmt {
        let guards: Vec<Guard> = guards.into_iter().copied().collect();
        guards.iter().rev().fold(body, |body, guard| {
            let bound = self.extent(guard.extent);
            Stmt::If { condition: Condition { index: guard.index, bound }, then: Box::new(body) }
        })
    }

    /// `body` inside a `LoadIndex` for every gather index `expr` reads,
    /// directly or through another gather index, earliest first.
    pub(super) fn wrap_loaded(&mut self, expr: &TensorExpr, body: Stmt) -> Stmt {
        self.wrap_loaded_in(expr, ReductionBodies::Include, body)
    }

    /// As `wrap_loaded`, but only for the indices read outside the
    /// reduction's body, which is all that is bound after the reduction.
    pub(super) fn wrap_loaded_outside_reduction(&mut self, expr: &TensorExpr, body: Stmt) -> Stmt {
        self.wrap_loaded_in(expr, ReductionBodies::Skip, body)
    }

    fn wrap_loaded_in(&mut self, expr: &TensorExpr, bodies: ReductionBodies, body: Stmt) -> Stmt {
        let mut used = BTreeSet::new();
        self.collect_loaded(expr, bodies, &mut used);
        used.into_iter().rev().fold(body, |body, var| {
            let IndexVar::Loaded { tensor, index, .. } = self.compute.vars()[var].clone() else {
                return body;
            };
            let kernel_var = self.loaded_var(var);
            let position = self.input_position(tensor, &index);
            Stmt::LoadIndex {
                var: kernel_var,
                buffer: self.inputs[tensor],
                index: position,
                body: Box::new(body),
            }
        })
    }

    fn collect_loaded(
        &self,
        expr: &TensorExpr,
        bodies: ReductionBodies,
        used: &mut BTreeSet<VarId>,
    ) {
        match expr {
            TensorExpr::Load { index, .. } => {
                for &id in index {
                    self.collect_index_loads(id, used);
                }
            }
            TensorExpr::Const(_) => {}
            TensorExpr::Index { index, .. } => self.collect_index_loads(*index, used),
            TensorExpr::Unary(_, value) | TensorExpr::Cast { value, .. } => {
                self.collect_loaded(value, bodies, used);
            }
            TensorExpr::Reduce { body, .. } | TensorExpr::ArgReduce { body, .. } => {
                if bodies == ReductionBodies::Include {
                    self.collect_loaded(body, bodies, used);
                }
            }
            TensorExpr::Binary(_, lhs, rhs) => {
                self.collect_loaded(lhs, bodies, used);
                self.collect_loaded(rhs, bodies, used);
            }
            TensorExpr::Select { condition, then, otherwise } => {
                self.collect_loaded(condition, bodies, used);
                self.collect_loaded(then, bodies, used);
                self.collect_loaded(otherwise, bodies, used);
            }
        }
    }

    fn collect_index_loads(&self, id: ExprId, used: &mut BTreeSet<VarId>) {
        for var in self.compute.exprs().vars(id) {
            if let IndexVar::Loaded { index, .. } = &self.compute.vars()[var]
                && used.insert(var)
            {
                for &inner in index {
                    self.collect_index_loads(inner, used);
                }
            }
        }
    }

    /// The loops of the nest that no axis binds, outermost first.
    pub(super) fn serial_loops(&self) -> Vec<LoopId> {
        self.tree
            .nest()
            .iter()
            .copied()
            .filter(|&id| !matches!(self.tree.loops()[id].annotation, LoopAnnotation::Bound(_)))
            .collect()
    }

    /// Whether a guard reads a reduction loop.
    pub(super) fn reads_reduction(&self, guard: &Guard) -> bool {
        self.tree
            .exprs()
            .vars(guard.index)
            .iter()
            .any(|var| self.tree.loops()[LoopId::new(var.index())].kind == LoopKind::Reduce)
    }
}
