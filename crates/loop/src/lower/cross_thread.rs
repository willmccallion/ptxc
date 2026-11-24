//! Lowers a reduction shared between threads: each thread reduces its
//! share, the partials meet in shared memory, a tree halves them with a
//! barrier between steps, and the first thread of each group stores the
//! result, as TVM's cross-thread allreduce does.

use super::bound_extent;
use super::combine::{Candidate, combine};
use super::context::Lowerer;
use super::reduction::Accumulator;
use crate::compute::TensorExpr;
use crate::expr::{ExprId, VarId};
use crate::schedule::{LoopAnnotation, LoopId, LoopKind, ThreadAxis};
use crate::stmt::{BufferId, Condition, MemScope, Stmt, ValueExpr};
use ptxc_core::{DType, Idx};

/// The thread axes in linear order: x varies fastest.
const THREAD_AXES: [ThreadAxis; 3] =
    [ThreadAxis::ThreadX, ThreadAxis::ThreadY, ThreadAxis::ThreadZ];

impl Lowerer<'_> {
    /// The loops computing the one output when reduction loop `shared` is
    /// bound to a thread axis (schedule validation leaves no serial spatial
    /// loop inside the serial reduction loops).
    pub(super) fn cross_thread(&mut self, reduction: &TensorExpr, shared: LoopId) -> Stmt {
        let serial = self.serial_loops();
        let first = serial
            .iter()
            .position(|&id| self.tree.loops()[id].kind == LoopKind::Reduce)
            .unwrap_or(serial.len());
        let (outer, inner) = serial.split_at(first);
        let flag = matches!(reduction, TensorExpr::ArgReduce { .. })
            .then(|| self.scratch("replace", DType::Bool, MemScope::Local, 1));
        let partial = self.partial_accumulator(reduction, flag);
        let tree = self.shared_partials(reduction, flag);
        let thread = self.thread_index();
        let lane = self.exprs.var(VarId::new(shared.index()));
        let lanes = bound_extent(self.tree.loops()[shared].extent.get());
        let stride = self.axis_stride(shared);

        let mut steps = vec![self.initialise(reduction, &partial)];
        let update = self.partial_update(reduction, &partial);
        let guards = self.tree.guards().to_vec();
        let update = self.wrap_guards(&guards, update);
        steps.push(self.wrap_loops(inner, update));
        steps.extend(Self::copy_partial(&partial, &tree, thread));
        steps.push(Stmt::Barrier);
        let mut half = lanes.next_power_of_two() / 2;
        while half >= 1 {
            steps.push(self.tree_step(
                reduction,
                &tree,
                TreeStep { thread, lane, lanes, half, stride },
            ));
            steps.push(Stmt::Barrier);
            half /= 2;
        }
        let result = ValueExpr::Load { buffer: tree.position.unwrap_or(tree.value), index: thread };
        let store = self.store_output(&result);
        let first_lane = self.constant(1);
        steps.push(Stmt::If {
            condition: Condition { index: lane, bound: first_lane },
            then: Box::new(store),
        });
        if !outer.is_empty() {
            steps.push(Stmt::Barrier);
        }
        let body = Self::allocate(&partial, Stmt::Seq(steps));
        let body = self.wrap_loops(outer, body);
        let shared = Accumulator { flag: None, ..tree };
        Self::allocate(&shared, body)
    }

    /// Each thread's one-element running result, deciding replacements
    /// through `flag`, which the partial owns and allocates.
    fn partial_accumulator(
        &mut self,
        reduction: &TensorExpr,
        flag: Option<BufferId>,
    ) -> Accumulator {
        let dtype = self.reduced_dtype(reduction);
        let index = self.constant(0);
        let value = self.scratch("partial", dtype, MemScope::Local, 1);
        let position = self.arg_position(reduction, "partial_position", MemScope::Local, 1);
        Accumulator { value, position, flag, index }
    }

    /// The block's partials in shared memory, one per thread; its steps
    /// decide replacements through the partial's `flag`.
    fn shared_partials(&mut self, reduction: &TensorExpr, flag: Option<BufferId>) -> Accumulator {
        let threads = u64::from(bound_extent(self.tree.block_threads().unwrap_or(1)));
        let dtype = self.reduced_dtype(reduction);
        let index = self.constant(0);
        let value = self.scratch("partials", dtype, MemScope::Shared { swizzle: None }, threads);
        let shared = MemScope::Shared { swizzle: None };
        let position = self.arg_position(reduction, "partial_positions", shared, threads);
        Accumulator { value, position, flag, index }
    }

    /// The position buffer an arg-reduction also needs.
    fn arg_position(
        &mut self,
        reduction: &TensorExpr,
        name: &str,
        scope: MemScope,
        size: u64,
    ) -> Option<BufferId> {
        matches!(reduction, TensorExpr::ArgReduce { .. })
            .then(|| self.scratch(name, DType::I64, scope, size))
    }

    /// The thread's flat position in its block: x, then y, then z.
    fn thread_index(&mut self) -> ExprId {
        let mut flat = self.constant(0);
        let mut stride = 1_i64;
        for axis in THREAD_AXES {
            let Some(id) = self.bound_loop(axis) else { continue };
            let var = self.exprs.var(VarId::new(id.index()));
            let scale = self.constant(stride);
            let term = self.exprs.mul(var, scale);
            flat = self.exprs.add(flat, term);
            stride *= i64::from(bound_extent(self.tree.loops()[id].extent.get()));
        }
        self.simplify(flat)
    }

    /// How far apart, in the flat thread position, neighbours along the
    /// axis `shared` is bound to are.
    fn axis_stride(&self, shared: LoopId) -> i64 {
        let LoopAnnotation::Bound(axis) = self.tree.loops()[shared].annotation else { return 1 };
        THREAD_AXES
            .iter()
            .take_while(|&&earlier| earlier != axis)
            .map(|&earlier| i64::from(bound_extent(self.tree.axis_extent(earlier))))
            .product()
    }

    fn bound_loop(&self, axis: ThreadAxis) -> Option<LoopId> {
        self.tree
            .nest()
            .iter()
            .copied()
            .find(|&id| self.tree.loops()[id].annotation == LoopAnnotation::Bound(axis))
    }

    /// Folds the body's value into the thread's partial.
    fn partial_update(&mut self, reduction: &TensorExpr, partial: &Accumulator) -> Stmt {
        let unused = ValueExpr::Const(crate::compute::Scalar::bool(false));
        let stores = match reduction {
            TensorExpr::Reduce { op, body, .. } => {
                let value = self.value(body, &unused);
                let dtype = value.dtype(&self.buffers);
                let current = ValueExpr::Load { buffer: partial.value, index: partial.index };
                vec![Stmt::Store {
                    buffer: partial.value,
                    index: partial.index,
                    value: combine(*op, dtype, current, value),
                }]
            }
            TensorExpr::ArgReduce { kind, tie, axis, body } => {
                let value = self.value(body, &unused);
                let dtype = value.dtype(&self.buffers);
                let position =
                    ValueExpr::Index { index: self.tree.axis_index(*axis), dtype: DType::I64 };
                self.arg_update(
                    *kind,
                    *tie,
                    dtype,
                    &Candidate { value, position },
                    partial,
                    partial.index,
                )
            }
            _ => Vec::new(),
        };
        self.wrap_loaded(reduction, Stmt::Seq(stores))
    }

    /// Copies the thread's partial into its shared slot.
    fn copy_partial(partial: &Accumulator, tree: &Accumulator, thread: ExprId) -> Vec<Stmt> {
        let mut copies = vec![Stmt::Store {
            buffer: tree.value,
            index: thread,
            value: ValueExpr::Load { buffer: partial.value, index: partial.index },
        }];
        if let (Some(shared), Some(local)) = (tree.position, partial.position) {
            copies.push(Stmt::Store {
                buffer: shared,
                index: thread,
                value: ValueExpr::Load { buffer: local, index: partial.index },
            });
        }
        copies
    }

    /// One halving step: lanes below `half` whose partner `half` lanes on
    /// exists fold the partner's partial into their own.
    fn tree_step(&mut self, reduction: &TensorExpr, tree: &Accumulator, step: TreeStep) -> Stmt {
        let offset = self.constant(i64::from(step.half) * step.stride);
        let partner = self.exprs.add(step.thread, offset);
        let partner = self.simplify(partner);
        let combine_step = match reduction {
            TensorExpr::Reduce { op, .. } => {
                let dtype = self.reduced_dtype(reduction);
                let mine = ValueExpr::Load { buffer: tree.value, index: step.thread };
                let theirs = ValueExpr::Load { buffer: tree.value, index: partner };
                Stmt::Store {
                    buffer: tree.value,
                    index: step.thread,
                    value: combine(*op, dtype, mine, theirs),
                }
            }
            TensorExpr::ArgReduce { kind, tie, .. } => {
                let dtype = self.reduced_dtype(reduction);
                let Some(positions) = tree.position else { return Stmt::Seq(Vec::new()) };
                let theirs = Candidate {
                    value: ValueExpr::Load { buffer: tree.value, index: partner },
                    position: ValueExpr::Load { buffer: positions, index: partner },
                };
                Stmt::Seq(self.arg_update(*kind, *tie, dtype, &theirs, tree, step.thread))
            }
            _ => Stmt::Seq(Vec::new()),
        };
        let half = self.constant(i64::from(step.half));
        let lanes = self.constant(i64::from(step.lanes));
        let reach = self.exprs.add(step.lane, half);
        let inner = Stmt::If {
            condition: Condition { index: reach, bound: lanes },
            then: Box::new(combine_step),
        };
        Stmt::If { condition: Condition { index: step.lane, bound: half }, then: Box::new(inner) }
    }
}

/// One step of the tree: the thread's flat position and lane, the lanes,
/// the distance to the partner in lanes, and the flat stride of a lane.
#[derive(Clone, Copy)]
struct TreeStep {
    thread: ExprId,
    lane: ExprId,
    lanes: u32,
    half: u32,
    stride: i64,
}
