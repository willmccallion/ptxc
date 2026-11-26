//! `Compilation<S>`: the pipeline as a typestate, so a stage cannot run
//! out of order: `Imported`, `Optimized`, `Partitioned`, `Lowered`,
//! `Scheduled`, then a `Program`.

use crate::emit::emit;
use crate::error::CompileError;
use ptxc_core::IndexVec;
use ptxc_graph::{Graph, GroupId, Partitioned as PartitionedGraph};
use ptxc_graph_opt::{GreedyPartitioner, Partitioner, PassManager};
use ptxc_loop::{ComputeDef, LoweredKernel};
use ptxc_lower::LoweredGroup;
use ptxc_onnx::{ImportOptions, import_model};
use ptxc_program::Program;
use ptxc_target::Target;
use std::sync::Arc;

/// The pipeline in state `S`, for `target`.
#[derive(Debug)]
pub struct Compilation<S> {
    state: S,
    target: Target,
}

/// A graph as imported.
#[derive(Debug)]
pub struct Imported {
    graph: Graph,
}

/// A graph after the graph passes.
#[derive(Debug)]
pub struct Optimized {
    graph: Graph,
}

/// A graph split into fused groups.
#[derive(Debug)]
pub struct Partitioned {
    partitioned: PartitionedGraph,
}

/// Each group lowered to stages.
#[derive(Debug)]
pub struct Lowered {
    partitioned: PartitionedGraph,
    groups: IndexVec<GroupId, LoweredGroup>,
}

/// Each stage scheduled and lowered to a kernel, in group then stage
/// order.
#[derive(Debug)]
pub struct Scheduled {
    pub(crate) partitioned: PartitionedGraph,
    pub(crate) groups: IndexVec<GroupId, LoweredGroup>,
    pub(crate) kernels: IndexVec<GroupId, Vec<LoweredKernel>>,
}

impl Compilation<Imported> {
    /// An ONNX model's graph, imported with `options`.
    pub fn import(
        onnx: &[u8],
        options: &ImportOptions,
        target: Target,
    ) -> Result<Self, CompileError> {
        Ok(Self::from_graph(import_model(onnx, options)?, target))
    }

    /// `graph`, as though imported.
    pub const fn from_graph(graph: Graph, target: Target) -> Self {
        Self { state: Imported { graph }, target }
    }

    /// The graph after the graph passes.
    pub fn optimize(self) -> Result<Compilation<Optimized>, CompileError> {
        let Imported { mut graph } = self.state;
        let _reports = PassManager::standard().run(&mut graph)?;
        Ok(Compilation { state: Optimized { graph }, target: self.target })
    }
}

impl Compilation<Optimized> {
    /// The graph split into fused groups by the greedy partitioner (D82).
    pub fn partition(self) -> Result<Compilation<Partitioned>, CompileError> {
        self.partition_with(&GreedyPartitioner)
    }

    /// The graph split into groups by `partitioner`.
    pub fn partition_with(
        self,
        partitioner: &dyn Partitioner,
    ) -> Result<Compilation<Partitioned>, CompileError> {
        let partitioned = partitioner.partition(self.state.graph)?;
        Ok(Compilation { state: Partitioned { partitioned }, target: self.target })
    }
}

impl Compilation<Partitioned> {
    /// Each group lowered to stages.
    pub fn lower(self) -> Result<Compilation<Lowered>, CompileError> {
        let Partitioned { partitioned } = self.state;
        let groups = ptxc_lower::lower(&partitioned)?;
        Ok(Compilation { state: Lowered { partitioned, groups }, target: self.target })
    }
}

impl Compilation<Lowered> {
    /// Each stage under its default schedule, lowered to a kernel, after
    /// each product that leaves the GPU idle is split along its sum (D75).
    // NOTE: ARCHITECTURE.md's `schedule` takes a tuning database; M2 has
    // only the default schedules, and the database comes with M5.
    pub fn schedule(self) -> Result<Compilation<Scheduled>, CompileError> {
        let Lowered { partitioned, groups } = self.state;
        let target = self.target;
        let split = |compute: &Arc<ComputeDef>| ptxc_schedule::split_k(compute, &target);
        let groups = groups
            .into_iter()
            .map(|group| group.with_split_reductions(split))
            .collect::<Result<IndexVec<_, _>, _>>()?;
        let kernels = groups
            .iter()
            .map(|group| {
                group
                    .stages()
                    .iter()
                    .map(|stage| {
                        let compute = Arc::clone(stage.compute());
                        Ok(ptxc_schedule::default_schedule(compute, &target)?.lower())
                    })
                    .collect::<Result<Vec<_>, CompileError>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(Compilation { state: Scheduled { partitioned, groups, kernels }, target })
    }
}

impl Compilation<Scheduled> {
    /// The program: every kernel emitted, every buffer placed.
    pub fn emit(self) -> Result<Program, CompileError> {
        emit(&self.state, &self.target)
    }
}
