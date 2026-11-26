//! Lowers a group: each node to its stages, then each stage computing a
//! value later stages of the group read inlined into them, so the group
//! runs in as few kernels as inlining allows (D80).

use crate::error::LowerError;
use crate::ops::{NodeContext, lower_node};
use crate::stage::{Binding, IntermediateId, LoweredGroup, Stage, TableId};
use ptxc_core::{HostTensor, Idx, IndexVec};
use ptxc_graph::{FusedGroup, Graph, GroupId, TensorType, ValueId};
use ptxc_loop::{InlineError, Inlined, InlinedInput, TensorParamId, inline};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// `group`, group `id` of `graph`, lowered with its stages inlined into
/// each other where they can be. Each node computes all its outputs; the
/// stages of those nothing reads are dropped with the rest.
pub(crate) fn lower_group(
    graph: &Graph,
    id: GroupId,
    group: &FusedGroup,
) -> Result<LoweredGroup, LowerError> {
    let mut merged = Merged::default();
    for &node in group.nodes() {
        let Some(contents) = graph.node(node) else { continue };
        let outputs: Vec<ValueId> = contents.outputs().iter().copied().collect();
        merged.append(lower_node(&NodeContext::new(graph, node, contents, &outputs))?);
    }
    merged.inline_within(id, group.outputs())?;
    Ok(merged.finish())
}

/// The stages of a group's nodes, in order, with their intermediates and
/// tables numbered in one group.
#[derive(Default)]
struct Merged {
    intermediates: IndexVec<IntermediateId, TensorType>,
    tables: IndexVec<TableId, HostTensor>,
    stages: Vec<Stage>,
}

impl Merged {
    /// Adds a node's stages, its intermediates and tables numbered after
    /// those already added.
    fn append(&mut self, lowered: LoweredGroup) {
        let (intermediates, tables, stages) = lowered.into_parts();
        let (intermediate_base, table_base) = (self.intermediates.len(), self.tables.len());
        let shifted = |binding: Binding| match binding {
            Binding::Value(_) => binding,
            Binding::Intermediate(id) => {
                Binding::Intermediate(IntermediateId::new(id.index() + intermediate_base))
            }
            Binding::Table(id) => Binding::Table(TableId::new(id.index() + table_base)),
        };
        for ty in intermediates {
            let _pushed = self.intermediates.push(ty);
        }
        for table in tables {
            let _pushed = self.tables.push(table);
        }
        for stage in stages {
            let inputs = stage.inputs().iter().map(|&binding| shifted(binding)).collect();
            let outputs = stage.outputs().iter().map(|&binding| shifted(binding)).collect();
            self.stages.push(Stage::new(Arc::clone(stage.compute()), inputs, outputs));
        }
    }

    /// Inlines each stage that computes one graph value alone into the
    /// later stages reading it, where `inline` allows, and drops each stage
    /// whose outputs neither leave the group nor are read by a stage kept.
    ///
    /// Stages are taken from the last, so each gathers the whole chain
    /// before it; and a stage's producers without a reduction go in first,
    /// so a reduction comes in once, as the epilogue of everything after it.
    fn inline_within(&mut self, group: GroupId, exported: &[ValueId]) -> Result<(), LowerError> {
        let producers: HashMap<ValueId, usize> = self
            .stages
            .iter()
            .enumerate()
            .filter_map(|(at, stage)| match stage.outputs() {
                [Binding::Value(value)] => Some((*value, at)),
                _ => None,
            })
            .collect();
        let mut live = vec![false; self.stages.len()];
        for consumer in (0..self.stages.len()).rev() {
            live[consumer] = self.is_live(consumer, &live, exported);
            if live[consumer] {
                self.inline_into(group, consumer, &producers)?;
            }
        }
        let mut position = 0..;
        self.stages.retain(|_| position.next().is_some_and(|at| live[at]));
        Ok(())
    }

    /// Whether stage `at` writes a value the group exports or a tensor a
    /// stage already found `live` reads; only later stages are found so
    /// far, and no stage reads its own outputs.
    fn is_live(&self, at: usize, live: &[bool], exported: &[ValueId]) -> bool {
        let read_by_live = |binding: &Binding| {
            self.stages.iter().zip(live).any(|(reader, &is_live)| {
                is_live && reader.inputs().iter().any(|read| read == binding)
            })
        };
        self.stages[at].outputs().iter().any(|binding| match binding {
            Binding::Value(value) if exported.contains(value) => true,
            other => read_by_live(other),
        })
    }

    /// Inlines into stage `consumer` each earlier stage computing a value it
    /// reads, those without a reduction first, while `inline` allows.
    fn inline_into(
        &mut self,
        group: GroupId,
        consumer: usize,
        producers: &HashMap<ValueId, usize>,
    ) -> Result<(), LowerError> {
        let mut refused = HashSet::new();
        while let Some(value) = self.next_read(consumer, producers, &refused) {
            let producer = &self.stages[producers[&value]];
            let holding: Vec<TensorParamId> = self.stages[consumer]
                .inputs()
                .iter_enumerated()
                .filter(|&(_, &binding)| binding == Binding::Value(value))
                .map(|(param, _)| param)
                .collect();
            match inline(self.stages[consumer].compute(), &holding, producer.compute()) {
                Ok(inlined) => {
                    self.stages[consumer] = rebound(&self.stages[consumer], producer, inlined);
                }
                Err(source @ (InlineError::Compute(_) | InlineError::Mismatch)) => {
                    return Err(LowerError::Inline { group, value, source });
                }
                Err(_) => {
                    let _new = refused.insert(value);
                }
            }
        }
        Ok(())
    }

    /// A value stage `consumer` reads that another stage of the group, an
    /// earlier one, computes alone, unless inlining it was `refused`: one
    /// whose producer has no reduction if any does, else the first read.
    fn next_read(
        &self,
        consumer: usize,
        producers: &HashMap<ValueId, usize>,
        refused: &HashSet<ValueId>,
    ) -> Option<ValueId> {
        let reads = self.stages[consumer].inputs().iter().filter_map(|binding| {
            let Binding::Value(value) = binding else { return None };
            let &at = producers.get(value)?;
            (!refused.contains(value)).then_some((*value, at))
        });
        let reduces = |at: usize| {
            self.stages[at]
                .compute()
                .outputs()
                .iter()
                .any(|output| output.expr.reduction().is_some())
        };
        reads
            .enumerate()
            .min_by_key(|&(order, (_, at))| (reduces(at), order))
            .map(|(_, (value, _))| value)
    }

    fn finish(self) -> LoweredGroup {
        LoweredGroup::new(self.intermediates, self.tables, self.stages)
    }
}

/// The stage `inlined` defines, `producer` inlined into `consumer`: its
/// inputs bound to the tensors the two bound them to.
fn rebound(consumer: &Stage, producer: &Stage, inlined: Inlined) -> Stage {
    let inputs = inlined
        .inputs
        .iter()
        .map(|&input| match input {
            InlinedInput::Consumer(id) => consumer.inputs()[id],
            InlinedInput::Producer(id) => producer.inputs()[id],
        })
        .collect();
    Stage::new(Arc::new(inlined.compute), inputs, consumer.outputs().to_vec())
}
