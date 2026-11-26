//! Constant folding by running the reference interpreter on constant
//! subgraphs.

use crate::pass::{Pass, PassError, PassReport};
use ptxc_core::{ByteSize, DType, HostTensor, Shape, SizeError};
use ptxc_graph::{Graph, Node, NodeId, TensorType, Use, ValueDef, ValueId};
use ptxc_interp::evaluate_op;
use std::collections::HashSet;

/// The bytes a fold may produce however few it reads: enough for the shapes
/// and indices shape computations make, far short of a materialised
/// broadcast.
const SMALL_RESULT: ByteSize = ByteSize::new(4096);

/// Replaces each node whose operands are all constants with constants of
/// its results, as XLA folds with its `HloEvaluator`.
///
/// The reference interpreter computes them (D12), so folding cannot
/// disagree with the golden model.
///
/// A node whose results take more bytes than the constants only it reads,
/// which folding frees, and more than `SMALL_RESULT`, is left to run:
/// folding it would inflate the weights, as materialising a broadcast,
/// dequantizing an int8 weight to f32 or copying a weight another node
/// still reads would.
#[derive(Clone, Copy, Debug)]
pub struct ConstFold;

impl Pass for ConstFold {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let mut changed = false;
        for node in graph.topological_order()? {
            let Some(results) = fold(graph, node)? else { continue };
            let mut rewriter = graph.rewriter();
            for (output, tensor) in results {
                let constant = rewriter.add_constant(tensor);
                rewriter.replace_all_uses(output, constant)?;
            }
            rewriter.erase_node(node)?;
            changed = true;
        }
        graph.rewriter().compact();
        Ok(PassReport { changed })
    }
}

/// Each of `node`'s outputs with its value, if every operand is a constant
/// and the values are small enough to fold.
fn fold(graph: &Graph, node: NodeId) -> Result<Option<Vec<(ValueId, HostTensor)>>, PassError> {
    let Some(contents) = graph.node(node) else { return Ok(None) };
    let mut operands = Vec::new();
    for &operand in contents.operands() {
        let ValueDef::Constant(constant) = graph.value(operand).def() else { return Ok(None) };
        operands.push(graph.constant(constant));
    }
    let types: Vec<&TensorType> =
        contents.outputs().iter().map(|&value| graph.value(value).ty()).collect();
    let freed = freed_bytes(graph, node, contents)?;
    let made = total_bytes(types.iter().map(|ty| (ty.dtype(), ty.shape())))?;
    if made > freed.max(SMALL_RESULT.get()) {
        return Ok(None);
    }
    let results = evaluate_op(contents.op(), &operands, &types)?;
    Ok(Some(contents.outputs().iter().copied().zip(results).collect()))
}

/// The bytes of the constants only `contents`, node `node`, reads, which
/// folding it frees.
fn freed_bytes(graph: &Graph, node: NodeId, contents: &Node) -> Result<u64, SizeError> {
    let distinct: HashSet<ValueId> = contents.operands().iter().copied().collect();
    let only_here = |value: &ValueId| {
        let here = |reader: &Use| matches!(reader, Use::Operand { node: by, .. } if *by == node);
        graph.uses(*value).iter().all(here)
    };
    let freed = distinct.into_iter().filter(only_here).map(|value| graph.value(value).ty());
    total_bytes(freed.map(|ty| (ty.dtype(), ty.shape())))
}

/// The bytes tensors of these element types and shapes take together,
/// saturating at 2^64.
fn total_bytes<'s>(
    mut tensors: impl Iterator<Item = (DType, &'s Shape)>,
) -> Result<u64, SizeError> {
    tensors.try_fold(0_u64, |total, (dtype, shape)| {
        Ok(total.saturating_add(shape.element_count().in_bytes(dtype)?.get()))
    })
}
