//! Constant folding by running the reference interpreter on constant
//! subgraphs.

use crate::pass::{Pass, PassError, PassReport};
use ptxc_core::{ByteSize, DType, Shape, SizeError};
use ptxc_graph::{Graph, Node, NodeId, TensorType, Use, ValueDef, ValueId};
use ptxc_interp::evaluate_op;
use std::collections::{HashMap, HashSet};

/// The bytes a fold may produce however few it reads: enough for the shapes
/// and indices shape computations make, far short of a materialised
/// broadcast.
const SMALL_RESULT: ByteSize = ByteSize::new(4096);

/// Replaces each node computed from constants alone with constants of its
/// results, as XLA folds with its `HloEvaluator`.
///
/// The reference interpreter computes them (D12), so folding cannot
/// disagree with the golden model.
///
/// Folding must not inflate the weights, as materialising a broadcast,
/// dequantizing an int8 weight to f32 or copying a weight another node
/// still reads would. So the nodes computed from constants are weighed in
/// cones, joined by the values they pass and the large constants they
/// share (D85): a cone folds whole if the results it leaves take no more bytes
/// than the constants only it reads, which folding frees, or than
/// `SMALL_RESULT`. A cone that does not fit is folded node by node under
/// the same rule.
#[derive(Clone, Copy, Debug)]
pub struct ConstFold;

impl Pass for ConstFold {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let mut changed = false;
        for cone in cones(graph, &graph.topological_order()?) {
            let whole = fits(graph, &cone, &cone_results(graph, &cone))?;
            for &node in &cone {
                let outputs = graph.node(node).map(|n| n.outputs().iter().copied().collect());
                let outputs: Vec<ValueId> = outputs.unwrap_or_default();
                if whole || fits(graph, &[node], &outputs)? {
                    changed |= fold(graph, node)?;
                }
            }
        }
        graph.rewriter().compact();
        Ok(PassReport { changed })
    }
}

/// The nodes computed from constants alone, in `order`, grouped into cones:
/// two share a cone if one reads the other's output or both read one
/// constant larger than `SMALL_RESULT`. A smaller shared constant joins
/// nothing: its bytes are within the allowance either way, and joining by
/// it would pool unrelated computations' freed bytes. Each cone is in
/// `order`, and the cones in the order of their first nodes.
fn cones(graph: &Graph, order: &[NodeId]) -> Vec<Vec<NodeId>> {
    let mut parent: Vec<usize> = Vec::new();
    let mut set_of: HashMap<NodeId, usize> = HashMap::new();
    let mut first_reader: HashMap<ValueId, usize> = HashMap::new();
    for &node in order {
        let Some(links) = graph.node(node).and_then(|contents| links(graph, contents, &set_of))
        else {
            continue;
        };
        let set = parent.len();
        parent.push(set);
        for link in links {
            let other = match link {
                Link::Node(other) => other,
                Link::Constant(value) => *first_reader.entry(value).or_insert(set),
            };
            let (root, other) = (find(&mut parent, set), find(&mut parent, other));
            parent[other] = root;
        }
        let _new = set_of.insert(node, set);
    }
    let mut grouped: Vec<(usize, Vec<NodeId>)> = Vec::new();
    for &node in order {
        let Some(&set) = set_of.get(&node) else { continue };
        let root = find(&mut parent, set);
        match grouped.iter_mut().find(|(other, _)| *other == root) {
            Some((_, cone)) => cone.push(node),
            None => grouped.push((root, vec![node])),
        }
    }
    grouped.into_iter().map(|(_, cone)| cone).collect()
}

/// What joins a node to a cone: a node it reads, by its set, or a constant.
enum Link {
    Node(usize),
    Constant(ValueId),
}

/// The links of a node reading `contents`' operands, if each is a constant
/// or an output of a node in `set_of`: the nodes it reads, and the
/// constants larger than `SMALL_RESULT`.
fn links(graph: &Graph, contents: &Node, set_of: &HashMap<NodeId, usize>) -> Option<Vec<Link>> {
    let mut links = Vec::new();
    for &operand in contents.operands() {
        match graph.value(operand).def() {
            ValueDef::Constant(_) if is_large(graph, operand) => {
                links.push(Link::Constant(operand));
            }
            ValueDef::Constant(_) => {}
            ValueDef::Output { node, .. } => links.push(Link::Node(*set_of.get(&node)?)),
            ValueDef::Input(_) => return None,
        }
    }
    Some(links)
}

/// Whether `value` takes more bytes than `SMALL_RESULT`.
fn is_large(graph: &Graph, value: ValueId) -> bool {
    let (dtype, shape) = typed(graph, value);
    shape.element_count().in_bytes(dtype).map_or(true, |bytes| bytes.get() > SMALL_RESULT.get())
}

/// The root of `set`'s tree in `parent`, a union-find forest.
const fn find(parent: &mut [usize], set: usize) -> usize {
    let mut root = set;
    while parent[root] != root {
        root = parent[root];
    }
    parent[set] = root;
    root
}

fn is_constant(graph: &Graph, value: ValueId) -> bool {
    matches!(graph.value(value).def(), ValueDef::Constant(_))
}

/// The values of `cone`'s nodes something outside it reads: what folding
/// it leaves.
fn cone_results(graph: &Graph, cone: &[NodeId]) -> Vec<ValueId> {
    let inside: HashSet<NodeId> = cone.iter().copied().collect();
    let outputs = cone.iter().filter_map(|&node| graph.node(node)).flat_map(|n| n.outputs().iter());
    outputs
        .copied()
        .filter(|&value| {
            graph.uses(value).iter().any(|reader| match reader {
                Use::Operand { node, .. } => !inside.contains(node),
                Use::GraphOutput(_) => true,
            })
        })
        .collect()
}

/// Whether folding `nodes`, leaving `results`, takes no more bytes than it
/// frees, or than `SMALL_RESULT`: the constants only `nodes` read are
/// freed.
fn fits(graph: &Graph, nodes: &[NodeId], results: &[ValueId]) -> Result<bool, SizeError> {
    let inside: HashSet<NodeId> = nodes.iter().copied().collect();
    let read: HashSet<ValueId> = nodes
        .iter()
        .filter_map(|&node| graph.node(node))
        .flat_map(|contents| contents.operands().iter().copied())
        .filter(|&value| is_constant(graph, value))
        .collect();
    let only_inside = |value: &ValueId| {
        graph
            .uses(*value)
            .iter()
            .all(|reader| matches!(reader, Use::Operand { node, .. } if inside.contains(node)))
    };
    let freed = total_bytes(read.iter().filter(|v| only_inside(v)).map(|&v| typed(graph, v)))?;
    let made = total_bytes(results.iter().map(|&v| typed(graph, v)))?;
    Ok(made <= freed.max(SMALL_RESULT.get()))
}

fn typed(graph: &Graph, value: ValueId) -> (DType, &Shape) {
    let ty = graph.value(value).ty();
    (ty.dtype(), ty.shape())
}

/// Replaces `node`'s outputs with constants of their values, if every
/// operand is a constant; whether it did.
fn fold(graph: &mut Graph, node: NodeId) -> Result<bool, PassError> {
    let Some(contents) = graph.node(node) else { return Ok(false) };
    let mut operands = Vec::new();
    for &operand in contents.operands() {
        let ValueDef::Constant(constant) = graph.value(operand).def() else { return Ok(false) };
        operands.push(graph.constant(constant));
    }
    let outputs: Vec<ValueId> = contents.outputs().iter().copied().collect();
    let types: Vec<&TensorType> = outputs.iter().map(|&value| graph.value(value).ty()).collect();
    let results = evaluate_op(contents.op(), &operands, &types)?;
    let mut rewriter = graph.rewriter();
    for (output, tensor) in outputs.into_iter().zip(results) {
        let constant = rewriter.add_constant(tensor);
        rewriter.replace_all_uses(output, constant)?;
    }
    rewriter.erase_node(node)?;
    Ok(true)
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
