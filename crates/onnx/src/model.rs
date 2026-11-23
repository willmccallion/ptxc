//! `ModelProto`-level import: IR version, opset imports, graph inputs and outputs.

use crate::context::{Entry, NodeContext, Values};
use crate::error::{ImportError, ImportErrorKind, Unsupported};
use crate::fold;
use crate::import::lower;
use crate::lowered::{Lowered, Operand};
use crate::opset::OpsetVersion;
use crate::tensor::import_tensor;
use crate::value_info::{check_declared, input_type};
use prost::Message;
use ptxc_core::{Dim, HostTensor};
use ptxc_graph::{Graph, ValueId, ValueName};
use ptxc_onnx_proto::{GraphProto, ModelProto, NodeProto};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

/// What the caller fixes about a model before import.
#[derive(Clone, Debug, Default)]
pub struct ImportOptions {
    /// Sizes for symbolic dimensions, by name, such as a batch size (D3).
    pub symbolic_dims: HashMap<String, Dim>,
    /// Graph inputs fixed to known values, for inputs a shape depends on;
    /// a `NeedsConstant` error names the ones a model needs. They become
    /// constants and are no longer graph inputs.
    pub constant_inputs: HashMap<String, HostTensor>,
    /// The directory external tensor data is read relative to.
    pub model_dir: Option<PathBuf>,
}

/// Imports a serialised `ModelProto`.
pub fn import_model(bytes: &[u8], options: &ImportOptions) -> Result<Graph, ImportError> {
    let model = ModelProto::decode(bytes).map_err(ImportErrorKind::Decode)?;
    import_model_proto(&model, options)
}

/// Imports a decoded `ModelProto`.
pub fn import_model_proto(
    model: &ModelProto,
    options: &ImportOptions,
) -> Result<Graph, ImportError> {
    let opset = OpsetVersion::from_imports(&model.opset_import)?;
    let graph = model.graph.as_ref().ok_or(ImportErrorKind::Missing("graph"))?;
    let mut values = Values::new();
    add_initializers(graph, options, &mut values)?;
    add_inputs(graph, options, &mut values)?;
    for (index, node) in graph.node.iter().enumerate() {
        import_node(node, &mut values, opset).map_err(|error| locate(error, graph, node, index))?;
    }
    finish_outputs(graph, options, values)
}

fn add_initializers(
    graph: &GraphProto,
    options: &ImportOptions,
    values: &mut Values,
) -> Result<(), ImportError> {
    for proto in &graph.initializer {
        let tensor = import_tensor(proto, options.model_dir.as_deref())?;
        values.insert(proto.name.as_deref().unwrap_or_default(), Entry::constant(tensor));
    }
    Ok(())
}

/// Graph inputs, skipping initializers that older models also list as
/// inputs, and making the caller's bound inputs constants.
fn add_inputs(
    graph: &GraphProto,
    options: &ImportOptions,
    values: &mut Values,
) -> Result<(), ImportError> {
    let initializers: HashSet<&str> =
        graph.initializer.iter().filter_map(|proto| proto.name.as_deref()).collect();
    for info in &graph.input {
        let name = info.name.as_deref().ok_or(ImportErrorKind::Missing("input name"))?;
        if initializers.contains(name) {
            continue;
        }
        if let Some(tensor) = options.constant_inputs.get(name) {
            values.insert(name, Entry::constant(tensor.clone()));
            continue;
        }
        let ty = input_type(info, &options.symbolic_dims)?;
        let value = values.builder.add_input(ty.clone());
        values.builder.set_name(value, ValueName::new(name.to_owned()));
        values.insert(name, Entry { ty, value: Some(value), constant: None });
    }
    Ok(())
}

fn import_node(
    node: &NodeProto,
    values: &mut Values,
    opset: Option<OpsetVersion>,
) -> Result<(), ImportError> {
    let op_type = node.op_type.as_deref().unwrap_or_default();
    let domain = node.domain.as_deref().unwrap_or_default();
    if !matches!(domain, "" | "ai.onnx") {
        return Err(Unsupported::Op { domain: domain.to_owned(), op: op_type.to_owned() }.into());
    }
    if matches!(op_type, "If" | "Loop" | "Scan") {
        return Err(Unsupported::ControlFlow(op_type.to_owned()).into());
    }
    let lowered = {
        let opset = opset.ok_or(ImportErrorKind::Missing("a default-domain opset import"))?;
        let mut context = NodeContext::new(node, values, opset);
        let lowered = lower(&mut context)?;
        context.attrs.finish()?;
        lowered
    };
    apply(node, lowered, values)
}

/// Records a node's lowered form: adds its graph node, or its constants, or
/// its alias, under the node's output names. Only values the node creates
/// take those names; an alias keeps its value's name, which may be a graph
/// input's.
fn apply(node: &NodeProto, lowered: Lowered, values: &mut Values) -> Result<(), ImportError> {
    let entries: Vec<Entry> = match lowered {
        Lowered::Alias(index) => vec![values.get(input_name(node, index)?)?.clone()],
        Lowered::Constants(tensors) => tensors.into_iter().map(Entry::constant).collect(),
        Lowered::Node { op, operands } => {
            let operands = resolve_all(node, operands, values)?;
            named_outputs(node, add_node(&op, &operands, values)?, values)
        }
        Lowered::Chain { first, then, rest } => {
            let input = resolve(node, Operand::Input(0), values)?;
            let mut operands = add_node(&first, &[input], values)?;
            operands.truncate(1);
            operands.extend(resolve_all(node, rest, values)?);
            named_outputs(node, add_node(&then, &operands, values)?, values)
        }
    };
    for (index, name) in node.output.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let entry = entries.get(index).cloned().ok_or_else(|| {
            let op = node.op_type.clone().unwrap_or_default();
            ImportError::from(Unsupported::Attribute { op, attribute: format!("output {index}") })
        })?;
        values.insert(name, entry);
    }
    Ok(())
}

/// An operand the builder has: its value and, if known, its contents.
struct Resolved {
    value: ValueId,
    constant: Option<HostTensor>,
}

fn resolve(
    node: &NodeProto,
    operand: Operand,
    values: &mut Values,
) -> Result<Resolved, ImportError> {
    Ok(match operand {
        Operand::Input(index) => {
            let name = input_name(node, index)?;
            let constant = values.get(name)?.constant.clone();
            Resolved { value: values.value(name)?, constant }
        }
        Operand::Constant(tensor) => {
            Resolved { value: values.builder.add_constant(tensor.clone()), constant: Some(tensor) }
        }
    })
}

fn resolve_all(
    node: &NodeProto,
    operands: Vec<Operand>,
    values: &mut Values,
) -> Result<Vec<Resolved>, ImportError> {
    operands.into_iter().map(|operand| resolve(node, operand, values)).collect()
}

/// Adds a graph node, evaluating its outputs when every operand is known.
fn add_node(
    op: &ptxc_graph::Op,
    operands: &[Resolved],
    values: &mut Values,
) -> Result<Vec<Resolved>, ImportError> {
    let ids = operands.iter().map(|operand| operand.value).collect();
    let outputs = values.builder.add_node(op.clone(), ids).map_err(ImportErrorKind::Node)?;
    let types: Vec<_> =
        outputs.iter().map(|&output| values.builder.value_type(output).clone()).collect();
    let known: Option<Vec<&HostTensor>> =
        operands.iter().map(|operand| operand.constant.as_ref()).collect();
    let mut folded =
        known.and_then(|tensors| fold::evaluate(op, &tensors, &types)).map(Vec::into_iter);
    Ok(outputs
        .into_iter()
        .map(|value| Resolved { value, constant: folded.as_mut().and_then(Iterator::next) })
        .collect())
}

/// The entries of a node's outputs, with their values named after them.
fn named_outputs(node: &NodeProto, outputs: Vec<Resolved>, values: &mut Values) -> Vec<Entry> {
    outputs
        .into_iter()
        .zip(&node.output)
        .map(|(output, name)| {
            if !name.is_empty() {
                values.builder.set_name(output.value, ValueName::new(name.clone()));
            }
            let ty = values.builder.value_type(output.value).clone();
            Entry { ty, value: Some(output.value), constant: output.constant }
        })
        .collect()
}

fn input_name(node: &NodeProto, index: usize) -> Result<&str, ImportError> {
    node.input
        .get(index)
        .filter(|name| !name.is_empty())
        .map(String::as_str)
        .ok_or_else(|| ImportErrorKind::Missing("a required input").into())
}

fn finish_outputs(
    graph: &GraphProto,
    options: &ImportOptions,
    mut values: Values,
) -> Result<Graph, ImportError> {
    let mut outputs = Vec::with_capacity(graph.output.len());
    for info in &graph.output {
        let name = info.name.as_deref().ok_or(ImportErrorKind::Missing("output name"))?;
        check_declared(info, &values.get(name)?.ty, &options.symbolic_dims)?;
        outputs.push(values.value(name)?);
    }
    values.builder.finish(outputs).map_err(|error| ImportErrorKind::Node(error).into())
}

/// Names the node an error happened at, and for an operand that must be
/// constant, the graph inputs it is computed from.
fn locate(error: ImportError, graph: &GraphProto, node: &NodeProto, index: usize) -> ImportError {
    let op = node.op_type.as_deref().unwrap_or("?");
    let label = node
        .name
        .as_deref()
        .filter(|name| !name.is_empty())
        .map_or_else(|| format!("{op} #{index}"), |name| format!("{name} ({op})"));
    let error = match error.kind() {
        ImportErrorKind::NeedsConstant { operand, .. } => {
            let operand = operand.clone();
            let inputs = upstream_inputs(graph, &operand).into_iter().collect();
            ImportError::from(ImportErrorKind::NeedsConstant { operand, inputs })
        }
        _ => error,
    };
    error.at_node(label)
}

/// The graph inputs (not initializers) `name` is computed from.
fn upstream_inputs(graph: &GraphProto, name: &str) -> BTreeSet<String> {
    let initializers: HashSet<&str> =
        graph.initializer.iter().filter_map(|proto| proto.name.as_deref()).collect();
    let inputs: HashSet<&str> = graph
        .input
        .iter()
        .filter_map(|info| info.name.as_deref())
        .filter(|n| !initializers.contains(n))
        .collect();
    let producers: HashMap<&str, &NodeProto> = graph
        .node
        .iter()
        .flat_map(|node| node.output.iter().map(move |output| (output.as_str(), node)))
        .collect();
    let mut found = BTreeSet::new();
    let mut pending = vec![name];
    let mut seen = HashSet::new();
    while let Some(current) = pending.pop() {
        if !seen.insert(current) {
            continue;
        }
        if inputs.contains(current) {
            let _new: bool = found.insert(current.to_owned());
        } else if let Some(node) = producers.get(current) {
            pending.extend(node.input.iter().filter(|input| !input.is_empty()).map(String::as_str));
        }
    }
    found
}
