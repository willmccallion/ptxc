//! Prints a graph in the text format.

use super::attr::write_attrs;
use super::lexer::is_word_char;
use super::ops::encode;
use super::tensor_text::{write_data, write_type};
use crate::graph::Graph;
use crate::node::NodeId;
use crate::value::{ValueDef, ValueId, ValueName};
use ptxc_core::IndexVec;
use std::collections::HashSet;

/// The graph as text: inputs in the header, then every constant, then the
/// nodes in topological order, then `return`. A cyclic graph, which the
/// verifier rejects, prints its nodes in id order instead.
pub fn print(graph: &Graph) -> String {
    let order =
        graph.topological_order().unwrap_or_else(|_| graph.nodes().map(|(id, _)| id).collect());
    let names = assign_names(graph, &order);
    let mut out = String::from("graph(");
    for (index, &input) in graph.inputs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&names[input]);
        out.push_str(": ");
        write_type(&mut out, &graph.values[input].ty);
    }
    out.push_str(") {\n");
    print_constants(graph, &names, &mut out);
    for node in order {
        print_node(graph, node, &names, &mut out);
    }
    out.push_str("  return ");
    out.push_str(&join(graph.outputs.iter().map(|&value| names[value].as_str())));
    out.push_str("\n}\n");
    out
}

fn print_constants(graph: &Graph, names: &IndexVec<ValueId, String>, out: &mut String) {
    for (value, contents) in graph.values.iter_enumerated() {
        let ValueDef::Constant(constant) = contents.def else { continue };
        out.push_str("  ");
        out.push_str(&names[value]);
        out.push_str(" = const ");
        write_type(out, &contents.ty);
        out.push(' ');
        write_data(out, &graph.constants[constant]);
        out.push('\n');
    }
}

fn print_node(graph: &Graph, id: NodeId, names: &IndexVec<ValueId, String>, out: &mut String) {
    let Some(node) = graph.node(id) else { return };
    let text = encode(&node.op);
    out.push_str("  ");
    out.push_str(&join(node.outputs.iter().map(|&value| names[value].as_str())));
    out.push_str(" = ");
    out.push_str(text.name);
    write_attrs(out, &text.attrs);
    if !node.operands.is_empty() {
        out.push(' ');
        out.push_str(&join(node.operands.iter().map(|&value| names[value].as_str())));
    }
    out.push_str(" : ");
    for (index, &value) in node.outputs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write_type(out, &graph.values[value].ty);
    }
    out.push('\n');
}

fn join<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names.collect::<Vec<_>>().join(", ")
}

/// Each value's printed name, `%` included, assigned in the order the text
/// defines values so a reparsed graph prints the same. Named values keep
/// their name, quoted if it is not bare word characters or starts with a
/// digit; unnamed values, and repeats of a name, get the next number.
fn assign_names(graph: &Graph, order: &[NodeId]) -> IndexVec<ValueId, String> {
    let mut names: IndexVec<ValueId, String> = graph.values.iter().map(|_| String::new()).collect();
    let constants = graph
        .values
        .ids()
        .filter(|&value| matches!(graph.values[value].def, ValueDef::Constant(_)));
    let node_outputs = order
        .iter()
        .filter_map(|&node| graph.node(node))
        .flat_map(|node| node.outputs.iter().copied());
    let mut taken = HashSet::new();
    let mut next_number = 0_usize;
    for value in graph.inputs.iter().copied().chain(constants).chain(node_outputs) {
        names[value] = match graph.values[value].name.as_ref().map(ValueName::as_str) {
            Some(name) if taken.insert(name.to_owned()) => format_name(name),
            _ => {
                next_number += 1;
                format!("%{}", next_number - 1)
            }
        };
    }
    names
}

fn format_name(name: &str) -> String {
    let bare = name.chars().all(is_word_char) && name.starts_with(|c: char| !c.is_ascii_digit());
    if bare {
        return format!("%{name}");
    }
    let escaped = name.replace('\\', "\\\\").replace('"', "\\\"");
    format!("%\"{escaped}\"")
}
