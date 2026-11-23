//! Imports every ONNX node test model (`cargo xtask fetch onnx-tests`
//! first): each one either imports, verifies and round-trips through the
//! text format, or is reported as unsupported. Any other error is a bug.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use prost::Message;
use ptxc_graph::text::{parse, print};
use ptxc_onnx::{ImportErrorKind, ImportOptions, import_model, tensor_from_bytes};
use ptxc_onnx_proto::ModelProto;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const NODE_TESTS: &str = "../../data/onnx/v1.22.0/node";

/// The outcome of importing one test.
enum Outcome {
    Imported,
    Unsupported(String),
    Failed(String),
}

#[test]
#[ignore = "needs the ONNX node tests: cargo xtask fetch onnx-tests"]
fn every_node_test_imports_or_is_reported_unsupported() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(NODE_TESTS);
    let mut tests: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|error| {
            panic!("{}: {error}; run `cargo xtask fetch onnx-tests`", root.display())
        })
        .map(|entry| entry.unwrap().path())
        .collect();
    tests.sort();
    let mut imported = 0;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for test in &tests {
        match import_test(test) {
            Outcome::Imported => imported += 1,
            Outcome::Unsupported(reason) => *reasons.entry(reason).or_default() += 1,
            Outcome::Failed(problem) => {
                failures
                    .push(format!("{}: {problem}", test.file_name().unwrap().to_string_lossy()));
            }
        }
    }
    let unsupported: usize = reasons.values().sum();
    eprintln!(
        "{imported} imported, {unsupported} unsupported, {} failed of {}",
        failures.len(),
        tests.len()
    );
    let mut by_count: Vec<_> = reasons.into_iter().collect();
    by_count.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    for (reason, count) in &by_count {
        eprintln!("  {count:4} {reason}");
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Imports a test's model, binding the inputs a `NeedsConstant` error names
/// to the test's own values, then verifies and round-trips the graph.
fn import_test(test: &Path) -> Outcome {
    let bytes = std::fs::read(test.join("model.onnx")).unwrap();
    let mut options = ImportOptions::default();
    loop {
        match import_model(&bytes, &options) {
            Ok(graph) => return check_graph(&graph),
            Err(error) => match error.kind() {
                ImportErrorKind::Unsupported(reason) => {
                    return Outcome::Unsupported(reason.to_string());
                }
                ImportErrorKind::NeedsConstant { inputs, .. }
                    if !inputs.is_empty()
                        && !inputs
                            .iter()
                            .all(|input| options.constant_inputs.contains_key(input)) =>
                {
                    for input in inputs {
                        let _replaced: Option<_> = options
                            .constant_inputs
                            .insert(input.clone(), test_input(test, &bytes, input));
                    }
                }
                ImportErrorKind::NeedsConstant { operand, .. } => {
                    return Outcome::Unsupported(format!(
                        "value-dependent shape ({})",
                        short(operand)
                    ));
                }
                _ => return Outcome::Failed(error.to_string()),
            },
        }
    }
}

/// The last part of a long generated name.
fn short(name: &str) -> &str {
    name.rsplit('_').next().unwrap_or(name)
}

fn check_graph(graph: &ptxc_graph::Graph) -> Outcome {
    if let Err(error) = graph.verify() {
        return Outcome::Failed(format!("does not verify: {error}"));
    }
    let text = print(graph);
    match parse(&text, &all_constants(graph)) {
        Ok(parsed) if print(&parsed) == text => Outcome::Imported,
        Ok(_) => Outcome::Failed("prints differently after a round trip".to_owned()),
        Err(error) => Outcome::Failed(format!("does not parse back: {error}")),
    }
}

/// Every constant a node or graph output reads, for elided constants in
/// the graph's text.
fn all_constants(graph: &ptxc_graph::Graph) -> Vec<ptxc_core::HostTensor> {
    let operands = graph.nodes().flat_map(|(_, node)| node.operands().iter().copied());
    operands
        .chain(graph.outputs().iter().copied())
        .filter_map(|value| match graph.value(value).def() {
            ptxc_graph::ValueDef::Constant(id) => Some(graph.constant(id).clone()),
            _ => None,
        })
        .collect()
}

/// The test's value for graph input `name`, from `test_data_set_0`.
fn test_input(test: &Path, model_bytes: &[u8], name: &str) -> ptxc_core::HostTensor {
    let model = ModelProto::decode(model_bytes).unwrap();
    let graph = model.graph.unwrap();
    let initializers: Vec<_> =
        graph.initializer.iter().filter_map(|tensor| tensor.name.clone()).collect();
    let position = graph
        .input
        .iter()
        .filter(|info| !initializers.contains(info.name.as_ref().unwrap()))
        .position(|info| info.name.as_deref() == Some(name))
        .unwrap();
    tensor_from_bytes(
        &std::fs::read(test.join(format!("test_data_set_0/input_{position}.pb"))).unwrap(),
    )
    .unwrap()
}
