//! Model-level rules: inputs, opsets, domains, attributes, constants that
//! shapes need, declared types, and where errors are reported.

use crate::builders::{
    D, FLOAT, INT64, fixed, import, import_with, int, ints, model, model_at, node, text, value_info,
};
use ptxc_core::{Dim, Shape};
use ptxc_onnx::{ImportErrorKind, ImportOptions, Unsupported};
use ptxc_onnx_proto::{ModelProto, OperatorSetIdProto};

fn relu_model(input: ptxc_onnx_proto::ValueInfoProto) -> ModelProto {
    model(vec![node("Relu", &["x"], &["y"], vec![])], vec![input], vec![], vec![])
}

fn unsupported(model: &ModelProto) -> Unsupported {
    let error = import(model).unwrap_err();
    error.unsupported().cloned().unwrap_or_else(|| panic!("not unsupported: {error}"))
}

#[test]
fn symbolic_dimensions_take_the_supplied_sizes() {
    let model = relu_model(value_info("x", FLOAT, &[D::S("batch"), D::N(3)]));
    let options = ImportOptions {
        symbolic_dims: [("batch".to_owned(), Dim::new(8))].into(),
        ..ImportOptions::default()
    };

    let graph = import_with(&model, &options).unwrap();

    let input = graph.inputs().iter().copied().next().unwrap();
    assert_eq!(
        graph.value(input).ty().shape(),
        &Shape::new(vec![Dim::new(8), Dim::new(3)]).unwrap()
    );
}

#[test]
fn a_symbolic_dimension_without_a_size_is_unsupported() {
    let model = relu_model(value_info("x", FLOAT, &[D::S("batch")]));

    assert_eq!(unsupported(&model), Unsupported::DynamicShape("batch".to_owned()));
}

#[test]
fn opsets_below_13_are_unsupported() {
    let model = model_at(
        12,
        vec![node("Relu", &["x"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );

    assert_eq!(unsupported(&model), Unsupported::Opset(12));
}

#[test]
fn ops_of_other_domains_are_unsupported_even_without_a_default_opset() {
    let mut model = relu_model(fixed("x", FLOAT, &[2]));
    model.opset_import =
        vec![OperatorSetIdProto { domain: Some("ai.onnx.ml".to_owned()), version: Some(3) }];
    model.graph.as_mut().unwrap().node[0].domain = Some("ai.onnx.ml".to_owned());

    assert_eq!(
        unsupported(&model),
        Unsupported::Op { domain: "ai.onnx.ml".to_owned(), op: "Relu".to_owned() }
    );
}

#[test]
fn control_flow_and_unknown_ops_are_unsupported() {
    let looped = model(
        vec![node("Loop", &["x"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );
    let unknown = model(
        vec![node("Frobnicate", &["x"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );

    assert_eq!(unsupported(&looped), Unsupported::ControlFlow("Loop".to_owned()));
    assert_eq!(
        unsupported(&unknown),
        Unsupported::Op { domain: String::new(), op: "Frobnicate".to_owned() }
    );
}

#[test]
fn an_attribute_no_importer_reads_is_unsupported() {
    let model = model(
        vec![node("Relu", &["x"], &["y"], vec![int("alpha", 1)])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );

    assert_eq!(
        unsupported(&model),
        Unsupported::Attribute { op: "Relu".to_owned(), attribute: "alpha".to_owned() }
    );
}

#[test]
fn an_attribute_of_the_wrong_type_is_invalid() {
    let model = model(
        vec![node("LeakyRelu", &["x"], &["y"], vec![int("alpha", 1)])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );

    let error = import(&model).unwrap_err();

    assert!(
        matches!(error.kind(), ImportErrorKind::Invalid { field: "attribute type", .. }),
        "{error}"
    );
    assert_eq!(error.node(), Some("LeakyRelu #0"));
}

#[test]
fn a_declared_output_that_differs_from_the_inferred_type_is_an_error() {
    let model = model(
        vec![node("Relu", &["x"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![fixed("y", FLOAT, &[3])],
    );

    let error = import(&model).unwrap_err();

    assert_eq!(error.to_string(), "`y` is declared as FLOAT[3] but has type f32[2]");
}

#[test]
fn identity_and_constants_add_no_nodes() {
    let nodes = vec![
        node("Identity", &["x"], &["y"], vec![]),
        node("Constant", &[], &["c"], vec![ints("value_ints", &[1, 2])]),
    ];
    let model = model(
        nodes,
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![fixed("y", FLOAT, &[2]), fixed("c", INT64, &[2])],
    );

    let graph = text(&model);

    assert_eq!(graph, "graph(%x: f32[2]) {\n  %c = const i64[2] [1, 2]\n  return %x, %c\n}\n");
}

#[test]
fn an_undefined_value_is_reported() {
    let model = model(
        vec![node("Relu", &["missing"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );

    let error = import(&model).unwrap_err();

    assert!(
        matches!(error.kind(), ImportErrorKind::UndefinedValue(name) if name == "missing"),
        "{error}"
    );
}

#[test]
fn optional_outputs_ptxc_does_not_produce_are_unsupported() {
    let dropout = model(
        vec![node("Dropout", &["x"], &["y", "mask"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    );
    let pool = model(
        vec![node("MaxPool", &["x"], &["y", "i"], vec![ints("kernel_shape", &[1])])],
        vec![fixed("x", FLOAT, &[1, 1, 2])],
        vec![],
        vec![],
    );

    assert_eq!(
        unsupported(&dropout),
        Unsupported::Attribute { op: "Dropout".to_owned(), attribute: "mask".to_owned() }
    );
    assert_eq!(
        unsupported(&pool),
        Unsupported::Attribute { op: "MaxPool".to_owned(), attribute: "Indices".to_owned() }
    );
}

#[test]
fn range_of_a_type_ptxc_does_not_evaluate_is_unsupported() {
    let scalar = |name: &str| ptxc_onnx_proto::TensorProto {
        name: Some(name.to_owned()),
        data_type: Some(10),
        int32_data: vec![0x3c00],
        ..Default::default()
    };
    let model = model(
        vec![node("Range", &["a", "b", "c"], &["y"], vec![])],
        vec![],
        vec![scalar("a"), scalar("b"), scalar("c")],
        vec![],
    );

    assert_eq!(unsupported(&model), Unsupported::DType("F16 for Range".to_owned()));
}

#[test]
fn errors_name_the_node_by_its_name_or_its_op_and_position() {
    let mut named = node("Relu", &["missing"], &["y"], vec![]);
    named.name = Some("act1".to_owned());
    let unnamed = node("Relu", &["missing"], &["y"], vec![]);
    let inputs = || vec![fixed("x", FLOAT, &[2])];

    let by_name = import(&model(vec![named], inputs(), vec![], vec![])).unwrap_err();
    let by_position = import(&model(vec![unnamed], inputs(), vec![], vec![])).unwrap_err();

    assert_eq!(by_name.node(), Some("act1 (Relu)"));
    assert_eq!(by_position.node(), Some("Relu #0"));
}
