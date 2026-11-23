//! Ops imported as values or aliases: `Constant`, `Shape`, `Size`,
//! `ConstantOfShape`, `Range`, `Identity` and `Dropout`.

use crate::builders::{
    BOOL, FLOAT, INT32, INT64, assert_lines, f32_tensor, fixed, float, i64_tensor, import, int,
    ints, model, node, output_line, tensor_attr,
};
use ptxc_onnx::{ImportErrorKind, Unsupported};
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::{AttributeProto, TensorProto};

fn constant(attrs: Vec<AttributeProto>, dtype: DataType, dims: &[i64]) -> String {
    output_line(vec![node("Constant", &[], &["c"], attrs)], vec![], fixed("c", dtype, dims))
}

#[test]
fn constants_come_from_whichever_value_attribute_is_set() {
    let floats = crate::builders::floats("value_floats", &[0.5, 1.5]);

    assert_lines(&[
        (
            constant(vec![tensor_attr("value", f32_tensor("", &[2], &[1.0, 2.0]))], FLOAT, &[2]),
            "const f32[2] [1.0, 2.0]",
        ),
        (constant(vec![float("value_float", 0.25)], FLOAT, &[]), "const f32[] [0.25]"),
        (constant(vec![floats], FLOAT, &[2]), "const f32[2] [0.5, 1.5]"),
        (constant(vec![int("value_int", 7)], INT64, &[]), "const i64[] [7]"),
        (constant(vec![ints("value_ints", &[1, 2])], INT64, &[2]), "const i64[2] [1, 2]"),
    ]);
}

#[test]
fn a_constant_needs_exactly_one_value() {
    let none = model(vec![node("Constant", &[], &["c"], vec![])], vec![], vec![], vec![]);
    let two = model(
        vec![node("Constant", &[], &["c"], vec![int("value_int", 1), float("value_float", 1.0)])],
        vec![],
        vec![],
        vec![],
    );

    assert_eq!(
        import(&none).unwrap_err().unsupported(),
        Some(&Unsupported::Attribute { op: "Constant".to_owned(), attribute: "value".to_owned() })
    );
    assert!(matches!(import(&two).unwrap_err().kind(), ImportErrorKind::Invalid { .. }));
}

#[test]
fn shape_and_size_read_the_input_type() {
    let x = || vec![fixed("x", FLOAT, &[2, 3, 4, 5])];
    let shape = |attrs, length| {
        output_line(vec![node("Shape", &["x"], &["s"], attrs)], x(), fixed("s", INT64, &[length]))
    };

    assert_lines(&[
        (shape(vec![], 4), "const i64[4] [2, 3, 4, 5]"),
        (shape(vec![int("start", -3), int("end", -1)], 2), "const i64[2] [3, 4]"),
        (shape(vec![int("start", 1), int("end", 100)], 3), "const i64[3] [3, 4, 5]"),
        (shape(vec![int("start", 3), int("end", 1)], 0), "const i64[0] []"),
        (
            output_line(vec![node("Size", &["x"], &["n"], vec![])], x(), fixed("n", INT64, &[])),
            "const i64[] [120]",
        ),
    ]);
}

#[test]
fn constant_of_shape_fills_with_zero_or_its_value() {
    let fill = |attrs, dtype| {
        let nodes = vec![
            node("Constant", &[], &["dims"], vec![ints("value_ints", &[2, 2])]),
            node("ConstantOfShape", &["dims"], &["y"], attrs),
        ];
        output_line(nodes, vec![], fixed("y", dtype, &[2, 2]))
    };

    assert_lines(&[
        (fill(vec![], FLOAT), "const f32[2, 2] [0.0, 0.0, 0.0, 0.0]"),
        (
            fill(vec![tensor_attr("value", i64_tensor("", &[1], &[7]))], INT64),
            "const i64[2, 2] [7, 7, 7, 7]",
        ),
    ]);
}

fn scalar(name: &str, dtype: DataType, value: f64) -> TensorProto {
    #[expect(clippy::cast_possible_truncation, reason = "test values are small and exact")]
    let tensor = match dtype {
        DataType::Float => f32_tensor(name, &[], &[value as f32]),
        DataType::Int32 => {
            TensorProto { int32_data: vec![value as i32], ..i64_tensor(name, &[], &[]) }
        }
        _ => i64_tensor(name, &[], &[value as i64]),
    };
    TensorProto { data_type: Some(dtype as i32), ..tensor }
}

fn range(dtype: DataType, start: f64, limit: f64, delta: f64, length: i64) -> String {
    let nodes = ["s", "l", "d"].iter().zip([start, limit, delta]).map(|(&name, value)| {
        node("Constant", &[], &[name], vec![tensor_attr("value", scalar("", dtype, value))])
    });
    let mut nodes: Vec<_> = nodes.collect();
    nodes.push(node("Range", &["s", "l", "d"], &["r"], vec![]));
    output_line(nodes, vec![], fixed("r", dtype, &[length]))
}

#[test]
fn ranges_count_ceil_of_span_over_delta() {
    assert_lines(&[
        (range(INT64, 1.0, 10.0, 3.0, 3), "const i64[3] [1, 4, 7]"),
        (range(INT64, 10.0, 1.0, -3.0, 3), "const i64[3] [10, 7, 4]"),
        (range(INT64, 5.0, 1.0, 1.0, 0), "const i64[0] []"),
        (range(INT64, 0.0, 0.0, 1.0, 0), "const i64[0] []"),
        (range(INT32, 0.0, 6.0, 2.0, 3), "const i32[3] [0, 2, 4]"),
        (range(FLOAT, 0.0, 1.0, 0.3, 4), "const f32[4] [0.0, 0.3, 0.6, 0.90000004]"),
        (range(FLOAT, 1.0, 0.0, 0.5, 0), "const f32[0] []"),
    ]);
}

#[test]
fn a_zero_delta_or_an_overlong_range_is_invalid() {
    let invalid = |dtype: DataType, limit: f64, delta: f64| {
        let nodes = vec![node("Range", &["s", "l", "d"], &["r"], vec![])];
        let constants =
            vec![scalar("s", dtype, 0.0), scalar("l", dtype, limit), scalar("d", dtype, delta)];
        import(&model(nodes, vec![], constants, vec![]))
            .is_err_and(|error| matches!(error.kind(), ImportErrorKind::Invalid { .. }))
    };

    assert!(invalid(INT64, 4.0, 0.0));
    assert!(invalid(FLOAT, 4.0, 0.0));
    assert!(invalid(INT64, 65_536.0, 1.0));
    assert!(!invalid(INT64, 65_535.0, 1.0));
    assert!(invalid(FLOAT, 70_000.0, 1.0));
}

#[test]
fn inference_dropout_is_its_input() {
    let dropout = |training: bool| {
        let flag = TensorProto {
            name: Some("t".to_owned()),
            data_type: Some(BOOL as i32),
            int32_data: vec![i32::from(training)],
            ..TensorProto::default()
        };
        let nodes = vec![node("Dropout", &["x", "", "t"], &["y"], vec![int("seed", 3)])];
        import(&model(
            nodes,
            vec![fixed("x", FLOAT, &[2])],
            vec![flag],
            vec![fixed("y", FLOAT, &[2])],
        ))
    };

    assert_eq!(
        ptxc_graph::text::print(&dropout(false).unwrap()),
        "graph(%x: f32[2]) {\n  return %x\n}\n"
    );
    assert_eq!(
        dropout(true).unwrap_err().unsupported(),
        Some(&Unsupported::Attribute {
            op: "Dropout".to_owned(),
            attribute: "training_mode".to_owned()
        })
    );
}
