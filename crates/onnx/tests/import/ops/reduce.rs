//! Reductions, arg-reductions and `CumSum`.

use crate::builders::{
    FLOAT, assert_lines, fixed, i64_tensor, import, int, ints, line_at, model, model_at, node,
    op_line, text,
};
use ptxc_onnx::ImportErrorKind;
use ptxc_onnx_proto::{AttributeProto, ModelProto};

#[test]
fn every_reduction_reads_its_axes_input() {
    let reduce = |op: &str| {
        let axes = vec![i64_tensor("a", &[1], &[-1])];
        line_at(
            18,
            node(op, &["x", "a"], &["y"], vec![int("keepdims", 0)]),
            vec![fixed("x", FLOAT, &[2, 3])],
            axes,
        )
    };

    assert_lines(&[
        (reduce("ReduceSum"), "reduce_sum<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceMean"), "reduce_mean<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceProd"), "reduce_prod<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceMax"), "reduce_max<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceMin"), "reduce_min<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceSumSquare"), "reduce_sum_square<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceL1"), "reduce_l1<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceL2"), "reduce_l2<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceLogSum"), "reduce_log_sum<axes=[1], keep_dims=drop> %x : f32[2]"),
        (reduce("ReduceLogSumExp"), "reduce_log_sum_exp<axes=[1], keep_dims=drop> %x : f32[2]"),
    ]);
}

#[test]
fn arg_reductions_read_axis_keepdims_and_tie_breaking() {
    let arg = |op: &str, attrs| {
        line_at(21, node(op, &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[2, 3])], vec![])
    };

    assert_lines(&[
        (arg("ArgMax", vec![]), "arg_max<axis=0, keep_dims=keep, tie=first> %x : i64[1, 3]"),
        (
            arg("ArgMin", vec![int("axis", -1), int("keepdims", 0), int("select_last_index", 1)]),
            "arg_min<axis=1, keep_dims=drop, tie=last> %x : i64[2]",
        ),
    ]);
}

#[test]
fn reduction_axes_come_from_the_attribute_before_opset_18_and_the_input_after() {
    let x = || vec![fixed("x", FLOAT, &[2, 3])];
    let by_attribute = op_line(&model_at(
        13,
        vec![node("ReduceMean", &["x"], &["y"], vec![ints("axes", &[-1]), int("keepdims", 0)])],
        x(),
        vec![],
        vec![],
    ));
    let by_input = op_line(&model_at(
        18,
        vec![node("ReduceMean", &["x", "a"], &["y"], vec![])],
        x(),
        vec![i64_tensor("a", &[1], &[0])],
        vec![],
    ));
    let every_axis =
        op_line(&model(vec![node("ReduceSum", &["x"], &["y"], vec![])], x(), vec![], vec![]));

    assert!(by_attribute.starts_with("reduce_mean<axes=[1], keep_dims=drop>"), "{by_attribute}");
    assert!(by_input.starts_with("reduce_mean<axes=[0], keep_dims=keep>"), "{by_input}");
    assert!(every_axis.starts_with("reduce_sum<axes=[0, 1], keep_dims=keep>"), "{every_axis}");
}

#[test]
fn a_noop_reduction_imports_as_nothing() {
    let graph = text(&model(
        vec![node("ReduceSum", &["x"], &["y"], vec![int("noop_with_empty_axes", 1)])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![fixed("y", FLOAT, &[2])],
    ));

    assert_eq!(graph, "graph(%x: f32[2]) {\n  return %x\n}\n");
}

fn cum_sum(attrs: Vec<AttributeProto>, axis: &[i64]) -> ModelProto {
    let axis_dims: &[i64] = if axis.len() == 1 { &[] } else { &[2] };
    model(
        vec![node("CumSum", &["x", "a"], &["y"], attrs)],
        vec![fixed("x", FLOAT, &[2, 3])],
        vec![i64_tensor("a", axis_dims, axis)],
        vec![],
    )
}

#[test]
fn cum_sum_reads_its_axis_input_and_direction_attributes() {
    let flags = vec![int("exclusive", 1), int("reverse", 1)];

    assert_eq!(
        op_line(&cum_sum(vec![], &[0])),
        "cum_sum<axis=0, inclusion=inclusive, direction=forward> %x : f32[2, 3]"
    );
    assert_eq!(
        op_line(&cum_sum(flags, &[-1])),
        "cum_sum<axis=1, inclusion=exclusive, direction=reverse> %x : f32[2, 3]"
    );
}

#[test]
fn a_cum_sum_axis_of_several_values_is_invalid() {
    let error = import(&cum_sum(vec![], &[0, 1])).unwrap_err();

    assert!(matches!(error.kind(), ImportErrorKind::Invalid { field: "axis", .. }), "{error}");
}
