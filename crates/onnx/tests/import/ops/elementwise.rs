//! Unary, binary and variadic ops, `IsInf`, `Mod`, `Where` and casts.

use crate::builders::{BOOL, FLOAT, INT64, assert_lines, fixed, int, line, node, string};
use ptxc_onnx_proto::tensor_proto::DataType;

fn unary(op: &str, dtype: DataType) -> String {
    line(node(op, &["x"], &["y"], vec![]), vec![fixed("x", dtype, &[2])], vec![])
}

fn binary(op: &str, dtype: DataType) -> String {
    let inputs = vec![fixed("a", dtype, &[2]), fixed("b", dtype, &[2])];
    line(node(op, &["a", "b"], &["y"], vec![]), inputs, vec![])
}

#[test]
fn unary_ops_map_one_to_one() {
    assert_lines(&[
        (unary("Abs", FLOAT), "abs %x : f32[2]"),
        (unary("Neg", FLOAT), "neg %x : f32[2]"),
        (unary("Exp", FLOAT), "exp %x : f32[2]"),
        (unary("Log", FLOAT), "log %x : f32[2]"),
        (unary("Sqrt", FLOAT), "sqrt %x : f32[2]"),
        (unary("Reciprocal", FLOAT), "reciprocal %x : f32[2]"),
        (unary("Floor", FLOAT), "floor %x : f32[2]"),
        (unary("Ceil", FLOAT), "ceil %x : f32[2]"),
        (unary("Round", FLOAT), "round %x : f32[2]"),
        (unary("Sign", FLOAT), "sign %x : f32[2]"),
        (unary("Erf", FLOAT), "erf %x : f32[2]"),
        (unary("Sin", FLOAT), "sin %x : f32[2]"),
        (unary("Cos", FLOAT), "cos %x : f32[2]"),
        (unary("Not", BOOL), "not %x : bool[2]"),
        (unary("IsNaN", FLOAT), "is_nan %x : bool[2]"),
    ]);
}

#[test]
fn binary_ops_map_one_to_one() {
    assert_lines(&[
        (binary("Add", FLOAT), "add %a, %b : f32[2]"),
        (binary("Sub", FLOAT), "sub %a, %b : f32[2]"),
        (binary("Mul", FLOAT), "mul %a, %b : f32[2]"),
        (binary("Div", FLOAT), "div %a, %b : f32[2]"),
        (binary("Pow", FLOAT), "pow %a, %b : f32[2]"),
        (binary("Equal", FLOAT), "equal %a, %b : bool[2]"),
        (binary("Less", FLOAT), "less %a, %b : bool[2]"),
        (binary("LessOrEqual", FLOAT), "less_or_equal %a, %b : bool[2]"),
        (binary("Greater", FLOAT), "greater %a, %b : bool[2]"),
        (binary("GreaterOrEqual", FLOAT), "greater_or_equal %a, %b : bool[2]"),
        (binary("And", BOOL), "and %a, %b : bool[2]"),
        (binary("Or", BOOL), "or %a, %b : bool[2]"),
        (binary("Xor", BOOL), "xor %a, %b : bool[2]"),
    ]);
}

#[test]
fn variadic_ops_take_every_input() {
    let variadic = |op: &str| {
        let inputs =
            vec![fixed("a", FLOAT, &[2]), fixed("b", FLOAT, &[1]), fixed("c", FLOAT, &[2])];
        line(node(op, &["a", "b", "c"], &["y"], vec![]), inputs, vec![])
    };

    assert_lines(&[
        (variadic("Max"), "max %a, %b, %c : f32[2]"),
        (variadic("Min"), "min %a, %b, %c : f32[2]"),
        (variadic("Sum"), "sum %a, %b, %c : f32[2]"),
        (variadic("Mean"), "mean %a, %b, %c : f32[2]"),
    ]);
}

#[test]
fn is_inf_detects_the_signs_asked_for() {
    let is_inf =
        |attrs| line(node("IsInf", &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[2])], vec![]);

    assert_lines(&[
        (is_inf(vec![]), "is_inf<detect=both> %x : bool[2]"),
        (is_inf(vec![int("detect_negative", 0)]), "is_inf<detect=positive> %x : bool[2]"),
        (is_inf(vec![int("detect_positive", 0)]), "is_inf<detect=negative> %x : bool[2]"),
        (
            is_inf(vec![int("detect_positive", 0), int("detect_negative", 0)]),
            "is_inf<detect=neither> %x : bool[2]",
        ),
    ]);
}

#[test]
fn mod_kind_comes_from_fmod_and_where_takes_three_inputs() {
    let integer = line(
        node("Mod", &["a", "b"], &["y"], vec![]),
        vec![fixed("a", INT64, &[2]), fixed("b", INT64, &[2])],
        vec![],
    );
    let inputs = vec![fixed("c", BOOL, &[2]), fixed("a", FLOAT, &[2]), fixed("b", FLOAT, &[1])];
    let picked = line(node("Where", &["c", "a", "b"], &["y"], vec![]), inputs, vec![]);
    let floats = vec![fixed("a", FLOAT, &[2]), fixed("b", FLOAT, &[2])];
    let fmod = line(node("Mod", &["a", "b"], &["y"], vec![int("fmod", 1)]), floats, vec![]);

    assert_lines(&[
        (integer, "mod<kind=integer> %a, %b : i64[2]"),
        (fmod, "mod<kind=fmod> %a, %b : f32[2]"),
        (picked, "where %c, %a, %b : f32[2]"),
    ]);
}

#[test]
fn casts_take_their_target_from_to_or_the_second_input() {
    let cast =
        |attrs| line(node("Cast", &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[2])], vec![]);
    let like = line(
        node("CastLike", &["x", "t"], &["y"], vec![]),
        vec![fixed("x", FLOAT, &[2]), fixed("t", INT64, &[1])],
        vec![],
    );

    assert_lines(&[
        (cast(vec![int("to", 17)]), "cast<to=f8e4m3fn, overflow=saturate> %x : f8e4m3fn[2]"),
        (
            cast(vec![int("to", 17), int("saturate", 0)]),
            "cast<to=f8e4m3fn, overflow=non_finite> %x : f8e4m3fn[2]",
        ),
        (cast(vec![int("to", 7), string("round_mode", "up")]), "cast<to=i64> %x : i64[2]"),
        (like, "cast<to=i64> %x : i64[2]"),
    ]);
}
