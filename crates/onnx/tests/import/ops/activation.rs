//! Activations with their default and given parameters, `Clip` and `PRelu`.

use crate::builders::{FLOAT, INT32, assert_lines, fixed, float, line, model, node, string, text};
use ptxc_onnx_proto::AttributeProto;

fn activation(op: &str, attrs: Vec<AttributeProto>) -> String {
    line(node(op, &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[2])], vec![])
}

#[test]
fn parameterless_activations_map_one_to_one() {
    assert_lines(&[
        (activation("Relu", vec![]), "relu %x : f32[2]"),
        (activation("Sigmoid", vec![]), "sigmoid %x : f32[2]"),
        (activation("Tanh", vec![]), "tanh %x : f32[2]"),
        (activation("Softplus", vec![]), "softplus %x : f32[2]"),
        (activation("Softsign", vec![]), "softsign %x : f32[2]"),
        (activation("HardSwish", vec![]), "hard_swish %x : f32[2]"),
        (activation("Mish", vec![]), "mish %x : f32[2]"),
        (activation("Gelu", vec![]), "gelu<approximate=none> %x : f32[2]"),
        (
            activation("Gelu", vec![string("approximate", "tanh")]),
            "gelu<approximate=tanh> %x : f32[2]",
        ),
    ]);
}

#[test]
fn parameters_take_onnx_defaults_unless_given() {
    assert_lines(&[
        (activation("LeakyRelu", vec![]), "leaky_relu<alpha=0.01> %x : f32[2]"),
        (activation("Elu", vec![]), "elu<alpha=1.0> %x : f32[2]"),
        (activation("Celu", vec![]), "celu<alpha=1.0> %x : f32[2]"),
        (activation("ThresholdedRelu", vec![]), "thresholded_relu<alpha=1.0> %x : f32[2]"),
        (activation("Selu", vec![]), "selu<alpha=1.6732632, gamma=1.050701> %x : f32[2]"),
        (activation("HardSigmoid", vec![]), "hard_sigmoid<alpha=0.2, beta=0.5> %x : f32[2]"),
        (activation("Elu", vec![float("alpha", 0.5)]), "elu<alpha=0.5> %x : f32[2]"),
        (
            activation("Selu", vec![float("alpha", 2.0), float("gamma", 3.0)]),
            "selu<alpha=2.0, gamma=3.0> %x : f32[2]",
        ),
        (
            activation("HardSigmoid", vec![float("alpha", 0.25), float("beta", 0.75)]),
            "hard_sigmoid<alpha=0.25, beta=0.75> %x : f32[2]",
        ),
    ]);
}

#[test]
fn clip_keeps_given_bounds_and_prelu_takes_a_slope() {
    let inputs = vec![fixed("x", FLOAT, &[2]), fixed("lo", FLOAT, &[]), fixed("hi", FLOAT, &[])];
    let bounded = line(node("Clip", &["x", "lo", "hi"], &["y"], vec![]), inputs, vec![]);
    let slope = vec![fixed("x", FLOAT, &[2]), fixed("s", FLOAT, &[1])];
    let prelu = line(node("PRelu", &["x", "s"], &["y"], vec![]), slope, vec![]);

    assert_lines(&[(bounded, "clip %x, %lo, %hi : f32[2]"), (prelu, "prelu %x, %s : f32[2]")]);
}

#[test]
fn missing_integer_clip_bounds_are_the_type_extremes() {
    let graph = crate::builders::text(&crate::builders::model(
        vec![node("Clip", &["x"], &["y"], vec![])],
        vec![fixed("x", INT32, &[2])],
        vec![],
        vec![],
    ));

    assert!(
        graph.contains("const i32[] [-2147483648]") && graph.contains("const i32[] [2147483647]"),
        "{graph}"
    );
}

#[test]
fn clip_gets_both_bounds() {
    let clip = text(&model(
        vec![node("Clip", &["x"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2])],
        vec![],
        vec![],
    ));

    assert!(clip.contains("const f32[] [-inf]") && clip.contains("const f32[] [inf]"), "{clip}");
}
