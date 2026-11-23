//! Softmax and batch, instance, layer and RMS normalisation.

use crate::builders::{
    FLOAT, assert_lines, fixed, float, import, int, line, line_at, model, node, op_line,
};
use ptxc_onnx::Unsupported;

#[test]
fn softmaxes_default_to_the_last_axis() {
    let softmax = |op: &str, attrs| {
        line(node(op, &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[2, 3])], vec![])
    };

    assert_lines(&[
        (softmax("Softmax", vec![]), "softmax<axis=1> %x : f32[2, 3]"),
        (softmax("Softmax", vec![int("axis", 0)]), "softmax<axis=0> %x : f32[2, 3]"),
        (softmax("LogSoftmax", vec![]), "log_softmax<axis=1> %x : f32[2, 3]"),
    ]);
}

#[test]
fn layer_and_rms_norms_read_axis_epsilon_stash_type_and_bias() {
    let inputs =
        || vec![fixed("x", FLOAT, &[2, 3]), fixed("s", FLOAT, &[3]), fixed("b", FLOAT, &[3])];
    let layer = line(
        node("LayerNormalization", &["x", "s", "b"], &["y"], vec![int("stash_type", 16)]),
        inputs(),
        vec![],
    );
    let attrs = vec![int("axis", 1), float("epsilon", 0.5)];
    let rms = line_at(23, node("RMSNormalization", &["x", "s"], &["y"], attrs), inputs(), vec![]);

    assert_lines(&[
        (
            layer,
            "layer_norm<axis=1, epsilon=1e-5, stash=bf16, outputs=normalized> %x, %s, %b : f32[2, 3]",
        ),
        (rms, "rms_norm<axis=1, epsilon=0.5, stash=f32> %x, %s : f32[2, 3]"),
    ]);
}

#[test]
fn batch_and_instance_norms_take_their_parameters_as_inputs() {
    let channels = |name: &str| fixed(name, FLOAT, &[2]);
    let mut batch_inputs = vec![fixed("x", FLOAT, &[1, 2, 3])];
    batch_inputs.extend(["s", "b", "m", "v"].map(channels));
    let batch = line(
        node(
            "BatchNormalization",
            &["x", "s", "b", "m", "v"],
            &["y"],
            vec![float("momentum", 0.5)],
        ),
        batch_inputs,
        vec![],
    );
    let instance_inputs = vec![fixed("x", FLOAT, &[1, 2, 3]), channels("s"), channels("b")];
    let instance = line(
        node("InstanceNormalization", &["x", "s", "b"], &["y"], vec![]),
        instance_inputs,
        vec![],
    );

    assert_lines(&[
        (batch, "batch_norm<epsilon=1e-5> %x, %s, %b, %m, %v : f32[1, 2, 3]"),
        (instance, "instance_norm<epsilon=1e-5> %x, %s, %b : f32[1, 2, 3]"),
    ]);
}

#[test]
fn training_mode_batch_norm_is_unsupported() {
    let mut inputs = vec![fixed("x", FLOAT, &[1, 2, 3])];
    inputs.extend(["s", "b", "m", "v"].map(|name| fixed(name, FLOAT, &[2])));
    let nodes = vec![node(
        "BatchNormalization",
        &["x", "s", "b", "m", "v"],
        &["y"],
        vec![int("training_mode", 1)],
    )];

    let error = import(&model(nodes, inputs, vec![], vec![])).unwrap_err();

    assert_eq!(
        error.unsupported(),
        Some(&Unsupported::Attribute {
            op: "BatchNormalization".to_owned(),
            attribute: "training_mode".to_owned()
        })
    );
}

#[test]
fn layer_norm_produces_statistics_only_when_asked() {
    let inputs = || vec![fixed("x", FLOAT, &[2, 4]), fixed("s", FLOAT, &[4])];
    let plain = op_line(&model(
        vec![node("LayerNormalization", &["x", "s"], &["y"], vec![])],
        inputs(),
        vec![],
        vec![],
    ));
    let stats = op_line(&model(
        vec![node("LayerNormalization", &["x", "s"], &["y", "", "inv"], vec![])],
        inputs(),
        vec![],
        vec![],
    ));

    assert!(plain.contains("outputs=normalized"), "{plain}");
    assert!(stats.contains("outputs=with_statistics"), "{stats}");
}

#[test]
fn default_axes_are_the_last_of_any_rank() {
    let x = || vec![fixed("x", FLOAT, &[2, 3, 4])];
    let softmax = line(node("Softmax", &["x"], &["y"], vec![]), x(), vec![]);
    let mut inputs = x();
    inputs.push(fixed("s", FLOAT, &[4]));
    let layer = line(node("LayerNormalization", &["x", "s"], &["y"], vec![]), inputs, vec![]);

    assert_lines(&[
        (softmax, "softmax<axis=2> %x : f32[2, 3, 4]"),
        (
            layer,
            "layer_norm<axis=2, epsilon=1e-5, stash=f32, outputs=normalized> %x, %s : f32[2, 3, 4]",
        ),
    ]);
}

#[test]
fn a_stash_type_other_than_f32_or_bf16_is_unsupported() {
    let inputs = vec![fixed("x", FLOAT, &[2, 3]), fixed("s", FLOAT, &[3])];
    let nodes = vec![node("LayerNormalization", &["x", "s"], &["y"], vec![int("stash_type", 10)])];

    let error = import(&model(nodes, inputs, vec![], vec![])).unwrap_err();

    assert_eq!(
        error.unsupported(),
        Some(&Unsupported::Attribute {
            op: "LayerNormalization".to_owned(),
            attribute: "stash_type".to_owned()
        })
    );
}
