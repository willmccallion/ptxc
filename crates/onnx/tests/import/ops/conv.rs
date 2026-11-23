//! `Conv`, and the window attributes it shares with pooling.

use crate::builders::{
    FLOAT, assert_lines, f32_tensor, fixed, import, int, ints, line, model, node, string,
};
use ptxc_onnx::ImportErrorKind;
use ptxc_onnx_proto::{AttributeProto, ModelProto};

/// A `Conv` of `x: f32[1, 2, width]` with weights `f32[2, 2 / group, 3]`.
fn conv_model(width: i64, attrs: Vec<AttributeProto>) -> ModelProto {
    let group = attrs
        .iter()
        .find(|attr| attr.name.as_deref() == Some("group"))
        .and_then(|attr| attr.i)
        .unwrap_or(1);
    let channels = 2 / group;
    let weights =
        f32_tensor("w", &[2, channels, 3], &vec![0.0; usize::try_from(6 * channels).unwrap()]);
    model(
        vec![node("Conv", &["x", "w"], &["y"], attrs)],
        vec![fixed("x", FLOAT, &[1, 2, width])],
        vec![weights],
        vec![],
    )
}

fn conv(width: i64, attrs: Vec<AttributeProto>) -> String {
    crate::builders::op_line(&conv_model(width, attrs))
}

#[test]
fn explicit_windows_read_strides_dilations_pads_and_groups() {
    let biased = line(
        node("Conv", &["x", "w", "b"], &["y"], vec![ints("kernel_shape", &[3])]),
        vec![fixed("x", FLOAT, &[1, 2, 5]), fixed("b", FLOAT, &[2])],
        vec![f32_tensor("w", &[2, 2, 3], &[0.0; 12])],
    );

    assert_lines(&[
        (conv(5, vec![]), "conv<groups=1, axes=[[1, 1, 0, 0]]> %x, %w : f32[1, 2, 3]"),
        (
            conv(9, vec![ints("strides", &[2]), ints("dilations", &[2]), ints("pads", &[1, 2])]),
            "conv<groups=1, axes=[[2, 2, 1, 2]]> %x, %w : f32[1, 2, 4]",
        ),
        (
            conv(5, vec![int("group", 2)]),
            "conv<groups=2, axes=[[1, 1, 0, 0]]> %x, %w : f32[1, 2, 3]",
        ),
        (
            conv(5, vec![string("auto_pad", "VALID")]),
            "conv<groups=1, axes=[[1, 1, 0, 0]]> %x, %w : f32[1, 2, 3]",
        ),
        (biased, "conv<groups=1, axes=[[1, 1, 0, 0]]> %x, %w, %b : f32[1, 2, 3]"),
    ]);
}

#[test]
fn same_padding_keeps_ceil_of_input_over_stride_outputs() {
    let same = |width, mode: &str, stride, dilation| {
        conv(
            width,
            vec![
                string("auto_pad", mode),
                ints("strides", &[stride]),
                ints("dilations", &[dilation]),
            ],
        )
    };

    assert_lines(&[
        (same(5, "SAME_UPPER", 2, 1), "conv<groups=1, axes=[[2, 1, 1, 1]]> %x, %w : f32[1, 2, 3]"),
        (same(4, "SAME_UPPER", 2, 1), "conv<groups=1, axes=[[2, 1, 0, 1]]> %x, %w : f32[1, 2, 2]"),
        (same(4, "SAME_LOWER", 2, 1), "conv<groups=1, axes=[[2, 1, 1, 0]]> %x, %w : f32[1, 2, 2]"),
        (same(6, "SAME_UPPER", 1, 2), "conv<groups=1, axes=[[1, 2, 2, 2]]> %x, %w : f32[1, 2, 6]"),
        (same(7, "SAME_UPPER", 3, 1), "conv<groups=1, axes=[[3, 1, 1, 1]]> %x, %w : f32[1, 2, 3]"),
    ]);
}

fn window_error(attrs: Vec<AttributeProto>) -> String {
    match import(&conv_model(5, attrs)).unwrap_err().kind() {
        ImportErrorKind::Invalid { problem, .. } => problem.clone(),
        other => panic!("not invalid: {other}"),
    }
}

#[test]
fn inconsistent_window_attributes_are_invalid() {
    assert_eq!(
        window_error(vec![ints("kernel_shape", &[2])]),
        "kernel_shape differs from the weights' spatial dims"
    );
    assert_eq!(
        window_error(vec![ints("pads", &[1])]),
        "pads must have two entries per spatial axis"
    );
    assert_eq!(
        window_error(vec![string("auto_pad", "VALID"), ints("pads", &[0, 0])]),
        "pads cannot be given with auto_pad"
    );
    assert_eq!(window_error(vec![ints("strides", &[0])]), "a stride is not positive");
    assert_eq!(window_error(vec![ints("dilations", &[-1])]), "a dilation is not positive");
    assert_eq!(
        window_error(vec![ints("strides", &[1, 1])]),
        "a window attribute needs one entry per spatial axis"
    );
}

#[test]
fn explicit_pads_list_every_begin_then_every_end() {
    let conv = line(
        node("Conv", &["x", "w"], &["y"], vec![ints("pads", &[1, 2, 3, 4])]),
        vec![fixed("x", FLOAT, &[1, 1, 5, 5])],
        vec![f32_tensor("w", &[1, 1, 3, 3], &[0.0; 9])],
    );

    assert_lines(&[(
        conv,
        "conv<groups=1, axes=[[1, 1, 1, 3], [1, 1, 2, 4]]> %x, %w : f32[1, 1, 7, 9]",
    )]);
}
