//! Max, average and global pooling.

use crate::builders::{FLOAT, assert_lines, fixed, import, int, ints, line, model, node};
use ptxc_onnx::Unsupported;
use ptxc_onnx_proto::AttributeProto;

fn pool(op: &str, mut attrs: Vec<AttributeProto>) -> String {
    attrs.push(ints("kernel_shape", &[2]));
    line(node(op, &["x"], &["y"], attrs), vec![fixed("x", FLOAT, &[1, 1, 5])], vec![])
}

#[test]
fn windowed_pools_read_counting_and_rounding() {
    assert_lines(&[
        (
            pool("MaxPool", vec![ints("strides", &[2])]),
            "max_pool<rounding=floor, axes=[[2, 2, 1, 0, 0]]> %x : f32[1, 1, 2]",
        ),
        (
            pool("MaxPool", vec![ints("strides", &[2]), int("ceil_mode", 1)]),
            "max_pool<rounding=ceil, axes=[[2, 2, 1, 0, 0]]> %x : f32[1, 1, 3]",
        ),
        (
            pool("AveragePool", vec![]),
            "average_pool<pads=exclude, rounding=floor, axes=[[2, 1, 1, 0, 0]]> %x : f32[1, 1, 4]",
        ),
        (
            pool("AveragePool", vec![int("count_include_pad", 1), ints("pads", &[1, 0])]),
            "average_pool<pads=include, rounding=floor, axes=[[2, 1, 1, 1, 0]]> %x : f32[1, 1, 5]",
        ),
    ]);
}

#[test]
fn global_pools_take_no_attributes() {
    let global = |op: &str| {
        line(node(op, &["x"], &["y"], vec![]), vec![fixed("x", FLOAT, &[1, 2, 3])], vec![])
    };

    assert_lines(&[
        (global("GlobalMaxPool"), "global_max_pool %x : f32[1, 2, 1]"),
        (global("GlobalAveragePool"), "global_average_pool %x : f32[1, 2, 1]"),
    ]);
}

#[test]
fn a_max_pool_storage_order_is_unsupported() {
    let attrs = vec![ints("kernel_shape", &[2]), int("storage_order", 1)];
    let model = model(
        vec![node("MaxPool", &["x"], &["y"], attrs)],
        vec![fixed("x", FLOAT, &[1, 1, 4])],
        vec![],
        vec![],
    );

    let error = import(&model).unwrap_err();

    assert_eq!(
        error.unsupported(),
        Some(&Unsupported::Attribute {
            op: "MaxPool".to_owned(),
            attribute: "storage_order".to_owned()
        })
    );
}

#[test]
fn a_max_pool_with_the_default_storage_order_imports() {
    assert_lines(&[(
        pool("MaxPool", vec![int("storage_order", 0)]),
        "max_pool<rounding=floor, axes=[[2, 1, 1, 0, 0]]> %x : f32[1, 1, 4]",
    )]);
}
