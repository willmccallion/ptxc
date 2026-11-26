use crate::run;
use ptxc_core::{Axis, DType, Dim, HostTensor, Shape};
use ptxc_graph::text::print;
use ptxc_graph::{BinaryOp, Graph, GraphBuilder, Layout, Op, TensorType, TransposeAttrs};
use ptxc_graph_opt::{ConstFold, Pass, PassReport};

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

#[test]
fn a_chain_of_constants_folds_to_one() {
    let text = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = const f32[2] [3.0, 4.0]
  %2 = neg %0 : f32[2]
  %3 = add %2, %1 : f32[2]
  %4 = mul %x, %3 : f32[2]
  return %4
}
";

    let (after, report) = run(&ConstFold, text);

    let expected = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [2.0, 2.0]
  %1 = mul %x, %0 : f32[2]
  return %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_weight_moved_into_another_layout_folds() {
    let text = "\
graph(%x: f32[3, 2]) {
  %0 = const f32[2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  %1 = transpose<permutation=[1, 0]> %0 : f32[3, 2]
  %2 = add %x, %1 : f32[3, 2]
  return %2
}
";

    let (after, report) = run(&ConstFold, text);

    let expected = "\
graph(%x: f32[3, 2]) {
  %0 = const f32[3, 2] [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
  %1 = add %x, %0 : f32[3, 2]
  return %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn every_output_of_a_node_folds() {
    let text = "\
graph() {
  %0 = const f32[3] [1.0, 2.0, 3.0]
  %1, %2 = split<axis=0, sizes=[1, 2]> %0 : f32[1], f32[2]
  return %1, %2
}
";

    let (after, report) = run(&ConstFold, text);

    let expected = "\
graph() {
  %0 = const f32[1] [1.0]
  %1 = const f32[2] [2.0, 3.0]
  return %0, %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_folded_output_keeps_its_name() {
    let text = "\
graph() {
  %0 = const f32[2] [1.0, -2.0]
  %y = neg %0 : f32[2]
  return %y
}
";

    let (after, _) = run(&ConstFold, text);

    let expected = "\
graph() {
  %0 = const f32[2] [-1.0, 2.0]
  return %0 as %y
}
";
    assert_eq!(after, expected);
}

#[test]
fn a_node_reading_anything_but_constants_is_left() {
    let text = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = add %x, %0 : f32[2]
  return %1
}
";

    let (after, report) = run(&ConstFold, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_fold_that_would_inflate_the_weights_is_left() {
    let text = "\
graph(%x: f32[2048]) {
  %0 = const f32[1] [0.5]
  %1 = expand<shape=[2048]> %0 : f32[2048]
  %2 = add %x, %1 : f32[2048]
  return %2
}
";

    let (after, report) = run(&ConstFold, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_small_result_folds_even_when_it_grows() {
    let text = "\
graph(%x: f32[1024]) {
  %0 = const f32[1] [0.5]
  %1 = expand<shape=[1024]> %0 : f32[1024]
  %2 = add %x, %1 : f32[1024]
  return %2
}
";

    let (after, report) = run(&ConstFold, text);

    assert!(after.contains("%0 = const f32[1024] "), "{after}");
    assert!(!after.contains("expand"), "{after}");
    assert_eq!(report, CHANGED);
}

/// `x + transpose(w)`, and `y + w` if `reread`, with `w` a weight of 64 by
/// 32 f32s, 8 KiB.
fn transposed_weight(reread: bool) -> Graph {
    let ty = |dims: [u64; 2]| {
        TensorType::new(
            DType::F32,
            Shape::new(dims.map(Dim::new).to_vec()).unwrap(),
            Layout::RowMajor,
        )
    };
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(ty([32, 64]));
    let y = builder.add_input(ty([64, 32]));
    let shape = Shape::new(vec![Dim::new(64), Dim::new(32)]).unwrap();
    let values: Vec<f32> = (0..2048_u16).map(f32::from).collect();
    let w = builder.add_constant(HostTensor::new(shape, values.into()).unwrap());
    let permutation = vec![Axis::new(1), Axis::new(0)];
    let t = builder.add_node(Op::Transpose(TransposeAttrs { permutation }), vec![w]).unwrap()[0];
    let mut outputs = builder.add_node(Op::Binary(BinaryOp::Add), vec![x, t]).unwrap();
    if reread {
        outputs.extend(builder.add_node(Op::Binary(BinaryOp::Add), vec![y, w]).unwrap());
    }
    builder.finish(outputs).unwrap()
}

#[test]
fn a_weight_only_one_node_reads_folds_however_large() {
    let mut graph = transposed_weight(false);

    let report = ConstFold.run(&mut graph).unwrap();

    assert_eq!(report, CHANGED);
    assert!(!print(&graph).contains("transpose"), "{}", print(&graph));
}

#[test]
fn a_fold_that_would_copy_a_weight_another_node_reads_is_left() {
    let mut graph = transposed_weight(true);

    let report = ConstFold.run(&mut graph).unwrap();

    assert_eq!(report, UNCHANGED);
    assert!(print(&graph).contains("transpose"), "{}", print(&graph));
}
