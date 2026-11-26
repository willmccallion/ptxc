use crate::{graph, run};
use ptxc_core::{Dim, Evaluate, HostTensor, Shape, TensorData};
use ptxc_graph::Graph;
use ptxc_graph_opt::{PassManager, PassReport, Simplify};
use ptxc_interp::Interpreter;

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

/// A 1 by 1 convolution of two channels into three in `conv_type`, `bias`
/// its optional bias operand, then a batch normalisation with parameters
/// of `parameter_type`, returning `returned`.
fn conv_then_batch_norm(
    bias: &str,
    returned: &str,
    [conv_type, parameter_type]: [&str; 2],
) -> String {
    let conv_operands =
        if bias.is_empty() { "%x, %0".to_owned() } else { format!("%x, %0, {bias}") };
    format!(
        "\
graph(%x: {conv_type}[1, 2, 2, 2]) {{
  %0 = const {conv_type}[3, 2, 1, 1] [1.0, -2.0, 0.5, 3.0, -1.5, 2.0]
  %1 = const {conv_type}[3] [0.25, -0.5, 1.0]
  %2 = const {parameter_type}[3] [2.0, 0.5, -1.0]
  %3 = const {parameter_type}[3] [0.5, -1.0, 0.25]
  %4 = const {parameter_type}[3] [0.125, 0.25, -0.375]
  %5 = const {parameter_type}[3] [4.0, 0.25, 1.0]
  %6 = conv<groups=1, axes=[[1, 1, 0, 0], [1, 1, 0, 0]]> {conv_operands} : {conv_type}[1, 3, 2, 2]
  %7 = batch_norm<epsilon=0.001> %6, %2, %3, %4, %5 : {conv_type}[1, 3, 2, 2]
  return {returned}
}}
"
    )
}

fn input() -> HostTensor {
    let shape = Shape::new([1, 2, 2, 2].map(Dim::new).to_vec()).unwrap();
    let values = vec![1.0_f32, -0.5, 2.0, 0.25, -1.0, 3.0, 0.5, -2.0];
    HostTensor::new(shape, values.into()).unwrap()
}

fn floats(graph: &Graph) -> Vec<f32> {
    let outputs = Interpreter::new(graph).evaluate(&[input()]).unwrap();
    let TensorData::F32(values) = outputs[0].data() else { panic!("{outputs:?}") };
    values.clone()
}

/// The ops of `graph`'s nodes, by mnemonic, in order.
fn ops(graph: &Graph) -> Vec<&'static str> {
    graph.nodes().map(|(_, node)| node.op().mnemonic()).collect()
}

#[test]
fn a_batch_normalisation_folds_into_the_convolution_before_it_with_or_without_a_bias() {
    for bias in ["", "%1"] {
        let text = conv_then_batch_norm(bias, "%7", ["f32"; 2]);
        let expected = floats(&graph(&text));
        let mut folded = graph(&text);

        let _reports = PassManager::standard().run(&mut folded).unwrap();

        assert_eq!(ops(&folded), ["conv"], "{bias}");
        let actual = floats(&folded);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                "{actual} {expected}"
            );
        }
    }
}

#[test]
fn a_batch_normalisation_folds_in_every_float_type() {
    for float in ["f16", "bf16"] {
        let mut folded = graph(&conv_then_batch_norm("", "%7", [float; 2]));

        let _reports = PassManager::standard().run(&mut folded).unwrap();

        assert_eq!(ops(&folded), ["conv"], "{float}");
    }
}

#[test]
fn a_batch_normalisation_stays_if_its_convolution_is_read_elsewhere() {
    let text = conv_then_batch_norm("", "%7, %6", ["f32"; 2]);

    let (after, report) = run(&Simplify, &text);

    assert_eq!(after, graph_text(&text));
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_batch_normalisation_stays_if_its_parameters_are_of_another_type() {
    let text = conv_then_batch_norm("", "%7", ["f32", "f16"]);

    let (after, report) = run(&Simplify, &text);

    assert_eq!(after, graph_text(&text));
    assert_eq!(report, UNCHANGED);
}

/// `text` as the printer writes it.
fn graph_text(text: &str) -> String {
    ptxc_graph::text::print(&graph(text))
}

#[test]
fn multiplying_by_one_and_adding_zero_on_either_side_are_removed() {
    let text = "\
graph(%x: f32[2, 3]) {
  %0 = const f32[3] [1.0, 1.0, 1.0]
  %1 = const f32[] [-0.0]
  %2 = mul %x, %0 : f32[2, 3]
  %3 = mul %0, %2 : f32[2, 3]
  %4 = add %3, %1 : f32[2, 3]
  %5 = add %1, %4 : f32[2, 3]
  %6 = neg %5 : f32[2, 3]
  return %6
}
";

    let (after, report) = run(&Simplify, text);

    let expected = "\
graph(%x: f32[2, 3]) {
  %0 = neg %x : f32[2, 3]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn negative_zero_and_one_are_recognised_in_every_float_type() {
    for float in ["f16", "bf16"] {
        let text = format!(
            "\
graph(%x: {float}[2]) {{
  %0 = const {float}[] [1.0]
  %1 = const {float}[] [-0.0]
  %2 = mul %x, %0 : {float}[2]
  %3 = add %2, %1 : {float}[2]
  %4 = neg %3 : {float}[2]
  return %4
}}
"
        );

        let (after, report) = run(&Simplify, &text);

        assert_eq!(
            after,
            format!("graph(%x: {float}[2]) {{\n  %0 = neg %x : {float}[2]\n  return %0\n}}\n")
        );
        assert_eq!(report, CHANGED);
    }
}

#[test]
fn a_constant_that_is_not_all_one_or_zero_or_that_broadcasts_its_operand_stays() {
    let text = "\
graph(%x: f32[3], %y: i32[2]) {
  %0 = const f32[3] [1.0, 2.0, 1.0]
  %1 = const f32[2, 3] [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  %2 = const i32[] [1]
  %3 = mul %x, %0 : f32[3]
  %4 = add %3, %1 : f32[2, 3]
  %5 = mul %y, %2 : i32[2]
  %6 = add %5, %2 : i32[2]
  return %4, %6
}
";

    let (after, report) = run(&Simplify, text);

    let expected = "\
graph(%x: f32[3], %y: i32[2]) {
  %0 = const f32[3] [1.0, 2.0, 1.0]
  %1 = const f32[2, 3] [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  %2 = const i32[] [1]
  %3 = mul %x, %0 : f32[3]
  %4 = add %3, %1 : f32[2, 3]
  %5 = add %y, %2 : i32[2]
  return %4, %5
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_transpose_of_a_transpose_is_one_transpose_of_both() {
    let text = "\
graph(%x: f32[2, 3, 4]) {
  %0 = transpose<permutation=[1, 2, 0]> %x : f32[3, 4, 2]
  %1 = transpose<permutation=[0, 2, 1]> %0 : f32[3, 2, 4]
  return %1
}
";

    let (after, report) = run(&Simplify, text);

    let expected = "\
graph(%x: f32[2, 3, 4]) {
  %0 = transpose<permutation=[1, 0, 2]> %x : f32[3, 2, 4]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_reshape_of_a_reshape_is_one_reshape() {
    let text = "\
graph(%x: f32[2, 3, 4]) {
  %0 = reshape<shape=[6, 4]> %x : f32[6, 4]
  %1 = reshape<shape=[4, 6]> %0 : f32[4, 6]
  %2 = neg %0 : f32[6, 4]
  return %1, %2
}
";

    let (after, report) = run(&Simplify, text);

    let expected = "\
graph(%x: f32[2, 3, 4]) {
  %0 = reshape<shape=[6, 4]> %x : f32[6, 4]
  %1 = neg %0 : f32[6, 4]
  %2 = reshape<shape=[4, 6]> %x : f32[4, 6]
  return %2, %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}
