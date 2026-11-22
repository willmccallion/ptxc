use super::assert_round_trips;
use crate::common::{add_one, binary, f32_type, shape, tensor_type, unary};
use ptxc_core::float::{F8E4M3Fn, F16};
use ptxc_core::{Axis, DType, Dim, HostTensor, TensorData};
use ptxc_graph::text::{parse, print};
use ptxc_graph::{BinaryOp, GraphBuilder, Op, SplitAttrs, UnaryOp, ValueDef, ValueName};

fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(shape(dims), data.into()).unwrap()
}

#[test]
fn a_graph_prints_inputs_constants_nodes_and_return() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[2, 3]));
    builder.set_name(x, ValueName::new("x".to_owned()));
    let c = builder.add_constant(tensor(&[3], vec![0.5_f32, 1.0, -2.0]));
    let sum = add_one(&mut builder, binary(BinaryOp::Add), &[x, c]);
    let out = add_one(&mut builder, unary(UnaryOp::Neg), &[sum]);
    let graph = builder.finish(vec![out]).unwrap();

    let expected = "\
graph(%x: f32[2, 3]) {
  %0 = const f32[3] [0.5, 1.0, -2.0]
  %1 = add %x, %0 : f32[2, 3]
  %2 = neg %1 : f32[2, 3]
  return %2
}
";
    assert_eq!(print(&graph), expected);
}

#[test]
fn attributes_and_several_results_print_in_order() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(f32_type(&[3, 4]));
    let split = Op::Split(SplitAttrs { axis: Axis::new(0), sizes: vec![Dim::new(1), Dim::new(2)] });
    let outputs = builder.add_node(split, vec![x]).unwrap();
    let graph = builder.finish(outputs).unwrap();

    let expected = "\
graph(%0: f32[3, 4]) {
  %1, %2 = split<axis=0, sizes=[1, 2]> %0 : f32[1, 4], f32[2, 4]
  return %1, %2
}
";
    assert_eq!(print(&graph), expected);
}

#[test]
fn names_that_are_not_bare_words_are_quoted_and_survive() {
    let mut builder = GraphBuilder::new();
    let names = ["onnx::Mul_1", "/layer1/conv/output_0", "7", "a \"quoted\" \\ name", ""];
    let values: Vec<_> = names
        .iter()
        .map(|name| {
            let value = builder.add_input(f32_type(&[1]));
            builder.set_name(value, ValueName::new((*name).to_owned()));
            value
        })
        .collect();
    let graph = builder.finish(values.clone()).unwrap();

    let text = print(&graph);
    let parsed = parse(&text, &[]).unwrap();

    assert!(text.contains(r#"%"onnx::Mul_1""#) && text.contains(r#"%"7""#), "{text}");
    for (&value, name) in parsed.inputs().iter().zip(names) {
        assert_eq!(parsed.value(value).name().map(ValueName::as_str), Some(name));
    }
    assert_eq!(values.len(), parsed.inputs().iter().count());
}

#[test]
fn a_repeated_name_falls_back_to_a_number() {
    let mut builder = GraphBuilder::new();
    let first = builder.add_input(f32_type(&[1]));
    let second = builder.add_input(f32_type(&[1]));
    for value in [first, second] {
        builder.set_name(value, ValueName::new("same".to_owned()));
    }
    let graph = builder.finish(vec![first, second]).unwrap();

    assert_eq!(print(&graph).lines().next(), Some("graph(%same: f32[1], %0: f32[1]) {"));
    assert_round_trips(&graph, &[]);
}

#[test]
fn small_constants_of_every_type_round_trip_exactly() {
    let f16_nan = F16::from_bits(0x7e01);
    let constants = [
        tensor(&[4], vec![f32::from_bits(0x7fc0_0001), -0.0_f32, 1e-30, f32::NEG_INFINITY]),
        tensor(&[3], vec![f16_nan, F16::from_f32(-0.5), F16::from_f32(65504.0)]),
        tensor(&[2], vec![F8E4M3Fn::from_bits(0x7f), F8E4M3Fn::from_bits(0x7e)]),
        tensor(&[2], vec![i64::MIN, i64::MAX]),
        tensor(&[2, 1], vec![-128_i8, 127]),
        tensor(&[2], vec![true, false]),
        tensor(&[0], Vec::<u8>::new()),
    ];
    let mut builder = GraphBuilder::new();
    let values: Vec<_> =
        constants.iter().cloned().map(|constant| builder.add_constant(constant)).collect();
    let graph = builder.finish(values).unwrap();

    let text = print(&graph);

    assert!(
        text.contains("nan_7e01") && text.contains("nan_7fc00001") && text.contains("-0.0"),
        "{text}"
    );
    assert_round_trips(&graph, &[]);
}

#[test]
fn large_constants_are_elided_and_parse_from_supplied_tensors() {
    let large = tensor(&[17], (0_i16..17).map(|value| f32::from(value) * 0.5).collect::<Vec<_>>());
    let mut builder = GraphBuilder::new();
    let value = builder.add_constant(large.clone());
    let graph = builder.finish(vec![value]).unwrap();

    let text = print(&graph);
    let supplied = [tensor(&[17], vec![1.0_f32; 17]), large];

    assert!(text.contains("<elided 0x7d234e6f6a5e085b>"), "{text}");
    assert_round_trips(&graph, &supplied);
    let parsed = parse(&text, &supplied).unwrap();
    let output = parsed.outputs().iter().copied().next().unwrap();
    let ValueDef::Constant(id) = parsed.value(output).def() else { panic!("not a constant") };
    let expected: Vec<f32> = (0_i16..17).map(|value| f32::from(value) * 0.5).collect();
    assert!(matches!(parsed.constant(id).data(), TensorData::F32(values) if *values == expected));
}

#[test]
fn content_hashes_match_an_independent_fnv_1a() {
    let pattern: Vec<bool> = (0..17).map(|index| index % 3 == 0).collect();
    let mut builder = GraphBuilder::new();
    let value = builder.add_constant(tensor(&[17], pattern));
    let graph = builder.finish(vec![value]).unwrap();

    assert!(print(&graph).contains("<elided 0xbd0adf16e3731002>"), "{}", print(&graph));
}

#[test]
fn sixteen_elements_print_inline_and_seventeen_do_not() {
    let mut builder = GraphBuilder::new();
    let inline = builder.add_constant(tensor(&[16], vec![0_i32; 16]));
    let elided = builder.add_constant(tensor(&[17], vec![0_i32; 17]));
    let graph = builder.finish(vec![inline, elided]).unwrap();

    let text = print(&graph);

    assert!(text.contains("const i32[16] [0, 0,"), "{text}");
    assert!(text.contains("const i32[17] <elided"), "{text}");
}

#[test]
fn comments_and_whitespace_are_ignored() {
    let text = "// a comment\ngraph( %x : f32[ 2 ] ) { // another\n  %y = neg %x : f32[2]\n\n  return %y }";

    let graph = parse(text, &[]).unwrap();

    assert_eq!(print(&graph), "graph(%x: f32[2]) {\n  %y = neg %x : f32[2]\n  return %y\n}\n");
}

#[test]
fn printed_dtypes_cover_every_element_type() {
    let dtypes = [
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::F8E4M3Fn,
        DType::F8E5M2,
        DType::I8,
        DType::U8,
        DType::I32,
        DType::I64,
        DType::Bool,
    ];
    let mut builder = GraphBuilder::new();
    let inputs: Vec<_> =
        dtypes.iter().map(|&dtype| builder.add_input(tensor_type(dtype, &[1]))).collect();
    let graph = builder.finish(inputs).unwrap();

    let header = print(&graph).lines().next().unwrap().to_owned();

    for word in ["f32", "f16", "bf16", "f8e4m3fn", "f8e5m2", "i8", "u8", "i32", "i64", "bool"] {
        assert!(header.contains(&format!(": {word}[1]")), "{word} in {header}");
    }
    assert_round_trips(&graph, &[]);
}
