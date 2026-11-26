//! The convolutions the loop and schedule tests share, against ONNX `Conv`
//! as the reference interpreter computes it, and their inputs.

use ptxc_core::{DType, Evaluate, HostTensor, TensorData};
use ptxc_graph::{
    ConvAttrs, Dilation, GraphBuilder, Groups, Layout, Op, Padding, Stride, TensorType, WindowAxis,
};
use ptxc_interp::Interpreter;
use ptxc_loop::ComputeDef;
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::convolution::{Channels, Conv, ConvTypes, conv, conv_inputs, typed_conv};

/// `tensor`, of `i32`, as `f32`: small integers, so exactly.
fn as_floats(tensor: &HostTensor) -> HostTensor {
    let TensorData::I32(values) = tensor.data() else { panic!("{tensor:?}") };
    let floats: Vec<f32> =
        values.iter().map(|&value| i16::try_from(value).unwrap().into()).collect();
    HostTensor::new(tensor.shape().clone(), floats.into()).unwrap()
}

fn floats(tensor: &HostTensor) -> &[f32] {
    let TensorData::F32(values) = tensor.data() else { panic!("{tensor:?}") };
    values
}

fn ints(tensor: &HostTensor) -> &[i32] {
    let TensorData::I32(values) = tensor.data() else { panic!("{tensor:?}") };
    values
}

/// `compute`'s outputs on `inputs` under its default schedule.
fn scheduled(compute: &std::sync::Arc<ComputeDef>, inputs: &[HostTensor]) -> Vec<HostTensor> {
    let target = preset(SmVersion::SM_89).unwrap();
    let kernel = ptxc_schedule::default_schedule(compute.clone(), &target).unwrap().lower();
    ptxc_loop::run(&kernel, inputs).unwrap()
}

/// ONNX `Conv` with `geometry`'s attributes on `inputs`, channels first.
fn onnx_conv(geometry: Conv, inputs: &[HostTensor]) -> Vec<HostTensor> {
    let axis = WindowAxis {
        stride: Stride::new(geometry.stride).unwrap(),
        dilation: Dilation::new(geometry.dilation).unwrap(),
        pad_begin: Padding::new(geometry.pad),
        pad_end: Padding::new(geometry.pad),
    };
    let attrs = ConvAttrs { axes: vec![axis; 2], groups: Groups::new(geometry.groups).unwrap() };
    let mut builder = GraphBuilder::new();
    let operands = inputs
        .iter()
        .map(|input| {
            builder.add_input(TensorType::new(DType::F32, input.shape().clone(), Layout::RowMajor))
        })
        .collect();
    let output = builder.add_node(Op::Conv(attrs), operands).unwrap();
    let graph = builder.finish(output).unwrap();
    Interpreter::new(&graph).evaluate(inputs).unwrap()
}

#[test]
fn a_shared_convolution_computes_what_onnx_conv_does() {
    let strided = Conv {
        batch: 1,
        size: 5,
        channels: 2,
        features: 4,
        kernel: 3,
        stride: 2,
        dilation: 1,
        pad: 1,
        groups: 2,
    };
    let dilated = Conv {
        batch: 2,
        size: 6,
        channels: 3,
        features: 2,
        kernel: 2,
        stride: 1,
        dilation: 2,
        pad: 0,
        groups: 1,
    };
    let types = ConvTypes { operands: DType::F32, sum: DType::F32 };
    for geometry in [strided, dilated] {
        let compute = typed_conv(geometry, Channels::First, types);
        let inputs: Vec<HostTensor> = conv_inputs(&compute).iter().map(as_floats).collect();

        let expected = onnx_conv(geometry, &inputs);
        let actual = scheduled(&compute, &inputs);

        assert_eq!(floats(&actual[0]), floats(&expected[0]), "{geometry:?}");
    }
}

#[test]
fn a_convolution_s_inputs_count_up_from_distinct_starts() {
    let geometry = Conv {
        batch: 1,
        size: 5,
        channels: 1,
        features: 1,
        kernel: 1,
        stride: 1,
        dilation: 1,
        pad: 0,
        groups: 1,
    };

    let inputs = conv_inputs(&conv(geometry, Channels::First));

    let starts: Vec<i32> = (1..=23).chain(1..=2).collect();
    assert_eq!(ints(&inputs[0]), starts);
    assert_eq!(ints(&inputs[1]), [-9]);
}
