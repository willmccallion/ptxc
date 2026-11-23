use crate::elementwise::lowered;
use crate::{stages, tensor_type};
use ptxc_core::float::Overflow;
use ptxc_core::{Affine, Axis, DType, Granularity, QuantDType, QuantParams, Scale, ZeroPoint};
use ptxc_graph::{
    ConvAttrs, DequantizeAttrs, Dilation, GraphBuilder, Groups, Op, Padding, QConvAttrs,
    QMatMulAttrs, QuantizeAttrs, Stride, WindowAxis,
};
use ptxc_loop::print_compute;
use ptxc_testkit::golden;

const fn affine(scale: f32, zero_point: i32) -> Affine {
    Affine { scale: Scale::new(scale), zero_point: ZeroPoint::new(zero_point) }
}

fn per_tensor(dtype: QuantDType, scale: f32, zero_point: i32) -> QuantParams {
    QuantParams::new(dtype, Granularity::PerTensor(affine(scale, zero_point))).unwrap()
}

#[test]
fn quantizing_divides_rounds_shifts_and_saturates() {
    let per_axis = QuantParams::new(
        QuantDType::I8,
        Granularity::PerAxis { axis: Axis::new(1), maps: vec![affine(0.5, -3), affine(2.0, 4)] },
    )
    .unwrap();
    let fp8 = per_tensor(QuantDType::F8E5M2, 0.25, 0);
    let definitions = [
        lowered(
            Op::Quantize(QuantizeAttrs::new(
                per_tensor(QuantDType::U8, 0.5, 7),
                Overflow::Saturate,
            )),
            &[(DType::F16, &[3])],
        ),
        lowered(
            Op::Quantize(QuantizeAttrs::new(per_axis, Overflow::Saturate)),
            &[(DType::F32, &[3, 2])],
        ),
        lowered(Op::Quantize(QuantizeAttrs::new(fp8, Overflow::NonFinite)), &[(DType::F32, &[3])]),
        lowered(
            Op::Dequantize(DequantizeAttrs {
                params: per_tensor(QuantDType::I32, 0.5, 0),
                output: DType::Bf16,
            }),
            &[(DType::I32, &[3])],
        ),
    ];

    golden::check("quant/quantize.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn per_axis_maps_become_tables() {
    let per_axis = QuantParams::new(
        QuantDType::U8,
        Granularity::PerAxis { axis: Axis::new(0), maps: vec![affine(0.5, 1), affine(0.25, 2)] },
    )
    .unwrap();
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::U8, &[2, 3]));
    let attrs = DequantizeAttrs { params: per_axis, output: DType::F32 };
    let out = builder.add_node(Op::Dequantize(attrs), vec![x]).unwrap();
    let graph = builder.finish(out).unwrap();
    let partitioned = ptxc_graph::Partitioned::one_node_per_group(graph).unwrap();

    let groups = ptxc_lower::lower(&partitioned).unwrap();

    let tables: Vec<String> = groups
        .iter()
        .flat_map(|group| group.tables().iter().map(|table| format!("{:?}", table.data())))
        .collect();
    assert_eq!(tables, ["F32([0.5, 0.25])", "I64([1, 2])"]);
}

#[test]
fn quantized_matmul_rescales_an_exact_integer_sum() {
    let attrs = QMatMulAttrs {
        a: per_tensor(QuantDType::U8, 0.5, 3),
        b: per_tensor(QuantDType::I8, 0.25, -1),
        output: per_tensor(QuantDType::I8, 2.0, 5),
    };
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(tensor_type(DType::U8, &[2, 3]));
    let rhs = builder.add_input(tensor_type(DType::I8, &[3, 4]));
    let out = builder.add_node(Op::QMatMul(attrs), vec![lhs, rhs]).unwrap();
    let graph = builder.finish(out).unwrap();

    let definitions: String =
        stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect();

    golden::check("quant/qmatmul.txt", &definitions).unwrap();
}

#[test]
fn quantized_convolution_adds_its_bias_before_rescaling_per_channel() {
    let weights = QuantParams::new(
        QuantDType::I8,
        Granularity::PerAxis { axis: Axis::new(0), maps: vec![affine(0.5, 0), affine(0.25, 0)] },
    )
    .unwrap();
    let window = WindowAxis {
        stride: Stride::new(1).unwrap(),
        dilation: Dilation::new(1).unwrap(),
        pad_begin: Padding::new(1),
        pad_end: Padding::new(0),
    };
    let attrs = QConvAttrs {
        conv: ConvAttrs { axes: vec![window], groups: Groups::new(1).unwrap() },
        input: per_tensor(QuantDType::U8, 0.5, 128),
        weights,
        output: per_tensor(QuantDType::U8, 1.0, 10),
    };
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::U8, &[1, 1, 4]));
    let w = builder.add_input(tensor_type(DType::I8, &[2, 1, 2]));
    let bias = builder.add_input(tensor_type(DType::I32, &[2]));
    let out = builder.add_node(Op::QConv(attrs.clone()), vec![x, w, bias]).unwrap();
    let graph = builder.finish(out).unwrap();

    let biased: String =
        stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect();
    let unbiased = lowered(Op::QConv(attrs), &[(DType::U8, &[1, 1, 4]), (DType::I8, &[2, 1, 2])]);

    golden::check("quant/qconv.txt", &[biased, unbiased].join("\n")).unwrap();
}
