use crate::common::{f32_type, tensor_type};
use ptxc_core::float::Overflow;
use ptxc_core::{
    Affine, Axis, DType, Dim, Granularity, QuantDType, QuantParams, Rank, Scale, ZeroPoint,
};
use ptxc_graph::{
    ConvAttrs, DequantizeAttrs, Dilation, Groups, Padding, QConvAttrs, QMatMulAttrs, QuantizeAttrs,
    Stride, TensorType, TypeError, TypeRule, WindowAxis,
};

const MAP: Affine = Affine { scale: Scale::new(0.5), zero_point: ZeroPoint::new(0) };

fn axis(index: i64) -> Axis {
    Axis::from_signed(index, Rank::new(8)).unwrap()
}

fn per_tensor(dtype: QuantDType) -> QuantParams {
    QuantParams::new(dtype, Granularity::PerTensor(MAP)).unwrap()
}

fn per_axis(dtype: QuantDType, index: i64, count: usize) -> QuantParams {
    QuantParams::new(dtype, Granularity::PerAxis { axis: axis(index), maps: vec![MAP; count] })
        .unwrap()
}

const fn quantize(params: QuantParams) -> QuantizeAttrs {
    QuantizeAttrs::new(params, Overflow::Saturate)
}

#[test]
fn quantize_gives_the_grid_type_with_the_input_shape() {
    for (dtype, element) in [
        (QuantDType::I8, DType::I8),
        (QuantDType::U8, DType::U8),
        (QuantDType::F8E4M3Fn, DType::F8E4M3Fn),
    ] {
        let inferred = quantize(per_tensor(dtype)).infer(&[&f32_type(&[2, 3])]);
        assert_eq!(inferred, Ok(vec![tensor_type(element, &[2, 3])]), "{dtype:?}");
    }
}

#[test]
fn quantize_takes_floats_and_never_targets_i32() {
    let attrs = quantize(per_tensor(QuantDType::I8));

    assert_eq!(
        attrs.infer(&[&tensor_type(DType::I32, &[2])]),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
    assert_eq!(
        quantize(per_tensor(QuantDType::I32)).infer(&[&f32_type(&[2])]),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
}

#[test]
fn per_axis_parameters_need_one_map_per_element_of_their_axis() {
    let input = f32_type(&[2, 3]);

    assert!(quantize(per_axis(QuantDType::I8, 1, 3)).infer(&[&input]).is_ok());
    assert_eq!(
        quantize(per_axis(QuantDType::I8, 1, 2)).infer(&[&input]),
        Err(TypeError::QuantMapCount { axis: axis(1), dim: Dim::new(3), maps: 2 })
    );
    assert!(matches!(
        quantize(per_axis(QuantDType::I8, 2, 3)).infer(&[&input]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}

#[test]
fn quantize_keeps_overflow_only_for_fp8_grids() {
    let integer = QuantizeAttrs::new(per_tensor(QuantDType::U8), Overflow::NonFinite);
    let fp8 = QuantizeAttrs::new(per_tensor(QuantDType::F8E5M2), Overflow::NonFinite);

    assert_eq!(integer.overflow(), Overflow::Saturate);
    assert_eq!(fp8.overflow(), Overflow::NonFinite);
    assert_eq!(fp8.params(), &per_tensor(QuantDType::F8E5M2));
}

#[test]
fn dequantize_gives_the_requested_float_type() {
    let attrs = DequantizeAttrs { params: per_tensor(QuantDType::I8), output: DType::F16 };

    assert_eq!(
        attrs.infer(&[&tensor_type(DType::I8, &[4])]),
        Ok(vec![tensor_type(DType::F16, &[4])])
    );
}

#[test]
fn dequantize_accepts_i32_biases() {
    let attrs = DequantizeAttrs { params: per_tensor(QuantDType::I32), output: DType::F32 };

    assert_eq!(attrs.infer(&[&tensor_type(DType::I32, &[8])]), Ok(vec![f32_type(&[8])]));
}

#[test]
fn dequantize_input_must_be_on_the_grid_and_output_a_float() {
    let attrs = DequantizeAttrs { params: per_tensor(QuantDType::I8), output: DType::F32 };

    assert_eq!(
        attrs.infer(&[&tensor_type(DType::U8, &[4])]),
        Err(TypeError::MixedDTypes { lhs: DType::I8, rhs: DType::U8 })
    );
    let to_integer = DequantizeAttrs { output: DType::I32, ..attrs.clone() };
    assert_eq!(
        to_integer.infer(&[&tensor_type(DType::I8, &[4])]),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
    let per_axis_attrs = DequantizeAttrs { params: per_axis(QuantDType::I8, 0, 3), ..attrs };
    assert!(matches!(
        per_axis_attrs.infer(&[&tensor_type(DType::I8, &[4])]),
        Err(TypeError::QuantMapCount { .. })
    ));
}

fn qmatmul(b: QuantParams, output: QuantParams) -> QMatMulAttrs {
    QMatMulAttrs { a: per_tensor(QuantDType::U8), b, output }
}

#[test]
fn qmatmul_follows_matmul_shapes_on_the_output_grid() {
    let attrs = qmatmul(per_axis(QuantDType::I8, 1, 4), per_tensor(QuantDType::U8));
    let (a, b) = (tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4]));

    assert_eq!(attrs.infer(&[&a, &b]), Ok(vec![tensor_type(DType::U8, &[2, 4])]));
}

#[test]
fn qmatmul_operands_and_result_must_match_their_grids() {
    let (a, b) = (tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4]));

    let wrong_b = qmatmul(per_tensor(QuantDType::U8), per_tensor(QuantDType::U8));
    assert!(matches!(wrong_b.infer(&[&a, &b]), Err(TypeError::MixedDTypes { .. })));
    let i32_output = qmatmul(per_tensor(QuantDType::I8), per_tensor(QuantDType::I32));
    assert_eq!(i32_output.infer(&[&a, &b]), Err(TypeError::UnsupportedDType(DType::I32)));
    let wide_b = qmatmul(per_axis(QuantDType::I8, 1, 5), per_tensor(QuantDType::U8));
    assert!(matches!(wide_b.infer(&[&a, &b]), Err(TypeError::QuantMapCount { .. })));
    let short = tensor_type(DType::I8, &[5, 4]);
    assert!(matches!(
        qmatmul(per_tensor(QuantDType::I8), per_tensor(QuantDType::U8)).infer(&[&a, &short]),
        Err(TypeError::ContractedDims { .. })
    ));
}

fn qconv(output: QuantParams) -> QConvAttrs {
    let window = WindowAxis {
        stride: Stride::new(1).unwrap(),
        dilation: Dilation::new(1).unwrap(),
        pad_begin: Padding::new(0),
        pad_end: Padding::new(0),
    };
    QConvAttrs {
        conv: ConvAttrs { axes: vec![window; 2], groups: Groups::new(1).unwrap() },
        input: per_tensor(QuantDType::U8),
        weights: per_axis(QuantDType::I8, 0, 8),
        output,
    }
}

fn qconv_operands() -> (TensorType, TensorType) {
    (tensor_type(DType::U8, &[1, 3, 5, 5]), tensor_type(DType::I8, &[8, 3, 3, 3]))
}

#[test]
fn qconv_follows_conv_shapes_on_the_output_grid() {
    let (x, w) = qconv_operands();

    let inferred =
        qconv(per_tensor(QuantDType::U8)).infer(&[&x, &w, &tensor_type(DType::I32, &[8])]);

    assert_eq!(inferred, Ok(vec![tensor_type(DType::U8, &[1, 8, 3, 3])]));
}

#[test]
fn qconv_bias_is_i32_with_one_entry_per_filter() {
    let (x, w) = qconv_operands();
    let attrs = qconv(per_tensor(QuantDType::U8));

    assert!(matches!(attrs.infer(&[&x, &w, &f32_type(&[8])]), Err(TypeError::MixedDTypes { .. })));
    assert_eq!(
        attrs.infer(&[&x, &w, &tensor_type(DType::I32, &[7])]),
        Err(TypeError::ParameterLength { expected: Dim::new(8), actual: Dim::new(7) })
    );
}

#[test]
fn qconv_operands_and_result_must_match_their_grids() {
    let (x, w) = qconv_operands();

    assert!(matches!(
        qconv(per_tensor(QuantDType::U8)).infer(&[&w, &w]),
        Err(TypeError::MixedDTypes { .. })
    ));
    assert_eq!(
        qconv(per_axis(QuantDType::U8, 1, 8)).infer(&[&x, &w]),
        Err(TypeError::QuantAxis(axis(1)))
    );
    assert_eq!(
        qconv(per_tensor(QuantDType::I32)).infer(&[&x, &w]),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
}

#[test]
fn qmatmul_maps_vary_only_by_row_of_a_and_column_of_b() {
    let (a, b) = (tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4]));
    let attrs = |a_params, b_params| QMatMulAttrs {
        a: a_params,
        b: b_params,
        output: per_tensor(QuantDType::U8),
    };

    let by_row = attrs(per_axis(QuantDType::U8, 0, 2), per_axis(QuantDType::I8, 1, 4));
    let a_by_inner = attrs(per_axis(QuantDType::U8, 1, 3), per_tensor(QuantDType::I8));
    let b_by_inner = attrs(per_tensor(QuantDType::U8), per_axis(QuantDType::I8, 0, 3));

    assert_eq!(by_row.infer(&[&a, &b]), Ok(vec![tensor_type(DType::U8, &[2, 4])]));
    assert_eq!(a_by_inner.infer(&[&a, &b]), Err(TypeError::QuantAxis(axis(1))));
    assert_eq!(b_by_inner.infer(&[&a, &b]), Err(TypeError::QuantAxis(axis(0))));
}

#[test]
fn qconv_maps_vary_only_by_output_channel_of_the_weights() {
    let (x, w) = qconv_operands();
    let per_input_channel =
        QConvAttrs { weights: per_axis(QuantDType::I8, 1, 3), ..qconv(per_tensor(QuantDType::U8)) };
    let per_channel_input =
        QConvAttrs { input: per_axis(QuantDType::U8, 1, 3), ..qconv(per_tensor(QuantDType::U8)) };

    assert_eq!(per_input_channel.infer(&[&x, &w]), Err(TypeError::QuantAxis(axis(1))));
    assert_eq!(per_channel_input.infer(&[&x, &w]), Err(TypeError::QuantAxis(axis(1))));
}
