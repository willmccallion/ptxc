use crate::common::{f32_type, tensor_type};
use ptxc_core::float::Overflow;
use ptxc_core::{
    Affine, Axis, DType, Dim, Granularity, QuantDType, QuantParams, Rank, Scale, ZeroPoint,
};
use ptxc_graph::{
    ConvAttrs, DequantizeAttrs, Dilation, Groups, Padding, QConvAttrs, QMatMulAttrs, QMatMulOutput,
    QuantizeAttrs, Stride, TensorType, TypeError, TypeRule, WindowAxis,
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
    QMatMulAttrs { a: per_tensor(QuantDType::U8), b, output: QMatMulOutput::Quantized(output) }
}

#[test]
fn qmatmul_follows_matmul_shapes_on_the_output_grid() {
    let attrs = qmatmul(per_axis(QuantDType::I8, 1, 4), per_tensor(QuantDType::U8));
    let (a, b) = (tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4]));

    assert_eq!(attrs.infer(&[&a, &b]), Ok(vec![tensor_type(DType::U8, &[2, 4])]));
}

/// `a × b` on `[grid_a, grid_b]`, dequantized to `output`.
fn dequantized([a, b]: [QuantDType; 2], output: DType) -> QMatMulAttrs {
    QMatMulAttrs { a: per_tensor(a), b: per_tensor(b), output: QMatMulOutput::Float(output) }
}

#[test]
fn a_dequantized_qmatmul_gives_its_float_type_from_integer_or_fp8_grids() {
    let fp8 = [QuantDType::F8E4M3Fn, QuantDType::F8E5M2];
    let fp8_operands = (tensor_type(DType::F8E4M3Fn, &[2, 3]), tensor_type(DType::F8E5M2, &[3, 4]));
    let integer_operands = (tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4]));

    let from_fp8 = dequantized(fp8, DType::F32).infer(&[&fp8_operands.0, &fp8_operands.1]);
    let integers = dequantized([QuantDType::U8, QuantDType::I8], DType::F16);
    let from_integers = integers.infer(&[&integer_operands.0, &integer_operands.1]);

    assert_eq!(from_fp8, Ok(vec![tensor_type(DType::F32, &[2, 4])]));
    assert_eq!(from_integers, Ok(vec![tensor_type(DType::F16, &[2, 4])]));
}

#[test]
fn fp8_operands_give_only_a_float_result_and_never_beside_an_integer_one() {
    let (e4m3, e5m2) = (tensor_type(DType::F8E4M3Fn, &[2, 3]), tensor_type(DType::F8E5M2, &[3, 4]));
    let i8 = tensor_type(DType::I8, &[3, 4]);
    let requantized = QMatMulAttrs {
        a: per_tensor(QuantDType::F8E4M3Fn),
        b: per_tensor(QuantDType::F8E5M2),
        output: QMatMulOutput::Quantized(per_tensor(QuantDType::I8)),
    };
    let mixed = dequantized([QuantDType::F8E4M3Fn, QuantDType::I8], DType::F32);
    let to_an_integer = dequantized([QuantDType::F8E4M3Fn, QuantDType::F8E5M2], DType::I32);
    let one_fp8 = |a, b| QMatMulAttrs {
        a: per_tensor(a),
        b: per_tensor(b),
        output: QMatMulOutput::Quantized(per_tensor(QuantDType::I8)),
    };
    let u8_a = tensor_type(DType::U8, &[2, 3]);
    let fp8_a = one_fp8(QuantDType::F8E4M3Fn, QuantDType::I8).infer(&[&e4m3, &i8]);
    let fp8_b = one_fp8(QuantDType::U8, QuantDType::F8E5M2).infer(&[&u8_a, &e5m2]);

    assert_eq!(
        requantized.infer(&[&e4m3, &e5m2]),
        Err(TypeError::UnsupportedDType(DType::F8E4M3Fn))
    );
    assert_eq!(fp8_a, Err(TypeError::UnsupportedDType(DType::F8E4M3Fn)));
    assert_eq!(fp8_b, Err(TypeError::UnsupportedDType(DType::F8E5M2)));
    assert_eq!(mixed.infer(&[&e4m3, &i8]), Err(TypeError::UnsupportedDType(DType::I8)));
    assert!(to_an_integer.infer(&[&e4m3, &e5m2]).is_err());
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
        output: QMatMulOutput::Quantized(per_tensor(QuantDType::U8)),
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
