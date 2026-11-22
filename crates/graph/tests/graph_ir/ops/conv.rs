use crate::common::{f32_type, tensor_type};
use ptxc_core::{DType, Dim, Rank};
use ptxc_graph::{
    ConvAttrs, Dilation, Groups, Padding, Stride, TensorType, TypeError, TypeRule, WindowAxis,
};

const fn axis(stride: u64, dilation: u64, pad: u64) -> WindowAxis {
    WindowAxis {
        stride: Stride::new(stride).unwrap(),
        dilation: Dilation::new(dilation).unwrap(),
        pad_begin: Padding::new(pad),
        pad_end: Padding::new(pad),
    }
}

const fn conv(axes: Vec<WindowAxis>, groups: u64) -> ConvAttrs {
    ConvAttrs { axes, groups: Groups::new(groups).unwrap() }
}

fn conv_2d(stride: u64, dilation: u64, pad: u64) -> ConvAttrs {
    conv(vec![axis(stride, dilation, pad); 2], 1)
}

fn infer(attrs: &ConvAttrs, x: &[u64], w: &[u64]) -> Result<Vec<TensorType>, TypeError> {
    attrs.infer(&[&f32_type(x), &f32_type(w)])
}

#[test]
fn an_unpadded_convolution_shrinks_by_the_kernel() {
    assert_eq!(
        infer(&conv_2d(1, 1, 0), &[1, 3, 5, 5], &[8, 3, 3, 3]),
        Ok(vec![f32_type(&[1, 8, 3, 3])])
    );
}

#[test]
fn padding_stride_and_dilation_follow_the_onnx_formula() {
    let (x, w) = (&[1, 3, 5, 5][..], &[8, 3, 3, 3][..]);

    assert_eq!(infer(&conv_2d(1, 1, 1), x, w), Ok(vec![f32_type(&[1, 8, 5, 5])]));
    assert_eq!(infer(&conv_2d(2, 1, 1), x, w), Ok(vec![f32_type(&[1, 8, 3, 3])]));
    assert_eq!(infer(&conv_2d(1, 2, 0), x, w), Ok(vec![f32_type(&[1, 8, 1, 1])]));
    assert_eq!(infer(&conv_2d(2, 1, 0), &[1, 3, 6, 6], w), Ok(vec![f32_type(&[1, 8, 2, 2])]));
}

#[test]
fn asymmetric_padding_counts_both_ends() {
    let mut window = axis(1, 1, 0);
    window.pad_end = Padding::new(2);

    let inferred = infer(&conv(vec![window], 1), &[1, 1, 4], &[1, 1, 3]);

    assert_eq!(inferred, Ok(vec![f32_type(&[1, 1, 4])]));
}

#[test]
fn one_dimensional_convolution_is_supported() {
    let inferred = infer(&conv(vec![axis(1, 1, 0)], 1), &[2, 3, 10], &[4, 3, 3]);

    assert_eq!(inferred, Ok(vec![f32_type(&[2, 4, 8])]));
}

#[test]
fn grouped_weights_hold_the_channels_of_one_group() {
    let grouped = conv(vec![axis(1, 1, 0); 2], 2);

    assert_eq!(infer(&grouped, &[1, 4, 5, 5], &[6, 2, 3, 3]), Ok(vec![f32_type(&[1, 6, 3, 3])]));
    assert_eq!(
        infer(&grouped, &[1, 4, 5, 5], &[6, 3, 3, 3]),
        Err(TypeError::Channels {
            input: Dim::new(4),
            weights: Dim::new(3),
            groups: Groups::new(2).unwrap()
        })
    );
    assert_eq!(
        infer(&grouped, &[1, 4, 5, 5], &[5, 2, 3, 3]),
        Err(TypeError::FiltersNotDivisible {
            filters: Dim::new(5),
            groups: Groups::new(2).unwrap()
        })
    );
}

#[test]
fn bias_has_one_entry_per_filter() {
    let attrs = conv_2d(1, 1, 0);
    let (x, w) = (f32_type(&[1, 3, 5, 5]), f32_type(&[8, 3, 3, 3]));

    assert!(attrs.infer(&[&x, &w, &f32_type(&[8])]).is_ok());
    assert_eq!(
        attrs.infer(&[&x, &w, &f32_type(&[7])]),
        Err(TypeError::ParameterLength { expected: Dim::new(8), actual: Dim::new(7) })
    );
    assert_eq!(
        attrs.infer(&[&x, &w, &f32_type(&[1, 8])]),
        Err(TypeError::Rank { expected: Rank::new(1), actual: Rank::new(2) })
    );
    assert_eq!(
        attrs.infer(&[&x, &w, &tensor_type(DType::F16, &[8])]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn operand_ranks_must_match_the_window() {
    let attrs = conv_2d(1, 1, 0);

    assert_eq!(
        infer(&attrs, &[1, 3, 5], &[8, 3, 3, 3]),
        Err(TypeError::Rank { expected: Rank::new(4), actual: Rank::new(3) })
    );
    assert!(infer(&attrs, &[1, 3, 5, 5], &[8, 3, 3]).is_err());
}

#[test]
fn a_kernel_larger_than_the_padded_input_is_rejected() {
    let error = infer(&conv_2d(1, 1, 0), &[1, 3, 2, 5], &[8, 3, 3, 3]);

    assert_eq!(error, Err(TypeError::WindowTooLarge { input: Dim::new(2), kernel: Dim::new(3) }));
    assert!(infer(&conv_2d(1, 1, 1), &[1, 3, 1, 5], &[8, 3, 3, 3]).is_ok());
}

#[test]
fn a_zero_length_kernel_is_rejected() {
    assert_eq!(infer(&conv_2d(1, 1, 0), &[1, 3, 5, 5], &[8, 3, 0, 3]), Err(TypeError::EmptyKernel));
}

#[test]
fn convolution_takes_only_floats_of_one_type() {
    let attrs = conv_2d(1, 1, 0);
    let x = tensor_type(DType::I32, &[1, 3, 5, 5]);
    let w = tensor_type(DType::I32, &[8, 3, 3, 3]);

    assert_eq!(attrs.infer(&[&x, &w]), Err(TypeError::UnsupportedDType(DType::I32)));
    assert_eq!(
        attrs.infer(&[&f32_type(&[1, 3, 5, 5]), &tensor_type(DType::Bf16, &[8, 3, 3, 3])]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::Bf16 })
    );
}

#[test]
fn convolution_takes_two_or_three_operands() {
    let x = f32_type(&[1, 3, 5, 5]);

    assert!(matches!(conv_2d(1, 1, 0).infer(&[&x]), Err(TypeError::ArityBetween { .. })));
    assert!(matches!(conv_2d(1, 1, 0).infer(&[&x; 4]), Err(TypeError::ArityBetween { .. })));
}

#[test]
fn window_newtypes_reject_zero_and_keep_their_value() {
    assert_eq!(Stride::new(0), None);
    assert_eq!(Dilation::new(0), None);
    assert_eq!(Groups::new(0), None);
    assert_eq!(Stride::new(3).map(Stride::get), Some(3));
    assert_eq!(Dilation::new(2).map(Dilation::get), Some(2));
    assert_eq!(Groups::new(4).map(Groups::get), Some(4));
    assert_eq!(Padding::new(5).get(), 5);
}
