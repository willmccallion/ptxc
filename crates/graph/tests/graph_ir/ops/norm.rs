use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::{Axis, DType, Dim, Rank};
use ptxc_graph::{
    BatchNormAttrs, F32Attr, InstanceNormAttrs, LayerNormAttrs, LayerNormKind, NormOutputs,
    SoftmaxAttrs, SoftmaxKind, StashType, TensorType, TypeError, TypeRule,
};

const EPSILON: F32Attr = F32Attr::new(1e-5);

fn axis(index: i64) -> Axis {
    Axis::from_signed(index, Rank::new(8)).unwrap()
}

fn layer_norm(index: i64, kind: LayerNormKind, stash: StashType) -> LayerNormAttrs {
    LayerNormAttrs { kind, axis: axis(index), epsilon: EPSILON, stash }
}

fn plain_layer_norm(index: i64) -> LayerNormAttrs {
    layer_norm(index, LayerNormKind::Layer(NormOutputs::Normalized), StashType::F32)
}

#[test]
fn softmaxes_keep_the_input_type() {
    let input = tensor_type(DType::F16, &[2, 5]);

    for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
        let attrs = SoftmaxAttrs { kind, axis: axis(1) };
        assert_eq!(attrs.infer(&[&input]), Ok(vec![input.clone()]), "{kind:?}");
    }
}

#[test]
fn softmax_needs_a_float_input_with_the_axis() {
    let attrs = SoftmaxAttrs { kind: SoftmaxKind::Softmax, axis: axis(2) };

    assert_eq!(
        attrs.infer(&[&f32_type(&[2, 5])]),
        Err(TypeError::AxisOutOfRange { axis: axis(2), rank: Rank::new(2) })
    );
    let integers = tensor_type(DType::I32, &[2, 5, 1]);
    assert_eq!(attrs.infer(&[&integers]), Err(TypeError::UnsupportedDType(DType::I32)));
}

#[test]
fn layer_norm_keeps_the_input_type_with_or_without_bias() {
    let (input, scale) = (f32_type(&[2, 3, 4]), f32_type(&[4]));

    assert_eq!(plain_layer_norm(2).infer(&[&input, &scale]), Ok(vec![input.clone()]));
    assert_eq!(plain_layer_norm(2).infer(&[&input, &scale, &scale]), Ok(vec![input.clone()]));
}

#[test]
fn layer_norm_statistics_keep_the_leading_axes_in_the_stash_type() {
    let input = tensor_type(DType::F16, &[2, 3, 4]);
    let scale = tensor_type(DType::F16, &[3, 4]);
    let attrs = layer_norm(1, LayerNormKind::Layer(NormOutputs::WithStatistics), StashType::Bf16);

    let statistics = tensor_type(DType::Bf16, &[2, 1, 1]);
    assert_eq!(
        attrs.infer(&[&input, &scale]),
        Ok(vec![input.clone(), statistics.clone(), statistics])
    );
}

#[test]
fn layer_norm_statistics_default_to_f32() {
    let input = tensor_type(DType::F16, &[2, 3]);
    let attrs = layer_norm(1, LayerNormKind::Layer(NormOutputs::WithStatistics), StashType::F32);

    let outputs = attrs.infer(&[&input, &tensor_type(DType::F16, &[3])]).unwrap();

    assert_eq!(outputs[1], f32_type(&[2, 1]));
}

#[test]
fn layer_norm_parameters_must_broadcast_to_the_normalized_shape() {
    let input = f32_type(&[2, 3, 4]);

    let error = plain_layer_norm(1).infer(&[&input, &f32_type(&[2, 3, 4])]);

    assert_eq!(
        error,
        Err(TypeError::NotUnidirectional { from: shape(&[2, 3, 4]), to: shape(&[3, 4]) })
    );
    assert!(plain_layer_norm(1).infer(&[&input, &f32_type(&[4]), &f32_type(&[5])]).is_err());
}

#[test]
fn layer_norm_parameters_share_the_input_type() {
    let input = f32_type(&[2, 4]);
    let half = tensor_type(DType::F16, &[4]);

    let error = plain_layer_norm(1).infer(&[&input, &half]);

    assert_eq!(error, Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 }));
    assert!(plain_layer_norm(1).infer(&[&input, &f32_type(&[4]), &half]).is_err());
}

#[test]
fn layer_norm_needs_its_axis_and_two_or_three_operands() {
    let input = f32_type(&[2, 4]);

    assert!(matches!(
        plain_layer_norm(2).infer(&[&input, &f32_type(&[4])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
    assert!(matches!(plain_layer_norm(1).infer(&[&input]), Err(TypeError::ArityBetween { .. })));
}

#[test]
fn rms_norm_takes_the_scale_type() {
    let attrs = layer_norm(1, LayerNormKind::Rms, StashType::F32);
    let input = f32_type(&[2, 4]);
    let scale = tensor_type(DType::F16, &[4]);

    assert_eq!(attrs.infer(&[&input, &scale]), Ok(vec![tensor_type(DType::F16, &[2, 4])]));
    assert!(matches!(attrs.infer(&[&input, &scale, &scale]), Err(TypeError::Arity { .. })));
    let integer_scale = tensor_type(DType::I32, &[4]);
    assert_eq!(
        attrs.infer(&[&input, &integer_scale]),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
}

fn batch_norm(
    input: &TensorType,
    scale: &TensorType,
    statistics: &TensorType,
) -> Result<Vec<TensorType>, TypeError> {
    BatchNormAttrs { epsilon: EPSILON }.infer(&[input, scale, scale, statistics, statistics])
}

#[test]
fn batch_norm_keeps_the_input_type() {
    let input = f32_type(&[2, 3, 4, 4]);

    assert_eq!(batch_norm(&input, &f32_type(&[3]), &f32_type(&[3])), Ok(vec![input.clone()]));
}

#[test]
fn batch_norm_treats_a_vector_as_one_channel() {
    let input = f32_type(&[5]);

    assert_eq!(batch_norm(&input, &f32_type(&[1]), &f32_type(&[1])), Ok(vec![input.clone()]));
}

#[test]
fn batch_norm_parameter_pairs_may_have_their_own_float_types() {
    let input = f32_type(&[2, 3]);
    let scale = tensor_type(DType::Bf16, &[3]);
    let statistics = tensor_type(DType::F16, &[3]);

    assert_eq!(batch_norm(&input, &scale, &statistics), Ok(vec![input.clone()]));
    let mixed = BatchNormAttrs { epsilon: EPSILON }.infer(&[
        &input,
        &scale,
        &f32_type(&[3]),
        &statistics,
        &statistics,
    ]);
    assert_eq!(mixed, Err(TypeError::MixedDTypes { lhs: DType::Bf16, rhs: DType::F32 }));
}

#[test]
fn batch_norm_parameters_have_one_entry_per_channel() {
    let input = f32_type(&[2, 3, 4]);

    assert_eq!(
        batch_norm(&input, &f32_type(&[3]), &f32_type(&[4])),
        Err(TypeError::ParameterLength { expected: Dim::new(3), actual: Dim::new(4) })
    );
    assert_eq!(
        batch_norm(&input, &f32_type(&[1, 3]), &f32_type(&[3])),
        Err(TypeError::Rank { expected: Rank::new(1), actual: Rank::new(2) })
    );
}

#[test]
fn batch_norm_needs_a_float_input_with_an_axis() {
    let scale = f32_type(&[1]);

    assert_eq!(
        batch_norm(&tensor_type(DType::I8, &[2, 1]), &scale, &scale),
        Err(TypeError::UnsupportedDType(DType::I8))
    );
    assert_eq!(
        batch_norm(&f32_type(&[]), &scale, &scale),
        Err(TypeError::RankBelow { minimum: Rank::new(1), actual: Rank::new(0) })
    );
    assert_eq!(
        batch_norm(&f32_type(&[2, 1]), &tensor_type(DType::I32, &[1]), &scale),
        Err(TypeError::UnsupportedDType(DType::I32))
    );
}

#[test]
fn instance_norm_keeps_the_input_type() {
    let input = f32_type(&[2, 3, 5]);
    let parameter = f32_type(&[3]);

    let inferred = InstanceNormAttrs { epsilon: EPSILON }.infer(&[&input, &parameter, &parameter]);

    assert_eq!(inferred, Ok(vec![input]));
}

#[test]
fn instance_norm_needs_spatial_axes_and_matching_parameters() {
    let attrs = InstanceNormAttrs { epsilon: EPSILON };
    let parameter = f32_type(&[3]);

    assert_eq!(
        attrs.infer(&[&f32_type(&[2, 3]), &parameter, &parameter]),
        Err(TypeError::RankBelow { minimum: Rank::new(3), actual: Rank::new(2) })
    );
    let half = tensor_type(DType::F16, &[3]);
    assert_eq!(
        attrs.infer(&[&f32_type(&[2, 3, 5]), &parameter, &half]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
    assert!(attrs.infer(&[&f32_type(&[2, 3, 5]), &f32_type(&[4]), &f32_type(&[4])]).is_err());
    assert!(attrs.infer(&[&tensor_type(DType::I32, &[2, 3, 5]), &parameter, &parameter]).is_err());
}
