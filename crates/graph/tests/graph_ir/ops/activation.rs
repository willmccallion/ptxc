use crate::common::{f32_type, tensor_type};
use ptxc_core::DType;
use ptxc_graph::{Activation, F32Attr, GeluApproximation, Op, TensorType, TypeError, TypeRule};
use std::hash::{Hash, Hasher};

const ALPHA: F32Attr = F32Attr::new(0.5);

/// One of each activation, attributes included.
const ALL: [Activation; 14] = [
    Activation::Relu,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Softplus,
    Activation::Softsign,
    Activation::HardSwish,
    Activation::Mish,
    Activation::Gelu(GeluApproximation::Tanh),
    Activation::LeakyRelu { alpha: ALPHA },
    Activation::Elu { alpha: ALPHA },
    Activation::Celu { alpha: ALPHA },
    Activation::ThresholdedRelu { alpha: ALPHA },
    Activation::Selu { alpha: ALPHA, gamma: ALPHA },
    Activation::HardSigmoid { alpha: ALPHA, beta: ALPHA },
];

fn scalar(dtype: DType) -> TensorType {
    tensor_type(dtype, &[])
}

#[test]
fn activations_keep_their_operand_type() {
    let operand = tensor_type(DType::Bf16, &[2, 3]);

    for activation in ALL {
        assert_eq!(activation.infer(&[&operand]), Ok(vec![operand.clone()]), "{activation:?}");
    }
}

#[test]
fn only_relu_accepts_integers() {
    let operand = tensor_type(DType::I32, &[4]);

    for activation in ALL {
        let accepted = activation.infer(&[&operand]).is_ok();
        assert_eq!(accepted, activation == Activation::Relu, "{activation:?}");
    }
}

#[test]
fn relu_rejects_unsigned_integers() {
    let operand = tensor_type(DType::U8, &[4]);

    assert_eq!(Activation::Relu.infer(&[&operand]), Err(TypeError::UnsupportedDType(DType::U8)));
}

#[test]
fn float_attributes_compare_by_bits() {
    assert_eq!(F32Attr::new(f32::NAN), F32Attr::new(f32::NAN));
    assert_ne!(F32Attr::new(0.0), F32Attr::new(-0.0));
    assert_eq!(F32Attr::new(0.125).get(), 0.125);
}

/// Ops are keyed in hash maps (CSE), so the hash must depend on the value.
#[test]
fn different_float_attributes_hash_differently() {
    let hash = |value: F32Attr| {
        let mut hasher = std::hash::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    };

    assert_ne!(hash(F32Attr::new(0.25)), hash(F32Attr::new(0.5)));
    assert_eq!(hash(F32Attr::new(0.25)), hash(F32Attr::new(0.25)));
}

#[test]
fn clip_keeps_the_input_type() {
    let input = tensor_type(DType::I8, &[2, 3]);
    let bound = scalar(DType::I8);

    assert_eq!(Op::Clip.infer(&[&input, &bound, &bound]), Ok(vec![input.clone()]));
}

#[test]
fn clip_bounds_must_be_scalars_of_the_input_type() {
    let input = f32_type(&[3]);
    let scalar_bound = scalar(DType::F32);
    let wide_bound = f32_type(&[1]);
    let half_bound = scalar(DType::F16);

    assert_eq!(
        Op::Clip.infer(&[&input, &scalar_bound, &wide_bound]),
        Err(TypeError::NotScalar(wide_bound.shape().clone()))
    );
    assert_eq!(
        Op::Clip.infer(&[&input, &half_bound, &scalar_bound]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}

#[test]
fn clip_rejects_bool() {
    let input = scalar(DType::Bool);

    assert_eq!(
        Op::Clip.infer(&[&input, &input, &input]),
        Err(TypeError::UnsupportedDType(DType::Bool))
    );
}

#[test]
fn prelu_slope_broadcasts_to_the_input() {
    let input = f32_type(&[2, 3, 4]);
    let slope = f32_type(&[3, 1]);

    assert_eq!(Op::PRelu.infer(&[&input, &slope]), Ok(vec![input.clone()]));
}

#[test]
fn prelu_slope_may_not_grow_the_input() {
    let input = f32_type(&[3, 1]);
    let slope = f32_type(&[2, 3, 4]);

    let error = Op::PRelu.infer(&[&input, &slope]);

    let expected =
        TypeError::NotUnidirectional { from: slope.shape().clone(), to: input.shape().clone() };
    assert_eq!(error, Err(expected));
    assert!(Op::PRelu.infer(&[&f32_type(&[3]), &f32_type(&[4])]).is_err());
}

#[test]
fn prelu_takes_wide_integers_but_not_i8() {
    let wide = tensor_type(DType::I64, &[3]);
    let narrow = tensor_type(DType::I8, &[3]);

    assert!(Op::PRelu.infer(&[&wide, &wide]).is_ok());
    assert_eq!(Op::PRelu.infer(&[&narrow, &narrow]), Err(TypeError::UnsupportedDType(DType::I8)));
    assert_eq!(
        Op::PRelu.infer(&[&f32_type(&[3]), &tensor_type(DType::F16, &[3])]),
        Err(TypeError::MixedDTypes { lhs: DType::F32, rhs: DType::F16 })
    );
}
