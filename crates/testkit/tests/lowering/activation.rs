//! Every activation, `Clip` and `PRelu`, at every element type its type rule
//! admits.

use crate::cases::{Case, Extent, OperandSpec, admitted, graphs};
use crate::harness::check_against_reference;
use proptest::prelude::*;
use ptxc_core::DType;
use ptxc_graph::{Activation, F32Attr, GeluApproximation, Op};

const fn attr(value: f32) -> F32Attr {
    F32Attr::new(value)
}

const ACTIVATIONS: [Activation; 17] = [
    Activation::Relu,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Softplus,
    Activation::Softsign,
    Activation::HardSwish,
    Activation::Mish,
    Activation::Gelu(GeluApproximation::None),
    Activation::Gelu(GeluApproximation::Tanh),
    Activation::LeakyRelu { alpha: attr(0.1) },
    Activation::Elu { alpha: attr(0.5) },
    Activation::Celu { alpha: attr(1.0) },
    Activation::Celu { alpha: attr(2.5) },
    Activation::ThresholdedRelu { alpha: attr(0.5) },
    Activation::Selu { alpha: attr(1.673_263_2), gamma: attr(1.050_701) },
    Activation::HardSigmoid { alpha: attr(0.2), beta: attr(0.5) },
    Activation::HardSigmoid { alpha: attr(-0.7), beta: attr(0.1) },
];

pub(crate) fn activation_cases() -> Vec<Case> {
    admitted(ACTIVATIONS.into_iter().flat_map(|activation| {
        DType::ALL.into_iter().map(move |dtype| Case::new(Op::Activation(activation), &[dtype]))
    }))
}

/// `Clip` on any input with scalar bounds that may cross.
pub(crate) fn clip_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().map(|dtype| {
        let bound = OperandSpec::any(dtype).with_extent(Extent::Scalar);
        let x = OperandSpec::any(dtype).with_extent(Extent::Full);
        Case { op: Op::Clip, operands: vec![x, bound, bound] }
    }))
}

/// `PRelu` with a slope broadcasting to the input.
pub(crate) fn prelu_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().map(|dtype| {
        let x = OperandSpec::any(dtype).with_extent(Extent::Full);
        Case { op: Op::PRelu, operands: vec![x, OperandSpec::any(dtype)] }
    }))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn activations_agree_with_the_reference((graph, inputs) in graphs(activation_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn clip_agrees_with_the_reference((graph, inputs) in graphs(clip_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn prelu_agrees_with_the_reference((graph, inputs) in graphs(prelu_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }
}
