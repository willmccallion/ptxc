use crate::elementwise::lowered;
use ptxc_core::DType;
use ptxc_graph::{Activation, F32Attr, GeluApproximation, Op};
use ptxc_testkit::golden;

#[test]
fn activations_compose_scalar_ops_in_f32_and_relu_stays_in_its_type() {
    let definitions = [
        lowered(Op::Activation(Activation::Relu), &[(DType::I32, &[2])]),
        lowered(Op::Activation(Activation::Sigmoid), &[(DType::F16, &[2])]),
        lowered(Op::Activation(Activation::Gelu(GeluApproximation::Tanh)), &[(DType::F32, &[2])]),
        lowered(
            Op::Activation(Activation::Celu { alpha: F32Attr::new(2.0) }),
            &[(DType::Bf16, &[2])],
        ),
    ];

    golden::check("activation/activations.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn clip_and_prelu_select_without_rounding() {
    let definitions = [
        lowered(Op::Clip, &[(DType::F16, &[2]), (DType::F16, &[]), (DType::F16, &[])]),
        lowered(Op::PRelu, &[(DType::F16, &[2, 3]), (DType::F16, &[3])]),
        lowered(Op::PRelu, &[(DType::I64, &[2]), (DType::I64, &[1])]),
    ];

    golden::check("activation/clip_prelu.txt", &definitions.join("\n")).unwrap();
}
