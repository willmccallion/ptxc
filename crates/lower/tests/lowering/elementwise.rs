use crate::{add_one, stages, tensor_type};
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_graph::{
    BinaryOp, CastAttrs, GraphBuilder, InfDetection, ModKind, Op, UnaryOp, VariadicOp,
};
use ptxc_loop::print_compute;
use ptxc_testkit::golden;

/// The printed definitions of a graph whose one node is `op` on inputs of
/// `operands`' types.
fn lowered(op: Op, operands: &[(DType, &[u64])]) -> String {
    let mut builder = GraphBuilder::new();
    let inputs: Vec<_> =
        operands.iter().map(|&(dtype, dims)| builder.add_input(tensor_type(dtype, dims))).collect();
    let out = add_one(&mut builder, op, &inputs);
    let graph = builder.finish(vec![out]).unwrap();
    stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect()
}

#[test]
fn arithmetic_on_narrow_floats_runs_in_f32_and_broadcasts_by_reading_index_zero() {
    let definitions = [
        lowered(Op::Binary(BinaryOp::Add), &[(DType::F16, &[2, 3]), (DType::F16, &[3])]),
        lowered(Op::Binary(BinaryOp::Less), &[(DType::Bf16, &[2, 1]), (DType::Bf16, &[1, 3])]),
        lowered(Op::Unary(UnaryOp::Exp), &[(DType::Bf16, &[4])]),
        lowered(Op::Unary(UnaryOp::IsNaN), &[(DType::F16, &[4])]),
    ];

    golden::check("elementwise/promotion.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn integer_ops_stay_in_their_type() {
    let definitions = [
        lowered(
            Op::Binary(BinaryOp::Mod(ModKind::Integer)),
            &[(DType::I8, &[3]), (DType::I8, &[])],
        ),
        lowered(Op::Binary(BinaryOp::Div), &[(DType::I32, &[3]), (DType::I32, &[3])]),
        lowered(Op::Unary(UnaryOp::Abs), &[(DType::I64, &[3])]),
    ];

    golden::check("elementwise/integer.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn pow_converts_its_exponent_by_the_base_type() {
    let definitions = [
        lowered(Op::Binary(BinaryOp::Pow), &[(DType::F16, &[3]), (DType::I32, &[3])]),
        lowered(Op::Binary(BinaryOp::Pow), &[(DType::I32, &[3]), (DType::I8, &[3])]),
        lowered(Op::Binary(BinaryOp::Pow), &[(DType::I32, &[3]), (DType::F32, &[3])]),
    ];

    golden::check("elementwise/pow.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn variadic_ops_fold_left_and_mean_divides_by_the_count() {
    let three = [(DType::F16, &[2][..]), (DType::F16, &[2]), (DType::F16, &[1])];
    let definitions = [
        lowered(Op::Variadic(VariadicOp::Mean), &three),
        lowered(Op::Variadic(VariadicOp::Max), &[(DType::I32, &[2]), (DType::I32, &[2])]),
        lowered(Op::Variadic(VariadicOp::Sum), &[(DType::F32, &[2])]),
    ];

    golden::check("elementwise/variadic.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn where_cast_and_constant_tests_need_no_arithmetic() {
    let saturating = CastAttrs::new(DType::F8E4M3Fn, Overflow::Saturate);
    let definitions = [
        lowered(Op::Where, &[(DType::Bool, &[2, 1]), (DType::F16, &[2, 2]), (DType::F16, &[])]),
        lowered(Op::Cast(saturating), &[(DType::F32, &[2])]),
        lowered(Op::Cast(CastAttrs::new(DType::I32, Overflow::Saturate)), &[(DType::I32, &[2])]),
        lowered(Op::Unary(UnaryOp::IsInf(InfDetection::Neither)), &[(DType::F32, &[2])]),
    ];

    golden::check("elementwise/select_cast.txt", &definitions.join("\n")).unwrap();
}
