//! Every elementwise op, `Where` and `Cast`, at every element type its type
//! rule admits, with broadcasting.

use crate::cases::{Case, OperandSpec, admitted, graphs};
use crate::harness::check_against_reference;
use crate::values::Domain;
use proptest::prelude::*;
use ptxc_core::DType;
use ptxc_core::float::Overflow;
use ptxc_graph::{BinaryOp, CastAttrs, InfDetection, ModKind, Op, UnaryOp, VariadicOp};

const UNARY: [UnaryOp; 19] = [
    UnaryOp::Abs,
    UnaryOp::Neg,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sqrt,
    UnaryOp::Reciprocal,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Sign,
    UnaryOp::Erf,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Not,
    UnaryOp::IsNaN,
    UnaryOp::IsInf(InfDetection::Both),
    UnaryOp::IsInf(InfDetection::Positive),
    UnaryOp::IsInf(InfDetection::Negative),
    UnaryOp::IsInf(InfDetection::Neither),
];

const BINARY: [BinaryOp; 15] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Pow,
    BinaryOp::Mod(ModKind::Integer),
    BinaryOp::Mod(ModKind::Fmod),
    BinaryOp::Equal,
    BinaryOp::Less,
    BinaryOp::LessOrEqual,
    BinaryOp::Greater,
    BinaryOp::GreaterOrEqual,
    BinaryOp::And,
    BinaryOp::Or,
    BinaryOp::Xor,
];

fn unary_cases() -> Vec<Case> {
    admitted(
        UNARY.into_iter().flat_map(|op| {
            DType::ALL.into_iter().map(move |dtype| Case::new(Op::Unary(op), &[dtype]))
        }),
    )
}

/// Every binary op on two operands of one type, and `Pow` with every
/// exponent type. An integer raised to a float exponent is computed in f32
/// (see the NOTE in lowering), so its operands stay small enough to be exact.
fn binary_cases() -> Vec<Case> {
    let same = BINARY.into_iter().flat_map(|op| {
        DType::ALL.into_iter().map(move |dtype| Case::new(Op::Binary(op), &[dtype, dtype]))
    });
    let pow = DType::ALL.into_iter().flat_map(|base| {
        DType::ALL.into_iter().map(move |exponent| {
            let domain = if !base.is_float() && exponent.is_float() {
                Domain::SmallInt
            } else {
                Domain::Any
            };
            let operands =
                [base, exponent].map(|dtype| OperandSpec::any(dtype).with_domain(domain));
            Case { op: Op::Binary(BinaryOp::Pow), operands: operands.into() }
        })
    });
    admitted(same.chain(pow))
}

fn variadic_cases() -> Vec<Case> {
    let ops = [VariadicOp::Max, VariadicOp::Min, VariadicOp::Sum, VariadicOp::Mean];
    admitted(ops.into_iter().flat_map(|op| {
        DType::ALL.into_iter().flat_map(move |dtype| {
            (1..=3).map(move |count| Case::new(Op::Variadic(op), &vec![dtype; count]))
        })
    }))
}

fn where_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().map(|dtype| Case::new(Op::Where, &[DType::Bool, dtype, dtype])))
}

fn cast_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().flat_map(|from| {
        DType::ALL.into_iter().flat_map(move |to| {
            [Overflow::Saturate, Overflow::NonFinite]
                .map(|overflow| Case::new(Op::Cast(CastAttrs::new(to, overflow)), &[from]))
        })
    }))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn unary_ops_agree_with_the_reference((graph, inputs) in graphs(unary_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn binary_ops_agree_with_the_reference((graph, inputs) in graphs(binary_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn variadic_ops_agree_with_the_reference((graph, inputs) in graphs(variadic_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn where_agrees_with_the_reference((graph, inputs) in graphs(where_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn casts_agree_with_the_reference((graph, inputs) in graphs(cast_cases())) {
        check_against_reference(graph, &inputs).map_err(TestCaseError::fail)?;
    }
}
