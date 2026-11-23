//! Every elementwise op, `Where` and `Cast`, at every element type its type
//! rule admits, with broadcasting.

use super::cases::{Case, OperandSpec, admitted};
use super::values::Domain;
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

/// Every unary op at every element type its type rule admits.
pub fn unary_cases() -> Vec<Case> {
    admitted(
        UNARY.into_iter().flat_map(|op| {
            DType::ALL.into_iter().map(move |dtype| Case::new(Op::Unary(op), &[dtype]))
        }),
    )
}

/// Every binary op on two operands of one type, and `Pow` with every
/// exponent type.
///
/// An integer raised to a float exponent is computed in f32
/// (see the NOTE in lowering), so its operands stay small enough to be exact.
pub fn binary_cases() -> Vec<Case> {
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

/// `Max`, `Min`, `Sum` and `Mean` of one to three operands of each type.
pub fn variadic_cases() -> Vec<Case> {
    let ops = [VariadicOp::Max, VariadicOp::Min, VariadicOp::Sum, VariadicOp::Mean];
    admitted(ops.into_iter().flat_map(|op| {
        DType::ALL.into_iter().flat_map(move |dtype| {
            (1..=3).map(move |count| Case::new(Op::Variadic(op), &vec![dtype; count]))
        })
    }))
}

/// `Where` choosing between two operands of each type.
pub fn where_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().map(|dtype| Case::new(Op::Where, &[DType::Bool, dtype, dtype])))
}

/// `Cast` between every pair of types, saturating and not.
pub fn cast_cases() -> Vec<Case> {
    admitted(DType::ALL.into_iter().flat_map(|from| {
        DType::ALL.into_iter().flat_map(move |to| {
            [Overflow::Saturate, Overflow::NonFinite]
                .map(|overflow| Case::new(Op::Cast(CastAttrs::new(to, overflow)), &[from]))
        })
    }))
}
