//! Every elementwise op, `Where` and `Cast`, at every element type its type
//! rule admits, with broadcasting.

use crate::harness::check_against_reference;
use crate::values::{Domain, operand_dims, output_dims, shape, tensor};
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::float::Overflow;
use ptxc_core::{DType, HostTensor};
use ptxc_graph::{
    BinaryOp, CastAttrs, Graph, GraphBuilder, InfDetection, Layout, ModKind, Op, TensorType,
    TypeRule, UnaryOp, VariadicOp,
};

/// An op and each operand's element type and values.
#[derive(Clone, Debug)]
struct Case {
    op: Op,
    operands: Vec<(DType, Domain)>,
}

impl Case {
    fn new(op: Op, dtypes: &[DType]) -> Self {
        Self { op, operands: dtypes.iter().map(|&dtype| (dtype, Domain::Any)).collect() }
    }

    fn is_admitted(&self) -> bool {
        let types: Vec<TensorType> = self
            .operands
            .iter()
            .map(|&(dtype, _)| TensorType::new(dtype, shape(&[1]), Layout::RowMajor))
            .collect();
        self.op.infer(&types.iter().collect::<Vec<_>>()).is_ok()
    }
}

fn admitted(cases: impl IntoIterator<Item = Case>) -> Vec<Case> {
    let admitted: Vec<Case> = cases.into_iter().filter(Case::is_admitted).collect();
    assert!(!admitted.is_empty());
    admitted
}

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
            Case {
                op: Op::Binary(BinaryOp::Pow),
                operands: vec![(base, domain), (exponent, domain)],
            }
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

/// A graph of one node of a case drawn from `cases`, and inputs for it.
fn graphs(cases: Vec<Case>) -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    select(cases)
        .prop_flat_map(|case| {
            output_dims().prop_flat_map(move |output| {
                let inputs: Vec<BoxedStrategy<HostTensor>> = case
                    .operands
                    .iter()
                    .map(|&(dtype, domain)| {
                        operand_dims(output.clone())
                            .prop_flat_map(move |dims| tensor(dtype, dims, domain))
                            .boxed()
                    })
                    .collect();
                (Just(case.op.clone()), inputs)
            })
        })
        .prop_map(|(op, inputs)| (single_node(op, &inputs), inputs))
}

fn single_node(op: Op, inputs: &[HostTensor]) -> Graph {
    let mut builder = GraphBuilder::new();
    let operands = inputs
        .iter()
        .map(|input| {
            builder.add_input(TensorType::new(
                input.dtype(),
                input.shape().clone(),
                Layout::RowMajor,
            ))
        })
        .collect();
    let outputs = builder.add_node(op, operands).unwrap();
    builder.finish(outputs).unwrap()
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
