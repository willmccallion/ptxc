//! Every reduction, arg-reduction and running sum at every element type its
//! type rule admits, over random axes.

use crate::cases::single_node;
use crate::harness::check_within;
use crate::values::{Domain, axis_dims, output_dims, shape, tensor};
use proptest::prelude::*;
use proptest::sample::{select, subsequence};
use ptxc_core::{Axis, DType, ElementCount, HostTensor};
use ptxc_graph::{
    ArgKind, ArgReduceAttrs, CumSumAttrs, Graph, Inclusion, KeepDims, Layout, Op, ReduceAttrs,
    ReduceOp, ScanDirection, TensorType, TieBreak, TypeRule,
};
use ptxc_testkit::compare::Tolerance;

const REDUCE_OPS: [ReduceOp; 10] = [
    ReduceOp::Sum,
    ReduceOp::Mean,
    ReduceOp::Prod,
    ReduceOp::Max,
    ReduceOp::Min,
    ReduceOp::SumSquare,
    ReduceOp::L1,
    ReduceOp::L2,
    ReduceOp::LogSum,
    ReduceOp::LogSumExp,
];

/// The `(op, dtype)` pairs `Reduce`'s type rule admits.
fn reduce_dtypes() -> Vec<(ReduceOp, DType)> {
    let vector = |dtype| TensorType::new(dtype, shape(&[2]), Layout::RowMajor);
    REDUCE_OPS
        .into_iter()
        .flat_map(|op| DType::ALL.into_iter().map(move |dtype| (op, dtype)))
        .filter(|&(op, dtype)| {
            let attrs = ReduceAttrs { op, axes: Vec::new(), keep_dims: KeepDims::Keep };
            attrs.infer(&[&vector(dtype)]).is_ok()
        })
        .collect()
}

/// The values a reduction's input takes: products and squares stay finite
/// in f32 where the f64 reference's would, and integer results computed in
/// f32 stay exact (see the NOTE in lowering).
const fn reduce_domain(op: ReduceOp, dtype: DType) -> Domain {
    match op {
        ReduceOp::Prod | ReduceOp::L2 if dtype.is_float() => Domain::Moderate,
        ReduceOp::Mean => Domain::Moderate,
        ReduceOp::L2 | ReduceOp::LogSum | ReduceOp::LogSumExp if !dtype.is_float() => {
            Domain::SmallInt
        }
        _ => Domain::Any,
    }
}

fn keep_dims() -> impl Strategy<Value = KeepDims> {
    prop_oneof![Just(KeepDims::Keep), Just(KeepDims::Drop)]
}

/// The tolerance for outputs that each combine up to all of `input`'s
/// elements.
fn reduction_tolerance(input: &HostTensor) -> impl Fn(DType) -> Tolerance {
    let length = input.shape().element_count().get().max(1);
    move |dtype| Tolerance::for_dtype(dtype).for_reduction(ElementCount::new(length))
}

pub(crate) fn reduce_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    (select(reduce_dtypes()), output_dims(), keep_dims())
        .prop_flat_map(|((op, dtype), dims, keep_dims)| {
            let rank = dims.len();
            let axes = subsequence((0..rank).collect::<Vec<_>>(), 0..=rank);
            let input = tensor(dtype, dims, reduce_domain(op, dtype));
            (Just(op), Just(keep_dims), axes, input)
        })
        .prop_map(|(op, keep_dims, axes, input)| {
            let axes = axes.into_iter().map(Axis::new).collect();
            let attrs = ReduceAttrs { op, axes, keep_dims };
            (single_node(Op::Reduce(attrs), std::slice::from_ref(&input)), vec![input])
        })
}

/// A rank-1-or-more input of a type `admits` accepts and one of its axes.
fn input_and_axis(dtypes: Vec<DType>) -> impl Strategy<Value = (HostTensor, Axis)> {
    (select(dtypes), axis_dims()).prop_flat_map(|(dtype, dims)| {
        let rank = dims.len();
        (tensor(dtype, dims, Domain::Any), (0..rank).prop_map(Axis::new))
    })
}

pub(crate) fn arg_reduce_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    let numeric =
        [DType::F32, DType::F16, DType::Bf16, DType::I8, DType::U8, DType::I32, DType::I64];
    let kind = prop_oneof![Just(ArgKind::Max), Just(ArgKind::Min)];
    let tie = prop_oneof![Just(TieBreak::First), Just(TieBreak::Last)];
    (input_and_axis(numeric.to_vec()), kind, tie, keep_dims()).prop_map(
        |((input, axis), kind, tie, keep_dims)| {
            let attrs = ArgReduceAttrs { kind, axis, keep_dims, tie };
            (single_node(Op::ArgReduce(attrs), std::slice::from_ref(&input)), vec![input])
        },
    )
}

pub(crate) fn cum_sum_graphs() -> impl Strategy<Value = (Graph, Vec<HostTensor>)> {
    let summable = [DType::F32, DType::F16, DType::Bf16, DType::I32, DType::I64];
    let inclusion = prop_oneof![Just(Inclusion::Inclusive), Just(Inclusion::Exclusive)];
    let direction = prop_oneof![Just(ScanDirection::Forward), Just(ScanDirection::Reverse)];
    (input_and_axis(summable.to_vec()), inclusion, direction).prop_map(
        |((input, axis), inclusion, direction)| {
            let attrs = CumSumAttrs { axis, inclusion, direction };
            (single_node(Op::CumSum(attrs), std::slice::from_ref(&input)), vec![input])
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn reductions_agree_with_the_reference((graph, inputs) in reduce_graphs()) {
        let tolerance = reduction_tolerance(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn arg_reductions_agree_with_the_reference((graph, inputs) in arg_reduce_graphs()) {
        check_within(graph, &inputs, Tolerance::for_dtype).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn running_sums_agree_with_the_reference((graph, inputs) in cum_sum_graphs()) {
        let tolerance = reduction_tolerance(&inputs[0]);
        check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)?;
    }
}
