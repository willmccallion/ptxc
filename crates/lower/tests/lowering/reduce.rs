use crate::elementwise::lowered;
use ptxc_core::{Axis, DType};
use ptxc_graph::{
    ArgKind, ArgReduceAttrs, CumSumAttrs, Inclusion, KeepDims, Op, ReduceAttrs, ReduceOp,
    ScanDirection, TieBreak,
};
use ptxc_testkit::golden;

fn reduce(op: ReduceOp, axes: &[usize], keep_dims: KeepDims) -> Op {
    Op::Reduce(ReduceAttrs { op, axes: axes.iter().copied().map(Axis::new).collect(), keep_dims })
}

#[test]
fn reductions_read_reduced_axes_through_reduction_axes() {
    let definitions = [
        lowered(reduce(ReduceOp::Mean, &[1], KeepDims::Keep), &[(DType::F16, &[2, 3, 4])]),
        lowered(reduce(ReduceOp::L2, &[0, 2], KeepDims::Drop), &[(DType::F32, &[2, 3, 4])]),
        lowered(reduce(ReduceOp::Max, &[], KeepDims::Keep), &[(DType::Bool, &[2])]),
    ];

    golden::check("reduce/reductions.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn integer_reductions_wrap_or_divide_exactly() {
    let definitions = [
        lowered(reduce(ReduceOp::SumSquare, &[0], KeepDims::Drop), &[(DType::I32, &[3])]),
        lowered(reduce(ReduceOp::Mean, &[0], KeepDims::Drop), &[(DType::I32, &[3])]),
        lowered(reduce(ReduceOp::LogSum, &[0], KeepDims::Drop), &[(DType::I64, &[3])]),
    ];

    golden::check("reduce/integer.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn log_sum_exp_shifts_by_a_maximum_computed_first() {
    let definition =
        lowered(reduce(ReduceOp::LogSumExp, &[1], KeepDims::Drop), &[(DType::Bf16, &[2, 3])]);

    golden::check("reduce/log_sum_exp.txt", &definition).unwrap();
}

#[test]
fn arg_reductions_and_running_sums() {
    let arg_min = ArgReduceAttrs {
        kind: ArgKind::Min,
        axis: Axis::new(0),
        keep_dims: KeepDims::Drop,
        tie: TieBreak::Last,
    };
    let reverse_exclusive = CumSumAttrs {
        axis: Axis::new(1),
        inclusion: Inclusion::Exclusive,
        direction: ScanDirection::Reverse,
    };
    let definitions = [
        lowered(Op::ArgReduce(arg_min), &[(DType::F16, &[3, 2])]),
        lowered(Op::CumSum(reverse_exclusive), &[(DType::F16, &[2, 3])]),
    ];

    golden::check("reduce/arg_cum_sum.txt", &definitions.join("\n")).unwrap();
}
