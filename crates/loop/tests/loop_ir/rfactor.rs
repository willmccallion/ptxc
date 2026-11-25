//! A reduction split into slices: the partial and combine definitions
//! `rfactor` builds, that together they compute what the whole does, and
//! the definitions it refuses.

use crate::compute_build::{binary, load, shape};
use crate::interp::{floats, ints, run_default, tensor};
use ptxc_core::float::{F16, Overflow};
use ptxc_core::{DType, Dim, HostTensor, Idx, TensorData};
use ptxc_loop::{
    BinaryOp, CombineInput, ComputeBuilder, ComputeDef, ComputeName, ExprId, IterVarId, ReduceOp,
    RfactorError, Rfactored, Scalar, Slices, TensorExpr, TensorName, TensorParamId, VarName,
    print_compute, rfactor,
};
use ptxc_testkit::golden;
use std::sync::Arc;

const fn slices(count: u64) -> Slices {
    Slices::new(count).unwrap()
}

fn widened(value: TensorExpr) -> TensorExpr {
    TensorExpr::Cast { to: DType::F32, overflow: Overflow::Saturate, value: Box::new(value) }
}

/// `out[i, j] = f16(2 * sum over k of a[k, i] * b[j, k] + bias[j])`, `a`
/// and `b` stored transposed, so the axis indexes `a`'s first dimension
/// and `b`'s last; and the summed axis.
fn product_with_epilogue(extents: [u64; 3]) -> (ComputeDef, IterVarId) {
    let [rows, columns, depth] = extents;
    let mut compute = ComputeBuilder::new(ComputeName::new("gemm"));
    let lhs = compute.input(TensorName::new("a"), DType::F16, shape(&[depth, rows]));
    let bias = compute.input(TensorName::new("bias"), DType::F16, shape(&[columns]));
    let rhs = compute.input(TensorName::new("b"), DType::F16, shape(&[columns, depth]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let product = binary(BinaryOp::Mul, widened(load(lhs, &[k, i])), widened(load(rhs, &[j, k])));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(product) };
    let scaled = binary(BinaryOp::Mul, TensorExpr::Const(Scalar::f32(2.0)), sum);
    let biased = binary(BinaryOp::Add, scaled, widened(load(bias, &[j])));
    let out =
        TensorExpr::Cast { to: DType::F16, overflow: Overflow::Saturate, value: Box::new(biased) };
    compute.output(TensorName::new("out"), out);
    (compute.finish().unwrap(), k_axis)
}

/// `out[i, j] = op over k of x[i, k, j]` on `i64`, whose results are exact
/// in any order.
fn int_reduction(op: ReduceOp, extents: [u64; 3]) -> (ComputeDef, IterVarId) {
    let [rows, depth, columns] = extents;
    let mut compute = ComputeBuilder::new(ComputeName::new("reduce"));
    let x = compute.input(TensorName::new("x"), DType::I64, shape(&extents));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let body = Box::new(load(x, &[i, k, j]));
    compute.output(TensorName::new("out"), TensorExpr::Reduce { op, axes: vec![k_axis], body });
    (compute.finish().unwrap(), k_axis)
}

/// The partial definition run on the loop interpreter, each of the
/// original's `inputs` viewed as the partial declares it.
fn partials(split: &Rfactored, inputs: &[HostTensor]) -> HostTensor {
    let viewed: Vec<HostTensor> = split
        .partial_inputs
        .iter()
        .zip(split.partial.inputs().iter())
        .map(|(&original, param)| {
            HostTensor::new(param.shape.clone(), inputs[original.index()].data().clone()).unwrap()
        })
        .collect();
    run_default(&Arc::new(split.partial.clone()), &viewed).unwrap().remove(0)
}

/// The split definitions run in turn on the loop interpreter, on the
/// original's `inputs`.
fn run_split(split: &Rfactored, inputs: &[HostTensor]) -> Vec<HostTensor> {
    let partials = partials(split, inputs);
    let arguments: Vec<HostTensor> = split
        .combine_inputs
        .iter()
        .map(|input| match input {
            CombineInput::Partials => partials.clone(),
            CombineInput::Original(original) => inputs[original.index()].clone(),
        })
        .collect();
    run_default(&Arc::new(split.combine.clone()), &arguments).unwrap()
}

/// `count` values counting up from `start` in steps of `step`.
fn counting(count: u64, start: i64, step: i64) -> Vec<i64> {
    (0..i64::try_from(count).unwrap()).map(|n| start + n * step).collect()
}

#[test]
fn a_split_product_sums_each_slice_then_the_slices_inside_its_epilogue() {
    let (compute, k) = product_with_epilogue([3, 4, 12]);

    let split = rfactor(&compute, k, slices(3)).unwrap();

    let text = format!("{}\n{}", print_compute(&split.partial), print_compute(&split.combine));
    golden::check("compute/rfactor_product.txt", &text).unwrap();
    assert_eq!(
        split.partial_inputs.iter().copied().collect::<Vec<_>>(),
        [TensorParamId::new(0), TensorParamId::new(2)]
    );
    let bias = CombineInput::Original(TensorParamId::new(1));
    assert_eq!(
        split.combine_inputs.iter().copied().collect::<Vec<_>>(),
        [CombineInput::Partials, bias]
    );
}

#[test]
fn a_split_float_product_computes_what_the_whole_does_where_its_sums_are_exact() {
    let (compute, k) = product_with_epilogue([3, 5, 12]);
    // Multiples of 1/8 in [-1, 1], whose products f32 sums exactly.
    let eighths = |count: u64| -> Vec<f32> {
        counting(count, -7, 3)
            .iter()
            .map(|&n| f32::from(i8::try_from(n % 9).unwrap()) / 8.0)
            .collect()
    };
    let inputs = [
        tensor(&[12, 3], half(&eighths(36))),
        tensor(&[5], half(&eighths(5))),
        tensor(&[5, 12], half(&eighths(60))),
    ];
    let whole = run_default(&Arc::new(compute.clone()), &inputs).unwrap();

    for count in [2, 3, 4, 6, 12] {
        let split = run_split(&rfactor(&compute, k, slices(count)).unwrap(), &inputs);

        assert_eq!(halves(&split[0]), halves(&whole[0]), "{count} slices");
    }
}

fn half(values: &[f32]) -> Vec<F16> {
    values.iter().map(|&value| F16::from_f32(value)).collect()
}

fn halves(output: &HostTensor) -> Vec<f32> {
    let TensorData::F16(values) = output.data() else { panic!("{:?}", output.dtype()) };
    values.iter().map(|value| value.to_f32()).collect()
}

#[test]
fn every_reduction_splits_into_partials_of_its_own_op() {
    for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
        let (compute, k) = int_reduction(op, [2, 6, 3]);
        let x = tensor(&[2, 6, 3], counting(36, -17, 5).iter().map(|n| n % 7).collect::<Vec<_>>());
        let whole = run_default(&Arc::new(compute.clone()), std::slice::from_ref(&x)).unwrap();

        let split = run_split(&rfactor(&compute, k, slices(2)).unwrap(), &[x]);

        assert_eq!(ints(&split[0]), ints(&whole[0]), "{op:?}");
    }
}

#[test]
fn the_partials_hold_each_slice_outermost_in_the_reductions_type() {
    let (compute, k) = product_with_epilogue([3, 4, 12]);

    let split = rfactor(&compute, k, slices(4)).unwrap();

    assert_eq!(split.partial.output_shape(), &shape(&[4, 3, 4]));
    assert_eq!(split.partial.outputs()[0].dtype, DType::F32);
    assert_eq!(split.partial.inputs()[TensorParamId::new(0)].shape, shape(&[4, 3, 3]));
    assert_eq!(split.partial.inputs()[TensorParamId::new(1)].shape, shape(&[4, 4, 3]));
    assert_eq!(split.combine.inputs()[TensorParamId::new(0)].shape, shape(&[4, 3, 4]));
}

#[test]
fn slices_number_at_least_two() {
    assert_eq!(Slices::new(0), None);
    assert_eq!(Slices::new(1), None);
    assert_eq!(Slices::new(2).map(Slices::get), Some(2));
}

#[test]
fn slices_that_do_not_divide_the_axis_are_refused() {
    let (compute, k) = product_with_epilogue([3, 4, 10]);

    let refused = rfactor(&compute, k, slices(3)).unwrap_err();

    assert_eq!(refused, RfactorError::Indivisible { extent: Dim::new(10), slices: slices(3) });
}

#[test]
fn only_a_reduction_over_the_axis_splits() {
    let (product, k) = product_with_epilogue([3, 4, 12]);
    let spatial = product.iter_vars().iter_enumerated().next().unwrap().0;
    let mut copy = ComputeBuilder::new(ComputeName::new("copy"));
    let x = copy.input(TensorName::new("x"), DType::I64, shape(&[4]));
    let (axis, i) = copy.spatial(VarName::new("i"), Dim::new(4));
    copy.output(TensorName::new("out"), load(x, &[i]));
    let copy = copy.finish().unwrap();

    assert_eq!(rfactor(&product, spatial, slices(2)).unwrap_err(), RfactorError::NoReduction);
    assert_eq!(rfactor(&copy, axis, slices(2)).unwrap_err(), RfactorError::NoReduction);
    assert!(rfactor(&product, k, slices(2)).is_ok());
}

/// `out[i] = sum over k of x[index(i, k)]`, `x` an `i64[length]`.
fn summed_through(
    length: u64,
    index: impl Fn(&mut ComputeBuilder, ExprId, ExprId) -> ExprId,
) -> (ComputeDef, IterVarId) {
    let mut compute = ComputeBuilder::new(ComputeName::new("window"));
    let x = compute.input(TensorName::new("x"), DType::I64, shape(&[length]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(2));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(4));
    let at = index(&mut compute, i, k);
    let body = Box::new(load(x, &[at]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    (compute.finish().unwrap(), k_axis)
}

#[test]
fn an_axis_read_other_than_as_a_whole_index_of_a_dimension_it_spans_is_refused() {
    let window = summed_through(5, |compute, i, k| compute.exprs().add(i, k));
    let offset = summed_through(4, |compute, _, k| {
        let zero = compute.exprs().constant(0);
        compute.exprs().add(k, zero)
    });
    let longer = summed_through(6, |_, _, k| k);
    let spanned = summed_through(4, |_, _, k| k);

    for (compute, k) in [window, offset, longer] {
        assert_eq!(rfactor(&compute, k, slices(2)).unwrap_err(), RfactorError::NotPlain);
    }
    assert!(rfactor(&spanned.0, spanned.1, slices(2)).is_ok());
}

#[test]
fn an_axis_read_as_a_value_is_refused() {
    let mut compute = ComputeBuilder::new(ComputeName::new("positions"));
    let (_, _) = compute.spatial(VarName::new("i"), Dim::new(2));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(4));
    let body = Box::new(TensorExpr::Index { index: k, dtype: DType::I64 });
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    let compute = compute.finish().unwrap();

    assert_eq!(rfactor(&compute, k_axis, slices(2)).unwrap_err(), RfactorError::NotPlain);
}

#[test]
fn a_gather_is_refused() {
    let mut compute = ComputeBuilder::new(ComputeName::new("gathered_sum"));
    let data = compute.input(TensorName::new("data"), DType::I64, shape(&[8]));
    let indices = compute.input(TensorName::new("indices"), DType::I64, shape(&[4]));
    let (_, _) = compute.spatial(VarName::new("i"), Dim::new(2));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(4));
    let row = compute.loaded_index(VarName::new("row"), indices, vec![k]);
    let eight = compute.exprs().constant(8);
    let wrapped = compute.exprs().floor_mod(row, eight);
    let body = Box::new(load(data, &[wrapped]));
    compute.output(
        TensorName::new("out"),
        TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body },
    );
    let compute = compute.finish().unwrap();

    assert_eq!(rfactor(&compute, k_axis, slices(2)).unwrap_err(), RfactorError::Gather);
}

#[test]
fn a_split_float_sum_keeps_each_slice_in_f32() {
    let (compute, k) = product_with_epilogue([1, 1, 4]);
    let inputs = [
        tensor(&[4, 1], half(&[1.0, 2.0, 3.0, 4.0])),
        tensor(&[1], half(&[0.5])),
        tensor(&[1, 4], half(&[0.25, 0.25, 0.25, 0.25])),
    ];

    let split = rfactor(&compute, k, slices(2)).unwrap();

    assert_eq!(floats(&partials(&split, &inputs)), [0.75, 1.75]);
}

/// `out[i] = sum over k of x[k] * y[i] * i`: one load the axis indexes, one
/// it does not, and the other axis read as a value.
#[test]
fn a_reduction_reading_other_axes_too_splits_only_the_loads_its_axis_indexes() {
    let mut compute = ComputeBuilder::new(ComputeName::new("scaled"));
    let x = compute.input(TensorName::new("x"), DType::I64, shape(&[6]));
    let y = compute.input(TensorName::new("y"), DType::I64, shape(&[3]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(3));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(6));
    let position = TensorExpr::Index { index: i, dtype: DType::I64 };
    let scale = binary(BinaryOp::Mul, load(y, &[i]), position);
    let body = Box::new(binary(BinaryOp::Mul, load(x, &[k]), scale));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body };
    compute.output(TensorName::new("out"), sum);
    let compute = compute.finish().unwrap();
    let inputs = [tensor(&[6], vec![1_i64, 2, 3, 4, 5, 6]), tensor(&[3], vec![7_i64, 8, 9])];

    let split = rfactor(&compute, k_axis, slices(3)).unwrap();

    assert_eq!(split.partial.inputs()[TensorParamId::new(0)].shape, shape(&[3, 2]));
    assert_eq!(split.partial.inputs()[TensorParamId::new(1)].shape, shape(&[3]));
    assert_eq!(ints(&run_split(&split, &inputs)[0]), [0, 8 * 21, 9 * 21 * 2]);
}
