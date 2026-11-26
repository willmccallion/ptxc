//! A producer inlined into a consumer: the definition `inline` builds
//! computes what the two compute in turn, and the inlinings it refuses.

use crate::compute_build::{binary, load, shape};
use crate::interp::{ints, run_default, tensor};
use ptxc_core::{DType, Dim, HostTensor, Idx, IndexVec};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, InlineError, InlinedInput, ReduceOp,
    Scalar, TensorExpr, TensorName, TensorParamId, VarName, inline,
};
use std::sync::Arc;

fn finish(compute: ComputeBuilder) -> ComputeDef {
    compute.finish().unwrap()
}

fn int(value: i64) -> TensorExpr {
    TensorExpr::Const(Scalar::int(DType::I32, value).unwrap())
}

/// `out[i, j] = sum over k of a[i, k] * b[k, j]` on `i32`, exact in any
/// order.
fn product(rows: u64, depth: u64, columns: u64) -> ComputeDef {
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let lhs = compute.input(TensorName::new("a"), DType::I32, shape(&[rows, depth]));
    let rhs = compute.input(TensorName::new("b"), DType::I32, shape(&[depth, columns]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let body = binary(BinaryOp::Mul, load(lhs, &[i, k]), load(rhs, &[k, j]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    finish(compute)
}

/// A consumer over `[rows, columns]` reading `p`, a `[rows, columns]` input,
/// first, then `extra` inputs of the shapes given, as `expr` makes the
/// output from the inputs and the row and column variables.
fn consumer(
    dims: [u64; 2],
    extra: &[&[u64]],
    expr: impl FnOnce(&mut ComputeBuilder, &[TensorParamId], [ExprId; 2]) -> TensorExpr,
) -> ComputeDef {
    let mut compute = ComputeBuilder::new(ComputeName::new("consumer"));
    let mut inputs = vec![compute.input(TensorName::new("p"), DType::I32, shape(&dims))];
    for (n, dims) in extra.iter().enumerate() {
        inputs.push(compute.input(TensorName::new(format!("e{n}")), DType::I32, shape(dims)));
    }
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(dims[0]));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(dims[1]));
    let out = expr(&mut compute, &inputs, [i, j]);
    compute.output(TensorName::new("out"), out);
    finish(compute)
}

fn run(compute: &ComputeDef, inputs: &[HostTensor]) -> Vec<i64> {
    ints(&run_default(&Arc::new(compute.clone()), inputs).unwrap()[0])
}

/// What `consumer` computes reading `producer`'s output, run in turn.
fn in_turn(
    producer: &ComputeDef,
    producer_inputs: &[HostTensor],
    consumer: &ComputeDef,
    rest: &[HostTensor],
) -> Vec<i64> {
    let produced = run_default(&Arc::new(producer.clone()), producer_inputs).unwrap().remove(0);
    let mut inputs = vec![produced];
    inputs.extend_from_slice(rest);
    run(consumer, &inputs)
}

/// The inputs `inlined` reads, in its order, from the consumer's `rest`
/// (its inputs after the inlined first) and the producer's.
fn inlined_inputs(
    inputs: &IndexVec<TensorParamId, InlinedInput>,
    rest: &[HostTensor],
    producer_inputs: &[HostTensor],
) -> Vec<HostTensor> {
    inputs
        .iter()
        .map(|input| match *input {
            InlinedInput::Consumer(id) => rest[id.index() - 1].clone(),
            InlinedInput::Producer(id) => producer_inputs[id.index()].clone(),
        })
        .collect()
}

fn values(count: i32, offset: i32) -> Vec<i32> {
    (0..count).map(|n| (n * 7 + offset) % 11 - 5).collect()
}

#[test]
fn an_elementwise_producer_is_computed_again_at_each_read() {
    let mut double = ComputeBuilder::new(ComputeName::new("double"));
    let x = double.input(TensorName::new("x"), DType::I32, shape(&[2, 6]));
    let (_, i) = double.spatial(VarName::new("i"), Dim::new(2));
    let (_, j) = double.spatial(VarName::new("j"), Dim::new(6));
    double.output(TensorName::new("out"), binary(BinaryOp::Mul, load(x, &[i, j]), int(2)));
    let double = finish(double);
    let reversed = consumer([2, 6], &[], |compute, inputs, [i, j]| {
        let last = compute.exprs().constant(5);
        let minus_one = compute.exprs().constant(-1);
        let back = compute.exprs().mul(minus_one, j);
        let mirrored = compute.exprs().add(last, back);
        binary(BinaryOp::Add, load(inputs[0], &[i, j]), load(inputs[0], &[i, mirrored]))
    });
    let x = [tensor(&[2, 6], values(12, 0))];

    let inlined = inline(&reversed, &[TensorParamId::new(0)], &double).unwrap();

    assert_eq!(
        inlined.inputs.iter().copied().collect::<Vec<_>>(),
        [InlinedInput::Producer(TensorParamId::new(0))]
    );
    let fused = run(&inlined.compute, &inlined_inputs(&inlined.inputs, &[], &x));
    assert_eq!(fused, in_turn(&double, &x, &reversed, &[]));
}

#[test]
fn a_transpose_inlines_into_a_products_reads() {
    let mut transpose = ComputeBuilder::new(ComputeName::new("transpose"));
    let weight = transpose.input(TensorName::new("w"), DType::I32, shape(&[5, 3]));
    let (_, j) = transpose.spatial(VarName::new("j"), Dim::new(3));
    let (_, k) = transpose.spatial(VarName::new("k"), Dim::new(5));
    transpose.output(TensorName::new("out"), load(weight, &[k, j]));
    let transpose = finish(transpose);
    let mut product = ComputeBuilder::new(ComputeName::new("product_nt"));
    let transposed = product.input(TensorName::new("t"), DType::I32, shape(&[3, 5]));
    let lhs = product.input(TensorName::new("a"), DType::I32, shape(&[4, 5]));
    let (_, i) = product.spatial(VarName::new("i"), Dim::new(4));
    let (_, j) = product.spatial(VarName::new("j"), Dim::new(3));
    let (k_axis, k) = product.reduce_axis(VarName::new("k"), Dim::new(5));
    let body = binary(BinaryOp::Mul, load(lhs, &[i, k]), load(transposed, &[j, k]));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body: Box::new(body) };
    product.output(TensorName::new("out"), sum);
    let product = finish(product);
    let (weight, lhs) = (tensor(&[5, 3], values(15, 1)), tensor(&[4, 5], values(20, 2)));

    let inlined = inline(&product, &[TensorParamId::new(0)], &transpose).unwrap();

    let order = [
        InlinedInput::Consumer(TensorParamId::new(1)),
        InlinedInput::Producer(TensorParamId::new(0)),
    ];
    assert_eq!(inlined.inputs.iter().copied().collect::<Vec<_>>(), order);
    let fused = run(&inlined.compute, &[lhs.clone(), weight.clone()]);
    let produced = run_default(&Arc::new(transpose), &[weight]).unwrap().remove(0);
    assert_eq!(fused, run(&product, &[produced, lhs]));
}

#[test]
fn a_product_becomes_the_reduction_of_its_epilogue() {
    let product = product(3, 4, 5);
    let relu = consumer([3, 5], &[&[5]], |_, inputs, [i, j]| {
        let biased = binary(BinaryOp::Add, load(inputs[0], &[i, j]), load(inputs[1], &[j]));
        binary(BinaryOp::Max, biased, int(0))
    });
    let operands = [tensor(&[3, 4], values(12, 3)), tensor(&[4, 5], values(20, 4))];
    let bias = [tensor(&[5], values(5, 5))];

    let inlined = inline(&relu, &[TensorParamId::new(0)], &product).unwrap();

    let fused = run(&inlined.compute, &inlined_inputs(&inlined.inputs, &bias, &operands));
    assert_eq!(fused, in_turn(&product, &operands, &relu, &bias));
}

#[test]
fn an_epilogue_may_read_the_product_more_than_once() {
    let product = product(3, 4, 5);
    let squared = consumer([3, 5], &[], |_, inputs, [i, j]| {
        binary(BinaryOp::Mul, load(inputs[0], &[i, j]), load(inputs[0], &[i, j]))
    });
    let operands = [tensor(&[3, 4], values(12, 6)), tensor(&[4, 5], values(20, 7))];

    let inlined = inline(&squared, &[TensorParamId::new(0)], &product).unwrap();

    let fused = run(&inlined.compute, &inlined_inputs(&inlined.inputs, &[], &operands));
    assert_eq!(fused, in_turn(&product, &operands, &squared, &[]));
}

#[test]
fn a_producer_held_by_two_inputs_inlines_into_both_at_once() {
    let product = product(3, 4, 5);
    let squared = consumer([3, 5], &[&[3, 5]], |_, inputs, [i, j]| {
        binary(BinaryOp::Mul, load(inputs[0], &[i, j]), load(inputs[1], &[i, j]))
    });
    let operands = [tensor(&[3, 4], values(12, 8)), tensor(&[4, 5], values(20, 9))];
    let both = [TensorParamId::new(0), TensorParamId::new(1)];

    let inlined = inline(&squared, &both, &product).unwrap();

    let producer_only = |input: &InlinedInput| matches!(input, InlinedInput::Producer(_));
    assert!(inlined.inputs.iter().all(producer_only));
    let fused = run(&inlined.compute, &inlined_inputs(&inlined.inputs, &[], &operands));
    let produced = run_default(&Arc::new(product), &operands).unwrap().remove(0);
    assert_eq!(fused, run(&squared, &[produced.clone(), produced]));
}

#[test]
fn a_broadcast_read_of_an_axis_of_one_element_is_the_consumers_own() {
    let product = product(1, 4, 5);
    let zeroed = consumer([1, 5], &[], |compute, inputs, [_, j]| {
        let zero = compute.exprs().constant(0);
        binary(BinaryOp::Add, load(inputs[0], &[zero, j]), int(1))
    });
    let operands = [tensor(&[1, 4], values(4, 10)), tensor(&[4, 5], values(20, 11))];

    let inlined = inline(&zeroed, &[TensorParamId::new(0)], &product).unwrap();

    let fused = run(&inlined.compute, &inlined_inputs(&inlined.inputs, &[], &operands));
    assert_eq!(fused, in_turn(&product, &operands, &zeroed, &[]));
}

#[test]
fn a_constant_read_of_an_axis_of_several_elements_is_refused() {
    let product = product(3, 2, 4);
    let first_row = consumer([3, 4], &[], |compute, inputs, [_, j]| {
        let zero = compute.exprs().constant(0);
        load(inputs[0], &[zero, j])
    });

    let refused = inline(&first_row, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::NotAnEpilogue);
}

#[test]
fn a_product_read_away_from_the_consumers_own_element_is_refused() {
    let product = product(4, 2, 4);
    let transposed = consumer([4, 4], &[], |_, inputs, [i, j]| load(inputs[0], &[j, i]));

    let refused = inline(&transposed, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::NotAnEpilogue);
}

#[test]
fn a_product_into_a_consumer_that_reduces_is_refused() {
    let product = product(3, 2, 4);
    let mut row_sum = ComputeBuilder::new(ComputeName::new("row_sum"));
    let p = row_sum.input(TensorName::new("p"), DType::I32, shape(&[3, 4]));
    let (_, i) = row_sum.spatial(VarName::new("i"), Dim::new(3));
    let (k_axis, k) = row_sum.reduce_axis(VarName::new("k"), Dim::new(4));
    let sum = TensorExpr::Reduce {
        op: ReduceOp::Sum,
        axes: vec![k_axis],
        body: Box::new(load(p, &[i, k])),
    };
    row_sum.output(TensorName::new("out"), sum);
    let row_sum = finish(row_sum);

    let refused = inline(&row_sum, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::NotAnEpilogue);
}

#[test]
fn a_padded_read_of_the_producer_is_refused() {
    let product = product(3, 2, 4);
    let padded = consumer([3, 4], &[], |_, inputs, [i, j]| TensorExpr::Load {
        tensor: inputs[0],
        index: vec![i, j],
        padding: Some(Scalar::zero(DType::I32)),
    });

    let refused = inline(&padded, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::Padded);
}

#[test]
fn a_producer_of_several_outputs_is_refused() {
    let mut two = ComputeBuilder::new(ComputeName::new("two"));
    let x = two.input(TensorName::new("x"), DType::I32, shape(&[3, 4]));
    let (_, i) = two.spatial(VarName::new("i"), Dim::new(3));
    let (_, j) = two.spatial(VarName::new("j"), Dim::new(4));
    two.output(TensorName::new("first"), load(x, &[i, j]));
    two.output(TensorName::new("second"), load(x, &[i, j]));
    let two = finish(two);
    let copy = consumer([3, 4], &[], |_, inputs, [i, j]| load(inputs[0], &[i, j]));

    let refused = inline(&copy, &[TensorParamId::new(0)], &two).unwrap_err();

    assert_eq!(refused, InlineError::SeveralOutputs);
}

#[test]
fn an_input_that_is_not_the_producers_output_is_refused() {
    let product = product(3, 2, 4);
    let wider = consumer([3, 5], &[], |_, inputs, [i, j]| load(inputs[0], &[i, j]));

    let refused = inline(&wider, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::Mismatch);
}

#[test]
fn a_consumer_that_gathers_is_refused() {
    let product = product(3, 2, 4);
    let gather = consumer([3, 4], &[&[3]], |compute, inputs, [i, j]| {
        let row = compute.loaded_index(VarName::new("row"), inputs[1], vec![i]);
        load(inputs[0], &[row, j])
    });

    let refused = inline(&gather, &[TensorParamId::new(0)], &product).unwrap_err();

    assert_eq!(refused, InlineError::Gather);
}
