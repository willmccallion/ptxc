//! The side evaluator against hand-computed results, with output types from
//! the graph's own type rules.

use super::evaluate;
use ptxc_core::float::Overflow;
use ptxc_core::{Axis, DType, Dim, HostTensor, Shape, TensorData};
use ptxc_graph::{
    BinaryOp, CastAttrs, GraphBuilder, KeepDims, ModKind, Op, ReduceAttrs, ReduceOp, RepeatCount,
    SliceAttrs, SliceAxis, SliceStep, TensorType, TileAttrs, TransposeAttrs, UnaryOp, VariadicOp,
};

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(shape(dims), data.into()).unwrap()
}

fn i64s(dims: &[u64], values: &[i64]) -> HostTensor {
    tensor(dims, values.to_vec())
}

fn bools(dims: &[u64], values: &[bool]) -> HostTensor {
    tensor(dims, values.to_vec())
}

/// The op's single output, evaluated, with the type its rule infers.
fn eval(op: &Op, operands: &[HostTensor]) -> Option<HostTensor> {
    let mut builder = GraphBuilder::new();
    let ids = operands.iter().map(|operand| builder.add_constant(operand.clone())).collect();
    let outputs = builder.add_node(op.clone(), ids).unwrap();
    let types: Vec<TensorType> =
        outputs.iter().map(|&output| builder.value_type(output).clone()).collect();
    let operands: Vec<&HostTensor> = operands.iter().collect();
    let mut values = evaluate(op, &operands, &types)?;
    assert_eq!(values.len(), 1);
    values.pop()
}

fn assert_i64(tensor: &HostTensor, dims: &[u64], expected: &[i64]) {
    assert_eq!(tensor.shape(), &shape(dims));
    assert!(matches!(tensor.data(), TensorData::I64(values) if values == expected), "{tensor:?}");
}

fn assert_bool(tensor: &HostTensor, dims: &[u64], expected: &[bool]) {
    assert_eq!(tensor.shape(), &shape(dims));
    assert!(matches!(tensor.data(), TensorData::Bool(values) if values == expected), "{tensor:?}");
}

fn binary(op: BinaryOp, a: &HostTensor, b: &HostTensor) -> HostTensor {
    eval(&Op::Binary(op), &[a.clone(), b.clone()]).unwrap()
}

#[test]
fn arithmetic_broadcasts_and_divides_toward_zero() {
    let a = i64s(&[3, 1], &[10, -7, 6]);
    let b = i64s(&[3], &[3, -3, 2]);
    let dims = [3, 3];

    assert_i64(&binary(BinaryOp::Add, &a, &b), &dims, &[13, 7, 12, -4, -10, -5, 9, 3, 8]);
    assert_i64(&binary(BinaryOp::Sub, &a, &b), &dims, &[7, 13, 8, -10, -4, -9, 3, 9, 4]);
    assert_i64(&binary(BinaryOp::Mul, &a, &b), &dims, &[30, -30, 20, -21, 21, -14, 18, -18, 12]);
    assert_i64(&binary(BinaryOp::Div, &a, &b), &dims, &[3, -3, 5, -2, 2, -3, 2, -2, 3]);
}

#[test]
fn fmod_takes_the_dividend_sign_and_integer_mod_the_divisor_sign() {
    let a = i64s(&[3, 1], &[10, -7, 6]);
    let b = i64s(&[3], &[3, -3, 2]);
    let dims = [3, 3];

    let fmod = binary(BinaryOp::Mod(ModKind::Fmod), &a, &b);
    let integer = binary(BinaryOp::Mod(ModKind::Integer), &a, &b);

    assert_i64(&fmod, &dims, &[1, 1, 0, -1, -1, -1, 0, 0, 0]);
    assert_i64(&integer, &dims, &[1, -2, 0, 2, -1, 1, 0, 0, 0]);
}

#[test]
fn comparisons_broadcast_a_scalar_and_give_bools() {
    let a = i64s(&[3], &[1, 2, 3]);
    let two = i64s(&[], &[2]);

    assert_bool(&binary(BinaryOp::Equal, &a, &two), &[3], &[false, true, false]);
    assert_bool(&binary(BinaryOp::Less, &a, &two), &[3], &[true, false, false]);
    assert_bool(&binary(BinaryOp::LessOrEqual, &a, &two), &[3], &[true, true, false]);
    assert_bool(&binary(BinaryOp::Greater, &a, &two), &[3], &[false, false, true]);
    assert_bool(&binary(BinaryOp::GreaterOrEqual, &a, &two), &[3], &[false, true, true]);
}

#[test]
fn logic_ops_follow_their_truth_tables() {
    let a = bools(&[4], &[false, false, true, true]);
    let b = bools(&[4], &[false, true, false, true]);

    assert_bool(&binary(BinaryOp::And, &a, &b), &[4], &[false, false, false, true]);
    assert_bool(&binary(BinaryOp::Or, &a, &b), &[4], &[false, true, true, true]);
    assert_bool(&binary(BinaryOp::Xor, &a, &b), &[4], &[false, true, true, false]);
}

#[test]
fn overflow_division_by_zero_and_pow_are_left_unknown() {
    let max = i64s(&[1], &[i64::MAX]);
    let zero = i64s(&[1], &[0]);
    let one = i64s(&[1], &[1]);

    assert!(eval(&Op::Binary(BinaryOp::Add), &[max.clone(), one.clone()]).is_none());
    assert!(eval(&Op::Binary(BinaryOp::Div), &[one.clone(), zero]).is_none());
    assert!(eval(&Op::Binary(BinaryOp::Pow), &[max, one]).is_none());
}

#[test]
fn unary_ops_negate_take_magnitudes_signs_and_complements() {
    let a = i64s(&[3], &[-2, 0, 5]);

    assert_i64(
        &eval(&Op::Unary(UnaryOp::Neg), std::slice::from_ref(&a)).unwrap(),
        &[3],
        &[2, 0, -5],
    );
    assert_i64(
        &eval(&Op::Unary(UnaryOp::Abs), std::slice::from_ref(&a)).unwrap(),
        &[3],
        &[2, 0, 5],
    );
    assert_i64(&eval(&Op::Unary(UnaryOp::Sign), &[a]).unwrap(), &[3], &[-1, 0, 1]);
    let not = eval(&Op::Unary(UnaryOp::Not), &[bools(&[2], &[true, false])]).unwrap();
    assert_bool(&not, &[2], &[false, true]);
    assert!(eval(&Op::Unary(UnaryOp::Neg), &[i64s(&[1], &[i64::MIN])]).is_none());
}

#[test]
fn max_and_min_broadcast_every_operand() {
    let operands = [i64s(&[2], &[1, 5]), i64s(&[2, 1], &[4, 2]), i64s(&[], &[3])];

    let max = eval(&Op::Variadic(VariadicOp::Max), &operands).unwrap();
    let min = eval(&Op::Variadic(VariadicOp::Min), &operands).unwrap();

    assert_i64(&max, &[2, 2], &[4, 5, 3, 5]);
    assert_i64(&min, &[2, 2], &[1, 3, 1, 2]);
}

#[test]
fn where_picks_by_a_broadcast_condition() {
    let condition = bools(&[2], &[true, false]);

    let picked = eval(&Op::Where, &[condition, i64s(&[2, 1], &[1, 2]), i64s(&[], &[9])]).unwrap();

    assert_i64(&picked, &[2, 2], &[1, 9, 2, 9]);
}

fn cast(to: DType, operand: HostTensor) -> Option<HostTensor> {
    eval(&Op::Cast(CastAttrs::new(to, Overflow::Saturate)), &[operand])
}

#[test]
fn casts_between_integer_types_and_to_bool() {
    let narrow = cast(DType::I32, i64s(&[3], &[1, 0, -2])).unwrap();
    let wide = cast(DType::I64, tensor(&[2], vec![7_i32, -1])).unwrap();
    let flags = cast(DType::Bool, i64s(&[3], &[1, 0, -2])).unwrap();

    assert!(matches!(narrow.data(), TensorData::I32(values) if *values == [1, 0, -2]));
    assert_i64(&wide, &[2], &[7, -1]);
    assert_bool(&flags, &[3], &[true, false, true]);
}

#[test]
fn a_cast_that_does_not_fit_or_leaves_the_integers_is_unknown() {
    assert!(cast(DType::I32, i64s(&[1], &[3_000_000_000])).is_none());
    assert!(cast(DType::F32, i64s(&[1], &[1])).is_none());
    assert!(cast(DType::F16, tensor(&[1], vec![1.0_f32])).is_none());
}

#[test]
fn reshapes_and_same_type_casts_carry_any_element_type() {
    let floats = tensor(&[2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);

    let reshaped = eval(&Op::Reshape(shape(&[4])), std::slice::from_ref(&floats)).unwrap();
    let cast = cast(DType::F32, floats).unwrap();

    assert_eq!(reshaped.shape(), &shape(&[4]));
    assert!(matches!(reshaped.data(), TensorData::F32(values) if *values == [1.0, 2.0, 3.0, 4.0]));
    assert_eq!(cast.shape(), &shape(&[2, 2]));
}

#[test]
fn tensors_past_the_limit_are_left_to_the_graph() {
    let at_limit = tensor(&[4096], vec![0.0_f32; 4096]);
    let past_limit = tensor(&[4097], vec![0.0_f32; 4097]);
    let sum = Op::Reduce(ReduceAttrs {
        op: ReduceOp::Sum,
        axes: vec![Axis::new(0)],
        keep_dims: KeepDims::Drop,
    });

    assert!(eval(&Op::Reshape(shape(&[64, 64])), &[at_limit]).is_some());
    assert!(eval(&Op::Reshape(shape(&[1, 4097])), &[past_limit]).is_none());
    assert!(eval(&Op::Expand(shape(&[4096])), &[i64s(&[1], &[1])]).is_some());
    assert!(eval(&Op::Expand(shape(&[4097])), &[i64s(&[1], &[1])]).is_none());
    assert!(eval(&sum, &[i64s(&[4096], &vec![1; 4096])]).is_some());
    assert!(eval(&sum, &[i64s(&[4097], &vec![1; 4097])]).is_none());
}

#[test]
fn expand_and_tile_repeat_elements() {
    let expanded = eval(&Op::Expand(shape(&[2, 3])), &[i64s(&[2, 1], &[1, 2])]).unwrap();
    let repeats = vec![RepeatCount::new(2), RepeatCount::new(2)];
    let tiled = eval(&Op::Tile(TileAttrs { repeats }), &[i64s(&[1, 2], &[1, 2])]).unwrap();

    assert_i64(&expanded, &[2, 3], &[1, 1, 1, 2, 2, 2]);
    assert_i64(&tiled, &[2, 4], &[1, 2, 1, 2, 1, 2, 1, 2]);
}

#[test]
fn concat_joins_blocks_along_an_inner_axis() {
    let joined =
        eval(&Op::Concat(Axis::new(1)), &[i64s(&[2, 1], &[1, 2]), i64s(&[2, 2], &[3, 4, 5, 6])]);

    assert_i64(&joined.unwrap(), &[2, 3], &[1, 3, 4, 2, 5, 6]);
}

#[test]
fn gather_wraps_negative_indices_and_rejects_out_of_range_ones() {
    let data = i64s(&[2, 3], &[1, 2, 3, 4, 5, 6]);
    let gather = Op::Gather(Axis::new(1));

    let gathered = eval(&gather, &[data.clone(), i64s(&[2], &[-1, 0])]).unwrap();

    assert_i64(&gathered, &[2, 2], &[3, 1, 6, 4]);
    assert!(eval(&gather, &[data.clone(), i64s(&[1], &[3])]).is_none());
    assert!(eval(&gather, &[data, i64s(&[1], &[-4])]).is_none());
}

#[test]
fn transpose_moves_input_axis_perm_i_to_output_axis_i() {
    let values: Vec<i64> = (0..12).collect();
    let permutation = vec![Axis::new(2), Axis::new(0), Axis::new(1)];

    let moved = eval(&Op::Transpose(TransposeAttrs { permutation }), &[i64s(&[2, 3, 2], &values)]);

    assert_i64(&moved.unwrap(), &[2, 2, 3], &[0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]);
}

#[test]
fn slices_step_from_their_start_in_either_direction() {
    let backwards = SliceAxis {
        axis: Axis::new(0),
        start: 4,
        step: SliceStep::new(-2).unwrap(),
        length: Dim::new(3),
    };
    let inner =
        SliceAxis { axis: Axis::new(1), start: 1, step: SliceStep::one(), length: Dim::new(2) };

    let reversed =
        eval(&Op::Slice(SliceAttrs { axes: vec![backwards] }), &[i64s(&[5], &[0, 1, 2, 3, 4])]);
    let cropped =
        eval(&Op::Slice(SliceAttrs { axes: vec![inner] }), &[i64s(&[2, 3], &[1, 2, 3, 4, 5, 6])]);

    assert_i64(&reversed.unwrap(), &[3], &[4, 2, 0]);
    assert_i64(&cropped.unwrap(), &[2, 2], &[2, 3, 5, 6]);
}

fn reduce(
    op: ReduceOp,
    axis: usize,
    keep_dims: KeepDims,
    input: &HostTensor,
) -> Option<HostTensor> {
    eval(
        &Op::Reduce(ReduceAttrs { op, axes: vec![Axis::new(axis)], keep_dims }),
        std::slice::from_ref(input),
    )
}

#[test]
fn reductions_combine_along_their_axes_keeping_or_dropping_them() {
    let input = i64s(&[2, 3], &[1, 2, 3, 4, 5, 6]);

    assert_i64(&reduce(ReduceOp::Sum, 1, KeepDims::Keep, &input).unwrap(), &[2, 1], &[6, 15]);
    assert_i64(&reduce(ReduceOp::Prod, 1, KeepDims::Drop, &input).unwrap(), &[2], &[6, 120]);
    assert_i64(&reduce(ReduceOp::Max, 0, KeepDims::Keep, &input).unwrap(), &[1, 3], &[4, 5, 6]);
    assert_i64(&reduce(ReduceOp::Min, 0, KeepDims::Drop, &input).unwrap(), &[3], &[1, 2, 3]);
}

#[test]
fn empty_reductions_give_the_identity_or_nothing() {
    let empty = i64s(&[2, 0], &[]);

    assert_i64(&reduce(ReduceOp::Sum, 1, KeepDims::Drop, &empty).unwrap(), &[2], &[0, 0]);
    assert_i64(&reduce(ReduceOp::Prod, 1, KeepDims::Drop, &empty).unwrap(), &[2], &[1, 1]);
    assert!(reduce(ReduceOp::Max, 1, KeepDims::Drop, &empty).is_none());
}

#[test]
fn a_lower_rank_operand_broadcasts_over_the_leading_axes() {
    let a = i64s(&[2, 1, 1], &[100, 200]);
    let b = i64s(&[2, 3], &[1, 2, 3, 4, 5, 6]);

    let sum = binary(BinaryOp::Add, &a, &b);

    assert_i64(&sum, &[2, 2, 3], &[101, 102, 103, 104, 105, 106, 201, 202, 203, 204, 205, 206]);
}

#[test]
fn gather_on_an_outer_axis_takes_whole_rows() {
    let data = i64s(&[3, 2], &[1, 2, 3, 4, 5, 6]);

    let rows = eval(&Op::Gather(Axis::new(0)), &[data, i64s(&[2], &[2, 0])]).unwrap();

    assert_i64(&rows, &[2, 2], &[5, 6, 1, 2]);
}

#[test]
fn reducing_an_inner_axis_keeps_the_outer_strides() {
    let input = i64s(&[2, 3, 2], &(0..12).collect::<Vec<_>>());

    let sums = reduce(ReduceOp::Sum, 2, KeepDims::Drop, &input).unwrap();

    assert_i64(&sums, &[2, 3], &[1, 5, 9, 13, 17, 21]);
}
