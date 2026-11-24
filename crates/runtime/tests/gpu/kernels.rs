//! Hand-built kernels covering what codegen selects: arithmetic and every
//! math program on awkward values, casts, integer edge cases, reductions
//! through shared memory, and unrolled loops.

use crate::device::{check_against_interpreter, match_interpreter};
use ptxc_core::float::{F16, Overflow};
use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeName, Primitive, ReduceOp, Schedule,
    SplitFactor, TensorExpr, TensorName, ThreadAxis, TieBreak, UnaryOp, VarName,
};
use ptxc_target::{SmVersion, preset};
use std::sync::Arc;

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// `y[i] = expr(x0[i], x1[i], ...)` over `length` elements.
fn elementwise(
    inputs: &[DType],
    length: u64,
    expr: impl FnOnce(&[TensorExpr]) -> TensorExpr,
) -> ComputeDef {
    let mut compute = ComputeBuilder::new(ComputeName::new("elementwise"));
    let tensors: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(n, &dtype)| compute.input(TensorName::new(format!("x{n}")), dtype, shape(&[length])))
        .collect();
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(length));
    let loads: Vec<TensorExpr> =
        tensors.into_iter().map(|tensor| TensorExpr::Load { tensor, index: vec![i] }).collect();
    compute.output(TensorName::new("y"), expr(&loads));
    compute.finish().unwrap()
}

fn vector(data: impl Into<TensorData>) -> HostTensor {
    HostTensor::vector(data.into())
}

/// Special values, then a spread of bit patterns over every exponent.
fn awkward_floats(count: u32) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        88.722_84,
        -103.972_09,
        std::f32::consts::PI,
        1e30,
        -3.575_242e35,
    ];
    let spread = (0..count).map(|n| f32::from_bits(n.wrapping_mul(0x9E37_79B9)));
    specials.into_iter().chain(spread).collect()
}

fn unary(op: UnaryOp, x: &TensorExpr) -> TensorExpr {
    TensorExpr::Unary(op, Box::new(x.clone()))
}

fn binary(op: BinaryOp, x: &TensorExpr, y: &TensorExpr) -> TensorExpr {
    TensorExpr::Binary(op, Box::new(x.clone()), Box::new(y.clone()))
}

fn cast(to: DType, overflow: Overflow, x: &TensorExpr) -> TensorExpr {
    TensorExpr::Cast { to, overflow, value: Box::new(x.clone()) }
}

#[test]
fn an_f32_add_matches_over_a_partial_last_block() {
    let a: Vec<f32> = (0..1000_u16).map(f32::from).collect();
    let b: Vec<f32> = a.iter().map(|x| 0.5 - x * 2.0).collect();
    let compute =
        elementwise(&[DType::F32, DType::F32], 1000, |x| binary(BinaryOp::Add, &x[0], &x[1]));

    check_against_interpreter(compute, &[vector(a), vector(b)]);
}

#[test]
fn every_unary_math_program_matches_bit_for_bit() {
    let values = awkward_floats(4096);
    let length = u64::try_from(values.len()).unwrap();
    let ops = [
        UnaryOp::Exp,
        UnaryOp::Expm1,
        UnaryOp::Log,
        UnaryOp::Log1p,
        UnaryOp::Tanh,
        UnaryOp::Erf,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Sqrt,
        UnaryOp::Reciprocal,
        UnaryOp::Round,
        UnaryOp::Sign,
    ];
    for op in ops {
        let compute = elementwise(&[DType::F32], length, |x| unary(op, &x[0]));
        check_against_interpreter(compute, &[vector(values.clone())]);
    }
}

#[test]
fn pow_and_remainder_match_bit_for_bit() {
    let x = awkward_floats(2048);
    let y: Vec<f32> = x.iter().rev().map(|v| v * 0.001 - 3.0).collect();
    let length = u64::try_from(x.len()).unwrap();
    for op in [BinaryOp::Pow, BinaryOp::TruncatedRem] {
        let compute = elementwise(&[DType::F32, DType::F32], length, |v| binary(op, &v[0], &v[1]));
        check_against_interpreter(compute, &[vector(x.clone()), vector(y.clone())]);
    }
}

#[test]
fn casts_from_f32_saturate_round_and_extend_as_the_interpreter_does() {
    let values = awkward_floats(1024);
    let length = u64::try_from(values.len()).unwrap();
    let targets =
        [DType::I8, DType::U8, DType::I32, DType::I64, DType::Bool, DType::F16, DType::Bf16];
    for to in targets {
        let compute = elementwise(&[DType::F32], length, |x| cast(to, Overflow::NonFinite, &x[0]));
        check_against_interpreter(compute, &[vector(values.clone())]);
    }
    for to in [DType::F8E4M3Fn, DType::F8E5M2] {
        for overflow in [Overflow::Saturate, Overflow::NonFinite] {
            let compute = elementwise(&[DType::F32], length, |x| cast(to, overflow, &x[0]));
            check_against_interpreter(compute, &[vector(values.clone())]);
        }
    }
}

/// An i8 result held in a 32-bit register feeds an add, so its sign
/// extension shows.
#[test]
fn a_cast_to_i8_feeds_arithmetic_sign_extended() {
    let values = vec![-200.0_f32, -128.5, -1.5, 0.0, 127.9, 300.0];
    let length = u64::try_from(values.len()).unwrap();
    let compute = elementwise(&[DType::F32], length, |x| {
        let narrow = cast(DType::I8, Overflow::NonFinite, &x[0]);
        cast(DType::I32, Overflow::NonFinite, &narrow)
    });

    check_against_interpreter(compute, &[vector(values)]);
}

#[test]
fn integer_division_overflow_and_pow_match() {
    let a = vec![i32::MIN, i32::MIN, 7, -7, 2, -1, 0, 3];
    let b = vec![-1, 1, -2, 2, 31, -3, -1, -2];
    let compute = |op| elementwise(&[DType::I32, DType::I32], 8, |x| binary(op, &x[0], &x[1]));
    for op in [BinaryOp::Div, BinaryOp::TruncatedRem, BinaryOp::FlooredRem, BinaryOp::Pow] {
        check_against_interpreter(compute(op), &[vector(a.clone()), vector(b.clone())]);
    }
    let wide = vec![3_i64, -2, i64::MIN, 0, 1];
    let powers = vec![40_i64, 63, 2, -5, -9];
    let compute =
        elementwise(&[DType::I64, DType::I64], 5, |x| binary(BinaryOp::Pow, &x[0], &x[1]));
    check_against_interpreter(compute, &[vector(wide), vector(powers)]);
}

#[test]
fn an_integer_cast_to_bf16_rounds_once() {
    let values = vec![(1_i64 << 24) + (1 << 15) + 1, -(1 << 40) - 3, i64::MAX, 7];
    let compute = elementwise(&[DType::I64], 4, |x| cast(DType::Bf16, Overflow::NonFinite, &x[0]));

    check_against_interpreter(compute, &[vector(values)]);
}

#[test]
fn an_f16_add_rounds_to_half_precision() {
    let a: Vec<F16> = (0..300_u16).map(|n| F16::from_f32(f32::from(n) * 0.37)).collect();
    let b: Vec<F16> =
        (0..300_u16).map(|n| F16::from_f32(f32::from(n).mul_add(-3.1, 1000.0))).collect();
    let compute =
        elementwise(&[DType::F16, DType::F16], 300, |x| binary(BinaryOp::Add, &x[0], &x[1]));

    check_against_interpreter(compute, &[vector(a), vector(b)]);
}

/// A row reduction of `dtype` values: a block per row, threads combined
/// through shared memory.
fn rows(
    dtype: DType,
    reduce: impl FnOnce(ptxc_loop::IterVarId, TensorExpr) -> TensorExpr,
) -> ComputeDef {
    let mut compute = ComputeBuilder::new(ComputeName::new("rows"));
    let x = compute.input(TensorName::new("x"), dtype, shape(&[3, 300]));
    let (_, row) = compute.spatial(VarName::new("row"), Dim::new(3));
    let (axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(300));
    compute.output(
        TensorName::new("y"),
        reduce(axis, TensorExpr::Load { tensor: x, index: vec![row, k] }),
    );
    compute.finish().unwrap()
}

#[test]
fn a_row_sum_and_argmax_reduce_through_shared_memory() {
    let values: Vec<f32> = (0..900_u16).map(|n| (f32::from(n) * 0.731).sin()).collect();
    let sum = rows(DType::F32, |axis, body| TensorExpr::Reduce {
        op: ReduceOp::Sum,
        axes: vec![axis],
        body: Box::new(body),
    });
    check_against_interpreter(
        sum,
        &[HostTensor::new(shape(&[3, 300]), values.clone().into()).unwrap()],
    );

    let halves: Vec<F16> = values.iter().map(|&v| F16::from_f32(v)).collect();
    let argmax = rows(DType::F16, |axis, body| TensorExpr::ArgReduce {
        kind: ArgKind::Max,
        tie: TieBreak::First,
        axis,
        body: Box::new(body),
    });
    check_against_interpreter(argmax, &[HostTensor::new(shape(&[3, 300]), halves.into()).unwrap()]);
}

/// `out[i, j] = sum over k of a[i, k] * b[k, j]`, its rows and columns
/// split by 2 and its reduction by 4, with a thread's 2 by 2 tile and each
/// step's depth unrolled; the uneven extents leave guards in the copies.
#[test]
fn an_unrolled_tile_of_accumulators_matches_bit_for_bit() {
    let mut compute = ComputeBuilder::new(ComputeName::new("tile"));
    let lhs = compute.input(TensorName::new("a"), DType::F32, shape(&[7, 11]));
    let rhs = compute.input(TensorName::new("b"), DType::F32, shape(&[11, 5]));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(7));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(5));
    let (axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(11));
    let product = binary(
        BinaryOp::Mul,
        &TensorExpr::Load { tensor: lhs, index: vec![row, k] },
        &TensorExpr::Load { tensor: rhs, index: vec![k, column] },
    );
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![axis], body: Box::new(product) };
    compute.output(TensorName::new("out"), sum);
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let by = |factor| vec![SplitFactor::new(factor).unwrap()];
    let rows = schedule.split(nest[0], by(2)).unwrap();
    let columns = schedule.split(nest[1], by(2)).unwrap();
    let steps = schedule.split(nest[2], by(4)).unwrap();
    let order = vec![rows[0], columns[0], steps[0], steps[1], rows[1], columns[1]];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    schedule.apply(Primitive::Bind { loop_: rows[0], axis: ThreadAxis::BlockX }).unwrap();
    schedule.apply(Primitive::Bind { loop_: columns[0], axis: ThreadAxis::ThreadX }).unwrap();
    for loop_ in [steps[1], rows[1], columns[1]] {
        schedule.apply(Primitive::Unroll { loop_ }).unwrap();
    }
    let kernel = schedule.validate(&preset(SmVersion::SM_89).unwrap()).unwrap().lower();
    let values = |count: u16, seed: f32| -> Vec<f32> {
        (0..count).map(|n| f32::from(n).mul_add(0.731, seed).sin()).collect()
    };
    let inputs = [
        HostTensor::new(shape(&[7, 11]), values(77, 1.0).into()).unwrap(),
        HostTensor::new(shape(&[11, 5]), values(55, 2.0).into()).unwrap(),
    ];

    let _outputs = match_interpreter(&kernel, &inputs, "").unwrap();
}

/// `floordiv(i - 9, d)` and `floormod(i - 9, d)` for powers of two `d` and
/// for 3, over operands from -9 to 10, as index values.
#[test]
fn floored_division_and_modulo_match_on_negative_operands() {
    let mut compute = ComputeBuilder::new(ComputeName::new("floored"));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(20));
    let exprs = compute.exprs();
    let offset = exprs.constant(-9);
    let shifted = exprs.add(i, offset);
    let mut outputs = Vec::new();
    for divisor in [1, 3, 4, 8] {
        let divisor = exprs.constant(divisor);
        outputs.push(exprs.floor_div(shifted, divisor));
        outputs.push(exprs.floor_mod(shifted, divisor));
    }
    for (n, index) in outputs.into_iter().enumerate() {
        let value = TensorExpr::Index { index, dtype: DType::I64 };
        compute.output(TensorName::new(format!("y{n}")), value);
    }

    check_against_interpreter(compute.finish().unwrap(), &[]);
}
