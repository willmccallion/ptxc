//! Value expressions: the buffers they read.

use ptxc_core::float::Overflow;
use ptxc_core::{DType, Idx};
use ptxc_loop::{BinaryOp, BufferId, ExprArena, Scalar, UnaryOp, ValueExpr};

fn load(buffer: usize) -> ValueExpr {
    ValueExpr::Load { buffer: BufferId::new(buffer), index: ExprArena::new().constant(0) }
}

#[test]
fn an_expression_reads_the_buffers_of_its_loads_at_any_depth() {
    let one = ValueExpr::Const(Scalar::one(DType::F32));
    let negated = ValueExpr::Unary(UnaryOp::Neg, Box::new(load(0)));
    let sum = ValueExpr::Binary(BinaryOp::Add, Box::new(one.clone()), Box::new(negated));
    let cast =
        ValueExpr::Cast { to: DType::F16, overflow: Overflow::Saturate, value: Box::new(sum) };
    let picked = ValueExpr::Select {
        condition: Box::new(load(1)),
        then: Box::new(one),
        otherwise: Box::new(cast),
    };

    let read: Vec<bool> = (0..3).map(|buffer| picked.reads(BufferId::new(buffer))).collect();

    assert_eq!(read, [true, true, false]);
}

#[test]
fn constants_and_indices_read_no_buffer() {
    let index = ValueExpr::Index { index: ExprArena::new().constant(3), dtype: DType::I64 };
    let constant = ValueExpr::Const(Scalar::i64(3));
    let then_side = ValueExpr::Select {
        condition: Box::new(ValueExpr::Const(Scalar::bool(true))),
        then: Box::new(load(2)),
        otherwise: Box::new(index.clone()),
    };

    assert!(!index.reads(BufferId::new(0)));
    assert!(!constant.reads(BufferId::new(0)));
    assert!(then_side.reads(BufferId::new(2)));
}

#[test]
fn a_multiply_add_reads_the_buffers_of_all_three_operands() {
    let fused = ValueExpr::MulAdd {
        lhs: Box::new(load(0)),
        rhs: Box::new(load(1)),
        addend: Box::new(load(2)),
    };

    assert!((0..3).all(|buffer| fused.reads(BufferId::new(buffer))));
    assert!(!fused.reads(BufferId::new(3)));
}
