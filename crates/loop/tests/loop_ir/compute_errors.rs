//! Each way a compute definition is rejected.

use crate::compute_build::{binary, f32_const, load, shape};
use ptxc_core::{Axis, DType, Dim, Idx, Rank};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeError, ComputeName, IterVarId, ReduceOp, TensorExpr,
    TensorName, TensorParamId, VarId, VarName,
};

/// A builder with `x: f32[4, 8]` and `idx: f32[4]`, a spatial axis `i`
/// over 4, and a reduction axis `k` over 8.
struct Setup {
    b: ComputeBuilder,
    x: TensorParamId,
    floats: TensorParamId,
    i: ptxc_loop::ExprId,
    k_axis: IterVarId,
    k: ptxc_loop::ExprId,
    i_axis: IterVarId,
}

fn setup() -> Setup {
    let mut b = ComputeBuilder::new(ComputeName::new("bad"));
    let x = b.input(TensorName::new("x"), DType::F32, shape(&[4, 8]));
    let floats = b.input(TensorName::new("idx"), DType::F32, shape(&[4]));
    let (i_axis, i) = b.spatial(VarName::new("i"), Dim::new(4));
    let (k_axis, k) = b.reduce_axis(VarName::new("k"), Dim::new(8));
    Setup { b, x, floats, i, k_axis, k, i_axis }
}

fn sum(axes: Vec<IterVarId>, body: TensorExpr) -> TensorExpr {
    TensorExpr::Reduce { op: ReduceOp::Sum, axes, body: Box::new(body) }
}

/// The error `finish` gives with one output, `expr`.
fn reject(mut setup: Setup, expr: TensorExpr) -> ComputeError {
    setup.b.output(TensorName::new("out"), expr);
    setup.b.finish().unwrap_err()
}

#[test]
fn a_definition_needs_an_output() {
    let error = ComputeBuilder::new(ComputeName::new("empty")).finish().unwrap_err();

    assert_eq!(error, ComputeError::NoOutputs);
}

#[test]
fn ids_from_elsewhere_are_unknown() {
    let s = setup();
    let foreign = TensorParamId::new(9);
    assert_eq!(reject(s, load(foreign, &[])), ComputeError::UnknownTensor(foreign));

    let mut s = setup();
    let stray = s.b.exprs().var(VarId::new(9));
    let (x, k_axis) = (s.x, s.k_axis);
    assert_eq!(
        reject(s, sum(vec![k_axis], load(x, &[stray, stray]))),
        ComputeError::UnknownVar(VarId::new(9))
    );

    let s = setup();
    let (x, i, k) = (s.x, s.i, s.k);
    let error = reject(s, sum(vec![IterVarId::new(9)], load(x, &[i, k])));
    assert_eq!(error, ComputeError::UnknownAxis(IterVarId::new(9)));
}

#[test]
fn a_load_needs_one_index_per_axis() {
    let s = setup();
    let (x, i) = (s.x, s.i);

    let error = reject(s, load(x, &[i]));

    let expected = ComputeError::IndexArity {
        tensor: TensorName::new("x"),
        expected: Rank::new(2),
        actual: Rank::new(1),
    };
    assert_eq!(error, expected);
}

#[test]
fn a_loaded_index_must_come_from_an_integer_tensor() {
    let mut s = setup();
    let row = s.b.loaded_index(VarName::new("row"), s.floats, vec![s.i]);
    let (x, k_axis, k) = (s.x, s.k_axis, s.k);

    let error = reject(s, sum(vec![k_axis], load(x, &[row, k])));

    assert_eq!(
        error,
        ComputeError::IndexNotInteger { tensor: TensorName::new("idx"), dtype: DType::F32 }
    );
}

#[test]
fn reduction_axes_are_read_only_inside_their_one_reduction() {
    let s = setup();
    let (x, i, k) = (s.x, s.i, s.k);
    assert_eq!(reject(s, load(x, &[i, k])), ComputeError::ReduceAxisOutOfScope(VarName::new("k")));

    let s = setup();
    let (x, i, k, i_axis) = (s.x, s.i, s.k, s.i_axis);
    assert_eq!(
        reject(s, sum(vec![i_axis], load(x, &[i, k]))),
        ComputeError::NotAReduceAxis(VarName::new("i"))
    );

    let s = setup();
    let (x, i, k, k_axis) = (s.x, s.i, s.k, s.k_axis);
    let listed_twice = sum(vec![k_axis, k_axis], load(x, &[i, k]));
    assert_eq!(reject(s, listed_twice), ComputeError::ReduceAxisReused(VarName::new("k")));

    let mut s = setup();
    let zero = s.b.exprs().constant(0);
    let (x, i) = (s.x, s.i);
    assert_eq!(reject(s, load(x, &[i, zero])), ComputeError::UnusedReduceAxis(VarName::new("k")));

    let s = setup();
    assert_eq!(reject(s, sum(Vec::new(), f32_const(1.0))), ComputeError::EmptyReduce);
}

#[test]
fn operations_reject_types_they_do_not_take() {
    let s = setup();
    let (x, i, k, k_axis) = (s.x, s.i, s.k, s.k_axis);
    let int = TensorExpr::Const(ptxc_loop::Scalar::int(DType::I32, 1).unwrap());
    let mixed = binary(BinaryOp::Add, sum(vec![k_axis], load(x, &[i, k])), int);
    assert_eq!(
        reject(s, mixed),
        ComputeError::MixedDTypes { op: "add", lhs: DType::F32, rhs: DType::I32 }
    );

    let s = setup();
    let (x, i, k, k_axis) = (s.x, s.i, s.k, s.k_axis);
    let select = TensorExpr::Select {
        condition: Box::new(f32_const(1.0)),
        then: Box::new(sum(vec![k_axis], load(x, &[i, k]))),
        otherwise: Box::new(f32_const(0.0)),
    };
    assert_eq!(
        reject(s, select),
        ComputeError::UnsupportedDType { op: "select", dtype: DType::F32 }
    );
}

#[test]
fn the_output_shape_must_fit_in_u64_elements() {
    let mut b = ComputeBuilder::new(ComputeName::new("huge"));
    let _ = b.spatial(VarName::new("a"), Dim::new(u64::MAX));
    let _ = b.spatial(VarName::new("b"), Dim::new(2));
    b.output(TensorName::new("out"), f32_const(0.0));

    assert!(matches!(b.finish(), Err(ComputeError::Shape(_))));
}

#[test]
fn a_definition_is_one_stage_with_at_most_one_reduction() {
    let mut s = setup();
    let (m_axis, second) = s.b.reduce_axis(VarName::new("m"), Dim::new(8));
    let (x, i, k, k_axis) = (s.x, s.i, s.k, s.k_axis);
    let two = binary(
        BinaryOp::Add,
        sum(vec![k_axis], load(x, &[i, k])),
        sum(vec![m_axis], load(x, &[i, second])),
    );
    assert_eq!(reject(s, two), ComputeError::MultipleReductions);

    let mut s = setup();
    let (x, i, k, k_axis) = (s.x, s.i, s.k, s.k_axis);
    s.b.output(TensorName::new("extra"), f32_const(1.0));
    assert_eq!(
        reject(s, sum(vec![k_axis], load(x, &[i, k]))),
        ComputeError::ReductionWithSeveralOutputs
    );
}

#[test]
fn a_load_must_stay_within_its_tensor_over_the_domain() {
    let out_of_bounds = |offset: i64| {
        let mut s = setup();
        let shift = s.b.exprs().constant(offset);
        let shifted = s.b.exprs().add(s.k, shift);
        let (x, i, k_axis) = (s.x, s.i, s.k_axis);
        reject(s, sum(vec![k_axis], load(x, &[i, shifted])))
    };
    let expected = ComputeError::IndexOutOfBounds {
        tensor: TensorName::new("x"),
        axis: Axis::new(1),
        extent: Dim::new(8),
    };

    assert_eq!(out_of_bounds(1), expected);
    assert_eq!(out_of_bounds(-1), expected);
}

#[test]
fn an_empty_axis_reads_nothing_so_any_index_is_allowed() {
    let mut compute = ComputeBuilder::new(ComputeName::new("empty"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&[2]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(0));
    let ten = compute.exprs().constant(10);
    let far = compute.exprs().add(i, ten);
    compute.output(TensorName::new("out"), load(x, &[far]));

    assert!(compute.finish().is_ok());
}

#[test]
fn extents_and_tensor_sizes_must_fit_an_index() {
    let mut compute = ComputeBuilder::new(ComputeName::new("huge_tensor"));
    let x = compute.input(TensorName::new("x"), DType::F32, shape(&[1 << 62, 2]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(2));
    let zero = compute.exprs().constant(0);
    compute.output(TensorName::new("out"), load(x, &[i, zero]));
    assert_eq!(compute.finish().unwrap_err(), ComputeError::ExtentTooLarge(Dim::new(1 << 63)));

    let mut compute = ComputeBuilder::new(ComputeName::new("huge_axis"));
    let _ = compute.spatial(VarName::new("i"), Dim::new(1 << 63));
    compute.output(TensorName::new("out"), f32_const(0.0));
    assert_eq!(compute.finish().unwrap_err(), ComputeError::ExtentTooLarge(Dim::new(1 << 63)));
}

#[test]
fn an_index_value_needs_an_integer_type_and_its_axes_in_scope() {
    let s = setup();
    let i = s.i;
    let as_float = TensorExpr::Index { index: i, dtype: DType::F32 };
    assert_eq!(
        reject(s, as_float),
        ComputeError::UnsupportedDType { op: "index", dtype: DType::F32 }
    );

    let s = setup();
    let k = s.k;
    let outside = TensorExpr::Index { index: k, dtype: DType::I64 };
    assert_eq!(reject(s, outside), ComputeError::ReduceAxisOutOfScope(VarName::new("k")));
}
