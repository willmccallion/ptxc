//! `MatMul` and `Gemm`: one reduction over the contracted axis, floats
//! accumulating in f32 (D43), with `Gemm`'s scaling and bias around it.

use super::NodeContext;
use crate::builder::{Operand, StageBuilder};
use crate::compose::{add, f32, mul};
use crate::error::LowerError;
use crate::precision::{convert, demote, promote};
use crate::stage::{Binding, LoweredGroup};
use ptxc_core::{DType, Dim};
use ptxc_graph::{F32Attr, GemmAttrs, Orientation, ValueId};
use ptxc_loop::{ExprId, IterVarId, ReduceOp, TensorExpr};

/// The dims of `value`.
fn dims(cx: &NodeContext<'_>, value: ValueId) -> Vec<Dim> {
    cx.graph.value(value).ty().shape().dims().to_vec()
}

/// `sum over k of x * y`: floats widened to f32, integers wrapping in
/// their own type.
fn dot(axis: IterVarId, x: &Operand, y: &Operand) -> TensorExpr {
    let product = mul(promote(x.expr.clone(), x.dtype), promote(y.expr.clone(), y.dtype));
    TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![axis], body: Box::new(product) }
}

/// `MatMul`, as NumPy's `matmul`: the last two axes multiply as matrices,
/// the rest broadcast, and a vector operand is a matrix whose added axis
/// the output drops.
pub(super) fn matmul(cx: &NodeContext<'_>) -> Result<LoweredGroup, LowerError> {
    let [lhs, rhs] = cx.values()?;
    cx.one_stage(|stage| {
        let product = product_reads(cx, stage, lhs, rhs)?;
        Ok(demote(dot(product.inner, &product.lhs, &product.rhs), product.lhs.dtype))
    })
}

/// A `MatMul`-shaped product's operands at one output element: each read
/// along the contracted axis, and the output's row and column, which a
/// vector operand does not have.
pub(super) struct Product {
    pub(super) lhs: Operand,
    pub(super) rhs: Operand,
    pub(super) inner: IterVarId,
    pub(super) row: Option<ExprId>,
    pub(super) column: Option<ExprId>,
}

/// `lhs` and `rhs` read for one element of their `MatMul` product.
pub(super) fn product_reads(
    cx: &NodeContext<'_>,
    stage: &mut StageBuilder<'_>,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<Product, LowerError> {
    let (lhs_dims, rhs_dims) = (dims(cx, lhs), dims(cx, rhs));
    let inner = *lhs_dims.last().ok_or(LowerError::Operands(cx.id))?;
    let spatial = stage.spatial().to_vec();
    let (has_rows, has_columns) = (lhs_dims.len() != 1, rhs_dims.len() != 1);
    let matrix_axes = usize::from(has_rows) + usize::from(has_columns);
    let batch_rank = spatial.len().checked_sub(matrix_axes).ok_or(LowerError::Operands(cx.id))?;
    let (batch, matrix) = spatial.split_at(batch_rank);
    let row = has_rows.then(|| matrix[0]);
    let column = has_columns.then(|| matrix[matrix.len() - 1]);
    let (inner_axis, k) = stage.reduce_axis(inner);
    let mut lhs_index = batch_index(stage, &lhs_dims, batch);
    lhs_index.extend(row);
    lhs_index.push(k);
    let mut rhs_index = batch_index(stage, &rhs_dims, batch);
    rhs_index.push(k);
    rhs_index.extend(column);
    let lhs = stage.load(Binding::Value(lhs), lhs_index);
    let rhs = stage.load(Binding::Value(rhs), rhs_index);
    Ok(Product { lhs, rhs, inner: inner_axis, row, column })
}

/// An operand's index along its batch axes, every axis but its last two
/// (none for a vector), aligned to the output's trailing batch axes; an
/// axis of extent 1 broadcasts, read at 0.
fn batch_index(stage: &mut StageBuilder<'_>, operand: &[Dim], batch: &[ExprId]) -> Vec<ExprId> {
    let own = &operand[..operand.len().saturating_sub(2)];
    let zero = stage.constant(0);
    let mut index: Vec<ExprId> = own
        .iter()
        .rev()
        .zip(batch.iter().rev())
        .map(|(dim, &axis)| if dim.get() == 1 { zero } else { axis })
        .collect();
    index.reverse();
    index
}

/// `Gemm`: `alpha * A' * B' + beta * C` on matrices, `A'` and `B'` read in
/// their orientations and `C` broadcast to the product.
pub(super) fn gemm(cx: &NodeContext<'_>, attrs: GemmAttrs) -> Result<LoweredGroup, LowerError> {
    let (lhs, rhs, bias) = match cx.operands()[..] {
        [lhs, rhs] => (lhs, rhs, None),
        [lhs, rhs, bias] => (lhs, rhs, Some(bias)),
        _ => return Err(LowerError::Operands(cx.id)),
    };
    let lhs_dims = dims(cx, lhs);
    let inner_axis = usize::from(attrs.a == Orientation::Transposed) ^ 1;
    let inner = *lhs_dims.get(inner_axis).ok_or(LowerError::Operands(cx.id))?;
    cx.one_stage(|stage| {
        let [row, column] = stage.spatial()[..] else { return Err(LowerError::Operands(cx.id)) };
        let (k_axis, k) = stage.reduce_axis(inner);
        let x = stage.load(Binding::Value(lhs), oriented(attrs.a, [row, k]));
        let y = stage.load(Binding::Value(rhs), oriented(attrs.b, [k, column]));
        let bias = bias.map(|bias| stage.broadcast(bias));
        if !x.dtype.is_float() {
            let product = dot(k_axis, &in_i64(&x), &in_i64(&y));
            return Ok(integer_gemm(product, attrs, bias.as_ref(), x.dtype));
        }
        let product = dot(k_axis, &x, &y);
        let scaled = scaled_sum(product, attrs, bias.map(|c| promote(c.expr, c.dtype)));
        Ok(demote(scaled, x.dtype))
    })
}

/// The index of element `[row, column]` of an operand as used, stored
/// as given or transposed.
fn oriented(orientation: Orientation, [row, column]: [ExprId; 2]) -> Vec<ExprId> {
    match orientation {
        Orientation::AsIs => vec![row, column],
        Orientation::Transposed => vec![column, row],
    }
}

/// `alpha * product + beta * bias` in f32, leaving out factors of 1.
fn scaled_sum(product: TensorExpr, attrs: GemmAttrs, bias: Option<TensorExpr>) -> TensorExpr {
    let scale = |factor: F32Attr, term: TensorExpr| {
        if factor == F32Attr::new(1.0) { term } else { mul(f32(factor.get()), term) }
    };
    let scaled = scale(attrs.alpha, product);
    match bias {
        Some(bias) => add(scaled, scale(attrs.beta, bias)),
        None => scaled,
    }
}

/// An integer operand widened to `i64`, where integer `Gemm` sums exactly.
fn in_i64(operand: &Operand) -> Operand {
    Operand { expr: convert(operand.expr.clone(), operand.dtype, DType::I64), dtype: DType::I64 }
}

/// Integer `Gemm` on an `i64` product: exact when `alpha` and `beta` are
/// 1, else scaled in f32 and truncated, as the reference truncates its
/// result.
fn integer_gemm(
    product: TensorExpr,
    attrs: GemmAttrs,
    bias: Option<&Operand>,
    dtype: DType,
) -> TensorExpr {
    let wide = |expr, from| convert(expr, from, DType::I64);
    let unit = F32Attr::new(1.0);
    if attrs.alpha == unit && attrs.beta == unit {
        let sum = match bias {
            Some(c) => add(product, wide(c.expr.clone(), c.dtype)),
            None => product,
        };
        return convert(sum, DType::I64, dtype);
    }
    // NOTE: the reference scales integer `Gemm` in f64; loop IR has no f64
    // (D42), so ptxc scales in f32, exact while the terms stay below 2^24.
    let to_f32 = |expr, from| convert(expr, from, DType::F32);
    let bias = bias.map(|c| to_f32(c.expr.clone(), c.dtype));
    let scaled = scaled_sum(to_f32(product, DType::I64), attrs, bias);
    convert(convert(scaled, DType::F32, DType::I64), DType::I64, dtype)
}
