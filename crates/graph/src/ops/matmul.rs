//! Matrix multiplication ops and their type rules.

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::elementwise::{broadcast, broadcast_to};
use super::{
    TypeError, TypeRule, check_axis, exact_operands, operands_with_optional, require_dtype,
};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Axis, DType, Dim, Rank, Shape, ZeroPoint};
use std::ops::RangeInclusive;

/// Whether a `Gemm` operand is used as given or transposed, from ONNX's
/// `transA` and `transB`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Orientation {
    /// As given.
    AsIs,
    /// Transposed.
    Transposed,
}

/// ONNX `Gemm`: `alpha * A' * B' + beta * C` on matrices, where `A'` and
/// `B'` are `A` and `B` in their orientations and the optional `C`
/// broadcasts to the product's shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GemmAttrs {
    /// How `A` is used.
    pub a: Orientation,
    /// How `B` is used.
    pub b: Orientation,
    /// The product's scale.
    pub alpha: F32Attr,
    /// `C`'s scale.
    pub beta: F32Attr,
}

impl TypeRule for GemmAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let ([a, b], bias) = operands_with_optional(operands)?;
        let dtype = matmul_dtype(a, b)?;
        let [m, k_a] = oriented(a.shape(), self.a)?;
        let [k_b, n] = oriented(b.shape(), self.b)?;
        if k_a != k_b {
            return Err(TypeError::ContractedDims { lhs: k_a, rhs: k_b });
        }
        let product = Shape::new(vec![m, n])?;
        if let Some(bias) = bias {
            check_bias(bias, dtype, &product)?;
        }
        Ok(vec![TensorType::new(dtype, product, Layout::RowMajor)])
    }
}

/// The zero points of one `MatMulInteger` operand: one for the whole
/// tensor, or one per index along an axis, the left operand's rows or the
/// right operand's columns.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ZeroPoints {
    /// One for every element.
    PerTensor(ZeroPoint),
    /// One per index along `axis`.
    PerAxis {
        /// The axis they vary along.
        axis: Axis,
        /// One per index along `axis`.
        points: Vec<ZeroPoint>,
    },
}

/// ONNX `MatMulInteger`: the `i32` matrix product of 8-bit integer
/// operands, each less its zero points. Shapes follow `MatMul`, and the
/// sum wraps in 32 bits, as ONNX allows.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct MatMulIntegerAttrs {
    /// The left operand's zero points, which may vary by row.
    pub a: ZeroPoints,
    /// The right operand's zero points, which may vary by column.
    pub b: ZeroPoints,
}

impl TypeRule for MatMulIntegerAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [a, b] = exact_operands(operands)?;
        let rank = |operand: &TensorType| operand.shape().rank().get();
        let rows = rank(a).checked_sub(2).map(Axis::new);
        let columns = (rank(b) >= 2).then(|| Axis::new(rank(b) - 1));
        check_zero_points(&self.a, a, rows)?;
        check_zero_points(&self.b, b, columns)?;
        let shape = matmul_shape(a.shape(), b.shape())?;
        Ok(vec![TensorType::new(DType::I32, shape, Layout::RowMajor)])
    }
}

/// Checks that `operand` is an 8-bit integer tensor and that `points` are
/// in its range, varying, if at all, along `allowed` with one per index.
fn check_zero_points(
    points: &ZeroPoints,
    operand: &TensorType,
    allowed: Option<Axis>,
) -> Result<(), TypeError> {
    let dtype = operand.dtype();
    Accepts::Int8.check(dtype)?;
    let values = match points {
        ZeroPoints::PerTensor(point) => std::slice::from_ref(point),
        ZeroPoints::PerAxis { axis, points } => {
            if Some(*axis) != allowed {
                return Err(TypeError::QuantAxis(*axis));
            }
            check_axis(operand.shape(), *axis)?;
            let dim = operand.shape().dims()[axis.index()];
            if usize::try_from(dim.get()) != Ok(points.len()) {
                return Err(TypeError::QuantMapCount { axis: *axis, dim, maps: points.len() });
            }
            points.as_slice()
        }
    };
    let range = int8_range(dtype);
    match values.iter().find(|point| !range.contains(&point.get())) {
        Some(&zero_point) => Err(TypeError::ZeroPointRange { zero_point, dtype }),
        None => Ok(()),
    }
}

/// The values of `dtype`, an 8-bit integer type.
fn int8_range(dtype: DType) -> RangeInclusive<i32> {
    if dtype == DType::U8 {
        i32::from(u8::MIN)..=i32::from(u8::MAX)
    } else {
        i32::from(i8::MIN)..=i32::from(i8::MAX)
    }
}

/// ONNX `MatMul`, as numpy's `matmul`: the last two axes multiply as matrices
/// and the rest broadcast. A 1-D operand is promoted to a matrix (`[1, K]`
/// on the left, `[K, 1]` on the right) and the added axis is removed from the
/// result.
pub(super) fn infer_matmul(operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
    let [a, b] = exact_operands(operands)?;
    let dtype = matmul_dtype(a, b)?;
    Ok(vec![TensorType::new(dtype, matmul_shape(a.shape(), b.shape())?, Layout::RowMajor)])
}

fn matmul_dtype(a: &TensorType, b: &TensorType) -> Result<DType, TypeError> {
    Accepts::FloatOrWideInt.check(a.dtype())?;
    require_dtype(a.dtype(), b)?;
    Ok(a.dtype())
}

pub(super) fn matmul_shape(a: &Shape, b: &Shape) -> Result<Shape, TypeError> {
    let (a_batch, [m, k_a]) = split_matrix(a, Side::Left)?;
    let (b_batch, [k_b, n]) = split_matrix(b, Side::Right)?;
    if k_a != k_b {
        return Err(TypeError::ContractedDims { lhs: k_a, rhs: k_b });
    }
    let batch = broadcast(&Shape::new(a_batch.to_vec())?, &Shape::new(b_batch.to_vec())?)?;
    let mut dims = batch.dims().to_vec();
    if a.rank() != Rank::new(1) {
        dims.push(m);
    }
    if b.rank() != Rank::new(1) {
        dims.push(n);
    }
    Ok(Shape::new(dims)?)
}

/// Which side of a product a matrix is on, which decides how a vector is
/// promoted.
#[derive(Clone, Copy, Debug)]
enum Side {
    Left,
    Right,
}

/// The batch dims and the `[rows, columns]` of an operand, promoting a vector.
fn split_matrix(shape: &Shape, side: Side) -> Result<(&[Dim], [Dim; 2]), TypeError> {
    let one = Dim::new(1);
    match (shape.dims(), side) {
        ([], _) => Err(TypeError::RankBelow { minimum: Rank::new(1), actual: shape.rank() }),
        ([k], Side::Left) => Ok((&[], [one, *k])),
        ([k], Side::Right) => Ok((&[], [*k, one])),
        ([batch @ .., rows, columns], _) => Ok((batch, [*rows, *columns])),
    }
}

/// `[rows, columns]` of a 2-D `Gemm` operand in its orientation.
fn oriented(shape: &Shape, orientation: Orientation) -> Result<[Dim; 2], TypeError> {
    let &[rows, columns] = shape.dims() else {
        return Err(TypeError::Rank { expected: Rank::new(2), actual: shape.rank() });
    };
    Ok(match orientation {
        Orientation::AsIs => [rows, columns],
        Orientation::Transposed => [columns, rows],
    })
}

fn check_bias(bias: &TensorType, dtype: DType, product: &Shape) -> Result<(), TypeError> {
    require_dtype(dtype, bias)?;
    broadcast_to(bias.shape(), product)
}
