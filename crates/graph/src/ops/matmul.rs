//! Matrix multiplication ops and their type rules.

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::elementwise::{broadcast, broadcast_to};
use super::{OperandCount, TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{DType, Dim, Rank, Shape};

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
        let (a, b, bias) = match *operands {
            [a, b] => (a, b, None),
            [a, b, c] => (a, b, Some(c)),
            _ => {
                return Err(TypeError::ArityBetween {
                    min: OperandCount(2),
                    max: OperandCount(3),
                    actual: OperandCount(operands.len()),
                });
            }
        };
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
    if a.dtype() != b.dtype() {
        return Err(TypeError::MixedDTypes { lhs: a.dtype(), rhs: b.dtype() });
    }
    Ok(a.dtype())
}

fn matmul_shape(a: &Shape, b: &Shape) -> Result<Shape, TypeError> {
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
    if bias.dtype() != dtype {
        return Err(TypeError::MixedDTypes { lhs: dtype, rhs: bias.dtype() });
    }
    broadcast_to(bias.shape(), product)
}
