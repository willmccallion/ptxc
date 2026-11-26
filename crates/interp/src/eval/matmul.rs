//! `MatMul`, `MatMulInteger` and `Gemm`, accumulating floats in f64 and
//! integers in i64.

use crate::error::{InterpError, Malformed};
use crate::extents::Extents;
use crate::numeric::{Elements, int_to_f64, store_ints, store_wide, truncate, widen};
use ptxc_core::{DType, HostTensor, TensorData};
use ptxc_graph::{GemmAttrs, MatMulIntegerAttrs, Orientation, TensorType, ZeroPoints};

/// An operand in the domain it accumulates in.
enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
}

fn values(tensor: &HostTensor) -> Result<Values, InterpError> {
    match widen(tensor.data()) {
        Elements::Float(v) => Ok(Values::Float(v.into_iter().map(f64::from).collect())),
        Elements::Int(v) => Ok(Values::Int(v)),
        Elements::Bool(_) => Err(Malformed::DType(tensor.dtype()).into()),
    }
}

/// `[batch…, rows, inner] x [batch…, inner, columns]` with the batch axes
/// broadcast.
pub(super) struct Batched {
    lhs_batch: Extents,
    rhs_batch: Extents,
    out_batch: Extents,
    pub(super) rows: usize,
    inner: usize,
    pub(super) columns: usize,
}

/// An operand's batch dims and `[rows, columns]`; a vector is `promoted`.
fn matrix(dims: &[usize], promoted: [usize; 2]) -> (Vec<usize>, [usize; 2]) {
    match dims {
        [batch @ .., rows, columns] => (batch.to_vec(), [*rows, *columns]),
        _ => (Vec::new(), promoted),
    }
}

impl Batched {
    /// NumPy's `matmul`: a vector on the left is `[1, inner]` and one on the
    /// right `[inner, 1]`. Dropping the added axis leaves the element order
    /// unchanged, so results are computed as `[batch…, rows, columns]`.
    pub(super) fn of(lhs: &Extents, rhs: &Extents) -> Self {
        let left_vector = [1, lhs.dims().last().copied().unwrap_or(1)];
        let right_vector = [rhs.dims().first().copied().unwrap_or(1), 1];
        let (lhs_batch, [rows, inner]) = matrix(lhs.dims(), left_vector);
        let (rhs_batch, [_, columns]) = matrix(rhs.dims(), right_vector);
        let rank = lhs_batch.len().max(rhs_batch.len());
        let aligned = |dims: &[usize], axis: usize| {
            (axis + dims.len()).checked_sub(rank).map_or(1, |position| dims[position])
        };
        // An extent of 1 broadcasts to the other, even to 0.
        let out_batch = (0..rank).map(|axis| {
            let (lhs, rhs) = (aligned(&lhs_batch, axis), aligned(&rhs_batch, axis));
            if lhs == 1 { rhs } else { lhs }
        });
        Self {
            out_batch: Extents::from_dims(out_batch.collect()),
            lhs_batch: Extents::from_dims(lhs_batch),
            rhs_batch: Extents::from_dims(rhs_batch),
            rows,
            inner,
            columns,
        }
    }

    /// Every product's elements, row-major, each folded over the inner axis
    /// from `zero` with `mul_add(lhs, rhs, acc)`.
    pub(super) fn products<T: Copy>(
        &self,
        lhs: &[T],
        rhs: &[T],
        zero: T,
        mul_add: impl Fn(T, T, T) -> T,
    ) -> Vec<T> {
        let (rows, inner, columns) = (self.rows, self.inner, self.columns);
        let mut out = Vec::with_capacity(self.out_batch.count() * rows * columns);
        for batch in self.out_batch.indices() {
            let lhs_base = self.lhs_batch.broadcast_offset(&batch) * rows * inner;
            let rhs_base = self.rhs_batch.broadcast_offset(&batch) * inner * columns;
            out.extend((0..rows * columns).map(|cell| {
                let (row, column) = (cell / columns, cell % columns);
                (0..inner).fold(zero, |acc, p| {
                    mul_add(
                        lhs[lhs_base + row * inner + p],
                        rhs[rhs_base + p * columns + column],
                        acc,
                    )
                })
            }));
        }
        out
    }
}

/// `MatMul`, as NumPy's `matmul`.
pub(super) fn matmul(
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let [lhs, rhs] = operands else { return Err(Malformed::Operands.into()) };
    let batched = Batched::of(&Extents::of(lhs.shape())?, &Extents::of(rhs.shape())?);
    match (values(lhs)?, values(rhs)?) {
        (Values::Float(l), Values::Float(r)) => {
            let out = batched.products(&l, &r, 0.0, f64::mul_add);
            store_wide(&out, output.dtype())
        }
        (Values::Int(l), Values::Int(r)) => {
            let out = batched.products(&l, &r, 0, |x, y, acc| x.wrapping_mul(y).wrapping_add(acc));
            store_ints(&out, output.dtype())
        }
        _ => Err(Malformed::DType(rhs.dtype()).into()),
    }
}

/// `MatMulInteger`: the product of the operands less their zero points,
/// `a`'s varying by row and `b`'s by column, wrapped to 32 bits.
pub(super) fn matmul_integer(
    attrs: &MatMulIntegerAttrs,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let [lhs, rhs] = operands else { return Err(Malformed::Operands.into()) };
    let batched = Batched::of(&Extents::of(lhs.shape())?, &Extents::of(rhs.shape())?);
    let (Values::Int(left), Values::Int(right)) = (values(lhs)?, values(rhs)?) else {
        return Err(Malformed::DType(lhs.dtype()).into());
    };
    let inner = lhs.shape().dims().last().map_or(1, |dim| dim.get());
    let inner = usize::try_from(inner).map_err(|_| Malformed::Operands)?;
    let left = less_zero_points(&left, &attrs.a, (inner, batched.rows));
    let right = less_zero_points(&right, &attrs.b, (1, batched.columns));
    let out = batched.products(&left, &right, 0, |x, y, acc| x * y + acc);
    store_ints(&out, DType::I32)
}

/// `values` less their zero points; element `i` takes the one at index
/// `(i / period) mod count` along the axis they vary along.
fn less_zero_points(
    values: &[i64],
    points: &ZeroPoints,
    (period, count): (usize, usize),
) -> Vec<i64> {
    let along = |position: usize| match points {
        ZeroPoints::PerTensor(point) => i64::from(point.get()),
        ZeroPoints::PerAxis { points, .. } => i64::from(points[position / period % count].get()),
    };
    values.iter().enumerate().map(|(position, value)| value - along(position)).collect()
}

/// A 2-D `Gemm` operand read in its orientation.
struct Oriented {
    values: Vec<f64>,
    orientation: Orientation,
    /// The stored matrix's column count.
    stride: usize,
}

impl Oriented {
    fn of(tensor: &HostTensor, orientation: Orientation) -> Result<Self, InterpError> {
        let stride = Extents::of(tensor.shape())?.dims()[1];
        Ok(Self { values: as_f64(tensor)?, orientation, stride })
    }

    /// Element `[row, column]` of the operand as used.
    fn at(&self, row: usize, column: usize) -> f64 {
        match self.orientation {
            Orientation::AsIs => self.values[row * self.stride + column],
            Orientation::Transposed => self.values[column * self.stride + row],
        }
    }
}

/// `Gemm`: `alpha * A' * B' + beta * C`, in f64. Integer `Gemm` truncates
/// the result back, as ONNX's reference casts its float result.
pub(super) fn gemm(
    attrs: &GemmAttrs,
    operands: &[&HostTensor],
    output: &TensorType,
) -> Result<TensorData, InterpError> {
    let (lhs, rhs, bias) = match operands {
        [lhs, rhs] => (lhs, rhs, None),
        [lhs, rhs, bias] => (lhs, rhs, Some(bias)),
        _ => return Err(Malformed::Operands.into()),
    };
    let out = Extents::of(output.shape())?;
    let lhs_dims = Extents::of(lhs.shape())?;
    let inner = if attrs.a == Orientation::AsIs { lhs_dims.dims()[1] } else { lhs_dims.dims()[0] };
    let (left, right) = (Oriented::of(lhs, attrs.a)?, Oriented::of(rhs, attrs.b)?);
    let bias = bias
        .map(|bias| Ok::<_, InterpError>((as_f64(bias)?, Extents::of(bias.shape())?)))
        .transpose()?;
    let (alpha, beta) = (f64::from(attrs.alpha.get()), f64::from(attrs.beta.get()));
    let result: Vec<f64> = out
        .indices()
        .map(|index| {
            let (row, column) = (index[0], index[1]);
            let product =
                (0..inner).fold(0.0, |acc, p| left.at(row, p).mul_add(right.at(p, column), acc));
            let added = bias
                .as_ref()
                .map_or(0.0, |(values, extents)| values[extents.broadcast_offset(&index)]);
            alpha.mul_add(product, beta * added)
        })
        .collect();
    if matches!(lhs.dtype(), ptxc_core::DType::I32 | ptxc_core::DType::I64) {
        store_ints(&result.into_iter().map(truncate).collect::<Vec<_>>(), output.dtype())
    } else {
        store_wide(&result, output.dtype())
    }
}

/// A float or integer operand's elements as f64.
fn as_f64(tensor: &HostTensor) -> Result<Vec<f64>, InterpError> {
    Ok(match values(tensor)? {
        Values::Float(v) => v,
        Values::Int(v) => v.into_iter().map(int_to_f64).collect(),
    })
}
