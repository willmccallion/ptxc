//! Values of small integer computations, known at import.
//!
//! Exported models compute shapes in the graph (`Shape -> Gather -> Concat
//! -> Reshape`). Static shapes make those values constants (D3), and the
//! canonical forms need them (D32), so the importer tracks them alongside
//! the graph: the ops stay in the graph, and this evaluator works out their
//! values for the shape operands that read them. It covers the integer ops
//! such chains use; anything else is not known at import.

#[cfg(test)]
mod tests;

use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};
use ptxc_graph::{BinaryOp, ModKind, Op, ReduceOp, TensorType, UnaryOp, VariadicOp};

/// Computations past this many elements are left to the graph.
const LIMIT: u64 = 4096;

/// The value of `op` applied to `operands`, given the output types the type
/// rule inferred, if every operand and output is a small integer tensor and
/// the op is one this evaluator covers.
pub(crate) fn evaluate(
    op: &Op,
    operands: &[&HostTensor],
    outputs: &[TensorType],
) -> Option<Vec<HostTensor>> {
    let [output] = outputs else { return None };
    if let Some(moved) = move_data(op, operands, output) {
        return Some(vec![moved]);
    }
    let ints: Vec<Ints> =
        operands.iter().map(|tensor| Ints::from_host(tensor)).collect::<Option<_>>()?;
    if output.shape().element_count().get() > LIMIT {
        return None;
    }
    let out_shape = output.shape();
    let values = match op {
        Op::Binary(op) => broadcast_map(&ints, out_shape, |v| binary(*op, v[0], v[1]))?,
        Op::Unary(op) => broadcast_map(&ints, out_shape, |v| unary(*op, v[0]))?,
        Op::Variadic(op) => broadcast_map(&ints, out_shape, |v| variadic(*op, v))?,
        Op::Where => {
            broadcast_map(&ints, out_shape, |v| Some(if v[0] != 0 { v[1] } else { v[2] }))?
        }
        Op::Cast(_) => ints.first()?.values.clone(),
        Op::Expand(_) => broadcast_map(&ints, out_shape, |v| Some(v[0]))?,
        Op::Concat(axis) => concat(&ints, axis.index()),
        Op::Gather(axis) => gather(ints.first()?, ints.get(1)?, axis.index())?,
        Op::Transpose(attrs) => {
            let permutation: Vec<usize> =
                attrs.permutation.iter().map(|axis| axis.index()).collect();
            gather_positions(ints.first()?, out_shape, |out| transpose_source(out, &permutation))
        }
        Op::Slice(attrs) => {
            let mut map = Vec::new();
            for slice in &attrs.axes {
                map.push((slice.axis.index(), slice.start, slice.step.get()));
            }
            gather_positions(ints.first()?, out_shape, |out| slice_source(out, &map))
        }
        Op::Tile(_) => {
            let input = ints.first()?;
            gather_positions(input, out_shape, |out| {
                out.iter().zip(input.shape.dims()).map(|(&i, d)| i % to_usize(*d)).collect()
            })
        }
        Op::Reduce(attrs) => reduce(
            ints.first()?,
            attrs.op,
            &attrs.axes.iter().map(|axis| axis.index()).collect::<Vec<_>>(),
            out_shape,
        )?,
        _ => return None,
    };
    Some(vec![Ints { dtype: output.dtype(), shape: out_shape.clone(), values }.into_host()?])
}

/// Ops that only move data, tracked for every element type: a cast to the
/// type the operand already has, and `Reshape`.
fn move_data(op: &Op, operands: &[&HostTensor], output: &TensorType) -> Option<HostTensor> {
    let [operand] = operands else { return None };
    let moves = match op {
        Op::Cast(attrs) => attrs.to() == operand.dtype(),
        Op::Reshape(_) => true,
        _ => false,
    };
    if !moves || operand.shape().element_count().get() > LIMIT {
        return None;
    }
    let (_, data) = (*operand).clone().into_parts();
    HostTensor::new(output.shape().clone(), data).ok()
}

/// An integer or bool tensor's elements as `i64`, row-major.
struct Ints {
    dtype: DType,
    shape: Shape,
    values: Vec<i64>,
}

impl Ints {
    fn from_host(tensor: &HostTensor) -> Option<Self> {
        if tensor.shape().element_count().get() > LIMIT {
            return None;
        }
        let values = match tensor.data() {
            TensorData::I64(values) => values.clone(),
            TensorData::I32(values) => values.iter().map(|&v| i64::from(v)).collect(),
            TensorData::Bool(values) => values.iter().map(|&v| i64::from(v)).collect(),
            _ => return None,
        };
        Some(Self { dtype: tensor.dtype(), shape: tensor.shape().clone(), values })
    }

    fn into_host(self) -> Option<HostTensor> {
        let data: TensorData = match self.dtype {
            DType::I64 => self.values.into(),
            DType::I32 => self
                .values
                .iter()
                .map(|&v| i32::try_from(v).ok())
                .collect::<Option<Vec<_>>>()?
                .into(),
            DType::Bool => self.values.iter().map(|&v| v != 0).collect::<Vec<_>>().into(),
            _ => return None,
        };
        HostTensor::new(self.shape, data).ok()
    }
}

fn to_usize(dim: Dim) -> usize {
    usize::try_from(dim.get()).unwrap_or(usize::MAX)
}

/// Row-major strides of `dims`.
fn strides(dims: &[Dim]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * to_usize(dims[axis + 1]);
    }
    strides
}

/// Every multi-index of `shape`, row-major.
fn indices(shape: &Shape) -> impl Iterator<Item = Vec<usize>> + '_ {
    let dims: Vec<usize> = shape.dims().iter().map(|&dim| to_usize(dim)).collect();
    let count = usize::try_from(shape.element_count().get()).unwrap_or(0);
    let strides = strides(shape.dims());
    (0..count).map(move |flat| {
        dims.iter().zip(&strides).map(|(&dim, &stride)| flat / stride % dim.max(1)).collect()
    })
}

/// The flat position in `shape` of an output multi-index, broadcasting.
fn broadcast_position(out: &[usize], shape: &Shape) -> usize {
    let dims = shape.dims();
    let offset = out.len() - dims.len();
    dims.iter()
        .zip(strides(dims))
        .enumerate()
        .map(
            |(axis, (&dim, stride))| {
                if dim == Dim::new(1) { 0 } else { out[offset + axis] * stride }
            },
        )
        .sum()
}

fn broadcast_map(
    ints: &[Ints],
    out_shape: &Shape,
    each: impl Fn(&[i64]) -> Option<i64>,
) -> Option<Vec<i64>> {
    indices(out_shape)
        .map(|out| {
            let args: Vec<i64> = ints
                .iter()
                .map(|operand| operand.values[broadcast_position(&out, &operand.shape)])
                .collect();
            each(&args)
        })
        .collect()
}

/// Each output element read from the input at the position `source` gives.
fn gather_positions(
    input: &Ints,
    out_shape: &Shape,
    source: impl Fn(&[usize]) -> Vec<usize>,
) -> Vec<i64> {
    let strides = strides(input.shape.dims());
    indices(out_shape)
        .map(|out| {
            input.values[source(&out).iter().zip(&strides).map(|(i, s)| i * s).sum::<usize>()]
        })
        .collect()
}

/// Output axis `i` reads input axis `permutation[i]`.
fn transpose_source(out: &[usize], permutation: &[usize]) -> Vec<usize> {
    let mut source = vec![0; out.len()];
    for (&index, &axis) in out.iter().zip(permutation) {
        source[axis] = index;
    }
    source
}

fn slice_source(out: &[usize], map: &[(usize, u64, i64)]) -> Vec<usize> {
    let mut source = out.to_vec();
    for &(axis, start, step) in map {
        let position =
            i128::from(start) + i128::from(step) * i128::try_from(out[axis]).unwrap_or(0);
        source[axis] = usize::try_from(position).unwrap_or(0);
    }
    source
}

fn binary(op: BinaryOp, a: i64, b: i64) -> Option<i64> {
    Some(match op {
        BinaryOp::Add => a.checked_add(b)?,
        BinaryOp::Sub => a.checked_sub(b)?,
        BinaryOp::Mul => a.checked_mul(b)?,
        BinaryOp::Div => a.checked_div(b)?,
        BinaryOp::Mod(ModKind::Fmod) => a.checked_rem(b)?,
        BinaryOp::Mod(ModKind::Integer) => {
            let remainder = a.checked_rem(b)?;
            if remainder != 0 && remainder.signum() != b.signum() {
                remainder + b
            } else {
                remainder
            }
        }
        BinaryOp::Equal => i64::from(a == b),
        BinaryOp::Less => i64::from(a < b),
        BinaryOp::LessOrEqual => i64::from(a <= b),
        BinaryOp::Greater => i64::from(a > b),
        BinaryOp::GreaterOrEqual => i64::from(a >= b),
        BinaryOp::And => i64::from(a != 0 && b != 0),
        BinaryOp::Or => i64::from(a != 0 || b != 0),
        BinaryOp::Xor => i64::from((a != 0) != (b != 0)),
        BinaryOp::Pow => return None,
    })
}

fn unary(op: UnaryOp, a: i64) -> Option<i64> {
    match op {
        UnaryOp::Neg => a.checked_neg(),
        UnaryOp::Abs => a.checked_abs(),
        UnaryOp::Sign => Some(a.signum()),
        UnaryOp::Not => Some(i64::from(a == 0)),
        _ => None,
    }
}

fn variadic(op: VariadicOp, values: &[i64]) -> Option<i64> {
    match op {
        VariadicOp::Max => values.iter().copied().max(),
        VariadicOp::Min => values.iter().copied().min(),
        VariadicOp::Sum | VariadicOp::Mean => None,
    }
}

fn concat(ints: &[Ints], axis: usize) -> Vec<i64> {
    let outer: usize = ints[0].shape.dims()[..axis].iter().map(|&d| to_usize(d)).product();
    let mut values = Vec::new();
    for block in 0..outer {
        for operand in ints {
            let inner: usize = operand.shape.dims()[axis..].iter().map(|&d| to_usize(d)).product();
            values.extend_from_slice(&operand.values[block * inner..(block + 1) * inner]);
        }
    }
    values
}

fn gather(data: &Ints, indices_tensor: &Ints, axis: usize) -> Option<Vec<i64>> {
    let dims = data.shape.dims();
    let extent = i64::try_from(dims[axis].get()).ok()?;
    let outer: usize = dims[..axis].iter().map(|&d| to_usize(d)).product();
    let inner: usize = dims[axis + 1..].iter().map(|&d| to_usize(d)).product();
    let mut values = Vec::new();
    for block in 0..outer {
        for &index in &indices_tensor.values {
            let index = if index < 0 { index + extent } else { index };
            let index = usize::try_from(index)
                .ok()
                .filter(|&i| i64::try_from(i).is_ok_and(|i| i < extent))?;
            let start = (block * to_usize(dims[axis]) + index) * inner;
            values.extend_from_slice(&data.values[start..start + inner]);
        }
    }
    Some(values)
}

fn reduce(input: &Ints, op: ReduceOp, axes: &[usize], out_shape: &Shape) -> Option<Vec<i64>> {
    let mut accumulated: Vec<Option<i64>> =
        vec![None; usize::try_from(out_shape.element_count().get()).ok()?];
    let out_strides = strides(out_shape.dims());
    let keep = out_shape.rank() == input.shape.rank();
    for (source, &value) in indices(&input.shape).zip(&input.values) {
        let kept = source.iter().enumerate().filter(|(axis, _)| keep || !axes.contains(axis));
        let position: usize = kept
            .zip(&out_strides)
            .map(|((axis, &i), &s)| if axes.contains(&axis) { 0 } else { i * s })
            .sum();
        accumulated[position] = Some(match (accumulated[position], op) {
            (None, ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Max | ReduceOp::Min) => value,
            (Some(acc), ReduceOp::Sum) => acc.checked_add(value)?,
            (Some(acc), ReduceOp::Prod) => acc.checked_mul(value)?,
            (Some(acc), ReduceOp::Max) => acc.max(value),
            (Some(acc), ReduceOp::Min) => acc.min(value),
            _ => return None,
        });
    }
    let empty = match op {
        ReduceOp::Sum => Some(0),
        ReduceOp::Prod => Some(1),
        _ => None,
    };
    accumulated.into_iter().map(|value| value.or(empty)).collect()
}
