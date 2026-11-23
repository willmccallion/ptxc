//! Ops that are values known at import, or aliases, rather than graph ops:
//! `Constant`, `Shape`, `Size`, `ConstantOfShape`, `Range`, `Identity` and
//! inference-mode `Dropout`. With static shapes all of them are known (D3).

use crate::context::{NodeContext, integers};
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use crate::tensor::{import_tensor, shape_from_dims};
use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "Constant" => constant(context).map(|tensor| Lowered::Constants(vec![tensor])),
        "Shape" => shape(context).map(|tensor| Lowered::Constants(vec![tensor])),
        "Size" => size(context).map(|tensor| Lowered::Constants(vec![tensor])),
        "ConstantOfShape" => {
            constant_of_shape(context).map(|tensor| Lowered::Constants(vec![tensor]))
        }
        "Range" => range(context).map(|tensor| Lowered::Constants(vec![tensor])),
        "Identity" => Ok(Lowered::Alias(0)),
        "Dropout" => dropout(context),
        _ => return None,
    })
}

fn invalid(problem: String) -> ImportError {
    ImportErrorKind::Invalid { field: "attribute", problem }.into()
}

/// `Constant`, from whichever single value attribute it has.
fn constant(context: &mut NodeContext<'_>) -> Result<HostTensor, ImportError> {
    let attrs = &mut context.attrs;
    let mut found = Vec::new();
    if let Some(tensor) = attrs.tensor("value")? {
        found.push(import_tensor(tensor, None)?);
    }
    if let Some(value) = attrs.float("value_float")? {
        found.push(tensor(Shape::scalar(), vec![value].into())?);
    }
    if let Some(values) = attrs.floats("value_floats")? {
        found.push(tensor(vector_shape(values.len())?, values.to_vec().into())?);
    }
    if let Some(value) = attrs.int("value_int")? {
        found.push(tensor(Shape::scalar(), vec![value].into())?);
    }
    if let Some(values) = attrs.ints("value_ints")? {
        found.push(tensor(vector_shape(values.len())?, values.to_vec().into())?);
    }
    match <[HostTensor; 1]>::try_from(found) {
        Ok([tensor]) => Ok(tensor),
        Err(found) if found.is_empty() => Err(context.attrs.unsupported("value")),
        Err(_) => Err(invalid("Constant has more than one value attribute".to_owned())),
    }
}

fn vector_shape(length: usize) -> Result<Shape, ImportError> {
    let length =
        u64::try_from(length).map_err(|_| invalid("a value list is too long".to_owned()))?;
    Shape::new(vec![Dim::new(length)]).map_err(|error| invalid(error.to_string()))
}

fn tensor(shape: Shape, data: TensorData) -> Result<HostTensor, ImportError> {
    HostTensor::new(shape, data).map_err(|error| invalid(error.to_string()))
}

/// `Shape`: the input's dims, from `start` to `end` (opset 15), clamped as
/// ONNX clamps them.
fn shape(context: &mut NodeContext<'_>) -> Result<HostTensor, ImportError> {
    let dims: Vec<i64> = context
        .ty(0)?
        .shape()
        .dims()
        .iter()
        .map(|dim| i64::try_from(dim.get()).unwrap_or(i64::MAX))
        .collect();
    let rank = i64::try_from(dims.len()).unwrap_or(i64::MAX);
    let clamp = |index: i64| {
        let index = if index < 0 { index + rank } else { index };
        usize::try_from(index.clamp(0, rank)).unwrap_or(0)
    };
    let start = clamp(context.attrs.int_or("start", 0)?);
    let end = clamp(context.attrs.int_or("end", rank)?);
    let taken = dims.get(start..end.max(start)).unwrap_or_default().to_vec();
    int64_vector(taken)
}

fn int64_vector(values: Vec<i64>) -> Result<HostTensor, ImportError> {
    tensor(vector_shape(values.len())?, values.into())
}

/// `Size`: the input's element count, as an `i64` scalar.
fn size(context: &NodeContext<'_>) -> Result<HostTensor, ImportError> {
    let count = context.ty(0)?.shape().element_count().get();
    let count = i64::try_from(count)
        .map_err(|_| invalid("the element count does not fit i64".to_owned()))?;
    tensor(Shape::scalar(), vec![count].into())
}

/// `ConstantOfShape`: a tensor of the given shape filled with `value`
/// (default an `f32` zero).
fn constant_of_shape(context: &mut NodeContext<'_>) -> Result<HostTensor, ImportError> {
    let shape = shape_from_dims(&context.ints(0)?)?;
    let fill = match context.attrs.tensor("value")? {
        Some(value) => import_tensor(value, None)?,
        None => tensor(vector_shape(1)?, vec![0.0_f32].into())?,
    };
    let count = usize::try_from(shape.element_count().get())
        .map_err(|_| invalid("the shape is too large".to_owned()))?;
    let data = ptxc_core::with_tensor_data!(fill.data(), values => {
        let &[value] = values.as_slice() else { return Err(invalid("ConstantOfShape's value must have one element".to_owned())) };
        TensorData::from(vec![value; count])
    });
    tensor(shape, data)
}

/// `Range(start, limit, delta)`: `max(ceil((limit - start) / delta), 0)`
/// values from `start`, `delta` apart, for integer and `f32` scalars.
fn range(context: &NodeContext<'_>) -> Result<HostTensor, ImportError> {
    let (start, limit, delta) = (context.constant(0)?, context.constant(1)?, context.constant(2)?);
    if !matches!(start.dtype(), DType::F32 | DType::I32 | DType::I64) {
        return Err(
            crate::error::Unsupported::DType(format!("{:?} for Range", start.dtype())).into()
        );
    }
    if let (TensorData::F32(s), TensorData::F32(l), TensorData::F32(d)) =
        (start.data(), limit.data(), delta.data())
    {
        let (&[s], &[l], &[d]) = (s.as_slice(), l.as_slice(), d.as_slice()) else {
            return Err(not_scalars());
        };
        let count = float_count(s, l, d)?;
        let values: Vec<f32> = (0..count).map(|k| f32::from(k).mul_add(d, s)).collect();
        return tensor(vector_shape(values.len())?, values.into());
    }
    let (s, l, d) = (single(start)?, single(limit)?, single(delta)?);
    let values: Vec<i64> = (0..integer_count(s, l, d)?).map(|k| s + d * k).collect();
    let data: TensorData = match start.dtype() {
        DType::I32 => values
            .iter()
            .map(|&v| i32::try_from(v))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("Range overflows i32".to_owned()))?
            .into(),
        _ => values.into(),
    };
    tensor(vector_shape(ptxc_core::with_tensor_data!(&data, v => v.len()))?, data)
}

/// `max(ceil((limit - start) / delta), 0)` for integers.
fn integer_count(start: i64, limit: i64, delta: i64) -> Result<i64, ImportError> {
    if delta == 0 {
        return Err(invalid("Range's delta is zero".to_owned()));
    }
    let span = i128::from(limit) - i128::from(start);
    let delta = i128::from(delta);
    let count = if span.signum() == delta.signum() {
        (span.abs() + delta.abs() - 1) / delta.abs()
    } else {
        0
    };
    i64::try_from(count)
        .ok()
        .filter(|&count| count <= 65_535)
        .ok_or_else(|| invalid("Range is too long to evaluate at import".to_owned()))
}

fn not_scalars() -> ImportError {
    invalid("Range takes scalars".to_owned())
}

fn single(tensor: &HostTensor) -> Result<i64, ImportError> {
    match integers(tensor).as_deref() {
        Some(&[value]) => Ok(value),
        _ => Err(not_scalars()),
    }
}

/// `max(ceil((limit - start) / delta), 0)` computed in `f32`, as ONNX
/// does, found as the first index that reaches it.
fn float_count(start: f32, limit: f32, delta: f32) -> Result<u16, ImportError> {
    if delta == 0.0 {
        return Err(invalid("Range's delta is zero".to_owned()));
    }
    let count = ((limit - start) / delta).ceil();
    (0..=u16::MAX)
        .find(|&k| f32::from(k) >= count)
        .ok_or_else(|| invalid("Range is too long to evaluate at import".to_owned()))
}

/// Inference-mode `Dropout` passes its input through. Its optional mask
/// output, and training mode, are not supported.
fn dropout(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let _seed: Option<i64> = context.attrs.int("seed")?;
    if context.has_input(2) {
        let training = context.constant(2)?;
        if !matches!(training.data(), TensorData::Bool(values) if values.iter().all(|&on| !on)) {
            return Err(context.attrs.unsupported("training_mode"));
        }
    }
    if context.wants_output(1) {
        return Err(context.attrs.unsupported("mask"));
    }
    Ok(Lowered::Alias(0))
}
