//! Declared types (`ValueInfoProto`) as tensor types, with symbolic
//! dimensions resolved from the caller's sizes.

use crate::error::{ImportError, ImportErrorKind, Unsupported};
use crate::tensor::dtype_from_onnx;
use ptxc_core::{Dim, Shape};
use ptxc_graph::text::format_type;
use ptxc_graph::{Layout, TensorType};
use ptxc_onnx_proto::ValueInfoProto;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::tensor_shape_proto::dimension::Value as DimValue;
use ptxc_onnx_proto::type_proto::{Tensor, Value as TypeValue};
use std::collections::HashMap;

/// A graph input's type. Every dimension must be known: fixed in the model,
/// or symbolic with a size the caller supplied (D3).
pub(crate) fn input_type(
    info: &ValueInfoProto,
    symbolic: &HashMap<String, Dim>,
) -> Result<TensorType, ImportError> {
    let tensor = tensor_type(info)?;
    let dtype =
        dtype_from_onnx(tensor.elem_type.ok_or(ImportErrorKind::Missing("input element type"))?)?;
    let declared = tensor.shape.as_ref().ok_or(ImportErrorKind::Missing("input shape"))?;
    let dims = declared
        .dim
        .iter()
        .map(|dim| match resolve(dim.value.as_ref(), symbolic) {
            Resolved::Known(extent) => Ok(extent),
            Resolved::Symbolic(name) => Err(ImportError::from(Unsupported::DynamicShape(name))),
            Resolved::Invalid => Err(invalid("input shape has a negative dimension")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shape = Shape::new(dims).map_err(|error| invalid(&error.to_string()))?;
    Ok(TensorType::new(dtype, shape, Layout::RowMajor))
}

/// Checks a declared output against its inferred type, on whatever the model
/// declares: the element type, the rank, and every dimension that is fixed
/// or has a supplied size.
pub(crate) fn check_declared(
    info: &ValueInfoProto,
    inferred: &TensorType,
    symbolic: &HashMap<String, Dim>,
) -> Result<(), ImportError> {
    let Ok(tensor) = tensor_type(info) else { return Ok(()) };
    let dtype_matches = tensor
        .elem_type
        .is_none_or(|code| dtype_from_onnx(code).is_ok_and(|dtype| dtype == inferred.dtype()));
    let shape_matches = tensor.shape.as_ref().is_none_or(|declared| {
        declared.dim.len() == inferred.shape().dims().len()
            && declared.dim.iter().zip(inferred.shape().dims()).all(|(dim, &actual)| match resolve(
                dim.value.as_ref(),
                symbolic,
            ) {
                Resolved::Known(extent) => extent == actual,
                Resolved::Symbolic(_) => true,
                Resolved::Invalid => false,
            })
    });
    if dtype_matches && shape_matches {
        return Ok(());
    }
    Err(ImportErrorKind::DeclaredType {
        name: info.name.clone().unwrap_or_default(),
        declared: describe(tensor),
        inferred: format_type(inferred),
    }
    .into())
}

fn tensor_type(info: &ValueInfoProto) -> Result<&Tensor, ImportError> {
    match info.r#type.as_ref().and_then(|ty| ty.value.as_ref()) {
        Some(TypeValue::TensorType(tensor)) => Ok(tensor),
        Some(_) => Err(Unsupported::DType("a non-tensor type".to_owned()).into()),
        None => Err(ImportErrorKind::Missing("value type").into()),
    }
}

/// One declared dimension.
enum Resolved {
    Known(Dim),
    Symbolic(String),
    Invalid,
}

fn resolve(dim: Option<&DimValue>, symbolic: &HashMap<String, Dim>) -> Resolved {
    match dim {
        Some(DimValue::DimValue(value)) => {
            u64::try_from(*value).map_or(Resolved::Invalid, |v| Resolved::Known(Dim::new(v)))
        }
        Some(DimValue::DimParam(name)) => symbolic
            .get(name)
            .map_or_else(|| Resolved::Symbolic(name.clone()), |&dim| Resolved::Known(dim)),
        None => Resolved::Symbolic(String::from("?")),
    }
}

fn invalid(problem: &str) -> ImportError {
    ImportErrorKind::Invalid { field: "value info", problem: problem.to_owned() }.into()
}

/// A declared type as written in the model, such as `FLOAT[batch, 3]`.
fn describe(tensor: &Tensor) -> String {
    let dtype = tensor
        .elem_type
        .and_then(|code| DataType::try_from(code).ok())
        .map_or("?", |dtype| dtype.as_str_name());
    let dims: Vec<String> = tensor
        .shape
        .iter()
        .flat_map(|shape| &shape.dim)
        .map(|dim| match &dim.value {
            Some(DimValue::DimValue(value)) => value.to_string(),
            Some(DimValue::DimParam(name)) => name.clone(),
            None => "?".to_owned(),
        })
        .collect();
    format!("{dtype}[{}]", dims.join(", "))
}
