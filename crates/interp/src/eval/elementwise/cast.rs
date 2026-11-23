//! `Cast` between any two element types.

use crate::error::InterpError;
use crate::eval::first;
use crate::numeric::{
    Elements, saturate_ints, store_floats, store_ints, store_ints_as_floats, truncate, widen,
};
use ptxc_core::float::Overflow;
use ptxc_core::{DType, HostTensor, TensorData};
use ptxc_graph::CastAttrs;

/// ONNX `Cast`: floats round to nearest-even (into FP8 as `saturate` says),
/// integers narrow by wrapping, and bools are 0 and 1.
pub(in crate::eval) fn cast(
    attrs: CastAttrs,
    operands: &[&HostTensor],
) -> Result<TensorData, InterpError> {
    let (to, overflow) = (attrs.to(), attrs.overflow());
    match widen(first(operands)?.data()) {
        Elements::Float(values) => from_floats(&values, to, overflow),
        Elements::Int(values) => from_ints(&values, to, overflow),
        Elements::Bool(values) => {
            from_ints(&values.into_iter().map(i64::from).collect::<Vec<_>>(), to, overflow)
        }
    }
}

const fn is_integer(dtype: DType) -> bool {
    matches!(dtype, DType::I8 | DType::U8 | DType::I32 | DType::I64)
}

fn from_floats(values: &[f32], to: DType, overflow: Overflow) -> Result<TensorData, InterpError> {
    if to == DType::Bool {
        return Ok(values.iter().map(|&v| v != 0.0).collect::<Vec<_>>().into());
    }
    if is_integer(to) {
        // NOTE: ONNX leaves out-of-range float-to-integer casts undefined;
        // ptxc truncates toward zero and saturates, NaN giving 0.
        let truncated: Vec<i64> = values.iter().map(|&v| truncate(f64::from(v))).collect();
        return saturate_ints(&truncated, to);
    }
    store_floats(values, to, overflow)
}

/// Integers into floats round once, from their exact value.
fn from_ints(values: &[i64], to: DType, overflow: Overflow) -> Result<TensorData, InterpError> {
    if to == DType::Bool {
        return Ok(values.iter().map(|&v| v != 0).collect::<Vec<_>>().into());
    }
    if is_integer(to) {
        return store_ints(values, to);
    }
    store_ints_as_floats(values, to, overflow)
}
