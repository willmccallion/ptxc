//! Tensors, graphs from text, and comparisons.

use ptxc_core::{Dim, Evaluate, HostTensor, Shape, TensorData};
use ptxc_graph::text::parse;
use ptxc_interp::{InterpError, Interpreter};

pub(crate) fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

pub(crate) fn tensor(dims: &[u64], data: impl Into<TensorData>) -> HostTensor {
    HostTensor::new(shape(dims), data.into()).unwrap()
}

pub(crate) fn f32s(dims: &[u64], values: &[f32]) -> HostTensor {
    tensor(dims, values.to_vec())
}

pub(crate) fn i64s(dims: &[u64], values: &[i64]) -> HostTensor {
    tensor(dims, values.to_vec())
}

/// The outputs of the graph written in `text` on `inputs`.
pub(crate) fn try_run(text: &str, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, InterpError> {
    let graph = parse(text, &[]).unwrap_or_else(|error| panic!("{error}\n{text}"));
    Interpreter::new(&graph).evaluate(inputs)
}

pub(crate) fn run(text: &str, inputs: &[HostTensor]) -> Vec<HostTensor> {
    try_run(text, inputs).unwrap_or_else(|error| panic!("{error}\n{text}"))
}

/// A float tensor's elements as f64.
pub(crate) fn values(tensor: &HostTensor) -> Vec<f64> {
    match tensor.data() {
        TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E4M3Fn(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E5M2(v) => v.iter().map(|x| x.to_f64()).collect(),
        other => panic!("not a float tensor: {other:?}"),
    }
}

/// Asserts the shape and that every element is within `1e-6` relative (or
/// both NaN, or equal infinities) of `expected`.
pub(crate) fn assert_close(actual: &HostTensor, dims: &[u64], expected: &[f64]) {
    assert_eq!(actual.shape(), &shape(dims), "{actual:?}");
    let got = values(actual);
    let close = |a: f64, e: f64| {
        (a.is_nan() && e.is_nan())
            || a.total_cmp(&e).is_eq()
            || (a - e).abs() <= 1e-6 * e.abs().max(1e-6)
    };
    assert!(
        got.len() == expected.len() && got.iter().zip(expected).all(|(&a, &e)| close(a, e)),
        "got {got:?}\nexpected {expected:?}"
    );
}
