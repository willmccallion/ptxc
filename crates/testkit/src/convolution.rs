//! Square 2-D convolutions as compute definitions, which the loop IR and
//! schedule tests share.
//!
//! They sum `i32`s, exactly in any order, and are built as `ptxc-lower`
//! builds a convolution: each tap reads the input through a load padded
//! with zero.
//!
//! NOTE: test code shared as a library. A builder handed a case it cannot
//! build has a bug, so it panics rather than returning an error (D52).

#![expect(
    clippy::unwrap_used,
    clippy::missing_panics_doc,
    reason = "test builders panic on cases they build wrong (D52)"
)]

use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprArena, ExprId, IterVarId, ReduceOp,
    Scalar, TensorExpr, TensorName, TensorParamId, VarName,
};
use std::sync::Arc;

/// A square 2-D convolution.
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    /// The images.
    pub batch: u64,
    /// The input's rows, and its columns.
    pub size: u64,
    /// The input channels each group reads.
    pub channels: u64,
    /// The output channels.
    pub features: u64,
    /// The kernel's rows, and its columns.
    pub kernel: u64,
    /// The step between windows.
    pub stride: u64,
    /// The step between a window's taps.
    pub dilation: u64,
    /// The padding on every side.
    pub pad: u64,
    /// The channel groups, which divide the output channels.
    pub groups: u64,
}

/// Where a convolution's channels lie in its input, weights and output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channels {
    /// Innermost: `x[n, h, w, c]`, `w[m, r, s, c]` and `out[n, p, q, m]`.
    Last,
    /// After the batch or output channel: `x[n, c, h, w]`,
    /// `w[m, c, r, s]` and `out[n, m, p, q]`, as ONNX lays them out.
    First,
}

impl Conv {
    /// The output size; the padded input holds at least one window.
    pub const fn output(self) -> u64 {
        let span = self.dilation * (self.kernel - 1) + 1;
        (self.size + 2 * self.pad - span) / self.stride + 1
    }
}

fn signed(value: u64) -> i64 {
    i64::try_from(value).unwrap()
}

/// `out = sum over r, s, c of x * w`, `x` zero where a tap reads its
/// padding, channels laid out as `channels` says.
pub fn conv(geometry: Conv, channels: Channels) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("conv"));
    let Conv { batch, size, channels: depth, features, kernel, groups, .. } = geometry;
    let inputs = groups * depth;
    let (x_shape, w_shape) = match channels {
        Channels::Last => ([batch, size, size, inputs], [features, kernel, kernel, depth]),
        Channels::First => ([batch, inputs, size, size], [features, depth, kernel, kernel]),
    };
    let image = compute.input(TensorName::new("x"), DType::I32, shape(&x_shape));
    let filter = compute.input(TensorName::new("w"), DType::I32, shape(&w_shape));
    let output = Dim::new(geometry.output());
    let mut spatial = |name: &str, extent: Dim| compute.spatial(VarName::new(name), extent).1;
    let sample = spatial("n", Dim::new(batch));
    let (row, column, feature) = match channels {
        Channels::Last => {
            let (row, column) = (spatial("p", output), spatial("q", output));
            (row, column, spatial("m", Dim::new(features)))
        }
        Channels::First => {
            let feature = spatial("m", Dim::new(features));
            (spatial("p", output), spatial("q", output), feature)
        }
    };
    let (channels_axis, channel) = compute.reduce_axis(VarName::new("c"), Dim::new(depth));
    let (tap_rows, tap_row) = compute.reduce_axis(VarName::new("r"), Dim::new(kernel));
    let (tap_columns, tap_column) = compute.reduce_axis(VarName::new("s"), Dim::new(kernel));
    let read_channel = group_channel(compute.exprs(), geometry, (feature, channel));
    let row_read = tap(compute.exprs(), geometry, (row, tap_row));
    let column_read = tap(compute.exprs(), geometry, (column, tap_column));
    let (x_index, w_index) = match channels {
        Channels::Last => (
            vec![sample, row_read, column_read, read_channel],
            vec![feature, tap_row, tap_column, channel],
        ),
        Channels::First => (
            vec![sample, read_channel, row_read, column_read],
            vec![feature, channel, tap_row, tap_column],
        ),
    };
    let padding = Some(Scalar::zero(DType::I32));
    let image = TensorExpr::Load { tensor: image, index: x_index, padding };
    let body = binary(BinaryOp::Mul, image, TensorExpr::load(filter, w_index));
    let axes = vec![channels_axis, tap_rows, tap_columns];
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes, body: Box::new(body) };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

/// Inputs of `compute`, a `conv`, counting up from distinct starts, so a
/// read of the wrong element shows.
pub fn conv_inputs(compute: &ComputeDef) -> Vec<HostTensor> {
    let starts = [1, -9];
    compute
        .inputs()
        .iter()
        .zip(starts)
        .map(|(param, start)| {
            let count = i32::try_from(param.shape.element_count().get()).unwrap();
            let values: Vec<i32> = (0..count).map(|value| start + value % 23).collect();
            HostTensor::new(param.shape.clone(), TensorData::I32(values)).unwrap()
        })
        .collect()
}

/// The input of `conv`, the first.
pub fn x() -> TensorParamId {
    ptxc_core::Idx::new(0)
}

/// The weights of `conv`, the second.
pub fn w() -> TensorParamId {
    ptxc_core::Idx::new(1)
}

/// The axis of `compute` named `name`: `conv`'s are `n`, `m`, `p` and `q`,
/// spatial, and `c`, `r` and `s`, summed.
pub fn axis(compute: &ComputeDef, name: &str) -> IterVarId {
    let found = compute.iter_vars().iter_enumerated().find(|(_, axis)| axis.name.as_str() == name);
    found.unwrap().0
}

/// The axes of `compute` named `names`, in order.
pub fn axes(compute: &ComputeDef, names: &[&str]) -> Vec<IterVarId> {
    names.iter().map(|name| axis(compute, name)).collect()
}

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

fn binary(op: BinaryOp, lhs: TensorExpr, rhs: TensorExpr) -> TensorExpr {
    TensorExpr::Binary(op, Box::new(lhs), Box::new(rhs))
}

/// The input channel channel `channel` of output channel `feature`'s group
/// reads: `channel` with one group, as `ptxc-lower` reads it.
fn group_channel(
    exprs: &mut ExprArena,
    geometry: Conv,
    (feature, channel): (ExprId, ExprId),
) -> ExprId {
    if geometry.groups == 1 {
        return channel;
    }
    let per_group = exprs.constant(signed(geometry.features / geometry.groups));
    let group = exprs.floor_div(feature, per_group);
    let width = exprs.constant(signed(geometry.channels));
    let first = exprs.mul(group, width);
    exprs.add(first, channel)
}

/// Where tap `tap` of output `out` reads along one spatial axis: in the
/// padding where it lies outside the input.
fn tap(exprs: &mut ExprArena, geometry: Conv, (out, tap): (ExprId, ExprId)) -> ExprId {
    let [stride, dilation] =
        [geometry.stride, geometry.dilation].map(|value| exprs.constant(signed(value)));
    let start = exprs.mul(out, stride);
    let offset = exprs.mul(tap, dilation);
    let padded = exprs.add(start, offset);
    let minus_pad = exprs.constant(-signed(geometry.pad));
    exprs.add(padded, minus_pad)
}
