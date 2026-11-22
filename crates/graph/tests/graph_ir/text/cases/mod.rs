//! One graph per op variant and attribute shape, for the round-trip test.

mod everyday;
mod special;

pub(super) use everyday::{activation, compute, elementwise, reduce_and_norm, shape_ops};
pub(super) use special::{attention, quantization, resize};

use ptxc_core::{Affine, Axis, Granularity, QuantDType, QuantParams, Scale, ZeroPoint};
use ptxc_graph::{Dilation, F32Attr, Padding, Stride, WindowAxis};

/// Float attribute values that are easy to lose: a NaN with a payload,
/// negative zero, a tiny value and infinity.
pub(super) const AWKWARD: [f32; 4] = [f32::from_bits(0x7fc0_0123), -0.0, 1e-30, f32::INFINITY];

pub(super) const fn f(value: f32) -> F32Attr {
    F32Attr::new(value)
}

pub(super) const fn window(stride: u64, dilation: u64, begin: u64, end: u64) -> WindowAxis {
    WindowAxis {
        stride: Stride::new(stride).unwrap(),
        dilation: Dilation::new(dilation).unwrap(),
        pad_begin: Padding::new(begin),
        pad_end: Padding::new(end),
    }
}

pub(super) fn params(dtype: QuantDType, granularity: Granularity) -> QuantParams {
    QuantParams::new(dtype, granularity).unwrap()
}

pub(super) fn per_tensor(dtype: QuantDType) -> QuantParams {
    params(
        dtype,
        Granularity::PerTensor(Affine { scale: Scale::new(0.5), zero_point: ZeroPoint::new(0) }),
    )
}

pub(super) fn per_axis(dtype: QuantDType, axis: usize, count: usize) -> QuantParams {
    let map = Affine { scale: Scale::new(0.25), zero_point: ZeroPoint::new(1) };
    params(dtype, Granularity::PerAxis { axis: Axis::new(axis), maps: vec![map; count] })
}
