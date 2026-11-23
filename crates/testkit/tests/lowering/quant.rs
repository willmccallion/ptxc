//! Quantization onto every grid type, per tensor and per axis, and the
//! quantized products, which compare within one step: their rescaling is
//! f32 where the reference's is f64.

use crate::cases::single_node;
use crate::harness::check_within;
use crate::values::{Domain, axis_dims, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::float::Overflow;
use ptxc_core::{
    Affine, Axis, DType, Granularity, HostTensor, QuantDType, QuantParams, Scale, ZeroPoint,
};
use ptxc_graph::{
    ConvAttrs, DequantizeAttrs, Dilation, Groups, Op, Padding, QConvAttrs, QMatMulAttrs,
    QuantizeAttrs, Stride, WindowAxis,
};
use ptxc_testkit::compare::Tolerance;

const GRIDS: [QuantDType; 4] =
    [QuantDType::I8, QuantDType::U8, QuantDType::F8E4M3Fn, QuantDType::F8E5M2];
const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::Bf16];

/// A map onto `dtype`'s grid: a scale from a few magnitudes, and a zero
/// point anywhere in the grid (0 for FP8 and `i32`, as ONNX requires).
fn affine(dtype: QuantDType) -> impl Strategy<Value = Affine> {
    let zero_point = match dtype {
        QuantDType::I8 => (-128_i32..=127).boxed(),
        QuantDType::U8 => (0_i32..=255).boxed(),
        _ => Just(0).boxed(),
    };
    (select(vec![0.05_f32, 0.5, 1.0, 3.0]), zero_point).prop_map(|(scale, zero_point)| Affine {
        scale: Scale::new(scale),
        zero_point: ZeroPoint::new(zero_point),
    })
}

/// Parameters on `dtype`'s grid: one map, or one per index along `axis`
/// if given, with `extent` indices.
fn params(dtype: QuantDType, along: Option<(usize, u64)>) -> BoxedStrategy<QuantParams> {
    match along {
        None => affine(dtype)
            .prop_map(move |map| QuantParams::new(dtype, Granularity::PerTensor(map)).unwrap())
            .boxed(),
        Some((axis, extent)) => vec(affine(dtype), usize::try_from(extent).unwrap())
            .prop_map(move |maps| {
                let granularity = Granularity::PerAxis { axis: Axis::new(axis), maps };
                QuantParams::new(dtype, granularity).unwrap()
            })
            .boxed(),
    }
}

/// Per tensor, or per axis along a random axis of `dims`.
fn params_for(dtype: QuantDType, dims: &[u64]) -> BoxedStrategy<QuantParams> {
    let dims = dims.to_vec();
    let rank = dims.len();
    prop_oneof![
        params(dtype, None),
        (0..rank).prop_flat_map(move |axis| params(dtype, Some((axis, dims[axis])))),
    ]
    .boxed()
}

fn overflow() -> impl Strategy<Value = Overflow> {
    prop_oneof![Just(Overflow::Saturate), Just(Overflow::NonFinite)]
}

fn quantize_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (select(GRIDS.to_vec()), select(FLOATS.to_vec()), axis_dims(), overflow())
        .prop_flat_map(|(grid, dtype, dims, overflow)| {
            let input = tensor(dtype, dims.clone(), Domain::Any);
            (params_for(grid, &dims), input, Just(overflow))
        })
        .prop_map(|(params, input, overflow)| {
            (Op::Quantize(QuantizeAttrs::new(params, overflow)), vec![input])
        })
}

fn dequantize_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let grids = [GRIDS.to_vec(), vec![QuantDType::I32]].concat();
    (select(grids), select(FLOATS.to_vec()), axis_dims())
        .prop_flat_map(|(grid, output, dims)| {
            let input = tensor(grid.dtype(), dims.clone(), Domain::Any);
            (params_for(grid, &dims), Just(output), input)
        })
        .prop_map(|(params, output, input)| {
            (Op::Dequantize(DequantizeAttrs { params, output }), vec![input])
        })
}

fn integer_grid() -> impl Strategy<Value = QuantDType> {
    prop_oneof![Just(QuantDType::I8), Just(QuantDType::U8)]
}

/// `[rows, inner] x [inner, columns]`, `a`'s maps maybe per row and `b`'s
/// maybe per column.
fn qmatmul_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let extent = 0_u64..=3;
    (
        integer_grid(),
        integer_grid(),
        integer_grid(),
        [extent.clone(), extent.clone(), extent],
        any::<[bool; 2]>(),
    )
        .prop_flat_map(|(a, b, y, [rows, inner, columns], per_axis)| {
            let a_params = params(a, per_axis[0].then_some((0, rows)));
            let b_params = params(b, per_axis[1].then_some((1, columns)));
            let lhs = tensor(a.dtype(), vec![rows, inner], Domain::Any);
            let rhs = tensor(b.dtype(), vec![inner, columns], Domain::Any);
            (a_params, b_params, params(y, None), lhs, rhs)
        })
        .prop_map(|(a, b, output, lhs, rhs)| {
            (Op::QMatMul(QMatMulAttrs { a, b, output }), vec![lhs, rhs])
        })
}

const fn window(stride: u64, pad: u64) -> WindowAxis {
    WindowAxis {
        stride: Stride::new(stride).unwrap(),
        dilation: Dilation::new(1).unwrap(),
        pad_begin: Padding::new(pad),
        pad_end: Padding::new(pad),
    }
}

/// A 1-D convolution of 1 or 2 channels, weights maybe per output channel,
/// and an optional `i32` bias.
fn qconv_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let shape = (1_u64..=2, 1_u64..=2, 2_u64..=5, 1_u64..=2, 1_u64..=2, 0_u64..=1);
    (integer_grid(), integer_grid(), integer_grid(), shape, any::<[bool; 2]>())
        .prop_flat_map(|(x, w, y, (channels, features, length, kernel, stride, pad), flags)| {
            let conv =
                ConvAttrs { axes: vec![window(stride, pad)], groups: Groups::new(1).unwrap() };
            let weights_params = params(w, flags[0].then_some((0, features)));
            let bias =
                vec(tensor(DType::I32, vec![features], Domain::Moderate), usize::from(flags[1]));
            (
                Just(conv),
                params(x, None),
                weights_params,
                params(y, None),
                tensor(x.dtype(), vec![1, channels, length], Domain::Any),
                tensor(w.dtype(), vec![features, channels, kernel], Domain::Any),
                bias,
            )
        })
        .prop_map(|(conv, input, weights, output, x, w, bias)| {
            let attrs = QConvAttrs { conv, input, weights, output };
            (Op::QConv(attrs), [vec![x, w], bias].concat())
        })
}

fn check(
    (op, inputs): (Op, Vec<HostTensor>),
    tolerance: fn(DType) -> Tolerance,
) -> Result<(), TestCaseError> {
    let graph = single_node(op, &inputs);
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn quantize_agrees_with_the_reference(case in quantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn dequantize_agrees_with_the_reference(case in dequantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn qmatmul_agrees_with_the_reference(case in qmatmul_cases()) {
        check(case, |_| Tolerance::QUANTIZED)?;
    }

    #[test]
    fn qconv_agrees_with_the_reference(case in qconv_cases()) {
        check(case, |_| Tolerance::QUANTIZED)?;
    }
}
