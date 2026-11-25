//! Cases for quantization, resize and attention.

use super::{f, per_axis, per_tensor, window};
use crate::common::{f32_type, tensor_type};
use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim, QuantDType};
use ptxc_graph::{
    Antialias, AttentionAttrs, AttentionLayout, AttentionMask, Causality, ConvAttrs,
    CoordinateTransform, DequantizeAttrs, Groups, HeadCount, NearestRounding, Op, OutsideWeights,
    Presence, QConvAttrs, QMatMulAttrs, QMatMulOutput, QkOutput, QuantizeAttrs, ResizeAttrs,
    ResizeAxis, ResizeMode, TensorType,
};

/// Quantization ops.
pub(in crate::text) fn quantization() -> Vec<(Op, Vec<TensorType>)> {
    let qconv = QConvAttrs {
        conv: ConvAttrs { axes: vec![window(1, 1, 0, 0); 2], groups: Groups::new(1).unwrap() },
        input: per_tensor(QuantDType::U8),
        weights: per_axis(QuantDType::I8, 0, 4),
        output: per_tensor(QuantDType::U8),
    };
    let qmatmul = QMatMulAttrs {
        a: per_tensor(QuantDType::U8),
        b: per_axis(QuantDType::I8, 1, 4),
        output: QMatMulOutput::Quantized(per_tensor(QuantDType::U8)),
    };
    vec![
        (
            Op::Quantize(QuantizeAttrs::new(per_tensor(QuantDType::I8), Overflow::Saturate)),
            vec![f32_type(&[2, 3])],
        ),
        (
            Op::Quantize(QuantizeAttrs::new(per_axis(QuantDType::U8, 1, 3), Overflow::Saturate)),
            vec![f32_type(&[2, 3])],
        ),
        (
            Op::Quantize(QuantizeAttrs::new(per_tensor(QuantDType::F8E5M2), Overflow::NonFinite)),
            vec![f32_type(&[2, 3])],
        ),
        (
            Op::Dequantize(DequantizeAttrs {
                params: per_tensor(QuantDType::I32),
                output: DType::F16,
            }),
            vec![tensor_type(DType::I32, &[4])],
        ),
        (
            Op::QMatMul(qmatmul),
            vec![tensor_type(DType::U8, &[2, 3]), tensor_type(DType::I8, &[3, 4])],
        ),
        (
            Op::QConv(qconv),
            vec![
                tensor_type(DType::U8, &[1, 3, 5, 5]),
                tensor_type(DType::I8, &[4, 3, 3, 3]),
                tensor_type(DType::I32, &[4]),
            ],
        ),
    ]
}

/// Every resize mode and coordinate transform shape.
pub(in crate::text) fn resize() -> Vec<(Op, Vec<TensorType>)> {
    let resize = |mode, coordinates, antialias| {
        let axes = vec![
            ResizeAxis { size: Dim::new(1), scale: f(1.0) },
            ResizeAxis { size: Dim::new(8), scale: f(2.5) },
        ];
        (Op::Resize(ResizeAttrs { mode, coordinates, antialias, axes }), vec![f32_type(&[1, 4])])
    };
    let crop = CoordinateTransform::TfCropAndResize {
        roi: vec![[f(0.0), f(1.0)], [f(0.25), f(0.75)]],
        extrapolation: f(-1.0),
    };
    let cubic = ResizeMode::Cubic { coefficient: f(-0.75), outside: OutsideWeights::Exclude };
    vec![
        resize(
            ResizeMode::Nearest(NearestRounding::RoundPreferCeil),
            CoordinateTransform::Asymmetric,
            Antialias::Off,
        ),
        resize(ResizeMode::Linear, CoordinateTransform::PytorchHalfPixel, Antialias::On),
        resize(cubic, CoordinateTransform::HalfPixelSymmetric, Antialias::Off),
        resize(ResizeMode::Linear, crop, Antialias::Off),
        resize(ResizeMode::Linear, CoordinateTransform::HalfPixel, Antialias::Off),
        resize(
            ResizeMode::Nearest(NearestRounding::Floor),
            CoordinateTransform::AlignCorners,
            Antialias::Off,
        ),
    ]
}

/// Attention in both layouts, with and without every optional part.
pub(in crate::text) fn attention() -> Vec<(Op, Vec<TensorType>)> {
    let full = AttentionAttrs {
        layout: AttentionLayout::Packed {
            q_heads: HeadCount::new(4).unwrap(),
            kv_heads: HeadCount::new(2).unwrap(),
        },
        mask: AttentionMask::Boolean,
        causality: Causality::Causal,
        past: Presence::Present,
        present: Presence::Present,
        nonpad_lengths: Presence::Absent,
        qk_output: Some(QkOutput::Masked),
        scale: f(0.125),
        softcap: Some(f(30.0)),
        softmax_precision: DType::F32,
    };
    let plain = AttentionAttrs {
        layout: AttentionLayout::Heads,
        mask: AttentionMask::None,
        causality: Causality::Full,
        past: Presence::Absent,
        present: Presence::Absent,
        qk_output: None,
        softcap: None,
        ..full
    };
    let full_operands = vec![
        f32_type(&[2, 5, 64]),
        f32_type(&[2, 7, 32]),
        f32_type(&[2, 7, 32]),
        tensor_type(DType::Bool, &[5, 10]),
        f32_type(&[2, 2, 3, 16]),
        f32_type(&[2, 2, 3, 16]),
    ];
    let plain_operands =
        vec![f32_type(&[2, 4, 5, 16]), f32_type(&[2, 2, 7, 16]), f32_type(&[2, 2, 7, 16])];
    vec![(Op::Attention(full), full_operands), (Op::Attention(plain), plain_operands)]
}
