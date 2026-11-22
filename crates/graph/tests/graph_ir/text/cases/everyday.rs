//! Cases for the everyday op families.

use super::{AWKWARD, f, window};
use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::float::Overflow;
use ptxc_core::{Axis, DType, Dim};
use ptxc_graph::{
    Activation, ArgKind, ArgReduceAttrs, AxisPadding, BatchNormAttrs, BinaryOp, CastAttrs,
    ConvAttrs, DiagonalOffset, GeluApproximation, GemmAttrs, GlobalPoolKind, Groups, InfDetection,
    InstanceNormAttrs, KeepDims, LayerNormAttrs, LayerNormKind, ModKind, NormOutputs, Op,
    Orientation, PadAttrs, PadCounting, PadMode, Padding, PoolAttrs, PoolAxis, PoolKind,
    ReduceAttrs, ReduceOp, RepeatCount, Rounding, SliceAttrs, SliceAxis, SliceStep, SoftmaxAttrs,
    SoftmaxKind, SplitAttrs, StashType, TensorType, TieBreak, TileAttrs, TransposeAttrs,
    TriangularPart, TriluAttrs, UnaryOp, VariadicOp,
};

/// Every elementwise variant.
pub(in crate::text) fn elementwise() -> Vec<(Op, Vec<TensorType>)> {
    let float = f32_type(&[2]);
    let boolean = tensor_type(DType::Bool, &[2]);
    let integer = tensor_type(DType::I32, &[2]);
    let mut cases = Vec::new();
    for op in [
        UnaryOp::Abs,
        UnaryOp::Neg,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Reciprocal,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Round,
        UnaryOp::Sign,
        UnaryOp::Erf,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::IsNaN,
    ] {
        cases.push((Op::Unary(op), vec![float.clone()]));
    }
    cases.push((Op::Unary(UnaryOp::Not), vec![boolean.clone()]));
    for detect in
        [InfDetection::Both, InfDetection::Positive, InfDetection::Negative, InfDetection::Neither]
    {
        cases.push((Op::Unary(UnaryOp::IsInf(detect)), vec![float.clone()]));
    }
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Mod(ModKind::Fmod),
        BinaryOp::Equal,
        BinaryOp::Less,
        BinaryOp::LessOrEqual,
        BinaryOp::Greater,
        BinaryOp::GreaterOrEqual,
    ] {
        cases.push((Op::Binary(op), vec![float.clone(), float.clone()]));
    }
    for op in [BinaryOp::And, BinaryOp::Or, BinaryOp::Xor] {
        cases.push((Op::Binary(op), vec![boolean.clone(), boolean.clone()]));
    }
    cases.push((Op::Binary(BinaryOp::Mod(ModKind::Integer)), vec![integer.clone(), integer]));
    cases.push((Op::Binary(BinaryOp::Pow), vec![float.clone(), tensor_type(DType::I64, &[2])]));
    for op in [VariadicOp::Max, VariadicOp::Min, VariadicOp::Sum, VariadicOp::Mean] {
        cases.push((Op::Variadic(op), vec![float.clone(), float.clone(), f32_type(&[1])]));
    }
    cases.push((Op::Where, vec![boolean, float.clone(), float.clone()]));
    for (to, overflow) in [
        (DType::F16, Overflow::Saturate),
        (DType::F8E5M2, Overflow::NonFinite),
        (DType::F8E4M3Fn, Overflow::Saturate),
    ] {
        cases.push((Op::Cast(CastAttrs::new(to, overflow)), vec![float.clone()]));
    }
    cases
}

/// Every activation, with awkward attribute values, plus `Clip` and `PRelu`.
pub(in crate::text) fn activation() -> Vec<(Op, Vec<TensorType>)> {
    let [nan, negative_zero, tiny, infinite] = AWKWARD.map(f);
    let activations = [
        Activation::Relu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Softplus,
        Activation::Softsign,
        Activation::HardSwish,
        Activation::Mish,
        Activation::Gelu(GeluApproximation::None),
        Activation::Gelu(GeluApproximation::Tanh),
        Activation::LeakyRelu { alpha: nan },
        Activation::Elu { alpha: negative_zero },
        Activation::Celu { alpha: tiny },
        Activation::ThresholdedRelu { alpha: infinite },
        Activation::Selu { alpha: f(1.673_263_2), gamma: f(1.050_701) },
        Activation::HardSigmoid { alpha: f(0.2), beta: f(0.5) },
    ];
    let mut cases: Vec<_> = activations
        .into_iter()
        .map(|activation| (Op::Activation(activation), vec![f32_type(&[2])]))
        .collect();
    cases.push((Op::Clip, vec![f32_type(&[2]), f32_type(&[]), f32_type(&[])]));
    cases.push((Op::PRelu, vec![f32_type(&[2, 3]), f32_type(&[3])]));
    cases
}

/// Matrix products, convolutions and pools.
pub(in crate::text) fn compute() -> Vec<(Op, Vec<TensorType>)> {
    let gemm = GemmAttrs {
        a: Orientation::Transposed,
        b: Orientation::AsIs,
        alpha: f(0.5),
        beta: f(-2.0),
    };
    let conv = ConvAttrs {
        axes: vec![window(2, 1, 1, 0), window(1, 2, 0, 2)],
        groups: Groups::new(2).unwrap(),
    };
    let pool_axis = |kernel| PoolAxis { kernel: Dim::new(kernel), window: window(2, 1, 0, 1) };
    vec![
        (Op::MatMul, vec![f32_type(&[2, 3]), f32_type(&[3, 4])]),
        (Op::Gemm(gemm), vec![f32_type(&[3, 2]), f32_type(&[3, 4]), f32_type(&[4])]),
        (Op::Conv(conv), vec![f32_type(&[1, 4, 7, 7]), f32_type(&[6, 2, 3, 3]), f32_type(&[6])]),
        (
            Op::Pool(PoolAttrs {
                kind: PoolKind::Max,
                axes: vec![pool_axis(3), pool_axis(2)],
                rounding: Rounding::Ceil,
            }),
            vec![f32_type(&[1, 2, 6, 6])],
        ),
        (
            Op::Pool(PoolAttrs {
                kind: PoolKind::Average(PadCounting::Include),
                axes: vec![pool_axis(2)],
                rounding: Rounding::Floor,
            }),
            vec![f32_type(&[1, 2, 6])],
        ),
        (Op::GlobalPool(GlobalPoolKind::Max), vec![f32_type(&[1, 2, 3, 3])]),
        (Op::GlobalPool(GlobalPoolKind::Average), vec![f32_type(&[1, 2, 3, 3])]),
    ]
}

/// Reductions and normalisations.
pub(in crate::text) fn reduce_and_norm() -> Vec<(Op, Vec<TensorType>)> {
    let input = f32_type(&[2, 3, 4]);
    let ops = [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Prod,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::SumSquare,
        ReduceOp::L1,
        ReduceOp::L2,
        ReduceOp::LogSum,
        ReduceOp::LogSumExp,
    ];
    let mut cases: Vec<_> = ops
        .into_iter()
        .zip([KeepDims::Keep, KeepDims::Drop].into_iter().cycle())
        .map(|(op, keep_dims)| {
            (
                Op::Reduce(ReduceAttrs { op, axes: vec![Axis::new(2), Axis::new(0)], keep_dims }),
                vec![input.clone()],
            )
        })
        .collect();
    for (kind, tie) in [(ArgKind::Max, TieBreak::Last), (ArgKind::Min, TieBreak::First)] {
        let attrs = ArgReduceAttrs { kind, axis: Axis::new(1), keep_dims: KeepDims::Drop, tie };
        cases.push((Op::ArgReduce(attrs), vec![input.clone()]));
    }
    for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
        cases.push((Op::Softmax(SoftmaxAttrs { kind, axis: Axis::new(1) }), vec![input.clone()]));
    }
    let layer = |kind, stash| {
        Op::LayerNorm(LayerNormAttrs { kind, axis: Axis::new(2), epsilon: f(1e-5), stash })
    };
    cases.push((
        layer(LayerNormKind::Layer(NormOutputs::Normalized), StashType::F32),
        vec![input.clone(), f32_type(&[4])],
    ));
    cases.push((
        layer(LayerNormKind::Layer(NormOutputs::WithStatistics), StashType::Bf16),
        vec![input.clone(), f32_type(&[4]), f32_type(&[4])],
    ));
    cases.push((layer(LayerNormKind::Rms, StashType::F32), vec![input.clone(), f32_type(&[4])]));
    let channel = f32_type(&[3]);
    cases.push((
        Op::BatchNorm(BatchNormAttrs { epsilon: f(1e-3) }),
        vec![input.clone(), channel.clone(), channel.clone(), channel.clone(), channel.clone()],
    ));
    cases.push((
        Op::InstanceNorm(InstanceNormAttrs { epsilon: f(2e-5) }),
        vec![input, channel.clone(), channel],
    ));
    cases
}

/// Shape ops and indexing.
pub(in crate::text) fn shape_ops() -> Vec<(Op, Vec<TensorType>)> {
    let input = f32_type(&[2, 3, 4]);
    let slice = SliceAttrs {
        axes: vec![SliceAxis {
            axis: Axis::new(2),
            start: 3,
            step: SliceStep::new(-2).unwrap(),
            length: Dim::new(2),
        }],
    };
    let pads = |mode| PadAttrs {
        mode,
        axes: vec![AxisPadding { begin: Padding::new(1), end: Padding::new(0) }; 3],
    };
    let mut cases = vec![
        (Op::Reshape(shape(&[6, 4])), vec![input.clone()]),
        (Op::Expand(shape(&[5, 1, 1, 1])), vec![input.clone()]),
        (
            Op::Transpose(TransposeAttrs {
                permutation: vec![Axis::new(2), Axis::new(0), Axis::new(1)],
            }),
            vec![input.clone()],
        ),
        (
            Op::Tile(TileAttrs { repeats: [2, 1, 3].map(RepeatCount::new).to_vec() }),
            vec![input.clone()],
        ),
        (Op::Concat(Axis::new(1)), vec![input.clone(), f32_type(&[2, 5, 4])]),
        (Op::Slice(slice), vec![input.clone()]),
        (
            Op::Split(SplitAttrs { axis: Axis::new(1), sizes: vec![Dim::new(1), Dim::new(2)] }),
            vec![input.clone()],
        ),
        (Op::Pad(pads(PadMode::Constant)), vec![input.clone(), f32_type(&[])]),
        (
            Op::Trilu(TriluAttrs {
                part: TriangularPart::Lower,
                diagonal: DiagonalOffset::new(-1),
            }),
            vec![input.clone()],
        ),
        (Op::Gather(Axis::new(1)), vec![input.clone(), tensor_type(DType::I64, &[5])]),
        (
            Op::GatherElements(Axis::new(2)),
            vec![input.clone(), tensor_type(DType::I32, &[2, 3, 1])],
        ),
    ];
    for mode in [PadMode::Reflect, PadMode::Edge, PadMode::Wrap] {
        cases.push((Op::Pad(pads(mode)), vec![input.clone()]));
    }
    cases
}
