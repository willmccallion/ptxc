//! Graph IR: an SSA dataflow graph in typed arenas, with op type rules, a verifier and a text format.

mod builder;
mod compact;
mod graph;
mod node;
mod ops;
mod partition;
mod rewrite;
mod tensor_type;
pub mod text;
mod traverse;
mod value;
mod verify;

pub use builder::GraphBuilder;
pub use graph::{Graph, NodeError};
pub use node::{Node, NodeId, NodeOutputIndex, OperandIndex};
pub use ops::{
    Activation, Antialias, ArgKind, ArgReduceAttrs, AttentionAttrs, AttentionLayout, AttentionMask,
    AxisPadding, BatchNormAttrs, BinaryOp, CastAttrs, Causality, ConvAttrs, CoordinateTransform,
    CumSumAttrs, DequantizeAttrs, DiagonalOffset, Dilation, F32Attr, GeluApproximation, GemmAttrs,
    GlobalPoolKind, Groups, HeadCount, Inclusion, InfDetection, InstanceNormAttrs, KeepDims,
    LayerNormAttrs, LayerNormKind, MatMulIntegerAttrs, ModKind, NearestRounding, NormOutputs, Op,
    OperandCount, Orientation, OutsideWeights, PadAttrs, PadCounting, PadMode, Padding, PoolAttrs,
    PoolAxis, PoolKind, Presence, QConvAttrs, QMatMulAttrs, QkOutput, QuantizeAttrs, ReduceAttrs,
    ReduceOp, RepeatCount, ResizeAttrs, ResizeAxis, ResizeMode, Rounding, ScanDirection,
    SliceAttrs, SliceAxis, SliceStep, SoftmaxAttrs, SoftmaxKind, SplitAttrs, StashType, Stride,
    TieBreak, TileAttrs, TransposeAttrs, TriangularPart, TriluAttrs, TypeError, TypeRule, UnaryOp,
    VariadicOp, WindowAxis, ZeroPoints,
};
pub use partition::{FusedGroup, GroupId, GroupKind, Partitioned};
pub use rewrite::{RewriteError, Rewriter};
pub use tensor_type::{Layout, TensorType};
pub use traverse::CycleError;
pub use value::{
    ConstantId, GraphInputIndex, GraphOutputIndex, Use, Value, ValueDef, ValueId, ValueName,
};
pub use verify::VerifyError;
