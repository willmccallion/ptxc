//! Graph IR: an SSA dataflow graph in typed arenas, with op type rules, a verifier and a text format.

mod builder;
mod compact;
mod graph;
mod node;
mod ops;
mod partition;
mod rewrite;
mod tensor_type;
mod text;
mod traverse;
mod value;
mod verify;

pub use builder::GraphBuilder;
pub use graph::{Graph, NodeError};
pub use node::{Node, NodeId, NodeOutputIndex, OperandIndex};
pub use ops::{
    Activation, ArgKind, ArgReduceAttrs, BatchNormAttrs, BinaryOp, CastAttrs, ConvAttrs, Dilation,
    F32Attr, GeluApproximation, GemmAttrs, GlobalPoolKind, Groups, InfDetection, InstanceNormAttrs,
    KeepDims, LayerNormAttrs, LayerNormKind, ModKind, NormOutputs, Op, OperandCount, Orientation,
    PadCounting, Padding, PoolAttrs, PoolAxis, PoolKind, ReduceAttrs, ReduceOp, Rounding,
    SoftmaxAttrs, SoftmaxKind, StashType, Stride, TieBreak, TypeError, TypeRule, UnaryOp,
    VariadicOp, WindowAxis,
};
pub use rewrite::{RewriteError, Rewriter};
pub use tensor_type::{Layout, TensorType};
pub use traverse::CycleError;
pub use value::{
    ConstantId, GraphInputIndex, GraphOutputIndex, Use, Value, ValueDef, ValueId, ValueName,
};
pub use verify::VerifyError;
