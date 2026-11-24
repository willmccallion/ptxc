//! Loop IR: compute definitions, schedules as replayable traces, lowered loop nests and a CPU interpreter for them.

mod compute;
mod expr;
mod interp;
mod lower;
pub mod math;
mod schedule;
mod stmt;
mod text;
mod unroll;
mod verify;

#[cfg(test)]
mod test_kernels;

pub use compute::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeError, ComputeName, Constant, IndexVar,
    InfSign, IterKind, IterVar, IterVarId, OutputDef, ReduceOp, Scalar, TensorExpr, TensorName,
    TensorParam, TensorParamId, TieBreak, UnaryOp, VarName,
};
pub use expr::{Expr, ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
pub use interp::{RunError, run};
pub use schedule::{
    CacheRead, Guard, Loop, LoopAnnotation, LoopId, LoopKind, LoopTree, Open, Primitive, Schedule,
    ScheduleError, SplitFactor, StageCount, Tensorized, ThreadAxis, Trace, Validated,
};
pub use stmt::{
    Access, Buffer, BufferId, Condition, ForKind, FragmentShape, FragmentTile, KernelParam,
    KernelVar, LaunchShape, LoweredKernel, MemScope, Stmt, Swizzle, TileAccess, TileOrder,
    ValueExpr,
};
pub use text::{print_compute, print_kernel, print_schedule};
pub use unroll::unrolled;
pub use verify::{VerifyError, verify};
