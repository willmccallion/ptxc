//! Loop IR: compute definitions, schedules as replayable traces, lowered loop nests and a CPU interpreter for them.

mod compute;
mod expr;
mod interp;
mod lower;
mod schedule;
mod stmt;
mod text;
mod verify;

pub use compute::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeError, ComputeName, IndexVar, InfSign,
    IterKind, IterVar, IterVarId, OutputDef, ReduceOp, Scalar, TensorExpr, TensorName, TensorParam,
    TensorParamId, TieBreak, UnaryOp, VarName,
};
pub use expr::{Expr, ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};
pub use schedule::{
    Guard, Loop, LoopAnnotation, LoopId, LoopKind, LoopTree, Open, Primitive, Schedule,
    ScheduleError, SplitFactor, ThreadAxis, Trace,
};
pub use text::{print_compute, print_schedule};
