//! Compute definitions: what a kernel computes, without saying how.

mod builder;
mod check;
mod compute_def;
mod error;
mod iter_var;
mod names;
mod rebuild;
mod rfactor;
mod scalar;
mod scalar_op;
mod tensor_expr;
pub(crate) mod types;

pub use builder::ComputeBuilder;
pub use compute_def::{ComputeDef, IndexVar, OutputDef, TensorParam};
pub use error::ComputeError;
pub use iter_var::{IterKind, IterVar, IterVarId};
pub use names::{ComputeName, TensorName, VarName};
pub use rfactor::{CombineInput, RfactorError, Rfactored, Slices, rfactor};
pub use scalar::{Constant, Scalar};
pub use scalar_op::{ArgKind, BinaryOp, InfSign, ReduceOp, TieBreak, UnaryOp};
pub use tensor_expr::{TensorExpr, TensorParamId};
