//! `ComputeDef`: iteration domain, inputs and output expressions.

use super::iter_var::{IterVar, IterVarId};
use super::names::{ComputeName, TensorName, VarName};
use super::tensor_expr::{TensorExpr, TensorParamId};
use crate::expr::{ExprArena, ExprId, VarId};
use ptxc_core::{DType, IndexVec, Shape};

/// An input tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorParam {
    /// The name it prints with.
    pub name: TensorName,
    /// The element type.
    pub dtype: DType,
    /// The shape, row-major.
    pub shape: Shape,
}

/// What a variable in an index expression stands for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum IndexVar {
    /// The current value of an iteration axis.
    Iter(IterVarId),
    /// An integer read from an index tensor, as a gather uses (D4).
    Loaded {
        /// The name it prints with.
        name: VarName,
        /// The `i32` or `i64` tensor read.
        tensor: TensorParamId,
        /// One index per axis, over variables defined before this one.
        index: Vec<ExprId>,
    },
}

/// An output tensor: its name, its element type, and each element's value.
/// Its shape is the spatial axes' extents, in order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OutputDef {
    /// The name it prints with.
    pub name: TensorName,
    /// The element type `expr` has.
    pub dtype: DType,
    /// Each element's value.
    pub expr: TensorExpr,
}

/// What a kernel computes, without saying how, as Halide's and TVM's
/// compute definitions.
///
/// Every output element is an expression over the spatial axes, with
/// reductions over reduction axes. Only `ComputeBuilder::finish` makes one,
/// after type-checking it, so a `ComputeDef` is well-formed.
#[derive(Clone, Debug)]
pub struct ComputeDef {
    pub(super) name: ComputeName,
    pub(super) exprs: ExprArena,
    pub(super) vars: IndexVec<VarId, IndexVar>,
    pub(super) iter_vars: IndexVec<IterVarId, IterVar>,
    pub(super) inputs: IndexVec<TensorParamId, TensorParam>,
    pub(super) outputs: Vec<OutputDef>,
    pub(super) output_shape: Shape,
}

impl ComputeDef {
    /// The definition's name.
    pub const fn name(&self) -> &ComputeName {
        &self.name
    }

    /// The index expressions the definition reads.
    pub const fn exprs(&self) -> &ExprArena {
        &self.exprs
    }

    /// What each index variable stands for.
    pub const fn vars(&self) -> &IndexVec<VarId, IndexVar> {
        &self.vars
    }

    /// The iteration axes.
    pub const fn iter_vars(&self) -> &IndexVec<IterVarId, IterVar> {
        &self.iter_vars
    }

    /// The input tensors.
    pub const fn inputs(&self) -> &IndexVec<TensorParamId, TensorParam> {
        &self.inputs
    }

    /// The outputs, which share the spatial axes.
    pub fn outputs(&self) -> &[OutputDef] {
        &self.outputs
    }

    /// The shape every output has: the spatial axes' extents.
    pub const fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    /// The name index variable `var` prints with.
    pub fn var_name(&self, var: VarId) -> &VarName {
        match &self.vars[var] {
            IndexVar::Iter(axis) => &self.iter_vars[*axis].name,
            IndexVar::Loaded { name, .. } => name,
        }
    }
}
