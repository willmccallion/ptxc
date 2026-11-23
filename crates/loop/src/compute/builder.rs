//! `ComputeBuilder`: builds a `ComputeDef` and checks it once.

use super::check::Checker;
use super::compute_def::{ComputeDef, IndexVar, OutputDef, TensorParam};
use super::error::ComputeError;
use super::iter_var::{IterKind, IterVar, IterVarId};
use super::names::{ComputeName, TensorName, VarName};
use super::tensor_expr::{TensorExpr, TensorParamId};
use crate::expr::{ExprArena, ExprId, VarId};
use ptxc_core::{DType, Dim, IndexVec, Shape};

/// Builds a compute definition: inputs, axes, index variables and outputs,
/// with index arithmetic in its `ExprArena`.
#[derive(Debug)]
pub struct ComputeBuilder {
    name: ComputeName,
    exprs: ExprArena,
    vars: IndexVec<VarId, IndexVar>,
    iter_vars: IndexVec<IterVarId, IterVar>,
    inputs: IndexVec<TensorParamId, TensorParam>,
    outputs: Vec<(TensorName, TensorExpr)>,
}

impl ComputeBuilder {
    /// An empty definition named `name`.
    pub fn new(name: ComputeName) -> Self {
        Self {
            name,
            exprs: ExprArena::new(),
            vars: IndexVec::new(),
            iter_vars: IndexVec::new(),
            inputs: IndexVec::new(),
            outputs: Vec::new(),
        }
    }

    /// Adds an input tensor.
    pub fn input(&mut self, name: TensorName, dtype: DType, shape: Shape) -> TensorParamId {
        self.inputs.push(TensorParam { name, dtype, shape })
    }

    /// Adds a spatial axis: the next output axis.
    pub fn spatial(&mut self, name: VarName, extent: Dim) -> (IterVarId, ExprId) {
        self.axis(name, extent, IterKind::Spatial)
    }

    /// Adds a reduction axis.
    pub fn reduce_axis(&mut self, name: VarName, extent: Dim) -> (IterVarId, ExprId) {
        self.axis(name, extent, IterKind::Reduce)
    }

    fn axis(&mut self, name: VarName, extent: Dim, kind: IterKind) -> (IterVarId, ExprId) {
        let axis = self.iter_vars.push(IterVar { name, extent, kind });
        let var = self.vars.push(IndexVar::Iter(axis));
        (axis, self.exprs.var(var))
    }

    /// Adds an index variable holding `tensor`'s integer at `index`.
    pub fn loaded_index(
        &mut self,
        name: VarName,
        tensor: TensorParamId,
        index: Vec<ExprId>,
    ) -> ExprId {
        let var = self.vars.push(IndexVar::Loaded { name, tensor, index });
        self.exprs.var(var)
    }

    /// The arena for building index expressions.
    pub const fn exprs(&mut self) -> &mut ExprArena {
        &mut self.exprs
    }

    /// Adds an output.
    pub fn output(&mut self, name: TensorName, expr: TensorExpr) {
        self.outputs.push((name, expr));
    }

    /// The definition, if it is well-formed: every tensor, axis and index
    /// variable exists, loads have one index per axis, element types fit
    /// every operation, and each reduction axis is reduced over exactly once
    /// and read only inside that reduction.
    pub fn finish(self) -> Result<ComputeDef, ComputeError> {
        if self.outputs.is_empty() {
            return Err(ComputeError::NoOutputs);
        }
        let mut checker = Checker::new(&self.exprs, &self.vars, &self.iter_vars, &self.inputs);
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for (name, expr) in self.outputs {
            let dtype = checker.output(&expr)?;
            outputs.push(OutputDef { name, dtype, expr });
        }
        checker.every_reduce_axis_used()?;
        if checker.has_reduction() && outputs.len() > 1 {
            return Err(ComputeError::ReductionWithSeveralOutputs);
        }
        let dims = self
            .iter_vars
            .iter()
            .filter(|var| var.kind == IterKind::Spatial)
            .map(|var| var.extent)
            .collect();
        let output_shape = Shape::new(dims)?;
        let sizes = self.iter_vars.iter().map(|var| var.extent);
        let sizes = sizes
            .chain(self.inputs.iter().map(|input| Dim::new(input.shape.element_count().get())));
        let sizes = sizes.chain(std::iter::once(Dim::new(output_shape.element_count().get())));
        if let Some(too_large) = sizes.into_iter().find(|size| i64::try_from(size.get()).is_err()) {
            return Err(ComputeError::ExtentTooLarge(too_large));
        }
        Ok(ComputeDef {
            name: self.name,
            exprs: self.exprs,
            vars: self.vars,
            iter_vars: self.iter_vars,
            inputs: self.inputs,
            outputs,
            output_shape,
        })
    }
}
