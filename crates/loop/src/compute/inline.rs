//! `inline`: a producer's computation in place of a consumer's reads of its
//! output, as TVM's `compute_inline` and XLA's producer-consumer fusion
//! merge two computations into one (D80).

use super::builder::ComputeBuilder;
use super::compute_def::{ComputeDef, IndexVar};
use super::error::ComputeError;
use super::iter_var::{IterKind, IterVarId};
use super::names::ComputeName;
use super::rebuild::{Rebuild, rebuild};
use super::scalar::Scalar;
use super::tensor_expr::{TensorExpr, TensorParamId};
use crate::expr::{Expr, ExprId, VarId};
use ptxc_core::IndexVec;
use std::collections::HashMap;
use thiserror::Error;

/// What one of an inlined definition's inputs holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InlinedInput {
    /// An input of the consumer.
    Consumer(TensorParamId),
    /// An input of the producer.
    Producer(TensorParamId),
}

/// A consumer with a producer inlined into it.
#[derive(Clone, Debug)]
pub struct Inlined {
    /// The consumer, computing the producer's output where it read it.
    pub compute: ComputeDef,
    /// What each of `compute`'s inputs holds: the consumer's other inputs,
    /// then the producer's.
    pub inputs: IndexVec<TensorParamId, InlinedInput>,
}

/// Why `inline` cannot inline a producer into a consumer.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum InlineError {
    /// The producer computes several outputs.
    #[error("the producer computes more than one output")]
    SeveralOutputs,
    /// The consumer's input is not of the producer's output's type and
    /// shape.
    #[error("the consumer's input is not the producer's output")]
    Mismatch,
    /// The producer reduces, and the consumer reads it other than at its
    /// own element or reduces too: a definition holds one reduction, read
    /// only by its epilogue.
    #[error("a producer that reduces inlines only as its consumer's whole reduction")]
    NotAnEpilogue,
    /// The consumer reads the producer with padding, which an inlined
    /// value does not have.
    #[error("the consumer reads the producer with padding")]
    Padded,
    /// The producer or the consumer gathers, reading loaded indices.
    #[error("a definition with loaded indices is not inlined")]
    Gather,
    /// The inlined definition is not well-formed.
    #[error(transparent)]
    Compute(#[from] ComputeError),
}

/// `consumer` with each read of its `inputs`, which all hold `producer`'s
/// output, replaced by that output computed at the read's index.
///
/// A producer without a reduction inlines at every read, at any index, and
/// is computed again at each. One with a reduction inlines only where the
/// consumer reads it at the consumer's own element and reduces nothing
/// itself, so the reduction stays the definition's one and the consumer
/// becomes its epilogue, which may read it more than once.
pub fn inline(
    consumer: &ComputeDef,
    inputs: &[TensorParamId],
    producer: &ComputeDef,
) -> Result<Inlined, InlineError> {
    let [output] = producer.outputs() else { return Err(InlineError::SeveralOutputs) };
    for &input in inputs {
        let read = consumer.inputs().get(input).ok_or(ComputeError::UnknownTensor(input))?;
        let matches =
            read.dtype == producer.dtype_of(&output.expr) && &read.shape == producer.output_shape();
        if !matches {
            return Err(InlineError::Mismatch);
        }
    }
    if [consumer, producer].iter().any(|compute| gathers(compute)) {
        return Err(InlineError::Gather);
    }
    if reads_padded(consumer, inputs) {
        return Err(InlineError::Padded);
    }
    if output.expr.reduction().is_some() && !is_epilogue(consumer, inputs) {
        return Err(InlineError::NotAnEpilogue);
    }
    let name = ComputeName::new(format!("{}_{}", producer.name(), consumer.name()));
    Inliner::new(consumer, inputs, producer, name).finish()
}

/// Whether `compute` reads loaded indices.
fn gathers(compute: &ComputeDef) -> bool {
    compute.vars().iter().any(|var| matches!(var, IndexVar::Loaded { .. }))
}

/// Whether `consumer` reduces nothing and reads `inputs` only at its own
/// element, each index its own variable in order.
fn is_epilogue(consumer: &ComputeDef, inputs: &[TensorParamId]) -> bool {
    if consumer.outputs().iter().any(|output| output.expr.reduction().is_some()) {
        return false;
    }
    let own: Vec<VarId> =
        consumer.iter_vars().ids().filter_map(|axis| consumer.axis_var(axis)).collect();
    let at_own = |read: &[ExprId], _: bool| {
        read.len() == own.len()
            && read.iter().zip(&own).all(|(&id, &var)| consumer.exprs().get(id) == Expr::Var(var))
    };
    consumer.outputs().iter().all(|output| every_read(&output.expr, inputs, &at_own))
}

/// Whether `consumer` reads any of `inputs` with padding.
fn reads_padded(consumer: &ComputeDef, inputs: &[TensorParamId]) -> bool {
    let unpadded = |_: &[ExprId], padded: bool| !padded;
    !consumer.outputs().iter().all(|output| every_read(&output.expr, inputs, &unpadded))
}

/// Whether `holds` holds of every read of `inputs` in `expr`, given the
/// read's index and whether it is padded.
fn every_read(
    expr: &TensorExpr,
    inputs: &[TensorParamId],
    holds: &impl Fn(&[ExprId], bool) -> bool,
) -> bool {
    match expr {
        TensorExpr::Load { tensor, index, padding } => {
            !inputs.contains(tensor) || holds(index, padding.is_some())
        }
        other => other.operands().into_iter().all(|operand| every_read(operand, inputs, holds)),
    }
}

/// Builds the inlined definition: the consumer's axes and inputs, the
/// producer's inputs and reduction axes after them.
struct Inliner<'c> {
    consumer: &'c ComputeDef,
    inlined: &'c [TensorParamId],
    producer: &'c ComputeDef,
    builder: ComputeBuilder,
    inputs: IndexVec<TensorParamId, InlinedInput>,
    consumer_vars: HashMap<VarId, ExprId>,
    consumer_axes: HashMap<IterVarId, IterVarId>,
    consumer_tensors: HashMap<TensorParamId, TensorParamId>,
    producer_axes: HashMap<IterVarId, IterVarId>,
    producer_reduce_vars: HashMap<VarId, ExprId>,
    producer_tensors: HashMap<TensorParamId, TensorParamId>,
}

impl<'c> Inliner<'c> {
    fn new(
        consumer: &'c ComputeDef,
        inlined: &'c [TensorParamId],
        producer: &'c ComputeDef,
        name: ComputeName,
    ) -> Self {
        let mut inliner = Self {
            consumer,
            inlined,
            producer,
            builder: ComputeBuilder::new(name),
            inputs: IndexVec::new(),
            consumer_vars: HashMap::new(),
            consumer_axes: HashMap::new(),
            consumer_tensors: HashMap::new(),
            producer_axes: HashMap::new(),
            producer_reduce_vars: HashMap::new(),
            producer_tensors: HashMap::new(),
        };
        inliner.declare_inputs();
        inliner.declare_axes();
        inliner
    }

    /// The consumer's inputs but the inlined ones, then the producer's.
    fn declare_inputs(&mut self) {
        for (id, param) in self.consumer.inputs().iter_enumerated() {
            if !self.inlined.contains(&id) {
                let new = self.builder.input(param.name.clone(), param.dtype, param.shape.clone());
                let _old = self.consumer_tensors.insert(id, new);
                let _pushed = self.inputs.push(InlinedInput::Consumer(id));
            }
        }
        for (id, param) in self.producer.inputs().iter_enumerated() {
            let new = self.builder.input(param.name.clone(), param.dtype, param.shape.clone());
            let _old = self.producer_tensors.insert(id, new);
            let _pushed = self.inputs.push(InlinedInput::Producer(id));
        }
    }

    /// The consumer's axes, then the producer's reduction axes: its spatial
    /// axes take each read's index instead.
    fn declare_axes(&mut self) {
        for (axis, iter) in self.consumer.iter_vars().iter_enumerated() {
            let (new, var) = match iter.kind {
                IterKind::Spatial => self.builder.spatial(iter.name.clone(), iter.extent),
                IterKind::Reduce => self.builder.reduce_axis(iter.name.clone(), iter.extent),
            };
            let _old = self.consumer_axes.insert(axis, new);
            if let Some(old_var) = self.consumer.axis_var(axis) {
                let _old = self.consumer_vars.insert(old_var, var);
            }
        }
        for (axis, iter) in self.producer.iter_vars().iter_enumerated() {
            if iter.kind == IterKind::Reduce {
                let (new, var) = self.builder.reduce_axis(iter.name.clone(), iter.extent);
                let _old = self.producer_axes.insert(axis, new);
                if let Some(old_var) = self.producer.axis_var(axis) {
                    let _old = self.producer_reduce_vars.insert(old_var, var);
                }
            }
        }
    }

    fn finish(mut self) -> Result<Inlined, InlineError> {
        for output in self.consumer.outputs() {
            let expr = rebuild(&output.expr, &mut Consumer(&mut self))?;
            self.builder.output(output.name.clone(), expr);
        }
        Ok(Inlined { compute: self.builder.finish()?, inputs: self.inputs })
    }

    /// The producer's output at `index`, an index in the new definition.
    fn produce(&mut self, index: &[ExprId]) -> Result<TensorExpr, ComputeError> {
        let producer = self.producer;
        let spatial = producer
            .iter_vars()
            .iter_enumerated()
            .filter(|(_, iter)| iter.kind == IterKind::Spatial);
        let mut vars = self.producer_reduce_vars.clone();
        for ((axis, _), &at) in spatial.zip(index) {
            if let Some(var) = producer.axis_var(axis) {
                let _old = vars.insert(var, at);
            }
        }
        rebuild(&producer.outputs()[0].expr, &mut Producer { inliner: self, vars })
    }
}

/// Rebuilds the consumer's expressions, computing the producer where it
/// read it.
struct Consumer<'i, 'c>(&'i mut Inliner<'c>);

impl Rebuild for Consumer<'_, '_> {
    fn index(&mut self, index: ExprId) -> Result<ExprId, ComputeError> {
        let (from, vars) = (self.0.consumer.exprs(), &self.0.consumer_vars);
        let imported = self.0.builder.exprs().import(from, index, &|var| vars.get(&var).copied());
        imported.map_err(ComputeError::UnknownVar)
    }

    fn axis(&self, axis: IterVarId) -> Result<IterVarId, ComputeError> {
        self.0.consumer_axes.get(&axis).copied().ok_or(ComputeError::UnknownAxis(axis))
    }

    fn load(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
        padding: Option<Scalar>,
    ) -> Result<TensorExpr, ComputeError> {
        let index = index.iter().map(|&part| self.index(part)).collect::<Result<Vec<_>, _>>()?;
        if self.0.inlined.contains(&tensor) {
            return self.0.produce(&index);
        }
        let new = self.0.consumer_tensors.get(&tensor).copied();
        Ok(TensorExpr::Load {
            tensor: new.ok_or(ComputeError::UnknownTensor(tensor))?,
            index,
            padding,
        })
    }
}

/// Rebuilds the producer's output at one read: its spatial variables are
/// that read's index.
struct Producer<'i, 'c> {
    inliner: &'i mut Inliner<'c>,
    vars: HashMap<VarId, ExprId>,
}

impl Rebuild for Producer<'_, '_> {
    fn index(&mut self, index: ExprId) -> Result<ExprId, ComputeError> {
        let (from, vars) = (self.inliner.producer.exprs(), &self.vars);
        let imported =
            self.inliner.builder.exprs().import(from, index, &|var| vars.get(&var).copied());
        imported.map_err(ComputeError::UnknownVar)
    }

    fn axis(&self, axis: IterVarId) -> Result<IterVarId, ComputeError> {
        self.inliner.producer_axes.get(&axis).copied().ok_or(ComputeError::UnknownAxis(axis))
    }

    fn load(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
        padding: Option<Scalar>,
    ) -> Result<TensorExpr, ComputeError> {
        let index = index.iter().map(|&part| self.index(part)).collect::<Result<Vec<_>, _>>()?;
        let new = self.inliner.producer_tensors.get(&tensor).copied();
        Ok(TensorExpr::Load {
            tensor: new.ok_or(ComputeError::UnknownTensor(tensor))?,
            index,
            padding,
        })
    }
}
