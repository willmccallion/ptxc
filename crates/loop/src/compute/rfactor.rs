//! `rfactor`: a definition's reduction split into slices of one of its
//! axes, as TVM's `rfactor` splits one for a parallel split-K (D75). The
//! partial definition reduces each slice, the slice its outermost spatial
//! axis; the combine definition reduces the slices and does with the result
//! what the original did with its reduction.

use super::builder::ComputeBuilder;
use super::compute_def::{ComputeDef, IndexVar, OutputDef, TensorParam};
use super::error::ComputeError;
use super::iter_var::{IterKind, IterVarId};
use super::names::{ComputeName, TensorName, VarName};
use super::scalar_op::ReduceOp;
use super::tensor_expr::{TensorExpr, TensorParamId};
use crate::expr::{Expr, ExprId, VarId};
use ptxc_core::{Dim, Idx, IndexVec, Shape};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use thiserror::Error;

/// How many slices a reduction is split into: at least 2.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Slices(u64);

impl Slices {
    /// `count`, if it is at least 2.
    pub const fn new(count: u64) -> Option<Self> {
        if count >= 2 { Some(Self(count)) } else { None }
    }

    /// The count.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// What one of a combine definition's inputs holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CombineInput {
    /// The partial definition's output.
    Partials,
    /// An input of the original definition.
    Original(TensorParamId),
}

/// A definition whose reduction `rfactor` split.
#[derive(Clone, Debug)]
pub struct Rfactored {
    /// Each slice's reduction, `[slices, ..spatial]`, in the reduction's
    /// own type.
    pub partial: ComputeDef,
    /// The original input each of `partial`'s inputs reads. One the axis
    /// indexes is viewed with that dimension split as `[slices, chunk]`,
    /// which leaves every row-major element where it was.
    pub partial_inputs: IndexVec<TensorParamId, TensorParamId>,
    /// The slices reduced, inside the original's output expression.
    pub combine: ComputeDef,
    /// What each of `combine`'s inputs holds.
    pub combine_inputs: IndexVec<TensorParamId, CombineInput>,
}

impl Rfactored {
    /// The partials `partial` writes and `combine` reads: `combine`'s first
    /// input.
    pub fn partials(&self) -> &TensorParam {
        &self.combine.inputs()[TensorParamId::new(0)]
    }
}

/// Why `rfactor` cannot split a definition.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RfactorError {
    /// The definition's output holds no `Reduce` over the axis.
    #[error("the definition has no reduction over the axis")]
    NoReduction,
    /// The slices do not divide the axis.
    #[error("{} slices do not divide the axis's {} values", .slices.get(), .extent.get())]
    Indivisible {
        /// The axis's extent.
        extent: Dim,
        /// The slices asked for.
        slices: Slices,
    },
    /// A load reads the axis other than as one whole index of a dimension
    /// as long as the axis, or another expression reads it.
    #[error("the axis is read other than as one whole index of a dimension it spans")]
    NotPlain,
    /// The definition gathers, so its indices read loaded values.
    #[error("a definition with loaded indices is not split")]
    Gather,
    /// A split definition is not well-formed.
    #[error(transparent)]
    Compute(#[from] ComputeError),
}

/// `compute`'s reduction over `axis` split into `slices` equal slices.
pub fn rfactor(
    compute: &ComputeDef,
    axis: IterVarId,
    slices: Slices,
) -> Result<Rfactored, RfactorError> {
    let split = Split::new(compute, axis, slices)?;
    let (partial, partial_inputs) = split.partial()?;
    let (combine, combine_inputs) = split.combine(&partial)?;
    Ok(Rfactored { partial, partial_inputs, combine, combine_inputs })
}

/// A reduction to split, and where each tensor its body loads reads the
/// axis.
struct Split<'c> {
    compute: &'c ComputeDef,
    axis: IterVarId,
    slices: Slices,
    chunk: Dim,
    op: ReduceOp,
    output: &'c OutputDef,
    reduction: &'c TensorExpr,
    /// The dimension each tensor the reduction loads reads the axis at, if
    /// it does.
    positions: BTreeMap<TensorParamId, Option<usize>>,
}

impl<'c> Split<'c> {
    fn new(compute: &'c ComputeDef, axis: IterVarId, slices: Slices) -> Result<Self, RfactorError> {
        let [output] = compute.outputs() else { return Err(RfactorError::NoReduction) };
        let reduction = output.expr.reduction().ok_or(RfactorError::NoReduction)?;
        let TensorExpr::Reduce { op, axes, body } = reduction else {
            return Err(RfactorError::NoReduction);
        };
        if !axes.contains(&axis) {
            return Err(RfactorError::NoReduction);
        }
        if compute.vars().iter().any(|var| matches!(var, IndexVar::Loaded { .. })) {
            return Err(RfactorError::Gather);
        }
        let extent = compute.iter_vars()[axis].extent;
        if !extent.get().is_multiple_of(slices.get()) {
            return Err(RfactorError::Indivisible { extent, slices });
        }
        let chunk = Dim::new(extent.get() / slices.get());
        let positions = positions(compute, axis, body)?;
        Ok(Self { compute, axis, slices, chunk, op: *op, output, reduction, positions })
    }

    /// The partial definition, and the original input each of its inputs
    /// reads.
    fn partial(
        &self,
    ) -> Result<(ComputeDef, IndexVec<TensorParamId, TensorParamId>), RfactorError> {
        let name = ComputeName::new(format!("{}_partial", self.compute.name()));
        let mut builder = ComputeBuilder::new(name);
        let mut inputs = IndexVec::new();
        let mut tensors = HashMap::new();
        for (&original, &position) in &self.positions {
            let param = &self.compute.inputs()[original];
            let shape = match position {
                Some(position) => self.split_shape(&param.shape, position)?,
                None => param.shape.clone(),
            };
            let _new =
                tensors.insert(original, builder.input(param.name.clone(), param.dtype, shape));
            let _pushed = inputs.push(original);
        }
        let (_, slice) = builder.spatial(self.slice_name(), Dim::new(self.slices.get()));
        let mut axes = HashMap::new();
        for (id, iter) in self.compute.iter_vars().iter_enumerated() {
            let extent = if id == self.axis { self.chunk } else { iter.extent };
            let axis = match iter.kind {
                IterKind::Spatial => builder.spatial(iter.name.clone(), extent),
                IterKind::Reduce => builder.reduce_axis(iter.name.clone(), extent),
            };
            let _new = axes.insert(id, axis);
        }
        let splits =
            self.positions.iter().filter_map(|(&tensor, &at)| Some((tensor, (at?, slice))));
        let renaming = Renaming { axes, tensors, splits: splits.collect() };
        let mut translator = Translator::new(self.compute, builder, renaming, None);
        let expr = translator.expr(self.reduction)?;
        Ok((translator.finish(self.output.name.clone(), expr)?, inputs))
    }

    /// The combine definition, reading the partial definition's output, and
    /// what each of its inputs holds.
    fn combine(
        &self,
        partial: &ComputeDef,
    ) -> Result<(ComputeDef, IndexVec<TensorParamId, CombineInput>), RfactorError> {
        let mut builder = ComputeBuilder::new(self.compute.name().clone());
        let dtype = self.compute.dtype_of(self.reduction);
        let partials =
            builder.input(TensorName::new("partials"), dtype, partial.output_shape().clone());
        let mut inputs = IndexVec::new();
        let _partials = inputs.push(CombineInput::Partials);
        let mut tensors = HashMap::new();
        for original in epilogue_tensors(&self.output.expr) {
            let param = &self.compute.inputs()[original];
            let id = builder.input(param.name.clone(), param.dtype, param.shape.clone());
            let _new = tensors.insert(original, id);
            let _pushed = inputs.push(CombineInput::Original(original));
        }
        let mut axes = HashMap::new();
        let mut index = Vec::new();
        for (id, iter) in self.compute.iter_vars().iter_enumerated() {
            if iter.kind == IterKind::Spatial {
                let (axis, var) = builder.spatial(iter.name.clone(), iter.extent);
                let _new = axes.insert(id, (axis, var));
                index.push(var);
            }
        }
        let (slice_axis, slice) =
            builder.reduce_axis(self.slice_name(), Dim::new(self.slices.get()));
        index.insert(0, slice);
        let body = Box::new(TensorExpr::load(partials, index));
        let reduced = TensorExpr::Reduce { op: self.op, axes: vec![slice_axis], body };
        let renaming = Renaming { axes, tensors, splits: HashMap::new() };
        let mut translator = Translator::new(self.compute, builder, renaming, Some(reduced));
        let expr = translator.expr(&self.output.expr)?;
        Ok((translator.finish(self.output.name.clone(), expr)?, inputs))
    }

    /// `shape` with dimension `position`, which the axis spans, split into
    /// the slices and their chunk.
    fn split_shape(&self, shape: &Shape, position: usize) -> Result<Shape, RfactorError> {
        let dims = shape.dims();
        let extent = self.compute.iter_vars()[self.axis].extent;
        if dims.get(position) != Some(&extent) {
            return Err(RfactorError::NotPlain);
        }
        let mut split = dims[..position].to_vec();
        split.extend([Dim::new(self.slices.get()), self.chunk]);
        split.extend_from_slice(&dims[position + 1..]);
        Ok(Shape::new(split).map_err(ComputeError::from)?)
    }

    fn slice_name(&self) -> VarName {
        VarName::new(format!("{}_slice", self.compute.iter_vars()[self.axis].name))
    }
}

/// The dimension at which each tensor `body` loads reads `axis`, if it
/// does. Every load of a tensor reads it at the same dimension, as the
/// whole index there, and nothing else reads it.
fn positions(
    compute: &ComputeDef,
    axis: IterVarId,
    body: &TensorExpr,
) -> Result<BTreeMap<TensorParamId, Option<usize>>, RfactorError> {
    let is_axis = |var: VarId| compute.vars()[var] == IndexVar::Iter(axis);
    let exprs = compute.exprs();
    let reads_axis = |id: ExprId| exprs.vars(id).into_iter().any(is_axis);
    let mut positions = BTreeMap::new();
    for node in nodes(body, &|_| true) {
        match node {
            TensorExpr::Load { tensor, index, .. } => {
                let whole = |id: ExprId| matches!(exprs.get(id), Expr::Var(var) if is_axis(var));
                let at: Vec<usize> = (0..index.len()).filter(|&at| reads_axis(index[at])).collect();
                let position = match at[..] {
                    [] => None,
                    [at] if whole(index[at]) => Some(at),
                    _ => return Err(RfactorError::NotPlain),
                };
                if *positions.entry(*tensor).or_insert(position) != position {
                    return Err(RfactorError::NotPlain);
                }
            }
            TensorExpr::Index { index, .. } if reads_axis(*index) => {
                return Err(RfactorError::NotPlain);
            }
            _ => {}
        }
    }
    Ok(positions)
}

/// The tensors `expr` loads outside its reduction.
fn epilogue_tensors(expr: &TensorExpr) -> BTreeSet<TensorParamId> {
    let outside = |node: &TensorExpr| {
        !matches!(node, TensorExpr::Reduce { .. } | TensorExpr::ArgReduce { .. })
    };
    let loaded = nodes(expr, &outside).into_iter().filter_map(|node| match node {
        TensorExpr::Load { tensor, .. } => Some(*tensor),
        _ => None,
    });
    loaded.collect()
}

/// `expr` and the nodes under it, entering the operands of those `enters`
/// accepts.
fn nodes<'e>(expr: &'e TensorExpr, enters: &impl Fn(&TensorExpr) -> bool) -> Vec<&'e TensorExpr> {
    let mut found = vec![expr];
    let mut next = 0;
    while let Some(&node) = found.get(next) {
        if enters(node) {
            found.extend(node.operands());
        }
        next += 1;
    }
    found
}

/// Where each axis and tensor of the original is in a new definition.
struct Renaming {
    /// Each axis, and its index variable's expression.
    axes: HashMap<IterVarId, (IterVarId, ExprId)>,
    tensors: HashMap<TensorParamId, TensorParamId>,
    /// Each tensor viewed with the split axis's dimension split: the
    /// dimension, and the slice's index, read before the chunk's.
    splits: HashMap<TensorParamId, (usize, ExprId)>,
}

/// Builds a new definition from expressions of `from`.
struct Translator<'c> {
    from: &'c ComputeDef,
    builder: ComputeBuilder,
    /// Each of `from`'s variables the new definition has, as its expression
    /// there.
    vars: HashMap<VarId, ExprId>,
    axes: HashMap<IterVarId, IterVarId>,
    tensors: HashMap<TensorParamId, TensorParamId>,
    /// Each tensor viewed with the split axis's dimension split: the
    /// dimension, and the slice's index, read before the chunk's.
    splits: HashMap<TensorParamId, (usize, ExprId)>,
    /// What `from`'s reduction becomes, if not its own translation.
    reduction: Option<TensorExpr>,
}

impl<'c> Translator<'c> {
    /// A translator into `builder` of `from` as `renaming` places it, with
    /// `from`'s reduction translated as `reduction` if given.
    fn new(
        from: &'c ComputeDef,
        builder: ComputeBuilder,
        renaming: Renaming,
        reduction: Option<TensorExpr>,
    ) -> Self {
        let Renaming { axes, tensors, splits } = renaming;
        let vars = from.vars().iter_enumerated().filter_map(|(var, index_var)| match index_var {
            IndexVar::Iter(axis) => Some((var, axes.get(axis)?.1)),
            IndexVar::Loaded { .. } => None,
        });
        let vars = vars.collect();
        let axes = axes.into_iter().map(|(original, (axis, _))| (original, axis)).collect();
        Self { from, builder, vars, axes, tensors, splits, reduction }
    }

    fn expr(&mut self, expr: &TensorExpr) -> Result<TensorExpr, RfactorError> {
        let boxed = |this: &mut Self, value: &TensorExpr| this.expr(value).map(Box::new);
        Ok(match expr {
            TensorExpr::Load { tensor, index, padding } => TensorExpr::Load {
                tensor: self.tensor(*tensor)?,
                index: self.load_index(*tensor, index)?,
                padding: *padding,
            },
            TensorExpr::Const(scalar) => TensorExpr::Const(*scalar),
            TensorExpr::Index { index, dtype } => {
                TensorExpr::Index { index: self.index(*index)?, dtype: *dtype }
            }
            TensorExpr::Unary(op, value) => TensorExpr::Unary(*op, boxed(self, value)?),
            TensorExpr::Binary(op, lhs, rhs) => {
                TensorExpr::Binary(*op, boxed(self, lhs)?, boxed(self, rhs)?)
            }
            TensorExpr::Cast { to, overflow, value } => {
                TensorExpr::Cast { to: *to, overflow: *overflow, value: boxed(self, value)? }
            }
            TensorExpr::Select { condition, then, otherwise } => TensorExpr::Select {
                condition: boxed(self, condition)?,
                then: boxed(self, then)?,
                otherwise: boxed(self, otherwise)?,
            },
            TensorExpr::Reduce { op, axes, body } => {
                if let Some(reduction) = &self.reduction {
                    return Ok(reduction.clone());
                }
                let axes = axes.iter().map(|&axis| self.axis(axis)).collect::<Result<_, _>>()?;
                TensorExpr::Reduce { op: *op, axes, body: boxed(self, body)? }
            }
            TensorExpr::ArgReduce { kind, tie, axis, body } => TensorExpr::ArgReduce {
                kind: *kind,
                tie: *tie,
                axis: self.axis(*axis)?,
                body: boxed(self, body)?,
            },
        })
    }

    /// A load's index, with the slice's index before a split dimension's.
    fn load_index(
        &mut self,
        tensor: TensorParamId,
        index: &[ExprId],
    ) -> Result<Vec<ExprId>, RfactorError> {
        let split = self.splits.get(&tensor).copied();
        let mut parts = Vec::with_capacity(index.len() + 1);
        for (position, &part) in index.iter().enumerate() {
            if let Some((at, slice)) = split
                && at == position
            {
                parts.push(slice);
            }
            parts.push(self.index(part)?);
        }
        Ok(parts)
    }

    fn index(&mut self, id: ExprId) -> Result<ExprId, RfactorError> {
        let vars = &self.vars;
        let imported =
            self.builder.exprs().import(self.from.exprs(), id, &|var| vars.get(&var).copied());
        Ok(imported.map_err(ComputeError::UnknownVar)?)
    }

    fn tensor(&self, tensor: TensorParamId) -> Result<TensorParamId, RfactorError> {
        Ok(self.tensors.get(&tensor).copied().ok_or(ComputeError::UnknownTensor(tensor))?)
    }

    fn axis(&self, axis: IterVarId) -> Result<IterVarId, RfactorError> {
        Ok(self.axes.get(&axis).copied().ok_or(ComputeError::UnknownAxis(axis))?)
    }

    fn finish(mut self, name: TensorName, expr: TensorExpr) -> Result<ComputeDef, RfactorError> {
        self.builder.output(name, expr);
        Ok(self.builder.finish()?)
    }
}
