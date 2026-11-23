//! `IterVar`: a spatial or reduction iteration axis with its extent.

use super::names::VarName;
use ptxc_core::Dim;

ptxc_core::define_index!(
    /// An iteration axis of a compute definition.
    pub IterVarId
);

/// What an iteration axis ranges over.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IterKind {
    /// An output axis: one output element per value.
    Spatial,
    /// An axis a reduction combines over, used inside exactly one
    /// `Reduce` or `ArgReduce`.
    Reduce,
}

/// An iteration axis: its name, its extent, and whether it is spatial.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IterVar {
    /// The name it prints with.
    pub name: VarName,
    /// The number of values, from 0.
    pub extent: Dim,
    /// Spatial or reduction.
    pub kind: IterKind,
}
