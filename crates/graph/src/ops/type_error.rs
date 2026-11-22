//! `TypeError`: why an op's type rule rejects its operands.

use super::conv::Groups;
use ptxc_core::{Axis, DType, Dim, Rank, Shape, ShapeError};
use thiserror::Error;

/// A number of operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OperandCount(pub(super) usize);

impl OperandCount {
    /// The count.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Operands an op's type rule rejects.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeError {
    /// The wrong number of operands.
    #[error("expected {} operands, got {}", .expected.get(), .actual.get())]
    Arity {
        /// What the op takes.
        expected: OperandCount,
        /// What it was given.
        actual: OperandCount,
    },
    /// A number of operands outside the op's range.
    #[error("expected {} to {} operands, got {}", .min.get(), .max.get(), .actual.get())]
    ArityBetween {
        /// The fewest the op takes.
        min: OperandCount,
        /// The most the op takes.
        max: OperandCount,
        /// What it was given.
        actual: OperandCount,
    },
    /// An operand of the wrong rank.
    #[error("expected rank {}, got rank {}", .expected.get(), .actual.get())]
    Rank {
        /// The rank the op needs.
        expected: Rank,
        /// The operand's rank.
        actual: Rank,
    },
    /// An operand of too low a rank.
    #[error("expected rank at least {}, got rank {}", .minimum.get(), .actual.get())]
    RankBelow {
        /// The lowest rank the op takes.
        minimum: Rank,
        /// The operand's rank.
        actual: Rank,
    },
    /// The dims a matrix product contracts differ.
    #[error("contracted dims differ: {} and {}", .lhs.get(), .rhs.get())]
    ContractedDims {
        /// The left operand's contracted dim.
        lhs: Dim,
        /// The right operand's contracted dim.
        rhs: Dim,
    },
    /// An axis beyond the operand's rank.
    #[error("axis {} is out of range for rank {}", .axis.index(), .rank.get())]
    AxisOutOfRange {
        /// The axis.
        axis: Axis,
        /// The operand's rank.
        rank: Rank,
    },
    /// An axis listed twice.
    #[error("axis {} is listed twice", .0.index())]
    RepeatedAxis(Axis),
    /// A window with a zero-length kernel.
    #[error("the kernel has a zero-length axis")]
    EmptyKernel,
    /// A window larger than the padded input along an axis.
    #[error("a kernel of {} does not fit an input of {} with its padding", .kernel.get(), .input.get())]
    WindowTooLarge {
        /// The input extent.
        input: Dim,
        /// The kernel extent.
        kernel: Dim,
    },
    /// A convolution whose input channels are not the weights' channels
    /// times the groups.
    #[error("input has {} channels, but weights have {} per group in {} groups", .input.get(), .weights.get(), .groups.get())]
    Channels {
        /// The input's channels.
        input: Dim,
        /// The weights' channels per group.
        weights: Dim,
        /// The groups.
        groups: Groups,
    },
    /// A convolution whose groups do not divide its filters.
    #[error("{} groups do not divide {} filters", .groups.get(), .filters.get())]
    FiltersNotDivisible {
        /// The filters (output channels).
        filters: Dim,
        /// The groups.
        groups: Groups,
    },
    /// A per-channel or per-filter parameter of the wrong length.
    #[error("expected a parameter of length {}, got {}", .expected.get(), .actual.get())]
    ParameterLength {
        /// The required length.
        expected: Dim,
        /// The bias's length.
        actual: Dim,
    },
    /// An op that needs at least one operand was given none.
    #[error("expected at least one operand")]
    NoOperands,
    /// Operands that must share an element type do not.
    #[error("operands have different element types: {lhs:?} and {rhs:?}")]
    MixedDTypes {
        /// The first operand's element type.
        lhs: DType,
        /// The second operand's element type.
        rhs: DType,
    },
    /// An element type the op does not accept.
    #[error("element type {0:?} is not accepted")]
    UnsupportedDType(DType),
    /// Shapes that do not broadcast together.
    #[error("shapes {:?} and {:?} do not broadcast", .lhs.dims(), .rhs.dims())]
    NotBroadcastable {
        /// The first operand's shape.
        lhs: Shape,
        /// The second operand's shape.
        rhs: Shape,
    },
    /// An operand that must be a scalar has a shape.
    #[error("expected a scalar, got shape {:?}", .0.dims())]
    NotScalar(Shape),
    /// A shape that must broadcast to another, without changing it, does not.
    #[error("shape {:?} does not broadcast to {:?}", .from.dims(), .to.dims())]
    NotUnidirectional {
        /// The shape being broadcast.
        from: Shape,
        /// The shape it must broadcast to.
        to: Shape,
    },
    /// An output shape that cannot exist.
    #[error(transparent)]
    Shape(#[from] ShapeError),
}
