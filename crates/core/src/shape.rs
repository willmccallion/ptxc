//! Tensor shapes: `Dim`, `Shape`, `Rank` and the always non-negative `Axis`.

use crate::size::ElementCount;
use thiserror::Error;

/// A shape that cannot exist, or an axis outside a shape.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ShapeError {
    /// The product of the dimensions does not fit in a `u64`.
    #[error("shape {dims:?} has more than 2^64 elements")]
    ElementCountOverflow {
        /// The dimensions.
        dims: Vec<Dim>,
    },
    /// A signed axis outside `[-rank, rank)`.
    #[error("axis {axis} is out of range for rank {}", .rank.get())]
    AxisOutOfRange {
        /// The axis as given.
        axis: i64,
        /// The rank it was resolved against.
        rank: Rank,
    },
}

/// The extent of one tensor dimension. Static in v0.1.0 (D3).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Dim(u64);

impl Dim {
    /// Wraps an extent.
    pub const fn new(extent: u64) -> Self {
        Self(extent)
    }

    /// The extent.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The number of dimensions of a tensor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Rank(usize);

impl Rank {
    /// Wraps a number of dimensions.
    pub const fn new(dims: usize) -> Self {
        Self(dims)
    }

    /// The number of dimensions.
    pub const fn get(self) -> usize {
        self.0
    }

    /// Every axis of a tensor of this rank, outermost first.
    pub fn axes(self) -> impl Iterator<Item = Axis> {
        (0..self.0).map(Axis)
    }
}

/// A dimension of a tensor, counted from the outermost. ONNX's negative axes
/// are resolved once, at import, by [`Axis::from_signed`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Axis(usize);

impl Axis {
    /// The axis at position `index`. Ops hold axes without a rank, so their
    /// type rules check each against the operand it applies to.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Resolves an ONNX axis in `[-rank, rank)`, where `-1` is the innermost.
    pub fn from_signed(axis: i64, rank: Rank) -> Result<Self, ShapeError> {
        let out_of_range = || ShapeError::AxisOutOfRange { axis, rank };
        let magnitude = usize::try_from(axis.unsigned_abs()).map_err(|_| out_of_range())?;
        let resolved = if axis < 0 { rank.0.checked_sub(magnitude) } else { Some(magnitude) };
        resolved.filter(|&index| index < rank.0).map(Self).ok_or_else(out_of_range)
    }

    /// The position of this axis, counted from the outermost.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// The dimensions of a tensor, outermost first. A shape always knows its
/// element count, so it cannot hold more elements than a `u64` counts.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shape {
    dims: Vec<Dim>,
    elements: ElementCount,
}

impl Shape {
    /// A shape with the given dimensions.
    pub fn new(dims: Vec<Dim>) -> Result<Self, ShapeError> {
        let elements = dims.iter().try_fold(1_u64, |product, dim| product.checked_mul(dim.0));
        match elements {
            Some(count) => Ok(Self { dims, elements: ElementCount::new(count) }),
            None => Err(ShapeError::ElementCountOverflow { dims }),
        }
    }

    /// The shape of a scalar: rank 0, one element.
    pub const fn scalar() -> Self {
        Self { dims: Vec::new(), elements: ElementCount::new(1) }
    }

    /// The dimensions, outermost first.
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// The extent of `axis`, or `None` if the shape has no such axis.
    pub fn dim(&self, axis: Axis) -> Option<Dim> {
        self.dims.get(axis.0).copied()
    }

    /// The number of dimensions.
    pub const fn rank(&self) -> Rank {
        Rank(self.dims.len())
    }

    /// The number of elements: the product of the dimensions.
    pub const fn element_count(&self) -> ElementCount {
        self.elements
    }
}
