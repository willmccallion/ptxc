//! `Pad` and `Trilu`.

use crate::ops::window::Padding;
use crate::ops::{TypeError, TypeRule, exact_operands, require_dtype, require_rank};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Dim, Rank, Shape, ShapeError};

/// How `Pad` fills the added elements, from ONNX's `mode`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PadMode {
    /// A constant, the second operand.
    Constant,
    /// The axis mirrored, without repeating its edge element.
    Reflect,
    /// The edge element repeated.
    Edge,
    /// The axis wrapped around.
    Wrap,
}

/// The elements added at each end of one axis. Negative ONNX pads (cropping)
/// are turned into a `Slice` by the importer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AxisPadding {
    /// Added before the first element.
    pub begin: Padding,
    /// Added after the last element.
    pub end: Padding,
}

/// ONNX `Pad`. In `Constant` mode the operands are the input and a scalar
/// pad value of its element type (the importer supplies 0 if ONNX omits
/// it); in every other mode, the input alone.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PadAttrs {
    /// How added elements are filled.
    pub mode: PadMode,
    /// One entry per input axis.
    pub axes: Vec<AxisPadding>,
}

impl TypeRule for PadAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let input = match self.mode {
            PadMode::Constant => {
                let [input, value] = exact_operands(operands)?;
                require_dtype(input.dtype(), value)?;
                require_rank(value.shape(), Rank::new(0))?;
                input
            }
            PadMode::Reflect | PadMode::Edge | PadMode::Wrap => {
                let [input] = exact_operands(operands)?;
                input
            }
        };
        let shape = input.shape();
        require_rank(shape, Rank::new(self.axes.len()))?;
        let mut dims = Vec::with_capacity(self.axes.len());
        for ((axis, &extent), padding) in shape.rank().axes().zip(shape.dims()).zip(&self.axes) {
            let added = padding.begin.get().checked_add(padding.end.get());
            let grown = added.and_then(|added| extent.get().checked_add(added));
            let grown = grown
                .ok_or_else(|| ShapeError::ElementCountOverflow { dims: shape.dims().to_vec() })?;
            if self.mode != PadMode::Constant && extent == Dim::new(0) && grown > 0 {
                return Err(TypeError::PaddedEmptyAxis(axis));
            }
            dims.push(Dim::new(grown));
        }
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}

/// Which triangle `Trilu` keeps, from ONNX's `upper`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TriangularPart {
    /// On and above the diagonal.
    Upper,
    /// On and below the diagonal.
    Lower,
}

/// How far `Trilu`'s diagonal is above the main one (negative: below), from
/// ONNX's `k` operand, which is a constant at import.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DiagonalOffset(i64);

impl DiagonalOffset {
    /// Wraps an offset.
    pub const fn new(offset: i64) -> Self {
        Self(offset)
    }

    /// The offset.
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// ONNX `Trilu` on the last two axes; the rest are a batch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TriluAttrs {
    /// The triangle kept.
    pub part: TriangularPart,
    /// The diagonal it is measured from.
    pub diagonal: DiagonalOffset,
}

impl TypeRule for TriluAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        let rank = input.shape().rank();
        if rank < Rank::new(2) {
            return Err(TypeError::RankBelow { minimum: Rank::new(2), actual: rank });
        }
        Ok(vec![TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor)])
    }
}
