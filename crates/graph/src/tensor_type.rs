//! `TensorType`: data type, shape and layout of a value.

use ptxc_core::{DType, Shape};

/// How a tensor's elements are ordered in memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    /// The last axis varies fastest, as ONNX stores every tensor. The layout
    /// pass (M4) adds the alternatives it chooses between.
    RowMajor,
}

/// The type of a value: element type, static shape and layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorType {
    dtype: DType,
    shape: Shape,
    layout: Layout,
}

impl TensorType {
    /// A tensor type.
    pub const fn new(dtype: DType, shape: Shape, layout: Layout) -> Self {
        Self { dtype, shape, layout }
    }

    /// The element type.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The dimensions.
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The memory order.
    pub const fn layout(&self) -> Layout {
        self.layout
    }
}
