//! Lowering through the public API: the definitions each op family builds,
//! and how stages bind graph values.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod activation;
mod elementwise;
mod gather;
mod groups;
mod matmul;
mod norm;
mod quant;
mod reduce;
mod resize;
mod shape;
mod window;

use ptxc_core::{DType, Dim, Shape};
use ptxc_graph::{Graph, GraphBuilder, Layout, Op, Partitioned, TensorType, ValueId};
use ptxc_lower::{LowerError, Stage, lower};

pub(crate) fn tensor_type(dtype: DType, dims: &[u64]) -> TensorType {
    let shape = Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    TensorType::new(dtype, shape, Layout::RowMajor)
}

/// Adds a single-output node and returns its output.
pub(crate) fn add_one(builder: &mut GraphBuilder, op: Op, operands: &[ValueId]) -> ValueId {
    builder.add_node(op, operands.to_vec()).unwrap()[0]
}

/// Every group's stages, flattened in order.
pub(crate) fn stages(graph: Graph) -> Result<Vec<Stage>, LowerError> {
    let partitioned = Partitioned::one_node_per_group(graph).unwrap();
    Ok(lower(&partitioned)?.iter().flat_map(|group| group.stages().to_vec()).collect())
}
