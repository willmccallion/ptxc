use ptxc_core::{DType, Dim, HostTensor, Shape};
use ptxc_graph::{BinaryOp, GraphBuilder, Layout, Op, TensorType, UnaryOp, ValueId};

pub(crate) fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

pub(crate) fn tensor_type(dtype: DType, dims: &[u64]) -> TensorType {
    TensorType::new(dtype, shape(dims), Layout::RowMajor)
}

pub(crate) fn f32_type(dims: &[u64]) -> TensorType {
    tensor_type(DType::F32, dims)
}

pub(crate) fn f32_constant(dims: &[u64]) -> HostTensor {
    let shape = shape(dims);
    let len = usize::try_from(shape.element_count().get()).unwrap();
    HostTensor::new(shape, vec![0.5_f32; len].into()).unwrap()
}

/// Adds a single-output node and returns its output.
pub(crate) fn add_one(builder: &mut GraphBuilder, op: Op, operands: &[ValueId]) -> ValueId {
    let [output] = builder.add_node(op, operands.to_vec()).unwrap()[..] else {
        panic!("expected a single-output op")
    };
    output
}

pub(crate) const fn unary(op: UnaryOp) -> Op {
    Op::Unary(op)
}

pub(crate) const fn binary(op: BinaryOp) -> Op {
    Op::Binary(op)
}
