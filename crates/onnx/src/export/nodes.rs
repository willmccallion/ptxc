//! Each exported op as an ONNX node, at opset 21, with any tensor inputs
//! ONNX takes in place of ptxc's attributes. Only the ops the fuzzer's
//! connectors draw are mapped, so their graphs can be checked against
//! ONNX Runtime (D54).

use super::{ExportError, onnx_dims};
use ptxc_graph::{Activation, BinaryOp, KeepDims, Op, ReduceOp, SoftmaxKind, UnaryOp, VariadicOp};
use ptxc_onnx_proto::attribute_proto::AttributeType;
use ptxc_onnx_proto::tensor_proto::DataType;
use ptxc_onnx_proto::{AttributeProto, NodeProto, TensorProto};

/// A node and the initializers it reads.
pub(super) struct Exported {
    pub(super) node: NodeProto,
    pub(super) initializers: Vec<TensorProto>,
}

/// `op` reading `inputs` and writing `outputs`, as an ONNX node.
pub(super) fn node(
    op: &Op,
    inputs: Vec<String>,
    outputs: Vec<String>,
) -> Result<Exported, ExportError> {
    let mut exported =
        Exported { node: proto(op_type(op)?, inputs, outputs), initializers: vec![] };
    match op {
        Op::Reduce(attrs) => {
            let axes =
                attrs.axes.iter().map(|axis| int64(axis.index())).collect::<Result<_, _>>()?;
            exported.tensor_input("axes", axes)?;
            let keep = i64::from(attrs.keep_dims == KeepDims::Keep);
            exported.node.attribute.push(int("keepdims", keep));
            // NOTE: ONNX reduces every axis when `axes` is empty; ptxc's
            // empty axes reduce none, which ONNX writes this way.
            exported.node.attribute.push(int("noop_with_empty_axes", 1));
        }
        Op::Softmax(attrs) => exported.node.attribute.push(int("axis", int64(attrs.axis.index())?)),
        Op::Transpose(attrs) => {
            let perm = attrs.permutation.iter().map(|axis| int64(axis.index()));
            exported.node.attribute.push(ints("perm", perm.collect::<Result<_, _>>()?));
        }
        Op::Reshape(shape) => {
            exported.tensor_input("shape", onnx_dims(shape)?)?;
            // NOTE: ONNX's Reshape copies the input's extent where the
            // shape holds 0; `allowzero` makes 0 an extent, as ptxc means.
            exported.node.attribute.push(int("allowzero", 1));
        }
        Op::Concat(axis) => exported.node.attribute.push(int("axis", int64(axis.index())?)),
        _ => {}
    }
    Ok(exported)
}

/// The ONNX op type of `op`, if the exporter maps it.
fn op_type(op: &Op) -> Result<&'static str, ExportError> {
    Ok(match op {
        Op::Unary(UnaryOp::Abs) => "Abs",
        Op::Unary(UnaryOp::Neg) => "Neg",
        Op::Activation(Activation::Relu) => "Relu",
        Op::Activation(Activation::Sigmoid) => "Sigmoid",
        Op::Activation(Activation::Tanh) => "Tanh",
        Op::Activation(Activation::Softsign) => "Softsign",
        Op::Binary(BinaryOp::Add) => "Add",
        Op::Binary(BinaryOp::Sub) => "Sub",
        Op::Binary(BinaryOp::Mul) => "Mul",
        Op::Variadic(VariadicOp::Max) => "Max",
        Op::Variadic(VariadicOp::Min) => "Min",
        Op::MatMul => "MatMul",
        Op::Reduce(attrs) => match attrs.op {
            ReduceOp::Sum => "ReduceSum",
            ReduceOp::Mean => "ReduceMean",
            ReduceOp::Max => "ReduceMax",
            ReduceOp::Min => "ReduceMin",
            other => return Err(ExportError::Op(format!("{other:?}"))),
        },
        Op::Softmax(attrs) => match attrs.kind {
            SoftmaxKind::Softmax => "Softmax",
            SoftmaxKind::LogSoftmax => "LogSoftmax",
        },
        Op::Transpose(_) => "Transpose",
        Op::Reshape(_) => "Reshape",
        Op::Concat(_) => "Concat",
        other => return Err(ExportError::Op(format!("{other:?}"))),
    })
}

impl Exported {
    /// Adds an `int64` initializer as the node's next input, named after
    /// its first output and `role`.
    fn tensor_input(&mut self, role: &str, values: Vec<i64>) -> Result<(), ExportError> {
        let output = self.node.output.first().map_or("", String::as_str);
        let name = format!("{output}_{role}");
        let dims = vec![int64(values.len())?];
        self.initializers.push(TensorProto {
            name: Some(name.clone()),
            data_type: Some(DataType::Int64 as i32),
            dims,
            int64_data: values,
            ..TensorProto::default()
        });
        self.node.input.push(name);
        Ok(())
    }
}

fn proto(op_type: &str, input: Vec<String>, output: Vec<String>) -> NodeProto {
    NodeProto { op_type: Some(op_type.to_owned()), input, output, ..NodeProto::default() }
}

fn int64(value: usize) -> Result<i64, ExportError> {
    i64::try_from(value).map_err(|_| ExportError::Index(value))
}

fn int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: Some(name.to_owned()),
        r#type: Some(AttributeType::Int as i32),
        i: Some(value),
        ..AttributeProto::default()
    }
}

fn ints(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: Some(name.to_owned()),
        r#type: Some(AttributeType::Ints as i32),
        ints: values,
        ..AttributeProto::default()
    }
}
