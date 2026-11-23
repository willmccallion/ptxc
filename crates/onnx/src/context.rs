//! The importer's table of ONNX values, and the view of it each node's
//! importer works through.

use crate::attribute::NodeAttrs;
use crate::error::{ImportError, ImportErrorKind};
use crate::opset::OpsetVersion;
use ptxc_core::{Axis, HostTensor, Rank, TensorData};
use ptxc_graph::{GraphBuilder, Layout, TensorType, ValueId, ValueName};
use ptxc_onnx_proto::NodeProto;
use std::collections::HashMap;

/// What the importer knows about one ONNX value: its type, the graph value
/// holding it (created on first use, for constants), and its value when it
/// is known at import.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) ty: TensorType,
    pub(crate) value: Option<ValueId>,
    pub(crate) constant: Option<HostTensor>,
}

impl Entry {
    pub(crate) fn constant(tensor: HostTensor) -> Self {
        let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
        Self { ty, value: None, constant: Some(tensor) }
    }
}

/// The graph being built and every ONNX value seen so far, by name.
pub(crate) struct Values {
    pub(crate) builder: GraphBuilder,
    entries: HashMap<String, Entry>,
}

impl Values {
    pub(crate) fn new() -> Self {
        Self { builder: GraphBuilder::new(), entries: HashMap::new() }
    }

    pub(crate) fn insert(&mut self, name: &str, entry: Entry) {
        let _replaced: Option<Entry> = self.entries.insert(name.to_owned(), entry);
    }

    pub(crate) fn get(&self, name: &str) -> Result<&Entry, ImportError> {
        self.entries
            .get(name)
            .ok_or_else(|| ImportErrorKind::UndefinedValue(name.to_owned()).into())
    }

    /// The graph value for `name`, adding its constant to the graph the first
    /// time a node reads it.
    pub(crate) fn value(&mut self, name: &str) -> Result<ValueId, ImportError> {
        let entry = self
            .entries
            .get_mut(name)
            .ok_or_else(|| ImportErrorKind::UndefinedValue(name.to_owned()))?;
        if let Some(value) = entry.value {
            return Ok(value);
        }
        let tensor = entry
            .constant
            .clone()
            .ok_or_else(|| ImportErrorKind::UndefinedValue(name.to_owned()))?;
        let value = self.builder.add_constant(tensor);
        self.builder.set_name(value, ValueName::new(name.to_owned()));
        entry.value = Some(value);
        Ok(value)
    }
}

/// A node's importer's view: its op type, attributes, inputs and opset.
pub(crate) struct NodeContext<'a> {
    pub(crate) attrs: NodeAttrs<'a>,
    pub(crate) opset: OpsetVersion,
    node: &'a NodeProto,
    values: &'a Values,
}

impl<'a> NodeContext<'a> {
    pub(crate) fn new(node: &'a NodeProto, values: &'a Values, opset: OpsetVersion) -> Self {
        let op = node.op_type.as_deref().unwrap_or_default();
        Self { attrs: NodeAttrs::new(op, &node.attribute), opset, node, values }
    }

    pub(crate) fn op_type(&self) -> &'a str {
        self.node.op_type.as_deref().unwrap_or_default()
    }

    /// Whether input `index` is given; ONNX leaves an optional input out
    /// with an empty name.
    pub(crate) fn has_input(&self, index: usize) -> bool {
        self.node.input.get(index).is_some_and(|name| !name.is_empty())
    }

    /// How many inputs the node lists, absent optional ones included.
    pub(crate) const fn input_count(&self) -> usize {
        self.node.input.len()
    }

    /// Whether output `index` is wanted; an unwanted optional output has an
    /// empty name or is left off the end.
    pub(crate) fn wants_output(&self, index: usize) -> bool {
        self.node.output.get(index).is_some_and(|name| !name.is_empty())
    }

    /// How many outputs the node lists, unwanted optional ones included.
    pub(crate) const fn output_count(&self) -> usize {
        self.node.output.len()
    }

    pub(crate) fn input_name(&self, index: usize) -> Result<&'a str, ImportError> {
        self.node
            .input
            .get(index)
            .filter(|name| !name.is_empty())
            .map(String::as_str)
            .ok_or_else(|| ImportErrorKind::Missing("a required input").into())
    }

    pub(crate) fn entry(&self, index: usize) -> Result<&'a Entry, ImportError> {
        self.values.get(self.input_name(index)?)
    }

    pub(crate) fn ty(&self, index: usize) -> Result<&'a TensorType, ImportError> {
        Ok(&self.entry(index)?.ty)
    }

    /// Input `index`'s value, which must be known at import: a shape-like
    /// operand under static shapes (D3, D32).
    pub(crate) fn constant(&self, index: usize) -> Result<&'a HostTensor, ImportError> {
        let name = self.input_name(index)?;
        self.values.get(name)?.constant.as_ref().ok_or_else(|| {
            ImportErrorKind::NeedsConstant { operand: name.to_owned(), inputs: Vec::new() }.into()
        })
    }

    /// An ONNX axis in `[-rank, rank)` as an `Axis`.
    pub(crate) fn axis(&self, axis: i64, rank: Rank) -> Result<Axis, ImportError> {
        Axis::from_signed(axis, rank).map_err(|error| {
            ImportErrorKind::Invalid {
                field: "axis",
                problem: format!("{}: {error}", self.op_type()),
            }
            .into()
        })
    }

    /// The axis attribute `name` (default `default`) resolved against `rank`.
    pub(crate) fn axis_attr(
        &mut self,
        name: &str,
        default: i64,
        rank: Rank,
    ) -> Result<Axis, ImportError> {
        let axis = self.attrs.int_or(name, default)?;
        self.axis(axis, rank)
    }

    /// Input `index` as integers, which must be known at import.
    pub(crate) fn ints(&self, index: usize) -> Result<Vec<i64>, ImportError> {
        integers(self.constant(index)?).ok_or_else(|| {
            ImportErrorKind::Invalid {
                field: "input",
                problem: format!("{} input {index} must be integers", self.op_type()),
            }
            .into()
        })
    }
}

/// A tensor's elements as `i64`, if it holds the `i32` or `i64` that ONNX's
/// shape-like operands use.
pub(crate) fn integers(tensor: &HostTensor) -> Option<Vec<i64>> {
    match tensor.data() {
        TensorData::I64(values) => Some(values.clone()),
        TensorData::I32(values) => Some(values.iter().map(|&value| i64::from(value)).collect()),
        _ => None,
    }
}
