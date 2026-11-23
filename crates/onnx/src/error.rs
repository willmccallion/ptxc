//! `ImportError`: what failed to import, on which node, and why.

use ptxc_graph::NodeError;
use std::path::PathBuf;
use thiserror::Error;

/// A model that cannot be imported, with the node it happened at.
#[derive(Debug, Error)]
#[error("{}{kind}", .node.as_ref().map(|node| format!("node {node}: ")).unwrap_or_default())]
pub struct ImportError {
    node: Option<String>,
    kind: ImportErrorKind,
}

impl ImportError {
    pub(crate) const fn new(kind: ImportErrorKind) -> Self {
        Self { node: None, kind }
    }

    /// Records the node being imported, as `name (OpType)` or `OpType #index`.
    pub(crate) fn at_node(mut self, node: String) -> Self {
        if self.node.is_none() {
            self.node = Some(node);
        }
        self
    }

    /// The node it happened at, if any.
    pub fn node(&self) -> Option<&str> {
        self.node.as_deref()
    }

    /// What went wrong.
    pub const fn kind(&self) -> &ImportErrorKind {
        &self.kind
    }

    /// Why the model is outside what ptxc supports, if that is the problem
    /// rather than a malformed model.
    pub const fn unsupported(&self) -> Option<&Unsupported> {
        match &self.kind {
            ImportErrorKind::Unsupported(reason) => Some(reason),
            _ => None,
        }
    }
}

impl From<ImportErrorKind> for ImportError {
    fn from(kind: ImportErrorKind) -> Self {
        Self::new(kind)
    }
}

impl From<Unsupported> for ImportError {
    fn from(reason: Unsupported) -> Self {
        Self::new(ImportErrorKind::Unsupported(reason))
    }
}

/// What went wrong importing a model.
#[derive(Debug, Error)]
pub enum ImportErrorKind {
    /// The model is valid ONNX but uses something ptxc does not support.
    #[error("unsupported: {0}")]
    Unsupported(Unsupported),
    /// The bytes are not a protobuf `ModelProto`.
    #[error("not an ONNX model: {0}")]
    Decode(#[from] prost::DecodeError),
    /// A field ONNX requires is absent.
    #[error("missing {0}")]
    Missing(&'static str),
    /// A field holds a value ONNX does not allow.
    #[error("{field}: {problem}")]
    Invalid {
        /// The field.
        field: &'static str,
        /// What is wrong with it.
        problem: String,
    },
    /// A name that no input, initializer or earlier node defines.
    #[error("`{0}` is not defined before it is used")]
    UndefinedValue(String),
    /// A shape-like operand that is not known at import. With static shapes
    /// it must be (D32): the caller can bind `inputs`, the graph inputs it
    /// is computed from, through `ImportOptions::constant_inputs`.
    #[error("`{operand}` must be known at import; it depends on inputs {inputs:?}")]
    NeedsConstant {
        /// The operand.
        operand: String,
        /// The graph inputs it is computed from.
        inputs: Vec<String>,
    },
    /// A tensor whose data does not fill its dims.
    #[error("tensor `{name}` has {actual} elements of data for {expected}")]
    DataLength {
        /// The tensor's name.
        name: String,
        /// The elements its dims hold.
        expected: u64,
        /// The elements its data holds.
        actual: u64,
    },
    /// External tensor data that cannot be read.
    #[error("reading external data {}: {source}", .path.display())]
    ExternalData {
        /// The file.
        path: PathBuf,
        /// Why it could not be read.
        source: std::io::Error,
    },
    /// A node the graph's type rules reject.
    #[error(transparent)]
    Node(#[from] NodeError),
    /// A declared shape that differs from the inferred one.
    #[error("`{name}` is declared as {declared} but has type {inferred}")]
    DeclaredType {
        /// The value.
        name: String,
        /// The declared type, as graph IR text.
        declared: String,
        /// The inferred type, as graph IR text.
        inferred: String,
    },
}

/// Why a valid model is outside what ptxc supports. These are the reasons
/// the conformance report counts (TESTING.md).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum Unsupported {
    /// An op ptxc does not implement.
    #[error("op {domain}{op}", domain = if .domain.is_empty() { String::new() } else { format!("{}.", .domain) })]
    Op {
        /// The op's domain; empty for the default domain.
        domain: String,
        /// The op type.
        op: String,
    },
    /// An element type outside ptxc's (D5), by its ONNX name.
    #[error("element type {0}")]
    DType(String),
    /// An attribute value, or an optional input or output, ptxc does not
    /// support.
    #[error("{op} attribute {attribute}")]
    Attribute {
        /// The op type.
        op: String,
        /// The attribute, input or output.
        attribute: String,
    },
    /// A default-domain opset below 13 (D28).
    #[error("opset {0}")]
    Opset(i64),
    /// `If`, `Loop` or `Scan` (D18).
    #[error("control flow ({0})")]
    ControlFlow(String),
    /// A symbolic dimension with no size supplied (D3).
    #[error("dynamic dimension `{0}`")]
    DynamicShape(String),
}
