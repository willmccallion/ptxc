//! What happened to one test, and why.

use serde::{Serialize, Serializer};
use std::fmt;

/// What happened to one test.
#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Outcome {
    /// Every data set's outputs matched.
    Pass,
    /// Something went wrong that ptxc should handle.
    Fail { failure: Failure },
    /// The test uses something outside ptxc's scope.
    Unsupported { reason: Reason },
}

/// Why a test is unsupported, as the importer reports it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Reason {
    /// An op ptxc does not import, prefixed by its domain outside the
    /// default one.
    Op(String),
    /// An element type ptxc does not support, in the model or its data.
    #[serde(rename = "dtype")]
    DType(String),
    /// An attribute value of an op ptxc does not support.
    Attribute { op: String, attribute: String },
    /// An opset below 13.
    Opset(i64),
    /// `If`, `Loop` or `Scan`.
    ControlFlow(String),
    /// A symbolic dimension with no size given.
    DynamicShape(String),
    /// A shape computed from values no input fixes at import.
    ValueDependentShape,
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Op(op) => write!(f, "op {op}"),
            Self::DType(dtype) => write!(f, "element type {dtype}"),
            Self::Attribute { op, attribute } => write!(f, "{op} attribute {attribute}"),
            Self::Opset(version) => write!(f, "opset {version}"),
            Self::ControlFlow(op) => write!(f, "control flow ({op})"),
            Self::DynamicShape(dim) => write!(f, "dynamic shape ({dim})"),
            Self::ValueDependentShape => write!(f, "value-dependent shape"),
        }
    }
}

/// Why a test failed.
#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Failure {
    /// An output differed from the expected one.
    Mismatch {
        data_set: usize,
        output: usize,
        message: String,
        worst_absolute: Option<Worst>,
        worst_relative: Option<Worst>,
    },
    /// The backend produced the wrong number of outputs.
    OutputCount { data_set: usize, expected: usize, actual: usize },
    /// The test's files could not be read.
    Load { message: String },
    /// The model failed to import for a reason that is not "unsupported".
    Import { message: String },
    /// The backend failed to evaluate the graph.
    Evaluate { message: String },
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { data_set, output, message, .. } => {
                write!(f, "data set {data_set}, output {output}: {message}")
            }
            Self::OutputCount { data_set, expected, actual } => {
                write!(f, "data set {data_set}: {actual} outputs where {expected} were expected")
            }
            Self::Load { message } => write!(f, "loading: {message}"),
            Self::Import { message } => write!(f, "importing: {message}"),
            Self::Evaluate { message } => write!(f, "evaluating: {message}"),
        }
    }
}

/// The worst element by one measure. Values are written as strings, since
/// JSON has no infinities or NaN.
#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Worst {
    pub(crate) position: Vec<u64>,
    #[serde(serialize_with = "as_text")]
    pub(crate) actual: f64,
    #[serde(serialize_with = "as_text")]
    pub(crate) expected: f64,
    #[serde(serialize_with = "as_text")]
    pub(crate) error: f64,
}

#[expect(clippy::trivially_copy_pass_by_ref, reason = "serde's serialize_with passes a reference")]
fn as_text<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(value)
}
