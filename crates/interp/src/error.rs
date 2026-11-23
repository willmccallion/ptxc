//! Why an evaluation failed.

use ptxc_core::{DType, TensorError};
use ptxc_graph::{CycleError, ValueId};
use thiserror::Error;

/// Why the interpreter could not evaluate a graph.
#[derive(Debug, Error)]
pub enum InterpError {
    /// The caller passed the wrong number of inputs.
    #[error("the graph takes {expected} inputs, but {actual} were given")]
    InputCount {
        /// The graph's input count.
        expected: usize,
        /// The number passed.
        actual: usize,
    },
    /// An input's type differs from the graph's declaration.
    #[error("input {index} is {actual}, but the graph declares {expected}")]
    InputType {
        /// The input's position.
        index: usize,
        /// The declared type, as graph IR text.
        expected: String,
        /// The given tensor's type, as graph IR text.
        actual: String,
    },
    /// The graph has a cycle, so it has no evaluation order.
    #[error(transparent)]
    Cycle(#[from] CycleError),
    /// An integer division or remainder by zero, which ONNX leaves undefined.
    #[error("{op} divides an integer by zero")]
    IntegerDivisionByZero {
        /// The op.
        op: &'static str,
    },
    /// A gather index lies outside the axis it indexes, which ONNX leaves
    /// undefined.
    #[error("index {index} is outside an axis of extent {extent}")]
    IndexOutOfRange {
        /// The index as given.
        index: i64,
        /// The axis's extent.
        extent: usize,
    },
    /// A tensor is too large to index on this machine.
    #[error("a tensor of {elements} elements is too large to index")]
    TooLarge {
        /// Its element count.
        elements: u64,
    },
    /// A value was read before it was computed, or a tensor's element type
    /// differs from its graph type: the graph broke an invariant its
    /// verifier checks.
    #[error("the graph is malformed: {0}")]
    Malformed(Malformed),
}

/// How a graph broke an invariant the interpreter relies on.
#[derive(Debug, Error)]
pub enum Malformed {
    /// A value has no tensor when it is read.
    #[error("{0:?} is read before it is defined")]
    Undefined(ValueId),
    /// An op received an element type its type rule rejects.
    #[error("an op received a {0:?} tensor its type rule rejects")]
    DType(DType),
    /// An op received fewer operands than its type rule requires.
    #[error("an op received fewer operands than its type rule requires")]
    Operands,
    /// A quantized op's grid varies along an axis its arithmetic sums over,
    /// which ONNX's quantized ops never allow.
    #[error("a quantized op's grid varies along an axis it sums over")]
    Granularity,
    /// An op's attributes reach outside its operand.
    #[error("an op's attributes reach outside its operand")]
    OutOfBounds,
    /// An op's result does not fill the type its rule inferred.
    #[error("a result does not fill its type: {0}")]
    Result(#[from] TensorError),
}

impl From<Malformed> for InterpError {
    fn from(malformed: Malformed) -> Self {
        Self::Malformed(malformed)
    }
}
