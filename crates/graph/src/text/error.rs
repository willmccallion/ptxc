//! Parse errors with the line and column they were found at.

use crate::graph::NodeError;
use crate::tensor_type::TensorType;
use std::fmt;
use thiserror::Error;

/// A line and column in graph IR text, both from 1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    line: u32,
    column: u32,
}

impl Position {
    pub(super) const START: Self = Self { line: 1, column: 1 };

    /// The line, from 1.
    pub const fn line(self) -> u32 {
        self.line
    }

    /// The column, from 1.
    pub const fn column(self) -> u32 {
        self.column
    }

    pub(super) const fn after(self, c: char) -> Self {
        if c == '\n' {
            Self { line: self.line + 1, column: 1 }
        } else {
            Self { line: self.line, column: self.column + 1 }
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Graph IR text that does not parse.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{position}: {kind}")]
pub struct ParseError {
    position: Position,
    kind: ParseErrorKind,
}

impl ParseError {
    pub(super) const fn new(position: Position, kind: ParseErrorKind) -> Self {
        Self { position, kind }
    }

    /// Where the problem is.
    pub const fn position(&self) -> Position {
        self.position
    }

    /// What the problem is.
    pub const fn kind(&self) -> &ParseErrorKind {
        &self.kind
    }

    pub(super) const fn unexpected_char(position: Position, found: char) -> Self {
        Self::new(position, ParseErrorKind::UnexpectedChar(found))
    }

    pub(super) const fn empty_name(position: Position) -> Self {
        Self::new(position, ParseErrorKind::EmptyName)
    }

    pub(super) const fn bad_escape(position: Position) -> Self {
        Self::new(position, ParseErrorKind::BadEscape)
    }

    pub(super) const fn unterminated_name(position: Position) -> Self {
        Self::new(position, ParseErrorKind::UnterminatedName)
    }
}

/// What is wrong with graph IR text.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ParseErrorKind {
    /// A character no token starts with.
    #[error("unexpected character {0:?}")]
    UnexpectedChar(char),
    /// `%` with no name after it.
    #[error("`%` must be followed by a name")]
    EmptyName,
    /// A `\` in a quoted name not followed by `"` or `\`.
    #[error("only `\\\"` and `\\\\` may be escaped in a name")]
    BadEscape,
    /// A quoted name with no closing quote.
    #[error("unterminated quoted name")]
    UnterminatedName,
    /// A token other than the one the grammar needs.
    #[error("expected {expected}, found {found}")]
    Unexpected {
        /// What the grammar needs.
        expected: &'static str,
        /// What is there, or `end of input`.
        found: String,
    },
    /// A number that does not fit the type it is read as.
    #[error("{text:?} is not a valid {wanted}")]
    BadNumber {
        /// The number as written.
        text: String,
        /// The type it was read as.
        wanted: &'static str,
    },
    /// A word that names no op.
    #[error("unknown op `{0}`")]
    UnknownOp(String),
    /// A word that is not one of the values an attribute takes.
    #[error("`{found}` is not one of {expected}")]
    UnknownWord {
        /// The values allowed.
        expected: &'static str,
        /// The word written.
        found: String,
    },
    /// An op missing an attribute it needs.
    #[error("missing attribute `{0}`")]
    MissingAttribute(&'static str),
    /// An attribute the op does not take, or one given twice.
    #[error("unexpected attribute `{0}`")]
    UnexpectedAttribute(String),
    /// An attribute value of the wrong form or out of range.
    #[error("attribute `{key}`: {problem}")]
    BadAttribute {
        /// The attribute.
        key: &'static str,
        /// What is wrong with it.
        problem: &'static str,
    },
    /// A value used before it is defined.
    #[error("`%{0}` is not defined")]
    UndefinedValue(String),
    /// A value name defined twice.
    #[error("`%{0}` is defined twice")]
    Redefined(String),
    /// A constant whose data does not match its shape.
    #[error("constant has {actual} elements, but its type holds {expected}")]
    DataLength {
        /// The elements its type holds.
        expected: usize,
        /// The elements written.
        actual: usize,
    },
    /// An elided constant the caller did not supply.
    #[error("no supplied constant matches elided {0}")]
    ElidedConstant(String),
    /// A node the graph builder rejects.
    #[error(transparent)]
    Node(#[from] NodeError),
    /// A node whose written results differ from what it defines.
    #[error("the op defines {inferred:?}, but the text says {written:?}")]
    Results {
        /// The types the op's type rule infers.
        inferred: Vec<TensorType>,
        /// The types written.
        written: Vec<TensorType>,
    },
}
