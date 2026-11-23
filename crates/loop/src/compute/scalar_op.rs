//! The scalar operations a tensor expression applies to elements: the
//! primitives every v0.1.0 graph op lowers onto. Activations, softmax and
//! normalisation are compositions of these, built by `ptxc-lower`.

/// An operation on one element.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    /// `|x|`.
    Abs,
    /// `-x`.
    Neg,
    /// `e^x`.
    Exp,
    /// The natural logarithm.
    Log,
    /// The square root.
    Sqrt,
    /// `1 / x`.
    Reciprocal,
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceil,
    /// Round to nearest, ties to even.
    Round,
    /// -1, 0 or 1 (and NaN for NaN).
    Sign,
    /// The error function.
    Erf,
    /// The sine.
    Sin,
    /// The cosine.
    Cos,
    /// The hyperbolic tangent.
    Tanh,
    /// Logical not, on booleans.
    Not,
    /// Whether a float is NaN.
    IsNaN,
    /// Whether a float is an infinity of the given sign.
    IsInf(InfSign),
}

/// Which infinities `IsInf` detects.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InfSign {
    /// `+inf` or `-inf`.
    Either,
    /// `+inf` only.
    Positive,
    /// `-inf` only.
    Negative,
}

/// An operation on two elements of the same type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    /// `a + b`; integers wrap.
    Add,
    /// `a - b`; integers wrap.
    Sub,
    /// `a * b`; integers wrap.
    Mul,
    /// `a / b`; integers truncate toward zero.
    Div,
    /// `a` to the power `b`, on floats.
    Pow,
    /// The larger; NaN if either is NaN, as NumPy's `maximum`.
    Max,
    /// The smaller; NaN if either is NaN, as NumPy's `minimum`.
    Min,
    /// The remainder with the dividend's sign: C's `fmod`, and `%` on
    /// integers.
    TruncatedRem,
    /// The remainder with the divisor's sign, as Python's `%`, on integers.
    FlooredRem,
    /// `a == b`, giving a boolean.
    Equal,
    /// `a < b`, giving a boolean.
    Less,
    /// `a <= b`, giving a boolean.
    LessOrEqual,
    /// `a > b`, giving a boolean.
    Greater,
    /// `a >= b`, giving a boolean.
    GreaterOrEqual,
    /// Logical and, on booleans.
    And,
    /// Logical or, on booleans.
    Or,
    /// Logical exclusive or, on booleans.
    Xor,
}

/// How a reduction combines the elements along its axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduceOp {
    /// The sum, from 0.
    Sum,
    /// The product, from 1.
    Prod,
    /// The largest, from the type's lowest value; NaN wins.
    Max,
    /// The smallest, from the type's highest value; NaN wins.
    Min,
}

/// Which extreme an arg-reduction looks for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArgKind {
    /// The position of the largest element.
    Max,
    /// The position of the smallest element.
    Min,
}

/// Which position an arg-reduction keeps among equal extremes. The first
/// NaN always wins, as in NumPy's `argmax`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TieBreak {
    /// The first.
    First,
    /// The last.
    Last,
}

impl UnaryOp {
    /// The operation's name in text and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Reciprocal => "reciprocal",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Sign => "sign",
            Self::Erf => "erf",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Not => "not",
            Self::IsNaN => "isnan",
            Self::IsInf(_) => "isinf",
        }
    }
}

impl InfSign {
    /// The sign's name in text.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Either => "either",
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }
}

impl BinaryOp {
    /// The operation's name in text and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::Max => "max",
            Self::Min => "min",
            Self::TruncatedRem => "truncated_rem",
            Self::FlooredRem => "floored_rem",
            Self::Equal => "eq",
            Self::Less => "lt",
            Self::LessOrEqual => "le",
            Self::Greater => "gt",
            Self::GreaterOrEqual => "ge",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }
}

impl ReduceOp {
    /// The reduction's name in text and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Max => "reduce_max",
            Self::Min => "reduce_min",
        }
    }
}

impl ArgKind {
    /// The arg-reduction's name in text and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Max => "argmax",
            Self::Min => "argmin",
        }
    }
}

impl TieBreak {
    /// The tie-break's name in text.
    pub const fn name(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Last => "last",
        }
    }
}
