//! `MathProgram`: straight-line code over float, integer and predicate
//! slots. Each step defines the next slot of its kind, so an operand of the
//! wrong kind cannot be written.

use ptxc_core::define_index;

define_index!(
    /// An f32 value a step defines.
    pub FloatSlot
);

define_index!(
    /// A 32-bit integer value a step defines, read as two's complement
    /// where a sign matters.
    pub IntSlot
);

define_index!(
    /// A boolean value a step defines.
    pub PredSlot
);

/// An IEEE f32 operation, rounding to nearest even with subnormals kept.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatOp {
    /// `add.rn.f32`.
    Add,
    /// `sub.rn.f32`.
    Sub,
    /// `mul.rn.f32`.
    Mul,
    /// `div.rn.f32`.
    Div,
}

/// A 32-bit integer operation. Additions, subtractions and products wrap;
/// a shift by 32 or more gives 0, or the sign bit everywhere for an
/// arithmetic right shift, as PTX clamps shift amounts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntOp {
    /// Wrapping addition.
    Add,
    /// Wrapping subtraction.
    Sub,
    /// The low 32 bits of the product.
    Mul,
    /// The high 32 bits of the unsigned product.
    MulHiUnsigned,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Shift left.
    Shl,
    /// Shift right, copying the sign bit.
    ShrArithmetic,
    /// Shift right, filling with zeros.
    ShrLogical,
}

/// A comparison: ordered on floats, so false when either is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Comparison {
    /// Less than.
    Lt,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Equal; `+0` equals `-0`.
    Eq,
}

/// A step defining a float.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatDef {
    /// The program's argument.
    Argument,
    /// A constant, as its bits.
    Const(u32),
    /// `a op b`.
    Binary(FloatOp, FloatSlot, FloatSlot),
    /// `a * b + c`, rounded once.
    Fma(FloatSlot, FloatSlot, FloatSlot),
    /// The nearest integer, ties to even, as a float.
    RoundToInteger(FloatSlot),
    /// A two's complement integer, rounded to nearest even.
    FromInt(IntSlot),
    /// An unsigned integer, rounded to nearest even.
    FromUnsigned(IntSlot),
    /// The float whose bits these are.
    FromBits(IntSlot),
    /// The first float where the predicate holds, else the second.
    Select(PredSlot, FloatSlot, FloatSlot),
}

/// A step defining an integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntDef {
    /// A constant.
    Const(u32),
    /// An integral float as two's complement, saturating; NaN gives 0.
    FromFloat(FloatSlot),
    /// A float's bits.
    Bits(FloatSlot),
    /// The number of leading zero bits, 32 for zero.
    LeadingZeros(IntSlot),
    /// `a op b`.
    Binary(IntOp, IntSlot, IntSlot),
    /// The first integer where the predicate holds, else the second.
    Select(PredSlot, IntSlot, IntSlot),
}

/// A step defining a predicate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredDef {
    /// Two floats compared.
    CompareFloat(Comparison, FloatSlot, FloatSlot),
    /// Two integers compared, signed.
    CompareInt(Comparison, IntSlot, IntSlot),
    /// Two integers compared, unsigned.
    CompareUnsigned(Comparison, IntSlot, IntSlot),
}

/// One step: the definition of the next slot of its kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Defines the next float slot.
    Float(FloatDef),
    /// Defines the next integer slot.
    Int(IntDef),
    /// Defines the next predicate slot.
    Pred(PredDef),
}

/// A function of one f32, as steps whose every operation is exact IEEE or
/// integer arithmetic.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MathProgram {
    pub(super) steps: Vec<Step>,
    pub(super) result: FloatSlot,
}

impl MathProgram {
    /// The steps, in order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The slot holding the result.
    pub const fn result(&self) -> FloatSlot {
        self.result
    }
}
