//! Elementwise ops over any number of tensors, and `Where`.

use super::{broadcast, broadcast_same_dtype};
use crate::ops::accepts::Accepts;
use crate::ops::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};

/// An elementwise op over one or more tensors of one element type, all
/// broadcast together.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VariadicOp {
    /// The largest operand.
    Max,
    /// The smallest operand.
    Min,
    /// The sum of the operands.
    Sum,
    /// The mean of the operands.
    Mean,
}

impl VariadicOp {
    const fn accepts(self) -> Accepts {
        match self {
            Self::Max | Self::Min => Accepts::Numeric,
            Self::Sum | Self::Mean => Accepts::Float,
        }
    }
}

impl TypeRule for VariadicOp {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let (dtype, shape) = broadcast_same_dtype(operands)?;
        self.accepts().check(dtype)?;
        Ok(vec![TensorType::new(dtype, shape, Layout::RowMajor)])
    }
}

/// `Where(condition, x, y)`: `x` where the bool condition holds, else `y`,
/// all three broadcast together.
pub(in crate::ops) fn infer_where(operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
    let [condition, on_true, on_false] = exact_operands(operands)?;
    Accepts::Bool.check(condition.dtype())?;
    let (dtype, values) = broadcast_same_dtype(&[on_true, on_false])?;
    let shape = broadcast(condition.shape(), &values)?;
    Ok(vec![TensorType::new(dtype, shape, Layout::RowMajor)])
}
