//! `Cast`: conversion to another element type.

use crate::ops::{TypeError, TypeRule, exact_operands};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::DType;
use ptxc_core::float::Overflow;

/// ONNX `Cast`: converts every element to `to`. Any element type converts to
/// any other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CastAttrs {
    to: DType,
    overflow: Overflow,
}

impl CastAttrs {
    /// A cast to `to`. `overflow` is ONNX's `saturate` attribute, which only
    /// affects FP8 targets; for any other target it is normalised to
    /// `Saturate`, so equal casts compare equal.
    pub const fn new(to: DType, overflow: Overflow) -> Self {
        let overflow = match to {
            DType::F8E4M3Fn | DType::F8E5M2 => overflow,
            _ => Overflow::Saturate,
        };
        Self { to, overflow }
    }

    /// The target element type.
    pub const fn to(self) -> DType {
        self.to
    }

    /// What an out-of-range value becomes in an FP8 target.
    pub const fn overflow(self) -> Overflow {
        self.overflow
    }
}

impl TypeRule for CastAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Ok(vec![TensorType::new(self.to, input.shape().clone(), Layout::RowMajor)])
    }
}
