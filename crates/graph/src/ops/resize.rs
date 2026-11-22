//! `Resize`: interpolation to new spatial sizes.

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::{TypeError, TypeRule, exact_operands, require_rank};
use crate::tensor_type::{Layout, TensorType};
use ptxc_core::{Dim, Rank, Shape};

/// How `Resize` computes an output element, from ONNX's `mode` and the
/// attributes that only apply to one mode.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ResizeMode {
    /// The nearest input element.
    Nearest(NearestRounding),
    /// Linear interpolation along every resized axis.
    Linear,
    /// Cubic interpolation along every resized axis.
    Cubic {
        /// ONNX's `cubic_coeff_a`.
        coefficient: F32Attr,
        /// ONNX's `exclude_outside`.
        outside: OutsideWeights,
    },
}

/// How a fractional source position picks the nearest element, from ONNX's
/// `nearest_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NearestRounding {
    /// Round half down.
    RoundPreferFloor,
    /// Round half up.
    RoundPreferCeil,
    /// Round down.
    Floor,
    /// Round up.
    Ceil,
}

/// Whether cubic taps that fall outside the input keep their weight.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OutsideWeights {
    /// `exclude_outside = 0`: they keep it.
    Include,
    /// `exclude_outside = 1`: they get zero and the rest are renormalised.
    Exclude,
}

/// How an output coordinate maps to an input coordinate, from ONNX's
/// `coordinate_transformation_mode`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CoordinateTransform {
    /// `half_pixel`.
    HalfPixel,
    /// `half_pixel_symmetric`.
    HalfPixelSymmetric,
    /// `pytorch_half_pixel`.
    PytorchHalfPixel,
    /// `align_corners`.
    AlignCorners,
    /// `asymmetric`.
    Asymmetric,
    /// `tf_crop_and_resize`, with its region of interest and the value used
    /// outside the input.
    TfCropAndResize {
        /// One `(start, end)` pair per input axis, in normalised coordinates.
        roi: Vec<[F32Attr; 2]>,
        /// ONNX's `extrapolation_value`.
        extrapolation: F32Attr,
    },
}

/// Whether downsampling stretches the filter, from ONNX's `antialias`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Antialias {
    /// No stretching.
    Off,
    /// Stretch linear and cubic filters by `max(1, 1 / scale)`.
    On,
}

/// One input axis of a resize: the output size and the scale the coordinate
/// transform uses. ONNX derives the scale from the sizes when a model gives
/// sizes, so both are recorded, resolved by the importer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ResizeAxis {
    /// The output extent.
    pub size: Dim,
    /// Output extent over input extent, as the coordinate transform sees it.
    pub scale: F32Attr,
}

/// ONNX `Resize` with constant scales, sizes and region of interest, so all
/// of them are attributes and the only operand is the input.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ResizeAttrs {
    /// How output elements are computed.
    pub mode: ResizeMode,
    /// How output coordinates map to input ones.
    pub coordinates: CoordinateTransform,
    /// Whether downsampling stretches the filter.
    pub antialias: Antialias,
    /// One entry per input axis; ONNX's `axes` is expanded by the importer.
    pub axes: Vec<ResizeAxis>,
}

impl TypeRule for ResizeAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let [input] = exact_operands(operands)?;
        Accepts::Numeric.check(input.dtype())?;
        let rank = Rank::new(self.axes.len());
        require_rank(input.shape(), rank)?;
        if let CoordinateTransform::TfCropAndResize { roi, .. } = &self.coordinates
            && roi.len() != self.axes.len()
        {
            return Err(TypeError::Rank { expected: rank, actual: Rank::new(roi.len()) });
        }
        let dims = self.axes.iter().map(|axis| axis.size).collect();
        Ok(vec![TensorType::new(input.dtype(), Shape::new(dims)?, Layout::RowMajor)])
    }
}
