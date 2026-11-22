//! Shape-manipulation ops and their type rules. With static shapes (D3),
//! the shape-like operands ONNX takes as tensors (target shapes, slice
//! bounds, pads, repeats, split sizes) are constants at import and become
//! attributes here. ONNX's `Squeeze`, `Unsqueeze` and `Flatten` only change
//! the shape, so the importer turns them into `Reshape`, as XLA's HLO has
//! only reshape.

mod pad;
mod slice;
mod view;

pub use pad::{AxisPadding, DiagonalOffset, PadAttrs, PadMode, TriangularPart, TriluAttrs};
pub(in crate::ops) use slice::infer_concat;
pub use slice::{SliceAttrs, SliceAxis, SliceStep, SplitAttrs};
pub use view::{RepeatCount, TileAttrs, TransposeAttrs};
pub(in crate::ops) use view::{infer_expand, infer_reshape};
