//! Host tensors: a `Shape` with typed element storage, `TensorData`.

use crate::dtype::DType;
use crate::float::{Bf16, F8E4M3Fn, F8E5M2, F16};
use crate::shape::Shape;
use crate::size::ElementCount;
use thiserror::Error;

/// A shape and data that do not describe one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum TensorError {
    /// The data holds a different number of elements than the shape.
    #[error("shape has {} elements but the data holds {data_len}", .shape_elements.get())]
    LengthMismatch {
        /// The shape's element count.
        shape_elements: ElementCount,
        /// The number of elements in the data.
        data_len: usize,
    },
}

/// Declares `TensorData`, with one variant per `DType` holding that type's
/// storage, and a `From<Vec<_>>` conversion for each.
macro_rules! tensor_data {
    ($($variant:ident($element:ty)),+ $(,)?) => {
        /// The elements of a tensor in row-major order. The variant is the
        /// element type, so the type and the storage cannot disagree.
        #[derive(Clone, Debug)]
        pub enum TensorData {
            $(
                #[doc = concat!("Elements of [`DType::", stringify!($variant), "`].")]
                $variant(Vec<$element>),
            )+
        }

        impl TensorData {
            /// The element type.
            pub const fn dtype(&self) -> DType {
                match self {
                    $(Self::$variant(_) => DType::$variant,)+
                }
            }
        }

        $(
            impl From<Vec<$element>> for TensorData {
                fn from(elements: Vec<$element>) -> Self {
                    Self::$variant(elements)
                }
            }
        )+
    };
}

tensor_data!(
    F32(f32),
    F16(F16),
    Bf16(Bf16),
    F8E4M3Fn(F8E4M3Fn),
    F8E5M2(F8E5M2),
    I8(i8),
    U8(u8),
    I32(i32),
    I64(i64),
    Bool(bool),
);

/// Runs `$body` with `$elements` bound to the typed `Vec` inside a
/// `TensorData` (or a reference to one), whatever its element type.
///
/// ```
/// use ptxc_core::{TensorData, with_tensor_data};
///
/// let data = TensorData::from(vec![1_i32, 2, 3]);
/// assert_eq!(with_tensor_data!(&data, elements => elements.len()), 3);
/// ```
#[macro_export]
macro_rules! with_tensor_data {
    ($data:expr, $elements:ident => $body:expr) => {
        match $data {
            $crate::TensorData::F32($elements) => $body,
            $crate::TensorData::F16($elements) => $body,
            $crate::TensorData::Bf16($elements) => $body,
            $crate::TensorData::F8E4M3Fn($elements) => $body,
            $crate::TensorData::F8E5M2($elements) => $body,
            $crate::TensorData::I8($elements) => $body,
            $crate::TensorData::U8($elements) => $body,
            $crate::TensorData::I32($elements) => $body,
            $crate::TensorData::I64($elements) => $body,
            $crate::TensorData::Bool($elements) => $body,
        }
    };
}

/// A tensor in host memory: graph constants, interpreter inputs and
/// outputs, and ONNX test data.
#[derive(Clone, Debug)]
pub struct HostTensor {
    shape: Shape,
    data: TensorData,
}

impl HostTensor {
    /// A tensor of `shape` holding `data` in row-major order.
    pub fn new(shape: Shape, data: TensorData) -> Result<Self, TensorError> {
        let shape_elements = shape.element_count();
        let data_len = with_tensor_data!(&data, elements => elements.len());
        if usize::try_from(shape_elements.get()) != Ok(data_len) {
            return Err(TensorError::LengthMismatch { shape_elements, data_len });
        }
        Ok(Self { shape, data })
    }

    /// The shape.
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The elements.
    pub const fn data(&self) -> &TensorData {
        &self.data
    }

    /// The element type.
    pub const fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// The shape and the elements, giving up the tensor.
    pub fn into_parts(self) -> (Shape, TensorData) {
        (self.shape, self.data)
    }
}
