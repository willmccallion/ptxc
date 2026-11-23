//! Size and offset newtypes: `ElementCount`, `ByteSize`, `ByteOffset` and
//! `Alignment`.

use crate::dtype::DType;
use thiserror::Error;

/// A size or offset computation that does not fit, or an invalid alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum SizeError {
    /// `count` elements of `dtype` take more than `u64::MAX` bytes.
    #[error("{} elements of {dtype:?} take more than 2^64 bytes", .count.get())]
    ByteSizeOverflow {
        /// The element count.
        count: ElementCount,
        /// The element type.
        dtype: DType,
    },
    /// Rounding `offset` up to `alignment` passes `u64::MAX`.
    #[error("aligning offset {} to {} bytes passes 2^64", .offset.get(), .alignment.get())]
    OffsetOverflow {
        /// The offset being aligned.
        offset: ByteOffset,
        /// The alignment.
        alignment: Alignment,
    },
    /// An alignment that is not a power of two.
    #[error("alignment {0} is not a power of two")]
    NotPowerOfTwo(u64),
}

/// A number of tensor elements.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ElementCount(u64);

impl ElementCount {
    /// Wraps a count.
    pub const fn new(count: u64) -> Self {
        Self(count)
    }

    /// The count.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The count as the nearest `f32`, for averages; exact below 2^24.
    #[expect(clippy::cast_precision_loss, reason = "rounds to nearest, as an average needs")]
    pub const fn to_f32(self) -> f32 {
        self.0 as f32
    }

    /// The bytes `self` elements of `dtype` occupy.
    pub const fn in_bytes(self, dtype: DType) -> Result<ByteSize, SizeError> {
        match self.0.checked_mul(dtype.element_size().get()) {
            Some(bytes) => Ok(ByteSize(bytes)),
            None => Err(SizeError::ByteSizeOverflow { count: self, dtype }),
        }
    }
}

/// A size in bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ByteSize(u64);

impl ByteSize {
    /// Wraps a size.
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// The size.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// An offset in bytes from the start of a buffer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ByteOffset(u64);

impl ByteOffset {
    /// Wraps an offset.
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// The offset.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The smallest offset at or after `self` that is a multiple of
    /// `alignment`.
    pub const fn align_up(self, alignment: Alignment) -> Result<Self, SizeError> {
        let mask = alignment.0 - 1;
        match self.0.checked_add(mask) {
            Some(bumped) => Ok(Self(bumped & !mask)),
            None => Err(SizeError::OffsetOverflow { offset: self, alignment }),
        }
    }
}

/// A byte alignment, always a power of two.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Alignment(u64);

impl Alignment {
    /// Parses an alignment in bytes.
    pub const fn new(bytes: u64) -> Result<Self, SizeError> {
        if bytes.is_power_of_two() { Ok(Self(bytes)) } else { Err(SizeError::NotPowerOfTwo(bytes)) }
    }

    /// The alignment in bytes.
    pub const fn get(self) -> u64 {
        self.0
    }
}
