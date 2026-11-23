//! Row-major index arithmetic: a shape's extents and strides as `usize`,
//! multi-indices, and broadcasting.

use crate::error::InterpError;
use ptxc_core::Shape;

/// A shape's extents, strides and element count, checked once to fit
/// `usize` so every index computed from them does.
#[derive(Clone, Debug)]
pub(crate) struct Extents {
    dims: Vec<usize>,
    strides: Vec<usize>,
    count: usize,
}

impl Extents {
    pub(crate) fn of(shape: &Shape) -> Result<Self, InterpError> {
        let elements = shape.element_count().get();
        let too_large = || InterpError::TooLarge { elements };
        let _fits: usize = usize::try_from(elements).map_err(|_| too_large())?;
        let dims = shape
            .dims()
            .iter()
            .map(|dim| usize::try_from(dim.get()).map_err(|_| too_large()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_dims(dims))
    }

    /// Extents already in `usize`, such as a sub-shape of checked extents.
    pub(crate) fn from_dims(dims: Vec<usize>) -> Self {
        let mut strides = vec![1; dims.len()];
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * dims[axis + 1];
        }
        let count = dims.iter().product();
        Self { dims, strides, count }
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub(crate) const fn count(&self) -> usize {
        self.count
    }

    pub(crate) const fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The flat position of a multi-index.
    pub(crate) fn offset(&self, index: &[usize]) -> usize {
        index.iter().zip(&self.strides).map(|(i, stride)| i * stride).sum()
    }

    /// Every multi-index, row-major.
    pub(crate) fn indices(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.count).map(|flat| self.index_of(flat))
    }

    /// The multi-index of a flat position.
    pub(crate) fn index_of(&self, flat: usize) -> Vec<usize> {
        self.dims.iter().zip(&self.strides).map(|(&dim, &stride)| flat / stride % dim).collect()
    }

    /// The flat position this tensor's element has when broadcast to an
    /// output multi-index: trailing axes align, and extent-1 axes repeat.
    pub(crate) fn broadcast_offset(&self, out: &[usize]) -> usize {
        let skipped = out.len() - self.dims.len();
        self.dims
            .iter()
            .zip(&self.strides)
            .zip(&out[skipped..])
            .map(|((&dim, &stride), &i)| if dim == 1 { 0 } else { i * stride })
            .sum()
    }
}

/// Each output element of `f` applied to the operands' elements broadcast
/// to `out`.
pub(crate) fn broadcast_map<T: Copy, U>(
    operands: &[(&[T], &Extents)],
    out: &Extents,
    f: impl Fn(&[T]) -> U,
) -> Vec<U> {
    let mut args = Vec::with_capacity(operands.len());
    out.indices()
        .map(|index| {
            args.clear();
            args.extend(
                operands.iter().map(|(values, extents)| values[extents.broadcast_offset(&index)]),
            );
            f(&args)
        })
        .collect()
}
