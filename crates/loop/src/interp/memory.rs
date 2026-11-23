//! Global, shared and local memory spaces: each element unset until stored.

use super::RunError;
use super::value::Value;
use crate::compute::TensorName;

/// One buffer's elements, each `None` until first stored.
#[derive(Clone, Debug)]
pub(crate) struct Memory {
    name: TensorName,
    elements: Vec<Option<Value>>,
}

impl Memory {
    /// `size` unset elements.
    pub(crate) fn unset(name: TensorName, size: usize) -> Self {
        Self { name, elements: vec![None; size] }
    }

    /// Elements set from `values`.
    pub(crate) fn of(name: TensorName, values: Vec<Value>) -> Self {
        Self { name, elements: values.into_iter().map(Some).collect() }
    }

    /// The element at `index`, which must be in range and set.
    pub(crate) fn read(&self, index: i64) -> Result<Value, RunError> {
        let slot = self.slot(index)?;
        self.elements[slot].ok_or_else(|| RunError::Unset { buffer: self.name.clone(), index })
    }

    /// Sets the element at `index`, which must be in range.
    pub(crate) fn write(&mut self, index: i64, value: Value) -> Result<(), RunError> {
        let slot = self.slot(index)?;
        self.elements[slot] = Some(value);
        Ok(())
    }

    /// Every element, which must all be set.
    pub(crate) fn values(&self) -> Result<Vec<Value>, RunError> {
        self.elements
            .iter()
            .enumerate()
            .map(|(slot, element)| {
                element.ok_or_else(|| RunError::Unwritten {
                    buffer: self.name.clone(),
                    index: i64::try_from(slot).unwrap_or(i64::MAX),
                })
            })
            .collect()
    }

    fn slot(&self, index: i64) -> Result<usize, RunError> {
        usize::try_from(index)
            .ok()
            .filter(|&slot| slot < self.elements.len())
            .ok_or_else(|| RunError::OutOfBounds { buffer: self.name.clone(), index })
    }
}
