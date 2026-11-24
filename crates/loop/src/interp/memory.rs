//! Global, shared and local memory spaces: each element unset until stored,
//! and a swizzled shared buffer's elements where its swizzle puts them.

use super::RunError;
use super::value::Value;
use crate::compute::TensorName;
use crate::stmt::Swizzle;

/// One buffer's elements, each `None` until first stored.
#[derive(Clone, Debug)]
pub(crate) struct Memory {
    name: TensorName,
    elements: Vec<Option<Value>>,
    /// The swizzle that places the elements, and their bytes.
    layout: Option<(Swizzle, u64)>,
}

impl Memory {
    /// `size` unset elements.
    pub(crate) fn unset(name: TensorName, size: usize) -> Self {
        Self { name, elements: vec![None; size], layout: None }
    }

    /// Elements set from `values`.
    pub(crate) fn of(name: TensorName, values: Vec<Value>) -> Self {
        Self { name, elements: values.into_iter().map(Some).collect(), layout: None }
    }

    /// `self`, its elements of `element_bytes` each placed by `swizzle`.
    pub(crate) const fn swizzled(mut self, swizzle: Swizzle, element_bytes: u64) -> Self {
        self.layout = Some((swizzle, element_bytes));
        self
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

    /// Where element `index` is held: its own slot, or where the swizzle
    /// puts its bytes.
    fn slot(&self, index: i64) -> Result<usize, RunError> {
        let out_of_bounds = || RunError::OutOfBounds { buffer: self.name.clone(), index };
        let logical = u64::try_from(index).map_err(|_| out_of_bounds())?;
        let physical = match self.layout {
            Some((swizzle, bytes)) => swizzle.apply(logical * bytes) / bytes,
            None => logical,
        };
        usize::try_from(physical)
            .ok()
            .filter(|&slot| slot < self.elements.len())
            .ok_or_else(out_of_bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::Memory;
    use crate::compute::TensorName;
    use crate::interp::value::Value;
    use crate::stmt::Swizzle;

    #[test]
    fn a_swizzled_buffer_holds_each_element_where_its_swizzle_puts_its_bytes() {
        // 64-byte rows of 32 halves: row 2's first 16 bytes move to its
        // second unit, so element 64 is held at element 72.
        let swizzle = Swizzle::for_rows(64).unwrap();
        let mut memory = Memory::unset(TensorName::new("tile"), 256).swizzled(swizzle, 2);

        memory.write(64, Value::Int(7)).unwrap();

        assert_eq!(memory.elements[72], Some(Value::Int(7)));
        assert_eq!(memory.elements[64], None);
        assert_eq!(memory.read(64).unwrap(), Value::Int(7));
    }
}
