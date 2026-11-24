//! Selects asynchronous copies as `cp.async`, and their commits and waits.

use super::{CodegenError, Selector};
use crate::ast::{B32, Compare, CopySize, PtxType, Reg, Signedness, Value};
use ptxc_core::Dim;
use ptxc_loop::{BufferId, Condition, ExprId};

impl Selector<'_> {
    /// Starts copying `run` elements from `source` to `target`, zero where
    /// one of `bounds` fails; verification makes both runs start aligned to
    /// the copy's size.
    pub(super) fn async_copy(
        &mut self,
        [(target, target_index), (source, source_index)]: [(BufferId, ExprId); 2],
        (run, bounds): (Dim, &[Condition]),
    ) -> Result<(), CodegenError> {
        let dtype = self.kernel.buffers()[target].dtype;
        let size = CopySize::of(run.get() * dtype.element_size().get())
            .ok_or(CodegenError::Unsupported { op: "a copy of other than 4, 8 or 16 bytes" })?;
        let copied = self.copied_bytes(size, bounds)?;
        let (_, src) = self.memory_element(source, source_index)?;
        let (_, dst) = self.memory_element(target, target_index)?;
        self.builder.copy_async(size, (dst, src), copied);
        Ok(())
    }

    /// Waits for the copies of all but the newest `pending` groups, after
    /// which their elements may differ from any value loaded before.
    pub(super) fn wait_copies(&mut self, pending: u32) {
        self.builder.wait_copies(pending);
        self.known.forget_values();
    }

    /// The bytes of a copy of `size` taken from its source: all where
    /// every one of `bounds` holds, else none; all, without bounds.
    fn copied_bytes(
        &mut self,
        size: CopySize,
        bounds: &[Condition],
    ) -> Result<Option<Reg<B32>>, CodegenError> {
        let all = i32::try_from(size.bytes()).map_err(|_| CodegenError::TooLarge)?;
        let mut copied = None;
        for condition in bounds {
            let index = self.index(condition.index)?;
            let bound = self.index_operand(condition.bound)?;
            let inside =
                self.builder.compare_int(Compare::Lt, Signedness::Signed, index.into(), bound);
            let so_far = copied.map_or_else(|| Value::<B32>::int(all), Value::from);
            let none = Value::<B32>::int(0);
            copied = Some(self.builder.select(PtxType::B32, so_far, none, inside));
        }
        Ok(copied)
    }
}
