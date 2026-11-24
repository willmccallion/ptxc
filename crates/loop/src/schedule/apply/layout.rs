//! Applies `swizzle`.

use crate::compute::{ComputeDef, TensorParamId};
use crate::schedule::error::ScheduleError;
use crate::schedule::loop_tree::LoopTree;
use crate::stmt::Swizzle;

impl LoopTree {
    /// Places the bytes of `tensor`'s staged tile by `swizzle`.
    pub(crate) fn swizzle(
        &mut self,
        compute: &ComputeDef,
        tensor: TensorParamId,
        swizzle: Swizzle,
    ) -> Result<(), ScheduleError> {
        let input = compute.inputs().get(tensor).ok_or(ScheduleError::UnknownTensor(tensor))?;
        let cache = self
            .caches
            .iter_mut()
            .find(|cache| cache.tensor == tensor)
            .ok_or_else(|| ScheduleError::SwizzleUnstaged(input.name.clone()))?;
        if cache.swizzle.is_some() {
            return Err(ScheduleError::AlreadySwizzled(input.name.clone()));
        }
        cache.swizzle = Some(swizzle);
        Ok(())
    }
}
