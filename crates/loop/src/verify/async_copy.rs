//! The rules an asynchronous copy keeps: from a global input into an
//! allocated shared buffer of its type, in bounds, 4, 8 or 16 bytes at a
//! time, both runs starting at a multiple of their length, as `cp.async`
//! copies.

use super::{Scope, VerifyError, same_type};
use crate::expr::ExprId;
use crate::stmt::{BufferId, Condition, MemScope};
use ptxc_core::Dim;

impl Scope<'_> {
    pub(super) fn async_copy(
        &self,
        [(target, target_index), (source, source_index)]: [(BufferId, ExprId); 2],
        (run, bounds): (Dim, &[Condition]),
    ) -> Result<(), VerifyError> {
        for condition in bounds {
            self.index(condition.index)?;
            self.index(condition.bound)?;
        }
        let buffers = self.kernel.buffers();
        let (written, read) = (&buffers[target], &buffers[source]);
        if written.scope != MemScope::Shared {
            return Err(VerifyError::AsyncTarget(written.name.clone()));
        }
        if read.scope != MemScope::Global {
            return Err(VerifyError::AsyncSource(read.name.clone()));
        }
        let bytes = run.get().saturating_mul(written.dtype.element_size().get());
        if !matches!(bytes, 4 | 8 | 16) {
            return Err(VerifyError::AsyncRun { bytes });
        }
        let starts = [target_index, source_index];
        if !starts.iter().all(|&start| self.kernel.exprs().is_multiple_of(start, run.get())) {
            return Err(VerifyError::AsyncUnaligned(written.name.clone()));
        }
        let place = format!("an asynchronous copy to {}", written.name);
        same_type(&place, written.dtype, read.dtype)?;
        self.load(source, source_index)?;
        self.access_run(target, target_index, run)
    }
}
