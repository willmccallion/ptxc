//! The asynchronous copy statements as text.

use super::kernel::KernelPrinter;
use crate::expr::ExprId;
use crate::stmt::{BufferId, Condition};
use ptxc_core::Dim;
use std::fmt::Write;

impl KernelPrinter<'_> {
    /// `%t[i] <- async %s[j] run 8 if a < b, c < d`.
    pub(super) fn async_copy(
        &self,
        out: &mut String,
        pad: &str,
        [(target, target_index), (source, source_index)]: [(BufferId, ExprId); 2],
        (run, bounds): (Dim, &[Condition]),
    ) {
        out.push_str(pad);
        self.access(out, target, target_index);
        out.push_str(" <- async ");
        self.access(out, source, source_index);
        let _infallible = write!(out, " run {}", run.get());
        for (position, condition) in bounds.iter().enumerate() {
            out.push_str(if position == 0 { " if " } else { ", " });
            self.index(out, condition.index);
            out.push_str(" < ");
            self.index(out, condition.bound);
        }
        out.push('\n');
    }
}
