//! Lowers a definition without a reduction: every output stored at each
//! spatial point, under every guard.

use super::context::Lowerer;
use crate::compute::Scalar;
use crate::stmt::{Stmt, ValueExpr};

impl Lowerer<'_> {
    /// The loops over every point, storing each output there.
    pub(super) fn elementwise(&mut self) -> Stmt {
        let position = self.output_position();
        let unused = ValueExpr::Const(Scalar::bool(false));
        let mut stores = Vec::new();
        for (output, &buffer) in self.compute.outputs().iter().zip(&self.outputs.clone()) {
            let value = self.value(&output.expr, &unused);
            let store = Stmt::Store { buffer, index: position, value };
            stores.push(self.wrap_loaded(&output.expr, store));
        }
        let guards = self.tree.guards().to_vec();
        let guarded = self.wrap_guards(&guards, Stmt::Seq(stores));
        self.wrap_loops(&self.serial_loops(), guarded)
    }
}
