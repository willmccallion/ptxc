//! Block-local value numbering: registers that already hold an index, an
//! element's address or a value, reused rather than computed again.
//!
//! Index expressions read only variables, so what they hold stays valid
//! for the rest of the block. A value may load from memory, so a store
//! forgets the values that load the buffer stored to, and a barrier or a
//! wait for asynchronous copies, after which other threads' stores or the
//! copies are visible, forgets every value. Entries made inside a loop or
//! branch body do not dominate the code after it, and a loop's later
//! iterations may see memory its body stored, so bodies are scoped.

use super::memory::Place;
use super::values::Held;
use crate::ast::{B64, Reg};
use ptxc_loop::{BufferId, ExprId, ValueExpr};
use std::collections::HashMap;

/// What earlier instructions of the current block left in registers.
#[derive(Clone, Default)]
pub(super) struct Known {
    indices: HashMap<ExprId, Reg<B64>>,
    addresses: HashMap<(BufferId, ExprId), Place>,
    values: HashMap<ValueExpr, Held>,
}

impl Known {
    pub(super) fn index(&self, id: ExprId) -> Option<Reg<B64>> {
        self.indices.get(&id).copied()
    }

    pub(super) fn remember_index(&mut self, id: ExprId, held: Reg<B64>) {
        let _previous = self.indices.insert(id, held);
    }

    pub(super) fn address(&self, buffer: BufferId, index: ExprId) -> Option<Place> {
        self.addresses.get(&(buffer, index)).cloned()
    }

    pub(super) fn remember_address(&mut self, buffer: BufferId, index: ExprId, at: Place) {
        let _previous = self.addresses.insert((buffer, index), at);
    }

    pub(super) fn value(&self, expr: &ValueExpr) -> Option<Held> {
        self.values.get(expr).copied()
    }

    pub(super) fn remember_value(&mut self, expr: &ValueExpr, held: Held) {
        let _previous = self.values.insert(expr.clone(), held);
    }

    /// Forgets the values that load from `buffer`, after a store to it.
    pub(super) fn forget_reads_of(&mut self, buffer: BufferId) {
        self.values.retain(|expr, _| !expr.reads(buffer));
    }

    /// Forgets every value, after any memory may have changed.
    pub(super) fn forget_values(&mut self) {
        self.values.clear();
    }

    /// Returns to `outer`, what was known before a loop or branch body,
    /// less any value the body may have stored over.
    pub(super) fn leave(&mut self, outer: Self) {
        *self = outer;
        self.forget_values();
    }
}

#[cfg(test)]
mod tests {
    use super::Known;
    use crate::ast::{Address, B64, F32, Reg, Space};
    use crate::select::memory::Place;
    use crate::select::values::Held;
    use ptxc_core::Idx;
    use ptxc_loop::{BufferId, ExprArena, ExprId, ValueExpr};

    fn load(buffer: usize) -> ValueExpr {
        ValueExpr::Load { buffer: BufferId::new(buffer), index: ExprArena::new().constant(0) }
    }

    fn holds(known: &Known, expr: &ValueExpr, index: u32) -> bool {
        matches!(known.value(expr), Some(Held::Float(reg)) if reg == Reg::<F32>::new(index))
    }

    fn knowing(indices: &[ExprId], values: &[(ValueExpr, u32)]) -> Known {
        let mut known = Known::default();
        for (n, &id) in (0..).zip(indices) {
            known.remember_index(id, Reg::new(n));
        }
        for (expr, index) in values {
            known.remember_value(expr, Held::Float(Reg::new(*index)));
        }
        known
    }

    #[test]
    fn what_is_remembered_is_found_and_nothing_else() {
        let (first, second) = (ExprId::new(0), ExprId::new(1));
        let at =
            Place::Memory(Space::Global, Address::Reg { base: Reg::<B64>::new(5).id(), offset: 0 });
        let mut known = knowing(&[first], &[(load(0), 7)]);

        known.remember_address(BufferId::new(2), first, at.clone());

        assert_eq!(known.index(first), Some(Reg::new(0)));
        assert_eq!(known.index(second), None);
        assert_eq!(known.address(BufferId::new(2), first), Some(at));
        assert_eq!(known.address(BufferId::new(3), first), None);
        assert!(holds(&known, &load(0), 7));
        assert!(known.value(&load(1)).is_none());
    }

    #[test]
    fn a_store_forgets_only_the_values_that_read_its_buffer() {
        let index = ExprId::new(0);
        let mut known = knowing(&[index], &[(load(0), 1), (load(1), 2)]);

        known.forget_reads_of(BufferId::new(0));

        assert!(known.value(&load(0)).is_none());
        assert!(holds(&known, &load(1), 2));
        assert_eq!(known.index(index), Some(Reg::new(0)));
    }

    #[test]
    fn forgetting_values_keeps_indices() {
        let index = ExprId::new(0);
        let mut known = knowing(&[index], &[(load(0), 1), (load(1), 2)]);

        known.forget_values();

        assert!(known.value(&load(0)).is_none() && known.value(&load(1)).is_none());
        assert_eq!(known.index(index), Some(Reg::new(0)));
    }

    #[test]
    fn leaving_a_body_drops_what_it_learned_and_every_value() {
        let (before, inside) = (ExprId::new(0), ExprId::new(1));
        let mut known = knowing(&[before], &[(load(0), 1)]);
        let outer = known.clone();
        known.remember_index(inside, Reg::new(9));

        known.leave(outer);

        assert_eq!(known.index(before), Some(Reg::new(0)));
        assert_eq!(known.index(inside), None);
        assert!(known.value(&load(0)).is_none());
    }
}
