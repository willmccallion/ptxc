//! A pipelined loop's tiles, filled `S - 1` iterations ahead for `S`
//! stages, as CUTLASS's multistage mainloop does. Each tile's buffer holds
//! `S` iterations' copies. Before the loop the block starts the first
//! `S - 1` iterations' copies, a group each. Each iteration waits for its
//! own group, passes a barrier, starts the copies of the iteration `S - 1`
//! ahead into the slot the previous iteration read, and computes on its
//! own slot. The barrier both shows every thread's landed copies to the
//! block and keeps the new copies from overwriting a slot still being read.

use super::cache::Tile;
use super::context::Lowerer;
use crate::expr::{ExprId, VarId};
use crate::schedule::StageCount;
use crate::stmt::{ForKind, Stmt};
use ptxc_core::Dim;

impl Lowerer<'_> {
    /// The loop over `var` to `extent`, running `body`, with `tiles`
    /// filled `stages - 1` iterations ahead.
    pub(super) fn pipelined(
        &mut self,
        (var, extent): (VarId, Dim),
        stages: StageCount,
        tiles: &[Tile],
        body: Stmt,
    ) -> Stmt {
        let ahead = stages.get() - 1;
        // The loop may run again inside an outer one, whose last iteration
        // may still be reading the slots the first fills write.
        let mut steps = vec![Stmt::Barrier];
        for iteration in 0..u64::from(ahead) {
            if iteration < extent.get() {
                let at = self.extent(Dim::new(iteration));
                steps.push(self.fill_iteration(tiles, var, at));
            }
            steps.push(Stmt::AsyncCommit);
        }
        let refill = self.refill(tiles, (var, extent), ahead);
        let iteration = vec![
            Stmt::AsyncWait { pending: ahead - 1 },
            Stmt::Barrier,
            refill,
            Stmt::AsyncCommit,
            body,
        ];
        steps.push(Stmt::For {
            var,
            extent,
            kind: ForKind::Serial,
            body: Box::new(Stmt::Seq(iteration)),
        });
        Stmt::Seq(steps)
    }

    /// Each iteration's copies of `tiles` for the iteration `ahead` later,
    /// where there is one; none in a loop the prologue fills entirely.
    fn refill(&mut self, tiles: &[Tile], (var, extent): (VarId, Dim), ahead: u32) -> Stmt {
        if u64::from(ahead) >= extent.get() {
            return Stmt::Seq(Vec::new());
        }
        let current = self.exprs.var(var);
        let offset = self.extent(Dim::new(u64::from(ahead)));
        let next = self.exprs.add(current, offset);
        let fills = self.fill_iteration(tiles, var, next);
        self.within(next, extent, fills)
    }

    /// The copies of `tiles` for the iteration where `var` is `at`, each
    /// into its slot for that iteration.
    fn fill_iteration(&mut self, tiles: &[Tile], var: VarId, at: ExprId) -> Stmt {
        let mut fills = Vec::new();
        for tile in tiles {
            let origin = tile
                .origin
                .iter()
                .map(|&axis| {
                    let substituted = self.exprs.substitute(axis, var, at);
                    self.simplify(substituted)
                })
                .collect();
            let shifted = Tile { origin, ..tile.clone() };
            let start = self.slot_start(tile, at);
            let run = self.run(&shifted);
            fills.push(self.fill(&shifted, run, start));
        }
        Stmt::Seq(fills)
    }
}
