//! The fragment statements as text.

use super::kernel::KernelPrinter;
use crate::compute::Scalar;
use crate::expr::VarId;
use crate::stmt::{BufferId, FragmentTile, Stmt, TileAccess, TileOrder};
use ptxc_target::MmaInstruction;
use std::fmt::Write;

impl KernelPrinter<'_> {
    /// `%f[t] = fill(value)`.
    pub(super) fn fill(&self, out: &mut String, pad: &str, fragment: FragmentTile, value: Scalar) {
        let _infallible = writeln!(out, "{pad}{} = fill({value})", self.fragment(fragment));
    }

    /// `%f[t] = tile %source[origin] stride s`.
    pub(super) fn load_tile(
        &self,
        out: &mut String,
        pad: &str,
        fragment: FragmentTile,
        (source, at): (BufferId, TileAccess),
    ) {
        let _infallible = write!(out, "{pad}{} = tile ", self.fragment(fragment));
        self.tile(out, source, at);
        out.push('\n');
    }

    /// `%c[t] = mma.m16n8k16.f16.f32(%a[t], %b[t], %c[t])`.
    pub(super) fn mma(
        &self,
        out: &mut String,
        pad: &str,
        instruction: MmaInstruction,
        [a, b, accumulator]: [FragmentTile; 3],
    ) {
        let shape = instruction.shape;
        let acc = self.fragment(accumulator);
        let _infallible = writeln!(
            out,
            "{pad}{acc} = mma.m{}n{}k{}.{}.{}({}, {}, {acc})",
            shape.m.get(),
            shape.n.get(),
            shape.k.get(),
            instruction.multiplicands.name(),
            instruction.accumulator.name(),
            self.fragment(a),
            self.fragment(b),
        );
    }

    /// `each %f[t] as (%row, %column) { body }`.
    pub(super) fn each(
        &self,
        out: &mut String,
        depth: usize,
        (fragment, [row, column]): (FragmentTile, [VarId; 2]),
        body: &Stmt,
    ) {
        let pad = "  ".repeat(depth);
        let (row, column) = (self.name(row), self.name(column));
        let fragment = self.fragment(fragment);
        let _infallible = writeln!(out, "{pad}each {fragment} as (%{row}, %{column}) {{");
        self.close(out, body, depth);
    }

    /// `%f[t]`.
    fn fragment(&self, fragment: FragmentTile) -> String {
        let mut out = String::new();
        self.access(&mut out, fragment.buffer, fragment.tile);
        out
    }

    /// `%buffer[origin] stride s`, then ` transposed` for a transposed tile.
    fn tile(&self, out: &mut String, buffer: BufferId, at: TileAccess) {
        self.access(out, buffer, at.origin);
        let _infallible = write!(out, " stride {}", at.stride.get());
        if at.order == TileOrder::Transposed {
            out.push_str(" transposed");
        }
    }
}
