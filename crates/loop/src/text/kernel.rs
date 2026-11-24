//! Lowered kernels as text: parameters, launch shape, thread indices and the
//! statement tree.

use super::cast::cast_name;
use super::index::write_index;
use crate::compute::UnaryOp;
use crate::expr::{ExprId, VarId};
use crate::stmt::{
    Access, BufferId, ForKind, KernelVar, LoweredKernel, MemScope, Stmt, TileAccess, TileOrder,
    ValueExpr,
};
use ptxc_target::MmaInstruction;
use std::fmt::Write;

/// The text of `kernel`:
///
/// ```text
/// kernel add(%x: f32[1000] read, %sum: f32[1000] write)
///   grid(4, 1, 1) block(256, 1, 1) smem 0
///   %i.0 = blockIdx.x
///   %i.1 = threadIdx.x
///   if %i.1 + %i.0 * 256 < 1000 {
///     %sum[%i.1 + %i.0 * 256] = ...
///   }
/// ```
pub fn print_kernel(kernel: &LoweredKernel) -> String {
    let printer = KernelPrinter { kernel };
    let params: Vec<String> = kernel
        .params()
        .iter()
        .map(|param| {
            let access = match param.access {
                Access::Read => "read",
                Access::Write => "write",
            };
            format!("{} {access}", printer.buffer_type(param.buffer))
        })
        .collect();
    let launch = kernel.launch();
    let mut out = format!("kernel {}({})\n", kernel.name(), params.join(", "));
    let _infallible = writeln!(
        out,
        "  grid({}, {}, {}) block({}, {}, {}) smem {}",
        launch.grid[0].get(),
        launch.grid[1].get(),
        launch.grid[2].get(),
        launch.block[0].get(),
        launch.block[1].get(),
        launch.block[2].get(),
        kernel.smem().get()
    );
    for definition in kernel.vars() {
        if let KernelVar::Thread { name, axis } = definition {
            let _infallible = writeln!(out, "  %{name} = {}", axis.name());
        }
    }
    printer.write_stmt(&mut out, kernel.body(), 1);
    out
}

struct KernelPrinter<'k> {
    kernel: &'k LoweredKernel,
}

impl KernelPrinter<'_> {
    fn name(&self, var: VarId) -> String {
        self.kernel.vars()[var].name().to_string()
    }

    fn index(&self, out: &mut String, index: ExprId) {
        write_index(out, self.kernel.exprs(), index, &|var| self.name(var));
    }

    /// `%x: f32[1000]`.
    fn buffer_type(&self, buffer: BufferId) -> String {
        let found = &self.kernel.buffers()[buffer];
        format!("%{}: {}[{}]", found.name, found.dtype.name(), found.size.get())
    }

    fn access(&self, out: &mut String, buffer: BufferId, index: ExprId) {
        let _infallible = write!(out, "%{}[", self.kernel.buffers()[buffer].name);
        self.index(out, index);
        out.push(']');
    }

    fn write_stmt(&self, out: &mut String, stmt: &Stmt, depth: usize) {
        let pad = "  ".repeat(depth);
        match stmt {
            Stmt::Seq(stmts) => stmts.iter().for_each(|stmt| self.write_stmt(out, stmt, depth)),
            Stmt::For { var, extent, kind, body } => {
                let kind = kind_suffix(*kind);
                let _infallible =
                    writeln!(out, "{pad}for %{} in {}{kind} {{", self.name(*var), extent.get());
                self.close(out, body, depth);
            }
            Stmt::If { condition, then } => {
                out.push_str(&pad);
                out.push_str("if ");
                self.index(out, condition.index);
                out.push_str(" < ");
                self.index(out, condition.bound);
                out.push_str(" {\n");
                self.close(out, then, depth);
            }
            Stmt::Alloc { buffer, body } => {
                let (typed, scope) = (self.buffer_type(*buffer), self.scope(*buffer));
                let _infallible = writeln!(out, "{pad}alloc {typed} {scope} {{");
                self.close(out, body, depth);
            }
            Stmt::LoadIndex { var, buffer, index, body } => {
                let _infallible = write!(out, "{pad}let %{} = ", self.name(*var));
                self.access(out, *buffer, *index);
                out.push_str(" {\n");
                self.close(out, body, depth);
            }
            Stmt::Store { buffer, index, value } => {
                out.push_str(&pad);
                self.access(out, *buffer, *index);
                out.push_str(" = ");
                self.value(out, value);
                out.push('\n');
            }
            Stmt::Barrier => {
                let _infallible = writeln!(out, "{pad}barrier");
            }
            Stmt::FragmentFill { fragment, value } => {
                let _infallible = writeln!(out, "{pad}{} = fill({value})", self.named(*fragment));
            }
            Stmt::FragmentLoad { fragment, source, at } => {
                let _infallible = write!(out, "{pad}{} = tile ", self.named(*fragment));
                self.tile(out, *source, *at);
                out.push('\n');
            }
            Stmt::Mma { instruction, a, b, accumulator } => {
                out.push_str(&pad);
                self.mma(out, *instruction, [*a, *b, *accumulator]);
            }
            Stmt::FragmentStore { fragment, target, at } => {
                out.push_str(&pad);
                out.push_str("tile ");
                self.tile(out, *target, *at);
                let _infallible = writeln!(out, " = {}", self.named(*fragment));
            }
        }
    }

    /// Where `buffer` lives, with a fragment's tile shape.
    fn scope(&self, buffer: BufferId) -> String {
        match self.kernel.buffers()[buffer].scope {
            MemScope::Global => "global".to_owned(),
            MemScope::Shared => "shared".to_owned(),
            MemScope::Local => "local".to_owned(),
            MemScope::Fragment(shape) => {
                format!("fragment<{}x{}>", shape.rows.get(), shape.columns.get())
            }
        }
    }

    /// `%c = mma.m16n8k16.f16.f32(%a, %b, %c)`.
    fn mma(
        &self,
        out: &mut String,
        instruction: MmaInstruction,
        [a, b, accumulator]: [BufferId; 3],
    ) {
        let shape = instruction.shape;
        let acc = self.named(accumulator);
        let _infallible = writeln!(
            out,
            "{acc} = mma.m{}n{}k{}.{}.{}({}, {}, {acc})",
            shape.m.get(),
            shape.n.get(),
            shape.k.get(),
            instruction.multiplicands.name(),
            instruction.accumulator.name(),
            self.named(a),
            self.named(b),
        );
    }

    /// `%name`.
    fn named(&self, buffer: BufferId) -> String {
        format!("%{}", self.kernel.buffers()[buffer].name)
    }

    /// `%buffer[origin] stride s`, then ` transposed` for a transposed tile.
    fn tile(&self, out: &mut String, buffer: BufferId, at: TileAccess) {
        self.access(out, buffer, at.origin);
        let _infallible = write!(out, " stride {}", at.stride.get());
        if at.order == TileOrder::Transposed {
            out.push_str(" transposed");
        }
    }

    /// Writes `body` one level in, then the closing brace.
    fn close(&self, out: &mut String, body: &Stmt, depth: usize) {
        self.write_stmt(out, body, depth + 1);
        let _infallible = writeln!(out, "{}}}", "  ".repeat(depth));
    }

    fn value(&self, out: &mut String, value: &ValueExpr) {
        match value {
            ValueExpr::Load { buffer, index } => self.access(out, *buffer, *index),
            ValueExpr::Const(scalar) => {
                let _infallible = write!(out, "{scalar}");
            }
            ValueExpr::Index { index, dtype } => {
                let _infallible = write!(out, "index<{}>(", dtype.name());
                self.index(out, *index);
                out.push(')');
            }
            ValueExpr::Unary(op, operand) => {
                out.push_str(op.name());
                if let UnaryOp::IsInf(sign) = op {
                    let _infallible = write!(out, "<{}>", sign.name());
                }
                self.args(out, &[operand]);
            }
            ValueExpr::Binary(op, lhs, rhs) => {
                out.push_str(op.name());
                self.args(out, &[lhs, rhs]);
            }
            ValueExpr::Cast { to, overflow, value } => {
                out.push_str(&cast_name(*to, *overflow));
                self.args(out, &[value]);
            }
            ValueExpr::Select { condition, then, otherwise } => {
                out.push_str("select");
                self.args(out, &[condition, then, otherwise]);
            }
            ValueExpr::MulAdd { lhs, rhs, addend } => {
                out.push_str("fma");
                self.args(out, &[lhs, rhs, addend]);
            }
        }
    }

    fn args(&self, out: &mut String, args: &[&ValueExpr]) {
        out.push('(');
        for (position, arg) in args.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            self.value(out, arg);
        }
        out.push(')');
    }
}

/// How a `for` line marks its loop's kind.
const fn kind_suffix(kind: ForKind) -> &'static str {
    match kind {
        ForKind::Serial => "",
        ForKind::Unrolled => " unroll",
        ForKind::Vectorized => " vectorize",
    }
}
