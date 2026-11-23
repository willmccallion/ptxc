//! Prints compute definitions, schedules and lowered kernels.

use super::index::write_index;
use crate::compute::{
    ComputeDef, IndexVar, IterKind, IterVarId, TensorExpr, TensorParamId, UnaryOp,
};
use crate::expr::{ExprId, VarId};
use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim, Shape};
use std::fmt::Write;

/// The text of `def`:
///
/// ```text
/// compute matmul(%A: f32[64, 32], %B: f32[32, 16]) {
///   %i = spatial 64
///   %k = reduce 32
///   %C: f32[64, 16] = sum[%k](mul(%A[%i, %k], %B[%k, %j]))
/// }
/// ```
pub fn print_compute(def: &ComputeDef) -> String {
    let printer = ComputePrinter { def };
    let mut out = format!("compute {}(", def.name());
    for (index, param) in def.inputs().iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        let _infallible =
            write!(out, "%{}: {}", param.name, type_text(param.dtype.name(), &param.shape));
    }
    out.push_str(") {\n");
    for (var, definition) in def.vars().iter_enumerated() {
        printer.write_var(&mut out, var, definition);
    }
    let output_type = |dtype: DType| type_text(dtype.name(), def.output_shape());
    for output in def.outputs() {
        let _infallible = write!(out, "  %{}: {} = ", output.name, output_type(output.dtype));
        printer.write_expr(&mut out, &output.expr);
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

/// `f32[2, 3]`.
fn type_text(dtype: &str, shape: &Shape) -> String {
    let dims: Vec<String> = shape.dims().iter().map(|dim: &Dim| dim.get().to_string()).collect();
    format!("{dtype}[{}]", dims.join(", "))
}

struct ComputePrinter<'d> {
    def: &'d ComputeDef,
}

impl ComputePrinter<'_> {
    fn var_name(&self, var: VarId) -> String {
        self.def.var_name(var).to_string()
    }

    fn write_var(&self, out: &mut String, var: VarId, definition: &IndexVar) {
        let _infallible = write!(out, "  %{} = ", self.var_name(var));
        match definition {
            IndexVar::Iter(axis) => {
                let iter = &self.def.iter_vars()[*axis];
                let kind = match iter.kind {
                    IterKind::Spatial => "spatial",
                    IterKind::Reduce => "reduce",
                };
                let _infallible = write!(out, "{kind} {}", iter.extent.get());
            }
            IndexVar::Loaded { tensor, index, .. } => {
                out.push_str("load ");
                self.write_load(out, *tensor, index);
            }
        }
        out.push('\n');
    }

    fn write_load(&self, out: &mut String, tensor: TensorParamId, index: &[ExprId]) {
        let _infallible = write!(out, "%{}[", self.def.inputs()[tensor].name);
        for (position, &expr) in index.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            write_index(out, self.def.exprs(), expr, &|var| self.var_name(var));
        }
        out.push(']');
    }

    fn write_expr(&self, out: &mut String, expr: &TensorExpr) {
        match expr {
            TensorExpr::Load { tensor, index } => self.write_load(out, *tensor, index),
            TensorExpr::Const(scalar) => {
                let _infallible = write!(out, "{scalar}");
            }
            TensorExpr::Unary(op, value) => {
                out.push_str(op.name());
                if let UnaryOp::IsInf(sign) = op {
                    let _infallible = write!(out, "<{}>", sign.name());
                }
                self.write_args(out, &[value]);
            }
            TensorExpr::Binary(op, lhs, rhs) => {
                out.push_str(op.name());
                self.write_args(out, &[lhs, rhs]);
            }
            TensorExpr::Cast { to, overflow, value } => {
                let overflow = match overflow {
                    Overflow::Saturate => "saturate",
                    Overflow::NonFinite => "nonfinite",
                };
                let _infallible = write!(out, "cast<{}, {overflow}>", to.name());
                self.write_args(out, &[value]);
            }
            TensorExpr::Select { condition, then, otherwise } => {
                out.push_str("select");
                self.write_args(out, &[condition, then, otherwise]);
            }
            TensorExpr::Reduce { op, axes, body } => {
                out.push_str(op.name());
                self.write_axes(out, axes);
                self.write_args(out, &[body]);
            }
            TensorExpr::ArgReduce { kind, tie, axis, body } => {
                let _infallible = write!(out, "{}<{}>", kind.name(), tie.name());
                self.write_axes(out, &[*axis]);
                self.write_args(out, &[body]);
            }
        }
    }

    fn write_args(&self, out: &mut String, args: &[&TensorExpr]) {
        out.push('(');
        for (position, arg) in args.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            self.write_expr(out, arg);
        }
        out.push(')');
    }

    fn write_axes(&self, out: &mut String, axes: &[IterVarId]) {
        let names: Vec<String> =
            axes.iter().map(|&axis| format!("%{}", self.def.iter_vars()[axis].name)).collect();
        let _infallible = write!(out, "[{}]", names.join(", "));
    }
}
