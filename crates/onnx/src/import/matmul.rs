//! Imports `MatMul` and `Gemm`.

use crate::context::NodeContext;
use crate::error::ImportError;
use crate::lowered::Lowered;
use ptxc_graph::{F32Attr, GemmAttrs, Op, Orientation};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "MatMul" => Ok(Lowered::on_inputs(Op::MatMul, 2)),
        "Gemm" => gemm(context),
        _ => return None,
    })
}

/// `Gemm` with its optional bias `C`, kept only when given.
fn gemm(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let orientation =
        |transposed: i64| if transposed == 0 { Orientation::AsIs } else { Orientation::Transposed };
    let attrs = GemmAttrs {
        a: orientation(context.attrs.int_or("transA", 0)?),
        b: orientation(context.attrs.int_or("transB", 0)?),
        alpha: F32Attr::new(context.attrs.float_or("alpha", 1.0)?),
        beta: F32Attr::new(context.attrs.float_or("beta", 1.0)?),
    };
    let operands = if context.has_input(2) { 3 } else { 2 };
    Ok(Lowered::on_inputs(Op::Gemm(attrs), operands))
}
