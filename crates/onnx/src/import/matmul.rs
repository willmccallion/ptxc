//! Imports `MatMul`, `MatMulInteger` and `Gemm`.

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind, Unsupported};
use crate::lowered::{Lowered, Operand};
use ptxc_core::{Axis, TensorData, ZeroPoint};
use ptxc_graph::{F32Attr, GemmAttrs, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    Some(match op {
        "MatMul" => Ok(Lowered::on_inputs(Op::MatMul, 2)),
        "MatMulInteger" => matmul_integer(context),
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

/// `MatMulInteger(a, b, a_zero_point?, b_zero_point?)`, the zero points
/// constants: one for the whole tensor, one per row of `a`, or one per
/// column of `b`.
fn matmul_integer(context: &NodeContext<'_>) -> Result<Lowered, ImportError> {
    let rank = |index| Ok::<_, ImportError>(context.ty(index)?.shape().rank().get());
    let rows = rank(0)?.checked_sub(2).map(Axis::new);
    let columns = rank(1)?.checked_sub(1).filter(|&axis| axis > 0).map(Axis::new);
    let attrs = MatMulIntegerAttrs {
        a: zero_points(context, 2, rows)?,
        b: zero_points(context, 3, columns)?,
    };
    Ok(Lowered::Node {
        op: Op::MatMulInteger(attrs),
        operands: vec![Operand::Input(0), Operand::Input(1)],
    })
}

/// The zero points input `index` gives, 0 when it is absent. More than one
/// must vary along `axis`, the operand's rows or columns, with no other dim
/// of their tensor above 1: ptxc does not take zero points that also vary
/// over the batch.
fn zero_points(
    context: &NodeContext<'_>,
    index: usize,
    axis: Option<Axis>,
) -> Result<ZeroPoints, ImportError> {
    if !context.has_input(index) {
        return Ok(ZeroPoints::PerTensor(ZeroPoint::new(0)));
    }
    let tensor = context.constant(index)?;
    let points: Vec<ZeroPoint> = match tensor.data() {
        TensorData::I8(values) => {
            values.iter().map(|&value| ZeroPoint::new(value.into())).collect()
        }
        TensorData::U8(values) => {
            values.iter().map(|&value| ZeroPoint::new(value.into())).collect()
        }
        other => {
            let problem = format!("{:?} zero points", other.dtype());
            return Err(ImportErrorKind::Invalid { field: "zero point", problem }.into());
        }
    };
    if let [only] = points[..] {
        return Ok(ZeroPoints::PerTensor(only));
    }
    let varying = tensor.shape().dims().iter().filter(|dim| dim.get() != 1).count();
    match axis {
        Some(axis) if varying == 1 => Ok(ZeroPoints::PerAxis { axis, points }),
        _ => Err(Unsupported::Attribute {
            op: "MatMulInteger".to_owned(),
            attribute: "zero points varying other than by row of a or column of b".to_owned(),
        }
        .into()),
    }
}
