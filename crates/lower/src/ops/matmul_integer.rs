//! `MatMulInteger`: the 8-bit operands' product in wrapping `i32`, its
//! zero points subtracted through the operands' sums along the contracted
//! axis, so the product itself runs on tensor cores (D61).

use super::NodeContext;
use super::matmul::{Product, contracted, dims, product_reads, with_inner};
use crate::builder::{GroupBuilder, Operand, StageBuilder};
use crate::compose::{add, mul, sub};
use crate::error::LowerError;
use crate::precision::convert;
use crate::stage::{Binding, IntermediateId, LoweredGroup, TableId};
use ptxc_core::{DType, HostTensor, Shape};
use ptxc_graph::{MatMulIntegerAttrs, ValueId, ZeroPoints};
use ptxc_loop::{ExprId, ReduceOp, Scalar, TensorExpr};

/// `MatMulInteger`: the 8-bit operands' product in wrapping `i32`, which
/// tensor cores multiply, with each zero point subtracted through the
/// other operand's sums along the contracted axis, as
/// `sum((a - za)(b - zb)) = sum(ab) - zb sum(a) - za sum(b) + k za zb`
/// holds in wrapping arithmetic (D61).
pub(super) fn matmul_integer(
    cx: &NodeContext<'_>,
    attrs: &MatMulIntegerAttrs,
) -> Result<LoweredGroup, LowerError> {
    let [lhs, rhs] = cx.values()?;
    let Some(output) = cx.needed_output() else { return Ok(LoweredGroup::default()) };
    let mut group = cx.group();
    let points = [Points::of(&mut group, &attrs.a), Points::of(&mut group, &attrs.b)];
    let lhs_axis = dims(cx, lhs).len() - 1;
    let rhs_axis = contracted(&dims(cx, rhs));
    let lhs_sums =
        points[1].is_nonzero().then(|| sums(cx, &mut group, (lhs, lhs_axis), "lhs_sums"));
    let rhs_sums =
        points[0].is_nonzero().then(|| sums(cx, &mut group, (rhs, rhs_axis), "rhs_sums"));
    let corrections = Corrections { points, sums: [lhs_sums.transpose()?, rhs_sums.transpose()?] };
    let mut stage = group.stage(cx.name(), cx.graph.value(output).ty().shape());
    let product = product_reads(cx, &mut stage, lhs, rhs)?;
    let wide = |operand: &Operand| convert(operand.expr.clone(), operand.dtype, DType::I32);
    let body = Box::new(mul(wide(&product.lhs), wide(&product.rhs)));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![product.inner], body };
    let depth = cx.signed(dims(cx, lhs)[lhs_axis])?;
    let corrected =
        corrections.applied(&mut stage, sum, &product, depth).ok_or(LowerError::Operands(cx.id))?;
    stage.output(Binding::Value(output), corrected);
    group.push(cx.finish(stage)?);
    Ok(group.finish())
}

/// What zero points add to a `MatMulInteger` sum: each operand's zero
/// points, and its sums along the contracted axis where the other
/// operand's zero points are not all 0.
struct Corrections {
    points: [Points; 2],
    sums: [Option<IntermediateId>; 2],
}

impl Corrections {
    /// `sum` with the terms for `product`'s element, whose contracted axis
    /// is `depth` long.
    fn applied(
        &self,
        stage: &mut StageBuilder<'_>,
        sum: TensorExpr,
        product: &Product,
        depth: i64,
    ) -> Option<TensorExpr> {
        let za = self.points[0].at(stage, product.row)?;
        let zb = self.points[1].at(stage, product.column)?;
        let mut corrected = sum;
        let mut read =
            |sums, index: &[ExprId]| stage.load(Binding::Intermediate(sums), index.to_vec()).expr;
        if let Some(sums) = self.sums[0] {
            corrected = sub(corrected, mul(zb.clone(), read(sums, &product.lhs_outer)));
        }
        if let Some(sums) = self.sums[1] {
            corrected = sub(corrected, mul(za.clone(), read(sums, &product.rhs_outer)));
        }
        if self.sums.iter().all(Option::is_some) {
            let depth = convert(TensorExpr::Const(Scalar::i64(depth)), DType::I64, DType::I32);
            corrected = add(corrected, mul(depth, mul(za, zb)));
        }
        Some(corrected)
    }
}

/// How a stage reads one operand's zero points: one constant, or a table
/// read at the row or column.
enum Points {
    Constant(i32),
    Table(TableId),
}

impl Points {
    /// `points`, as a table in `group` if they vary and are not all 0.
    fn of(group: &mut GroupBuilder<'_>, points: &ZeroPoints) -> Self {
        match points {
            ZeroPoints::PerTensor(point) => Self::Constant(point.get()),
            ZeroPoints::PerAxis { points, .. } if points.iter().all(|point| point.get() == 0) => {
                Self::Constant(0)
            }
            ZeroPoints::PerAxis { points, .. } => {
                let values: Vec<i32> = points.iter().map(|point| point.get()).collect();
                Self::Table(group.table(HostTensor::vector(values.into())))
            }
        }
    }

    const fn is_nonzero(&self) -> bool {
        !matches!(self, Self::Constant(0))
    }

    /// The zero point at `index` along the axis a table varies along; a
    /// constant needs none.
    fn at(&self, stage: &mut StageBuilder<'_>, index: Option<ExprId>) -> Option<TensorExpr> {
        match *self {
            Self::Constant(point) => {
                Scalar::int(DType::I32, i64::from(point)).map(TensorExpr::Const)
            }
            Self::Table(table) => Some(stage.load(Binding::Table(table), vec![index?]).expr),
        }
    }
}

/// `operand`'s sums along its contracted `axis` in wrapping `i32`: an
/// intermediate over its other axes, computed by a stage named `part`.
fn sums(
    cx: &NodeContext<'_>,
    group: &mut GroupBuilder<'_>,
    (operand, axis): (ValueId, usize),
    part: &str,
) -> Result<IntermediateId, LowerError> {
    let mut kept = dims(cx, operand);
    let depth = kept.remove(axis);
    let shape = Shape::new(kept).map_err(|_| LowerError::Operands(cx.id))?;
    let sums = group.intermediate_of(DType::I32, &shape);
    let mut stage = group.stage(cx.named(part), &shape);
    let (depth_axis, k) = stage.reduce_axis(depth);
    let index = with_inner(stage.spatial(), axis, k);
    let element = stage.load(Binding::Value(operand), index);
    let body = Box::new(convert(element.expr, element.dtype, DType::I32));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![depth_axis], body };
    stage.output(Binding::Intermediate(sums), sum);
    group.push(cx.finish(stage)?);
    Ok(sums)
}
