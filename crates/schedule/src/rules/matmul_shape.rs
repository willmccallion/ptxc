//! Recognises a matrix product, as Ansor's multi-level tiling recognises
//! the data reuse that tiling pays for: a sum over one axis of two loads
//! multiplied, the left reading a row axis the right does not, the right a
//! column axis the left does not. Any other axis only one operand reads is
//! a batch axis the other broadcasts.

use ptxc_loop::{
    BinaryOp, ComputeDef, Expr, IndexVar, IterKind, IterVarId, ReduceOp, TensorExpr, TensorParamId,
};

/// A matrix product's axes and operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Product {
    /// The innermost output axis only the left operand reads.
    pub(crate) row: IterVarId,
    /// The innermost output axis only the right operand reads.
    pub(crate) column: IterVarId,
    /// The summed axis both operands read.
    pub(crate) depth: IterVarId,
    /// The left operand.
    pub(crate) lhs: TensorParamId,
    /// The right operand.
    pub(crate) rhs: TensorParamId,
}

/// `compute` as a matrix product with work to tile: one output holding a
/// sum over one axis of the product of two loads, every index part a plain
/// axis or a constant, each operand read by its one load alone, and no
/// empty axis. An empty product is left to the reduction rule, which keeps
/// empty loops serial.
pub(crate) fn product(compute: &ComputeDef) -> Option<Product> {
    let [output] = compute.outputs() else { return None };
    let TensorExpr::Reduce { op: ReduceOp::Sum, axes, body } = output.expr.reduction()? else {
        return None;
    };
    let &[depth] = axes.as_slice() else { return None };
    let TensorExpr::Binary(BinaryOp::Mul, lhs, rhs) = uncast(body) else { return None };
    let (lhs, lhs_axes) = load_axes(compute, uncast(lhs))?;
    let (rhs, rhs_axes) = load_axes(compute, uncast(rhs))?;
    let innermost_only_in = |reads: &[IterVarId], other: &[IterVarId]| {
        let spatial = |axis: &IterVarId| compute.iter_vars()[*axis].kind == IterKind::Spatial;
        reads.iter().copied().filter(|axis| spatial(axis) && !other.contains(axis)).max()
    };
    let row = innermost_only_in(&lhs_axes, &rhs_axes)?;
    let column = innermost_only_in(&rhs_axes, &lhs_axes)?;
    let has_work = compute.iter_vars().iter().all(|axis| axis.extent.get() > 0);
    let read_once = [lhs, rhs].iter().all(|&tensor| read_once(&output.expr, body, tensor));
    let reads_depth = lhs_axes.contains(&depth) && rhs_axes.contains(&depth);
    (lhs != rhs && read_once && reads_depth && has_work).then_some(Product {
        row,
        column,
        depth,
        lhs,
        rhs,
    })
}

/// `expr` without the casts around it.
pub(crate) fn uncast(expr: &TensorExpr) -> &TensorExpr {
    match expr {
        TensorExpr::Cast { value, .. } => uncast(value),
        other => other,
    }
}

/// The tensor `expr` loads and the axes its index reads, if it is a load
/// whose every index part is a plain axis or a constant.
fn load_axes(compute: &ComputeDef, expr: &TensorExpr) -> Option<(TensorParamId, Vec<IterVarId>)> {
    let TensorExpr::Load { tensor, index, .. } = expr else { return None };
    let mut axes = Vec::new();
    for &part in index {
        match compute.exprs().get(part) {
            Expr::Const(_) => {}
            Expr::Var(var) => match &compute.vars()[var] {
                IndexVar::Iter(axis) => axes.push(*axis),
                IndexVar::Loaded { .. } => return None,
            },
            _ => return None,
        }
    }
    Some((*tensor, axes))
}

/// Whether `tensor` is read once in `body`, the body of `expr`'s one
/// reduction, and nowhere else in `expr`: an epilogue may read the
/// reduction more than once (D80), each read the same body.
pub(crate) fn read_once(expr: &TensorExpr, body: &TensorExpr, tensor: TensorParamId) -> bool {
    loads(body, tensor) == 1 && loads_outside_reductions(expr, tensor) == 0
}

/// How many loads of `tensor` `expr` holds.
pub(crate) fn loads(expr: &TensorExpr, tensor: TensorParamId) -> usize {
    match expr {
        TensorExpr::Reduce { body, .. } | TensorExpr::ArgReduce { body, .. } => loads(body, tensor),
        other => count_operands(other, tensor, loads),
    }
}

/// How many loads of `tensor` `expr` holds outside its reductions.
fn loads_outside_reductions(expr: &TensorExpr, tensor: TensorParamId) -> usize {
    match expr {
        TensorExpr::Reduce { .. } | TensorExpr::ArgReduce { .. } => 0,
        other => count_operands(other, tensor, loads_outside_reductions),
    }
}

/// The loads of `tensor` `expr` holds itself, plus those `count` finds in
/// its operands.
fn count_operands(
    expr: &TensorExpr,
    tensor: TensorParamId,
    count: fn(&TensorExpr, TensorParamId) -> usize,
) -> usize {
    let own = matches!(expr, TensorExpr::Load { tensor: read, .. } if *read == tensor);
    usize::from(own)
        + expr.operands().into_iter().map(|operand| count(operand, tensor)).sum::<usize>()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{Product, loads, product};
    use ptxc_core::{DType, Dim, Idx, Shape};
    use ptxc_loop::{
        BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, IterVarId, ReduceOp, TensorExpr,
        TensorName, TensorParamId, VarName,
    };

    /// Where each operand of a two-operand product reads, by axis name:
    /// `b`, `i`, `j` spatial (`b` outermost), `k` summed, `0` a constant.
    struct Reads {
        lhs: &'static [&'static str],
        rhs: &'static [&'static str],
    }

    /// What a product's definition does besides its sum.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Variant {
        /// Two distinct operands, and nothing else.
        Plain,
        /// `x` as both operands.
        Square,
        /// `x[0, ..]` added to the sum, so `x` is read twice.
        ReadAgain,
        /// The sum times itself, the one reduction read twice.
        Squared,
    }

    /// `out[b, i, j] = sum over k of x[reads.lhs] * y[reads.rhs]`, every
    /// extent `extent`, varied as `variant` says.
    fn compute(reads: &Reads, extent: u64, variant: Variant) -> ComputeDef {
        let mut compute = ComputeBuilder::new(ComputeName::new("product"));
        let shape = |rank: usize| Shape::new(vec![Dim::new(extent.max(1)); rank]).unwrap();
        let x = compute.input(TensorName::new("x"), DType::F32, shape(reads.lhs.len()));
        let y = if variant == Variant::Square {
            x
        } else {
            compute.input(TensorName::new("y"), DType::F32, shape(reads.rhs.len()))
        };
        let names = ["b", "i", "j"];
        let spatial: Vec<ExprId> = names
            .iter()
            .map(|&name| compute.spatial(VarName::new(name), Dim::new(extent)).1)
            .collect();
        let (axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(extent));
        let zero = compute.exprs().constant(0);
        let index = |parts: &[&str]| -> Vec<ExprId> {
            parts
                .iter()
                .map(|&part| match part {
                    "k" => k,
                    "0" => zero,
                    name => spatial[names.iter().position(|&n| n == name).unwrap()],
                })
                .collect()
        };
        let load = |tensor, parts: &[&str]| TensorExpr::load(tensor, index(parts));
        let body = TensorExpr::Binary(
            BinaryOp::Mul,
            Box::new(load(x, reads.lhs)),
            Box::new(load(y, reads.rhs)),
        );
        let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![axis], body: Box::new(body) };
        let first = TensorExpr::load(x, vec![zero; reads.lhs.len()]);
        let output = match variant {
            Variant::ReadAgain => TensorExpr::Binary(BinaryOp::Add, Box::new(sum), Box::new(first)),
            Variant::Squared => {
                TensorExpr::Binary(BinaryOp::Mul, Box::new(sum.clone()), Box::new(sum))
            }
            Variant::Plain | Variant::Square => sum,
        };
        compute.output(TensorName::new("out"), output);
        compute.finish().unwrap()
    }

    fn found(reads: &Reads) -> Option<Product> {
        product(&compute(reads, 3, Variant::Plain))
    }

    #[test]
    fn a_batched_product_finds_its_row_column_and_depth() {
        let found = found(&Reads { lhs: &["b", "i", "k"], rhs: &["b", "k", "j"] });

        let expected = Product {
            row: IterVarId::new(1),
            column: IterVarId::new(2),
            depth: IterVarId::new(3),
            lhs: TensorParamId::new(0),
            rhs: TensorParamId::new(1),
        };
        assert_eq!(found, Some(expected));
    }

    #[test]
    fn transposed_and_broadcast_operands_are_products() {
        let transposed = found(&Reads { lhs: &["0", "k", "i"], rhs: &["j", "b", "k"] });

        assert_eq!(
            transposed.map(|product| (product.row, product.column)),
            Some((IterVarId::new(1), IterVarId::new(2)))
        );
    }

    #[test]
    fn a_product_needs_a_row_a_column_and_the_depth_in_both_operands() {
        assert_eq!(found(&Reads { lhs: &["b", "k"], rhs: &["b", "k", "j"] }), None);
        assert_eq!(found(&Reads { lhs: &["b", "i", "k"], rhs: &["b", "k", "i"] }), None);
        assert_eq!(found(&Reads { lhs: &["b", "i", "j"], rhs: &["k", "j"] }), None);
        assert_eq!(found(&Reads { lhs: &["b", "i"], rhs: &["k", "j"] }), None);
    }

    #[test]
    fn an_operand_read_twice_or_an_empty_product_is_not_tiled() {
        let reads = Reads { lhs: &["b", "i", "k"], rhs: &["b", "k", "j"] };

        assert_eq!(product(&compute(&reads, 3, Variant::Square)), None);
        assert_eq!(product(&compute(&reads, 3, Variant::ReadAgain)), None);
        assert_eq!(product(&compute(&reads, 0, Variant::Plain)), None);
    }

    #[test]
    fn a_product_its_epilogue_reads_twice_is_still_tiled() {
        let reads = Reads { lhs: &["b", "i", "k"], rhs: &["b", "k", "j"] };

        let squared = product(&compute(&reads, 3, Variant::Squared));

        assert_eq!(squared, product(&compute(&reads, 3, Variant::Plain)));
        assert!(squared.is_some());
    }

    #[test]
    fn every_load_of_a_tensor_counts_wherever_it_sits() {
        let (x, y) = (TensorParamId::new(0), TensorParamId::new(1));
        let load = |tensor| Box::new(TensorExpr::load(tensor, Vec::new()));
        let chosen = TensorExpr::Select { condition: load(x), then: load(x), otherwise: load(y) };
        let added = TensorExpr::Binary(BinaryOp::Add, Box::new(chosen.clone()), load(y));

        assert_eq!([loads(&chosen, x), loads(&chosen, y)], [2, 1]);
        assert_eq!([loads(&added, x), loads(&added, y)], [2, 2]);
    }
}
