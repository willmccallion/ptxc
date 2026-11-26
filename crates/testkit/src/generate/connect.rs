//! Connectors: ops reading a value already in the recipe, chosen from those
//! continuous in their operands. A last-bit difference between two
//! executors then stays a last-bit difference, rather than flipping a
//! floor, a sign or a comparison, so graphs of them on moderate inputs are
//! compared with the reference.

use super::inputs::{Input, moderate};
use super::recipe::{Node, Operand, Recipe, tensor_type};
use super::shapes::{Rng, dims, extent};
use ptxc_core::{Axis, DType, Dim, Shape};
use ptxc_graph::{
    Activation, BinaryOp, KeepDims, Op, ReduceAttrs, ReduceOp, SoftmaxAttrs, SoftmaxKind,
    TensorType, TransposeAttrs, UnaryOp, VariadicOp,
};

/// The element types of the inputs connectors start from, f32 most often.
const DTYPES: [DType; 6] =
    [DType::F32, DType::F32, DType::F32, DType::F16, DType::Bf16, DType::I32];
const UNARY: [UnaryOp; 2] = [UnaryOp::Abs, UnaryOp::Neg];
const ACTIVATIONS: [Activation; 4] =
    [Activation::Relu, Activation::Sigmoid, Activation::Tanh, Activation::Softsign];
const BINARY: [Op; 5] = [
    Op::Binary(BinaryOp::Add),
    Op::Binary(BinaryOp::Sub),
    Op::Binary(BinaryOp::Mul),
    Op::Variadic(VariadicOp::Max),
    Op::Variadic(VariadicOp::Min),
];
const REDUCTIONS: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];

/// Whether `op` is one the connectors draw, continuous in its operands.
pub(crate) fn is_continuous(op: &Op) -> bool {
    match op {
        Op::Unary(unary) => UNARY.contains(unary),
        Op::Activation(activation) => ACTIVATIONS.contains(activation),
        Op::Binary(_) | Op::Variadic(_) => BINARY.contains(op),
        Op::Reduce(attrs) => REDUCTIONS.contains(&attrs.op),
        Op::Softmax(attrs) => attrs.kind == SoftmaxKind::Softmax,
        Op::MatMul | Op::Transpose(_) | Op::Reshape(_) | Op::Concat(_) => true,
        _ => false,
    }
}

/// Adds a connector reading one of `values`, the recipe's values: most
/// often the last, or a new moderate input if there are none. `None` if the
/// drawn op needs an axis the value lacks.
pub(crate) fn connect(
    rng: &mut Rng,
    recipe: &mut Recipe,
    values: &[(Operand, TensorType)],
) -> Option<()> {
    let (operand, ty) = match values {
        [] => fresh(rng, recipe),
        [.., last] if rng.chance(3, 4) => last.clone(),
        _ => rng.pick(values).clone(),
    };
    let node = propose(rng, recipe, operand, &ty)?;
    recipe.nodes.push(node);
    Some(())
}

/// A new moderate input to start from, of rank 0 to 3, rank 2 most often.
fn fresh(rng: &mut Rng, recipe: &mut Recipe) -> (Operand, TensorType) {
    let dtype = *rng.pick(&DTYPES);
    let rank = *rng.pick(&[0, 1, 2, 2, 3]);
    let extents = dims(rng, rank);
    let input = moderate(rng, dtype, &extents);
    let ty = tensor_type(&input.tensor);
    (push(recipe, input), ty)
}

/// A connector reading `operand`, of type `ty`, possibly with a new input
/// added to `recipe`; `None` if the drawn op needs an axis `ty` lacks. The
/// caller drops proposals the op's type rule rejects.
fn propose(rng: &mut Rng, recipe: &mut Recipe, operand: Operand, ty: &TensorType) -> Option<Node> {
    let rank = ty.shape().dims().len();
    let node = |op, operands| Node { op, operands };
    Some(match rng.below(9) {
        0 => node(Op::Unary(*rng.pick(&UNARY)), vec![operand]),
        1 => node(Op::Activation(*rng.pick(&ACTIVATIONS)), vec![operand]),
        2 => {
            let extents = broadcastable(rng, ty.shape());
            let other = new_input(rng, recipe, ty.dtype(), &extents);
            node(rng.pick(&BINARY).clone(), vec![operand, other])
        }
        3 if rank >= 1 => {
            let k = ty.shape().dims()[rank - 1].get();
            let extents = [k, extent(rng)];
            let weights = new_input(rng, recipe, ty.dtype(), &extents);
            node(Op::MatMul, vec![operand, weights])
        }
        4 if rank >= 1 => node(Op::Reduce(reduction(rng, rank)), vec![operand]),
        5 if rank >= 1 => {
            let axis = Axis::new(rng.index(rank));
            node(Op::Softmax(SoftmaxAttrs { kind: SoftmaxKind::Softmax, axis }), vec![operand])
        }
        6 => node(Op::Transpose(permutation(rng, rank)), vec![operand]),
        7 => node(Op::Reshape(flattened(ty.shape())), vec![operand]),
        8 if rank >= 1 => {
            let axis = rng.index(rank);
            let mut extents: Vec<u64> = ty.shape().dims().iter().map(|dim| dim.get()).collect();
            extents[axis] = extent(rng);
            let other = new_input(rng, recipe, ty.dtype(), &extents);
            node(Op::Concat(Axis::new(axis)), vec![operand, other])
        }
        _ => return None,
    })
}

/// A new moderate input of `dtype` and `extents`.
fn new_input(rng: &mut Rng, recipe: &mut Recipe, dtype: DType, extents: &[u64]) -> Operand {
    let input = moderate(rng, dtype, extents);
    push(recipe, input)
}

/// `input` added to `recipe`, as an operand.
pub(crate) fn push(recipe: &mut Recipe, input: Input) -> Operand {
    recipe.inputs.push(input);
    Operand::Input(recipe.inputs.len() - 1)
}

/// Extents broadcasting to `shape`: leading axes dropped, some set to 1.
fn broadcastable(rng: &mut Rng, shape: &Shape) -> Vec<u64> {
    let extents: Vec<u64> = shape.dims().iter().map(|dim| dim.get()).collect();
    let dropped = rng.index(extents.len() + 1);
    extents[dropped..].iter().map(|&extent| if rng.chance(1, 3) { 1 } else { extent }).collect()
}

/// One of the reductions over a non-empty set of axes.
fn reduction(rng: &mut Rng, rank: usize) -> ReduceAttrs {
    let mut axes: Vec<Axis> = (0..rank).filter(|_| rng.chance(1, 2)).map(Axis::new).collect();
    if axes.is_empty() {
        axes.push(Axis::new(rng.index(rank)));
    }
    let keep_dims = if rng.chance(1, 2) { KeepDims::Keep } else { KeepDims::Drop };
    ReduceAttrs { op: *rng.pick(&REDUCTIONS), axes, keep_dims }
}

/// A random permutation of `rank` axes.
fn permutation(rng: &mut Rng, rank: usize) -> TransposeAttrs {
    let mut axes: Vec<Axis> = (0..rank).map(Axis::new).collect();
    for i in (1..rank).rev() {
        axes.swap(i, rng.index(i + 1));
    }
    TransposeAttrs { permutation: axes }
}

/// `shape` as one axis of all its elements.
fn flattened(shape: &Shape) -> Shape {
    Shape::new(vec![Dim::new(shape.element_count().get())]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::{BINARY, connect, is_continuous, moderate, push};
    use crate::generate::recipe::{Node, Operand, Recipe};
    use crate::generate::shapes::Rng;
    use ptxc_core::{Axis, DType};
    use ptxc_graph::{
        Activation, BinaryOp, F32Attr, KeepDims, Op, ReduceAttrs, ReduceOp, SoftmaxAttrs,
        SoftmaxKind, UnaryOp,
    };
    use std::mem::discriminant;

    /// Three moderate `f32` inputs of shape `[2, 3]`.
    fn three_inputs() -> Recipe {
        let mut rng = Rng::new(0);
        let mut recipe = Recipe::default();
        for _ in 0..3 {
            let _operand = push(&mut recipe, moderate(&mut rng, DType::F32, &[2, 3]));
        }
        recipe
    }

    /// The connector each of the first `seeds` seeds adds to `recipe`.
    fn connectors(recipe: &Recipe, seeds: u64) -> Vec<Node> {
        let values = recipe.values().unwrap();
        (0..seeds)
            .filter_map(|seed| {
                let mut next = recipe.clone();
                connect(&mut Rng::new(seed), &mut next, &values)?;
                next.nodes.pop()
            })
            .collect()
    }

    #[test]
    fn a_connector_reads_the_last_value_most_often_but_not_always() {
        let recipe = three_inputs();

        let nodes = connectors(&recipe, 200);

        let last = Operand::Input(2);
        let at_last = nodes.iter().filter(|node| node.operands[0] == last).count();
        assert!(at_last > nodes.len() / 2 && at_last < nodes.len(), "{at_last} of {}", nodes.len());
    }

    #[test]
    fn connectors_draw_every_kind_of_op_they_name() {
        let recipe = three_inputs();

        let kinds: Vec<_> =
            connectors(&recipe, 400).iter().map(|node| discriminant(&node.op)).collect();

        let unary = Op::Unary(UnaryOp::Abs);
        let named = [
            unary,
            Op::Activation(Activation::Relu),
            BINARY[0].clone(),
            Op::MatMul,
            Op::Reduce(ReduceAttrs { op: ReduceOp::Sum, axes: vec![], keep_dims: KeepDims::Keep }),
            Op::Softmax(SoftmaxAttrs { kind: SoftmaxKind::Softmax, axis: Axis::new(0) }),
            Op::Transpose(ptxc_graph::TransposeAttrs { permutation: vec![] }),
            Op::Reshape(ptxc_core::Shape::new(vec![]).unwrap()),
            Op::Concat(Axis::new(0)),
        ];
        for op in named {
            assert!(kinds.contains(&discriminant(&op)), "no {op:?}");
        }
    }

    #[test]
    fn transposes_draw_every_order_of_two_axes() {
        let recipe = three_inputs();

        let orders: Vec<Vec<usize>> = connectors(&recipe, 400)
            .iter()
            .filter_map(|node| match &node.op {
                Op::Transpose(attrs) => Some(attrs.permutation.iter().map(|a| a.index()).collect()),
                _ => None,
            })
            .collect();

        assert!(orders.contains(&vec![0, 1]) && orders.contains(&vec![1, 0]), "{orders:?}");
    }

    #[test]
    fn only_the_connectors_ops_are_continuous() {
        let reduce = |op| Op::Reduce(ReduceAttrs { op, axes: vec![], keep_dims: KeepDims::Keep });
        let softmax = |kind| Op::Softmax(SoftmaxAttrs { kind, axis: Axis::new(0) });
        let hard_sigmoid =
            Activation::HardSigmoid { alpha: F32Attr::new(0.2), beta: F32Attr::new(0.5) };

        let continuous = [
            Op::Unary(UnaryOp::Neg),
            Op::Activation(Activation::Tanh),
            Op::Binary(BinaryOp::Mul),
            reduce(ReduceOp::Max),
            softmax(SoftmaxKind::Softmax),
        ];
        let discontinuous = [
            Op::Unary(UnaryOp::Floor),
            Op::Activation(hard_sigmoid),
            Op::Binary(BinaryOp::Div),
            reduce(ReduceOp::Prod),
            softmax(SoftmaxKind::LogSoftmax),
        ];

        assert!(continuous.iter().all(is_continuous));
        assert!(!discontinuous.iter().any(is_continuous));
    }
}
