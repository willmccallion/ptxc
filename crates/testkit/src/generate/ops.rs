//! Proposes a node reading a value already in the recipe, from ops whose
//! results are continuous in their operands: a last-bit difference between
//! two executors then stays a last-bit difference, rather than flipping a
//! floor, a sign or a comparison.

use super::graph::{Node, Operand, Recipe};
use super::shapes::{Rng, dims, extent};
use ptxc_core::{Axis, Dim, Shape};
use ptxc_graph::{
    Activation, BinaryOp, KeepDims, Op, ReduceAttrs, ReduceOp, SoftmaxAttrs, SoftmaxKind,
    TensorType, TransposeAttrs, UnaryOp, VariadicOp,
};

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

/// A node reading `operand`, of type `ty`, possibly with a new input added
/// to `recipe`; `None` if the drawn op needs an axis `ty` lacks. The caller
/// drops proposals the op's type rule rejects.
pub(crate) fn propose(
    rng: &mut Rng,
    recipe: &mut Recipe,
    operand: Operand,
    ty: &TensorType,
) -> Option<Node> {
    let rank = ty.shape().dims().len();
    let node = |op, operands| Node { op, operands };
    Some(match rng.below(9) {
        0 => node(Op::Unary(*rng.pick(&UNARY)), vec![operand]),
        1 => node(Op::Activation(*rng.pick(&ACTIVATIONS)), vec![operand]),
        2 => {
            let other = new_input(recipe, broadcastable(rng, ty.shape()));
            node(rng.pick(&BINARY).clone(), vec![operand, other])
        }
        3 if rank >= 1 => {
            let k = ty.shape().dims()[rank - 1].get();
            let weights = new_input(recipe, vec![k, extent(rng)]);
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
            let other = new_input(recipe, extents);
            node(Op::Concat(Axis::new(axis)), vec![operand, other])
        }
        _ => return None,
    })
}

/// A new input with `extents`.
fn new_input(recipe: &mut Recipe, extents: Vec<u64>) -> Operand {
    recipe.inputs.push(extents);
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

/// Extents for a first input of rank 0 to 3, rank 2 most often.
pub(crate) fn first_input(rng: &mut Rng) -> Vec<u64> {
    let rank = *rng.pick(&[0, 1, 2, 2, 3]);
    dims(rng, rank)
}
