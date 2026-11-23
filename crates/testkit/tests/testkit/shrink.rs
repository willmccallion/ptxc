use ptxc_core::{Axis, DType};
use ptxc_graph::{BinaryOp, KeepDims, Op, ReduceAttrs, ReduceOp, UnaryOp};
use ptxc_testkit::generate::{Node, Operand, Recipe, Seed, generate};
use ptxc_testkit::shrink::shrink;

fn node(op: Op, operands: &[Operand]) -> Node {
    Node { op, operands: operands.to_vec() }
}

fn sum_over_first_axis() -> Op {
    Op::Reduce(ReduceAttrs {
        op: ReduceOp::Sum,
        axes: vec![Axis::new(0)],
        keep_dims: KeepDims::Keep,
    })
}

/// Whether some reduction reads a matrix product.
fn reduces_a_product(recipe: &Recipe) -> bool {
    recipe.nodes.iter().any(|reader| {
        matches!(reader.op, Op::Reduce(_))
            && reader.operands.iter().any(|&operand| match operand {
                Operand::Node(n) => recipe.nodes[n].op == Op::MatMul,
                Operand::Input(_) => false,
            })
    })
}

#[test]
fn a_failure_shrinks_to_the_nodes_and_extents_that_cause_it() {
    let failing = Recipe {
        dtype: DType::F32,
        inputs: vec![vec![2, 4], vec![4], vec![4, 2]],
        nodes: vec![
            node(Op::Binary(BinaryOp::Add), &[Operand::Input(0), Operand::Input(1)]),
            node(Op::MatMul, &[Operand::Node(0), Operand::Input(2)]),
            node(sum_over_first_axis(), &[Operand::Node(1)]),
        ],
        data: 3,
    };

    let shrunk = shrink(failing, reduces_a_product);

    let expected = Recipe {
        dtype: DType::F32,
        inputs: vec![vec![1, 1], vec![1, 1]],
        nodes: vec![
            node(Op::MatMul, &[Operand::Input(0), Operand::Input(1)]),
            node(sum_over_first_axis(), &[Operand::Node(0)]),
        ],
        data: 3,
    };
    assert_eq!(shrunk, expected);
}

#[test]
fn a_narrow_float_failure_shrinks_to_f32_and_keeps_a_node() {
    let failing = Recipe {
        dtype: DType::F16,
        inputs: vec![vec![1]],
        nodes: vec![node(Op::Unary(UnaryOp::Abs), &[Operand::Input(0)])],
        data: 3,
    };

    let shrunk = shrink(failing.clone(), |_| true);

    assert_eq!(shrunk, Recipe { dtype: DType::F32, ..failing });
}

#[test]
fn a_generated_failure_shrinks_to_one_node_on_extents_of_at_most_one() {
    let has_matmul = |recipe: &Recipe| recipe.nodes.iter().any(|node| node.op == Op::MatMul);
    let failing = (0..400)
        .map(|seed| generate(Seed(seed)))
        .find(|recipe| has_matmul(recipe) && recipe.nodes.len() >= 3)
        .unwrap();

    let shrunk = shrink(failing, has_matmul);

    assert_eq!(shrunk.nodes.len(), 1, "{shrunk:?}");
    assert_eq!(shrunk.nodes[0].op, Op::MatMul);
    assert_eq!(shrunk.inputs.len(), 2, "{shrunk:?}");
    assert!(shrunk.inputs.iter().flatten().all(|&extent| extent <= 1), "{shrunk:?}");
}
