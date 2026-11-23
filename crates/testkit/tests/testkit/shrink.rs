use ptxc_core::{Axis, Dim, HostTensor};
use ptxc_graph::{BinaryOp, KeepDims, Op, ReduceAttrs, ReduceOp, SplitAttrs, UnaryOp};
use ptxc_testkit::families::values::shape;
use ptxc_testkit::generate::{Generator, Input, Node, Operand, Recipe, Seed, Source};
use ptxc_testkit::shrink::shrink;

fn node(op: Op, operands: &[Operand]) -> Node {
    Node { op, operands: operands.to_vec() }
}

const fn output(node: usize) -> Operand {
    Operand::Output { node, index: 0 }
}

/// An f32 input of `dims` holding 1, 2, 3, ...
fn counting(dims: &[u64]) -> Input {
    let count = shape(dims).element_count().get();
    let data: Vec<f32> = (1..=count).map(|n| f32::from(u16::try_from(n).unwrap())).collect();
    Input { tensor: HostTensor::new(shape(dims), data.into()).unwrap(), source: Source::Moderate }
}

fn extents(recipe: &Recipe) -> Vec<Vec<u64>> {
    let dims = |input: &Input| input.tensor.shape().dims().iter().map(|dim| dim.get()).collect();
    recipe.inputs.iter().map(dims).collect()
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
                Operand::Output { node, .. } => recipe.nodes[node].op == Op::MatMul,
                Operand::Input(_) => false,
            })
    })
}

#[test]
fn a_failure_shrinks_to_the_nodes_and_extents_that_cause_it() {
    let failing = Recipe {
        inputs: vec![counting(&[2, 4]), counting(&[4]), counting(&[4, 2])],
        nodes: vec![
            node(Op::Binary(BinaryOp::Add), &[Operand::Input(0), Operand::Input(1)]),
            node(Op::MatMul, &[output(0), Operand::Input(2)]),
            node(sum_over_first_axis(), &[output(1)]),
        ],
    };

    let shrunk = shrink(failing, reduces_a_product);

    let expected = vec![
        node(Op::MatMul, &[Operand::Input(0), Operand::Input(1)]),
        node(sum_over_first_axis(), &[output(0)]),
    ];
    assert_eq!(shrunk.nodes, expected);
    assert_eq!(extents(&shrunk), [[1, 1], [1, 1]]);
}

#[test]
fn a_node_with_several_outputs_is_bypassed_whole() {
    let halves = SplitAttrs { axis: Axis::new(0), sizes: vec![Dim::new(2), Dim::new(2)] };
    let failing = Recipe {
        inputs: vec![counting(&[4])],
        nodes: vec![
            node(Op::Split(halves), &[Operand::Input(0)]),
            node(Op::Unary(UnaryOp::Neg), &[Operand::Output { node: 0, index: 1 }]),
        ],
    };
    let negates = |recipe: &Recipe| recipe.nodes.iter().any(|n| n.op == Op::Unary(UnaryOp::Neg));

    let shrunk = shrink(failing, negates);

    assert_eq!(shrunk.nodes, [node(Op::Unary(UnaryOp::Neg), &[Operand::Input(0)])]);
    assert_eq!(extents(&shrunk), [[1]]);
}

#[test]
fn a_generated_failure_shrinks_to_one_node_on_extents_of_at_most_one() {
    let has_matmul = |recipe: &Recipe| recipe.nodes.iter().any(|node| node.op == Op::MatMul);
    let generator = Generator::default();
    let failing = (0..400)
        .map(|seed| generator.generate(Seed(seed)))
        .find(|recipe| has_matmul(recipe) && recipe.nodes.len() >= 3)
        .unwrap();

    let shrunk = shrink(failing, has_matmul);

    assert_eq!(shrunk.nodes.len(), 1, "{shrunk:?}");
    assert_eq!(shrunk.nodes[0].op, Op::MatMul);
    assert!(shrunk.inputs.len() <= 2, "{shrunk:?}");
    assert!(extents(&shrunk).iter().flatten().all(|&extent| extent <= 1), "{shrunk:?}");
}
