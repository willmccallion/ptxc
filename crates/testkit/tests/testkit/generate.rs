use ptxc_core::{Evaluate, HostTensor, TensorData};
use ptxc_graph::Op;
use ptxc_interp::Interpreter;
use ptxc_testkit::generate::{Operand, Recipe, Seed, generate};
use std::collections::HashSet;

const SEEDS: u64 = 400;

fn recipes() -> Vec<Recipe> {
    (0..SEEDS).map(|seed| generate(Seed(seed))).collect()
}

fn values(tensor: &HostTensor) -> Vec<f64> {
    match tensor.data() {
        TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        other => panic!("unexpected input {:?}", other.dtype()),
    }
}

#[test]
fn a_seed_builds_the_same_recipe_every_time() {
    assert_eq!(generate(Seed(7)), generate(Seed(7)));
    assert_ne!(generate(Seed(7)), generate(Seed(8)));
}

#[test]
fn every_recipe_builds_a_graph_with_outputs_the_reference_runs() {
    for (seed, recipe) in recipes().iter().enumerate() {
        let case =
            recipe.build().unwrap_or_else(|| panic!("seed {seed} does not build: {recipe:?}"));

        let outputs = Interpreter::new(&case.graph).evaluate(&case.inputs);

        assert!(!case.graph.outputs().is_empty(), "seed {seed}");
        assert!(outputs.is_ok(), "seed {seed}: {outputs:?}");
    }
}

#[test]
fn the_seeds_draw_every_op_and_element_type() {
    let recipes = recipes();
    let ops: HashSet<&str> = recipes
        .iter()
        .flat_map(|recipe| &recipe.nodes)
        .map(|node| match node.op {
            Op::Unary(_) => "unary",
            Op::Activation(_) => "activation",
            Op::Binary(_) | Op::Variadic(_) => "binary",
            Op::MatMul => "matmul",
            Op::Reduce(_) => "reduce",
            Op::Softmax(_) => "softmax",
            Op::Transpose(_) => "transpose",
            Op::Reshape(_) => "reshape",
            Op::Concat(_) => "concat",
            _ => "other",
        })
        .collect();
    let dtypes: HashSet<_> = recipes.iter().map(|recipe| recipe.dtype).collect();

    let expected = [
        "unary",
        "activation",
        "binary",
        "matmul",
        "reduce",
        "softmax",
        "transpose",
        "reshape",
        "concat",
    ];
    assert_eq!(ops, expected.into_iter().collect());
    assert_eq!(dtypes.len(), 4);
}

#[test]
fn nodes_read_earlier_nodes_and_binary_operands_broadcast() {
    let recipes = recipes();
    let chained = recipes
        .iter()
        .flat_map(|recipe| &recipe.nodes)
        .any(|node| node.operands.iter().any(|operand| matches!(operand, Operand::Node(_))));
    let broadcast = recipes.iter().any(|recipe| {
        let case = recipe.build().unwrap();
        recipe.nodes.iter().any(|node| {
            let Some(&Operand::Input(second)) = node.operands.get(1) else { return false };
            matches!(node.op, Op::Binary(_) | Op::Variadic(_))
                && case.inputs[second].shape().dims().iter().any(|dim| dim.get() == 1)
        })
    });

    assert!(chained);
    assert!(broadcast);
}

#[test]
fn inputs_draw_empty_unit_and_longer_extents() {
    let extents: HashSet<u64> =
        recipes().into_iter().flat_map(|recipe| recipe.inputs).flatten().collect();

    assert!(extents.contains(&0) && extents.contains(&1));
    assert!(extents.iter().any(|&extent| extent > 16));
}

#[test]
fn input_values_are_small_and_take_both_signs() {
    let all: Vec<f64> = recipes()
        .iter()
        .flat_map(|recipe| recipe.build().unwrap().inputs)
        .flat_map(|input| values(&input))
        .collect();

    assert!(all.iter().all(|x| (-8.0..=8.0).contains(x)), "out of range");
    assert!(all.iter().any(|&x| x < 0.0) && all.iter().any(|&x| x > 0.0));
    assert!(all.iter().any(|&x| x.fract() != 0.0), "no fractional floats");
}
