use ptxc_core::convert::int_to_f64;
use ptxc_core::{DType, HostTensor, TensorData};
use ptxc_testkit::families;
use ptxc_testkit::generate::{Generator, Operand, Recipe, Seed, Source};
use std::collections::HashSet;

const SEEDS: u64 = 400;

fn recipes() -> Vec<Recipe> {
    let generator = Generator::default();
    (0..SEEDS).map(|seed| generator.generate(Seed(seed))).collect()
}

fn values(tensor: &HostTensor) -> Vec<f64> {
    match tensor.data() {
        TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E4M3Fn(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E5M2(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::I8(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::U8(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::I64(v) => v.iter().map(|&x| int_to_f64(x)).collect(),
        TensorData::Bool(v) => v.iter().map(|&x| f64::from(u8::from(x))).collect(),
    }
}

#[test]
fn a_seed_builds_the_same_recipe_every_time() {
    let generator = Generator::default();

    let first = format!("{:?}", generator.generate(Seed(7)));
    let again = format!("{:?}", Generator::default().generate(Seed(7)));
    let other = format!("{:?}", generator.generate(Seed(8)));

    assert_eq!(first, again);
    assert_ne!(first, other);
}

#[test]
fn every_recipe_builds_a_graph_with_outputs() {
    for (seed, recipe) in recipes().iter().enumerate() {
        let case = recipe.build().unwrap_or_else(|| panic!("seed {seed} does not build"));

        assert!(!case.graph.outputs().is_empty(), "seed {seed}");
    }
}

#[test]
fn the_seeds_splice_in_every_op_family() {
    let spliced: HashSet<&str> = recipes()
        .iter()
        .flat_map(|recipe| &recipe.inputs)
        .filter_map(|input| match input.source {
            Source::Family(name) => Some(name),
            Source::Moderate => None,
        })
        .collect();

    let every: HashSet<&str> = families::all().iter().map(|family| family.name).collect();
    assert_eq!(spliced, every);
}

#[test]
fn nodes_read_earlier_outputs_including_spliced_ones() {
    let recipes = recipes();
    let reads_output =
        |operands: &[Operand]| operands.iter().any(|o| matches!(o, Operand::Output { .. }));
    let spliced_reads_output = recipes.iter().any(|recipe| {
        recipe.nodes.iter().any(|node| {
            reads_output(&node.operands)
                && node.operands.iter().any(|&operand| match operand {
                    Operand::Input(n) => matches!(recipe.inputs[n].source, Source::Family(_)),
                    Operand::Output { .. } => false,
                })
        })
    });

    assert!(recipes.iter().flat_map(|r| &r.nodes).any(|node| reads_output(&node.operands)));
    assert!(spliced_reads_output);
}

#[test]
fn some_recipes_compare_with_the_reference_and_some_do_not() {
    let recipes = recipes();

    let comparable = recipes.iter().filter(|recipe| recipe.compares_with_reference()).count();

    assert!(comparable > 0 && comparable < recipes.len(), "{comparable}");
}

#[test]
fn moderate_inputs_are_small_and_take_both_signs() {
    let moderate: Vec<f64> = recipes()
        .iter()
        .flat_map(|recipe| &recipe.inputs)
        .filter(|input| input.source == Source::Moderate)
        .flat_map(|input| values(&input.tensor))
        .collect();

    assert!(moderate.iter().all(|x| (-8.0..=8.0).contains(x)), "out of range");
    assert!(moderate.iter().any(|&x| x < 0.0) && moderate.iter().any(|&x| x > 0.0));
    assert!(moderate.iter().any(|&x| x.fract() != 0.0), "no fractional floats");
}

#[test]
fn moderate_integer_inputs_take_both_signs() {
    let integers: Vec<f64> = recipes()
        .iter()
        .flat_map(|recipe| &recipe.inputs)
        .filter(|input| input.source == Source::Moderate && input.tensor.dtype() == DType::I32)
        .flat_map(|input| values(&input.tensor))
        .collect();

    assert!(integers.iter().any(|&x| x < 0.0) && integers.iter().any(|&x| x > 0.0));
}

#[test]
fn inputs_draw_empty_unit_and_longer_extents() {
    let extents: HashSet<u64> = recipes()
        .iter()
        .flat_map(|recipe| &recipe.inputs)
        .flat_map(|input| input.tensor.shape().dims().iter().map(|dim| dim.get()))
        .collect();

    assert!(extents.contains(&0) && extents.contains(&1));
    assert!(extents.iter().any(|&extent| extent > 16));
}
