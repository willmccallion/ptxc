//! Splices an op-family case into the recipe: the family's op, each operand
//! either an earlier value of the same type or the family's own input,
//! whose values reach that op's edge cases.

use super::connect::push;
use super::inputs::{Input, Source};
use super::recipe::{Node, Operand, Recipe, tensor_type};
use super::shapes::Rng;
use crate::families::Family;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use ptxc_core::HostTensor;
use ptxc_graph::TensorType;

/// Adds a node of a case drawn from one of `families`, reading
/// `values`, the recipe's values, where their types allow. `None` if the
/// family drew nothing.
pub(crate) fn splice(
    rng: &mut Rng,
    families: &[Family],
    recipe: &mut Recipe,
    values: &[(Operand, TensorType)],
) -> Option<()> {
    let family = rng.pick(families);
    let mut runner = seeded_runner(rng);
    let (graph, inputs) = family.graphs.new_tree(&mut runner).ok()?.current();
    let (_, node) = graph.nodes().next()?;
    let op = node.op().clone();
    let operands = inputs
        .into_iter()
        .map(|tensor| operand(rng, recipe, values, tensor, family.name))
        .collect();
    recipe.nodes.push(Node { op, operands });
    Some(())
}

/// Half the time, an earlier value of `tensor`'s type if there is one;
/// otherwise `tensor` as a new input.
fn operand(
    rng: &mut Rng,
    recipe: &mut Recipe,
    values: &[(Operand, TensorType)],
    tensor: HostTensor,
    family: &'static str,
) -> Operand {
    let ty = tensor_type(&tensor);
    let same: Vec<Operand> =
        values.iter().filter(|(_, other)| *other == ty).map(|&(operand, _)| operand).collect();
    if !same.is_empty() && rng.chance(1, 2) {
        return *rng.pick(&same);
    }
    push(recipe, Input { tensor, source: Source::Family(family) })
}

/// A proptest runner whose `ChaCha` stream is seeded from `rng`, so a seed
/// draws the same family case on every platform.
// NOTE: D13 has a seed reproduce its case exactly; through proptest that
// holds only while proptest samples the same way, so a recipe keeps its
// tensors and a shrunk case never depends on the seed (D53).
fn seeded_runner(rng: &mut Rng) -> TestRunner {
    let seed: Vec<u8> = (0..4).flat_map(|_| rng.next().to_le_bytes()).collect();
    TestRunner::new_with_rng(Config::default(), TestRng::from_seed(RngAlgorithm::ChaCha, &seed))
}
