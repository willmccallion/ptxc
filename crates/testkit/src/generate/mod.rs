//! Random well-typed graphs from a seed (D13). The choices that build each
//! graph are kept as a `Recipe`, which `shrink` edits.
//!
//! NOTE: test code shared as a library, like the op-family generators: it
//! panics on its own invariants rather than reporting them (D52).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test generators panic on cases they build wrong (D52)"
)]

mod graph;
mod ops;
mod shapes;

pub use graph::{Case, Node, Operand, Recipe};

use crate::families::values::shape;
use ops::{first_input, propose};
use ptxc_core::DType;
use ptxc_graph::{Layout, TensorType};
use shapes::Rng;

/// A fuzz case's seed: the same seed builds the same recipe everywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Seed(pub u64);

/// The element types drawn, f32 most often.
// NOTE: TESTING.md first planned an element type per subgraph; one type per
// graph keeps every binary op well typed without casts.
const DTYPES: [DType; 6] =
    [DType::F32, DType::F32, DType::F32, DType::F16, DType::Bf16, DType::I32];

/// The most nodes a recipe has.
const MAX_NODES: u64 = 6;

/// Proposals tried for each node before giving up on it.
const ATTEMPTS: usize = 16;

/// The recipe `seed` draws: an element type, a first input, and up to
/// `MAX_NODES` nodes, each reading an earlier value.
pub fn generate(seed: Seed) -> Recipe {
    let mut rng = Rng::new(seed.0);
    let mut recipe = Recipe {
        dtype: *rng.pick(&DTYPES),
        inputs: vec![first_input(&mut rng)],
        nodes: Vec::new(),
        data: rng.next(),
    };
    for _ in 0..=rng.below(MAX_NODES) {
        for _ in 0..ATTEMPTS {
            let candidate = extended(&mut rng, &recipe);
            if let Some(candidate) = candidate.filter(|c| c.node_types().is_some()) {
                recipe = candidate;
                break;
            }
        }
    }
    recipe
}

/// `recipe` with one more node, reading its last node or one of its inputs.
fn extended(rng: &mut Rng, recipe: &Recipe) -> Option<Recipe> {
    let mut next = recipe.clone();
    let types = recipe.node_types()?;
    let operand = match types.len() {
        0 => Operand::Input(0),
        n if rng.chance(3, 4) => Operand::Node(n - 1),
        n => Operand::Node(rng.index(n)),
    };
    let ty = match operand {
        Operand::Node(n) => types[n].clone(),
        Operand::Input(n) => {
            TensorType::new(recipe.dtype, shape(&recipe.inputs[n]), Layout::RowMajor)
        }
    };
    let node = propose(rng, &mut next, operand, &ty)?;
    next.nodes.push(node);
    Some(next)
}
