//! Random well-typed graphs from a seed (D13, D53).
//!
//! Each node either splices in an op-family case, reaching every op in the
//! v0.1.0 set, or connects a continuous op to an earlier value. The inputs
//! and nodes are kept as a `Recipe`, which `shrink` edits.
//!
//! NOTE: test code shared as a library, like the op-family generators: it
//! panics on its own invariants rather than reporting them (D52).

#![expect(clippy::unwrap_used, reason = "test generators panic on cases they build wrong (D52)")]

mod connect;
mod inputs;
mod recipe;
mod shapes;
mod splice;

pub use inputs::{Input, Source};
pub use recipe::{Case, Node, Operand, Recipe};

use crate::families::{self, Family};
use connect::connect;
use shapes::Rng;
use splice::splice;

/// A fuzz case's seed: the same seed builds the same recipe everywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Seed(pub u64);

/// The most nodes a recipe has.
const MAX_NODES: u64 = 6;

/// Proposals tried for each node before giving up on it.
const ATTEMPTS: usize = 16;

/// Draws recipes, splicing from every op family.
#[derive(Debug)]
pub struct Generator {
    families: Vec<Family>,
}

impl Default for Generator {
    fn default() -> Self {
        Self { families: families::all() }
    }
}

impl Generator {
    /// The recipe `seed` draws: up to `MAX_NODES` nodes, each an op-family
    /// case or a connector, half and half.
    pub fn generate(&self, seed: Seed) -> Recipe {
        let mut rng = Rng::new(seed.0);
        let mut recipe = Recipe::default();
        for _ in 0..=rng.below(MAX_NODES) {
            for _ in 0..ATTEMPTS {
                if let Some(next) = self.extended(&mut rng, &recipe) {
                    recipe = next;
                    break;
                }
            }
        }
        recipe
    }

    /// `recipe` with one more node, or `None` if the drawn node does not
    /// apply or its type rule rejects it.
    fn extended(&self, rng: &mut Rng, recipe: &Recipe) -> Option<Recipe> {
        let values = recipe.values()?;
        let mut next = recipe.clone();
        if rng.chance(1, 2) {
            splice(rng, &self.families, &mut next, &values)?;
        } else {
            connect(rng, &mut next, &values)?;
        }
        next.values().map(|_| next)
    }
}
