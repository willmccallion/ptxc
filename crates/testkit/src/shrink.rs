//! Shrinks a failing recipe to a smaller one that still fails (D13).
//!
//! Each round tries bypassing a node, dropping an input nothing reads,
//! halving an extent everywhere it appears, and computing in f32, and keeps
//! the first candidate that builds and still fails, until none does.

use crate::generate::{Operand, Recipe};
use ptxc_core::DType;

/// `recipe` made as small as its candidates allow while `fails` holds of
/// it. `fails` is only asked about recipes that build.
pub fn shrink(mut recipe: Recipe, mut fails: impl FnMut(&Recipe) -> bool) -> Recipe {
    loop {
        let smaller = candidates(&recipe)
            .into_iter()
            .find(|candidate| candidate.build().is_some() && fails(candidate));
        match smaller {
            Some(candidate) => recipe = candidate,
            None => return recipe,
        }
    }
}

/// Every one-step edit of `recipe`, the largest reductions first.
fn candidates(recipe: &Recipe) -> Vec<Recipe> {
    let mut all = Vec::new();
    for node in (0..recipe.nodes.len()).rev() {
        for operand in 0..recipe.nodes[node].operands.len() {
            all.push(bypassed(recipe, node, operand));
        }
    }
    for input in 0..recipe.inputs.len() {
        if !reads_input(recipe, input) {
            all.push(without_input(recipe, input));
        }
    }
    let mut extents: Vec<u64> =
        recipe.inputs.iter().flatten().copied().filter(|&e| e > 1).collect();
    extents.sort_unstable();
    extents.dedup();
    for extent in extents {
        all.push(with_extent_halved(recipe, extent));
    }
    if matches!(recipe.dtype, DType::F16 | DType::Bf16) {
        all.push(Recipe { dtype: DType::F32, ..recipe.clone() });
    }
    all
}

/// `recipe` with every input extent equal to `extent` halved together, so
/// extents that inputs must share, such as a matrix product's inner one,
/// shrink as one.
fn with_extent_halved(recipe: &Recipe, extent: u64) -> Recipe {
    let mut next = recipe.clone();
    for e in next.inputs.iter_mut().flatten() {
        if *e == extent {
            *e = extent / 2;
        }
    }
    next
}

/// `recipe` without node `node`, whose readers read its operand `operand`
/// instead.
fn bypassed(recipe: &Recipe, node: usize, operand: usize) -> Recipe {
    let replacement = recipe.nodes[node].operands[operand];
    let mut next = recipe.clone();
    let _removed = next.nodes.remove(node);
    for later in &mut next.nodes {
        for reference in &mut later.operands {
            *reference = match *reference {
                Operand::Node(n) if n == node => replacement,
                Operand::Node(n) if n > node => Operand::Node(n - 1),
                other => other,
            };
        }
    }
    next
}

fn reads_input(recipe: &Recipe, input: usize) -> bool {
    recipe.nodes.iter().any(|node| node.operands.contains(&Operand::Input(input)))
}

/// `recipe` without input `input`, which nothing reads.
fn without_input(recipe: &Recipe, input: usize) -> Recipe {
    let mut next = recipe.clone();
    let _removed = next.inputs.remove(input);
    for node in &mut next.nodes {
        for reference in &mut node.operands {
            if let Operand::Input(n) = *reference
                && n > input
            {
                *reference = Operand::Input(n - 1);
            }
        }
    }
    next
}
