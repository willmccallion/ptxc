//! Shrinks a failing recipe to a smaller one that still fails (D13).
//!
//! Each round tries bypassing a node, dropping an input nothing reads, and
//! halving an extent everywhere it appears, and keeps the first candidate
//! that builds and still fails, until none does.

use crate::families::cases::single_node;
use crate::generate::{Operand, Recipe};
use ptxc_core::{Axis, Dim, Evaluate, HostTensor};
use ptxc_graph::{Op, SliceAttrs, SliceAxis, SliceStep};
use ptxc_interp::Interpreter;

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
        if !recipe.reads(Operand::Input(input)) {
            all.push(without_input(recipe, input));
        }
    }
    let mut extents: Vec<u64> = recipe
        .inputs
        .iter()
        .flat_map(|input| input.tensor.shape().dims().iter().map(|dim| dim.get()))
        .filter(|&extent| extent > 1)
        .collect();
    extents.sort_unstable();
    extents.dedup();
    all.extend(extents.into_iter().filter_map(|extent| with_extent_halved(recipe, extent)));
    all
}

/// `recipe` with every input axis of extent `extent` cut to its first half,
/// so extents that inputs must share, such as a matrix product's inner one,
/// shrink as one.
fn with_extent_halved(recipe: &Recipe, extent: u64) -> Option<Recipe> {
    let mut next = recipe.clone();
    for input in &mut next.inputs {
        input.tensor = halved(&input.tensor, extent)?;
    }
    Some(next)
}

/// `tensor` cut to the first half of every axis of extent `extent`, by the
/// reference interpreter's `Slice`.
fn halved(tensor: &HostTensor, extent: u64) -> Option<HostTensor> {
    let axes: Vec<SliceAxis> = tensor
        .shape()
        .dims()
        .iter()
        .enumerate()
        .filter(|(_, dim)| dim.get() == extent)
        .map(|(axis, _)| SliceAxis {
            axis: Axis::new(axis),
            start: 0,
            step: SliceStep::one(),
            length: Dim::new(extent / 2),
        })
        .collect();
    if axes.is_empty() {
        return Some(tensor.clone());
    }
    let inputs = std::slice::from_ref(tensor);
    let graph = single_node(Op::Slice(SliceAttrs { axes }), inputs);
    Interpreter::new(&graph).evaluate(inputs).ok()?.pop()
}

/// `recipe` without node `node`, whose readers read its operand `operand`
/// in place of any of its outputs.
fn bypassed(recipe: &Recipe, node: usize, operand: usize) -> Recipe {
    let replacement = recipe.nodes[node].operands[operand];
    let mut next = recipe.clone();
    let _removed = next.nodes.remove(node);
    for later in &mut next.nodes {
        for reference in &mut later.operands {
            *reference = match *reference {
                Operand::Output { node: n, .. } if n == node => replacement,
                Operand::Output { node: n, index } if n > node => {
                    Operand::Output { node: n - 1, index }
                }
                other => other,
            };
        }
    }
    next
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
