//! Declarative rewrite rules and the engines that apply them.

mod engine;
mod greedy;
mod pattern;
mod rule;
mod rules;
mod template;

pub(crate) use engine::RewriteEngine;
pub(crate) use greedy::GreedyEngine;
pub(crate) use rule::RewriteRule;
pub(crate) use rules::{algebraic, canonical};
