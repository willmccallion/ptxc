//! Canonicalisations that give each computation one form.

use crate::rewrite::pattern::{Capture, Pattern};
use crate::rewrite::rule::{Guard, RewriteRule};
use crate::rewrite::template::Template;

/// The rules' operand.
const INPUT: Capture = Capture("x");

/// The rules' matched op's output.
const OUTPUT: Capture = Capture("y");

/// The canonical rules: a reshape to its input's own shape, a transpose
/// that keeps every axis and a cast to its input's own type each read
/// their input unchanged, so they are removed, as MLIR's folders remove
/// them.
pub(crate) fn canonical() -> Vec<RewriteRule> {
    vec![
        removed("reshape-to-own-shape", "reshape", Guard::SameType(INPUT, OUTPUT)),
        removed("transpose-in-place", "transpose", Guard::IdentityTranspose(OUTPUT)),
        removed("cast-to-own-type", "cast", Guard::SameType(INPUT, OUTPUT)),
    ]
}

/// The rule `name`, which removes an `op` of one operand where `guard`
/// holds, its readers reading the operand.
fn removed(name: &'static str, op: &'static str, guard: Guard) -> RewriteRule {
    let lhs = Pattern::Op { op, operands: vec![Pattern::Any(INPUT)], capture: OUTPUT };
    RewriteRule { name, lhs, rhs: Template::Captured(INPUT), guard }
}
