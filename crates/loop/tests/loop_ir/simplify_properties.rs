//! Properties of the simplifier on random expressions: it never changes a
//! value within the variables' ranges, and it is idempotent.

use proptest::prelude::*;
use ptxc_core::Idx;
use ptxc_loop::{ExprArena, ExprId, Interval, Simplifier, VarId, VarRanges};

/// Variables the generated expressions read.
const VARIABLES: usize = 3;

/// A random expression, built into an arena by `build`.
#[derive(Clone, Debug)]
enum Tree {
    Var(usize),
    Const(i64),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    FloorDiv(Box<Self>, Box<Self>),
    FloorMod(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
}

/// Expressions up to depth 5 over small constants, weighted toward the
/// division and modulo by constants that schedules produce. Values stay far
/// below `i64`'s range, so evaluation fails only on division by zero.
fn tree() -> impl Strategy<Value = Tree> {
    let leaf =
        prop_oneof![(0..VARIABLES).prop_map(Tree::Var), (-16_i64..=16).prop_map(Tree::Const)];
    leaf.prop_recursive(5, 32, 2, |inner| {
        let divisor = prop_oneof![(1_i64..=8).prop_map(Tree::Const), inner.clone()];
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Add(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Mul(Box::new(a), Box::new(b))),
            (inner.clone(), divisor.clone())
                .prop_map(|(a, b)| Tree::FloorDiv(Box::new(a), Box::new(b))),
            (inner.clone(), divisor).prop_map(|(a, b)| Tree::FloorMod(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Min(Box::new(a), Box::new(b))),
            (inner.clone(), inner).prop_map(|(a, b)| Tree::Max(Box::new(a), Box::new(b))),
        ]
    })
}

fn build(arena: &mut ExprArena, tree: &Tree) -> ExprId {
    let pair = |arena: &mut ExprArena, a: &Tree, b: &Tree| (build(arena, a), build(arena, b));
    match tree {
        Tree::Var(index) => arena.var(VarId::new(*index)),
        Tree::Const(value) => arena.constant(*value),
        Tree::Add(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.add(a, b)
        }
        Tree::Mul(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.mul(a, b)
        }
        Tree::FloorDiv(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.floor_div(a, b)
        }
        Tree::FloorMod(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.floor_mod(a, b)
        }
        Tree::Min(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.min(a, b)
        }
        Tree::Max(a, b) => {
            let (a, b) = pair(arena, a, b);
            arena.max(a, b)
        }
    }
}

/// Each variable's range, as `(low, width)`, and eight assignments within
/// them, as offsets that `assignment` wraps into each range.
fn ranges_and_assignments() -> impl Strategy<Value = (Vec<(i64, i64)>, Vec<Vec<i64>>)> {
    (
        prop::collection::vec((-12_i64..=12, 0_i64..=12), VARIABLES),
        prop::collection::vec(prop::collection::vec(0_i64..=1000, VARIABLES), 8),
    )
}

fn var_ranges(ranges: &[(i64, i64)]) -> VarRanges {
    let mut var_ranges = VarRanges::new();
    for (index, &(low, width)) in ranges.iter().enumerate() {
        var_ranges.insert(VarId::new(index), Interval::new(low, low + width).unwrap());
    }
    var_ranges
}

/// The value of each variable: its offset wrapped into its range.
fn assignment(ranges: &[(i64, i64)], offsets: &[i64]) -> Vec<i64> {
    ranges.iter().zip(offsets).map(|(&(low, width), &offset)| low + offset % (width + 1)).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn simplifying_never_changes_a_value_within_the_ranges(
        tree in tree(),
        (ranges, assignments) in ranges_and_assignments(),
    ) {
        let mut arena = ExprArena::new();
        let original = build(&mut arena, &tree);
        let var_ranges = var_ranges(&ranges);
        let simplified = Simplifier::new(&var_ranges).simplify(&mut arena, original);

        for offsets in &assignments {
            let values = assignment(&ranges, offsets);
            let lookup = |var: VarId| values.get(var.index()).copied();
            if let Some(expected) = arena.evaluate(original, &lookup) {
                prop_assert_eq!(arena.evaluate(simplified, &lookup), Some(expected), "values {:?}", values);
            }
        }
    }

    #[test]
    fn simplifying_twice_changes_nothing(tree in tree(), (ranges, _) in ranges_and_assignments()) {
        let mut arena = ExprArena::new();
        let original = build(&mut arena, &tree);
        let var_ranges = var_ranges(&ranges);
        let once = Simplifier::new(&var_ranges).simplify(&mut arena, original);

        let twice = Simplifier::new(&var_ranges).simplify(&mut arena, once);

        prop_assert_eq!(twice, once);
    }
}
