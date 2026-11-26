use crate::run;
use ptxc_graph_opt::{Canonicalize, PassReport};

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

#[test]
fn ops_that_leave_their_input_as_it_was_are_removed_to_a_fixed_point() {
    let text = "\
graph(%x: f32[2, 3]) {
  %0 = reshape<shape=[2, 3]> %x : f32[2, 3]
  %1 = transpose<permutation=[0, 1]> %0 : f32[2, 3]
  %2 = cast<to=f32> %1 : f32[2, 3]
  %3 = neg %2 : f32[2, 3]
  return %3
}
";

    let (after, report) = run(&Canonicalize, text);

    let expected = "\
graph(%x: f32[2, 3]) {
  %0 = neg %x : f32[2, 3]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_graph_output_an_identity_gave_is_read_from_its_input() {
    let text = "\
graph(%x: f32[4]) {
  %0 = cast<to=f32> %x : f32[4]
  return %0
}
";

    let (after, report) = run(&Canonicalize, text);

    assert_eq!(after, "graph(%x: f32[4]) {\n  return %x\n}\n");
    assert_eq!(report, CHANGED);
}

#[test]
fn a_transpose_that_swaps_a_square_s_axes_stays_though_its_type_is_its_input_s() {
    let text = "\
graph(%x: f32[3, 3]) {
  %0 = transpose<permutation=[1, 0]> %x : f32[3, 3]
  return %0
}
";

    let (after, report) = run(&Canonicalize, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_reshape_or_cast_that_changes_the_type_stays() {
    let text = "\
graph(%x: f32[2, 3]) {
  %0 = reshape<shape=[3, 2]> %x : f32[3, 2]
  %1 = cast<to=f16> %0 : f16[3, 2]
  return %1
}
";

    let (after, report) = run(&Canonicalize, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_graph_with_nothing_to_canonicalise_is_left_as_it_was() {
    let text = "\
graph(%x: f32[4]) {
  %0 = const f32[] [2.0]
  %1 = neg %x : f32[4]
  return %1
}
";

    let (after, report) = run(&Canonicalize, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}
