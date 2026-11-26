use crate::run;
use ptxc_graph_opt::{Cse, PassReport};

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

#[test]
fn duplicate_chains_merge_into_the_first() {
    let text = "\
graph(%x: f32[4]) {
  %0 = neg %x : f32[4]
  %1 = neg %x : f32[4]
  %2 = abs %0 : f32[4]
  %3 = abs %1 : f32[4]
  %4 = add %2, %3 : f32[4]
  return %4
}
";

    let (after, report) = run(&Cse, text);

    let expected = "\
graph(%x: f32[4]) {
  %0 = neg %x : f32[4]
  %1 = abs %0 : f32[4]
  %2 = add %1, %1 : f32[4]
  return %2
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn nodes_that_differ_in_an_attribute_or_an_operand_stay_apart() {
    let text = "\
graph(%x: f32[3, 4], %y: f32[3, 4]) {
  %0, %1 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  %2, %3 = split<axis=0, sizes=[2, 1]> %x : f32[2, 4], f32[1, 4]
  %4 = neg %x : f32[3, 4]
  %5 = neg %y : f32[3, 4]
  return %0, %3, %4, %5
}
";

    let (after, report) = run(&Cse, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn every_output_of_a_repeated_node_moves_to_the_first() {
    let text = "\
graph(%x: f32[3, 4]) {
  %0, %1 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  %2, %3 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  return %0, %3, %2, %1
}
";

    let (after, report) = run(&Cse, text);

    let expected = "\
graph(%x: f32[3, 4]) {
  %0, %1 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  return %0, %1, %0, %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn equal_constants_merge_and_so_do_the_nodes_reading_them() {
    let text = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = const f32[2] [1.0, 2.0]
  %2 = add %x, %0 : f32[2]
  %3 = add %x, %1 : f32[2]
  return %2, %3
}
";

    let (after, report) = run(&Cse, text);

    let expected = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = add %x, %0 : f32[2]
  return %1, %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn constants_merge_only_with_the_same_type_and_bytes() {
    let text = "\
graph() {
  %0 = const f32[1] [0.0]
  %1 = const f32[1] [-0.0]
  %2 = const i32[1] [0]
  %3 = const f32[1, 1] [0.0]
  return %0, %1, %2, %3
}
";

    let (after, report) = run(&Cse, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}
