use crate::run;
use ptxc_graph_opt::{Dce, PassReport};

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

#[test]
fn a_chain_no_output_reads_is_removed() {
    let text = "\
graph(%x: f32[4]) {
  %0 = neg %x : f32[4]
  %1 = abs %0 : f32[4]
  %2 = exp %x : f32[4]
  return %2
}
";

    let (after, report) = run(&Dce, text);

    let expected = "\
graph(%x: f32[4]) {
  %0 = exp %x : f32[4]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_node_stays_while_any_of_its_outputs_is_read() {
    let text = "\
graph(%x: f32[3, 4]) {
  %0, %1 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  %2 = neg %1 : f32[2, 4]
  return %2
}
";

    let (after, report) = run(&Dce, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}

#[test]
fn a_node_none_of_whose_outputs_is_read_is_removed() {
    let text = "\
graph(%x: f32[3, 4]) {
  %0, %1 = split<axis=0, sizes=[1, 2]> %x : f32[1, 4], f32[2, 4]
  %2 = neg %x : f32[3, 4]
  return %2
}
";

    let (after, report) = run(&Dce, text);

    let expected = "\
graph(%x: f32[3, 4]) {
  %0 = neg %x : f32[3, 4]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_constant_only_dead_nodes_read_goes_with_them() {
    let text = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = const f32[2] [3.0, 4.0]
  %2 = add %x, %0 : f32[2]
  %3 = mul %x, %1 : f32[2]
  return %3
}
";

    let (after, report) = run(&Dce, text);

    let expected = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [3.0, 4.0]
  %1 = mul %x, %0 : f32[2]
  return %1
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn a_constant_nothing_reads_is_removed_alone() {
    let text = "\
graph(%x: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = neg %x : f32[2]
  return %1
}
";

    let (after, report) = run(&Dce, text);

    let expected = "\
graph(%x: f32[2]) {
  %0 = neg %x : f32[2]
  return %0
}
";
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn inputs_outputs_and_what_they_need_are_kept() {
    let text = "\
graph(%x: f32[2], %y: f32[2]) {
  %0 = const f32[2] [1.0, 2.0]
  %1 = add %x, %0 : f32[2]
  return %1, %x, %0
}
";

    let (after, report) = run(&Dce, text);

    assert_eq!(after, text);
    assert_eq!(report, UNCHANGED);
}
