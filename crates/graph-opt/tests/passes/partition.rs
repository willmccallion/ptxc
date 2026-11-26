use crate::graph;
use ptxc_graph::{Graph, NodeId};
use ptxc_graph_opt::{GreedyPartitioner, Partitioner};

/// The groups the greedy partitioner makes of the graph `text`, each as
/// the positions of its nodes in topological order.
fn groups(text: &str) -> Vec<Vec<usize>> {
    let graph: Graph = graph(text);
    let order: Vec<NodeId> = graph.topological_order().unwrap();
    let partitioned = GreedyPartitioner.partition(graph).unwrap();
    partitioned
        .groups()
        .iter()
        .map(|group| {
            group.nodes().iter().map(|node| order.iter().position(|n| n == node).unwrap()).collect()
        })
        .collect()
}

#[test]
fn a_product_takes_its_bias_and_activation_as_its_epilogue() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[8, 6], %b: f32[6]) {
  %0 = matmul %x, %w : f32[4, 6]
  %1 = add %0, %b : f32[4, 6]
  %2 = relu %1 : f32[4, 6]
  return %2
}
";

    assert_eq!(groups(text), [vec![0, 1, 2]]);
}

#[test]
fn gelu_after_a_product_joins_it_though_it_reads_the_product_twice() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[8, 6]) {
  %0 = const f32[] [0.5]
  %1 = matmul %x, %w : f32[4, 6]
  %2 = erf %1 : f32[4, 6]
  %3 = add %2, %0 : f32[4, 6]
  %4 = mul %1, %3 : f32[4, 6]
  return %4
}
";

    assert_eq!(groups(text), [vec![0, 1, 2, 3]]);
}

#[test]
fn a_transpose_joins_the_product_it_feeds_but_arithmetic_does_not() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[6, 8], %v: f32[8, 6]) {
  %0 = transpose<permutation=[1, 0]> %w : f32[8, 6]
  %1 = matmul %x, %0 : f32[4, 6]
  %2 = neg %v : f32[8, 6]
  %3 = matmul %x, %2 : f32[4, 6]
  return %1, %3
}
";

    assert_eq!(groups(text), [vec![0, 1], vec![2], vec![3]]);
}

#[test]
fn elementwise_ops_feed_a_reduction_through_its_reads() {
    let text = "\
graph(%x: f32[3, 10]) {
  %0 = neg %x : f32[3, 10]
  %1 = exp %0 : f32[3, 10]
  %2 = reduce_sum<axes=[1], keep_dims=drop> %1 : f32[3]
  %3 = log %2 : f32[3]
  return %3
}
";

    assert_eq!(groups(text), [vec![0, 1, 2, 3]]);
}

#[test]
fn a_value_read_outside_its_group_is_not_fused_away() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[8, 6]) {
  %0 = matmul %x, %w : f32[4, 6]
  %1 = relu %0 : f32[4, 6]
  return %1, %0
}
";

    assert_eq!(groups(text), [vec![0], vec![1]]);
}

#[test]
fn a_producer_two_groups_read_joins_neither() {
    let text = "\
graph(%x: f32[4, 6]) {
  %0 = neg %x : f32[4, 6]
  %1 = abs %0 : f32[4, 6]
  %2 = exp %0 : f32[4, 6]
  return %1, %2
}
";

    assert_eq!(groups(text), [vec![0], vec![1], vec![2]]);
}

#[test]
fn two_reductions_stay_apart() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[8, 6]) {
  %0 = matmul %x, %w : f32[4, 6]
  %1 = reduce_sum<axes=[1], keep_dims=drop> %0 : f32[4]
  return %1
}
";

    assert_eq!(groups(text), [vec![0], vec![1]]);
}

#[test]
fn an_epilogue_that_broadcasts_its_product_stays_apart() {
    let text = "\
graph(%x: f32[4, 8], %w: f32[8, 1], %y: f32[4, 6]) {
  %0 = matmul %x, %w : f32[4, 1]
  %1 = add %0, %y : f32[4, 6]
  return %1
}
";

    assert_eq!(groups(text), [vec![0], vec![1]]);
}
