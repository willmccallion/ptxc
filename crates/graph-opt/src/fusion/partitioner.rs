//! The `Partitioner` trait.

use ptxc_graph::{Graph, PartitionError, Partitioned};

/// Splits a graph into fused groups, each run as one unit, as XLA's fusion
/// passes do.
pub trait Partitioner: std::fmt::Debug {
    /// `graph` split into groups.
    fn partition(&self, graph: Graph) -> Result<Partitioned, PartitionError>;
}

/// Each node its own group, as M2 ran every node as its own kernel: for
/// tests of what happens between groups, and to compare against fusion.
#[derive(Clone, Copy, Debug)]
pub struct OneNodePerGroup;

impl Partitioner for OneNodePerGroup {
    fn partition(&self, graph: Graph) -> Result<Partitioned, PartitionError> {
        Ok(Partitioned::one_node_per_group(graph)?)
    }
}
