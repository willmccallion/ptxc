//! Topological order of a graph's nodes.

use crate::graph::Graph;
use crate::node::NodeId;
use ptxc_core::IndexVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use thiserror::Error;

/// The graph has a cycle; the node is on it or downstream of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("the graph has a cycle reaching node {0:?}")]
pub struct CycleError(pub NodeId);

impl Graph {
    /// The live nodes ordered so every node comes after the producers of its
    /// operands (Kahn's algorithm). The lowest ready id always goes next, so
    /// the order is deterministic, and a graph straight from `GraphBuilder`
    /// comes out in the order its nodes were added.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError> {
        let mut pending: IndexVec<NodeId, usize> = self.nodes.iter().map(|_| 0).collect();
        let mut consumers: IndexVec<NodeId, Vec<NodeId>> =
            self.nodes.iter().map(|_| Vec::new()).collect();
        for (node, contents) in self.nodes() {
            for &operand in contents.operands() {
                if let Some(producer) = self.producer(operand) {
                    pending[node] += 1;
                    consumers[producer].push(node);
                }
            }
        }

        let mut ready: BinaryHeap<Reverse<NodeId>> =
            self.nodes().map(|(id, _)| id).filter(|&id| pending[id] == 0).map(Reverse).collect();
        let mut order = Vec::new();
        while let Some(Reverse(node)) = ready.pop() {
            order.push(node);
            for &consumer in &consumers[node] {
                pending[consumer] -= 1;
                if pending[consumer] == 0 {
                    ready.push(Reverse(consumer));
                }
            }
        }

        match self.nodes().find(|&(id, _)| pending[id] > 0) {
            Some((stuck, _)) => Err(CycleError(stuck)),
            None => Ok(order),
        }
    }
}
