//! Fusion partitioning.

mod greedy;
mod partitioner;
mod policy;

pub use greedy::GreedyPartitioner;
pub use partitioner::{OneNodePerGroup, Partitioner};
