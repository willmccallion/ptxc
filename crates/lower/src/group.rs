//! Lowers each group of a partitioned graph to its stages.

use crate::error::LowerError;
use crate::fuse::lower_group;
use crate::stage::LoweredGroup;
use ptxc_core::IndexVec;
use ptxc_graph::{GroupId, Partitioned};

/// Every group lowered, in the partition's order: each group reads only
/// graph values computed by earlier groups.
pub fn lower(partitioned: &Partitioned) -> Result<IndexVec<GroupId, LoweredGroup>, LowerError> {
    partitioned
        .groups()
        .iter_enumerated()
        .map(|(id, group)| lower_group(partitioned.graph(), id, group))
        .collect()
}
