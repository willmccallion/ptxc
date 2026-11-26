//! The `MemoryPlanner` trait: offsets in one arena for buffers with
//! lifetimes.

use super::lifetime::Lifetime;
use ptxc_core::{ByteOffset, ByteSize};

/// A buffer to place in the arena.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Request {
    /// Its size.
    pub(crate) size: ByteSize,
    /// When it is live; `None` if no step touches it.
    pub(crate) lifetime: Option<Lifetime>,
}

/// A plan: an offset for each buffer and the arena holding them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ArenaPlan {
    /// Each request's offset, in the order of the requests.
    pub(crate) offsets: Vec<ByteOffset>,
    /// The arena's size: the end of the buffer ending last.
    pub(crate) size: ByteSize,
}

/// Places buffers in one arena so that two live at the same step never
/// share a byte, each at an offset aligned for any buffer.
pub(crate) trait MemoryPlanner {
    /// A plan for `requests`; `None` if an offset passes 2^64.
    fn plan(&self, requests: &[Request]) -> Option<ArenaPlan>;
}
