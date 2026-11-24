//! Applying each primitive to a loop tree.

mod bind;
pub(super) mod cache;
mod compute_at;
mod fuse;
mod layout;
mod pipeline;
mod reorder;
mod split;
mod tensorize;
mod unroll;
mod vectorize;
