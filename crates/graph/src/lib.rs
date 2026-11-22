//! Graph IR: an SSA dataflow graph in typed arenas, with op type rules, a verifier and a text format.

mod builder;
mod graph;
mod node;
mod ops;
mod partition;
mod rewrite;
mod tensor_type;
mod text;
mod traverse;
mod value;
mod verify;
