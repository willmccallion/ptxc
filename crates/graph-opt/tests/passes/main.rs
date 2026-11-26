//! Graph passes through their public API, on graphs written as text.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod canonicalize;
mod const_fold;
mod cse;
mod dce;
mod manager;
mod partition;
mod simplify;

use ptxc_graph::Graph;
use ptxc_graph::text::{parse, print};
use ptxc_graph_opt::{Pass, PassReport};

/// The graph `text` describes.
pub(crate) fn graph(text: &str) -> Graph {
    parse(text, &[]).unwrap()
}

/// `pass` run over the graph `text` describes: the graph it leaves,
/// printed, and its report.
pub(crate) fn run(pass: &dyn Pass, text: &str) -> (String, PassReport) {
    let mut graph = graph(text);
    let report = pass.run(&mut graph).unwrap();
    assert_eq!(graph.verify(), Ok(()));
    (print(&graph), report)
}
