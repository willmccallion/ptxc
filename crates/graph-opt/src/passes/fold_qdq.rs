//! QDQ folding: quantization nodes folded into the compute they surround.

use super::rewrite_greedily;
use crate::pass::{Pass, PassError, PassReport};
use crate::rewrite::qdq;
use ptxc_graph::Graph;

/// Applies the QDQ folding rules until none matches, as TensorRT's
/// explicit quantization folds a QDQ model's quantization nodes into
/// quantized compute.
///
/// It runs before constant folding, which would otherwise dequantize
/// constant weights into float ones.
#[derive(Clone, Copy, Debug)]
pub struct FoldQdq;

impl Pass for FoldQdq {
    fn name(&self) -> &'static str {
        "fold-qdq"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        rewrite_greedily(graph, &qdq())
    }
}
