//! The suite's problems: each layer of `bench/conv.json` at each of its
//! batch sizes, and the names its results use. The PyTorch baseline reads
//! the same file, so both time the same problems.

use ptxc_core::DType;
use serde::{Deserialize, Serialize};

/// The suite's definition, which the baseline shares.
const DEFINITION: &str = include_str!("../../../../../bench/conv.json");

/// `bench/conv.json`: the element type, the batch sizes and the layers.
#[derive(Deserialize)]
struct Definition {
    dtype: ConvType,
    batches: Vec<u32>,
    layers: Vec<Layer>,
}

/// The element type a suite's inputs, weights and outputs have; the sums
/// are f32.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum ConvType {
    F16,
}

impl ConvType {
    pub(super) const fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
        }
    }
}

/// The suite: its element type and its problems, each layer at each
/// batch size.
pub(super) struct Suite {
    pub(super) dtype: ConvType,
    pub(super) problems: Vec<ConvProblem>,
}

/// One convolution layer, NCHW, square: input channels, output channels
/// (`features`), input size, kernel size, stride and padding on each side.
#[derive(Clone, PartialEq, Eq, Debug, Deserialize)]
pub(super) struct Layer {
    pub(super) name: String,
    pub(super) channels: u32,
    pub(super) features: u32,
    pub(super) size: u32,
    pub(super) kernel: u32,
    pub(super) stride: u32,
    pub(super) pad: u32,
}

impl Layer {
    /// The output size, for a window the padded input holds.
    pub(super) const fn output(&self) -> u32 {
        (self.size + 2 * self.pad - self.kernel) / self.stride + 1
    }

    /// The products each output element sums: `channels × kernel²`.
    pub(super) const fn depth(&self) -> u32 {
        self.channels * self.kernel * self.kernel
    }
}

/// One problem: a layer at a batch size.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct ConvProblem {
    pub(super) layer: Layer,
    pub(super) batch: u32,
}

/// A case's parameters, as its results record them.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct ConvParams {
    pub(crate) layer: String,
    pub(crate) batch: u32,
    pub(crate) channels: u32,
    pub(crate) features: u32,
    pub(crate) size: u32,
    pub(crate) kernel: u32,
    pub(crate) stride: u32,
    pub(crate) pad: u32,
    pub(crate) dtype: String,
}

/// The suite `bench/conv.json` defines.
pub(super) fn suite() -> Result<Suite, serde_json::Error> {
    let definition: Definition = serde_json::from_str(DEFINITION)?;
    let problems = definition
        .layers
        .iter()
        .flat_map(|layer| {
            definition.batches.iter().map(|&batch| ConvProblem { layer: layer.clone(), batch })
        })
        .collect();
    Ok(Suite { dtype: definition.dtype, problems })
}

/// The problem's name, `conv/<type>/<layer>/b<batch>`.
pub(super) fn id(problem: &ConvProblem, dtype: ConvType) -> String {
    format!("conv/{}/{}/b{}", dtype.name(), problem.layer.name, problem.batch)
}

/// The problem's parameters.
pub(super) fn params(problem: &ConvProblem, dtype: ConvType) -> ConvParams {
    let Layer { name, channels, features, size, kernel, stride, pad } = problem.layer.clone();
    let (layer, batch, dtype) = (name, problem.batch, dtype.name().to_owned());
    ConvParams { layer, batch, channels, features, size, kernel, stride, pad, dtype }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{ConvType, id, suite};
    use std::collections::HashSet;

    #[test]
    fn the_suite_runs_resnet50s_23_layers_at_three_batch_sizes_in_f16() {
        let suite = suite().unwrap();

        let ids: HashSet<String> =
            suite.problems.iter().map(|problem| id(problem, suite.dtype)).collect();

        assert_eq!(suite.dtype, ConvType::F16);
        assert_eq!(suite.problems.len(), 23 * 3);
        assert_eq!(ids.len(), suite.problems.len());
        assert!(ids.contains("conv/f16/stem/b32"));
        assert!(ids.contains("conv/f16/layer4-projection/b1"));
    }

    #[test]
    fn a_layers_depth_counts_every_channel_and_tap() {
        let suite = suite().unwrap();
        let depth = |name: &str| {
            let found = suite.problems.iter().find(|problem| problem.layer.name == name).unwrap();
            found.layer.depth()
        };

        assert_eq!([depth("stem"), depth("layer4-3x3"), depth("layer4-reduce")], [147, 4608, 2048]);
    }

    #[test]
    fn every_layer_has_resnet50s_output_sizes() {
        let outputs: HashSet<(String, u32)> = suite()
            .unwrap()
            .problems
            .into_iter()
            .map(|problem| (problem.layer.name.clone(), problem.layer.output()))
            .collect();

        for (layer, size) in [
            ("stem", 112),
            ("layer1-3x3", 56),
            ("layer2-3x3-strided", 28),
            ("layer2-projection", 28),
            ("layer3-3x3-strided", 14),
            ("layer4-3x3-strided", 7),
            ("layer4-projection", 7),
        ] {
            assert!(outputs.contains(&(layer.to_owned(), size)), "{layer}");
        }
    }
}
