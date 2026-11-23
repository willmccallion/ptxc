//! The sweep's problems: BENCHMARKING.md's shapes, in every type and
//! layout, and the names its results use.

use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};
use serde::{Deserialize, Serialize};

/// Square problems.
const SQUARES: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];

/// M for the transformer shapes: batch 1, 8 and 32 at sequence length 128.
const TRANSFORMER_M: [u32; 3] = [128, 1024, 4096];

/// N and K of BERT-base's and GPT-2's projections.
const TRANSFORMER_NK: [(u32, u32); 4] = [(768, 768), (3072, 768), (768, 3072), (2304, 768)];

/// Decoding's skinny M, with the transformer's N and K.
const SKINNY_M: [u32; 3] = [1, 4, 16];

/// Squares no power-of-two tile divides.
const UNEVEN: [u32; 2] = [1000, 3000];

const TYPES: [GemmType; 5] =
    [GemmType::F16, GemmType::Bf16, GemmType::E4m3, GemmType::I8, GemmType::F32];
const LAYOUTS: [Layout; 3] = [Layout::Nn, Layout::Nt, Layout::Tn];

/// A case's parameters, as its results record them.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct GemmParams {
    pub(crate) m: u32,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) dtype: String,
    pub(crate) layout: String,
    /// For the best of cuBLAS's algorithms: which of them, from 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) algorithm: Option<u8>,
}

/// Every problem the sweep runs.
pub(super) fn problems() -> Vec<GemmProblem> {
    let squares =
        SQUARES.iter().chain(&UNEVEN).map(|&side| GemmShape { m: side, n: side, k: side });
    let transformer = TRANSFORMER_M
        .iter()
        .chain(&SKINNY_M)
        .flat_map(|&m| TRANSFORMER_NK.iter().map(move |&(n, k)| GemmShape { m, n, k }));
    let shapes: Vec<GemmShape> = squares.chain(transformer).collect();
    TYPES
        .iter()
        .flat_map(|&dtype| LAYOUTS.iter().map(move |&layout| (dtype, layout)))
        .flat_map(|(dtype, layout)| {
            shapes.iter().map(move |&shape| GemmProblem { shape, dtype, layout })
        })
        .collect()
}

/// The problem's name, `gemm/<type>/<layout>/<m>x<n>x<k>`.
pub(super) fn id(problem: GemmProblem) -> String {
    let GemmShape { m, n, k } = problem.shape;
    format!("gemm/{}/{}/{m}x{n}x{k}", type_name(problem.dtype), layout_name(problem.layout))
}

/// The problem's parameters, for the given algorithm of cuBLAS's.
pub(super) fn params(problem: GemmProblem, algorithm: Option<u8>) -> GemmParams {
    let GemmShape { m, n, k } = problem.shape;
    let dtype = type_name(problem.dtype).to_owned();
    GemmParams { m, n, k, dtype, layout: layout_name(problem.layout).to_owned(), algorithm }
}

const fn type_name(dtype: GemmType) -> &'static str {
    match dtype {
        GemmType::F32 => "f32",
        GemmType::F16 => "f16",
        GemmType::Bf16 => "bf16",
        GemmType::E4m3 => "e4m3",
        GemmType::I8 => "i8",
    }
}

const fn layout_name(layout: Layout) -> &'static str {
    match layout {
        Layout::Nn => "NN",
        Layout::Nt => "NT",
        Layout::Tn => "TN",
    }
}

#[cfg(test)]
mod tests {
    use super::{id, problems};
    use std::collections::HashSet;

    #[test]
    fn the_sweep_covers_each_shape_in_every_type_and_layout_once() {
        let all = problems();

        let ids: HashSet<String> = all.iter().map(|&problem| id(problem)).collect();

        assert_eq!(all.len(), (6 + 2 + 3 * 4 + 3 * 4) * 5 * 3);
        assert_eq!(ids.len(), all.len());
        assert!(ids.contains("gemm/f16/NN/4096x4096x4096"));
        assert!(ids.contains("gemm/e4m3/NT/1x3072x768"));
        assert!(ids.contains("gemm/i8/TN/1000x1000x1000"));
    }
}
