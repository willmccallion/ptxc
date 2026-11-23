//! `ptxc fuzz-export`: writes fuzz cases the reference can check as ONNX
//! models and input sets, for `ptxc-baselines fuzz-reference` to run on
//! ONNX Runtime (D54).

use crate::args::FuzzExportArgs;
use crate::ui;
use anyhow::Context;
use ptxc_core::DType;
use ptxc_graph::{Graph, Op, ReduceOp};
use ptxc_onnx::{model_to_bytes, tensor_to_bytes};
use ptxc_testkit::generate::{Case, Generator, Seed};
use std::path::Path;
use std::process::ExitCode;

/// Each case's model, in its directory.
pub(super) const MODEL: &str = "model.onnx";

/// Each case's input set, `input_k.pb` per input, in its directory.
pub(super) const INPUTS: &str = "inputs";

/// Writes `args.cases` cases, the first comparable seeds from `args.seed`,
/// each to `seed-<n>/` under `args.out`, which must be new or empty.
pub(crate) fn run(args: &FuzzExportArgs) -> anyhow::Result<ExitCode> {
    ui::section("Fuzz export");
    ensure_empty(&args.out)?;
    let generator = Generator::default();
    let mut seed = args.seed;
    let mut written = 0;
    while written < args.cases {
        let recipe = generator.generate(Seed(seed));
        if recipe.compares_with_reference() {
            let case = recipe.build().context("a generated recipe always builds")?;
            if has_kernels(&case) && is_defined(&case.graph) {
                write_case(&args.out.join(format!("seed-{seed}")), &case)?;
                written += 1;
            }
        }
        seed = seed.checked_add(1).context("the seeds run past u64::MAX")?;
    }
    ui::stat("written", &format!("{written} to {}", args.out.display()));
    ui::stat("seeds", &format!("{} to {}", args.seed, seed - 1));
    Ok(ExitCode::SUCCESS)
}

/// Whether ONNX Runtime's CPU provider can run the case: it has no bf16
/// kernels. A case the reference can check is all connectors, which keep
/// their inputs' types, so the inputs' types are every value's.
fn has_kernels(case: &Case) -> bool {
    case.inputs.iter().all(|input| input.dtype() != DType::Bf16)
}

/// Whether ONNX defines every result of `graph`. It leaves the mean of no
/// elements undefined: ptxc computes NaN, as numpy does, and ONNX Runtime
/// computes 0.
fn is_defined(graph: &Graph) -> bool {
    graph.nodes().all(|(_, node)| match node.op() {
        Op::Reduce(attrs) if attrs.op == ReduceOp::Mean => {
            let Some(&input) = node.operands().iter().next() else { return false };
            let dims = graph.value(input).ty().shape().dims();
            attrs.axes.iter().all(|axis| dims.get(axis.index()).is_some_and(|dim| dim.get() > 0))
        }
        _ => true,
    })
}

/// Fails unless `directory` is missing or empty, so old cases never mix
/// with new ones.
fn ensure_empty(directory: &Path) -> anyhow::Result<()> {
    let Ok(mut entries) = std::fs::read_dir(directory) else { return Ok(()) };
    if entries.next().is_some() {
        anyhow::bail!("{} is not empty; remove it or choose another --out", directory.display());
    }
    Ok(())
}

/// `case` as `model.onnx` and `inputs/input_k.pb` in `directory`.
pub(super) fn write_case(directory: &Path, case: &Case) -> anyhow::Result<()> {
    let inputs = directory.join(INPUTS);
    std::fs::create_dir_all(&inputs).with_context(|| format!("creating {}", inputs.display()))?;
    let model = directory.join(MODEL);
    std::fs::write(&model, model_to_bytes(&case.graph)?)
        .with_context(|| format!("writing {}", model.display()))?;
    for (k, tensor) in case.inputs.iter().enumerate() {
        let path = inputs.join(format!("input_{k}.pb"));
        std::fs::write(&path, tensor_to_bytes(tensor, &format!("x{k}"))?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{INPUTS, MODEL, ensure_empty, has_kernels, is_defined, write_case};
    use ptxc_core::float::Bf16;
    use ptxc_core::{Axis, Dim, HostTensor, Shape, TensorData};
    use ptxc_graph::{KeepDims, Op, ReduceAttrs, ReduceOp};
    use ptxc_onnx::{ImportOptions, import_model};
    use ptxc_testkit::generate::{Case, Input, Node, Operand, Recipe, Source};
    use ptxc_testkit::reference::inputs;

    fn case(op: Op, input: HostTensor) -> Case {
        let recipe = Recipe {
            inputs: vec![Input { tensor: input, source: Source::Moderate }],
            nodes: vec![Node { op, operands: vec![Operand::Input(0)] }],
        };
        recipe.build().unwrap()
    }

    fn reduce(op: ReduceOp) -> Op {
        Op::Reduce(ReduceAttrs { op, axes: vec![Axis::new(0)], keep_dims: KeepDims::Keep })
    }

    fn empty() -> HostTensor {
        let shape = Shape::new(vec![Dim::new(0), Dim::new(2)]).unwrap();
        HostTensor::new(shape, TensorData::F32(Vec::new())).unwrap()
    }

    fn pair() -> HostTensor {
        HostTensor::vector(vec![1.0_f32, 2.0].into())
    }

    #[test]
    fn a_mean_of_no_elements_is_undefined_and_other_reductions_are_not() {
        assert!(!is_defined(&case(reduce(ReduceOp::Mean), empty()).graph));
        assert!(is_defined(&case(reduce(ReduceOp::Sum), empty()).graph));
        assert!(is_defined(&case(reduce(ReduceOp::Mean), pair()).graph));
    }

    #[test]
    fn onnx_runtime_has_no_bf16_kernels() {
        let bf16 = HostTensor::vector(vec![Bf16::from_f64(1.0)].into());

        assert!(!has_kernels(&case(reduce(ReduceOp::Sum), bf16)));
        assert!(has_kernels(&case(reduce(ReduceOp::Sum), pair())));
    }

    #[test]
    fn a_written_case_reads_back_as_its_graph_and_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let written = case(reduce(ReduceOp::Sum), pair());

        write_case(dir.path(), &written).unwrap();

        let bytes = std::fs::read(dir.path().join(MODEL)).unwrap();
        let graph = import_model(&bytes, &ImportOptions::default()).unwrap();
        let read = inputs(&dir.path().join(INPUTS)).unwrap();
        assert_eq!(ptxc_graph::text::print(&graph), ptxc_graph::text::print(&written.graph));
        assert_eq!(format!("{read:?}"), format!("{:?}", written.inputs));
    }

    #[test]
    fn only_a_missing_or_empty_directory_is_written_to() {
        let dir = tempfile::tempdir().unwrap();
        let full = dir.path().join("full");
        std::fs::create_dir_all(full.join("seed-0")).unwrap();

        assert!(ensure_empty(&dir.path().join("missing")).is_ok());
        assert!(ensure_empty(&full).is_err());
        assert!(ensure_empty(&full.join("seed-0")).is_ok());
    }
}
