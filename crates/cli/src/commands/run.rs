//! `ptxc run`: executes a model on given inputs, on GPU 0.

use crate::args::RunArgs;
use crate::commands::compile::{compile_model, describe, extents};
use crate::ui;
use anyhow::Context;
use ptxc_core::{Evaluate, HostTensor};
use ptxc_cuda::{DeviceOrdinal, Driver};
use ptxc_program::{FormatError, Program};
use ptxc_runtime::{Executor, load};
use std::path::Path;
use std::process::ExitCode;

/// Runs `args.model` on its inputs and reports, and optionally writes, the
/// outputs.
pub(crate) fn run(args: &RunArgs) -> anyhow::Result<ExitCode> {
    ui::section("Run");
    ui::header("Model", &args.model.display().to_string());
    let program = program(&args.model, args)?;
    describe(&program);
    let inputs =
        args.inputs.iter().map(|path| read_tensor(path)).collect::<anyhow::Result<Vec<_>>>()?;
    let device = Driver::load()?.device(DeviceOrdinal::new(0))?;
    let names: Vec<String> = program.outputs().iter().map(|binding| binding.name.clone()).collect();
    let mut executor = Executor::eager(load(program, &device)?);
    let outputs = executor.evaluate(&inputs)?;
    for (name, output) in names.iter().zip(&outputs) {
        ui::stat(name, &format!("{} {}", output.dtype().name(), extents(output.shape())));
    }
    if let Some(dir) = &args.out {
        write_outputs(dir, &names, &outputs)?;
    }
    Ok(ExitCode::SUCCESS)
}

/// The program in `path`: a program file as it is, an ONNX model compiled.
fn program(path: &Path, args: &RunArgs) -> anyhow::Result<Program> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    match ptxc_program::load(&bytes) {
        Ok(program) => Ok(program),
        Err(FormatError::Magic) => compile_model(path, &args.dims),
        Err(error) => Err(error).with_context(|| format!("loading {}", path.display())),
    }
}

fn read_tensor(path: &Path) -> anyhow::Result<HostTensor> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ptxc_onnx::tensor_from_bytes(&bytes).with_context(|| format!("decoding {}", path.display()))
}

/// Each output as `output_<k>.pb` in `dir`, named as the program names it.
fn write_outputs(dir: &Path, names: &[String], outputs: &[HostTensor]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for (index, (name, output)) in names.iter().zip(outputs).enumerate() {
        let path = dir.join(format!("output_{index}.pb"));
        let bytes = ptxc_onnx::tensor_to_bytes(output, name)?;
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]

    use super::write_outputs;
    use ptxc_core::HostTensor;

    #[test]
    fn outputs_are_written_in_order_as_named_tensor_protos() {
        let dir = tempfile::tempdir().unwrap();
        let outputs =
            [HostTensor::vector(vec![1.5_f32].into()), HostTensor::vector(vec![7_i64, 8].into())];

        write_outputs(dir.path(), &["y".into(), "z".into()], &outputs).unwrap();

        let read = |n: usize| {
            ptxc_onnx::tensor_from_bytes(
                &std::fs::read(dir.path().join(format!("output_{n}.pb"))).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(read(0).data().to_le_bytes(), outputs[0].data().to_le_bytes());
        assert_eq!(read(1).data().to_le_bytes(), outputs[1].data().to_le_bytes());
    }
}
