//! `ptxc compile`: ONNX model to Program.

use crate::args::CompileArgs;
use crate::ui;
use anyhow::Context;
use ptxc_compiler::{Compilation, arena_use};
use ptxc_core::{Dim, Shape};
use ptxc_onnx::ImportOptions;
use ptxc_program::Program;
use ptxc_target::{SmVersion, preset};
use std::path::Path;
use std::process::ExitCode;

/// Compiles `args.model` and writes the program file.
pub(crate) fn run(args: &CompileArgs) -> anyhow::Result<ExitCode> {
    ui::section("Compile");
    ui::header("Model", &args.model.display().to_string());
    let program = compile_model(&args.model, &args.dims)?;
    let bytes = ptxc_program::save(&program).context("encoding the program")?;
    std::fs::write(&args.out, &bytes).with_context(|| format!("writing {}", args.out.display()))?;
    describe(&program)?;
    ui::stat("Written", &args.out.display().to_string());
    Ok(ExitCode::SUCCESS)
}

/// The program for the ONNX model at `path`, its symbolic dimensions sized
/// by `dims`, for `sm_89`.
pub(crate) fn compile_model(path: &Path, dims: &[(String, Dim)]) -> anyhow::Result<Program> {
    let onnx = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let options = ImportOptions {
        symbolic_dims: dims.iter().cloned().collect(),
        model_dir: path.parent().map(Path::to_path_buf),
        ..ImportOptions::default()
    };
    let target = preset(SmVersion::SM_89)?;
    let compiled = Compilation::import(&onnx, &options, target)
        .and_then(|imported| imported.optimize()?.partition()?.lower()?.schedule()?.emit())
        .with_context(|| format!("compiling {}", path.display()))?;
    Ok(compiled)
}

/// The program's shape: kernels, memory and bindings.
pub(crate) fn describe(program: &Program) -> anyhow::Result<()> {
    ui::stat("Kernels", &program.kernels().len().to_string());
    let arena = arena_use(program).context("measuring the arena")?;
    ui::stat("Arena", &format!("{} bytes", arena.size.get()));
    ui::stat("Arena lower bound", &format!("{} bytes", arena.lower_bound.get()));
    ui::stat("Constants", &format!("{} bytes", program.constants().len()));
    for binding in program.inputs() {
        ui::stat("Input", &format!("{} {}", binding.name, extents(&binding.shape)));
    }
    for binding in program.outputs() {
        ui::stat("Output", &format!("{} {}", binding.name, extents(&binding.shape)));
    }
    Ok(())
}

/// A shape as its extents, such as `[2, 4]`.
pub(crate) fn extents(shape: &Shape) -> String {
    let extents: Vec<String> = shape.dims().iter().map(|dim| dim.get().to_string()).collect();
    format!("[{}]", extents.join(", "))
}
