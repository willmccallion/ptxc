//! `ptxc-bench gemm`: the GEMM sweep against cuBLAS (BENCHMARKING.md). Every
//! framework's result is checked before it is timed; a failed check is
//! reported and not timed.

mod data;
mod frameworks;
mod problems;

use crate::clocks::{ClockError, GpuIdentity, GpuState, ensure_idle};
use crate::cublas::{CublasError, GemmError, Handle, LtHandle, lt};
use crate::host::HostError;
use crate::problem::GemmProblem;
use crate::results::{Case, Framework, Results, ResultsError};
use crate::timing::{Series, time};
use data::{Inputs, check};
use frameworks::{Default, Operands, enqueue_candidate};
use problems::{GemmParams, id, params, problems};
use ptxc_cuda::{Context, CudaError, Device, DeviceBuffer, DeviceOrdinal, Driver, Stream};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Why the sweep stopped.
#[derive(Debug, Error)]
pub(crate) enum SweepError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Cublas(#[from] CublasError),
    #[error(transparent)]
    Gemm(#[from] GemmError),
    #[error(transparent)]
    Clock(#[from] ClockError),
    #[error(transparent)]
    Results(#[from] ResultsError),
    #[error(transparent)]
    Host(#[from] HostError),
    #[error("compiling ptxc's GEMM: {0}")]
    Compile(#[from] ptxc_compiler::CompileError),
    #[error("running ptxc's GEMM: {0}")]
    Runtime(#[from] ptxc_runtime::RuntimeError),
    #[error("building ptxc's GEMM: {0}")]
    Graph(#[from] ptxc_graph::NodeError),
    #[error(transparent)]
    Target(#[from] ptxc_target::UnsupportedSm),
    #[error(transparent)]
    Shape(#[from] ptxc_core::ShapeError),
    #[error(transparent)]
    Tensor(#[from] ptxc_core::TensorError),
    #[error(transparent)]
    Bytes(#[from] ptxc_core::BytesError),
    #[error("a problem is too large for this host")]
    TooLarge,
    #[error("ptxc's GEMM program has no output")]
    NoOutput,
    #[error("cuBLASLt proposed no algorithm")]
    NoAlgorithm,
    #[error("{} results failed their checks:\n{}", .0.len(), .0.join("\n"))]
    Failed(Vec<String>),
}

/// The sweep's arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct GemmArgs {
    /// The directory the results are written to, one file per framework.
    #[arg(long, default_value = "bench/results/scratch")]
    pub(crate) out: PathBuf,
    /// Only the problems whose names contain this, such as `f16/NN`.
    #[arg(long)]
    pub(crate) only: Option<String>,
}

/// Each framework's cases, and the checks that failed.
#[derive(Default)]
struct Collected {
    ptxc: Vec<Case<GemmParams>>,
    cublas: Vec<Case<GemmParams>>,
    best: Vec<Case<GemmParams>>,
    failed: Vec<String>,
}

/// Runs the sweep and writes each framework's results under `args.out`,
/// recording `repository`'s commit.
pub(crate) fn run(args: &GemmArgs, repository: &Path) -> Result<(), SweepError> {
    ensure_idle()?;
    let device = Driver::load()?.device(DeviceOrdinal::new(0))?;
    let stream = device.primary_context().stream()?;
    let selected: Vec<GemmProblem> = problems()
        .into_iter()
        .filter(|&problem| {
            args.only.as_ref().is_none_or(|only| id(problem).contains(only.as_str()))
        })
        .collect();
    let mut collected = Collected::default();
    for (index, &problem) in selected.iter().enumerate() {
        eprintln!("[{}/{}] {}", index + 1, selected.len(), id(problem));
        run_problem(problem, (&device, &stream), &mut collected)?;
    }
    let version = Handle::new(&stream)?.version()?;
    write((&args.out, repository), &version, collected)
}

/// Checks and times `problem` on every framework that runs it.
fn run_problem(
    problem: GemmProblem,
    (device, stream): (&Device, &Stream),
    collected: &mut Collected,
) -> Result<(), SweepError> {
    let context = device.primary_context();
    let on = Timing { context: &context, stream };
    let inputs = Inputs::new(problem)?;
    if let Some(executor) = frameworks::ptxc(problem, device, &inputs)? {
        let bound = executor.bind(&[inputs.a.clone(), inputs.b.clone()])?;
        bound.enqueue(stream)?;
        stream.synchronize()?;
        let output = bound.outputs()?;
        let data = output.first().ok_or(SweepError::NoOutput)?.data();
        let enqueue = || Ok(bound.enqueue(stream)?);
        if let Some(case) = on.timed((problem, None), check(problem, data), collected, enqueue)? {
            collected.ptxc.push(case);
        }
    }
    let mut operands = Operands::new(device, problem, &inputs)?;
    let workspace = context.alloc(lt::MAX_WORKSPACE)?;
    if let Some(mut default) = Default::new(stream, problem)? {
        let mut enqueue = |operands: &Operands| default.enqueue(problem, operands, &workspace);
        let checked = checked_run(problem, stream, &mut operands, &mut enqueue)?;
        let operands = &operands;
        if let Some(case) = on.timed((problem, None), checked, collected, || enqueue(operands))? {
            collected.cublas.push(case);
        }
    }
    if let Some(case) = best_of(problem, on, (&mut operands, &workspace), collected)? {
        collected.best.push(case);
    }
    Ok(())
}

/// The fastest of cuBLASLt's top algorithms that passes its check, if any.
fn best_of(
    problem: GemmProblem,
    on: Timing<'_>,
    (operands, workspace): (&mut Operands, &DeviceBuffer),
    collected: &mut Collected,
) -> Result<Option<Case<GemmParams>>, SweepError> {
    let mut handle = LtHandle::new(on.stream)?;
    let plan = lt::plan(&handle, problem)?;
    let mut passed = Vec::new();
    for (index, candidate) in plan.candidates.iter().enumerate() {
        let mut enqueue = |operands: &Operands| {
            enqueue_candidate(&mut handle, (&plan, candidate), problem, operands, workspace)
        };
        let checked = checked_run(problem, on.stream, operands, &mut enqueue)?;
        let algorithm = u8::try_from(index).map_err(|_| SweepError::TooLarge)?;
        let operands = &*operands;
        let timed =
            on.timed((problem, Some(algorithm)), checked, collected, || enqueue(operands))?;
        passed.extend(timed);
    }
    Ok(fastest(passed))
}

/// The case with the lowest median, the first of any that tie.
fn fastest(cases: Vec<Case<GemmParams>>) -> Option<Case<GemmParams>> {
    cases.into_iter().reduce(|best, case| if case.median_us < best.median_us { case } else { best })
}

/// C's check after one run of `enqueue` on poisoned C, so a run that
/// writes nothing fails.
fn checked_run(
    problem: GemmProblem,
    stream: &Stream,
    operands: &mut Operands,
    enqueue: &mut impl FnMut(&Operands) -> Result<(), SweepError>,
) -> Result<Result<(), String>, SweepError> {
    operands.poison_c()?;
    enqueue(operands)?;
    stream.synchronize()?;
    Ok(check(problem, &operands.c(problem)?))
}

/// The context and stream a series is timed on.
#[derive(Clone, Copy)]
struct Timing<'a> {
    context: &'a Context,
    stream: &'a Stream,
}

impl Timing<'_> {
    /// The case for `problem` and cuBLASLt's `algorithm`, timed with
    /// `enqueue` if `checked` passed; a failed check is recorded in
    /// `collected` and nothing is timed.
    fn timed(
        self,
        (problem, algorithm): (GemmProblem, Option<u8>),
        checked: Result<(), String>,
        collected: &mut Collected,
        mut enqueue: impl FnMut() -> Result<(), SweepError>,
    ) -> Result<Option<Case<GemmParams>>, SweepError> {
        if let Err(message) = checked {
            collected.failed.push(format!("{} {algorithm:?}: {message}", id(problem)));
            return Ok(None);
        }
        let before = GpuState::read()?;
        let series: Series = time(self.context, self.stream, &mut enqueue)?;
        let after = GpuState::read()?;
        let case =
            Case::new(id(problem), params(problem, algorithm), true, (before, after), &series)?;
        Ok(Some(case))
    }
}

/// Writes each framework's results under `out`, ptxc's at `repository`'s
/// commit and cuBLAS's at its `version` with the same commit, then fails
/// if any check failed.
fn write(
    (out, repository): (&Path, &Path),
    version: &str,
    collected: Collected,
) -> Result<(), SweepError> {
    let gpu = GpuIdentity::read()?;
    let ptxc = Framework::ptxc(repository)?;
    let cublas = |name: &str| Framework {
        name: name.to_owned(),
        version: version.to_owned(),
        ..ptxc.clone()
    };
    let files = [
        ("gemm-ptxc.json", ptxc.clone(), collected.ptxc),
        ("gemm-cublas.json", cublas("cublas"), collected.cublas),
        ("gemm-cublas-best-of-8.json", cublas("cublas-best-of-8"), collected.best),
    ];
    for (file, framework, cases) in files {
        Results::new("gemm", framework, gpu.clone(), cases).write(&out.join(file))?;
    }
    if collected.failed.is_empty() { Ok(()) } else { Err(SweepError::Failed(collected.failed)) }
}

#[cfg(test)]
mod tests;
