//! `ptxc-bench conv`: ptxc on ResNet-50's convolution layers at batch sizes
//! 1, 8 and 32 (BENCHMARKING.md), each compiled as an ONNX `Conv`. Every
//! result is checked before it is timed; a failed check is reported and
//! not timed.

mod data;
mod layers;

use super::Timing;
use crate::clocks::{ClockError, GpuIdentity, ensure_idle};
use crate::host::HostError;
use crate::results::{Case, Framework, Results, ResultsError};
use data::{Inputs, check};
use layers::{ConvParams, ConvProblem, ConvType, id, params, suite};
use ptxc_compiler::Compilation;
use ptxc_core::HostTensor;
use ptxc_cuda::{CudaError, Device, DeviceOrdinal, Driver, Stream};
use ptxc_graph::{
    ConvAttrs, Dilation, Graph, GraphBuilder, Groups, Layout, Op, Padding, Stride, TensorType,
    ValueName, WindowAxis,
};
use ptxc_runtime::{Eager, Executor, load};
use ptxc_target::{SmVersion, preset};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Why the suite stopped.
#[derive(Debug, Error)]
pub(crate) enum ConvError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Clock(#[from] ClockError),
    #[error(transparent)]
    Results(#[from] ResultsError),
    #[error(transparent)]
    Host(#[from] HostError),
    #[error("reading bench/conv.json: {0}")]
    Definition(#[from] serde_json::Error),
    #[error("compiling ptxc's convolution: {0}")]
    Compile(#[from] ptxc_compiler::CompileError),
    #[error("running ptxc's convolution: {0}")]
    Runtime(#[from] ptxc_runtime::RuntimeError),
    #[error("building ptxc's convolution: {0}")]
    Graph(#[from] ptxc_graph::NodeError),
    #[error(transparent)]
    Target(#[from] ptxc_target::UnsupportedSm),
    #[error(transparent)]
    Shape(#[from] ptxc_core::ShapeError),
    #[error(transparent)]
    Tensor(#[from] ptxc_core::TensorError),
    #[error("a layer has a zero stride")]
    Stride,
    #[error("ptxc's convolution program has no output")]
    NoOutput,
    #[error("{} results failed their checks:\n{}", .0.len(), .0.join("\n"))]
    Failed(Vec<String>),
}

/// The suite's arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct ConvArgs {
    /// The directory the results are written to.
    #[arg(long, default_value = "bench/results/scratch")]
    pub(crate) out: PathBuf,
    /// Only the problems whose names contain this, such as `layer1` or
    /// `b32`.
    #[arg(long)]
    pub(crate) only: Option<String>,
}

/// Runs the suite and writes ptxc's results under `args.out`, recording
/// `repository`'s commit.
pub(crate) fn run(args: &ConvArgs, repository: &Path) -> Result<(), ConvError> {
    ensure_idle()?;
    let device = Driver::load()?.device(DeviceOrdinal::new(0))?;
    let stream = device.primary_context().stream()?;
    let suite = suite()?;
    let selected: Vec<&ConvProblem> = suite
        .problems
        .iter()
        .filter(|problem| {
            args.only.as_ref().is_none_or(|only| id(problem, suite.dtype).contains(only.as_str()))
        })
        .collect();
    let (mut cases, mut failed) = (Vec::new(), Vec::new());
    for (index, problem) in selected.iter().enumerate() {
        eprintln!("[{}/{}] {}", index + 1, selected.len(), id(problem, suite.dtype));
        match run_problem(problem, suite.dtype, (&device, &stream))? {
            Ok(case) => cases.push(case),
            Err(message) => failed.push(format!("{}: {message}", id(problem, suite.dtype))),
        }
    }
    let results = Results::new("conv", Framework::ptxc(repository)?, GpuIdentity::read()?, cases);
    results.write(&args.out.join("conv-ptxc.json"))?;
    if failed.is_empty() { Ok(()) } else { Err(ConvError::Failed(failed)) }
}

/// `problem` run once and checked, then timed; or how its check failed.
fn run_problem(
    problem: &ConvProblem,
    dtype: ConvType,
    (device, stream): (&Device, &Stream),
) -> Result<Result<Case<ConvParams>, String>, ConvError> {
    let inputs = Inputs::new(problem, dtype)?;
    let executor = compile(problem, &inputs, device)?;
    let bound = executor.bind(&[inputs.x, inputs.w])?;
    bound.enqueue(stream)?;
    stream.synchronize()?;
    let outputs = bound.outputs()?;
    let output = outputs.first().ok_or(ConvError::NoOutput)?;
    if let Err(message) = check(problem, dtype, output.data()) {
        return Ok(Err(message));
    }
    let context = device.primary_context();
    let on = Timing { context: &context, stream };
    let case = (id(problem, dtype), params(problem, dtype));
    Ok(Ok(on.case(case, || Ok::<(), ConvError>(bound.enqueue(stream)?))?))
}

/// ptxc's program for `problem`, compiled for the device's target.
fn compile(
    problem: &ConvProblem,
    inputs: &Inputs,
    device: &Device,
) -> Result<Executor<Eager>, ConvError> {
    let graph = conv_graph(problem, inputs)?;
    let target = preset(SmVersion::SM_89)?;
    let program = Compilation::from_graph(graph, target)
        .optimize()?
        .partition()?
        .lower()?
        .schedule()?
        .emit()?;
    Ok(Executor::eager(load(program, device)?))
}

/// `y0 = Conv(x0, x1)` over two spatial axes, each with the layer's
/// stride and padding.
fn conv_graph(problem: &ConvProblem, inputs: &Inputs) -> Result<Graph, ConvError> {
    let layer = &problem.layer;
    let stride = Stride::new(u64::from(layer.stride)).ok_or(ConvError::Stride)?;
    let pad = Padding::new(u64::from(layer.pad));
    let one = Dilation::new(1).ok_or(ConvError::Stride)?;
    let axis = WindowAxis { stride, dilation: one, pad_begin: pad, pad_end: pad };
    let groups = Groups::new(1).ok_or(ConvError::Stride)?;
    let attrs = ConvAttrs { axes: vec![axis, axis], groups };
    let mut builder = GraphBuilder::new();
    let mut input = |name: &str, tensor: &HostTensor| {
        let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), Layout::RowMajor);
        let value = builder.add_input(ty);
        builder.set_name(value, ValueName::new(name.to_owned()));
        value
    };
    let (x, w) = (input("x0", &inputs.x), input("x1", &inputs.w));
    let outputs = builder.add_node(Op::Conv(attrs), vec![x, w])?;
    for &output in &outputs {
        builder.set_name(output, ValueName::new("y0".to_owned()));
    }
    Ok(builder.finish(outputs)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::conv_graph;
    use super::data::Inputs;
    use super::layers::{ConvType, suite};
    use ptxc_core::DType;
    use ptxc_graph::Op;

    #[test]
    fn a_layer_is_one_onnx_conv_with_its_stride_and_padding() {
        let suite = suite().unwrap();
        let stem = suite.problems.iter().find(|problem| problem.layer.name == "stem").unwrap();
        let inputs = Inputs::new(stem, ConvType::F16).unwrap();

        let graph = conv_graph(stem, &inputs).unwrap();

        let node = graph.nodes().next().unwrap().1;
        let Op::Conv(attrs) = node.op() else { panic!("{:?}", node.op()) };
        let axis = attrs.axes[0];
        assert_eq!(attrs.axes.len(), 2);
        assert_eq!((axis.stride.get(), axis.pad_begin.get(), axis.pad_end.get()), (2, 3, 3));
        let output = graph.value(*graph.outputs().iter().next().unwrap()).ty();
        let dims: Vec<u64> = output.shape().dims().iter().map(|dim| dim.get()).collect();
        assert_eq!((output.dtype(), dims), (DType::F16, vec![stem.batch.into(), 64, 112, 112]));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn one_problem_is_checked_timed_and_written() {
        use super::{ConvArgs, run};
        use crate::suites::testing::repository;

        let _gpu = crate::timing::exclusive_gpu();
        let dir = tempfile::tempdir().unwrap();
        repository(dir.path());
        let out = dir.path().join("results");
        let only = Some("conv/f16/layer4-projection/b1".to_owned());

        run(&ConvArgs { out: out.clone(), only }, dir.path()).unwrap();

        let text = std::fs::read_to_string(out.join("conv-ptxc.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!((&json["suite"], &json["framework"]["name"]), (&"conv".into(), &"ptxc".into()));
        let cases = json["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0]["id"], "conv/f16/layer4-projection/b1");
        assert_eq!(cases[0]["verified"], true);
        assert_eq!(cases[0]["params"]["features"], 2048);
        assert!(cases[0]["median_us"].as_f64().unwrap() > 0.0);
    }
}
