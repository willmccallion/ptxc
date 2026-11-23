//! Each framework's GEMM, run once to be checked and then timed: ptxc's
//! program, compiled as an ONNX `Gemm`; cuBLAS's default; and each of
//! cuBLASLt's top algorithms.

use super::SweepError;
use super::data::Inputs;
use crate::cublas::{Candidate, Handle, LtHandle, Plan, gemm, lt};
use crate::problem::{GemmProblem, GemmType, Layout};
use ptxc_compiler::Compilation;
use ptxc_core::{ByteSize, HostTensor};
use ptxc_cuda::{Device, DeviceBuffer, Stream};
use ptxc_graph::F32Attr;
use ptxc_graph::{
    GemmAttrs, Graph, GraphBuilder, Layout as ValueLayout, Op, Orientation, TensorType, ValueName,
};
use ptxc_runtime::{Eager, Executor, load};
use ptxc_target::{SmVersion, preset};

/// ptxc's program for `problem`, or `None` for types its `Gemm` does not
/// take as the sweep defines them: E4M3 to f32 and int8 to i32 come with
/// the quantized matmul of #107.
pub(super) fn ptxc(
    problem: GemmProblem,
    device: &Device,
    inputs: &Inputs,
) -> Result<Option<Executor<Eager>>, SweepError> {
    if matches!(problem.dtype, GemmType::E4m3 | GemmType::I8) {
        return Ok(None);
    }
    let graph = gemm_graph(problem.layout, inputs)?;
    let target = preset(SmVersion::SM_89)?;
    let program = Compilation::from_graph(graph, target)
        .optimize()?
        .partition()?
        .lower()?
        .schedule()?
        .emit()?;
    Ok(Some(Executor::eager(load(program, device)?)))
}

/// `y0 = Gemm(x0, x1)`, each input transposed as `layout` stores it.
fn gemm_graph(layout: Layout, inputs: &Inputs) -> Result<Graph, SweepError> {
    let orientation =
        |transposed| if transposed { Orientation::Transposed } else { Orientation::AsIs };
    let attrs = GemmAttrs {
        a: orientation(layout.a_transposed()),
        b: orientation(layout.b_transposed()),
        alpha: F32Attr::new(1.0),
        beta: F32Attr::new(0.0),
    };
    let mut builder = GraphBuilder::new();
    let mut input = |name: &str, tensor: &HostTensor| {
        let ty = TensorType::new(tensor.dtype(), tensor.shape().clone(), ValueLayout::RowMajor);
        let value = builder.add_input(ty);
        builder.set_name(value, ValueName::new(name.to_owned()));
        value
    };
    let (a, b) = (input("x0", &inputs.a), input("x1", &inputs.b));
    let outputs = builder.add_node(Op::Gemm(attrs), vec![a, b])?;
    for &output in &outputs {
        builder.set_name(output, ValueName::new("y0".to_owned()));
    }
    Ok(builder.finish(outputs)?)
}

/// The byte C is filled with before a checked run: all ones, which is NaN
/// in every float type and -1 in i32.
const POISON: u8 = 0xFF;

/// A and B on the device, and room for C.
pub(super) struct Operands {
    pub(super) a: DeviceBuffer,
    pub(super) b: DeviceBuffer,
    pub(super) c: DeviceBuffer,
}

impl Operands {
    pub(super) fn new(
        device: &Device,
        problem: GemmProblem,
        inputs: &Inputs,
    ) -> Result<Self, SweepError> {
        let context = device.primary_context();
        let upload = |tensor: &HostTensor| -> Result<DeviceBuffer, SweepError> {
            let bytes = tensor.data().to_le_bytes();
            let mut buffer = context.alloc(ByteSize::new(
                u64::try_from(bytes.len()).map_err(|_| SweepError::TooLarge)?,
            ))?;
            buffer.copy_from_host(&bytes)?;
            Ok(buffer)
        };
        let [_, _, c_bytes] = problem.sizes().ok_or(SweepError::TooLarge)?;
        Ok(Self { a: upload(&inputs.a)?, b: upload(&inputs.b)?, c: context.alloc(c_bytes)? })
    }

    /// Fills C with `POISON`, so a run that writes nothing fails its check
    /// rather than passing on what an earlier run left.
    pub(super) fn poison_c(&mut self) -> Result<(), SweepError> {
        let size = usize::try_from(self.c.size().get()).map_err(|_| SweepError::TooLarge)?;
        Ok(self.c.copy_from_host(&vec![POISON; size])?)
    }

    /// C, read back.
    pub(super) fn c(&self, problem: GemmProblem) -> Result<ptxc_core::TensorData, SweepError> {
        let mut bytes =
            vec![0; usize::try_from(self.c.size().get()).map_err(|_| SweepError::TooLarge)?];
        self.c.copy_to_host(&mut bytes)?;
        Ok(ptxc_core::TensorData::from_le_bytes(problem.dtype.output(), &bytes)?)
    }
}

/// cuBLAS's default for `problem`: `GemmEx`'s heuristic, or cuBLASLt's
/// first algorithm for the types only it runs; `None` if neither can.
pub(super) enum Default<'s> {
    GemmEx(Handle<'s>),
    Lt(LtHandle<'s>, Plan),
}

impl<'s> Default<'s> {
    pub(super) fn new(
        stream: &'s Stream,
        problem: GemmProblem,
    ) -> Result<Option<Self>, SweepError> {
        if matches!(problem.dtype, GemmType::E4m3 | GemmType::I8) {
            let handle = LtHandle::new(stream)?;
            let plan = lt::plan(&handle, problem)?;
            return Ok((!plan.candidates.is_empty()).then_some(Self::Lt(handle, plan)));
        }
        Ok(Some(Self::GemmEx(Handle::new(stream)?)))
    }

    /// Queues the default on its stream.
    pub(super) fn enqueue(
        &mut self,
        problem: GemmProblem,
        operands: &Operands,
        workspace: &DeviceBuffer,
    ) -> Result<(), SweepError> {
        let Operands { a, b, c } = operands;
        match self {
            Self::GemmEx(handle) => gemm::gemm(handle, problem, [a, b], c)?,
            Self::Lt(handle, plan) => {
                let first = *plan.candidates.first().ok_or(SweepError::NoAlgorithm)?;
                lt::run(handle, (plan, &first), problem, [a, b, c], workspace)?;
            }
        }
        Ok(())
    }
}

/// Queues `candidate` of `plan` on `handle`'s stream.
pub(super) fn enqueue_candidate(
    handle: &mut LtHandle<'_>,
    (plan, candidate): (&Plan, &Candidate),
    problem: GemmProblem,
    operands: &Operands,
    workspace: &DeviceBuffer,
) -> Result<(), SweepError> {
    let Operands { a, b, c } = operands;
    Ok(lt::run(handle, (plan, candidate), problem, [a, b, c], workspace)?)
}
