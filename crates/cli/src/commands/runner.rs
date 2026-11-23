//! What runs a graph: the reference interpreter, or the compiler and the
//! GPU runtime on a device opened once.

use crate::args::Backend;
use ptxc_compiler::Compilation;
use ptxc_core::{Evaluate, HostTensor};
use ptxc_cuda::{Device, DeviceOrdinal, Driver};
use ptxc_graph::Graph;
use ptxc_interp::Interpreter;
use ptxc_runtime::{Executor, load};
use ptxc_target::{SmVersion, preset};

/// A backend ready to run cases.
pub(crate) enum Runner {
    Interp,
    Gpu(Device),
}

impl Runner {
    /// The runner for `backend`, opening GPU 0 for the GPU.
    pub(crate) fn new(backend: Backend) -> anyhow::Result<Self> {
        Ok(match backend {
            Backend::Interp => Self::Interp,
            Backend::Gpu => Self::Gpu(Driver::load()?.device(DeviceOrdinal::new(0))?),
        })
    }

    /// `graph`'s outputs for `inputs`, or why it did not run.
    pub(crate) fn evaluate(
        &self,
        graph: &Graph,
        inputs: &[HostTensor],
    ) -> Result<Vec<HostTensor>, String> {
        match self {
            Self::Interp => {
                Interpreter::new(graph).evaluate(inputs).map_err(|error| error.to_string())
            }
            Self::Gpu(device) => on_gpu(device, graph, inputs),
        }
    }
}

/// `graph` compiled for `sm_89` and run eagerly on `device`.
fn on_gpu(
    device: &Device,
    graph: &Graph,
    inputs: &[HostTensor],
) -> Result<Vec<HostTensor>, String> {
    let target = preset(SmVersion::SM_89).map_err(|error| error.to_string())?;
    let program = Compilation::from_graph(graph.clone(), target)
        .optimize()
        .and_then(|c| c.partition()?.lower()?.schedule()?.emit())
        .map_err(|error| format!("compiling: {error}"))?;
    let loaded = load(program, device).map_err(|error| format!("loading: {error}"))?;
    Executor::eager(loaded).evaluate(inputs).map_err(|error| error.to_string())
}
