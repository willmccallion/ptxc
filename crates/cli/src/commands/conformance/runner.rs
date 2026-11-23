//! What runs each case: the reference interpreter, or the compiler and the
//! GPU runtime on a device opened once.

use crate::args::Backend;
use ptxc_compiler::Compilation;
use ptxc_core::{Evaluate, HostTensor};
use ptxc_cuda::{Device, DeviceOrdinal, Driver};
use ptxc_interp::Interpreter;
use ptxc_runtime::{Executor, load};
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::onnx_tests::Runnable;

/// A backend ready to run cases.
pub(super) enum Runner {
    Interp,
    Gpu(Device),
}

impl Runner {
    /// The runner for `backend`, opening GPU 0 for the GPU.
    pub(super) fn new(backend: Backend) -> anyhow::Result<Self> {
        Ok(match backend {
            Backend::Interp => Self::Interp,
            Backend::Gpu => Self::Gpu(Driver::load()?.device(DeviceOrdinal::new(0))?),
        })
    }

    /// `runnable`'s outputs, or why it did not run.
    pub(super) fn evaluate(&self, runnable: &Runnable) -> Result<Vec<HostTensor>, String> {
        match self {
            Self::Interp => Interpreter::new(&runnable.graph)
                .evaluate(&runnable.inputs)
                .map_err(|error| error.to_string()),
            Self::Gpu(device) => on_gpu(device, runnable),
        }
    }
}

/// `runnable` compiled for `sm_89` and run eagerly on `device`.
fn on_gpu(device: &Device, runnable: &Runnable) -> Result<Vec<HostTensor>, String> {
    let target = preset(SmVersion::SM_89).map_err(|error| error.to_string())?;
    let program = Compilation::from_graph(runnable.graph.clone(), target)
        .optimize()
        .and_then(|c| c.partition()?.lower()?.schedule()?.emit())
        .map_err(|error| format!("compiling: {error}"))?;
    let loaded = load(program, device).map_err(|error| format!("loading: {error}"))?;
    Executor::eager(loaded).evaluate(&runnable.inputs).map_err(|error| error.to_string())
}
