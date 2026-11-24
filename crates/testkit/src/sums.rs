//! How a run of lowered stages rounds its sums.
//!
//! That decides how closely the GPU must match the loop interpreter on it
//! (D60): exactly, unless a stage multiplies on tensor cores, which round a
//! sum their own way (D58).

use crate::compare::Tolerance;
use crate::lowered::{LoweredError, run_lowered};
use ptxc_core::{DType, ElementCount, HostTensor};
use ptxc_graph::Graph;
use ptxc_loop::{LoweredKernel, RunError};

/// How the GPU rounds the sums of a run of stages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SumRounding {
    /// As the loop interpreter does.
    Interpreted,
    /// Some on tensor cores, in stages whose arguments' longest axis has
    /// `longest` elements, which bounds their reductions.
    TensorCores {
        /// The longest axis.
        longest: ElementCount,
    },
}

impl SumRounding {
    /// `self` after a stage running `kernel` on `arguments`.
    #[must_use]
    pub fn after(self, kernel: &LoweredKernel, arguments: &[HostTensor]) -> Self {
        if !kernel.multiplies_on_tensor_cores() {
            return self;
        }
        let stage = longest_axis(arguments);
        let longest = match self {
            Self::Interpreted => stage,
            Self::TensorCores { longest } => longest.max(stage),
        };
        Self::TensorCores { longest }
    }

    /// How closely a GPU output of `dtype` must match the interpreter's.
    pub fn tolerance(self, dtype: DType) -> Tolerance {
        match self {
            Self::Interpreted => Tolerance::Exact,
            Self::TensorCores { longest } => Tolerance::for_dtype(dtype).for_reduction(longest),
        }
    }
}

/// The most elements along any axis of `tensors`, at least one.
fn longest_axis(tensors: &[HostTensor]) -> ElementCount {
    let dims = tensors.iter().flat_map(|tensor| tensor.shape().dims());
    ElementCount::new(dims.map(|dim| dim.get()).max().unwrap_or(1).max(1))
}

/// `graph`'s outputs from the loop interpreter, its lowered stages run one
/// by one, and how the GPU rounds their sums.
pub fn interpreted(
    graph: Graph,
    inputs: &[HostTensor],
) -> Result<(Vec<HostTensor>, SumRounding), LoweredError<RunError>> {
    let mut rounding = SumRounding::Interpreted;
    let outputs = run_lowered(graph, inputs, |kernel, arguments| {
        rounding = rounding.after(kernel, arguments);
        ptxc_loop::run(kernel, arguments)
    })?;
    Ok((outputs, rounding))
}
