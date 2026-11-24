//! Instruction selection from lowered loop IR.

mod async_copy;
mod cast;
mod control;
mod expr;
mod fp8;
mod known;
mod math;
mod memory;
mod mma;
mod ops;
mod values;

use crate::ast::{B16, B64, F32, FunctionBuilder, PtxModule, Reg};
use known::Known;
use memory::Storage;
use ptxc_loop::{BufferId, LoweredKernel, VarId};
use ptxc_target::Target;
use std::collections::HashMap;
use thiserror::Error;
use values::Narrow;

/// A kernel codegen cannot emit.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CodegenError {
    /// An operation with no PTX selection yet.
    #[error("`{op}` cannot be emitted yet")]
    Unsupported {
        /// The operation's name.
        op: &'static str,
    },
    /// A variable read outside the statement binding it, or a buffer
    /// outside its allocation, which `ptxc_loop::verify` rejects.
    #[error("the kernel reads a variable or buffer outside its scope")]
    Unbound,
    /// An extent, offset or array too large for 64-bit index arithmetic.
    #[error("an extent or buffer is too large for 64-bit addressing")]
    TooLarge,
}

/// The PTX module holding `kernel`, for `target`, with its unrolled loops
/// expanded first.
pub fn generate(kernel: &LoweredKernel, target: &Target) -> Result<PtxModule, CodegenError> {
    let kernel = ptxc_loop::unrolled(kernel);
    let mut selector = Selector::new(&kernel);
    selector.parameters();
    selector.thread_indices();
    selector.statement(kernel.body())?;
    let function = selector.builder.finish();
    Ok(PtxModule { version: target.ptx_isa(), target: target.sm(), functions: vec![function] })
}

/// The state of selection: the function being built, what each kernel
/// variable and buffer is bound to, and what registers already hold.
struct Selector<'k> {
    builder: FunctionBuilder,
    kernel: &'k LoweredKernel,
    vars: HashMap<VarId, Reg<B64>>,
    storage: HashMap<BufferId, Storage>,
    known: Known,
    /// Each f32 register widened from a narrow float's memory form, with
    /// that form, which a store of the register reuses.
    narrow_forms: HashMap<Reg<F32>, (Reg<B16>, Narrow)>,
}

impl<'k> Selector<'k> {
    fn new(kernel: &'k LoweredKernel) -> Self {
        Self {
            builder: FunctionBuilder::new(kernel.name().as_str()),
            kernel,
            vars: HashMap::new(),
            storage: HashMap::new(),
            known: Known::default(),
            narrow_forms: HashMap::new(),
        }
    }
}
