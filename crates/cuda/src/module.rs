//! Loading PTX modules and looking up kernels.

use crate::context::Context;
use crate::driver::driver_error;
use crate::error::CudaError;
use crate::sys::{
    CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CUDA_ERROR_INVALID_VALUE,
    CUDA_SUCCESS, CuFunction, CuModule,
};
use ptxc_core::{ByteSize, KernelName};
use std::ffi::{CString, c_void};
use std::sync::Arc;

/// The room given to the JIT for its error log.
const JIT_LOG_BYTES: usize = 16_384;

/// A loaded PTX module. Its kernels keep it loaded.
#[derive(Debug)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

#[derive(Debug)]
struct ModuleInner {
    context: Context,
    raw: CuModule,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // A failed unload cannot be reported from `drop`; the module stays
        // loaded until its context is released.
        let _failed = self.context.call("cuModuleUnload", |api| {
            // SAFETY: the module was loaded in this context and is unloaded
            // exactly once, here, after every kernel from it is dropped.
            unsafe { (api.module_unload)(self.raw) }
        });
    }
}

impl Module {
    /// JIT-compiles `ptx`; a failure carries the JIT's error log.
    pub(crate) fn load(context: &Context, ptx: &str) -> Result<Self, CudaError> {
        let image = CString::new(ptx).map_err(|_| CudaError::InteriorNul("PTX"))?;
        let mut log = vec![0_u8; JIT_LOG_BYTES];
        let mut options = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut values: [*mut c_void; 2] =
            [log.as_mut_ptr().cast(), std::ptr::without_provenance_mut(JIT_LOG_BYTES)];
        let mut raw = CuModule::null();
        context.make_current()?;
        // SAFETY: `image` is NUL-terminated PTX; each option is paired with
        // its value, the log buffer is `JIT_LOG_BYTES` long and outlives the
        // call, and `raw` is a valid place for the module.
        let status = unsafe {
            (context.api().module_load_data_ex)(
                &raw mut raw,
                image.as_ptr().cast(),
                2,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != CUDA_SUCCESS {
            let detail = jit_log(&log);
            return Err(driver_error(context.api(), "cuModuleLoadDataEx", status, detail));
        }
        Ok(Self { inner: Arc::new(ModuleInner { context: context.clone(), raw }) })
    }

    /// The kernel whose `.entry` is `name`, with its parameter sizes.
    pub fn function(&self, name: &KernelName) -> Result<Function, CudaError> {
        let entry =
            CString::new(name.as_str()).map_err(|_| CudaError::InteriorNul("kernel name"))?;
        let mut raw = CuFunction::null();
        self.inner.context.call("cuModuleGetFunction", |api| {
            // SAFETY: the module is loaded, `entry` is NUL-terminated, and
            // `raw` is a valid place for the function.
            unsafe { (api.module_get_function)(&raw mut raw, self.inner.raw, entry.as_ptr()) }
        })?;
        let params = parameter_sizes(&self.inner.context, raw)?;
        Ok(Function { module: Arc::clone(&self.inner), raw, name: name.clone(), params })
    }
}

/// The JIT's log up to its terminating NUL, if it wrote one.
fn jit_log(log: &[u8]) -> Option<String> {
    let end = log.iter().position(|&byte| byte == 0).unwrap_or(log.len());
    let text = String::from_utf8_lossy(&log[..end]).trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Each parameter's size, asked of the driver until it reports no more.
fn parameter_sizes(context: &Context, function: CuFunction) -> Result<Vec<ByteSize>, CudaError> {
    context.make_current()?;
    let mut sizes = Vec::new();
    loop {
        let (mut offset, mut size) = (0_usize, 0_usize);
        // SAFETY: `function` is a kernel of a loaded module, and `offset`
        // and `size` are valid places for the driver to write.
        let status = unsafe {
            (context.api().func_get_param_info)(
                function,
                sizes.len(),
                &raw mut offset,
                &raw mut size,
            )
        };
        match status {
            CUDA_SUCCESS => sizes.push(ByteSize::new(size as u64)),
            CUDA_ERROR_INVALID_VALUE => return Ok(sizes),
            failed => return Err(driver_error(context.api(), "cuFuncGetParamInfo", failed, None)),
        }
    }
}

/// A kernel in a loaded module.
#[derive(Debug)]
pub struct Function {
    module: Arc<ModuleInner>,
    raw: CuFunction,
    name: KernelName,
    params: Vec<ByteSize>,
}

impl Function {
    /// The kernel's name.
    pub const fn name(&self) -> &KernelName {
        &self.name
    }

    /// Each parameter's size, as the PTX declares it.
    pub fn parameter_sizes(&self) -> &[ByteSize] {
        &self.params
    }

    pub(crate) fn context(&self) -> &Context {
        &self.module.context
    }

    pub(crate) const fn raw(&self) -> CuFunction {
        self.raw
    }
}
