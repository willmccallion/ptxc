//! The function-pointer table: each driver API function ptxc calls,
//! resolved from `libcuda.so.1` by its exported name. Where `cuda.h` maps a
//! name to a versioned symbol (`cuMemAlloc` to `cuMemAlloc_v2`), the table
//! uses the versioned one, as a compiled C caller would.

use super::library::Library;
use super::types::{
    CuContext, CuDevice, CuDevicePtr, CuEvent, CuFunction, CuJitOption, CuModule, CuResult,
    CuStream,
};
use std::ffi::{c_char, c_int, c_uint, c_void};

macro_rules! api {
    ($($(#[$doc:meta])* $field:ident = $symbol:literal: fn($($arg:ty),* $(,)?);)*) => {
        /// The resolved driver functions.
        pub(crate) struct Api {
            $($(#[$doc])* pub(crate) $field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
        }

        impl Api {
            /// Resolves every function from `library`.
            pub(crate) fn resolve(library: &Library) -> Result<Self, String> {
                Ok(Self {$($field: {
                    let address = library.symbol($symbol)?;
                    // SAFETY: `$symbol` is exported by the CUDA driver with
                    // the C prototype `cuda.h` gives, which this signature
                    // transcribes.
                    unsafe {
                        std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn($($arg),*) -> CuResult,
                        >(address.as_ptr())
                    }
                },)*})
            }
        }
    };
}

api! {
    /// `cuInit`.
    init = c"cuInit": fn(c_uint);
    /// `cuGetErrorName`.
    get_error_name = c"cuGetErrorName": fn(CuResult, *mut *const c_char);
    /// `cuDeviceGet`.
    device_get = c"cuDeviceGet": fn(*mut CuDevice, c_int);
    /// `cuDevicePrimaryCtxRetain`.
    primary_ctx_retain = c"cuDevicePrimaryCtxRetain": fn(*mut CuContext, CuDevice);
    /// `cuDevicePrimaryCtxRelease`.
    primary_ctx_release = c"cuDevicePrimaryCtxRelease_v2": fn(CuDevice);
    /// `cuCtxSetCurrent`.
    ctx_set_current = c"cuCtxSetCurrent": fn(CuContext);
    /// `cuModuleLoadDataEx`.
    module_load_data_ex = c"cuModuleLoadDataEx":
        fn(*mut CuModule, *const c_void, c_uint, *mut CuJitOption, *mut *mut c_void);
    /// `cuModuleUnload`.
    module_unload = c"cuModuleUnload": fn(CuModule);
    /// `cuModuleGetFunction`.
    module_get_function = c"cuModuleGetFunction": fn(*mut CuFunction, CuModule, *const c_char);
    /// `cuFuncGetParamInfo`.
    func_get_param_info = c"cuFuncGetParamInfo": fn(CuFunction, usize, *mut usize, *mut usize);
    /// `cuMemAlloc`.
    mem_alloc = c"cuMemAlloc_v2": fn(*mut CuDevicePtr, usize);
    /// `cuMemFree`.
    mem_free = c"cuMemFree_v2": fn(CuDevicePtr);
    /// `cuMemcpyHtoD`.
    memcpy_htod = c"cuMemcpyHtoD_v2": fn(CuDevicePtr, *const c_void, usize);
    /// `cuMemcpyDtoH`.
    memcpy_dtoh = c"cuMemcpyDtoH_v2": fn(*mut c_void, CuDevicePtr, usize);
    /// `cuStreamCreate`.
    stream_create = c"cuStreamCreate": fn(*mut CuStream, c_uint);
    /// `cuStreamDestroy`.
    stream_destroy = c"cuStreamDestroy_v2": fn(CuStream);
    /// `cuStreamSynchronize`.
    stream_synchronize = c"cuStreamSynchronize": fn(CuStream);
    /// `cuEventCreate`.
    event_create = c"cuEventCreate": fn(*mut CuEvent, c_uint);
    /// `cuEventDestroy`.
    event_destroy = c"cuEventDestroy_v2": fn(CuEvent);
    /// `cuEventRecord`.
    event_record = c"cuEventRecord": fn(CuEvent, CuStream);
    /// `cuEventSynchronize`.
    event_synchronize = c"cuEventSynchronize": fn(CuEvent);
    /// `cuEventElapsedTime`.
    event_elapsed_time = c"cuEventElapsedTime_v2": fn(*mut f32, CuEvent, CuEvent);
    /// `cuLaunchKernel`.
    launch_kernel = c"cuLaunchKernel": fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    );
}
