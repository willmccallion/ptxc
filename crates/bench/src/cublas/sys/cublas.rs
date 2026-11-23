//! cuBLAS's entry points, loaded once with ptxc-cuda's `dlopen` wrapper
//! (D24), and the calls into them, each typed as `cublas_api.h` declares
//! it.

#![allow(unsafe_code, reason = "calls into the cuBLAS library it loads (D55)")]

use ptxc_cuda::{DeviceAddress, Library, Stream};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;
use thiserror::Error;

/// The library, as the dev shell's `LD_LIBRARY_PATH` finds it.
const LIBRARY: &CStr = c"libcublas.so.13";

/// `CUBLAS_STATUS_SUCCESS`.
const SUCCESS: c_int = 0;

/// `CUBLAS_GEMM_DEFAULT`: the library's heuristic choice of algorithm.
const GEMM_DEFAULT: c_int = -1;

type Status = c_int;
type RawHandle = *mut c_void;
type GemmEx = unsafe extern "C" fn(
    RawHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> Status;

/// The resolved cuBLAS functions.
struct Api {
    create: unsafe extern "C" fn(*mut RawHandle) -> Status,
    destroy: unsafe extern "C" fn(RawHandle) -> Status,
    set_stream: unsafe extern "C" fn(RawHandle, *mut c_void) -> Status,
    status_name: unsafe extern "C" fn(Status) -> *const c_char,
    gemm_ex: GemmEx,
}

/// Why cuBLAS could not be loaded or a call failed.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub(crate) enum CublasError {
    #[error("loading cuBLAS: {0}")]
    Load(String),
    #[error("{call} failed: {name}")]
    Call { call: &'static str, name: String },
}

static API: OnceLock<Result<Api, CublasError>> = OnceLock::new();

/// A `cublasOperation_t`: whether cuBLAS reads an operand transposed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cublas) enum Operation {
    /// `CUBLAS_OP_N`.
    AsIs = 0,
    /// `CUBLAS_OP_T`.
    Transposed = 1,
}

/// A `cudaDataType`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cublas) enum DataType {
    /// `CUDA_R_32F`.
    F32 = 0,
    /// `CUDA_R_16F`.
    F16 = 2,
    /// `CUDA_R_16BF`.
    Bf16 = 14,
}

/// A `cublasComputeType_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cublas) enum ComputeType {
    /// `CUBLAS_COMPUTE_32F`: f32 arithmetic, never TF32.
    F32 = 68,
}

/// An operand of `cublasGemmEx`, in cuBLAS's column-major terms.
#[derive(Clone, Copy, Debug)]
pub(in crate::cublas) struct Operand {
    pub(in crate::cublas) address: DeviceAddress,
    pub(in crate::cublas) leading: c_int,
}

/// A `cublasGemmEx` call: `c = op(a) op(b)` with `alpha = 1`, `beta = 0`.
#[derive(Clone, Copy, Debug)]
pub(in crate::cublas) struct GemmCall {
    pub(in crate::cublas) operations: [Operation; 2],
    pub(in crate::cublas) extents: [c_int; 3],
    pub(in crate::cublas) a: Operand,
    pub(in crate::cublas) b: Operand,
    pub(in crate::cublas) c: Operand,
    pub(in crate::cublas) data: DataType,
    pub(in crate::cublas) output: DataType,
    pub(in crate::cublas) compute: ComputeType,
}

/// A cuBLAS handle queuing its work on one stream.
pub(crate) struct Handle<'s> {
    api: &'static Api,
    raw: RawHandle,
    _stream: &'s Stream,
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from cublasCreate and is destroyed exactly
        // once, here. A failed destroy cannot be reported from `drop`.
        let _status = unsafe { (self.api.destroy)(self.raw) };
    }
}

impl<'s> Handle<'s> {
    /// A handle whose work queues on `stream`.
    pub(crate) fn new(stream: &'s Stream) -> Result<Self, CublasError> {
        let api = API.get_or_init(Api::load).as_ref().map_err(Clone::clone)?;
        let mut raw: RawHandle = std::ptr::null_mut();
        // SAFETY: `raw` is a valid place for cuBLAS to write the handle.
        let created = unsafe { (api.create)(&raw mut raw) };
        api.check("cublasCreate", created)?;
        let handle = Self { api, raw, _stream: stream };
        // SAFETY: the handle is alive, and the stream outlives it, as
        // `'s` ensures.
        let set = unsafe { (api.set_stream)(handle.raw, stream.handle()) };
        api.check("cublasSetStream", set)?;
        Ok(handle)
    }

    /// Queues `call` on the handle's stream.
    pub(in crate::cublas) fn gemm_ex(&mut self, call: &GemmCall) -> Result<(), CublasError> {
        let (alpha, beta) = (1.0_f32, 0.0_f32);
        let [m, n, k] = call.extents;
        let address = |operand: Operand| operand.address.get() as *mut c_void;
        // SAFETY: the handle is alive; `alpha` and `beta` are f32s, which
        // CUBLAS_COMPUTE_32F reads from host memory before returning; the
        // operands are device addresses, and no host memory is mapped into
        // the device's address space (D37), so a wrong extent can only
        // fault on the device, which cuBLAS reports as an error.
        let status = unsafe {
            (self.api.gemm_ex)(
                self.raw,
                call.operations[0] as c_int,
                call.operations[1] as c_int,
                m,
                n,
                k,
                (&raw const alpha).cast(),
                address(call.a),
                call.data as c_int,
                call.a.leading,
                address(call.b),
                call.data as c_int,
                call.b.leading,
                (&raw const beta).cast(),
                address(call.c),
                call.output as c_int,
                call.c.leading,
                call.compute as c_int,
                GEMM_DEFAULT,
            )
        };
        self.api.check("cublasGemmEx", status)
    }
}

impl Api {
    fn load() -> Result<Self, CublasError> {
        let library = Library::open(LIBRARY).map_err(CublasError::Load)?;
        // SAFETY (each): the field's type transcribes the symbol's
        // declaration in cublas_api.h, and the library is never closed.
        Ok(Self {
            // SAFETY: cublasStatus_t cublasCreate_v2(cublasHandle_t*).
            create: unsafe { resolve(&library, c"cublasCreate_v2")? },
            // SAFETY: cublasStatus_t cublasDestroy_v2(cublasHandle_t).
            destroy: unsafe { resolve(&library, c"cublasDestroy_v2")? },
            // SAFETY: cublasStatus_t cublasSetStream_v2(cublasHandle_t, cudaStream_t).
            set_stream: unsafe { resolve(&library, c"cublasSetStream_v2")? },
            // SAFETY: const char* cublasGetStatusName(cublasStatus_t).
            status_name: unsafe { resolve(&library, c"cublasGetStatusName")? },
            // SAFETY: cublasGemmEx's 19 parameters, as `GemmEx` lists them.
            gemm_ex: unsafe { resolve(&library, c"cublasGemmEx")? },
        })
    }

    /// `status` as a result, naming the failed `call` and the status.
    fn check(&self, call: &'static str, status: Status) -> Result<(), CublasError> {
        if status == SUCCESS {
            return Ok(());
        }
        // SAFETY: cublasGetStatusName takes any status and returns a static
        // NUL-terminated string, or null.
        let name = unsafe { (self.status_name)(status) };
        let name = if name.is_null() {
            format!("status {status}")
        } else {
            // SAFETY: a non-null result is a static NUL-terminated string.
            unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
        };
        Err(CublasError::Call { call, name })
    }
}

/// `symbol` from `library`, as the function pointer type `F`.
///
/// # Safety
///
/// `F` must be a function pointer type matching the symbol's C declaration.
unsafe fn resolve<F: Copy>(library: &Library, symbol: &CStr) -> Result<F, CublasError> {
    let address = library.symbol(symbol).map_err(CublasError::Load)?.as_ptr();
    if size_of::<F>() != size_of::<*mut c_void>() {
        return Err(CublasError::Load(format!(
            "{} is not a function pointer",
            symbol.to_string_lossy()
        )));
    }
    // SAFETY: `F` is the size of a pointer, checked above, and the caller
    // promises it is the symbol's function pointer type.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{ComputeType, CublasError, DataType, GemmCall, Handle, Operand, Operation};
    use ptxc_core::ByteSize;
    use ptxc_cuda::{DeviceOrdinal, Driver};

    #[test]
    fn a_failed_call_is_an_error_naming_cublas_status() {
        let context =
            Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context();
        let buffer = context.alloc(ByteSize::new(64)).unwrap();
        let stream = context.stream().unwrap();
        let mut handle = Handle::new(&stream).unwrap();
        let operand = Operand { address: buffer.device_address(), leading: 0 };
        let call = GemmCall {
            operations: [Operation::AsIs, Operation::AsIs],
            extents: [2, 2, 2],
            a: operand,
            b: operand,
            c: operand,
            data: DataType::F32,
            output: DataType::F32,
            compute: ComputeType::F32,
        };

        let error = handle.gemm_ex(&call).unwrap_err();

        let expected = CublasError::Call {
            call: "cublasGemmEx",
            name: "CUBLAS_STATUS_INVALID_VALUE".to_owned(),
        };
        assert_eq!(error, expected);
    }
}
