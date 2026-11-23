//! cuBLASLt's entry points, for the types and algorithm choices `GemmEx`
//! lacks, each typed as `cublasLt.h` declares it.

#![allow(unsafe_code, reason = "calls into the cuBLASLt library it loads (D55)")]

use super::cublas::{ComputeType, CublasError, DataType, Operation, device_pointer};
use ptxc_core::ByteSize;
use ptxc_cuda::{DeviceAddress, DeviceBuffer, Library, Stream};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

const LIBRARY: &CStr = c"libcublasLt.so.13";

/// `CUBLAS_STATUS_SUCCESS` and `CUBLAS_STATUS_NOT_SUPPORTED`.
const SUCCESS: c_int = 0;
const NOT_SUPPORTED: c_int = 15;

/// `CUBLASLT_MATMUL_DESC_TRANSA` and `_TRANSB`.
const DESC_TRANSA: c_int = 3;
const DESC_TRANSB: c_int = 4;

/// `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`.
const PREF_MAX_WORKSPACE: c_int = 1;

type Status = c_int;
type Raw = *mut c_void;

/// `cublasLtMatmulAlgo_t`: an opaque algorithm and its configuration.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Algo {
    data: [u64; 8],
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct HeuristicResult {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [c_int; 4],
}

type Heuristic = unsafe extern "C" fn(
    Raw,
    Raw,
    Raw,
    Raw,
    Raw,
    Raw,
    Raw,
    c_int,
    *mut HeuristicResult,
    *mut c_int,
) -> Status;
type Matmul = unsafe extern "C" fn(
    Raw,
    Raw,
    *const c_void,
    *const c_void,
    Raw,
    *const c_void,
    Raw,
    *const c_void,
    *const c_void,
    Raw,
    *mut c_void,
    Raw,
    *const Algo,
    *mut c_void,
    usize,
    *mut c_void,
) -> Status;

struct Api {
    create: unsafe extern "C" fn(*mut Raw) -> Status,
    destroy: unsafe extern "C" fn(Raw) -> Status,
    desc_create: unsafe extern "C" fn(*mut Raw, c_int, c_int) -> Status,
    desc_destroy: unsafe extern "C" fn(Raw) -> Status,
    desc_set: unsafe extern "C" fn(Raw, c_int, *const c_void, usize) -> Status,
    layout_create: unsafe extern "C" fn(*mut Raw, c_int, u64, u64, i64) -> Status,
    layout_destroy: unsafe extern "C" fn(Raw) -> Status,
    pref_create: unsafe extern "C" fn(*mut Raw) -> Status,
    pref_destroy: unsafe extern "C" fn(Raw) -> Status,
    pref_set: unsafe extern "C" fn(Raw, c_int, *const c_void, usize) -> Status,
    heuristic: Heuristic,
    matmul: Matmul,
    status_name: unsafe extern "C" fn(Status) -> *const c_char,
}

static API: OnceLock<Result<Api, CublasError>> = OnceLock::new();

/// A matrix as cuBLASLt stores it, column-major: rows, columns and the
/// leading dimension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cublas) struct MatrixLayout {
    pub(in crate::cublas) rows: u64,
    pub(in crate::cublas) columns: u64,
    pub(in crate::cublas) leading: i64,
}

/// A matmul `d = op(a) op(b)`, with `alpha = 1` and `beta = 0`, in
/// cuBLASLt's terms.
#[derive(Clone, Copy, Debug)]
pub(in crate::cublas) struct LtProblem {
    pub(in crate::cublas) operations: [Operation; 2],
    pub(in crate::cublas) layouts: [MatrixLayout; 3],
    pub(in crate::cublas) data: DataType,
    pub(in crate::cublas) output: DataType,
    pub(in crate::cublas) compute: ComputeType,
}

/// One algorithm the heuristic proposed, and the workspace it needs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    algo: Algo,
    pub(crate) workspace: ByteSize,
}

/// A cuBLASLt handle queuing its work on one stream.
pub(crate) struct LtHandle<'s> {
    api: &'static Api,
    raw: Raw,
    stream: &'s Stream,
}

/// A problem's descriptors, and the algorithms proposed for it, best
/// first; none if cuBLASLt cannot run it.
pub(crate) struct Plan {
    api: &'static Api,
    desc: Raw,
    layouts: [Raw; 3],
    compute: ComputeType,
    pub(crate) candidates: Vec<Candidate>,
}

impl Drop for LtHandle<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from cublasLtCreate and is destroyed once.
        let _status = unsafe { (self.api.destroy)(self.raw) };
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        for layout in self.layouts.into_iter().filter(|layout| !layout.is_null()) {
            // SAFETY: each non-null layout came from cublasLtMatrixLayoutCreate
            // and is destroyed once.
            let _status = unsafe { (self.api.layout_destroy)(layout) };
        }
        if !self.desc.is_null() {
            // SAFETY: the descriptor came from cublasLtMatmulDescCreate and is
            // destroyed once.
            let _status = unsafe { (self.api.desc_destroy)(self.desc) };
        }
    }
}

impl<'s> LtHandle<'s> {
    /// A handle whose work queues on `stream`.
    pub(crate) fn new(stream: &'s Stream) -> Result<Self, CublasError> {
        let api = API.get_or_init(Api::load).as_ref().map_err(Clone::clone)?;
        let mut raw: Raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid place for cuBLASLt to write the handle.
        api.check("cublasLtCreate", unsafe { (api.create)(&raw mut raw) })?;
        Ok(Self { api, raw, stream })
    }

    /// `problem`'s descriptors, and up to `count` algorithms the heuristic
    /// proposes within `workspace` bytes.
    pub(in crate::cublas) fn plan(
        &self,
        problem: &LtProblem,
        count: u8,
        workspace: ByteSize,
    ) -> Result<Plan, CublasError> {
        let api = self.api;
        let mut plan = Plan {
            api,
            desc: std::ptr::null_mut(),
            layouts: [std::ptr::null_mut(); 3],
            compute: problem.compute,
            candidates: Vec::new(),
        };
        let scale = scale_type(problem.compute) as c_int;
        // SAFETY: `plan.desc` is a valid place for the descriptor.
        let created =
            unsafe { (api.desc_create)(&raw mut plan.desc, problem.compute as c_int, scale) };
        api.check("cublasLtMatmulDescCreate", created)?;
        for (attribute, operation) in [DESC_TRANSA, DESC_TRANSB].into_iter().zip(problem.operations)
        {
            let value = operation as c_int;
            // SAFETY: the descriptor is alive, and both attributes are a
            // `cublasOperation_t`, an int, read from `value` before return.
            let set = unsafe {
                (api.desc_set)(plan.desc, attribute, (&raw const value).cast(), size_of::<c_int>())
            };
            api.check("cublasLtMatmulDescSetAttribute", set)?;
        }
        let types = [problem.data, problem.data, problem.output];
        for ((slot, layout), data) in plan.layouts.iter_mut().zip(problem.layouts).zip(types) {
            // SAFETY: `slot` is a valid place for the layout.
            let created = unsafe {
                (api.layout_create)(
                    slot,
                    data as c_int,
                    layout.rows,
                    layout.columns,
                    layout.leading,
                )
            };
            api.check("cublasLtMatrixLayoutCreate", created)?;
        }
        plan.candidates = self.heuristic(&plan, count, workspace)?;
        Ok(plan)
    }

    /// Up to `count` algorithms for `plan` within `workspace` bytes, best
    /// first; none if cuBLASLt has none.
    fn heuristic(
        &self,
        plan: &Plan,
        count: u8,
        workspace: ByteSize,
    ) -> Result<Vec<Candidate>, CublasError> {
        let api = self.api;
        let mut preference: Raw = std::ptr::null_mut();
        // SAFETY: `preference` is a valid place for the preference.
        api.check("cublasLtMatmulPreferenceCreate", unsafe {
            (api.pref_create)(&raw mut preference)
        })?;
        let limit = workspace.get();
        // SAFETY: the preference is alive, and its maximum workspace is a
        // `uint64_t`, read from `limit` before return.
        let set = unsafe {
            (api.pref_set)(
                preference,
                PREF_MAX_WORKSPACE,
                (&raw const limit).cast(),
                size_of::<u64>(),
            )
        };
        let empty = HeuristicResult {
            algo: Algo { data: [0; 8] },
            workspace_size: 0,
            state: SUCCESS,
            waves_count: 0.0,
            reserved: [0; 4],
        };
        let mut results = vec![empty; usize::from(count)];
        let mut found: c_int = 0;
        let [a, b, d] = plan.layouts;
        // SAFETY: the handle, descriptor, layouts and preference are alive;
        // `results` holds `count` entries for cuBLASLt to fill, and `found`
        // is a valid place for how many it did. C and D share D's layout.
        let status = unsafe {
            (api.heuristic)(
                self.raw,
                plan.desc,
                a,
                b,
                d,
                d,
                preference,
                c_int::from(count),
                results.as_mut_ptr(),
                &raw mut found,
            )
        };
        // SAFETY: the preference came from cublasLtMatmulPreferenceCreate
        // and is destroyed once.
        let _destroyed = unsafe { (api.pref_destroy)(preference) };
        api.check("cublasLtMatmulPreferenceSetAttribute", set)?;
        if status == NOT_SUPPORTED {
            return Ok(Vec::new());
        }
        api.check("cublasLtMatmulAlgoGetHeuristic", status)?;
        let found = usize::try_from(found).map_err(|_| CublasError::Call {
            call: "cublasLtMatmulAlgoGetHeuristic",
            name: format!("returned {found} algorithms"),
        })?;
        results.truncate(found);
        results
            .into_iter()
            .filter(|result| result.state == SUCCESS)
            .map(|result| {
                let bytes = u64::try_from(result.workspace_size)
                    .map_err(|_| CublasError::Size(u64::MAX))?;
                Ok(Candidate { algo: result.algo, workspace: ByteSize::new(bytes) })
            })
            .collect()
    }

    /// Queues `plan` with `candidate` on the handle's stream: `d = a b`,
    /// with `workspace` at least the candidate's.
    pub(in crate::cublas) fn matmul(
        &mut self,
        plan: &Plan,
        candidate: &Candidate,
        [a, b, d]: [DeviceAddress; 3],
        workspace: &DeviceBuffer,
    ) -> Result<(), CublasError> {
        if workspace.size().get() < candidate.workspace.get() {
            return Err(CublasError::Workspace {
                needed: candidate.workspace,
                actual: workspace.size(),
            });
        }
        let (f32_one, f32_zero, i32_one, i32_zero) = (1.0_f32, 0.0_f32, 1_i32, 0_i32);
        let (alpha, beta): (*const c_void, *const c_void) = match plan.compute {
            ComputeType::F32 => ((&raw const f32_one).cast(), (&raw const f32_zero).cast()),
            ComputeType::I32 => ((&raw const i32_one).cast(), (&raw const i32_zero).cast()),
        };
        let [a, b, d, scratch] = [a, b, d, workspace.device_address()].map(device_pointer);
        let (a, b, d, scratch) = (a?, b?, d?, scratch?);
        let [a_layout, b_layout, d_layout] = plan.layouts;
        let bytes = workspace.size().get();
        let size = usize::try_from(bytes).map_err(|_| CublasError::Size(bytes))?;
        // SAFETY: the handle, descriptor and layouts are alive; alpha and
        // beta point to the scale type the compute type implies, read
        // before return; the operands and workspace are device addresses,
        // and no host memory is mapped into the device's address space
        // (D37), so a wrong extent can only fault on the device; the stream
        // outlives the handle.
        let status = unsafe {
            (self.api.matmul)(
                self.raw,
                plan.desc,
                alpha,
                a,
                a_layout,
                b,
                b_layout,
                beta,
                d,
                d_layout,
                d,
                d_layout,
                &raw const candidate.algo,
                scratch,
                size,
                self.stream.handle(),
            )
        };
        self.api.check("cublasLtMatmul", status)
    }
}

/// The type alpha and beta have for `compute`.
const fn scale_type(compute: ComputeType) -> DataType {
    match compute {
        ComputeType::F32 => DataType::F32,
        ComputeType::I32 => DataType::I32,
    }
}

impl Api {
    fn load() -> Result<Self, CublasError> {
        let library = Library::open(LIBRARY).map_err(CublasError::Load)?;
        // SAFETY (each): the field's type transcribes the symbol's
        // declaration in cublasLt.h, and the library is never closed.
        Ok(Self {
            // SAFETY: cublasLtCreate(cublasLtHandle_t*).
            create: unsafe { resolve(&library, c"cublasLtCreate")? },
            // SAFETY: cublasLtDestroy(cublasLtHandle_t).
            destroy: unsafe { resolve(&library, c"cublasLtDestroy")? },
            // SAFETY: cublasLtMatmulDescCreate(desc*, computeType, scaleType).
            desc_create: unsafe { resolve(&library, c"cublasLtMatmulDescCreate")? },
            // SAFETY: cublasLtMatmulDescDestroy(desc).
            desc_destroy: unsafe { resolve(&library, c"cublasLtMatmulDescDestroy")? },
            // SAFETY: cublasLtMatmulDescSetAttribute(desc, attr, buf, size).
            desc_set: unsafe { resolve(&library, c"cublasLtMatmulDescSetAttribute")? },
            // SAFETY: cublasLtMatrixLayoutCreate(layout*, type, rows, cols, ld).
            layout_create: unsafe { resolve(&library, c"cublasLtMatrixLayoutCreate")? },
            // SAFETY: cublasLtMatrixLayoutDestroy(layout).
            layout_destroy: unsafe { resolve(&library, c"cublasLtMatrixLayoutDestroy")? },
            // SAFETY: cublasLtMatmulPreferenceCreate(pref*).
            pref_create: unsafe { resolve(&library, c"cublasLtMatmulPreferenceCreate")? },
            // SAFETY: cublasLtMatmulPreferenceDestroy(pref).
            pref_destroy: unsafe { resolve(&library, c"cublasLtMatmulPreferenceDestroy")? },
            // SAFETY: cublasLtMatmulPreferenceSetAttribute(pref, attr, buf, size).
            pref_set: unsafe { resolve(&library, c"cublasLtMatmulPreferenceSetAttribute")? },
            // SAFETY: cublasLtMatmulAlgoGetHeuristic's ten parameters, as
            // `Heuristic` lists them.
            heuristic: unsafe { resolve(&library, c"cublasLtMatmulAlgoGetHeuristic")? },
            // SAFETY: cublasLtMatmul's sixteen parameters, as `Matmul` lists them.
            matmul: unsafe { resolve(&library, c"cublasLtMatmul")? },
            // SAFETY: const char* cublasLtGetStatusName(cublasStatus_t).
            status_name: unsafe { resolve(&library, c"cublasLtGetStatusName")? },
        })
    }

    fn check(&self, call: &'static str, status: Status) -> Result<(), CublasError> {
        if status == SUCCESS {
            return Ok(());
        }
        // SAFETY: cublasLtGetStatusName takes any status and returns a
        // static NUL-terminated string, or null.
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
    // SAFETY: the caller's promise is `resolve`'s own.
    unsafe { super::cublas::resolve(library, symbol) }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{LtHandle, LtProblem, MatrixLayout};
    use crate::cublas::sys::{ComputeType, CublasError, DataType, Operation};
    use ptxc_core::ByteSize;
    use ptxc_cuda::{DeviceOrdinal, Driver};

    #[test]
    fn a_layout_whose_leading_dimension_is_below_its_rows_is_an_error_naming_the_call() {
        let context =
            Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap().primary_context();
        let stream = context.stream().unwrap();
        let handle = LtHandle::new(&stream).unwrap();
        let bad = MatrixLayout { rows: 16, columns: 16, leading: 4 };
        let problem = LtProblem {
            operations: [Operation::AsIs, Operation::AsIs],
            layouts: [bad; 3],
            data: DataType::F32,
            output: DataType::F32,
            compute: ComputeType::F32,
        };

        let outcome = handle.plan(&problem, 8, ByteSize::new(0));

        let Err(CublasError::Call { call, name }) = outcome.map(|plan| plan.candidates.len())
        else {
            panic!("planned a bad layout")
        };
        assert!(call.starts_with("cublasLt"), "{call}");
        assert!(name.starts_with("CUBLAS_STATUS_"), "{name}");
    }
}
