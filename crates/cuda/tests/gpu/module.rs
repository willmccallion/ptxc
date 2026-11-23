//! Loading PTX and looking up kernels.

use crate::{VECTOR_ADD, context};
use ptxc_core::{ByteSize, KernelName};
use ptxc_cuda::CudaError;

#[test]
fn a_kernel_reports_its_parameter_sizes() {
    let module = context().load_ptx(VECTOR_ADD).unwrap();

    let kernel = module.function(&KernelName::new("vector_add")).unwrap();

    let expected: Vec<ByteSize> = [8, 8, 8, 4].map(ByteSize::new).to_vec();
    assert_eq!(kernel.parameter_sizes(), expected);
    assert_eq!(kernel.name().as_str(), "vector_add");
}

#[test]
fn malformed_ptx_fails_with_the_jit_log() {
    let broken = VECTOR_ADD.replace("add.f32 %f3", "frobnicate.f32 %f3");

    let error = context().load_ptx(&broken).unwrap_err();

    let CudaError::Driver { call, detail, .. } = &error else { panic!("{error}") };
    assert_eq!(*call, "cuModuleLoadDataEx");
    assert!(detail.as_deref().is_some_and(|log| log.contains("frobnicate")), "{error}");
}

#[test]
fn an_unknown_kernel_is_not_found() {
    let module = context().load_ptx(VECTOR_ADD).unwrap();

    let error = module.function(&KernelName::new("vector_sub")).unwrap_err();

    let CudaError::Driver { name, .. } = &error else { panic!("{error}") };
    assert_eq!(name, "CUDA_ERROR_NOT_FOUND");
}

#[test]
fn ptx_with_a_nul_byte_is_rejected_before_the_driver() {
    let error = context().load_ptx("\0").unwrap_err();

    assert_eq!(error, CudaError::InteriorNul("PTX"));
}
