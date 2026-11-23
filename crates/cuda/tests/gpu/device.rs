//! Devices and the driver's error names.

use ptxc_cuda::{CudaError, DeviceOrdinal, Driver};

#[test]
fn a_device_that_does_not_exist_is_an_error() {
    let error = Driver::load().unwrap().device(DeviceOrdinal::new(999)).unwrap_err();

    let CudaError::Driver { call, name, .. } = &error else { panic!("{error}") };
    assert_eq!((*call, name.as_str()), ("cuDeviceGet", "CUDA_ERROR_INVALID_DEVICE"));
}

#[test]
fn the_device_reports_a_compute_capability() {
    let device =
        ptxc_cuda::Driver::load().unwrap().device(ptxc_cuda::DeviceOrdinal::new(0)).unwrap();

    let capability = device.compute_capability().unwrap();

    assert!(capability.major >= 5, "{capability:?}");
}
