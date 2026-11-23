//! Devices and the driver's error names.

use ptxc_cuda::{CudaError, DeviceOrdinal, Driver};

#[test]
fn a_device_that_does_not_exist_is_an_error() {
    let error = Driver::load().unwrap().device(DeviceOrdinal::new(999)).unwrap_err();

    let CudaError::Driver { call, name, .. } = &error else { panic!("{error}") };
    assert_eq!((*call, name.as_str()), ("cuDeviceGet", "CUDA_ERROR_INVALID_DEVICE"));
}
