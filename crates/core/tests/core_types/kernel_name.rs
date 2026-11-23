//! Kernel names.

use ptxc_core::KernelName;

#[test]
fn a_kernel_name_prints_as_written() {
    let name = KernelName::new("matmul_0");

    assert_eq!(name.as_str(), "matmul_0");
    assert_eq!(name.to_string(), "matmul_0");
}
