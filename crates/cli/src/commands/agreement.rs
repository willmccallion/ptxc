//! Whether two executors' outputs agree.

use ptxc_core::{DType, HostTensor};
use ptxc_testkit::compare::{Tolerance, compare};

/// Whether `actual` matches `expected` output by output, each within
/// `tolerance` of its element type.
pub(crate) fn agree(
    actual: &[HostTensor],
    expected: &[HostTensor],
    tolerance: impl Fn(DType) -> Tolerance,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{} outputs, expected {}", actual.len(), expected.len()));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        compare(actual, expected, tolerance(expected.dtype()))
            .map_err(|mismatch| format!("output {index}: {mismatch}"))?;
    }
    Ok(())
}
