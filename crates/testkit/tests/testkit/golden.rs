//! `golden::check` against this crate's own `tests/golden`.

use ptxc_testkit::golden::{GoldenError, check};

#[test]
fn a_matching_result_passes_and_a_changed_one_fails() {
    check("sample.txt", "line one\nline two\n").unwrap();

    assert!(matches!(check("sample.txt", "line one\nline 2\n"), Err(GoldenError::Differs { .. })));
}
