//! Comparing, reporting and blessing golden files in a temporary directory.

use super::{Difference, GoldenError, LineNumber, Mode, check_in, first_difference};
use std::path::Path;

#[test]
fn equal_texts_have_no_difference() {
    assert_eq!(first_difference("a\nb\n", "a\nb\n"), None);
}

#[test]
fn the_first_differing_line_is_reported_with_both_sides() {
    let difference = first_difference("a\nb\nc\n", "a\nx\nc\n");

    assert_eq!(
        difference,
        Some(Difference {
            line: LineNumber(2),
            expected: Some("b".to_owned()),
            actual: Some("x".to_owned())
        })
    );
}

#[test]
fn a_text_that_ends_early_has_no_line_there() {
    let shorter = first_difference("a\nb", "a").unwrap();
    let longer = first_difference("a", "a\nb").unwrap();

    assert_eq!((shorter.line, shorter.actual), (LineNumber(2), None));
    assert_eq!((longer.line, longer.expected), (LineNumber(2), None));
}

#[test]
fn blessing_writes_the_file_and_checking_then_passes() {
    let dir = tempfile::tempdir().unwrap();
    let path = Path::new("nested/out.txt");

    check_in(dir.path(), path, "result\n", Mode::Bless).unwrap();

    assert_eq!(std::fs::read_to_string(dir.path().join(path)).unwrap(), "result\n");
    check_in(dir.path(), path, "result\n", Mode::Compare).unwrap();
}

#[test]
fn a_changed_result_reports_where_and_how_to_accept_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("out.txt"), "one\ntwo\n").unwrap();

    let error = check_in(dir.path(), Path::new("out.txt"), "one\n2\n", Mode::Compare).unwrap_err();

    let message = format!("{error:?}");
    assert!(matches!(error, GoldenError::Differs { .. }));
    assert!(message.contains("at line 2:\n  expected \"two\"\n       got \"2\""), "{message}");
    assert!(message.contains("PTXC_BLESS=1"), "{message}");
}

#[test]
fn a_missing_file_says_how_to_create_it() {
    let dir = tempfile::tempdir().unwrap();

    let error = check_in(dir.path(), Path::new("absent.txt"), "x", Mode::Compare).unwrap_err();

    assert!(matches!(error, GoldenError::Missing { .. }));
    assert!(
        error.to_string().ends_with("does not exist; run with PTXC_BLESS=1 to create it"),
        "{error}"
    );
}

#[test]
fn only_a_bless_value_of_one_blesses() {
    assert_eq!(Mode::from_bless(Some("1".as_ref())), Mode::Bless);
    assert_eq!(Mode::from_bless(Some("0".as_ref())), Mode::Compare);
    assert_eq!(Mode::from_bless(None), Mode::Compare);
}

#[test]
fn an_unreadable_golden_file_is_an_io_error_not_a_missing_one() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("a_directory")).unwrap();

    let error = check_in(dir.path(), Path::new("a_directory"), "x", Mode::Compare).unwrap_err();

    assert!(matches!(error, GoldenError::Io { .. }), "{error}");
}
