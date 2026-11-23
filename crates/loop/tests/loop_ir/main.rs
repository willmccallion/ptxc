//! Loop IR through its public API.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

mod compute_build;
mod compute_errors;
mod compute_types;
mod expressions;
mod interp;
mod lower;
mod schedule_properties;
mod schedule_rules;
mod schedule_support;
mod simplify_properties;
mod simplify_rules;
mod validate;
mod values;

/// Property tests run `default` cases, or `PROPTEST_CASES` when set, so a
/// longer run needs no code change (`ProptestConfig::with_cases` alone
/// ignores the variable).
pub(crate) fn cases(default: u32) -> proptest::prelude::ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|cases| cases.parse().ok())
        .unwrap_or(default);
    proptest::prelude::ProptestConfig::with_cases(cases)
}
