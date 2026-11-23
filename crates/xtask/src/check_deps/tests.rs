use super::crates::Workspace;
use super::rules::{Violation, violations};
use super::sources::{Manifests, SourceFile, files_under};
use serde_json::json;

/// Every crate the rules name, so none is reported unknown.
const NAMED: [&str; 18] = [
    "core",
    "target",
    "onnx-proto",
    "onnx",
    "graph",
    "interp",
    "graph-opt",
    "loop",
    "lower",
    "schedule",
    "codegen",
    "cost-model",
    "program",
    "compiler",
    "testkit",
    "cuda",
    "runtime",
    "autotune",
];

const ROOT: &str = "[workspace.lints.rust]\nunsafe_code = \"deny\"\n";
const MEMBER: &str = "[package]\nname = \"x\"\n\n[lints]\nworkspace = true\n";

/// A workspace of `NAMED` and `bench`, with `edges` as (from, to, kind).
fn workspace(edges: &[(&str, &str, Option<&str>)]) -> Workspace {
    let packages: Vec<_> = NAMED
        .iter()
        .chain(&["bench", "cli"])
        .map(|&name| {
            let dependencies: Vec<_> = edges
                .iter()
                .filter(|(from, ..)| *from == name)
                .map(|(_, to, kind)| json!({ "name": format!("ptxc-{to}"), "kind": kind }))
                .chain([json!({ "name": "serde", "kind": null })])
                .collect();
            json!({
                "name": format!("ptxc-{name}"),
                "manifest_path": format!("/w/crates/{name}/Cargo.toml"),
                "dependencies": dependencies,
            })
        })
        .collect();
    Workspace::parse(&json!({ "packages": packages }).to_string()).unwrap()
}

fn clean_manifests() -> Manifests {
    Manifests { root: ROOT.to_owned(), crates: vec![("core".to_owned(), MEMBER.to_owned())] }
}

fn file(krate: &str, text: &str) -> SourceFile {
    SourceFile {
        krate: krate.to_owned(),
        path: format!("crates/{krate}/src/lib.rs"),
        text: text.to_owned(),
    }
}

fn chain(names: &[&str]) -> Vec<String> {
    names.iter().map(|&name| name.to_owned()).collect()
}

fn graph_violations(edges: &[(&str, &str, Option<&str>)]) -> Vec<Violation> {
    violations(&workspace(edges), &clean_manifests(), &[])
}

#[test]
fn the_workspace_follows_every_rule() {
    let outcome = super::run(&crate::workspace_root());

    assert!(outcome.is_ok(), "{}", outcome.unwrap_err());
}

#[test]
fn a_broken_workspace_fails_with_each_violation_listed() {
    let dir = tempfile::tempdir().unwrap();
    let files = [
        (
            "Cargo.toml",
            "[workspace]\nmembers = [\"crates/core\"]\nresolver = \"3\"\n\n[workspace.lints.rust]\nunsafe_code = \"deny\"\n",
        ),
        (
            "crates/core/Cargo.toml",
            "[package]\nname = \"ptxc-core\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
        ),
        ("crates/core/src/lib.rs", "//! Faster than cuBLAS.\n"),
    ];
    for (path, text) in files {
        let path = dir.path().join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    let message = super::run(dir.path()).unwrap_err().to_string();

    assert!(
        message.contains("\n  crates/core/src/lib.rs mentions cuBLAS outside `bench`"),
        "{message}"
    );
    assert!(message.contains("\n  a rule names `cuda`, which is not a crate"), "{message}");
}

#[test]
fn a_workspace_without_offending_edges_breaks_no_rule() {
    let found = graph_violations(&[("compiler", "codegen", None), ("cli", "runtime", None)]);

    assert_eq!(found, []);
}

#[test]
fn a_gpu_free_crate_reaching_a_gpu_crate_through_another_is_reported_with_its_chain() {
    let found = graph_violations(&[("onnx", "graph", None), ("graph", "runtime", None)]);

    assert_eq!(
        found,
        [
            Violation::ReachesGpu(chain(&["onnx", "graph", "runtime"])),
            Violation::ReachesGpu(chain(&["graph", "runtime"])),
        ]
    );
}

#[test]
fn a_crates_own_test_dependency_counts_for_the_gpu_rule() {
    let found = graph_violations(&[("compiler", "runtime", Some("dev"))]);

    assert_eq!(found, [Violation::ReachesGpu(chain(&["compiler", "runtime"]))]);
}

#[test]
fn a_dependencys_test_dependencies_do_not_count() {
    let found = graph_violations(&[("cli", "loop", None), ("loop", "cuda", Some("dev"))]);

    assert_eq!(found, [Violation::ReachesGpu(chain(&["loop", "cuda"]))]);
}

#[test]
fn interp_and_lower_may_meet_only_in_tests() {
    let in_tests =
        graph_violations(&[("lower", "testkit", Some("dev")), ("testkit", "interp", None)]);
    let in_library =
        graph_violations(&[("interp", "graph-opt", None), ("graph-opt", "lower", None)]);

    assert_eq!(in_tests, []);
    assert_eq!(in_library, [Violation::InterpAndLower(chain(&["interp", "graph-opt", "lower"]))]);
}

#[test]
fn depending_on_bench_is_reported() {
    let found = graph_violations(&[("cli", "bench", Some("dev"))]);

    assert_eq!(found, [Violation::DependsOnBench("cli".to_owned())]);
}

#[test]
fn a_rule_naming_a_missing_crate_is_reported() {
    let packages =
        json!({ "packages": [{ "name": "ptxc-core", "manifest_path": "/w", "dependencies": [] }] });
    let lone = Workspace::parse(&packages.to_string()).unwrap();

    let found = violations(&lone, &clean_manifests(), &[]);

    assert!(found.contains(&Violation::UnknownCrate("cost-model")), "{found:?}");
    assert!(!found.contains(&Violation::UnknownCrate("core")), "{found:?}");
}

#[test]
fn only_bench_may_mention_cublas() {
    let sources = [file("bench", "dlopen(\"libcublas.so\")"), file("codegen", "// beats CUBLAS")];

    let found = violations(&workspace(&[]), &clean_manifests(), &sources);

    assert_eq!(found, [Violation::MentionsCublas("crates/codegen/src/lib.rs".to_owned())]);
}

#[test]
fn only_cuda_may_name_the_unsafe_code_lint() {
    let sources = [file("cuda", "#![allow(unsafe_code)]"), file("core", "#![allow(unsafe_code)]")];

    let found = violations(&workspace(&[]), &clean_manifests(), &sources);

    assert_eq!(found, [Violation::RelaxesUnsafe("crates/core/src/lib.rs".to_owned())]);
}

#[test]
fn of_bench_only_its_cublas_bindings_may_name_the_unsafe_code_lint() {
    let bindings = SourceFile {
        krate: "bench".to_owned(),
        path: "crates/bench/src/cublas/sys.rs".to_owned(),
        text: "#![allow(unsafe_code)]".to_owned(),
    };
    let sources = [bindings, file("bench", "#![allow(unsafe_code)]")];

    let found = violations(&workspace(&[]), &clean_manifests(), &sources);

    assert_eq!(found, [Violation::RelaxesUnsafe("crates/bench/src/lib.rs".to_owned())]);
}

#[test]
fn every_crate_inherits_lints_that_deny_unsafe_code() {
    let manifests = Manifests {
        root: "[workspace.lints.rust]\nunsafe_code = \"warn\"\n".to_owned(),
        crates: vec![
            ("core".to_owned(), MEMBER.to_owned()),
            ("graph".to_owned(), "[lints]\n\n[dependencies]\nworkspace = true\n".to_owned()),
        ],
    };

    let found = violations(&workspace(&[]), &manifests, &[]);

    assert_eq!(found, [Violation::OwnLints("graph".to_owned()), Violation::UnsafeNotDenied]);
}

#[test]
fn an_unknown_dependency_kind_is_a_parse_error() {
    let packages = json!({ "packages": [{
        "name": "ptxc-core",
        "manifest_path": "/w",
        "dependencies": [{ "name": "serde", "kind": "optional" }],
    }] });

    assert!(Workspace::parse(&packages.to_string()).is_err());
}

#[test]
fn the_file_walk_finds_rust_sources_and_manifests_outside_the_skipped_directory() {
    let dir = tempfile::tempdir().unwrap();
    for path in ["Cargo.toml", "src/lib.rs", "src/notes.md", "src/skipped/mod.rs"] {
        let path = dir.path().join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "").unwrap();
    }

    let mut found = files_under(dir.path(), &dir.path().join("src/skipped")).unwrap();
    found.sort();

    assert_eq!(found, [dir.path().join("Cargo.toml"), dir.path().join("src/lib.rs")]);
}
