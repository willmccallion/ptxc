//! The dependency rules of ARCHITECTURE.md, checked against the workspace
//! and its sources. Cargo already refuses a cyclic graph, rule 1.

use super::crates::{Scope, Workspace};
use super::sources::{Manifests, SourceFile};
use std::fmt;

/// Crates that never reach a GPU, rule 2, from their tests either.
const GPU_FREE: [&str; 15] = [
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
];

/// Crates that may reach a GPU.
const GPU: [&str; 4] = ["cuda", "runtime", "autotune", "bench"];

/// The one crate that mentions cuBLAS, rule 3.
const CUBLAS_CRATE: &str = "bench";

/// The one crate that allows unsafe code, rule 4.
const UNSAFE_CRATE: &str = "cuda";

/// The crates whose libraries never depend on each other, rule 5. Their
/// tests compare them through `testkit`.
const SEPARATE: [&str; 2] = ["interp", "lower"];

/// A broken rule.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Violation {
    /// A rule names a crate the workspace lacks.
    UnknownCrate(&'static str),
    /// A GPU-free crate reaches a GPU crate by this chain.
    ReachesGpu(Vec<String>),
    /// A crate depends on `bench`.
    DependsOnBench(String),
    /// A file outside `bench` mentions cuBLAS.
    MentionsCublas(String),
    /// A crate does not inherit the workspace's lints.
    OwnLints(String),
    /// The workspace does not deny `unsafe_code`.
    UnsafeNotDenied,
    /// A file outside `cuda` names the `unsafe_code` lint.
    RelaxesUnsafe(String),
    /// `interp` and `lower` are joined by this chain.
    InterpAndLower(Vec<String>),
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCrate(name) => write!(f, "a rule names `{name}`, which is not a crate"),
            Self::ReachesGpu(chain) => {
                write!(f, "GPU-free `{}` reaches a GPU crate: {}", chain[0], chain.join(" -> "))
            }
            Self::DependsOnBench(name) => write!(f, "`{name}` depends on `bench`"),
            Self::MentionsCublas(file) => write!(f, "{file} mentions cuBLAS outside `bench`"),
            Self::OwnLints(name) => write!(f, "`{name}` does not set `[lints] workspace = true`"),
            Self::UnsafeNotDenied => write!(f, "the workspace lints do not deny `unsafe_code`"),
            Self::RelaxesUnsafe(file) => write!(f, "{file} names `unsafe_code` outside `cuda`"),
            Self::InterpAndLower(chain) => write!(f, "{}", chain.join(" -> ")),
        }
    }
}

/// Every rule `workspace`, its manifests and its `sources` break.
pub(super) fn violations(
    workspace: &Workspace,
    manifests: &Manifests,
    sources: &[SourceFile],
) -> Vec<Violation> {
    let mut found = unknown_crates(workspace);
    found.extend(gpu_reach(workspace));
    found.extend(bench_dependents(workspace));
    found.extend(cublas_mentions(sources));
    found.extend(unsafe_rules(manifests, sources));
    found.extend(interp_lower(workspace));
    found
}

fn unknown_crates(workspace: &Workspace) -> Vec<Violation> {
    let named = GPU_FREE.iter().chain(&GPU).chain(&SEPARATE).chain([&CUBLAS_CRATE, &UNSAFE_CRATE]);
    named.filter(|&&name| !workspace.has(name)).map(|&name| Violation::UnknownCrate(name)).collect()
}

fn gpu_reach(workspace: &Workspace) -> Vec<Violation> {
    GPU_FREE
        .iter()
        .flat_map(|from| GPU.iter().filter_map(|to| workspace.path(from, to, Scope::WithTests)))
        .map(Violation::ReachesGpu)
        .collect()
}

fn bench_dependents(workspace: &Workspace) -> Vec<Violation> {
    workspace
        .crates
        .iter()
        .filter(|krate| krate.dependencies.iter().any(|dependency| dependency.name == CUBLAS_CRATE))
        .map(|krate| Violation::DependsOnBench(krate.name.clone()))
        .collect()
}

fn cublas_mentions(sources: &[SourceFile]) -> Vec<Violation> {
    sources
        .iter()
        .filter(|file| file.krate != CUBLAS_CRATE && file.text.to_lowercase().contains("cublas"))
        .map(|file| Violation::MentionsCublas(file.path.clone()))
        .collect()
}

fn unsafe_rules(manifests: &Manifests, sources: &[SourceFile]) -> Vec<Violation> {
    let mut found: Vec<Violation> = manifests
        .crates
        .iter()
        .filter(|(_, manifest)| !inherits_lints(manifest))
        .map(|(name, _)| Violation::OwnLints(name.clone()))
        .collect();
    if !denies_unsafe(&manifests.root) {
        found.push(Violation::UnsafeNotDenied);
    }
    found.extend(
        sources
            .iter()
            .filter(|file| file.krate != UNSAFE_CRATE && file.text.contains("unsafe_code"))
            .map(|file| Violation::RelaxesUnsafe(file.path.clone())),
    );
    found
}

fn interp_lower(workspace: &Workspace) -> Vec<Violation> {
    let [first, second] = SEPARATE;
    [workspace.path(first, second, Scope::Library), workspace.path(second, first, Scope::Library)]
        .into_iter()
        .flatten()
        .map(Violation::InterpAndLower)
        .collect()
}

/// Whether a crate manifest's `[lints]` table is `workspace = true`.
fn inherits_lints(manifest: &str) -> bool {
    table_lines(manifest, "[lints]").any(|line| line == "workspace = true")
}

/// Whether the root manifest's Rust lints deny `unsafe_code`.
fn denies_unsafe(root: &str) -> bool {
    table_lines(root, "[workspace.lints.rust]").any(|line| line == "unsafe_code = \"deny\"")
}

/// The trimmed lines of `manifest`'s table headed `header`.
fn table_lines<'a>(manifest: &'a str, header: &'a str) -> impl Iterator<Item = &'a str> {
    manifest
        .lines()
        .map(str::trim)
        .skip_while(move |line| *line != header)
        .skip(1)
        .take_while(|line| !line.starts_with('['))
}
