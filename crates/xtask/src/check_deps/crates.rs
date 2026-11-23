//! The workspace's crates and the dependencies between them, as
//! `cargo metadata --no-deps` reports them.

use serde::Deserialize;
use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

/// The prefix every workspace crate's package name carries.
const PREFIX: &str = "ptxc-";

/// The workspace's crates.
#[derive(Debug)]
pub(super) struct Workspace {
    pub(super) crates: Vec<Crate>,
}

/// A workspace crate.
#[derive(Debug)]
pub(super) struct Crate {
    /// The name without `ptxc-`, as ARCHITECTURE.md writes it.
    pub(super) name: String,
    /// Its `Cargo.toml`.
    pub(super) manifest: PathBuf,
    /// Its dependencies on other workspace crates.
    pub(super) dependencies: Vec<Dependency>,
}

/// A dependency on a workspace crate.
#[derive(Debug)]
pub(super) struct Dependency {
    /// The crate depended on, without `ptxc-`.
    pub(super) name: String,
    /// Which builds use it.
    pub(super) kind: Kind,
}

/// Which of a crate's own builds a dependency path starts from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Scope {
    /// The library and its build script.
    Library,
    /// Those and the crate's tests, examples and benchmarks.
    WithTests,
}

/// Which builds use a dependency.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    /// The crate itself.
    Normal,
    /// Its build script.
    Build,
    /// Its tests, examples and benchmarks only.
    Dev,
}

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    manifest_path: PathBuf,
    dependencies: Vec<PackageDependency>,
}

#[derive(Deserialize)]
struct PackageDependency {
    name: String,
    /// `null` for a normal dependency.
    kind: Option<SpecialKind>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum SpecialKind {
    Dev,
    Build,
}

impl Workspace {
    /// The workspace `cargo metadata --no-deps --format-version 1` describes.
    pub(super) fn parse(json: &str) -> Result<Self, serde_json::Error> {
        let metadata: Metadata = serde_json::from_str(json)?;
        let members: BTreeSet<String> =
            metadata.packages.iter().map(|package| package.name.clone()).collect();
        let crates = metadata
            .packages
            .into_iter()
            .map(|package| Crate {
                name: short(&package.name),
                manifest: package.manifest_path,
                dependencies: package
                    .dependencies
                    .into_iter()
                    .filter(|dependency| members.contains(&dependency.name))
                    .map(|dependency| Dependency {
                        name: short(&dependency.name),
                        kind: match dependency.kind {
                            None => Kind::Normal,
                            Some(SpecialKind::Build) => Kind::Build,
                            Some(SpecialKind::Dev) => Kind::Dev,
                        },
                    })
                    .collect(),
            })
            .collect();
        Ok(Self { crates })
    }

    /// The chain of crates by which `from` reaches `to`, starting with
    /// `from`: its own dependencies in `scope`, then the normal and build
    /// dependencies of those, since another crate's dev-dependencies never
    /// build into it. `None` if it does not.
    pub(super) fn path(&self, from: &str, to: &str, scope: Scope) -> Option<Vec<String>> {
        let mut queue = VecDeque::from([vec![from.to_owned()]]);
        let mut seen = BTreeSet::from([from.to_owned()]);
        while let Some(chain) = queue.pop_front() {
            let last = chain.last()?;
            let follows_dev = chain.len() == 1 && scope == Scope::WithTests;
            for dependency in self.dependencies(last) {
                let skipped = dependency.kind == Kind::Dev && !follows_dev;
                if skipped || !seen.insert(dependency.name.clone()) {
                    continue;
                }
                let mut next = chain.clone();
                next.push(dependency.name.clone());
                if dependency.name == to {
                    return Some(next);
                }
                queue.push_back(next);
            }
        }
        None
    }

    /// Whether the workspace has a crate called `name`.
    pub(super) fn has(&self, name: &str) -> bool {
        self.crates.iter().any(|krate| krate.name == name)
    }

    fn dependencies(&self, name: &str) -> &[Dependency] {
        self.crates
            .iter()
            .find(|krate| krate.name == name)
            .map_or(&[], |krate| krate.dependencies.as_slice())
    }
}

fn short(package: &str) -> String {
    package.strip_prefix(PREFIX).unwrap_or(package).to_owned()
}
