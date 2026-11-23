//! A suite's results as JSON, in BENCHMARKING.md's schema: who ran it and
//! where, and each case's samples and summary, with the GPU's clocks
//! around its series.

use crate::clocks::{GpuIdentity, GpuState, clocks_held};
use crate::timing::Series;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use thiserror::Error;

/// The schema version this harness writes.
const SCHEMA: u32 = 1;

/// One suite's results for one framework.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct Results<P> {
    pub(crate) schema: u32,
    pub(crate) suite: String,
    pub(crate) framework: Framework,
    pub(crate) environment: Environment,
    pub(crate) cases: Vec<Case<P>>,
}

/// The framework timed, and the code it was built from.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct Framework {
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) commit: String,
    /// Whether the working tree had uncommitted changes, which published
    /// results never come from.
    pub(crate) dirty: bool,
}

/// The machine a suite ran on.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct Environment {
    pub(crate) gpu: String,
    pub(crate) driver: String,
}

/// One timed case: its parameters, whether its result was checked, its
/// samples and summary, and the clocks around its series.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct Case<P> {
    pub(crate) id: String,
    pub(crate) params: P,
    pub(crate) verified: bool,
    pub(crate) clocks: SeriesClocks,
    pub(crate) samples_us: Vec<f64>,
    pub(crate) median_us: f64,
    pub(crate) p10_us: f64,
    pub(crate) p90_us: f64,
}

/// The GPU's state before and after a series, and whether its clocks held.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub(crate) struct SeriesClocks {
    pub(crate) before: GpuState,
    pub(crate) after: GpuState,
    pub(crate) held: bool,
}

/// Why results could not be made or written.
#[derive(Debug, Error)]
pub(crate) enum ResultsError {
    #[error("a case's series has no samples")]
    Empty,
    #[error("running git: {0}")]
    Git(String),
    #[error("writing {path}: {source}")]
    Write { path: String, source: std::io::Error },
    #[error("encoding the results: {0}")]
    Encode(#[from] serde_json::Error),
}

impl<P> Case<P> {
    /// A case from its timed `series` and the GPU's state around it.
    pub(crate) fn new(
        id: String,
        params: P,
        verified: bool,
        clocks: (GpuState, GpuState),
        series: &Series,
    ) -> Result<Self, ResultsError> {
        let summary = series.summary().ok_or(ResultsError::Empty)?;
        let (before, after) = clocks;
        let held = clocks_held(&before, &after);
        Ok(Self {
            id,
            params,
            verified,
            clocks: SeriesClocks { before, after, held },
            samples_us: series.samples.iter().map(|sample| sample.get()).collect(),
            median_us: summary.median.get(),
            p10_us: summary.p10.get(),
            p90_us: summary.p90.get(),
        })
    }
}

impl<P: Serialize> Results<P> {
    /// Results of `suite` for `framework`, run on `gpu`.
    pub(crate) fn new(
        suite: &str,
        framework: Framework,
        gpu: GpuIdentity,
        cases: Vec<Case<P>>,
    ) -> Self {
        let environment = Environment { gpu: gpu.name, driver: gpu.driver };
        Self { schema: SCHEMA, suite: suite.to_owned(), framework, environment, cases }
    }

    /// Writes the results to `path` as indented JSON, creating its
    /// directory.
    pub(crate) fn write(&self, path: &Path) -> Result<(), ResultsError> {
        let failed = |source| ResultsError::Write { path: path.display().to_string(), source };
        if let Some(directory) = path.parent() {
            std::fs::create_dir_all(directory).map_err(failed)?;
        }
        let text = serde_json::to_string_pretty(self)? + "\n";
        std::fs::write(path, text).map_err(failed)
    }
}

impl Framework {
    /// ptxc at the commit checked out in `repository`.
    pub(crate) fn ptxc(repository: &Path) -> Result<Self, ResultsError> {
        let commit = git(repository, &["rev-parse", "--short=12", "HEAD"])?;
        Ok(Self::ptxc_at(&commit, &git(repository, &["status", "--porcelain"])?))
    }

    /// ptxc at the commit `git rev-parse` printed, its working tree as
    /// `git status --porcelain` printed it.
    fn ptxc_at(commit: &str, status: &str) -> Self {
        Self {
            name: "ptxc".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            commit: commit.trim().to_owned(),
            dirty: !status.trim().is_empty(),
        }
    }
}

/// `git`'s output for `arguments` in `repository`.
fn git(repository: &Path, arguments: &[&str]) -> Result<String, ResultsError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .map_err(|error| ResultsError::Git(error.to_string()))?;
    if !output.status.success() {
        return Err(ResultsError::Git(String::from_utf8_lossy(&output.stderr).trim().to_owned()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{Case, Framework, Results, ResultsError};
    use crate::clocks::{Celsius, ClockEvent, GpuIdentity, GpuState, Megahertz};
    use crate::timing::Series;
    use ptxc_core::Microseconds;
    use serde::{Deserialize, Serialize};
    use std::path::Path;
    use std::process::Command;

    #[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
    struct Shape {
        m: u64,
        dtype: String,
    }

    fn state(sm: u32) -> GpuState {
        GpuState {
            sm: Megahertz(sm),
            memory: Megahertz(9001),
            temperature: Celsius(60),
            events: vec![ClockEvent::SwPowerCap],
        }
    }

    fn series(samples: &[f64]) -> Series {
        Series { samples: samples.iter().copied().map(Microseconds::new).collect() }
    }

    fn results(cases: Vec<Case<Shape>>) -> Results<Shape> {
        let framework = Framework {
            name: "ptxc".to_owned(),
            version: "0.1.0".to_owned(),
            commit: "abc123".to_owned(),
            dirty: false,
        };
        let gpu = GpuIdentity {
            name: "NVIDIA GeForce RTX 4060 Ti".to_owned(),
            driver: "595.112.07".to_owned(),
        };
        Results::new("gemm", framework, gpu, cases)
    }

    fn case(sm_after: u32) -> Case<Shape> {
        let params = Shape { m: 64, dtype: "f16".to_owned() };
        Case::new(
            "gemm/f16/64".to_owned(),
            params,
            true,
            (state(2520), state(sm_after)),
            &series(&[3.0, 1.0, 2.0]),
        )
        .unwrap()
    }

    #[test]
    fn a_case_keeps_its_samples_in_order_and_summarizes_them() {
        let made = case(2520);

        assert_eq!(made.samples_us, [3.0, 1.0, 2.0]);
        assert!((made.median_us - 2.0).abs() < 1e-12);
        assert!((made.p10_us - 1.2).abs() < 1e-12 && (made.p90_us - 2.8).abs() < 1e-12);
    }

    #[test]
    fn a_case_records_whether_its_clocks_held() {
        assert!(case(2520).clocks.held);
        assert!(!case(2505).clocks.held);
    }

    #[test]
    fn a_case_without_samples_is_an_error() {
        let made = Case::new("x".to_owned(), (), true, (state(1), state(1)), &series(&[]));

        assert!(matches!(made, Err(ResultsError::Empty)));
    }

    #[test]
    fn results_are_written_in_the_documented_shape() {
        let json: serde_json::Value = serde_json::to_value(results(vec![case(2520)])).unwrap();

        assert_eq!(json["schema"], 1);
        assert_eq!(json["suite"], "gemm");
        assert_eq!(json["framework"]["commit"], "abc123");
        assert_eq!(json["environment"]["driver"], "595.112.07");
        let first = &json["cases"][0];
        assert_eq!(first["id"], "gemm/f16/64");
        assert_eq!(first["params"]["dtype"], "f16");
        assert_eq!(first["clocks"]["before"]["sm_mhz"], 2520);
        assert_eq!(first["clocks"]["after"]["events"][0], "sw_power_cap");
        assert_eq!(first["clocks"]["held"], true);
        assert_eq!(first["median_us"], 2.0);
    }

    #[test]
    fn written_results_read_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scratch").join("gemm.json");
        let written = results(vec![case(2520), case(2505)]);

        written.write(&path).unwrap();

        let read: Results<Shape> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(read, written);
    }

    /// Runs `git` in `directory`, ignoring any global hooks and signing.
    fn run_git(directory: &Path, arguments: &[&str]) {
        let isolated = ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"];
        let identity = ["-c", "user.name=bench", "-c", "user.email=bench@example.invalid"];
        let status = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(isolated)
            .args(identity)
            .args(arguments)
            .status()
            .unwrap();
        assert!(status.success(), "git {arguments:?}");
    }

    #[test]
    fn the_framework_is_read_from_the_repository_it_was_built_in() {
        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "-q"]);
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        run_git(dir.path(), &["add", "a.txt"]);
        run_git(dir.path(), &["commit", "-q", "-m", "a"]);

        let clean = Framework::ptxc(dir.path()).unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        let dirty = Framework::ptxc(dir.path()).unwrap();

        assert_eq!(clean.commit.len(), 12, "{clean:?}");
        assert!(clean.commit.chars().all(|c| c.is_ascii_hexdigit()), "{clean:?}");
        assert!(!clean.dirty && dirty.dirty);
    }

    #[test]
    fn a_directory_outside_any_repository_is_an_error() {
        let dir = tempfile::tempdir().unwrap();

        let outcome = Framework::ptxc(dir.path());

        assert!(matches!(outcome, Err(ResultsError::Git(_))), "{outcome:?}");
    }

    #[test]
    fn the_framework_is_ptxc_at_its_commit_dirty_if_anything_is_uncommitted() {
        let clean = Framework::ptxc_at("abc123def456\n", "");
        let dirty = Framework::ptxc_at("abc123def456\n", " M crates/bench/src/results.rs\n");

        assert_eq!(
            (clean.name.as_str(), clean.commit.as_str(), clean.dirty),
            ("ptxc", "abc123def456", false)
        );
        assert_eq!(clean.version, env!("CARGO_PKG_VERSION"));
        assert!(dirty.dirty);
    }
}
