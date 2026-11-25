//! Benchmark suites, and timing one of their cases.

pub(crate) mod conv;
pub(crate) mod gemm;
mod models;

use crate::clocks::{ClockError, GpuState};
use crate::results::{Case, ResultsError};
use crate::timing::time;
use ptxc_cuda::{Context, CudaError, Stream};

/// The context and stream a series is timed on.
#[derive(Clone, Copy)]
pub(crate) struct Timing<'a> {
    pub(crate) context: &'a Context,
    pub(crate) stream: &'a Stream,
}

impl Timing<'_> {
    /// The case `id` with `params`, verified, its series timed with
    /// `enqueue` between readings of the GPU's state taken after its
    /// warmup.
    pub(crate) fn case<P, E>(
        self,
        (id, params): (String, P),
        mut enqueue: impl FnMut() -> Result<(), E>,
    ) -> Result<Case<P>, E>
    where
        E: From<CudaError> + From<ClockError> + From<ResultsError>,
    {
        let read = || GpuState::read().map_err(E::from);
        let timed = time(self.context, self.stream, &mut enqueue, read)?;
        Ok(Case::new(id, params, true, (timed.before, timed.after), &timed.series)?)
    }
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
pub(crate) mod testing {
    use std::path::Path;
    use std::process::Command;

    /// A repository with one commit at `directory`, for a suite to record,
    /// ignoring any global hooks and signing.
    pub(crate) fn repository(directory: &Path) {
        let git = |arguments: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(directory)
                .args(["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"])
                .args(["-c", "user.name=bench", "-c", "user.email=bench@example.invalid"])
                .args(arguments)
                .status()
                .unwrap();
            assert!(status.success(), "git {arguments:?}");
        };
        git(&["init", "-q"]);
        git(&["commit", "-q", "--allow-empty", "-m", "empty"]);
    }
}
