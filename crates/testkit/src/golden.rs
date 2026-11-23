//! Golden files under a crate's `tests/golden`, rewritten with `PTXC_BLESS=1`.

#[cfg(test)]
mod tests;

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// The environment variable that rewrites golden files instead of checking
/// them.
const BLESS: &str = "PTXC_BLESS";

/// The golden directory, relative to the crate under test, which is where
/// `cargo test` runs tests.
const GOLDEN_DIR: &str = "tests/golden";

/// Whether golden files are compared or rewritten.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Compare,
    Bless,
}

impl Mode {
    /// The mode `PTXC_BLESS` asks for: blessing only when it is `1`.
    fn from_bless(value: Option<&std::ffi::OsStr>) -> Self {
        if value.is_some_and(|value| value == "1") { Self::Bless } else { Self::Compare }
    }
}

/// A line of a text, counting from 1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LineNumber(pub usize);

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The first line where a result differs from its golden file. A side that
/// has ended has no line.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Difference {
    /// Where.
    pub line: LineNumber,
    /// The golden file's line.
    pub expected: Option<String>,
    /// The result's line.
    pub actual: Option<String>,
}

impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show = |line: &Option<String>| {
            line.as_ref().map_or_else(|| "(end of text)".to_owned(), |l| format!("{l:?}"))
        };
        write!(
            f,
            "line {}:\n  expected {}\n       got {}",
            self.line,
            show(&self.expected),
            show(&self.actual)
        )
    }
}

/// Why a golden check failed. Its `Debug` form is its message, so a test's
/// `unwrap()` shows the difference readably.
#[derive(Error)]
pub enum GoldenError {
    /// The golden file does not exist.
    #[error("{} does not exist; run with {BLESS}=1 to create it", .file.display())]
    Missing {
        /// The golden file.
        file: PathBuf,
    },
    /// The result differs from the golden file.
    #[error("{} differs from the result at {difference}\nrun with {BLESS}=1 to accept the result, and review the diff", .file.display())]
    Differs {
        /// The golden file.
        file: PathBuf,
        /// The first difference.
        difference: Difference,
    },
    /// Reading or writing the golden file failed.
    #[error("{}: {source}", .file.display())]
    Io {
        /// The golden file.
        file: PathBuf,
        /// The failure.
        source: io::Error,
    },
}

impl fmt::Debug for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Compares `actual` with `tests/golden/<path>` of the crate under test, or
/// writes it there when `PTXC_BLESS=1`.
pub fn check(path: impl AsRef<Path>, actual: &str) -> Result<(), GoldenError> {
    let root = std::env::current_dir()
        .map_err(|source| GoldenError::Io { file: PathBuf::from(GOLDEN_DIR), source })?
        .join(GOLDEN_DIR);
    let mode = Mode::from_bless(std::env::var_os(BLESS).as_deref());
    check_in(&root, path.as_ref(), actual, mode)
}

fn check_in(root: &Path, path: &Path, actual: &str, mode: Mode) -> Result<(), GoldenError> {
    let file = root.join(path);
    let io = |source| GoldenError::Io { file: file.clone(), source };
    if mode == Mode::Bless {
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        return std::fs::write(&file, actual).map_err(io);
    }
    let expected = match std::fs::read_to_string(&file) {
        Ok(expected) => expected,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(GoldenError::Missing { file });
        }
        Err(error) => return Err(io(error)),
    };
    first_difference(&expected, actual)
        .map_or(Ok(()), |difference| Err(GoldenError::Differs { file, difference }))
}

/// The first line where the texts differ, or `None` if they are equal.
fn first_difference(expected: &str, actual: &str) -> Option<Difference> {
    if expected == actual {
        return None;
    }
    let (mut expected_lines, mut actual_lines) = (expected.split('\n'), actual.split('\n'));
    (1..).find_map(|line| {
        let (e, a) = (expected_lines.next(), actual_lines.next());
        (e != a).then(|| Difference {
            line: LineNumber(line),
            expected: e.map(str::to_owned),
            actual: a.map(str::to_owned),
        })
    })
}
