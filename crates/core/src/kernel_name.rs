//! `KernelName`: the name of a generated kernel, as PTX declares its entry.

use std::fmt;

/// The name of a kernel's `.entry` in its PTX module.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct KernelName(String);

impl KernelName {
    /// Wraps a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
