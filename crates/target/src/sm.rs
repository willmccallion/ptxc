//! `SmVersion` and `PtxIsaVersion`.

use std::fmt;

/// A compute capability, written `sm_<major><minor>` in PTX.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SmVersion {
    major: u8,
    minor: u8,
}

impl SmVersion {
    /// Ada: the RTX 40 series, including the RTX 4060 Ti ptxc is developed on.
    pub const SM_89: Self = Self::new(8, 9);

    /// Compute capability `major.minor`.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// The major version.
    pub const fn major(self) -> u8 {
        self.major
    }

    /// The minor version.
    pub const fn minor(self) -> u8 {
        self.minor
    }
}

impl fmt::Display for SmVersion {
    /// The PTX `.target` name, such as `sm_89`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

/// A PTX ISA version, written `.version <major>.<minor>` in PTX.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PtxIsaVersion {
    major: u8,
    minor: u8,
}

impl PtxIsaVersion {
    /// PTX ISA `major.minor`.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// The major version.
    pub const fn major(self) -> u8 {
        self.major
    }

    /// The minor version.
    pub const fn minor(self) -> u8 {
        self.minor
    }
}

impl fmt::Display for PtxIsaVersion {
    /// The `.version` operand, such as `7.8`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}
