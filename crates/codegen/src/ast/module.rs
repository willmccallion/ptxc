//! `PtxModule`: version, target, address size and functions.

use super::function::Function;
use ptxc_target::{PtxIsaVersion, SmVersion};

/// A PTX module: what it targets and the functions it holds.
#[derive(Clone, PartialEq, Debug)]
pub struct PtxModule {
    pub(crate) version: PtxIsaVersion,
    pub(crate) target: SmVersion,
    pub(crate) functions: Vec<Function>,
}
