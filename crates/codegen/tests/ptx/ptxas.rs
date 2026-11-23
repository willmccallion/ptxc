//! Assembles emitted modules with ptxas.

use ptxc_codegen::PtxModule;

/// Assembles `module` for its target, panicking with ptxas's diagnostics.
pub(crate) fn assemble(module: &PtxModule) {
    let gpu = crate::target().sm().to_string();
    if let Err(error) = ptxc_testkit::ptxas::assemble(&module.to_string(), &gpu) {
        panic!("{error}");
    }
}
