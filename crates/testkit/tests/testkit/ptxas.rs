use ptxc_testkit::ptxas::{PtxasError, assemble};

const EMPTY_KERNEL: &str =
    ".version 8.4\n.target sm_89\n.address_size 64\n\n.visible .entry k()\n{\n\tret;\n}\n";

#[test]
fn a_well_formed_module_assembles() {
    assemble(EMPTY_KERNEL, "sm_89").unwrap();
}

#[test]
fn a_malformed_module_is_rejected_with_ptxas_diagnostics() {
    let malformed = EMPTY_KERNEL.replace("ret;", "add.f32 %f0, %f1;");

    let error = assemble(&malformed, "sm_89").unwrap_err();

    let PtxasError::Rejected { diagnostics, text } = error else { panic!("{error}") };
    assert!(diagnostics.contains("error"), "{diagnostics}");
    assert_eq!(text, malformed);
}

#[test]
fn a_module_for_another_gpu_is_rejected() {
    let newer = EMPTY_KERNEL.replace(".target sm_89", ".target sm_90a");

    let error = assemble(&newer, "sm_89").unwrap_err();

    assert!(matches!(error, PtxasError::Rejected { .. }), "{error}");
}
