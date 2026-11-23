//! How SM and PTX ISA versions print and compare.

use ptxc_target::{PtxIsaVersion, SmVersion};

#[test]
fn versions_print_as_ptx_writes_them() {
    assert_eq!(SmVersion::SM_89.to_string(), "sm_89");
    assert_eq!(SmVersion::new(10, 0).to_string(), "sm_100");
    assert_eq!(PtxIsaVersion::new(7, 8).to_string(), "7.8");
}

#[test]
fn isa_versions_order_by_major_then_minor() {
    assert!(PtxIsaVersion::new(7, 8) < PtxIsaVersion::new(8, 0));
    assert!(PtxIsaVersion::new(8, 0) < PtxIsaVersion::new(8, 10));
    assert!(PtxIsaVersion::new(9, 2) > PtxIsaVersion::new(8, 8));
}

#[test]
fn sm_versions_expose_their_parts() {
    assert_eq!((SmVersion::SM_89.major(), SmVersion::SM_89.minor()), (8, 9));
    assert_eq!((PtxIsaVersion::new(7, 8).major(), PtxIsaVersion::new(7, 8).minor()), (7, 8));
}
