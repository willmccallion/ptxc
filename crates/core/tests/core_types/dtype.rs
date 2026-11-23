//! Element type names.

use ptxc_core::DType;

#[test]
fn every_element_type_is_found_by_its_name() {
    for dtype in DType::ALL {
        assert_eq!(DType::from_name(dtype.name()), Some(dtype));
    }
}

#[test]
fn names_are_the_text_formats_words() {
    assert_eq!(DType::F8E4M3Fn.name(), "f8e4m3fn");
    assert_eq!(DType::Bf16.name(), "bf16");
    assert_eq!(DType::from_name("float32"), None);
}
