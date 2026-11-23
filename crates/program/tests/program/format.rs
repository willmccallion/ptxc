use crate::sample;
use ptxc_program::{FORMAT_VERSION, FormatError, Program, load, save};

fn saved() -> Vec<u8> {
    save(&Program::new(sample()).unwrap()).unwrap()
}

/// The saved bytes with the header's JSON rewritten by `edit`.
fn with_header(edit: impl FnOnce(String) -> String) -> Vec<u8> {
    let bytes = saved();
    let length = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())).unwrap();
    let header = edit(String::from_utf8(bytes[16..16 + length].to_vec()).unwrap());
    let mut rewritten = bytes[..8].to_vec();
    rewritten.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
    rewritten.extend_from_slice(header.as_bytes());
    rewritten.extend_from_slice(&bytes[16 + length..]);
    rewritten
}

#[test]
fn a_program_round_trips_through_its_file() {
    let program = Program::new(sample()).unwrap();

    let loaded = load(&save(&program).unwrap()).unwrap();

    assert_eq!(loaded, program);
}

#[test]
fn the_file_starts_with_the_magic_and_version_and_ends_with_the_blob() {
    let bytes = saved();

    assert_eq!(&bytes[..4], b"PTXC");
    assert_eq!(bytes[4..8], FORMAT_VERSION.to_le_bytes());
    assert!(bytes.ends_with(&[7; 16]));
}

#[test]
fn foreign_old_and_truncated_files_are_rejected() {
    let mut foreign = saved();
    foreign[0] = b'X';
    let mut old = saved();
    old[4] = 0;
    let truncated = &saved()[..20];

    assert!(matches!(load(&foreign), Err(FormatError::Magic)));
    assert!(matches!(load(&old), Err(FormatError::Version(0))));
    assert!(matches!(load(truncated), Err(FormatError::Truncated)));
    assert!(matches!(load(b"PTX"), Err(FormatError::Truncated)));
}

#[test]
fn a_header_with_unknown_fields_or_types_is_rejected() {
    let unknown_field = with_header(|h| h.replacen('{', "{\"extra\":1,", 1));
    let unknown_type = with_header(|h| h.replace("\"f32\"", "\"f31\""));
    let bad_binding = with_header(|h| h.replace("\"buffer\":3", "\"buffer\":1"));

    assert!(matches!(load(&unknown_field), Err(FormatError::Header(_))));
    assert!(matches!(load(&unknown_type), Err(FormatError::DType(name)) if name == "f31"));
    assert!(matches!(load(&bad_binding), Err(FormatError::Program(_))));
}
