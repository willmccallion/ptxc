//! Gzipped tarballs built in memory for the fetch tests.

use flate2::Compression;
use flate2::write::GzEncoder;
use tar::{Builder, EntryType, Header};

/// A gzipped tarball with one regular file per `(path, contents)`, its path
/// written verbatim so tests can include hostile names.
pub(super) fn tarball(files: &[(&str, &str)]) -> Vec<u8> {
    let mut builder = Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
    for (path, contents) in files {
        let mut header = file_header(path, EntryType::Regular);
        header.set_size(contents.len().try_into().unwrap());
        header.set_cksum();
        builder.append(&header, contents.as_bytes()).unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

/// A header naming `path` verbatim, with the given entry type.
pub(super) fn file_header(path: &str, kind: EntryType) -> Header {
    let mut header = Header::new_old();
    header.as_old_mut().name[..path.len()].copy_from_slice(path.as_bytes());
    header.set_entry_type(kind);
    header.set_mode(0o644);
    header
}
