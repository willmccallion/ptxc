//! Unpacking one directory of a gzipped tarball.

use super::FetchError;
use flate2::read::GzDecoder;
use std::io::Read;
use std::path::{Component, Path};
use tar::{Archive, Entry, EntryType};

/// What an unpacked entry was.
#[derive(Debug, PartialEq, Eq)]
enum Unpacked {
    File,
    Directory,
}

/// Unpacks every entry under `prefix` in the gzipped tarball `archive` into
/// `destination`, with `prefix` removed from its path. Entries elsewhere are
/// skipped; it is an error if no file is under `prefix`.
pub(super) fn extract_prefix(
    archive: &[u8],
    prefix: &'static str,
    destination: &Path,
) -> Result<(), FetchError> {
    let mut tarball = Archive::new(GzDecoder::new(archive));
    let mut has_files = false;
    for entry in tarball.entries().map_err(FetchError::ReadArchive)? {
        let mut entry = entry.map_err(FetchError::ReadArchive)?;
        let path = entry.path().map_err(FetchError::ReadArchive)?.into_owned();
        let Ok(relative) = path.strip_prefix(prefix) else { continue };
        has_files |= unpack_entry(&mut entry, relative, destination)? == Unpacked::File;
    }
    if has_files { Ok(()) } else { Err(FetchError::NothingUnderPrefix { prefix }) }
}

/// Unpacks a file or directory entry to `destination/relative`.
fn unpack_entry<R: Read>(
    entry: &mut Entry<'_, R>,
    relative: &Path,
    destination: &Path,
) -> Result<Unpacked, FetchError> {
    if !relative.components().all(|component| matches!(component, Component::Normal(_))) {
        return Err(FetchError::UnsafePath(relative.to_owned()));
    }
    let target = destination.join(relative);
    let write_error = |source| FetchError::Write { path: target.clone(), source };
    match entry.header().entry_type() {
        EntryType::Directory => {
            std::fs::create_dir_all(&target).map_err(write_error)?;
            Ok(Unpacked::Directory)
        }
        EntryType::Regular => {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(write_error)?;
            }
            let _: tar::Unpacked = entry.unpack(&target).map_err(write_error)?;
            Ok(Unpacked::File)
        }
        kind => Err(FetchError::UnsupportedEntry {
            path: relative.to_owned(),
            kind: format!("{kind:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::test_archive::{file_header, tarball};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tar::Builder;

    const PREFIX: &str = "onnx-1.22.0/onnx/backend/test/data/node";

    #[test]
    fn unpacks_files_under_the_prefix_without_it() {
        let archive = tarball(&[
            (&format!("{PREFIX}/test_abs/model.onnx"), "model"),
            (&format!("{PREFIX}/test_abs/test_data_set_0/input_0.pb"), "input"),
        ]);
        let destination = tempfile::tempdir().unwrap();

        extract_prefix(&archive, PREFIX, destination.path()).unwrap();

        let read = |path: &str| std::fs::read_to_string(destination.path().join(path)).unwrap();
        assert_eq!(read("test_abs/model.onnx"), "model");
        assert_eq!(read("test_abs/test_data_set_0/input_0.pb"), "input");
    }

    #[test]
    fn skips_entries_outside_the_prefix() {
        let archive = tarball(&[
            ("onnx-1.22.0/setup.py", "setup"),
            (&format!("{PREFIX}_extra/test_x/model.onnx"), "sibling"),
            (&format!("{PREFIX}/test_abs/model.onnx"), "model"),
        ]);
        let destination = tempfile::tempdir().unwrap();

        extract_prefix(&archive, PREFIX, destination.path()).unwrap();

        let mut names: Vec<_> = std::fs::read_dir(destination.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, ["test_abs"]);
    }

    #[test]
    fn rejects_entries_that_climb_out_of_the_destination() {
        let archive = tarball(&[(&format!("{PREFIX}/../../escaped"), "x")]);
        let destination = tempfile::tempdir().unwrap();

        let error = extract_prefix(&archive, PREFIX, destination.path()).unwrap_err();

        assert!(matches!(error, FetchError::UnsafePath(_)), "{error}");
    }

    #[test]
    fn rejects_symlinks() {
        let mut builder = Builder::new(GzEncoder::new(Vec::new(), Compression::fast()));
        let mut header = file_header(&format!("{PREFIX}/link"), EntryType::Symlink);
        header.set_size(0);
        header.set_link_name("/etc/passwd").unwrap();
        header.set_cksum();
        builder.append(&header, std::io::empty()).unwrap();
        let archive = builder.into_inner().unwrap().finish().unwrap();
        let destination = tempfile::tempdir().unwrap();

        let error = extract_prefix(&archive, PREFIX, destination.path()).unwrap_err();

        assert!(matches!(error, FetchError::UnsupportedEntry { .. }), "{error}");
    }

    #[test]
    fn fails_when_nothing_is_under_the_prefix() {
        let archive = tarball(&[("onnx-1.22.0/setup.py", "setup")]);
        let destination = tempfile::tempdir().unwrap();

        let error = extract_prefix(&archive, PREFIX, destination.path()).unwrap_err();

        assert!(matches!(error, FetchError::NothingUnderPrefix { .. }), "{error}");
    }

    #[test]
    fn rejects_a_corrupt_archive() {
        let destination = tempfile::tempdir().unwrap();

        let error = extract_prefix(b"not gzip", PREFIX, destination.path()).unwrap_err();

        assert!(matches!(error, FetchError::ReadArchive(_)), "{error}");
    }
}
