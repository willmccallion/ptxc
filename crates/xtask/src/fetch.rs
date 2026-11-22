//! `cargo xtask fetch onnx-tests`: downloads the pinned ONNX release, checks
//! its SHA-256, and unpacks its node tests into `data/onnx/<tag>/node/`.

mod download;
mod extract;
#[cfg(test)]
mod test_archive;

use crate::pin::{ONNX_SDIST, ONNX_TAG, PinnedArchive};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// A failure to fetch pinned test data.
#[derive(Debug, Error)]
pub(crate) enum FetchError {
    #[error("downloading {url}: {source}")]
    Download { url: &'static str, source: Box<ureq::Error> },
    #[error("{url} has SHA-256 {actual}, but {expected} is pinned")]
    Checksum { url: &'static str, expected: &'static str, actual: String },
    #[error("reading the archive: {0}")]
    ReadArchive(std::io::Error),
    #[error("writing {}: {source}", .path.display())]
    Write { path: PathBuf, source: std::io::Error },
    #[error("archive entry {} leaves its directory", .0.display())]
    UnsafePath(PathBuf),
    #[error("archive entry {} is a {kind}, not a file or directory", .path.display())]
    UnsupportedEntry { path: PathBuf, kind: String },
    #[error("the archive has no files under {prefix}")]
    NothingUnderPrefix { prefix: &'static str },
}

/// Fetches the node tests of the pinned ONNX release, unless they are
/// already in place.
pub(crate) fn onnx_tests(workspace: &Path) -> Result<(), FetchError> {
    let relative = Path::new("data/onnx").join(ONNX_TAG).join("node");
    let destination = workspace.join(&relative);
    if destination.is_dir() {
        eprintln!("{} already holds the ONNX {ONNX_TAG} node tests", relative.display());
        return Ok(());
    }
    eprintln!("downloading {}", ONNX_SDIST.url);
    let archive = download::download(ONNX_SDIST.url)
        .map_err(|source| FetchError::Download { url: ONNX_SDIST.url, source: Box::new(source) })?;
    verify_checksum(&archive, &ONNX_SDIST)?;
    unpack_atomically(&archive, ONNX_SDIST.node_tests_prefix, &destination)?;
    eprintln!("unpacked the ONNX {ONNX_TAG} node tests into {}", relative.display());
    Ok(())
}

/// Checks `archive` against the pinned SHA-256.
fn verify_checksum(archive: &[u8], pin: &PinnedArchive) -> Result<(), FetchError> {
    let actual = download::sha256_hex(archive);
    if actual != pin.sha256 {
        return Err(FetchError::Checksum { url: pin.url, expected: pin.sha256, actual });
    }
    Ok(())
}

/// Unpacks into a staging directory beside `destination` and renames it into
/// place, so an interrupted fetch never leaves a partial `destination`.
fn unpack_atomically(
    archive: &[u8],
    prefix: &'static str,
    destination: &Path,
) -> Result<(), FetchError> {
    let staging = destination.with_extension("partial");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .map_err(|source| FetchError::Write { path: staging.clone(), source })?;
    }
    extract::extract_prefix(archive, prefix, &staging)?;
    std::fs::rename(&staging, destination)
        .map_err(|source| FetchError::Write { path: destination.to_owned(), source })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIN: PinnedArchive = PinnedArchive {
        url: "https://example.invalid/archive.tar.gz",
        // SHA-256 of "abc", from FIPS 180-2.
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        node_tests_prefix: "node",
    };

    #[test]
    fn matching_archive_passes_the_checksum() {
        assert!(verify_checksum(b"abc", &PIN).is_ok());
    }

    #[test]
    fn changed_archive_fails_the_checksum_with_both_digests() {
        let error = verify_checksum(b"abd", &PIN).unwrap_err();

        let FetchError::Checksum { expected, actual, .. } = error else { panic!("{error}") };
        assert_eq!(expected, PIN.sha256);
        assert_eq!(actual, download::sha256_hex(b"abd"));
        assert_ne!(actual, expected);
    }

    #[test]
    fn unpacking_creates_the_destination_with_the_files() {
        let workspace = tempfile::tempdir().unwrap();
        let destination = workspace.path().join("node");
        let archive = test_archive::tarball(&[("pkg/node/test_abs/model.onnx", "model")]);

        unpack_atomically(&archive, "pkg/node", &destination).unwrap();

        let model = std::fs::read_to_string(destination.join("test_abs/model.onnx")).unwrap();
        assert_eq!(model, "model");
        assert!(!destination.with_extension("partial").exists());
    }

    #[test]
    fn unpacking_replaces_a_leftover_staging_directory() {
        let workspace = tempfile::tempdir().unwrap();
        let destination = workspace.path().join("node");
        let staging = destination.with_extension("partial");
        std::fs::create_dir(&staging).unwrap();
        std::fs::write(staging.join("leftover"), "stale").unwrap();
        let archive = test_archive::tarball(&[("pkg/node/test_abs/model.onnx", "model")]);

        unpack_atomically(&archive, "pkg/node", &destination).unwrap();

        assert!(destination.join("test_abs/model.onnx").exists());
        assert!(!destination.join("leftover").exists());
    }

    #[test]
    fn a_failed_unpack_leaves_no_destination() {
        let workspace = tempfile::tempdir().unwrap();
        let destination = workspace.path().join("node");
        let archive = test_archive::tarball(&[("pkg/other/file", "x")]);

        let error = unpack_atomically(&archive, "pkg/node", &destination).unwrap_err();

        assert!(matches!(error, FetchError::NothingUnderPrefix { .. }), "{error}");
        assert!(!destination.exists());
    }

    #[test]
    fn pinned_sdist_digest_is_lowercase_sha256_hex() {
        let digest = ONNX_SDIST.sha256;

        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    }
}
