//! Downloading an archive into memory and hashing it.

use crate::pin::ArchiveSize;
use sha2::{Digest, Sha256};

/// The body of a GET of `url`, at most `limit`; HTTP error statuses, and a
/// body past the limit, are errors rather than a truncated archive.
pub(super) fn download(url: &str, limit: ArchiveSize) -> Result<Vec<u8>, ureq::Error> {
    ureq::get(url).call()?.body_mut().with_config().limit(limit.0).read_to_vec()
}

/// The SHA-256 of `bytes` in lowercase hex.
pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        hex.push(char::from(DIGITS[usize::from(byte >> 4)]));
        hex.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    hex
}
