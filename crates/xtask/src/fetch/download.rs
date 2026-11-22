//! Downloading an archive into memory and hashing it.

use sha2::{Digest, Sha256};

/// 64 MiB, larger than any archive we pin; a response past it is an error,
/// not a truncated archive.
const MAX_ARCHIVE_BYTES: u64 = 67_108_864;

/// The body of a GET of `url`; HTTP error statuses are errors.
pub(super) fn download(url: &str) -> Result<Vec<u8>, ureq::Error> {
    ureq::get(url).call()?.body_mut().with_config().limit(MAX_ARCHIVE_BYTES).read_to_vec()
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
