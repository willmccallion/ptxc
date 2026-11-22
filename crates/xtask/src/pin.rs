//! The pinned ONNX release (D31): `onnx.proto` and the node tests both come
//! from it.

/// The ONNX release tag.
pub(crate) const ONNX_TAG: &str = "v1.22.0";

/// A downloadable archive, its expected SHA-256 and where the node tests
/// sit inside it.
#[derive(Debug)]
pub(crate) struct PinnedArchive {
    pub(crate) url: &'static str,
    /// Lowercase hex, as PyPI publishes it.
    pub(crate) sha256: &'static str,
    pub(crate) node_tests_prefix: &'static str,
}

/// ONNX's source distribution on PyPI, which never replaces a published file.
pub(crate) const ONNX_SDIST: PinnedArchive = PinnedArchive {
    url: "https://files.pythonhosted.org/packages/04/19/8ea73a64b368b75fe339771a20a02bc61ea1f551484c9e3d9d0bfbd0450f/onnx-1.22.0.tar.gz",
    sha256: "ef40c0aaf0b643857ea9306fc7eddce17eaf9fb0407e4801f1fc5758443a38e0",
    node_tests_prefix: "onnx-1.22.0/onnx/backend/test/data/node",
};
