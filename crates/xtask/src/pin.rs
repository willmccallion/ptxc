//! The pinned ONNX release (D31): `onnx.proto` and the node tests both come
//! from it.

/// The ONNX release tag.
pub(crate) const ONNX_TAG: &str = "v1.22.0";

/// An archive's exact size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArchiveSize(pub(crate) u64);

/// How an archive is packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Packing {
    /// A gzipped tarball.
    TarGz,
    /// A plain tarball.
    Tar,
}

/// A downloadable archive, its expected SHA-256, how it is packed, and the
/// directory inside it that is unpacked.
#[derive(Debug)]
pub(crate) struct PinnedArchive {
    pub(crate) url: &'static str,
    /// Lowercase hex.
    pub(crate) sha256: &'static str,
    /// The archive's exact size; the download stops past it.
    pub(crate) size: ArchiveSize,
    pub(crate) packing: Packing,
    pub(crate) prefix: &'static str,
}

/// ONNX's source distribution on PyPI, which never replaces a published file.
pub(crate) const ONNX_SDIST: PinnedArchive = PinnedArchive {
    url: "https://files.pythonhosted.org/packages/04/19/8ea73a64b368b75fe339771a20a02bc61ea1f551484c9e3d9d0bfbd0450f/onnx-1.22.0.tar.gz",
    sha256: "ef40c0aaf0b643857ea9306fc7eddce17eaf9fb0407e4801f1fc5758443a38e0",
    size: ArchiveSize(12_024_721),
    packing: Packing::TarGz,
    prefix: "onnx-1.22.0/onnx/backend/test/data/node",
};

/// ImageNetV2's matched-frequency test set (MIT licence), an interim stand-in
/// for the ImageNet validation set (D35). The file is a plain tarball despite
/// its `.tar.gz` name; the digest is its Git LFS object id.
pub(crate) const IMAGENET_V2: PinnedArchive = PinnedArchive {
    url: "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/d626240be2538720e83103a0e1178d24aca8b12c/imagenetv2-matched-frequency.tar.gz",
    sha256: "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c",
    size: ArchiveSize(1_264_079_360),
    packing: Packing::Tar,
    prefix: "imagenetv2-matched-frequency-format-val",
};
