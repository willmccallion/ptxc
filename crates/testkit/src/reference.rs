//! Input sets and the reference outputs `ptxc-baselines` caches for them,
//! keyed by model, input set and tool version (TESTING.md).

use crate::data::{DataError, baselines_command, workspace_root};
use crate::onnx_tests::{LoadError, tensors};
use ptxc_core::{Dim, HostTensor};
use sha2::{Digest, Sha256};
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// The seed of a random input set's generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed(u64);

impl Seed {
    /// The seed `value`.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The seed's value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Where an image input set's images come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// ImageNetV2's matched-frequency images, evenly spaced (D35).
    ImageNetV2,
    /// The first of D25's ImageNet validation images.
    ImageNet,
}

/// An input set `ptxc-baselines` makes, under `data/inputs/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputSet {
    /// Images preprocessed for ResNet-50, as one batch.
    Images {
        /// Where they come from.
        source: ImageSource,
        /// How many, which is the batch size.
        count: Dim,
    },
    /// Seeded random inputs that are valid for a target model.
    Random {
        /// The target model.
        model: &'static str,
        /// The batch size.
        batch: Dim,
        /// The sequence length, for models that have one.
        sequence: Dim,
        /// The generator's seed.
        seed: Seed,
    },
}

impl InputSet {
    /// The set's directory name, as `ptxc-baselines` writes it.
    pub fn name(&self) -> String {
        match self {
            Self::Images { source: ImageSource::ImageNetV2, count } => {
                format!("imagenetv2-{}", count.get())
            }
            Self::Images { source: ImageSource::ImageNet, count } => {
                format!("imagenet-val-{}", count.get())
            }
            Self::Random { model, batch, sequence, seed } => format!(
                "{model}-batch{}-sequence{}-seed{}",
                batch.get(),
                sequence.get(),
                seed.get()
            ),
        }
    }

    /// The command that makes the set.
    pub fn command(&self) -> String {
        match self {
            Self::Images { source, count } => {
                let source = match source {
                    ImageSource::ImageNetV2 => "imagenetv2",
                    ImageSource::ImageNet => "imagenet",
                };
                baselines_command(&format!("images --source {source} --count {}", count.get()))
            }
            Self::Random { model, batch, sequence, seed } => baselines_command(&format!(
                "random --model {model} --batch {} --sequence {} --seed {}",
                batch.get(),
                sequence.get(),
                seed.get()
            )),
        }
    }

    /// The set's directory, which must exist.
    pub fn directory(&self) -> Result<PathBuf, DataError> {
        let path = workspace_root()?.join("data").join("inputs").join(self.name());
        if path.is_dir() {
            Ok(path)
        } else {
            Err(DataError::NotFetched { path, fetch: self.command() })
        }
    }
}

/// Why reference outputs could not be read.
#[derive(Debug, Error)]
pub enum ReferenceError {
    /// A model, input set or reference is missing.
    #[error(transparent)]
    Missing(#[from] DataError),
    /// Several tool versions have cached outputs, so which to use is unclear.
    #[error("{} holds outputs from several tools: {}", .directory.display(), .found.join(", "))]
    Ambiguous {
        /// The input set's reference directory.
        directory: PathBuf,
        /// The tool directories found.
        found: Vec<String>,
    },
    /// A file could not be read.
    #[error("{}: {source}", .path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
    /// A cached tensor could not be loaded.
    #[error(transparent)]
    Load(#[from] LoadError),
}

/// Bytes as lowercase hex.
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&byte| [DIGITS[usize::from(byte >> 4)], DIGITS[usize::from(byte & 0xf)]])
        .map(char::from)
        .collect()
}

/// A file's length as `u64`, which holds any `usize`.
const fn length(bytes: &[u8]) -> u64 {
    bytes.len() as u64
}

/// An input set's id: SHA-256, in hex, over each `input_k.pb` in order, its
/// length as 8 little-endian bytes and then its bytes, as `ptxc-baselines`
/// computes it.
pub fn input_set_id(directory: &Path) -> Result<String, ReferenceError> {
    let mut digest = Sha256::new();
    for k in 0.. {
        let path = directory.join(format!("input_{k}.pb"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound && k > 0 => break,
            Err(source) => return Err(ReferenceError::Io { path, source }),
        };
        digest.update(length(&bytes).to_le_bytes());
        digest.update(&bytes);
    }
    Ok(hex(&digest.finalize()))
}

/// The input set's tensors, in input order.
pub fn inputs(directory: &Path) -> Result<Vec<HostTensor>, ReferenceError> {
    Ok(tensors(directory, "input")?)
}

/// ONNX Runtime's cached outputs for `model` on the input set in `inputs`,
/// from the workspace's `data/reference/`.
pub fn ort_reference(model: &Path, inputs: &Path) -> Result<Vec<HostTensor>, ReferenceError> {
    let root = workspace_root()?.join("data").join("reference");
    ort_reference_under(&root, model, inputs)
}

/// ONNX Runtime's cached outputs for `model` on the input set in `inputs`,
/// from `root/<model sha256>/<input set id>/ort-<version>/`. Exactly one
/// version must be cached.
pub fn ort_reference_under(
    root: &Path,
    model: &Path,
    inputs: &Path,
) -> Result<Vec<HostTensor>, ReferenceError> {
    let bytes = std::fs::read(model)
        .map_err(|source| ReferenceError::Io { path: model.to_path_buf(), source })?;
    let directory = root.join(hex(&Sha256::digest(&bytes))).join(input_set_id(inputs)?);
    let missing = || {
        let arguments =
            format!("reference --model {} --inputs {}", model.display(), inputs.display());
        let path = directory.join("ort-<version>");
        ReferenceError::Missing(DataError::NotFetched {
            path,
            fetch: baselines_command(&arguments),
        })
    };
    let found: Vec<String> = std::fs::read_dir(&directory)
        .map_err(|_| missing())?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("ort-"))
        .collect();
    match found.as_slice() {
        [] => Err(missing()),
        [tool] => Ok(tensors(&directory.join(tool), "output")?),
        _ => Err(ReferenceError::Ambiguous { directory, found }),
    }
}
