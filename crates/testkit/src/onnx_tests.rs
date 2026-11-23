//! Loads ONNX backend test cases: a model and its input and output protobufs.

use prost::Message;
use ptxc_core::HostTensor;
use ptxc_graph::Graph;
use ptxc_onnx::{ImportError, ImportErrorKind, ImportOptions, import_model, tensor_from_bytes};
use ptxc_onnx_proto::ModelProto;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Why a test case could not be loaded.
#[derive(Debug, Error)]
pub enum LoadError {
    /// A file or directory could not be read.
    #[error("{}: {source}", .path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The failure.
        source: io::Error,
    },
    /// The model is not a valid `ModelProto`.
    #[error("{}: {source}", .path.display())]
    Model {
        /// The model file.
        path: PathBuf,
        /// The decoding failure.
        source: prost::DecodeError,
    },
    /// A test tensor could not be imported, perhaps because its element
    /// type is unsupported.
    #[error("{}: {source}", .path.display())]
    Tensor {
        /// The tensor file.
        path: PathBuf,
        /// The import failure.
        source: Box<ImportError>,
    },
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> LoadError + '_ {
    move |source| LoadError::Io { path: path.to_path_buf(), source }
}

/// One test case: its name, its model, and where its data sets are.
#[derive(Debug)]
pub struct TestCase {
    name: String,
    dir: PathBuf,
    model: Vec<u8>,
    /// The graph inputs that are not initializers, in order; input file `k`
    /// holds the `k`th.
    inputs: Vec<String>,
}

/// One data set: the inputs to feed and the outputs to expect.
#[derive(Debug)]
pub struct DataSet {
    /// One tensor per graph input that is not an initializer, in order.
    pub inputs: Vec<HostTensor>,
    /// One tensor per graph output, in order.
    pub outputs: Vec<HostTensor>,
}

/// A case's model imported for one data set, and the inputs it still takes.
#[derive(Debug)]
pub struct Runnable {
    /// The graph.
    pub graph: Graph,
    /// The data set's inputs that are still graph inputs, in order.
    pub inputs: Vec<HostTensor>,
}

/// Every test case under `root`, in name order.
pub fn cases(root: &Path) -> Result<Vec<TestCase>, LoadError> {
    let mut dirs = subdirectories(root)?;
    dirs.sort();
    dirs.iter().map(|dir| TestCase::load(dir)).collect()
}

fn subdirectories(dir: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(io_error(dir))? {
        let path = entry.map_err(io_error(dir))?.path();
        if path.is_dir() {
            found.push(path);
        }
    }
    Ok(found)
}

impl TestCase {
    /// Loads the case in `dir`: `model.onnx` and `test_data_set_*`.
    pub fn load(dir: &Path) -> Result<Self, LoadError> {
        let path = dir.join("model.onnx");
        let model = std::fs::read(&path).map_err(io_error(&path))?;
        let proto = ModelProto::decode(model.as_slice())
            .map_err(|source| LoadError::Model { path, source })?;
        let graph = proto.graph.unwrap_or_default();
        let initializers: Vec<&str> =
            graph.initializer.iter().filter_map(|tensor| tensor.name.as_deref()).collect();
        let inputs = graph
            .input
            .iter()
            .filter_map(|info| info.name.clone())
            .filter(|name| !initializers.contains(&name.as_str()))
            .collect();
        let name =
            dir.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        Ok(Self { name, dir: dir.to_path_buf(), model, inputs })
    }

    /// The case's name, its directory's.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The serialised model.
    pub fn model(&self) -> &[u8] {
        &self.model
    }

    /// Every data set, in name order.
    pub fn data_sets(&self) -> Result<Vec<DataSet>, LoadError> {
        let mut dirs: Vec<PathBuf> = subdirectories(&self.dir)?
            .into_iter()
            .filter(|dir| {
                dir.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("test_data_set_"))
            })
            .collect();
        dirs.sort();
        dirs.iter()
            .map(|dir| {
                Ok(DataSet { inputs: tensors(dir, "input")?, outputs: tensors(dir, "output")? })
            })
            .collect()
    }

    /// Imports the model for `data_set`, binding as constants the inputs a
    /// shape depends on, as the importer asks (`NeedsConstant`).
    pub fn import(&self, data_set: &DataSet) -> Result<Runnable, ImportError> {
        let mut options = ImportOptions::default();
        loop {
            match import_model(&self.model, &options) {
                Ok(graph) => {
                    return Ok(Runnable { graph, inputs: self.unbound(data_set, &options) });
                }
                Err(error) => match error.kind() {
                    ImportErrorKind::NeedsConstant { inputs, .. }
                        if self.binds_more(inputs, &options) =>
                    {
                        for name in inputs {
                            if let Some(tensor) = self.input(data_set, name) {
                                let _replaced =
                                    options.constant_inputs.insert(name.clone(), tensor.clone());
                            }
                        }
                    }
                    _ => return Err(error),
                },
            }
        }
    }

    /// Whether binding `names` would bind an input not yet bound.
    fn binds_more(&self, names: &[String], options: &ImportOptions) -> bool {
        names
            .iter()
            .any(|name| self.inputs.contains(name) && !options.constant_inputs.contains_key(name))
    }

    fn input<'d>(&self, data_set: &'d DataSet, name: &str) -> Option<&'d HostTensor> {
        self.inputs.iter().position(|input| input == name).and_then(|k| data_set.inputs.get(k))
    }

    fn unbound(&self, data_set: &DataSet, options: &ImportOptions) -> Vec<HostTensor> {
        self.inputs
            .iter()
            .zip(&data_set.inputs)
            .filter(|(name, _)| !options.constant_inputs.contains_key(*name))
            .map(|(_, tensor)| tensor.clone())
            .collect()
    }
}

/// `<prefix>_0.pb`, `<prefix>_1.pb`, … in `dir`, up to the first missing.
fn tensors(dir: &Path, prefix: &str) -> Result<Vec<HostTensor>, LoadError> {
    let mut found = Vec::new();
    for k in 0.. {
        let path = dir.join(format!("{prefix}_{k}.pb"));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(source) => return Err(LoadError::Io { path, source }),
        };
        found.push(
            tensor_from_bytes(&bytes)
                .map_err(|source| LoadError::Tensor { path, source: Box::new(source) })?,
        );
    }
    Ok(found)
}
