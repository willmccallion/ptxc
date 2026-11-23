//! Typed attribute reading: each attribute is parsed once into the type the op needs.

use crate::error::{ImportError, ImportErrorKind, Unsupported};
use ptxc_onnx_proto::attribute_proto::AttributeType;
use ptxc_onnx_proto::{AttributeProto, TensorProto};

/// A node's attributes, read by name and type. Every attribute must be read:
/// one left over means the model relies on behaviour ptxc does not handle,
/// so `finish` reports it as unsupported rather than ignoring it.
pub(crate) struct NodeAttrs<'a> {
    op: &'a str,
    attrs: &'a [AttributeProto],
    read: Vec<bool>,
}

impl<'a> NodeAttrs<'a> {
    pub(crate) fn new(op: &'a str, attrs: &'a [AttributeProto]) -> Self {
        Self { op, attrs, read: vec![false; attrs.len()] }
    }

    /// The attribute `name`, checked to be of type `expected`.
    fn find(
        &mut self,
        name: &str,
        expected: AttributeType,
    ) -> Result<Option<&'a AttributeProto>, ImportError> {
        let Some(index) = self.attrs.iter().position(|attr| attr.name.as_deref() == Some(name))
        else {
            return Ok(None);
        };
        self.read[index] = true;
        let attr = &self.attrs[index];
        if attr.r#type != Some(expected as i32) {
            let problem =
                format!("{} attribute `{name}` is not of type {}", self.op, expected.as_str_name());
            return Err(ImportErrorKind::Invalid { field: "attribute type", problem }.into());
        }
        Ok(Some(attr))
    }

    pub(crate) fn int(&mut self, name: &str) -> Result<Option<i64>, ImportError> {
        Ok(self.find(name, AttributeType::Int)?.and_then(|attr| attr.i))
    }

    pub(crate) fn int_or(&mut self, name: &str, default: i64) -> Result<i64, ImportError> {
        Ok(self.int(name)?.unwrap_or(default))
    }

    pub(crate) fn float(&mut self, name: &str) -> Result<Option<f32>, ImportError> {
        Ok(self.find(name, AttributeType::Float)?.and_then(|attr| attr.f))
    }

    pub(crate) fn float_or(&mut self, name: &str, default: f32) -> Result<f32, ImportError> {
        Ok(self.float(name)?.unwrap_or(default))
    }

    pub(crate) fn ints(&mut self, name: &str) -> Result<Option<&'a [i64]>, ImportError> {
        Ok(self.find(name, AttributeType::Ints)?.map(|attr| attr.ints.as_slice()))
    }

    pub(crate) fn floats(&mut self, name: &str) -> Result<Option<&'a [f32]>, ImportError> {
        Ok(self.find(name, AttributeType::Floats)?.map(|attr| attr.floats.as_slice()))
    }

    pub(crate) fn string(&mut self, name: &str) -> Result<Option<String>, ImportError> {
        let Some(attr) = self.find(name, AttributeType::String)? else { return Ok(None) };
        let bytes = attr.s.as_deref().unwrap_or_default();
        String::from_utf8(bytes.to_vec()).map(Some).map_err(|_| {
            ImportErrorKind::Invalid {
                field: "attribute",
                problem: format!("`{name}` is not UTF-8"),
            }
            .into()
        })
    }

    pub(crate) fn string_or(&mut self, name: &str, default: &str) -> Result<String, ImportError> {
        Ok(self.string(name)?.unwrap_or_else(|| default.to_owned()))
    }

    pub(crate) fn tensor(&mut self, name: &str) -> Result<Option<&'a TensorProto>, ImportError> {
        Ok(self.find(name, AttributeType::Tensor)?.and_then(|attr| attr.t.as_ref()))
    }

    /// The unsupported-attribute error for this op's `attribute`, for values
    /// the importer reads but ptxc does not support.
    pub(crate) fn unsupported(&self, attribute: &str) -> ImportError {
        Unsupported::Attribute { op: self.op.to_owned(), attribute: attribute.to_owned() }.into()
    }

    /// Fails on the first attribute no importer read.
    pub(crate) fn finish(self) -> Result<(), ImportError> {
        match self.read.iter().position(|&read| !read) {
            Some(index) => {
                Err(self.unsupported(self.attrs[index].name.as_deref().unwrap_or_default()))
            }
            None => Ok(()),
        }
    }
}
