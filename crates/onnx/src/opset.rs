//! `OpsetVersion` and per-op version dispatch.

use crate::error::{ImportError, ImportErrorKind, Unsupported};
use ptxc_onnx_proto::OperatorSetIdProto;

/// The lowest default-domain opset ptxc imports (D28).
const MINIMUM: i64 = 13;

/// The default-domain opset a model imports, at least 13.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct OpsetVersion(i64);

impl OpsetVersion {
    /// Whether this opset is `version` or later, for ops whose attributes
    /// became inputs (or changed meaning) at `version`.
    pub(crate) const fn at_least(self, version: i64) -> bool {
        self.0 >= version
    }

    /// The default domain's version from a model's opset imports, if it
    /// imports the default domain. Models of only other domains import none;
    /// their ops are reported as unsupported when used.
    pub(crate) fn from_imports(
        imports: &[OperatorSetIdProto],
    ) -> Result<Option<Self>, ImportError> {
        let default = imports
            .iter()
            .find(|import| matches!(import.domain.as_deref(), None | Some("" | "ai.onnx")))
            .map(|import| {
                import.version.ok_or(ImportErrorKind::Missing("the default domain's opset version"))
            })
            .transpose()?;
        match default {
            Some(version) if version < MINIMUM => Err(Unsupported::Opset(version).into()),
            version => Ok(version.map(Self)),
        }
    }
}
