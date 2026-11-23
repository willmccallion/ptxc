//! Per-family node importers, mirroring `ptxc_graph::ops`, plus the ops that
//! become constants or aliases at import.

mod activation;
mod constant;
mod conv;
mod elementwise;
mod gather;
mod matmul;
mod norm;
mod pool;
mod quant;
mod reduce;
mod shape;

use crate::context::NodeContext;
use crate::error::{ImportError, Unsupported};
use crate::lowered::Lowered;

/// A family's importer: `None` if the op is not one of its own.
type FamilyImporter = fn(&str, &mut NodeContext<'_>) -> Option<Result<Lowered, ImportError>>;

/// The node in canonical form (D32), or why it cannot be imported.
pub(crate) fn lower(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let families: [FamilyImporter; 10] = [
        constant::lower,
        elementwise::lower,
        activation::lower,
        matmul::lower,
        conv::lower,
        pool::lower,
        reduce::lower,
        norm::lower,
        shape::lower,
        gather::lower,
    ];
    let op = context.op_type();
    families
        .iter()
        .find_map(|import| import(op, context))
        .unwrap_or_else(|| Err(Unsupported::Op { domain: String::new(), op: op.to_owned() }.into()))
}
