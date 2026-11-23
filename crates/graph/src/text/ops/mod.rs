//! Each op's text: a mnemonic and its attributes, one module per family.

mod activation;
mod attention;
mod elementwise;
mod matmul;
mod norm;
mod quant;
mod reduce;
mod resize;
mod shape;
mod window;

use super::attr::{AttrValue, Attrs};
use super::error::{ParseError, ParseErrorKind, Position};
use crate::ops::Op;

/// An op as text: `name<key=value, ...>`.
pub(super) struct OpText {
    pub(super) name: &'static str,
    pub(super) attrs: Vec<(&'static str, AttrValue)>,
}

impl OpText {
    /// An op with no attributes.
    const fn plain(name: &'static str) -> Self {
        Self { name, attrs: Vec::new() }
    }

    const fn new(name: &'static str, attrs: Vec<(&'static str, AttrValue)>) -> Self {
        Self { name, attrs }
    }
}

impl Op {
    /// The op's name in the text format, such as `add` or `reduce_sum`.
    pub fn mnemonic(&self) -> &'static str {
        encode(self).name
    }
}

/// The text of `op`. The match is exhaustive, so every op has a text form.
pub(super) fn encode(op: &Op) -> OpText {
    match op {
        Op::Unary(op) => elementwise::encode_unary(*op),
        Op::Binary(op) => elementwise::encode_binary(*op),
        Op::Variadic(op) => elementwise::encode_variadic(*op),
        Op::Where => OpText::plain("where"),
        Op::Cast(attrs) => elementwise::encode_cast(*attrs),
        Op::Activation(activation) => activation::encode(*activation),
        Op::Clip => OpText::plain("clip"),
        Op::PRelu => OpText::plain("prelu"),
        Op::MatMul => OpText::plain("matmul"),
        Op::Gemm(attrs) => matmul::encode_gemm(*attrs),
        Op::Conv(attrs) => window::encode_conv(attrs),
        Op::Pool(attrs) => window::encode_pool(attrs),
        Op::GlobalPool(kind) => window::encode_global_pool(*kind),
        Op::Reduce(attrs) => reduce::encode_reduce(attrs),
        Op::ArgReduce(attrs) => reduce::encode_arg_reduce(*attrs),
        Op::CumSum(attrs) => reduce::encode_cum_sum(*attrs),
        Op::Softmax(attrs) => norm::encode_softmax(*attrs),
        Op::LayerNorm(attrs) => norm::encode_layer_norm(*attrs),
        Op::BatchNorm(attrs) => norm::encode_epsilon("batch_norm", attrs.epsilon),
        Op::InstanceNorm(attrs) => norm::encode_epsilon("instance_norm", attrs.epsilon),
        Op::Reshape(target) => shape::encode_target("reshape", target),
        Op::Expand(target) => shape::encode_target("expand", target),
        Op::Transpose(attrs) => shape::encode_transpose(attrs),
        Op::Tile(attrs) => shape::encode_tile(attrs),
        Op::Concat(axis) => shape::encode_axis("concat", *axis),
        Op::Slice(attrs) => shape::encode_slice(attrs),
        Op::Split(attrs) => shape::encode_split(attrs),
        Op::Pad(attrs) => shape::encode_pad(attrs),
        Op::Trilu(attrs) => shape::encode_trilu(*attrs),
        Op::Gather(axis) => shape::encode_axis("gather", *axis),
        Op::GatherElements(axis) => shape::encode_axis("gather_elements", *axis),
        Op::GatherNd(batch_dims) => shape::encode_gather_nd(*batch_dims),
        Op::Quantize(attrs) => quant::encode_quantize(attrs),
        Op::Dequantize(attrs) => quant::encode_dequantize(attrs),
        Op::QMatMul(attrs) => quant::encode_qmatmul(attrs),
        Op::QConv(attrs) => quant::encode_qconv(attrs),
        Op::Resize(attrs) => resize::encode(attrs),
        Op::Attention(attrs) => attention::encode(attrs),
    }
}

/// A family's decoder: `None` if `name` is not one of its ops.
type Decoder = fn(&str, &mut Attrs) -> Option<Result<Op, ParseError>>;

/// The op `name` with `attrs`, every one of which must be used.
pub(super) fn decode(name: &str, mut attrs: Attrs, position: Position) -> Result<Op, ParseError> {
    let decoders: [Decoder; 10] = [
        elementwise::decode,
        activation::decode,
        matmul::decode,
        window::decode,
        reduce::decode,
        norm::decode,
        shape::decode,
        quant::decode,
        resize::decode,
        attention::decode,
    ];
    let op = decoders.iter().find_map(|decode| decode(name, &mut attrs)).unwrap_or_else(|| {
        Err(ParseError::new(position, ParseErrorKind::UnknownOp(name.to_owned())))
    })?;
    attrs.finish()?;
    Ok(op)
}
