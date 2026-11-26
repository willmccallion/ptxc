//! Which ops may fuse with which (D82).

use ptxc_graph::Op;

/// What an op is to fusion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    /// It reduces or multiplies, so a group holds at most one: products,
    /// convolutions, windows, reductions and normalisations.
    Anchor(Anchor),
    /// Each output element reads its operands at that element, broadcast:
    /// it fuses as an epilogue, or into the reads of its consumers.
    Elementwise,
    /// It moves elements without arithmetic; fused only into the reads of
    /// its consumers, where it costs nothing.
    Transpose,
    /// It stays a group of its own: gathers, which read loaded indices,
    /// views, the other data movement, and ops of several outputs.
    Apart,
}

/// The kind of op a group is built around.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Anchor {
    /// A matrix product: its operands must stay plain reads to be tiled.
    Product,
    /// A convolution or a pooling window: it reads its input padded.
    Window,
    /// A reduction or normalisation, or attention.
    Reduction,
}

/// `op`'s role in fusion.
pub(super) const fn role(op: &Op) -> Role {
    match op {
        Op::MatMul | Op::Gemm(_) | Op::MatMulInteger(_) | Op::QMatMul(_) => {
            Role::Anchor(Anchor::Product)
        }
        Op::Conv(_) | Op::QConv(_) | Op::Pool(_) => Role::Anchor(Anchor::Window),
        Op::GlobalPool(_)
        | Op::Reduce(_)
        | Op::ArgReduce(_)
        | Op::CumSum(_)
        | Op::Softmax(_)
        | Op::LayerNorm(_)
        | Op::InstanceNorm(_)
        | Op::Attention(_) => Role::Anchor(Anchor::Reduction),
        Op::Unary(_)
        | Op::Binary(_)
        | Op::Variadic(_)
        | Op::Where
        | Op::Cast(_)
        | Op::Activation(_)
        | Op::Clip
        | Op::PRelu
        | Op::BatchNorm(_)
        | Op::Quantize(_)
        | Op::Dequantize(_) => Role::Elementwise,
        Op::Transpose(_) => Role::Transpose,
        Op::Reshape(_)
        | Op::Expand(_)
        | Op::Tile(_)
        | Op::Concat(_)
        | Op::Slice(_)
        | Op::Split(_)
        | Op::Pad(_)
        | Op::Trilu(_)
        | Op::Gather(_)
        | Op::GatherElements(_)
        | Op::GatherNd(_)
        | Op::Resize(_) => Role::Apart,
    }
}

/// Whether a group of ops with these `roles`, no anchor among them, may be
/// inlined into the reads of `reader`.
pub(super) fn takes_into_reads(reader: &Op, roles: &[Role]) -> bool {
    match role(reader) {
        Role::Anchor(Anchor::Product) => roles.iter().all(|&role| role == Role::Transpose),
        Role::Anchor(Anchor::Window) | Role::Apart => false,
        Role::Anchor(Anchor::Reduction) | Role::Elementwise | Role::Transpose => {
            !matches!(reader, Op::Attention(_))
        }
    }
}
