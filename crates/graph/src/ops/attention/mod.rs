//! `Attention`: fused scaled dot-product attention, as ONNX's `Attention`
//! (opset 23+). The attention matcher (M6) produces the same op.

mod extents;

use super::accepts::Accepts;
use super::attr::F32Attr;
use super::elementwise::broadcast_to;
use super::{OperandCount, TypeError, TypeRule, require_dtype, require_rank};
use crate::tensor_type::{Layout, TensorType};
use extents::{Extents, check_dims, heads_extents, packed_extents, past_total, product};
use ptxc_core::{DType, Dim, Rank, Shape};
use std::num::NonZeroU64;

/// A number of attention heads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HeadCount(NonZeroU64);

impl HeadCount {
    /// The count, or `None` for zero.
    pub const fn new(count: u64) -> Option<Self> {
        match NonZeroU64::new(count) {
            Some(count) => Some(Self(count)),
            None => None,
        }
    }

    /// The count.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// How Q, K, V and Y lay out their heads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AttentionLayout {
    /// 4-D `(batch, heads, sequence, head size)`.
    Heads,
    /// 3-D `(batch, sequence, heads * head size)`, with ONNX's `q_num_heads`
    /// and `kv_num_heads`.
    Packed {
        /// Query heads.
        q_heads: HeadCount,
        /// Key and value heads.
        kv_heads: HeadCount,
    },
}

/// The optional attention mask operand, from ONNX's `attn_mask`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AttentionMask {
    /// No mask operand.
    None,
    /// A bool mask: `true` takes part in attention.
    Boolean,
    /// A float mask of Q's type, added to the scores.
    Additive,
}

/// Whether a lower-triangular (upper-left aligned) mask applies, from ONNX's
/// `is_causal`. Independent of the mask operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Causality {
    /// Every key is visible.
    Full,
    /// Keys after the query's position are hidden.
    Causal,
}

/// Whether an optional operand pair or output pair is there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Presence {
    /// Not there.
    Absent,
    /// There.
    Present,
}

/// Which intermediate ONNX's optional `qk_matmul_output` holds, from
/// `qk_matmul_output_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QkOutput {
    /// Mode 0: `Q * K^T`, scaled.
    MatMul,
    /// Mode 1: after the softcap.
    Softcapped,
    /// Mode 2: after the softcap and mask.
    Masked,
    /// Mode 3: after the softmax.
    Softmax,
}

/// ONNX `Attention`, with defaults (`scale`, softmax precision) resolved by
/// the importer.
///
/// Operands, in order: `Q`, `K`, `V`, then the mask if `mask` is not `None`,
/// `past_key` and `past_value` if `past` is present, and `nonpad_kv_seqlen`
/// if `nonpad_lengths` is present. Outputs: `Y`, then `present_key` and
/// `present_value` if `present` is present, then the QK intermediate if
/// `qk_output` is set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttentionAttrs {
    /// How heads are laid out.
    pub layout: AttentionLayout,
    /// The mask operand, if any.
    pub mask: AttentionMask,
    /// Whether causal masking applies.
    pub causality: Causality,
    /// Whether the past key and value operands are there.
    pub past: Presence,
    /// Whether the present key and value outputs are there.
    pub present: Presence,
    /// Whether the per-sample valid key count operand is there.
    pub nonpad_lengths: Presence,
    /// The intermediate output, if any.
    pub qk_output: Option<QkOutput>,
    /// The factor applied to `Q * K^T`.
    pub scale: F32Attr,
    /// `softcap * tanh(scores / softcap)`, if set.
    pub softcap: Option<F32Attr>,
    /// The float type the softmax is computed in.
    pub softmax_precision: DType,
}

impl TypeRule for AttentionAttrs {
    fn infer(&self, operands: &[&TensorType]) -> Result<Vec<TensorType>, TypeError> {
        let operands = self.split_operands(operands)?;
        let [q, k, v] = operands.qkv;
        Accepts::Float.check(q.dtype())?;
        Accepts::Float.check(v.dtype())?;
        Accepts::Float.check(self.softmax_precision)?;
        require_dtype(q.dtype(), k)?;
        let extents = self.extents(q, k, v)?;
        let total = past_total(&operands, extents)?;
        if let Some(mask) = operands.mask {
            self.check_mask(mask, q.dtype(), extents, total)?;
        }
        if let Some(lengths) = operands.nonpad_lengths {
            require_dtype(DType::I64, lengths)?;
            require_rank(lengths.shape(), Rank::new(1))?;
            check_dims(lengths.shape(), &[(0, extents.batch)])?;
        }
        self.outputs(q.dtype(), v.dtype(), extents, total)
    }
}

/// Attention's operands, split by role.
struct Operands<'a> {
    qkv: [&'a TensorType; 3],
    mask: Option<&'a TensorType>,
    past: Option<[&'a TensorType; 2]>,
    nonpad_lengths: Option<&'a TensorType>,
}

impl AttentionAttrs {
    fn split_operands<'a>(&self, operands: &[&'a TensorType]) -> Result<Operands<'a>, TypeError> {
        let has_mask = self.mask != AttentionMask::None;
        let has_past = self.past == Presence::Present;
        let has_lengths = self.nonpad_lengths == Presence::Present;
        if has_lengths && (has_past || self.present == Presence::Present) {
            return Err(TypeError::NonpadWithCache);
        }
        let expected =
            3 + usize::from(has_mask) + 2 * usize::from(has_past) + usize::from(has_lengths);
        if operands.len() != expected {
            return Err(TypeError::Arity {
                expected: OperandCount(expected),
                actual: OperandCount(operands.len()),
            });
        }
        let mut rest = operands[3..].iter().copied();
        let mask = if has_mask { rest.next() } else { None };
        let past = if has_past { rest.next().zip(rest.next()).map(<[_; 2]>::from) } else { None };
        let nonpad_lengths = if has_lengths { rest.next() } else { None };
        Ok(Operands { qkv: [operands[0], operands[1], operands[2]], mask, past, nonpad_lengths })
    }

    fn extents(
        &self,
        q: &TensorType,
        k: &TensorType,
        v: &TensorType,
    ) -> Result<Extents, TypeError> {
        let extents = match self.layout {
            AttentionLayout::Heads => heads_extents(q.shape(), k.shape(), v.shape())?,
            AttentionLayout::Packed { q_heads, kv_heads } => {
                packed_extents(q.shape(), k.shape(), v.shape(), [q_heads, kv_heads])?
            }
        };
        if extents.q_heads.checked_rem(extents.kv_heads) != Some(0) {
            let (q_heads, kv_heads) = (Dim::new(extents.q_heads), Dim::new(extents.kv_heads));
            return Err(TypeError::HeadsNotDivisible { q_heads, kv_heads });
        }
        Ok(extents)
    }

    /// The mask broadcasts to `(batch, q_heads, q_sequence, total)`, except
    /// that its last axis may be shorter than `total` (ONNX pads it with
    /// negative infinity).
    fn check_mask(
        &self,
        mask: &TensorType,
        q_dtype: DType,
        extents: Extents,
        total: Dim,
    ) -> Result<(), TypeError> {
        let expected = match self.mask {
            AttentionMask::Boolean => DType::Bool,
            AttentionMask::Additive | AttentionMask::None => q_dtype,
        };
        require_dtype(expected, mask)?;
        let mut dims = mask.shape().dims().to_vec();
        if let Some(last) = dims.last_mut().filter(|last| **last <= total) {
            *last = total;
        }
        let target =
            Shape::new(vec![extents.batch, Dim::new(extents.q_heads), extents.q_sequence, total])?;
        broadcast_to(&Shape::new(dims)?, &target)
    }

    fn outputs(
        &self,
        q_dtype: DType,
        v_dtype: DType,
        extents: Extents,
        total: Dim,
    ) -> Result<Vec<TensorType>, TypeError> {
        let Extents { batch, q_heads, q_sequence, head_size, kv_heads, v_head_size, .. } = extents;
        let y_dims = match self.layout {
            AttentionLayout::Heads => vec![batch, Dim::new(q_heads), q_sequence, v_head_size],
            AttentionLayout::Packed { .. } => {
                vec![batch, q_sequence, product(q_heads, v_head_size)?]
            }
        };
        let mut outputs = vec![TensorType::new(q_dtype, Shape::new(y_dims)?, Layout::RowMajor)];
        if self.present == Presence::Present {
            for (dtype, size) in [(q_dtype, head_size), (v_dtype, v_head_size)] {
                let dims = vec![batch, Dim::new(kv_heads), total, size];
                outputs.push(TensorType::new(dtype, Shape::new(dims)?, Layout::RowMajor));
            }
        }
        if self.qk_output.is_some() {
            let dims = vec![batch, Dim::new(q_heads), q_sequence, total];
            outputs.push(TensorType::new(q_dtype, Shape::new(dims)?, Layout::RowMajor));
        }
        Ok(outputs)
    }
}
