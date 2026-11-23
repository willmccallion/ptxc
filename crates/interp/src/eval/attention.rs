//! ONNX `Attention`: `softmax(mask(softcap(scale * Q K^T))) V` per head, in
//! f64, with grouped key and value heads and an optional past.

use super::{result, round_to_output};
use crate::error::{InterpError, Malformed};
use crate::extents::Extents;
use crate::numeric::{ints, wide_floats};
use ptxc_core::{HostTensor, TensorData};
use ptxc_graph::{
    AttentionAttrs, AttentionLayout, AttentionMask, Causality, Presence, QkOutput, TensorType,
};

/// A tensor as `(batch, heads, sequence, size)`, in f64.
struct HeadTensor {
    values: Vec<f64>,
    heads: usize,
    sequence: usize,
    size: usize,
}

impl HeadTensor {
    /// 4-D operands are already in this layout; packed 3-D ones are
    /// `(batch, sequence, heads * size)` and are regrouped.
    fn of(
        tensor: &HostTensor,
        layout: AttentionLayout,
        packed_heads: usize,
    ) -> Result<Self, InterpError> {
        let values = wide_floats(tensor.data())?;
        let extents = Extents::of(tensor.shape())?;
        match (layout, extents.dims()) {
            (AttentionLayout::Heads, &[_, heads, sequence, size]) => {
                Ok(Self { values, heads, sequence, size })
            }
            (AttentionLayout::Packed { .. }, &[batch, sequence, hidden]) => {
                let size = hidden / packed_heads;
                let regrouped = Extents::from_dims(vec![batch, packed_heads, sequence, size])
                    .indices()
                    .map(|i| values[((i[0] * sequence + i[2]) * packed_heads + i[1]) * size + i[3]])
                    .collect();
                Ok(Self { values: regrouped, heads: packed_heads, sequence, size })
            }
            _ => Err(Malformed::Operands.into()),
        }
    }

    fn at(&self, batch: usize, head: usize, position: usize) -> &[f64] {
        let start = ((batch * self.heads + head) * self.sequence + position) * self.size;
        &self.values[start..start + self.size]
    }

    /// `past` followed by `self` along the sequence axis.
    fn after(self, past: &Self, batch: usize) -> Self {
        let sequence = past.sequence + self.sequence;
        let mut values = Vec::with_capacity(batch * self.heads * sequence * self.size);
        for b in 0..batch {
            for h in 0..self.heads {
                (0..past.sequence).for_each(|p| values.extend_from_slice(past.at(b, h, p)));
                (0..self.sequence).for_each(|p| values.extend_from_slice(self.at(b, h, p)));
            }
        }
        Self { values, heads: self.heads, sequence, size: self.size }
    }
}

/// Attention's operands, split by role as the op's attributes say.
struct Operands<'t> {
    q: &'t HostTensor,
    k: &'t HostTensor,
    v: &'t HostTensor,
    mask: Option<&'t HostTensor>,
    past: Option<[&'t HostTensor; 2]>,
    lengths: Option<&'t HostTensor>,
}

fn split<'t>(
    attrs: &AttentionAttrs,
    operands: &[&'t HostTensor],
) -> Result<Operands<'t>, InterpError> {
    let [q, k, v, rest @ ..] = operands else { return Err(Malformed::Operands.into()) };
    let mut rest = rest.iter().copied();
    let mask = (attrs.mask != AttentionMask::None).then(|| rest.next()).flatten();
    let past = (attrs.past == Presence::Present)
        .then(|| rest.next().zip(rest.next()).map(<[_; 2]>::from))
        .flatten();
    let lengths = (attrs.nonpad_lengths == Presence::Present).then(|| rest.next()).flatten();
    Ok(Operands { q, k, v, mask, past, lengths })
}

/// The mask operand's elements.
enum MaskValues {
    Boolean(Vec<bool>),
    Additive(Vec<f64>),
}

/// What is added to each score before the softmax: the mask (`false` and
/// positions past a short mask's last axis as -inf), causal hiding of keys
/// after the query's position counted from the end of the past, and hiding
/// of keys past a sample's non-pad length.
struct Bias {
    mask: Option<(MaskValues, Extents)>,
    causal: bool,
    past: usize,
    lengths: Option<Vec<i64>>,
}

impl Bias {
    fn of(
        attrs: &AttentionAttrs,
        operands: &Operands<'_>,
        past: usize,
    ) -> Result<Self, InterpError> {
        let mask = match operands.mask {
            Some(mask) => {
                let values = match mask.data() {
                    TensorData::Bool(flags) => MaskValues::Boolean(flags.clone()),
                    data => MaskValues::Additive(wide_floats(data)?),
                };
                Some((values, Extents::of(mask.shape())?))
            }
            None => None,
        };
        let lengths = operands.lengths.map(|lengths| ints(lengths.data())).transpose()?;
        Ok(Self { mask, causal: attrs.causality == Causality::Causal, past, lengths })
    }

    /// The bias of score `[b, h, i, j]`.
    fn at(&self, index: [usize; 4]) -> f64 {
        let [b, _, i, j] = index;
        let hidden_by_length = self
            .lengths
            .as_ref()
            .is_some_and(|lengths| i64::try_from(j).map_or(true, |j| j >= lengths[b]));
        if (self.causal && j > i + self.past) || hidden_by_length {
            return f64::NEG_INFINITY;
        }
        let Some((values, extents)) = &self.mask else { return 0.0 };
        match (mask_offset(extents, index), values) {
            (None, _) => f64::NEG_INFINITY,
            (Some(offset), MaskValues::Boolean(flags)) => {
                if flags[offset] {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            (Some(offset), MaskValues::Additive(added)) => added[offset],
        }
    }
}

/// The mask element of score `index`: the leading axes broadcast from the
/// right, and a last axis shorter than the key sequence reads nothing past
/// its end.
fn mask_offset(extents: &Extents, index: [usize; 4]) -> Option<usize> {
    let (dims, strides) = (extents.dims(), extents.strides());
    let skipped = index.len() - dims.len();
    let last = dims.len().checked_sub(1)?;
    let mut offset = 0;
    for axis in 0..dims.len() {
        let i = index[skipped + axis];
        if axis == last && i >= dims[axis] {
            return None;
        }
        if axis == last || dims[axis] != 1 {
            offset += i * strides[axis];
        }
    }
    Some(offset)
}

/// One query row's scores at each stage, and its output.
struct Row {
    stages: [Vec<f64>; 4],
    output: Vec<f64>,
}

/// Where a query row reads its keys and values: the sample and kv head.
#[derive(Clone, Copy)]
struct Source {
    batch: usize,
    kv_head: usize,
}

fn row(
    attrs: &AttentionAttrs,
    kv: [&HeadTensor; 2],
    query: &[f64],
    bias: impl Fn(usize) -> f64,
    source: Source,
) -> Row {
    let [keys, values] = kv;
    let Source { batch, kv_head } = source;
    let scale = f64::from(attrs.scale.get());
    let scaled: Vec<f64> = (0..keys.sequence)
        .map(|position| {
            scale
                * query
                    .iter()
                    .zip(keys.at(batch, kv_head, position))
                    .map(|(q, k)| q * k)
                    .sum::<f64>()
        })
        .collect();
    let capped: Vec<f64> = attrs.softcap.filter(|cap| cap.get() > 0.0).map_or_else(
        || scaled.clone(),
        |cap| {
            let cap = f64::from(cap.get());
            scaled.iter().map(|score| cap * (score / cap).tanh()).collect()
        },
    );
    let masked: Vec<f64> =
        capped.iter().enumerate().map(|(position, score)| score + bias(position)).collect();
    let max = masked.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = masked.iter().map(|score| (score - max).exp()).collect();
    let sum: f64 = exponentials.iter().sum();
    let probabilities: Vec<f64> = exponentials.iter().map(|e| e / sum).collect();
    let mut output = vec![0.0; values.size];
    for (position, weight) in probabilities.iter().enumerate() {
        for (out, value) in output.iter_mut().zip(values.at(batch, kv_head, position)) {
            *out = weight.mul_add(*value, *out);
        }
    }
    Row { stages: [scaled, capped, masked, probabilities], output }
}

/// The head counts the layout gives, for regrouping packed operands.
fn head_counts(layout: AttentionLayout) -> Result<[usize; 2], InterpError> {
    let count =
        |heads: u64| usize::try_from(heads).map_err(|_| InterpError::TooLarge { elements: heads });
    match layout {
        AttentionLayout::Heads => Ok([0, 0]),
        AttentionLayout::Packed { q_heads, kv_heads } => {
            Ok([count(q_heads.get())?, count(kv_heads.get())?])
        }
    }
}

/// Which stage the QK output holds.
const fn stage(output: QkOutput) -> usize {
    match output {
        QkOutput::MatMul => 0,
        QkOutput::Softcapped => 1,
        QkOutput::Masked => 2,
        QkOutput::Softmax => 3,
    }
}

/// `(batch, heads, sequence, size)` back to packed `(batch, sequence,
/// heads * size)`.
fn repack(heads: &[f64], dims: [usize; 4]) -> Vec<f64> {
    let [batch, count, sequence, size] = dims;
    Extents::from_dims(vec![batch, sequence, count, size])
        .indices()
        .map(|i| heads[((i[0] * count + i[2]) * sequence + i[1]) * size + i[3]])
        .collect()
}

/// `Attention`'s outputs: `Y`, then the present key and value if asked for,
/// then the QK intermediate if asked for.
pub(super) fn attention(
    attrs: &AttentionAttrs,
    operands: &[&HostTensor],
    outputs: &[&TensorType],
) -> Result<Vec<HostTensor>, InterpError> {
    let operands = split(attrs, operands)?;
    let [q_heads, kv_heads] = head_counts(attrs.layout)?;
    let query = HeadTensor::of(operands.q, attrs.layout, q_heads)?;
    let mut keys = HeadTensor::of(operands.k, attrs.layout, kv_heads)?;
    let mut values = HeadTensor::of(operands.v, attrs.layout, kv_heads)?;
    let batch = Extents::of(operands.q.shape())?.dims()[0];
    let mut past_length = 0;
    if let Some([past_key, past_value]) = operands.past {
        let past_keys = HeadTensor::of(past_key, AttentionLayout::Heads, 0)?;
        past_length = past_keys.sequence;
        keys = keys.after(&past_keys, batch);
        values = values.after(&HeadTensor::of(past_value, AttentionLayout::Heads, 0)?, batch);
    }
    let bias = Bias::of(attrs, &operands, past_length)?;
    let group = query.heads.checked_div(keys.heads).ok_or(Malformed::Operands)?;
    let (mut y, mut qk) = (Vec::new(), Vec::new());
    for index in Extents::from_dims(vec![batch, query.heads, query.sequence]).indices() {
        let [sample, head, position] = [index[0], index[1], index[2]];
        let source = Source { batch: sample, kv_head: head / group };
        let row = row(
            attrs,
            [&keys, &values],
            query.at(sample, head, position),
            |key| bias.at([sample, head, position, key]),
            source,
        );
        y.extend(row.output);
        if let Some(output) = attrs.qk_output {
            qk.extend_from_slice(&row.stages[stage(output)]);
        }
    }
    if matches!(attrs.layout, AttentionLayout::Packed { .. }) {
        y = repack(&y, [batch, query.heads, query.sequence, values.size]);
    }
    let mut stored = Vec::with_capacity(outputs.len());
    let present = attrs.present == Presence::Present;
    let produced = std::iter::once(y)
        .chain(present.then_some([keys.values, values.values]).into_iter().flatten())
        .chain(attrs.qk_output.map(|_| qk));
    for (values, ty) in produced.zip(outputs) {
        stored.push(result(ty, round_to_output(&values, ty)?)?);
    }
    Ok(stored)
}
