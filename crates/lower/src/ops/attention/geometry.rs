//! Attention's operands and extents in `(batch, heads, sequence, size)`
//! terms, whatever the layout.

use crate::builder::StageBuilder;
use crate::error::LowerError;
use crate::ops::NodeContext;
use ptxc_core::Dim;
use ptxc_graph::{AttentionAttrs, AttentionLayout, AttentionMask, Causality, Presence, ValueId};
use ptxc_loop::ExprId;

/// Attention's operands by role, and its extents.
pub(super) struct Geometry {
    pub(super) layout: AttentionLayout,
    pub(super) q: ValueId,
    pub(super) k: ValueId,
    pub(super) v: ValueId,
    pub(super) past_key: Option<ValueId>,
    pub(super) past_value: Option<ValueId>,
    pub(super) mask: Option<ValueId>,
    pub(super) mask_kind: AttentionMask,
    pub(super) lengths: Option<ValueId>,
    pub(super) causality: Causality,
    pub(super) batch: Dim,
    pub(super) q_heads: Dim,
    pub(super) kv_heads: Dim,
    pub(super) q_sequence: Dim,
    pub(super) kv_sequence: Dim,
    pub(super) past: Dim,
    pub(super) head_size: Dim,
    pub(super) v_head_size: Dim,
}

/// `[batch, q_heads, q_sequence, head_size]` and `[kv_heads, kv_sequence,
/// v_head_size]`.
type Extents = ([Dim; 4], [Dim; 3]);

fn extents(
    cx: &NodeContext<'_>,
    attrs: &AttentionAttrs,
    [q, k, v]: [ValueId; 3],
) -> Result<Extents, LowerError> {
    let dims = |value: ValueId| cx.graph.value(value).ty().shape().dims().to_vec();
    let missing = || LowerError::Operands(cx.id);
    match attrs.layout {
        AttentionLayout::Heads => {
            let [batch, q_heads, q_sequence, head_size] = dims(q)[..] else {
                return Err(missing());
            };
            let [_, kv_heads, kv_sequence, _] = dims(k)[..] else { return Err(missing()) };
            let [.., v_head_size] = dims(v)[..] else { return Err(missing()) };
            Ok(([batch, q_heads, q_sequence, head_size], [kv_heads, kv_sequence, v_head_size]))
        }
        AttentionLayout::Packed { q_heads, kv_heads } => {
            let [batch, q_sequence, q_hidden] = dims(q)[..] else { return Err(missing()) };
            let [_, kv_sequence, _] = dims(k)[..] else { return Err(missing()) };
            let [.., v_hidden] = dims(v)[..] else { return Err(missing()) };
            let (q_count, kv_count) = (q_heads.get(), kv_heads.get());
            let per_head = |hidden: Dim, heads: u64| Dim::new(hidden.get() / heads);
            Ok((
                [batch, Dim::new(q_count), q_sequence, per_head(q_hidden, q_count)],
                [Dim::new(kv_count), kv_sequence, per_head(v_hidden, kv_count)],
            ))
        }
    }
}

impl Geometry {
    /// Operands in ONNX's order: Q, K and V, then the mask, the past key
    /// and value, and the non-pad lengths, each where the attributes say.
    pub(super) fn of(cx: &NodeContext<'_>, attrs: &AttentionAttrs) -> Result<Self, LowerError> {
        let operands = cx.operands();
        let [q, k, v, ref rest @ ..] = operands[..] else {
            return Err(LowerError::Operands(cx.id));
        };
        let mut rest = rest.iter().copied();
        let mask = (attrs.mask != AttentionMask::None).then(|| rest.next()).flatten();
        let past = (attrs.past == Presence::Present).then(|| (rest.next(), rest.next()));
        let lengths = (attrs.nonpad_lengths == Presence::Present).then(|| rest.next()).flatten();
        let (past_key, past_value) = past.unwrap_or((None, None));
        let ([batch, q_heads, q_sequence, head_size], [kv_heads, kv_sequence, v_head_size]) =
            extents(cx, attrs, [q, k, v])?;
        let past_dims = past_key.map(|key| cx.graph.value(key).ty().shape().dims().to_vec());
        let past = past_dims.and_then(|dims| dims.get(2).copied()).unwrap_or(Dim::new(0));
        Ok(Self {
            layout: attrs.layout,
            q,
            k,
            v,
            past_key,
            past_value,
            mask,
            mask_kind: attrs.mask,
            lengths,
            causality: attrs.causality,
            batch,
            q_heads,
            kv_heads,
            q_sequence,
            kv_sequence,
            past,
            head_size,
            v_head_size,
        })
    }

    /// Every key position: the past's, then the current ones.
    pub(super) const fn total(&self) -> Dim {
        Dim::new(self.past.get().saturating_add(self.kv_sequence.get()))
    }

    /// The scores' dims, `(batch, q_heads, q_sequence, total)`.
    pub(super) const fn score_dims(&self) -> [Dim; 4] {
        [self.batch, self.q_heads, self.q_sequence, self.total()]
    }

    /// The per-row statistics' dims, `(batch, q_heads, q_sequence, 1)`.
    pub(super) const fn row_dims(&self) -> [Dim; 4] {
        [self.batch, self.q_heads, self.q_sequence, Dim::new(1)]
    }

    /// The key and value head that query head `head` reads.
    pub(super) fn kv_head(
        &self,
        cx: &NodeContext<'_>,
        stage: &mut StageBuilder<'_>,
        head: ExprId,
    ) -> Result<ExprId, LowerError> {
        let group = self.q_heads.get() / self.kv_heads.get().max(1);
        if group == 1 {
            return Ok(head);
        }
        let group = stage.constant(cx.signed(Dim::new(group))?);
        Ok(stage.exprs().floor_div(head, group))
    }

    /// The `(batch, head, query, element)` point of a `Y` stage's output
    /// element: 4-D directly, or packed `(batch, query, head * size +
    /// element)` split.
    pub(super) fn y_point(
        &self,
        cx: &NodeContext<'_>,
        stage: &mut StageBuilder<'_>,
    ) -> Result<[ExprId; 4], LowerError> {
        let missing = || LowerError::Operands(cx.id);
        let spatial = stage.spatial().to_vec();
        match (self.layout, &spatial[..]) {
            (AttentionLayout::Heads, &[batch, head, at, element]) => Ok([batch, head, at, element]),
            (AttentionLayout::Packed { .. }, &[batch, at, packed]) => {
                let size = stage.constant(cx.signed(self.v_head_size)?);
                let head = stage.exprs().floor_div(packed, size);
                let element = stage.exprs().floor_mod(packed, size);
                Ok([batch, head, at, element])
            }
            _ => Err(missing()),
        }
    }
}
