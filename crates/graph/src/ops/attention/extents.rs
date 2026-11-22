//! Reading attention's extents from Q, K, V and the past key and value.

use super::{HeadCount, Operands};
use crate::ops::{TypeError, require_dtype, require_rank};
use ptxc_core::{Dim, Rank, Shape, ShapeError};

/// The extents attention works with, read from Q, K and V.
#[derive(Clone, Copy, Debug)]
pub(super) struct Extents {
    pub(super) batch: Dim,
    pub(super) q_heads: u64,
    pub(super) q_sequence: Dim,
    pub(super) head_size: Dim,
    pub(super) kv_heads: u64,
    pub(super) kv_sequence: Dim,
    pub(super) v_head_size: Dim,
}

/// Extents of 4-D Q `(B, Hq, Sq, D)`, K `(B, Hkv, Skv, D)` and V
/// `(B, Hkv, Skv, Dv)`.
pub(super) fn heads_extents(
    q_shape: &Shape,
    k_shape: &Shape,
    v_shape: &Shape,
) -> Result<Extents, TypeError> {
    for shape in [q_shape, k_shape, v_shape] {
        require_rank(shape, Rank::new(4))?;
    }
    let (q, k, v) = (q_shape.dims(), k_shape.dims(), v_shape.dims());
    let extents = Extents {
        batch: q[0],
        q_heads: q[1].get(),
        q_sequence: q[2],
        head_size: q[3],
        kv_heads: k[1].get(),
        kv_sequence: k[2],
        v_head_size: v[3],
    };
    check_dims(k_shape, &[(0, q[0]), (3, q[3])])?;
    check_dims(v_shape, &[(0, q[0]), (1, k[1]), (2, k[2])])?;
    Ok(extents)
}

/// Extents of 3-D Q `(B, Sq, Hq * D)`, K `(B, Skv, Hkv * D)` and V
/// `(B, Skv, Hkv * Dv)`.
pub(super) fn packed_extents(
    q_shape: &Shape,
    k_shape: &Shape,
    v_shape: &Shape,
    heads: [HeadCount; 2],
) -> Result<Extents, TypeError> {
    let [q_heads, kv_heads] = heads;
    for shape in [q_shape, k_shape, v_shape] {
        require_rank(shape, Rank::new(3))?;
    }
    let (q, k, v) = (q_shape.dims(), k_shape.dims(), v_shape.dims());
    let head_size = per_head(q[2], q_heads)?;
    let v_head_size = per_head(v[2], kv_heads)?;
    check_dims(k_shape, &[(0, q[0]), (2, product(kv_heads.get(), head_size)?)])?;
    check_dims(v_shape, &[(0, q[0]), (1, k[1])])?;
    Ok(Extents {
        batch: q[0],
        q_heads: q_heads.get(),
        q_sequence: q[1],
        head_size,
        kv_heads: kv_heads.get(),
        kv_sequence: k[1],
        v_head_size,
    })
}

/// The per-head size of a packed hidden axis.
const fn per_head(hidden: Dim, heads: HeadCount) -> Result<Dim, TypeError> {
    if hidden.get().is_multiple_of(heads.get()) {
        Ok(Dim::new(hidden.get() / heads.get()))
    } else {
        Err(TypeError::HiddenNotDivisible { hidden, heads })
    }
}

/// The total key sequence: past plus new. The past key and value are 4-D
/// `(B, Hkv, Sp, D)` and `(B, Hkv, Sp, Dv)` with Q's and V's types.
pub(super) fn past_total(operands: &Operands<'_>, extents: Extents) -> Result<Dim, TypeError> {
    let Some([past_key, past_value]) = operands.past else {
        return Ok(extents.kv_sequence);
    };
    let [q, _, v] = operands.qkv;
    require_dtype(q.dtype(), past_key)?;
    require_dtype(v.dtype(), past_value)?;
    for (past, size) in [(past_key, extents.head_size), (past_value, extents.v_head_size)] {
        require_rank(past.shape(), Rank::new(4))?;
        check_dims(
            past.shape(),
            &[(0, extents.batch), (1, Dim::new(extents.kv_heads)), (3, size)],
        )?;
    }
    let past_sequence = past_key.shape().dims()[2];
    check_dims(past_value.shape(), &[(2, past_sequence)])?;
    past_sequence.get().checked_add(extents.kv_sequence.get()).map(Dim::new).ok_or_else(|| {
        ShapeError::ElementCountOverflow { dims: vec![past_sequence, extents.kv_sequence] }.into()
    })
}

/// Checks that `shape` has each `(axis position, extent)` listed.
pub(super) fn check_dims(shape: &Shape, expected: &[(usize, Dim)]) -> Result<(), TypeError> {
    for (axis, &actual) in shape.rank().axes().zip(shape.dims()) {
        let wanted = expected.iter().find(|&&(position, _)| position == axis.index());
        if let Some(&(_, expected)) = wanted.filter(|&&(_, extent)| extent != actual) {
            return Err(TypeError::DimMismatch { axis, expected, actual });
        }
    }
    Ok(())
}

/// `heads * size`, the extent of a packed hidden axis.
pub(super) fn product(heads: u64, size: Dim) -> Result<Dim, TypeError> {
    heads.checked_mul(size.get()).map(Dim::new).ok_or_else(|| {
        ShapeError::ElementCountOverflow { dims: vec![Dim::new(heads), size] }.into()
    })
}
