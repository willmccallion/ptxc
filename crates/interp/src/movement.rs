//! Ops that only move data, written once for every element type: each
//! output element is a source element at a computed position.

use crate::error::{InterpError, Malformed};
use ptxc_core::{TensorData, with_tensor_data};

/// The elements at `positions`, in order.
pub(crate) fn take(data: &TensorData, positions: &[usize]) -> TensorData {
    with_tensor_data!(data, elements => {
        TensorData::from(positions.iter().map(|&position| elements[position]).collect::<Vec<_>>())
    })
}

/// Several tensors' elements end to end, so positions can index any of them.
pub(crate) fn join(parts: &[&TensorData]) -> Result<TensorData, InterpError> {
    let Some((first, rest)) = parts.split_first() else {
        return Ok(TensorData::from(Vec::<f32>::new()));
    };
    let mut joined = (*first).clone();
    for part in rest {
        append(&mut joined, part)?;
    }
    Ok(joined)
}

fn append(joined: &mut TensorData, part: &TensorData) -> Result<(), InterpError> {
    macro_rules! extend {
        ($($variant:ident),+) => {
            match (joined, part) {
                $((TensorData::$variant(into), TensorData::$variant(from)) => into.extend_from_slice(from),)+
                (_, other) => return Err(Malformed::DType(other.dtype()).into()),
            }
        };
    }
    extend!(F32, F16, Bf16, F8E4M3Fn, F8E5M2, I8, U8, I32, I64, Bool);
    Ok(())
}

/// One zero element of `dtype`, the fill of `Trilu`.
pub(crate) fn zero(dtype: ptxc_core::DType) -> TensorData {
    use ptxc_core::DType;
    use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16};
    match dtype {
        DType::F32 => vec![0.0_f32].into(),
        DType::F16 => vec![F16::from_bits(0)].into(),
        DType::Bf16 => vec![Bf16::from_bits(0)].into(),
        DType::F8E4M3Fn => vec![F8E4M3Fn::from_bits(0)].into(),
        DType::F8E5M2 => vec![F8E5M2::from_bits(0)].into(),
        DType::I8 => vec![0_i8].into(),
        DType::U8 => vec![0_u8].into(),
        DType::I32 => vec![0_i32].into(),
        DType::I64 => vec![0_i64].into(),
        DType::Bool => vec![false].into(),
    }
}
