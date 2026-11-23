//! Strategies for shapes that broadcast and tensors whose values reach
//! each op's edge cases.

use proptest::collection::vec;
use proptest::prelude::*;
use ptxc_core::convert::f64_to_f32;
use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};

/// The values an operand takes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Domain {
    /// Anything its type holds, specials included.
    Any,
    /// Integers from 0 to 6, where a computation is exact in any type.
    SmallInt,
}

/// An output shape: rank 0 to 3, extents mostly 1 to 3, sometimes 0.
pub(crate) fn output_dims() -> impl Strategy<Value = Vec<u64>> {
    let extent = prop_oneof![1 => Just(0_u64), 12 => 1_u64..=3];
    vec(extent, 0..=3)
}

/// A shape broadcasting to `output`: some leading axes dropped, some
/// extents replaced by 1.
pub(crate) fn operand_dims(output: Vec<u64>) -> impl Strategy<Value = Vec<u64>> {
    let rank = output.len();
    (0..=rank, vec(any::<bool>(), rank)).prop_map(move |(dropped, ones)| {
        output[dropped..]
            .iter()
            .zip(&ones[dropped..])
            .map(|(&extent, &is_one)| if is_one { 1 } else { extent })
            .collect()
    })
}

pub(crate) fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// A tensor of `dtype` and `dims` with values from `domain`.
pub(crate) fn tensor(dtype: DType, dims: Vec<u64>, domain: Domain) -> BoxedStrategy<HostTensor> {
    let len = usize::try_from(shape(&dims).element_count().get()).unwrap();
    data(dtype, len, domain)
        .prop_map(move |data| HostTensor::new(shape(&dims), data).unwrap())
        .boxed()
}

fn data(dtype: DType, len: usize, domain: Domain) -> BoxedStrategy<TensorData> {
    if matches!(domain, Domain::SmallInt) {
        return vec(0_i64..=6, len).prop_map(move |values| small_ints(dtype, &values)).boxed();
    }
    match dtype {
        DType::I8 => vec(any::<i8>(), len).prop_map(TensorData::from).boxed(),
        DType::U8 => vec(any::<u8>(), len).prop_map(TensorData::from).boxed(),
        DType::I32 => {
            vec(prop_oneof![-50_i32..50, any::<i32>()], len).prop_map(TensorData::from).boxed()
        }
        DType::I64 => {
            vec(prop_oneof![-50_i64..50, any::<i64>()], len).prop_map(TensorData::from).boxed()
        }
        DType::Bool => vec(any::<bool>(), len).prop_map(TensorData::from).boxed(),
        _ => vec(float(), len).prop_map(move |values| floats(dtype, &values)).boxed(),
    }
}

/// Mostly ordinary values, with zeros of both signs, infinities, NaN and
/// values past f16's range.
fn float() -> impl Strategy<Value = f64> {
    let special = prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(f64::NAN),
        Just(1e30),
        Just(-70_000.0),
    ];
    prop_oneof![6 => -8.0..8.0_f64, 3 => (-4_i32..=4).prop_map(f64::from), 1 => special]
}

fn floats(dtype: DType, values: &[f64]) -> TensorData {
    let each = values.iter().copied();
    match dtype {
        DType::F32 => each.map(f64_to_f32).collect::<Vec<_>>().into(),
        DType::F16 => each.map(F16::from_f64).collect::<Vec<_>>().into(),
        DType::Bf16 => each.map(Bf16::from_f64).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => {
            each.map(|x| F8E4M3Fn::from_f64(x, Overflow::Saturate)).collect::<Vec<_>>().into()
        }
        DType::F8E5M2 => {
            each.map(|x| F8E5M2::from_f64(x, Overflow::Saturate)).collect::<Vec<_>>().into()
        }
        other => panic!("{other:?} is not a float type"),
    }
}

fn small_ints(dtype: DType, values: &[i64]) -> TensorData {
    let each = values.iter().copied();
    match dtype {
        DType::I8 => each.map(|x| i8::try_from(x).unwrap()).collect::<Vec<_>>().into(),
        DType::U8 => each.map(|x| u8::try_from(x).unwrap()).collect::<Vec<_>>().into(),
        DType::I32 => each.map(|x| i32::try_from(x).unwrap()).collect::<Vec<_>>().into(),
        DType::I64 => values.to_vec().into(),
        DType::Bool => each.map(|x| x != 0).collect::<Vec<_>>().into(),
        _ => floats(dtype, &each.map(ptxc_core::convert::int_to_f64).collect::<Vec<_>>()),
    }
}
