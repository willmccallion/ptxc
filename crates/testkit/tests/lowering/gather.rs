//! `Gather`, `GatherElements` and `GatherND` with valid indices of both
//! index types, negative ones included, on every element type.

use crate::cases::single_node;
use crate::harness::check_against_reference;
use crate::values::{Domain, axis_dims, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{Axis, DType, Dim, HostTensor, Rank, Shape, TensorData};
use ptxc_graph::Op;

fn any_dtype() -> impl Strategy<Value = DType> {
    select(DType::ALL.to_vec())
}

fn index_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![Just(DType::I32), Just(DType::I64)]
}

/// An index tensor of `dims` whose element at flat position `p` is a valid
/// index, negative or not, into an axis of `extents[p % extents.len()]`
/// elements. An index into an empty axis has no valid value, so such a
/// tensor must be empty.
fn indices(dtype: DType, dims: &[u64], extents: &[u64]) -> BoxedStrategy<HostTensor> {
    let extents = extents.to_vec();
    let shape = Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    let count = usize::try_from(shape.element_count().get()).unwrap();
    vec(0.0..1.0_f64, count)
        .prop_map(move |fractions| {
            let values: Vec<i64> = fractions
                .iter()
                .enumerate()
                .map(|(position, fraction)| {
                    let extent = i64::try_from(extents[position % extents.len()]).unwrap();
                    let span = ptxc_core::convert::int_to_f64(2 * extent);
                    ptxc_core::convert::truncate(fraction * span) - extent
                })
                .collect();
            let data: TensorData = match dtype {
                DType::I32 => {
                    values.iter().map(|&v| i32::try_from(v).unwrap()).collect::<Vec<_>>().into()
                }
                _ => values.into(),
            };
            HostTensor::new(shape.clone(), data).unwrap()
        })
        .boxed()
}

/// `dims` emptied if any of `extents` is 0, since no index into an empty
/// axis is valid.
fn emptied_if_unindexable(mut dims: Vec<u64>, extents: &[u64]) -> Vec<u64> {
    if extents.contains(&0) {
        dims.insert(0, 0);
    }
    dims
}

fn gather_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), index_dtype(), axis_dims(), vec(1_u64..=3, 0..=2))
        .prop_flat_map(|(dtype, index_type, dims, index_dims)| {
            let rank = dims.len();
            (0..rank).prop_flat_map(move |axis| {
                let extent = dims[axis];
                let index_dims = emptied_if_unindexable(index_dims.clone(), &[extent]);
                let data = tensor(dtype, dims.clone(), Domain::Any);
                (Just(axis), data, indices(index_type, &index_dims, &[extent]))
            })
        })
        .prop_map(|(axis, data, indices)| (Op::Gather(Axis::new(axis)), vec![data, indices]))
}

fn gather_elements_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), index_dtype(), axis_dims())
        .prop_flat_map(|(dtype, index_type, dims)| {
            let rank = dims.len();
            (0..rank, vec(0_u64..=3, rank)).prop_flat_map(move |(axis, lengths)| {
                let extent = dims[axis];
                let index_dims: Vec<u64> = dims
                    .iter()
                    .zip(&lengths)
                    .enumerate()
                    .map(
                        |(position, (&dim, &length))| {
                            if position == axis { length } else { length.min(dim) }
                        },
                    )
                    .collect();
                let index_dims = if extent == 0 { vec![0; rank] } else { index_dims };
                let data = tensor(dtype, dims.clone(), Domain::Any);
                (Just(axis), data, indices(index_type, &index_dims, &[extent]))
            })
        })
        .prop_map(|(axis, data, indices)| {
            (Op::GatherElements(Axis::new(axis)), vec![data, indices])
        })
}

/// `GatherND` with `batch` leading axes shared, maybe one more outer axis,
/// and tuples one to all of the remaining axes long.
fn gather_nd_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), index_dtype(), axis_dims(), 0_u64..=3, any::<bool>())
        .prop_flat_map(|(dtype, index_type, dims, free_extent, has_free)| {
            let rank = dims.len();
            (0..rank, 1..=rank).prop_filter_map(
                "tuple past the last axis",
                move |(batch, length)| {
                    let tuple_axes = dims.get(batch..batch + length)?.to_vec();
                    Some(gather_nd_case(
                        dtype,
                        index_type,
                        &dims,
                        batch,
                        &tuple_axes,
                        has_free.then_some(free_extent),
                    ))
                },
            )
        })
        .prop_flat_map(|case| case)
}

/// The data and indices of one `GatherND` case. No index into an empty axis
/// is valid, so then the free outer axis is empty and there are no tuples.
fn gather_nd_case(
    dtype: DType,
    index_type: DType,
    dims: &[u64],
    batch: usize,
    tuple_axes: &[u64],
    free_extent: Option<u64>,
) -> BoxedStrategy<(Op, Vec<HostTensor>)> {
    let free = if tuple_axes.contains(&0) { Some(0) } else { free_extent };
    let length = u64::try_from(tuple_axes.len()).unwrap();
    let index_dims = [&dims[..batch], free.as_slice(), &[length]].concat();
    let data = tensor(dtype, dims.to_vec(), Domain::Any);
    (data, indices(index_type, &index_dims, tuple_axes))
        .prop_map(move |(data, indices)| (Op::GatherNd(Rank::new(batch)), vec![data, indices]))
        .boxed()
}

fn check((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let graph = single_node(op, &inputs);
    check_against_reference(graph, &inputs).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn gather_agrees_with_the_reference(case in gather_cases()) { check(case)?; }

    #[test]
    fn gather_elements_agrees_with_the_reference(case in gather_elements_cases()) { check(case)?; }

    #[test]
    fn gather_nd_agrees_with_the_reference(case in gather_nd_cases()) { check(case)?; }
}
