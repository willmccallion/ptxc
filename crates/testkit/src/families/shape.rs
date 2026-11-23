//! Every data-movement op at random shapes and attributes, on every element
//! type.

use super::cases::single_node;
use super::values::{Domain, axis_dims, operand_dims, output_dims, shape, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{Axis, DType, Dim, HostTensor};
use ptxc_graph::{
    AxisPadding, DiagonalOffset, Graph, Op, PadAttrs, PadMode, Padding, RepeatCount, SliceAttrs,
    SliceAxis, SliceStep, SplitAttrs, TileAttrs, TransposeAttrs, TriangularPart, TriluAttrs,
    TypeRule,
};

fn any_dtype() -> impl Strategy<Value = DType> {
    select(DType::ALL.to_vec())
}

fn any_tensor(dtype: DType, dims: Vec<u64>) -> BoxedStrategy<HostTensor> {
    tensor(dtype, dims, Domain::Any)
}

/// The graph of `op` on `inputs`, if its type rule accepts them.
fn admitted(op: &Op, inputs: &[HostTensor]) -> bool {
    let types: Vec<_> = inputs
        .iter()
        .map(|input| {
            ptxc_graph::TensorType::new(
                input.dtype(),
                input.shape().clone(),
                ptxc_graph::Layout::RowMajor,
            )
        })
        .collect();
    op.infer(&types.iter().collect::<Vec<_>>()).is_ok()
}

/// A single-node graph of `op` over `inputs`, with the inputs.
pub fn graph_of((op, inputs): (Op, Vec<HostTensor>)) -> (Graph, Vec<HostTensor>) {
    (single_node(op, &inputs), inputs)
}

/// Shapes holding as many elements as `dims`: flattened, reversed, with a 1
/// added at either end, and with each adjacent pair merged.
fn regroupings(dims: &[u64]) -> Vec<Vec<u64>> {
    let count = dims.iter().product::<u64>();
    let mut shapes = vec![vec![count], dims.iter().rev().copied().collect()];
    shapes.push([&[1], dims].concat());
    shapes.push([dims, &[1]].concat());
    for pair in 0..dims.len().saturating_sub(1) {
        let merged = dims[pair] * dims[pair + 1];
        shapes.push([&dims[..pair], &[merged], &dims[pair + 2..]].concat());
    }
    if count == 1 {
        shapes.push(Vec::new());
    }
    shapes
}

/// `Reshape` to any regrouping of the input's extents.
pub fn reshape_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), output_dims())
        .prop_flat_map(|(dtype, dims)| (select(regroupings(&dims)), any_tensor(dtype, dims)))
        .prop_map(|(target, input)| (Op::Reshape(shape(&target)), vec![input]))
}

/// `Expand` to a shape the input broadcasts to.
pub fn expand_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), output_dims())
        .prop_flat_map(|(dtype, dims)| {
            let input = operand_dims(dims.clone()).prop_flat_map(move |d| any_tensor(dtype, d));
            (operand_dims(dims), input)
        })
        .prop_map(|(target, input)| (Op::Expand(shape(&target)), vec![input]))
}

/// `Transpose` by any permutation.
pub fn transpose_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), output_dims())
        .prop_flat_map(|(dtype, dims)| {
            let permutation = Just((0..dims.len()).collect::<Vec<_>>()).prop_shuffle();
            (permutation, any_tensor(dtype, dims))
        })
        .prop_map(|(permutation, input)| {
            let permutation = permutation.into_iter().map(Axis::new).collect();
            (Op::Transpose(TransposeAttrs { permutation }), vec![input])
        })
}

/// `Tile` by small repeats, zero included.
pub fn tile_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), output_dims())
        .prop_flat_map(|(dtype, dims)| (vec(0_u64..=3, dims.len()), any_tensor(dtype, dims)))
        .prop_map(|(repeats, input)| {
            let repeats = repeats.into_iter().map(RepeatCount::new).collect();
            (Op::Tile(TileAttrs { repeats }), vec![input])
        })
}

/// One to three operands agreeing on every axis but `axis`, where each has
/// 0 to 3 elements.
pub fn concat_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), axis_dims()).prop_flat_map(|(dtype, dims)| {
        let rank = dims.len();
        (0..rank, vec(0_u64..=3, 1..=3)).prop_flat_map(move |(axis, extents)| {
            let inputs: Vec<_> = extents
                .into_iter()
                .map(|extent| {
                    let mut own = dims.clone();
                    own[axis] = extent;
                    any_tensor(dtype, own)
                })
                .collect();
            (Just(Op::Concat(Axis::new(axis))), inputs)
        })
    })
}

/// A slice of an axis of `extent`: a start inside it, a step of either
/// sign, and any length that stays inside.
fn slice_axis(axis: usize, extent: u64) -> BoxedStrategy<Option<SliceAxis>> {
    if extent == 0 {
        return Just(None).boxed();
    }
    let step = prop_oneof![-3_i64..=-1, 1_i64..=3];
    let sliced = (0..extent, step).prop_flat_map(move |(start, step)| {
        let longest = if step > 0 {
            (extent - start).div_ceil(step.unsigned_abs())
        } else {
            start / step.unsigned_abs() + 1
        };
        (0..=longest).prop_map(move |length| {
            Some(SliceAxis {
                axis: Axis::new(axis),
                start,
                step: SliceStep::new(step).unwrap(),
                length: Dim::new(length),
            })
        })
    });
    prop_oneof![1 => Just(None), 3 => sliced].boxed()
}

/// `Slice` with any starts, ends and steps, negative and clamped ones
/// included.
pub fn slice_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), output_dims())
        .prop_flat_map(|(dtype, dims)| {
            let axes: Vec<_> =
                dims.iter().enumerate().map(|(axis, &extent)| slice_axis(axis, extent)).collect();
            (axes, any_tensor(dtype, dims))
        })
        .prop_map(|(axes, input)| {
            let axes = axes.into_iter().flatten().collect();
            (Op::Slice(SliceAttrs { axes }), vec![input])
        })
}

/// An axis cut into one to three consecutive pieces, some maybe empty.
pub fn split_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    (any_dtype(), axis_dims()).prop_flat_map(|(dtype, dims)| {
        let rank = dims.len();
        (0..rank).prop_flat_map(move |axis| {
            let extent = dims[axis];
            (vec(0..=extent, 0..=2), any_tensor(dtype, dims.clone())).prop_map(
                move |(mut cuts, input)| {
                    cuts.sort_unstable();
                    let bounds = [&[0], &cuts[..], &[extent]].concat();
                    let sizes = bounds.windows(2).map(|w| Dim::new(w[1] - w[0])).collect();
                    (Op::Split(SplitAttrs { axis: Axis::new(axis), sizes }), vec![input])
                },
            )
        })
    })
}

/// `Pad` in every mode, negative pads included.
pub fn pad_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let mode = prop_oneof![
        Just(PadMode::Constant),
        Just(PadMode::Reflect),
        Just(PadMode::Edge),
        Just(PadMode::Wrap),
    ];
    (any_dtype(), output_dims(), mode)
        .prop_flat_map(|(dtype, dims, mode)| {
            let rank = dims.len();
            let value = any_tensor(dtype, Vec::new());
            (Just(mode), vec((0_u64..=3, 0_u64..=3), rank), any_tensor(dtype, dims), value)
        })
        .prop_map(|(mode, widths, input, value)| {
            let axes = widths
                .into_iter()
                .map(|(begin, end)| AxisPadding {
                    begin: Padding::new(begin),
                    end: Padding::new(end),
                })
                .collect();
            let inputs = if mode == PadMode::Constant { vec![input, value] } else { vec![input] };
            (Op::Pad(PadAttrs { mode, axes }), inputs)
        })
        .prop_filter("pads an empty axis", |(op, inputs)| admitted(op, inputs))
}

/// `Trilu`, upper and lower, at any diagonal offset.
pub fn trilu_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let part = prop_oneof![Just(TriangularPart::Upper), Just(TriangularPart::Lower)];
    let dims = vec(prop_oneof![1 => Just(0_u64), 12 => 1_u64..=4], 2..=3);
    (any_dtype(), dims, part, -4_i64..=4)
        .prop_flat_map(|(dtype, dims, part, diagonal)| {
            (Just(part), Just(diagonal), any_tensor(dtype, dims))
        })
        .prop_map(|(part, diagonal, input)| {
            let attrs = TriluAttrs { part, diagonal: DiagonalOffset::new(diagonal) };
            (Op::Trilu(attrs), vec![input])
        })
}
