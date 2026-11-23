//! Convolution and pooling over random windows: strides, dilations,
//! padding, ceil rounding and channel groups.

use crate::cases::single_node;
use crate::harness::check_within;
use crate::values::{Domain, tensor};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use ptxc_core::{DType, Dim, ElementCount, HostTensor};
use ptxc_graph::{
    ConvAttrs, Dilation, GlobalPoolKind, Groups, Layout, Op, PadCounting, Padding, PoolAttrs,
    PoolAxis, PoolKind, Rounding, Stride, TensorType, TypeRule, WindowAxis,
};
use ptxc_testkit::compare::Tolerance;

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::Bf16];

fn window_axis() -> impl Strategy<Value = WindowAxis> {
    (1_u64..=2, 1_u64..=2, 0_u64..=2, 0_u64..=2).prop_map(|(stride, dilation, begin, end)| {
        WindowAxis {
            stride: Stride::new(stride).unwrap(),
            dilation: Dilation::new(dilation).unwrap(),
            pad_begin: Padding::new(begin),
            pad_end: Padding::new(end),
        }
    })
}

/// One spatial axis: an input extent that, padded, holds at least one
/// dilated kernel, the kernel, and how the window moves.
fn spatial_axis() -> impl Strategy<Value = (u64, u64, WindowAxis)> {
    (1_u64..=3, window_axis(), 0_u64..=3).prop_map(|(kernel, window, extra)| {
        let span = (kernel - 1) * window.dilation.get() + 1;
        let padding = window.pad_begin.get() + window.pad_end.get();
        (span.saturating_sub(padding) + extra, kernel, window)
    })
}

/// Whether `op`'s type rule accepts inputs typed as `inputs`.
fn admitted(op: &Op, inputs: &[HostTensor]) -> bool {
    let types: Vec<TensorType> = inputs
        .iter()
        .map(|input| TensorType::new(input.dtype(), input.shape().clone(), Layout::RowMajor))
        .collect();
    op.infer(&types.iter().collect::<Vec<_>>()).is_ok()
}

fn rounding() -> impl Strategy<Value = Rounding> {
    prop_oneof![Just(Rounding::Floor), Just(Rounding::Ceil)]
}

fn pool_kind() -> impl Strategy<Value = PoolKind> {
    prop_oneof![
        Just(PoolKind::Max),
        Just(PoolKind::Average(PadCounting::Exclude)),
        Just(PoolKind::Average(PadCounting::Include)),
    ]
}

/// A pool on a type its kind takes: max pooling also takes 8-bit integers.
pub(crate) fn pool_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let leading = [0_u64..=2, 1_u64..=2];
    let spatial = vec(spatial_axis(), 1..=2);
    let kind_and_type = pool_kind().prop_flat_map(|kind| {
        let mut types = FLOATS.to_vec();
        if kind == PoolKind::Max {
            types.extend([DType::I8, DType::U8]);
        }
        (Just(kind), select(types))
    });
    (kind_and_type, rounding(), leading, spatial)
        .prop_flat_map(|((kind, dtype), rounding, [n, c], spatial)| {
            let dims = [vec![n, c], spatial.iter().map(|&(extent, ..)| extent).collect()].concat();
            let axes = spatial
                .iter()
                .map(|&(_, kernel, window)| PoolAxis { kernel: Dim::new(kernel), window })
                .collect();
            let op = Op::Pool(PoolAttrs { kind, axes, rounding });
            (Just(op), tensor(dtype, dims, Domain::Moderate).prop_map(|input| vec![input]))
        })
        .prop_filter("rejected by the type rule", |(op, inputs)| admitted(op, inputs))
}

pub(crate) fn global_pool_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let kind = prop_oneof![Just(GlobalPoolKind::Max), Just(GlobalPoolKind::Average)];
    (select(FLOATS.to_vec()), kind, vec(0_u64..=3, 3..=4)).prop_flat_map(|(dtype, kind, dims)| {
        (Just(Op::GlobalPool(kind)), tensor(dtype, dims, Domain::Moderate).prop_map(|x| vec![x]))
    })
}

/// A convolution of `groups * group_channels` input channels into `groups *
/// per_group` outputs, with an optional bias.
pub(crate) fn conv_cases() -> impl Strategy<Value = (Op, Vec<HostTensor>)> {
    let channels = (1_u64..=2, 1_u64..=2, 1_u64..=2);
    let spatial = vec(spatial_axis(), 1..=2);
    (select(FLOATS.to_vec()), 1_u64..=2, channels, spatial, any::<bool>())
        .prop_flat_map(|(dtype, batch, (groups, group_channels, per_group), spatial, has_bias)| {
            let extents = spatial.iter().map(|&(extent, ..)| extent);
            let input = [vec![batch, groups * group_channels], extents.collect()].concat();
            let kernel = spatial.iter().map(|&(_, kernel, _)| kernel);
            let weights = [vec![groups * per_group, group_channels], kernel.collect()].concat();
            let axes = spatial.iter().map(|&(.., window)| window).collect();
            let op = Op::Conv(ConvAttrs { axes, groups: Groups::new(groups).unwrap() });
            let bias = vec(
                tensor(dtype, vec![groups * per_group], Domain::Moderate),
                usize::from(has_bias),
            );
            (
                Just(op),
                tensor(dtype, input, Domain::Moderate),
                tensor(dtype, weights, Domain::Moderate),
                bias,
            )
        })
        .prop_map(|(op, input, weights, bias)| (op, [vec![input, weights], bias].concat()))
        .prop_filter("rejected by the type rule", |(op, inputs)| admitted(op, inputs))
}

fn check((op, inputs): (Op, Vec<HostTensor>)) -> Result<(), TestCaseError> {
    let length: u64 = inputs.iter().map(|input| input.shape().element_count().get()).sum();
    let graph = single_node(op, &inputs);
    let length = ElementCount::new(length.max(1));
    let tolerance = |dtype| Tolerance::for_dtype(dtype).for_reduction(length);
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn pooling_agrees_with_the_reference(case in pool_cases()) { check(case)?; }

    #[test]
    fn global_pooling_agrees_with_the_reference(case in global_pool_cases()) { check(case)?; }

    #[test]
    fn convolution_agrees_with_the_reference(case in conv_cases()) { check(case)?; }
}
