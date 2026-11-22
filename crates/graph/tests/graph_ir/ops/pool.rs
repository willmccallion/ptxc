use crate::common::{f32_type, tensor_type};
use ptxc_core::{DType, Dim, Rank};
use ptxc_graph::{
    Dilation, GlobalPoolKind, PadCounting, Padding, PoolAttrs, PoolAxis, PoolKind, Rounding,
    Stride, TensorType, TypeError, TypeRule, WindowAxis,
};

const fn pool_axis(kernel: u64, stride: u64, pad_end: u64) -> PoolAxis {
    PoolAxis {
        kernel: Dim::new(kernel),
        window: WindowAxis {
            stride: Stride::new(stride).unwrap(),
            dilation: Dilation::new(1).unwrap(),
            pad_begin: Padding::new(0),
            pad_end: Padding::new(pad_end),
        },
    }
}

const fn pool(kind: PoolKind, axes: Vec<PoolAxis>, rounding: Rounding) -> PoolAttrs {
    PoolAttrs { kind, axes, rounding }
}

/// The spatial extent a 1-D max pool gives an input of `input` elements.
fn extent_1d(input: u64, axis: PoolAxis, rounding: Rounding) -> Result<Vec<TensorType>, TypeError> {
    pool(PoolKind::Max, vec![axis], rounding).infer(&[&f32_type(&[1, 1, input])])
}

#[test]
fn floor_rounding_counts_only_whole_windows() {
    assert_eq!(extent_1d(5, pool_axis(2, 2, 0), Rounding::Floor), Ok(vec![f32_type(&[1, 1, 2])]));
}

#[test]
fn ceil_rounding_counts_a_partial_last_window() {
    assert_eq!(extent_1d(5, pool_axis(2, 2, 0), Rounding::Ceil), Ok(vec![f32_type(&[1, 1, 3])]));
}

#[test]
fn ceil_rounding_matches_onnx_maxpool_2d_ceil() {
    let axes = vec![pool_axis(3, 2, 0); 2];
    let input = f32_type(&[1, 1, 4, 4]);

    let ceil = pool(PoolKind::Max, axes.clone(), Rounding::Ceil).infer(&[&input]);
    let floor = pool(PoolKind::Max, axes, Rounding::Floor).infer(&[&input]);

    assert_eq!(ceil, Ok(vec![f32_type(&[1, 1, 2, 2])]));
    assert_eq!(floor, Ok(vec![f32_type(&[1, 1, 1, 1])]));
}

#[test]
fn ceil_rounding_drops_a_window_that_would_start_in_the_end_padding() {
    assert_eq!(extent_1d(4, pool_axis(2, 2, 1), Rounding::Ceil), Ok(vec![f32_type(&[1, 1, 2])]));
    assert_eq!(extent_1d(5, pool_axis(2, 2, 1), Rounding::Ceil), Ok(vec![f32_type(&[1, 1, 3])]));
}

#[test]
fn average_pool_gives_the_same_shape_whatever_it_counts() {
    let input = f32_type(&[2, 3, 6, 6]);

    for counting in [PadCounting::Exclude, PadCounting::Include] {
        let attrs = pool(PoolKind::Average(counting), vec![pool_axis(2, 2, 0); 2], Rounding::Floor);
        assert_eq!(attrs.infer(&[&input]), Ok(vec![f32_type(&[2, 3, 3, 3])]), "{counting:?}");
    }
}

#[test]
fn max_pool_takes_8_bit_integers_but_average_pool_does_not() {
    let axes = vec![pool_axis(2, 1, 0)];
    let max = pool(PoolKind::Max, axes.clone(), Rounding::Floor);
    let average = pool(PoolKind::Average(PadCounting::Exclude), axes, Rounding::Floor);

    for dtype in [DType::I8, DType::U8] {
        let input = tensor_type(dtype, &[1, 1, 4]);
        assert!(max.infer(&[&input]).is_ok(), "{dtype:?}");
        assert_eq!(average.infer(&[&input]), Err(TypeError::UnsupportedDType(dtype)));
    }
    let wide = tensor_type(DType::I32, &[1, 1, 4]);
    assert_eq!(max.infer(&[&wide]), Err(TypeError::UnsupportedDType(DType::I32)));
}

#[test]
fn the_input_rank_must_match_the_window() {
    let attrs = pool(PoolKind::Max, vec![pool_axis(2, 1, 0); 2], Rounding::Floor);

    let error = attrs.infer(&[&f32_type(&[1, 1, 4])]);

    assert_eq!(error, Err(TypeError::Rank { expected: Rank::new(4), actual: Rank::new(3) }));
}

#[test]
fn a_window_larger_than_the_input_is_rejected() {
    let error = extent_1d(2, pool_axis(3, 1, 0), Rounding::Ceil);

    assert_eq!(error, Err(TypeError::WindowTooLarge { input: Dim::new(2), kernel: Dim::new(3) }));
}

#[test]
fn global_pools_keep_every_axis_as_one() {
    for kind in [GlobalPoolKind::Max, GlobalPoolKind::Average] {
        assert_eq!(kind.infer(&[&f32_type(&[2, 3, 5, 7])]), Ok(vec![f32_type(&[2, 3, 1, 1])]));
        assert_eq!(kind.infer(&[&f32_type(&[2, 3])]), Ok(vec![f32_type(&[2, 3])]));
    }
}

#[test]
fn global_pools_need_batch_and_channel_axes_of_floats() {
    let error = GlobalPoolKind::Average.infer(&[&f32_type(&[3])]);

    assert_eq!(error, Err(TypeError::RankBelow { minimum: Rank::new(2), actual: Rank::new(1) }));
    let integer = tensor_type(DType::I8, &[1, 1, 2]);
    assert_eq!(GlobalPoolKind::Max.infer(&[&integer]), Err(TypeError::UnsupportedDType(DType::I8)));
}
