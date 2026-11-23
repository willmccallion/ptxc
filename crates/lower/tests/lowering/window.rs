use crate::elementwise::lowered;
use ptxc_core::{DType, Dim};
use ptxc_graph::{
    ConvAttrs, Dilation, GlobalPoolKind, Groups, Op, PadCounting, Padding, PoolAttrs, PoolAxis,
    PoolKind, Rounding, Stride, WindowAxis,
};
use ptxc_testkit::golden;

const fn window(stride: u64, dilation: u64, pads: (u64, u64)) -> WindowAxis {
    WindowAxis {
        stride: Stride::new(stride).unwrap(),
        dilation: Dilation::new(dilation).unwrap(),
        pad_begin: Padding::new(pads.0),
        pad_end: Padding::new(pads.1),
    }
}

fn pool(kind: PoolKind, kernel: u64, axis: WindowAxis, rounding: Rounding) -> Op {
    Op::Pool(PoolAttrs {
        kind,
        axes: vec![PoolAxis { kernel: Dim::new(kernel), window: axis }],
        rounding,
    })
}

#[test]
fn convolution_masks_padded_taps_and_reads_its_group_channels() {
    let padded = ConvAttrs { axes: vec![window(2, 2, (1, 1))], groups: Groups::new(2).unwrap() };
    let dense = ConvAttrs { axes: vec![window(1, 1, (0, 0))], groups: Groups::new(1).unwrap() };
    let definitions = [
        lowered(
            Op::Conv(padded),
            &[(DType::F16, &[1, 4, 5]), (DType::F16, &[6, 2, 2]), (DType::F16, &[6])],
        ),
        lowered(Op::Conv(dense), &[(DType::F32, &[1, 1, 4]), (DType::F32, &[1, 1, 3])]),
        lowered(
            Op::Conv(ConvAttrs {
                axes: vec![window(1, 1, (1, 1))],
                groups: Groups::new(1).unwrap(),
            }),
            &[(DType::F32, &[1, 1, 0]), (DType::F32, &[2, 1, 1]), (DType::F32, &[2])],
        ),
    ];

    golden::check("window/conv.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn pooling_masks_padding_and_averages_over_counted_taps() {
    let padded = window(2, 1, (1, 1));
    let definitions = [
        lowered(pool(PoolKind::Max, 3, padded, Rounding::Ceil), &[(DType::I8, &[1, 1, 5])]),
        lowered(
            pool(PoolKind::Average(PadCounting::Exclude), 3, padded, Rounding::Floor),
            &[(DType::F32, &[1, 1, 5])],
        ),
        lowered(
            pool(PoolKind::Average(PadCounting::Include), 2, window(1, 1, (0, 0)), Rounding::Floor),
            &[(DType::F32, &[1, 1, 3])],
        ),
        lowered(Op::GlobalPool(GlobalPoolKind::Average), &[(DType::F16, &[1, 2, 3, 3])]),
    ];

    golden::check("window/pool.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn pooling_masks_exactly_the_windows_that_leave_the_input() {
    let empty = pool(PoolKind::Max, 2, window(1, 1, (1, 1)), Rounding::Floor);
    let end_padded = window(1, 1, (0, 1));
    let counted = window(2, 1, (0, 2));
    let definitions = [
        lowered(empty, &[(DType::F32, &[1, 1, 0])]),
        lowered(
            pool(PoolKind::Average(PadCounting::Exclude), 2, end_padded, Rounding::Floor),
            &[(DType::F32, &[1, 1, 3])],
        ),
        lowered(
            pool(PoolKind::Average(PadCounting::Include), 3, counted, Rounding::Ceil),
            &[(DType::F32, &[1, 1, 4])],
        ),
    ];

    golden::check("window/edges.txt", &definitions.join("\n")).unwrap();
}
