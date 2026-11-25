//! ResNet-50's convolution layers as implicit GEMMs on tensor cores: each
//! distinct layer's channels, kernel, stride and padding, at a spatial size
//! small enough to interpret, in either channel layout. On inputs whose
//! sums are exact they match the loop interpreter bit for bit (D58, D70).

use crate::device::{match_interpreter_exactly, target};
use ptxc_schedule::default_schedule;
use ptxc_testkit::convolution::{Channels, Conv, ConvTypes, typed_conv};
use ptxc_testkit::tensorized::exact_inputs;
use std::sync::Arc;

/// A layer's input and output channels, kernel size, stride and padding.
type Layer = (u64, u64, u64, u64, u64);

/// ResNet-50 v1.5's distinct convolution layers: the stem, then each
/// stage's bottleneck 1 by 1 and 3 by 3 layers, its strided 3 by 3 and
/// its strided projection.
const RESNET50: [Layer; 23] = [
    (3, 64, 7, 2, 3),
    (64, 64, 1, 1, 0),
    (64, 64, 3, 1, 1),
    (64, 256, 1, 1, 0),
    (256, 64, 1, 1, 0),
    (256, 128, 1, 1, 0),
    (128, 128, 3, 2, 1),
    (128, 512, 1, 1, 0),
    (512, 128, 1, 1, 0),
    (128, 128, 3, 1, 1),
    (256, 512, 1, 2, 0),
    (512, 256, 1, 1, 0),
    (256, 256, 3, 2, 1),
    (256, 1024, 1, 1, 0),
    (1024, 256, 1, 1, 0),
    (256, 256, 3, 1, 1),
    (512, 1024, 1, 2, 0),
    (1024, 512, 1, 1, 0),
    (512, 512, 3, 2, 1),
    (512, 2048, 1, 1, 0),
    (2048, 512, 1, 1, 0),
    (512, 512, 3, 1, 1),
    (1024, 2048, 1, 2, 0),
];

/// The output size each layer is run at, to keep the interpreter's work
/// small: four by four positions of one image.
const OUTPUT: u64 = 4;

/// `layer` of one image whose output is `OUTPUT` square.
const fn geometry((channels, features, kernel, stride, pad): Layer) -> Conv {
    let size = (OUTPUT - 1) * stride + kernel - 2 * pad;
    Conv { batch: 1, size, channels, features, kernel, stride, dilation: 1, pad, groups: 1 }
}

/// Layers of `RESNET50` with every kernel size, stride and padding it
/// has, and few enough channels to interpret in a debug build: the stem, a
/// 3 by 3, a strided 3 by 3 and a strided projection.
const REPRESENTATIVE: [Layer; 4] = [RESNET50[0], RESNET50[2], RESNET50[6], RESNET50[10]];

/// Each of `layers`, in either channel layout, run on tensor cores and
/// matched with the interpreter exactly.
fn match_exactly(layers: &[Layer]) {
    let instruction = *target().mma().first().unwrap();
    for &layer in layers {
        for channels in [Channels::First, Channels::Last] {
            let compute = typed_conv(geometry(layer), channels, ConvTypes::of(instruction));
            let kernel = default_schedule(Arc::clone(&compute), &target()).unwrap().lower();
            let case = format!("{layer:?}, {channels:?}, {instruction:?}");

            assert!(kernel.multiplies_on_tensor_cores(), "{case}");
            let _outputs =
                match_interpreter_exactly(&kernel, &exact_inputs(&compute), &case).unwrap();
        }
    }
}

#[test]
fn resnet50s_kinds_of_convolution_layer_match_the_interpreter_exactly_on_tensor_cores() {
    match_exactly(&REPRESENTATIVE);
}

#[test]
#[ignore = "interprets every layer at its channel counts; run with --release --features gpu -- --ignored"]
fn every_resnet50_convolution_layer_matches_the_interpreter_exactly_on_tensor_cores() {
    match_exactly(&RESNET50);
}
