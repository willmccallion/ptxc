//! Split-K under the default schedule: which products split, into how
//! many slices, and how the two stages are scheduled (D75).

use crate::matmul::{PLAIN, product};
use crate::{launch_dims, stage};
use ptxc_core::DType;
use ptxc_schedule::{default_schedule, split_k};
use ptxc_target::{SmVersion, Target, preset};
use std::sync::Arc;

fn target() -> Target {
    preset(SmVersion::SM_89).unwrap()
}

#[test]
fn a_skinny_product_splits_into_slices_its_partials_tile_on_tensor_cores() {
    // One block unsplit; 3072 splits into at most 24 slices of 128.
    let compute = product(DType::F16, [1, 4, 64, 3072], PLAIN);

    let split = split_k(&compute, &target()).unwrap().unwrap();

    assert_eq!(split.partial.output_shape().dims().first().map(|dim| dim.get()), Some(24));
    let partial = default_schedule(Arc::new(split.partial), &target()).unwrap().lower();
    let combine = default_schedule(Arc::new(split.combine), &target()).unwrap().lower();
    assert!(partial.multiplies_on_tensor_cores());
    assert_eq!(launch_dims(partial.launch()), ([24, 1, 1], [32, 2, 2]));
    // 256 outputs, a thread each, each adding its 24 slices.
    assert_eq!(launch_dims(combine.launch()), ([1, 1, 1], [256, 1, 1]));
}

#[test]
fn a_product_that_fills_the_gpu_or_is_no_product_is_not_split() {
    let filling = product(DType::F16, [1, 1024, 512, 1024], PLAIN);
    let copy = stage(&[4], 0, None);

    assert!(split_k(&filling, &target()).unwrap().is_none());
    assert!(split_k(&copy, &target()).unwrap().is_none());
}
