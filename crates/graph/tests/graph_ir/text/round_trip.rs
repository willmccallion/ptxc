use super::cases;
use super::{assert_round_trips, single_op};
use ptxc_graph::{Op, TensorType};

fn check_all(cases: Vec<(Op, Vec<TensorType>)>) {
    for (op, types) in cases {
        assert_round_trips(&single_op(op, &types), &[]);
    }
}

#[test]
fn every_elementwise_op_round_trips() {
    check_all(cases::elementwise());
}

#[test]
fn every_activation_round_trips_with_awkward_floats() {
    check_all(cases::activation());
}

#[test]
fn matrix_products_convolutions_and_pools_round_trip() {
    check_all(cases::compute());
}

#[test]
fn reductions_and_normalisations_round_trip() {
    check_all(cases::reduce_and_norm());
}

#[test]
fn shape_and_indexing_ops_round_trip() {
    check_all(cases::shape_ops());
}

#[test]
fn quantization_ops_round_trip() {
    check_all(cases::quantization());
}

#[test]
fn every_resize_shape_round_trips() {
    check_all(cases::resize());
}

#[test]
fn attention_round_trips_with_and_without_its_options() {
    check_all(cases::attention());
}

#[test]
fn the_cases_cover_every_op_variant() {
    let all: Vec<Op> = [
        cases::elementwise(),
        cases::activation(),
        cases::compute(),
        cases::reduce_and_norm(),
        cases::shape_ops(),
        cases::quantization(),
        cases::resize(),
        cases::attention(),
    ]
    .into_iter()
    .flatten()
    .map(|(op, _)| op)
    .collect();

    let variants: std::collections::HashSet<_> = all.iter().map(std::mem::discriminant).collect();

    assert_eq!(variants.len(), VARIANT_COUNT, "a variant of Op has no round-trip case");
}

/// The number of `Op` variants; update with the enum.
const VARIANT_COUNT: usize = 39;
