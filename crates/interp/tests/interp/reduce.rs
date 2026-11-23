//! Reductions and arg-reductions, including empty reductions, and running
//! sums.

use crate::builders::{assert_close, assert_exact, bools, f32s, i64s, op, tensor};

const INF: f64 = f64::INFINITY;

/// `[[1, 2, 3], [4, 5, 6]]`.
fn x() -> ptxc_core::HostTensor {
    f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

fn reduce(
    name: &str,
    axes: &str,
    keep: &str,
    result: &str,
    input: ptxc_core::HostTensor,
) -> ptxc_core::HostTensor {
    op(&format!("{name}<axes={axes}, keep_dims={keep}>"), result, &[input])
}

#[test]
fn sums_means_and_products_along_axes_kept_or_dropped() {
    assert_close(&reduce("reduce_sum", "[1]", "keep", "f32[2, 1]", x()), &[2, 1], &[6.0, 15.0]);
    assert_close(&reduce("reduce_mean", "[0]", "drop", "f32[3]", x()), &[3], &[2.5, 3.5, 4.5]);
    assert_close(
        &reduce("reduce_mean", "[0]", "keep", "f32[1, 3]", x()),
        &[1, 3],
        &[2.5, 3.5, 4.5],
    );
    assert_close(&reduce("reduce_prod", "[1]", "drop", "f32[2]", x()), &[2], &[6.0, 120.0]);
    assert_close(&reduce("reduce_max", "[0, 1]", "drop", "f32[]", x()), &[], &[6.0]);
    assert_close(&reduce("reduce_min", "[0, 1]", "keep", "f32[1, 1]", x()), &[1, 1], &[1.0]);
}

#[test]
fn norms_and_logarithmic_reductions() {
    let signed = || f32s(&[2, 3], &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]);

    assert_close(
        &reduce("reduce_sum_square", "[1]", "drop", "f32[2]", signed()),
        &[2],
        &[14.0, 77.0],
    );
    assert_close(&reduce("reduce_l1", "[1]", "drop", "f32[2]", signed()), &[2], &[6.0, 15.0]);
    assert_close(
        &reduce("reduce_l2", "[1]", "drop", "f32[2]", signed()),
        &[2],
        &[14f64.sqrt(), 77f64.sqrt()],
    );
    assert_close(
        &reduce("reduce_log_sum", "[1]", "drop", "f32[2]", x()),
        &[2],
        &[6f64.ln(), 15f64.ln()],
    );
    assert_close(
        &reduce("reduce_log_sum_exp", "[1]", "drop", "f32[2]", x()),
        &[2],
        &[3.407_605_964_444_38, 6.407_605_964_444_38],
    );
    let large = f32s(&[2], &[1000.0, 1000.0]);
    assert_close(
        &reduce("reduce_log_sum_exp", "[0]", "drop", "f32[]", large),
        &[],
        &[1000.0 + 2f64.ln()],
    );
}

#[test]
fn empty_reductions_give_each_op_its_identity() {
    let empty = || f32s(&[2, 0], &[]);
    let over_empty = |name: &str| reduce(name, "[1]", "drop", "f32[2]", empty());

    assert_close(&over_empty("reduce_sum"), &[2], &[0.0, 0.0]);
    assert_close(&over_empty("reduce_prod"), &[2], &[1.0, 1.0]);
    assert_close(&over_empty("reduce_max"), &[2], &[-INF, -INF]);
    assert_close(&over_empty("reduce_min"), &[2], &[INF, INF]);
    assert_close(&over_empty("reduce_mean"), &[2], &[f64::NAN, f64::NAN]);
    assert_close(&over_empty("reduce_log_sum"), &[2], &[-INF, -INF]);
    assert_close(&over_empty("reduce_log_sum_exp"), &[2], &[-INF, -INF]);
    assert_close(&over_empty("reduce_l2"), &[2], &[0.0, 0.0]);
}

#[test]
fn nan_propagates_through_max_and_min() {
    let with_nan = || f32s(&[2], &[1.0, f32::NAN]);

    assert_close(&reduce("reduce_max", "[0]", "drop", "f32[]", with_nan()), &[], &[f64::NAN]);
    assert_close(&reduce("reduce_min", "[0]", "drop", "f32[]", with_nan()), &[], &[f64::NAN]);
}

#[test]
fn integer_reductions_wrap_or_truncate_and_empty_ones_take_the_type_extremes() {
    let large = || tensor(&[2, 2], vec![2_000_000_000_i32, 2_000_000_000, -3, -4]);
    let extreme = tensor(&[2, 2], vec![i32::MIN, 1, -5, 2]);
    let empty = || tensor(&[1, 0], Vec::<i32>::new());
    let to_i32 = |name: &str, input| reduce(name, "[1]", "drop", "i32[2]", input);

    assert_exact(&to_i32("reduce_sum", large()), &[2], &[-294_967_296, -7]);
    assert_exact(&to_i32("reduce_prod", large()), &[2], &[-1_651_507_200, 12]);
    assert_exact(&to_i32("reduce_sum_square", large()), &[2], &[991_952_896, 25]);
    assert_exact(&to_i32("reduce_l1", extreme), &[2], &[-2_147_483_647, 7]);
    assert_exact(&to_i32("reduce_mean", tensor(&[2, 2], vec![1_i32, 2, -3, -2])), &[2], &[1, -2]);
    assert_exact(&reduce("reduce_l2", "[1]", "drop", "i64[1]", i64s(&[1, 2], &[3, 4])), &[1], &[5]);
    assert_exact(
        &reduce("reduce_max", "[1]", "drop", "i32[1]", empty()),
        &[1],
        &[i64::from(i32::MIN)],
    );
    assert_exact(
        &reduce("reduce_min", "[1]", "drop", "i32[1]", empty()),
        &[1],
        &[i64::from(i32::MAX)],
    );
    let small = tensor(&[2, 2], vec![100_i8, -100, -3, 4]);
    assert_exact(&reduce("reduce_max", "[0]", "drop", "i8[2]", small.clone()), &[2], &[100, 4]);
    assert_exact(&reduce("reduce_min", "[1]", "drop", "i8[2]", small), &[2], &[-100, -3]);
}

#[test]
fn bool_max_is_any_and_min_is_all() {
    let flags = || bools(&[2, 2], &[true, false, false, false]);

    assert_exact(&reduce("reduce_max", "[1]", "drop", "bool[2]", flags()), &[2], &[1, 0]);
    assert_exact(&reduce("reduce_min", "[0]", "drop", "bool[2]", flags()), &[2], &[0, 0]);
    assert_exact(
        &reduce("reduce_min", "[1]", "drop", "bool[1]", bools(&[1, 2], &[true, true])),
        &[1],
        &[1],
    );
}

fn arg(
    name: &str,
    axis: usize,
    keep: &str,
    tie: &str,
    result: &str,
    input: ptxc_core::HostTensor,
) -> ptxc_core::HostTensor {
    op(&format!("{name}<axis={axis}, keep_dims={keep}, tie={tie}>"), result, &[input])
}

#[test]
fn arg_reductions_pick_the_first_or_last_extreme() {
    let x = || f32s(&[2, 3], &[1.0, 3.0, 3.0, 2.0, 0.0, -1.0]);

    assert_exact(&arg("arg_max", 1, "keep", "first", "i64[2, 1]", x()), &[2, 1], &[1, 0]);
    assert_exact(&arg("arg_max", 1, "drop", "last", "i64[2]", x()), &[2], &[2, 0]);
    assert_exact(&arg("arg_min", 1, "drop", "first", "i64[2]", x()), &[2], &[0, 2]);
    assert_exact(
        &arg(
            "arg_min",
            0,
            "drop",
            "last",
            "i64[3]",
            f32s(&[2, 3], &[1.0, 0.0, 5.0, 1.0, 0.0, 2.0]),
        ),
        &[3],
        &[1, 1, 1],
    );
    assert_exact(&arg("arg_max", 0, "drop", "first", "i64[3]", x()), &[3], &[1, 0, 0]);
    assert_exact(
        &arg("arg_max", 0, "drop", "first", "i64[3]", i64s(&[2, 3], &[5, 1, 7, 6, 1, 2])),
        &[3],
        &[1, 0, 0],
    );
}

#[test]
fn nan_wins_arg_reductions_at_its_first_position() {
    let with_nan = || f32s(&[4], &[1.0, f32::NAN, f32::NAN, 0.0]);

    assert_exact(&arg("arg_max", 0, "drop", "first", "i64[]", with_nan()), &[], &[1]);
    assert_exact(&arg("arg_min", 0, "drop", "last", "i64[]", with_nan()), &[], &[1]);
}

#[test]
fn reducing_a_middle_axis_keeps_the_others_in_order() {
    let cube = f32s(&[2, 2, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    assert_close(
        &reduce("reduce_sum", "[1]", "drop", "f32[2, 2]", cube),
        &[2, 2],
        &[2.0, 4.0, 10.0, 12.0],
    );
}

#[test]
fn arg_min_breaks_ties_by_first_or_last() {
    let tied = || f32s(&[3], &[3.0, 1.0, 1.0]);

    assert_exact(&arg("arg_min", 0, "drop", "first", "i64[]", tied()), &[], &[1]);
    assert_exact(&arg("arg_min", 0, "drop", "last", "i64[]", tied()), &[], &[2]);
}

fn cum_sum(attrs: &str, input: ptxc_core::HostTensor, result: &str) -> ptxc_core::HostTensor {
    op(&format!("cum_sum<{attrs}>"), result, &[input])
}

#[test]
fn cum_sum_includes_or_excludes_each_element_going_either_way() {
    let sums = |inclusion, direction| {
        let attrs = format!("axis=1, inclusion={inclusion}, direction={direction}");
        cum_sum(&attrs, x(), "f32[2, 3]")
    };

    assert_close(&sums("inclusive", "forward"), &[2, 3], &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    assert_close(&sums("exclusive", "forward"), &[2, 3], &[0.0, 1.0, 3.0, 0.0, 4.0, 9.0]);
    assert_close(&sums("inclusive", "reverse"), &[2, 3], &[6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);
    assert_close(&sums("exclusive", "reverse"), &[2, 3], &[5.0, 3.0, 0.0, 11.0, 6.0, 0.0]);
}

#[test]
fn cum_sum_runs_along_its_axis_only() {
    let down = cum_sum("axis=0, inclusion=inclusive, direction=forward", x(), "f32[2, 3]");

    assert_close(&down, &[2, 3], &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn integer_cum_sums_wrap_in_their_type() {
    let attrs = "axis=0, inclusion=inclusive, direction=forward";
    let narrow = cum_sum(attrs, tensor(&[3], vec![i32::MAX, 1, 1]), "i32[3]");
    let wide = cum_sum(attrs, i64s(&[2], &[i64::MAX, 1]), "i64[2]");

    assert_exact(
        &narrow,
        &[3],
        &[i64::from(i32::MAX), i64::from(i32::MIN), i64::from(i32::MIN) + 1],
    );
    assert_exact(&wide, &[2], &[i64::MAX, i64::MIN]);
}
