//! The op families' generators reach the edge cases their docs promise:
//! the lowering tests pass on whatever they draw, so a generator that
//! stopped drawing a case would go unseen.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use ptxc_core::convert::{f64_to_f32, int_to_f64, truncate};
use ptxc_core::{DType, Granularity, HostTensor, QuantDType, TensorData};
use ptxc_graph::{Activation, BinaryOp, Op, PoolKind, Presence};
use ptxc_testkit::families::activation::activation_cases;
use ptxc_testkit::families::attention::attention_cases;
use ptxc_testkit::families::elementwise::binary_cases;
use ptxc_testkit::families::gather::gather_cases;
use ptxc_testkit::families::matmul::matmul_cases;
use ptxc_testkit::families::quant::quantize_cases;
use ptxc_testkit::families::resize::resize_cases;
use ptxc_testkit::families::shape::{graph_of, slice_cases, trilu_cases};
use ptxc_testkit::families::values::Domain;
use ptxc_testkit::families::window::{conv_cases, pool_cases};

/// `count` values drawn from `strategy`, deterministically.
fn drawn<T>(strategy: &impl Strategy<Value = T>, count: usize) -> Vec<T> {
    let mut runner = TestRunner::deterministic();
    (0..count).map(|_| strategy.new_tree(&mut runner).unwrap().current()).collect()
}

/// An integer tensor's values.
fn integers(tensor: &HostTensor) -> Vec<i64> {
    match tensor.data() {
        TensorData::I32(values) => values.iter().map(|&value| i64::from(value)).collect(),
        TensorData::I64(values) => values.clone(),
        other => panic!("{other:?} is not an index tensor"),
    }
}

#[test]
fn hard_sigmoid_is_drawn_with_slopes_of_both_signs() {
    let slopes: Vec<f32> = activation_cases()
        .iter()
        .filter_map(|case| match case.op {
            Op::Activation(Activation::HardSigmoid { alpha, .. }) => Some(alpha.get()),
            _ => None,
        })
        .collect();

    assert!(slopes.iter().any(|&slope| slope < 0.0), "{slopes:?}");
    assert!(slopes.iter().any(|&slope| slope > 0.0), "{slopes:?}");
}

#[test]
fn only_an_integer_raised_to_a_float_exponent_draws_small_integers() {
    let powers = binary_cases().into_iter().filter(|case| case.op == Op::Binary(BinaryOp::Pow));

    for case in powers {
        let [base, exponent] = [case.operands[0], case.operands[1]];
        let small = matches!(base.domain, Domain::SmallInt);
        assert_eq!(small, !base.dtype.is_float() && exponent.dtype.is_float(), "{case:?}");
    }
}

#[test]
fn gathers_draw_indices_of_both_types_on_both_sides_of_zero() {
    let cases = drawn(&gather_cases(), 200);

    let indices: Vec<&HostTensor> = cases.iter().map(|(_, inputs)| &inputs[1]).collect();
    let values: Vec<i64> = indices.iter().flat_map(|tensor| integers(tensor)).collect();
    assert!(indices.iter().any(|tensor| tensor.dtype() == DType::I32));
    assert!(indices.iter().any(|tensor| tensor.dtype() == DType::I64));
    assert!(values.iter().any(|&value| value < 0) && values.iter().any(|&value| value > 0));
}

#[test]
fn products_draw_matrices_as_well_as_vectors() {
    let cases = drawn(&matmul_cases(), 100);

    let operands = cases.iter().flat_map(|(_, inputs)| inputs);
    let ranks: Vec<usize> = operands.map(|operand| operand.shape().dims().len()).collect();
    assert!(ranks.contains(&1) && ranks.iter().any(|&rank| rank >= 2), "{ranks:?}");
}

#[test]
fn integer_quantization_draws_zero_points_away_from_zero() {
    let cases = drawn(&quantize_cases(), 300);

    let zero_points = |grid: QuantDType| -> Vec<i32> {
        let params = cases.iter().filter_map(|(op, _)| match op {
            Op::Quantize(attrs) if attrs.params().dtype() == grid => Some(attrs.params()),
            _ => None,
        });
        params
            .flat_map(|params| match params.granularity() {
                Granularity::PerTensor(map) => vec![*map],
                Granularity::PerAxis { maps, .. } => maps.clone(),
            })
            .map(|map| map.zero_point.get())
            .collect()
    };
    assert!(zero_points(QuantDType::I8).iter().any(|&point| point < 0));
    assert!(zero_points(QuantDType::U8).iter().any(|&point| point > 0));
}

#[test]
fn a_resized_axis_has_the_scale_its_size_implies_or_the_size_its_scale_floors_to() {
    let cases = drawn(&resize_cases(), 200);
    let mut by_size_off_the_list = 0;

    for (op, inputs) in &cases {
        let Op::Resize(attrs) = op else { panic!("{op:?}") };
        for (axis, dim) in attrs.axes.iter().zip(inputs[0].shape().dims()) {
            let wide = |n: u64| int_to_f64(i64::try_from(n).unwrap());
            let (size, scale, input) = (axis.size.get(), axis.scale.get(), dim.get());
            let ratio = if input == 0 { 1.0 } else { f64_to_f32(wide(size) / wide(input)) };
            let floored = truncate(f64::from(scale) * wide(input));
            let implied = scale.total_cmp(&ratio).is_eq();
            let floors = i64::try_from(size).unwrap() == floored;
            assert!(implied || floors, "{axis:?} of {input}");
            by_size_off_the_list += usize::from(![0.5, 0.75, 1.0, 1.5, 2.0, 3.0].contains(&scale));
        }
    }
    assert!(by_size_off_the_list > 0);
}

#[test]
fn slices_with_a_negative_step_reach_the_start_of_their_axis() {
    let cases = drawn(&slice_cases(), 300);

    let reaching = cases.iter().flat_map(|(op, _)| match op {
        Op::Slice(attrs) => attrs.axes.clone(),
        other => panic!("{other:?}"),
    });
    let reaching = reaching.filter(|axis| {
        let step = axis.step.get();
        step < 0 && axis.length.get() == axis.start / step.unsigned_abs() + 1
    });
    assert!(reaching.count() > 0);
}

#[test]
fn trilu_draws_diagonals_on_both_sides_of_the_main_one() {
    let cases = drawn(&trilu_cases(), 100);

    let diagonals: Vec<i64> = cases
        .iter()
        .map(|(op, _)| match op {
            Op::Trilu(attrs) => attrs.diagonal.get(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert!(diagonals.iter().any(|&d| d < 0) && diagonals.iter().any(|&d| d > 0));
}

#[test]
fn a_case_becomes_a_graph_of_its_one_node_on_its_inputs() {
    let [case] = &drawn(&trilu_cases(), 1)[..] else { unreachable!() };

    let (graph, inputs) = graph_of(case.clone());

    assert_eq!(graph.nodes().count(), 1);
    assert_eq!(graph.inputs().len(), inputs.len());
    assert_eq!(inputs.len(), case.1.len());
}

#[test]
fn attention_draws_non_pad_lengths_past_zero() {
    let cases = drawn(&attention_cases(), 300);

    let lengths: Vec<i64> = cases
        .iter()
        .filter_map(|(op, inputs)| match op {
            Op::Attention(attrs) if attrs.nonpad_lengths == Presence::Present => inputs.last(),
            _ => None,
        })
        .flat_map(integers)
        .collect();
    assert!(lengths.iter().any(|&length| length > 0), "{lengths:?}");
}

#[test]
fn a_window_s_padded_input_holds_one_dilated_kernel_and_at_most_3_elements_more() {
    let cases = drawn(&conv_cases(), 300);
    let (mut tight, mut loose_and_dilated) = (0, 0);

    for (op, inputs) in &cases {
        let Op::Conv(attrs) = op else { panic!("{op:?}") };
        let (input, weights) = (inputs[0].shape().dims(), inputs[1].shape().dims());
        for (axis, window) in attrs.axes.iter().enumerate() {
            let (extent, kernel) = (input[2 + axis].get(), weights[2 + axis].get());
            let span = (kernel - 1) * window.dilation.get() + 1;
            let padding = window.pad_begin.get() + window.pad_end.get();
            let least = span.saturating_sub(padding);
            assert!((least..=least + 3).contains(&extent), "{extent} for {kernel} {window:?}");
            tight += usize::from(padding > 0 && least > 0 && extent == least);
            let dilated = window.dilation.get() > 1 && kernel > 1;
            loose_and_dilated += usize::from(dilated && least > 0 && extent == least + 3);
        }
    }
    assert!(tight > 0 && loose_and_dilated > 0, "{tight} {loose_and_dilated}");
}

#[test]
fn max_pools_and_only_max_pools_are_drawn_over_8_bit_integers() {
    let cases = drawn(&pool_cases(), 300);

    let integer = |dtype: DType| matches!(dtype, DType::I8 | DType::U8);
    let mut max_over_integers = 0;
    for (op, inputs) in &cases {
        let Op::Pool(attrs) = op else { panic!("{op:?}") };
        let max = attrs.kind == PoolKind::Max;
        assert!(max || !integer(inputs[0].dtype()), "{op:?} on {:?}", inputs[0].dtype());
        max_over_integers += usize::from(max && integer(inputs[0].dtype()));
    }
    assert!(max_over_integers > 0);
}
