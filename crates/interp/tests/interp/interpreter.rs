//! Evaluation order, inputs, constants and outputs.

use crate::builders::{assert_close, f32s, i64s, run, try_run};
use ptxc_interp::InterpError;

const CHAIN: &str = "graph(%x: f32[2], %y: f32[2]) {
  %c = const f32[2] [10.0, 20.0]
  %s = add %x, %c : f32[2]
  %p = mul %s, %y : f32[2]
  return %p, %s, %x, %c
}
";

#[test]
fn nodes_run_after_their_operands_and_outputs_keep_their_order() {
    let outputs = run(CHAIN, &[f32s(&[2], &[1.0, 2.0]), f32s(&[2], &[3.0, 4.0])]);

    assert_eq!(outputs.len(), 4);
    assert_close(&outputs[0], &[2], &[33.0, 88.0]);
    assert_close(&outputs[1], &[2], &[11.0, 22.0]);
    assert_close(&outputs[2], &[2], &[1.0, 2.0]);
    assert_close(&outputs[3], &[2], &[10.0, 20.0]);
}

#[test]
fn the_wrong_number_of_inputs_is_an_error() {
    let error = try_run(CHAIN, &[f32s(&[2], &[1.0, 2.0])]).unwrap_err();

    assert!(matches!(error, InterpError::InputCount { expected: 2, actual: 1 }), "{error}");
}

#[test]
fn an_input_of_the_wrong_type_is_an_error() {
    let wrong_shape =
        try_run(CHAIN, &[f32s(&[2], &[1.0, 2.0]), f32s(&[3], &[1.0, 2.0, 3.0])]).unwrap_err();
    let wrong_dtype = try_run(CHAIN, &[i64s(&[2], &[1, 2]), f32s(&[2], &[1.0, 2.0])]).unwrap_err();

    assert_eq!(wrong_shape.to_string(), "input 1 is f32[3], but the graph declares f32[2]");
    assert!(matches!(wrong_dtype, InterpError::InputType { index: 0, .. }), "{wrong_dtype}");
}
