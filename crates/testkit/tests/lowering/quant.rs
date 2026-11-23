//! Quantization onto every grid type, per tensor and per axis, and the
//! quantized products, which compare within one step: their rescaling is
//! f32 where the reference's is f64.

use crate::harness::check_within;
use proptest::prelude::*;
use ptxc_core::{DType, HostTensor};
use ptxc_graph::Op;
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::families::cases::single_node;
use ptxc_testkit::families::quant::{dequantize_cases, qconv_cases, qmatmul_cases, quantize_cases};

fn check(
    (op, inputs): (Op, Vec<HostTensor>),
    tolerance: fn(DType) -> Tolerance,
) -> Result<(), TestCaseError> {
    let graph = single_node(op, &inputs);
    check_within(graph, &inputs, tolerance).map_err(TestCaseError::fail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(crate::cases(1024)))]

    #[test]
    fn quantize_agrees_with_the_reference(case in quantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn dequantize_agrees_with_the_reference(case in dequantize_cases()) {
        check(case, Tolerance::for_dtype)?;
    }

    #[test]
    fn qmatmul_agrees_with_the_reference(case in qmatmul_cases()) {
        check(case, |_| Tolerance::QUANTIZED)?;
    }

    #[test]
    fn qconv_agrees_with_the_reference(case in qconv_cases()) {
        check(case, |_| Tolerance::QUANTIZED)?;
    }
}
