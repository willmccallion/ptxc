//! Vectorized loops that copy or clear runs of 8 or 16 bytes become one
//! vector access, where their indices step by one from an aligned start;
//! every other vectorized loop stays serial.

use crate::ptxas::assemble;
use crate::target;
use ptxc_codegen::generate;
use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{
    ComputeBuilder, ComputeName, ExprId, Primitive, Scalar, Schedule, SplitFactor, TensorExpr,
    TensorName, ThreadAxis, VarName,
};
use std::sync::Arc;

/// What `y[b, i]` holds for `b < 4` and `i < 8 * width`.
#[derive(Clone, Copy)]
enum Element {
    /// `x[b, 2 * i]` scaled by `stride`, plus `offset`, read from `x`.
    Read { stride: i64, offset: i64, by_row: bool },
    /// A constant.
    Fill(Scalar),
    /// `y` and a second output `z` both read at `[b, i]`.
    Twice,
}

/// The PTX of `y[b, i]` of `dtype`, `i` split into runs of `width`
/// vectorized, `b` on `threadIdx.y` and the runs on `threadIdx.x`.
fn vectorized(dtype: DType, width: u64, element: Element) -> String {
    let columns = 8 * width;
    let shape = |dims: &[u64]| Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    let mut compute = ComputeBuilder::new(ComputeName::new("copy"));
    let x = compute.input(TensorName::new("x"), dtype, shape(&[4, 4 * columns]));
    let (_, b) = compute.spatial(VarName::new("b"), Dim::new(4));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(columns));
    let expr = match element {
        Element::Read { stride, offset, by_row } => {
            let index = read_index(compute.exprs(), [b, i], [stride, offset], by_row);
            TensorExpr::load(x, index)
        }
        Element::Fill(value) => TensorExpr::Const(value),
        Element::Twice => {
            let read = TensorExpr::load(x, vec![b, i]);
            compute.output(TensorName::new("z"), read.clone());
            read
        }
    };
    compute.output(TensorName::new("y"), expr);
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let parts = schedule.split(nest[1], vec![SplitFactor::new(width).unwrap()]).unwrap();
    for (loop_, axis) in [(nest[0], ThreadAxis::ThreadY), (parts[0], ThreadAxis::ThreadX)] {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    schedule.apply(Primitive::Vectorize { loop_: parts[1] }).unwrap();
    let kernel = schedule.validate(&target()).unwrap().lower();
    let module = generate(&kernel, &target()).unwrap();
    assemble(&module);
    module.to_string()
}

/// `[b, i * stride + offset]`, or `[0, b + i * stride + offset]` by row.
fn read_index(
    exprs: &mut ptxc_loop::ExprArena,
    [b, i]: [ExprId; 2],
    [stride, offset]: [i64; 2],
    by_row: bool,
) -> Vec<ExprId> {
    let (stride, offset) = (exprs.constant(stride), exprs.constant(offset));
    let scaled = exprs.mul(i, stride);
    let column = exprs.add(scaled, offset);
    if by_row {
        let zero = exprs.constant(0);
        vec![zero, exprs.add(b, column)]
    } else {
        vec![b, column]
    }
}

const COPY: Element = Element::Read { stride: 1, offset: 0, by_row: false };

#[test]
fn a_run_of_16_or_8_bytes_copies_as_one_vector_and_one_of_4_stays_serial() {
    let quads = vectorized(DType::F32, 4, COPY);
    let pairs = vectorized(DType::F32, 2, COPY);
    let halves = vectorized(DType::F16, 4, COPY);
    let words = vectorized(DType::F16, 2, COPY);

    assert!(quads.contains("ld.global.v4.b32") && quads.contains("st.global.v4.b32"));
    assert!(pairs.contains("ld.global.v2.b32") && pairs.contains("st.global.v2.b32"));
    assert!(halves.contains("ld.global.v2.b32") && halves.contains("st.global.v2.b32"));
    assert!(!words.contains("ld.global.v") && !words.contains("st.global.v"), "{words}");
}

#[test]
fn only_a_run_of_zero_bits_clears_as_one_vector() {
    let float = |value| Element::Fill(Scalar::float(DType::F32, value).unwrap());
    let int = |value| Element::Fill(Scalar::int(DType::I32, value).unwrap());
    let [zero, one, negative_zero] =
        [0.0, 1.0, -0.0].map(|value| vectorized(DType::F32, 4, float(value)));
    let [int_zero, int_one] = [0, 1].map(|value| vectorized(DType::I32, 4, int(value)));

    assert!(zero.contains("st.global.v4.b32") && int_zero.contains("st.global.v4.b32"));
    for kept in [one, negative_zero, int_one] {
        assert!(!kept.contains("st.global.v4"), "{kept}");
    }
}

#[test]
fn a_loop_storing_two_outputs_stays_serial() {
    let module = vectorized(DType::F32, 4, Element::Twice);

    assert!(!module.contains("ld.global.v") && !module.contains("st.global.v"), "{module}");
}

#[test]
fn a_run_read_at_a_stride_or_from_an_unaligned_start_stays_serial() {
    let strided = Element::Read { stride: 2, offset: 0, by_row: false };
    let shifted = Element::Read { stride: 1, offset: 1, by_row: false };
    let by_row = Element::Read { stride: 1, offset: 0, by_row: true };

    for element in [strided, shifted, by_row] {
        let module = vectorized(DType::F32, 4, element);

        assert!(!module.contains("ld.global.v"), "{module}");
    }
}
