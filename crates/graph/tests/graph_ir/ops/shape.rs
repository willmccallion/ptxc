use crate::common::{f32_type, shape, tensor_type};
use ptxc_core::{Axis, DType, Dim, Rank};
use ptxc_graph::{
    AxisPadding, DiagonalOffset, Op, PadAttrs, PadMode, Padding, RepeatCount, SliceAttrs,
    SliceAxis, SliceStep, SplitAttrs, TensorType, TileAttrs, TransposeAttrs, TriangularPart,
    TriluAttrs, TypeError, TypeRule,
};

fn axis(index: i64) -> Axis {
    Axis::from_signed(index, Rank::new(8)).unwrap()
}

fn infer_one(op: &Op, input: &TensorType) -> Result<Vec<TensorType>, TypeError> {
    op.infer(&[input])
}

fn slice(index: i64, start: u64, step: i64, length: u64) -> SliceAttrs {
    let step = SliceStep::new(step).unwrap();
    SliceAttrs {
        axes: vec![SliceAxis { axis: axis(index), start, step, length: Dim::new(length) }],
    }
}

const fn padding(begin: u64, end: u64) -> AxisPadding {
    AxisPadding { begin: Padding::new(begin), end: Padding::new(end) }
}

#[test]
fn reshape_keeps_the_elements_and_type() {
    let input = tensor_type(DType::I64, &[2, 3, 4]);

    assert_eq!(
        infer_one(&Op::Reshape(shape(&[6, 4])), &input),
        Ok(vec![tensor_type(DType::I64, &[6, 4])])
    );
    assert_eq!(infer_one(&Op::Reshape(shape(&[1])), &f32_type(&[])), Ok(vec![f32_type(&[1])]));
}

#[test]
fn reshape_to_a_different_element_count_is_rejected() {
    let error = infer_one(&Op::Reshape(shape(&[7])), &f32_type(&[2, 3]));

    assert_eq!(
        error,
        Err(TypeError::ElementCountMismatch { from: shape(&[2, 3]), to: shape(&[7]) })
    );
}

#[test]
fn expand_broadcasts_in_both_directions() {
    assert_eq!(
        infer_one(&Op::Expand(shape(&[2, 1, 4])), &f32_type(&[3, 1])),
        Ok(vec![f32_type(&[2, 3, 4])])
    );
    assert_eq!(infer_one(&Op::Expand(shape(&[1])), &f32_type(&[3])), Ok(vec![f32_type(&[3])]));
    assert!(infer_one(&Op::Expand(shape(&[4])), &f32_type(&[3])).is_err());
}

#[test]
fn transpose_permutes_the_axes() {
    let attrs = TransposeAttrs { permutation: vec![axis(2), axis(0), axis(1)] };

    assert_eq!(attrs.infer(&[&f32_type(&[2, 3, 4])]), Ok(vec![f32_type(&[4, 2, 3])]));
}

#[test]
fn transpose_needs_every_axis_exactly_once() {
    let input = f32_type(&[2, 3]);
    let short = TransposeAttrs { permutation: vec![axis(0)] };
    let repeated = TransposeAttrs { permutation: vec![axis(1), axis(1)] };
    let beyond = TransposeAttrs { permutation: vec![axis(0), axis(2)] };

    assert_eq!(
        short.infer(&[&input]),
        Err(TypeError::Rank { expected: Rank::new(1), actual: Rank::new(2) })
    );
    assert_eq!(repeated.infer(&[&input]), Err(TypeError::RepeatedAxis(axis(1))));
    assert_eq!(
        beyond.infer(&[&input]),
        Err(TypeError::AxisOutOfRange { axis: axis(2), rank: Rank::new(2) })
    );
}

#[test]
fn tile_multiplies_each_axis_by_its_count() {
    let tile = |counts: &[u64]| TileAttrs {
        repeats: counts.iter().copied().map(RepeatCount::new).collect(),
    };

    assert_eq!(tile(&[2, 1]).infer(&[&f32_type(&[2, 3])]), Ok(vec![f32_type(&[4, 3])]));
    assert_eq!(tile(&[0, 1]).infer(&[&f32_type(&[2, 3])]), Ok(vec![f32_type(&[0, 3])]));
    assert!(matches!(tile(&[2]).infer(&[&f32_type(&[2, 3])]), Err(TypeError::Rank { .. })));
    assert_eq!(RepeatCount::new(5).get(), 5);
}

#[test]
fn concat_joins_along_its_axis() {
    let joined =
        Op::Concat(axis(1)).infer(&[&f32_type(&[2, 3]), &f32_type(&[2, 5]), &f32_type(&[2, 0])]);

    assert_eq!(joined, Ok(vec![f32_type(&[2, 8])]));
    assert_eq!(Op::Concat(axis(0)).infer(&[&f32_type(&[4])]), Ok(vec![f32_type(&[4])]));
}

#[test]
fn concat_operands_must_agree_off_the_axis() {
    let error = Op::Concat(axis(1)).infer(&[&f32_type(&[2, 3]), &f32_type(&[4, 3])]);

    assert_eq!(
        error,
        Err(TypeError::DimMismatch { axis: axis(0), expected: Dim::new(2), actual: Dim::new(4) })
    );
    assert!(matches!(
        Op::Concat(axis(0)).infer(&[&f32_type(&[2, 3]), &f32_type(&[2])]),
        Err(TypeError::Rank { .. })
    ));
    assert!(matches!(
        Op::Concat(axis(0)).infer(&[&f32_type(&[2]), &tensor_type(DType::F16, &[2])]),
        Err(TypeError::MixedDTypes { .. })
    ));
}

#[test]
fn concat_needs_an_operand_with_its_axis() {
    assert_eq!(Op::Concat(axis(0)).infer(&[]), Err(TypeError::NoOperands));
    assert!(matches!(
        Op::Concat(axis(1)).infer(&[&f32_type(&[2])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}

#[test]
fn slice_takes_length_elements_of_its_axes() {
    assert_eq!(slice(0, 2, 3, 3).infer(&[&f32_type(&[10, 5])]), Ok(vec![f32_type(&[3, 5])]));
    assert_eq!(slice(0, 9, -2, 5).infer(&[&f32_type(&[10, 5])]), Ok(vec![f32_type(&[5, 5])]));
    assert_eq!(slice(1, 99, 1, 0).infer(&[&f32_type(&[10, 5])]), Ok(vec![f32_type(&[10, 0])]));
}

#[test]
fn slices_that_leave_their_axis_are_rejected() {
    let input = f32_type(&[10]);

    for attrs in [slice(0, 2, 3, 4), slice(0, 10, 1, 1), slice(0, 1, -1, 3)] {
        assert_eq!(attrs.infer(&[&input]), Err(TypeError::SliceOutOfBounds(axis(0))), "{attrs:?}");
    }
    assert!(slice(0, 0, 1, 10).infer(&[&input]).is_ok());
}

#[test]
fn slice_axes_must_be_distinct_and_in_range() {
    let mut twice = slice(0, 0, 1, 1);
    twice.axes.push(twice.axes[0]);

    assert_eq!(twice.infer(&[&f32_type(&[4])]), Err(TypeError::RepeatedAxis(axis(0))));
    assert!(matches!(
        slice(1, 0, 1, 1).infer(&[&f32_type(&[4])]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
    assert_eq!(SliceStep::new(0), None);
    assert_eq!(SliceStep::new(-3).map(SliceStep::get), Some(-3));
}

#[test]
fn the_unit_slice_step_is_one() {
    assert_eq!(SliceStep::one().get(), 1);
}

#[test]
fn split_gives_one_output_per_size() {
    let sizes = vec![Dim::new(1), Dim::new(0), Dim::new(5)];
    let attrs = SplitAttrs { axis: axis(0), sizes };

    let pieces = attrs.infer(&[&f32_type(&[6, 4])]);

    assert_eq!(pieces, Ok(vec![f32_type(&[1, 4]), f32_type(&[0, 4]), f32_type(&[5, 4])]));
}

#[test]
fn split_sizes_must_add_up_to_the_axis() {
    let short = SplitAttrs { axis: axis(0), sizes: vec![Dim::new(2), Dim::new(3)] };
    let none = SplitAttrs { axis: axis(0), sizes: Vec::new() };
    let input = f32_type(&[6]);

    assert_eq!(
        short.infer(&[&input]),
        Err(TypeError::SplitSizes { extent: Dim::new(6), sizes: short.sizes.clone() })
    );
    assert!(matches!(none.infer(&[&input]), Err(TypeError::SplitSizes { .. })));
    assert!(matches!(
        SplitAttrs { axis: axis(1), ..short }.infer(&[&input]),
        Err(TypeError::AxisOutOfRange { .. })
    ));
}

#[test]
fn constant_padding_grows_each_axis_by_both_ends() {
    let attrs = PadAttrs { mode: PadMode::Constant, axes: vec![padding(1, 0), padding(2, 2)] };
    let value = f32_type(&[]);

    assert_eq!(attrs.infer(&[&f32_type(&[2, 3]), &value]), Ok(vec![f32_type(&[3, 7])]));
    assert_eq!(attrs.infer(&[&f32_type(&[0, 3]), &value]), Ok(vec![f32_type(&[1, 7])]));
}

#[test]
fn constant_padding_needs_a_scalar_value_of_the_input_type() {
    let attrs = PadAttrs { mode: PadMode::Constant, axes: vec![padding(1, 1)] };
    let input = f32_type(&[2]);

    assert!(matches!(attrs.infer(&[&input]), Err(TypeError::Arity { .. })));
    assert_eq!(
        attrs.infer(&[&input, &f32_type(&[1])]),
        Err(TypeError::Rank { expected: Rank::new(0), actual: Rank::new(1) })
    );
    assert!(matches!(
        attrs.infer(&[&input, &tensor_type(DType::I32, &[])]),
        Err(TypeError::MixedDTypes { .. })
    ));
}

#[test]
fn copy_padding_takes_only_the_input_and_needs_elements_to_copy() {
    for mode in [PadMode::Reflect, PadMode::Edge, PadMode::Wrap] {
        let attrs = PadAttrs { mode, axes: vec![padding(1, 1), padding(0, 0)] };
        assert_eq!(attrs.infer(&[&f32_type(&[3, 0])]), Ok(vec![f32_type(&[5, 0])]), "{mode:?}");
        assert_eq!(
            attrs.infer(&[&f32_type(&[0, 3])]),
            Err(TypeError::PaddedEmptyAxis(axis(0))),
            "{mode:?}"
        );
        assert!(matches!(
            attrs.infer(&[&f32_type(&[3, 3]), &f32_type(&[])]),
            Err(TypeError::Arity { .. })
        ));
    }
}

#[test]
fn padding_needs_an_entry_per_axis() {
    let attrs = PadAttrs { mode: PadMode::Edge, axes: vec![padding(1, 1)] };

    assert_eq!(
        attrs.infer(&[&f32_type(&[2, 2])]),
        Err(TypeError::Rank { expected: Rank::new(1), actual: Rank::new(2) })
    );
}

#[test]
fn trilu_keeps_the_shape_of_matrices() {
    let attrs = TriluAttrs { part: TriangularPart::Lower, diagonal: DiagonalOffset::new(3) };

    assert_eq!(
        attrs.infer(&[&tensor_type(DType::I64, &[2, 3, 4])]),
        Ok(vec![tensor_type(DType::I64, &[2, 3, 4])])
    );
    assert_eq!(
        attrs.infer(&[&f32_type(&[4])]),
        Err(TypeError::RankBelow { minimum: Rank::new(2), actual: Rank::new(1) })
    );
    assert_eq!(attrs.infer(&[&f32_type(&[2, 2])]), Ok(vec![f32_type(&[2, 2])]));
    assert_eq!(attrs.diagonal.get(), 3);
}
