use crate::elementwise::lowered;
use crate::{stages, tensor_type};
use ptxc_core::{Axis, DType, Dim, Shape};
use ptxc_graph::{
    AxisPadding, DiagonalOffset, GraphBuilder, Op, PadAttrs, PadMode, Padding, RepeatCount,
    SliceAttrs, SliceAxis, SliceStep, SplitAttrs, TileAttrs, TransposeAttrs, TriangularPart,
    TriluAttrs,
};
use ptxc_loop::print_compute;
use ptxc_testkit::golden;

fn tile(repeats: &[u64]) -> TileAttrs {
    TileAttrs { repeats: repeats.iter().copied().map(RepeatCount::new).collect() }
}

const fn padding(begin: u64, end: u64) -> AxisPadding {
    AxisPadding { begin: Padding::new(begin), end: Padding::new(end) }
}

#[test]
fn views_and_permutations_read_through_index_arithmetic() {
    let target = Shape::new(vec![Dim::new(3), Dim::new(2)]).unwrap();
    let permutation = [2, 0, 1].map(Axis::new).to_vec();
    let definitions = [
        lowered(Op::Reshape(target), &[(DType::F32, &[2, 3])]),
        lowered(Op::Transpose(TransposeAttrs { permutation }), &[(DType::I8, &[2, 3, 4])]),
        lowered(
            Op::Expand(Shape::new(vec![Dim::new(2), Dim::new(1)]).unwrap()),
            &[(DType::Bool, &[3])],
        ),
        lowered(Op::Tile(tile(&[2, 3])), &[(DType::F16, &[0, 2])]),
    ];

    golden::check("shape/views.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn concat_and_slice_select_and_step_through_their_inputs() {
    let backwards = SliceAxis {
        axis: Axis::new(0),
        start: 4,
        step: SliceStep::new(-2).unwrap(),
        length: Dim::new(3),
    };
    let definitions = [
        lowered(
            Op::Concat(Axis::new(1)),
            &[(DType::F16, &[2, 1]), (DType::F16, &[2, 0]), (DType::F16, &[2, 3])],
        ),
        lowered(Op::Slice(SliceAttrs { axes: vec![backwards] }), &[(DType::F32, &[5])]),
    ];

    golden::check("shape/concat_slice.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn split_is_a_stage_per_piece() {
    let attrs = SplitAttrs { axis: Axis::new(0), sizes: vec![Dim::new(1), Dim::new(3)] };
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[4]));
    let pieces = builder.add_node(Op::Split(attrs), vec![x]).unwrap();
    let graph = builder.finish(pieces).unwrap();

    let definitions: String =
        stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect();

    golden::check("shape/split.txt", &definitions).unwrap();
}

#[test]
fn padding_clamps_folds_or_reflects_its_reads() {
    let constant = PadAttrs { mode: PadMode::Constant, axes: vec![padding(1, 0), padding(0, 0)] };
    let reflect = PadAttrs {
        mode: PadMode::Reflect,
        axes: vec![padding(2, 1), padding(1, 0), padding(0, 0)],
    };
    let edge = PadAttrs { mode: PadMode::Edge, axes: vec![padding(0, 2)] };
    let wrap = PadAttrs { mode: PadMode::Wrap, axes: vec![padding(1, 2)] };
    let lower = TriluAttrs { part: TriangularPart::Lower, diagonal: DiagonalOffset::new(-1) };
    let definitions = [
        lowered(Op::Pad(constant), &[(DType::F32, &[2, 2]), (DType::F32, &[])]),
        lowered(Op::Pad(reflect), &[(DType::F32, &[4, 1, 2])]),
        lowered(Op::Pad(edge), &[(DType::F32, &[3])]),
        lowered(Op::Pad(wrap), &[(DType::F32, &[3])]),
        lowered(Op::Trilu(lower), &[(DType::I32, &[3, 3])]),
    ];

    golden::check("shape/pad_trilu.txt", &definitions.join("\n")).unwrap();
}
