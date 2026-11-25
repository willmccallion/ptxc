use crate::{add_one, stages, tensor_type};
use ptxc_core::{DType, Idx};
use ptxc_graph::{BinaryOp, GraphBuilder, Op, Partitioned, UnaryOp, ValueId};
use ptxc_loop::{ComputeDef, RfactorError, Rfactored, Slices, rfactor};
use ptxc_lower::{Binding, IntermediateId, LoweredGroup, lower};
use std::sync::Arc;

#[test]
fn a_stage_binds_each_graph_value_once() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[4]));
    let doubled = add_one(&mut builder, Op::Binary(BinaryOp::Add), &[x, x]);
    let graph = builder.finish(vec![doubled]).unwrap();

    let stages = stages(graph).unwrap();

    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].inputs().iter().copied().collect::<Vec<_>>(), [Binding::Value(x)]);
    assert_eq!(stages[0].outputs(), [Binding::Value(doubled)]);
    assert_eq!(stages[0].compute().name().as_str(), "add_0");
}

#[test]
fn a_node_whose_output_no_one_reads_has_no_stage() {
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F32, &[4]));
    let _unread = add_one(&mut builder, Op::Unary(UnaryOp::Neg), &[x]);
    let out = add_one(&mut builder, Op::Unary(UnaryOp::Abs), &[x]);
    let graph = builder.finish(vec![out]).unwrap();

    let stages = stages(graph).unwrap();

    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].outputs(), [Binding::Value(out)]);
}

/// The one group of `a * b`, a matrix product of f32 `[4, 8]` and `[8, 6]`.
fn product_group() -> (LoweredGroup, [ValueId; 3]) {
    let mut builder = GraphBuilder::new();
    let a = builder.add_input(tensor_type(DType::F32, &[4, 8]));
    let b = builder.add_input(tensor_type(DType::F32, &[8, 6]));
    let product = add_one(&mut builder, Op::MatMul, &[a, b]);
    let graph = builder.finish(vec![product]).unwrap();
    let partitioned = Partitioned::one_node_per_group(graph).unwrap();
    let group = lower(&partitioned).unwrap().into_iter().next().unwrap();
    (group, [a, b, product])
}

/// `compute`'s sum split into two slices.
fn halves(compute: &Arc<ComputeDef>) -> Result<Option<Rfactored>, RfactorError> {
    let axis = compute.iter_vars().iter_enumerated().last().unwrap().0;
    rfactor(compute, axis, Slices::new(2).unwrap()).map(Some)
}

#[test]
fn a_split_stage_becomes_a_partial_and_a_combine_joined_by_its_partials() {
    let (group, [a, b, product]) = product_group();

    let split = group.with_split_reductions(halves).unwrap();

    let partials = Binding::Intermediate(IntermediateId::new(0));
    let [partial, combine] = split.stages() else { panic!("{:?}", split.stages()) };
    assert_eq!(
        partial.inputs().iter().copied().collect::<Vec<_>>(),
        [Binding::Value(a), Binding::Value(b)]
    );
    assert_eq!(partial.outputs(), [partials]);
    assert_eq!(combine.inputs().iter().copied().collect::<Vec<_>>(), [partials]);
    assert_eq!(combine.outputs(), [Binding::Value(product)]);
    assert_eq!(split.intermediates()[IntermediateId::new(0)], tensor_type(DType::F32, &[2, 4, 6]));
}

#[test]
fn a_stage_left_unsplit_keeps_its_bindings_and_a_refusal_is_returned() {
    let (group, _) = product_group();
    let original = group.stages()[0].clone();

    let kept = group.clone().with_split_reductions(|_| Ok::<_, RfactorError>(None)).unwrap();
    let refused = group.with_split_reductions(|_| Err(RfactorError::Gather));

    let [stage] = kept.stages() else { panic!("{:?}", kept.stages()) };
    assert_eq!((stage.inputs(), stage.outputs()), (original.inputs(), original.outputs()));
    assert_eq!(kept.intermediates().iter().count(), 0);
    assert_eq!(refused.unwrap_err(), RfactorError::Gather);
}
