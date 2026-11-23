use crate::elementwise::lowered;
use crate::{add_one, stages, tensor_type};
use ptxc_core::{Axis, DType};
use ptxc_graph::{
    BatchNormAttrs, F32Attr, GraphBuilder, InstanceNormAttrs, LayerNormAttrs, LayerNormKind,
    NormOutputs, Op, SoftmaxAttrs, SoftmaxKind, StashType,
};
use ptxc_loop::print_compute;
use ptxc_testkit::golden;

const EPSILON: F32Attr = F32Attr::new(1e-5);

#[test]
fn softmax_is_a_maximum_a_sum_and_an_elementwise_stage() {
    let attrs = SoftmaxAttrs { kind: SoftmaxKind::LogSoftmax, axis: Axis::new(1) };

    let definitions = lowered(Op::Softmax(attrs), &[(DType::F16, &[2, 3])]);

    golden::check("norm/log_softmax.txt", &definitions).unwrap();
}

#[test]
fn layer_norm_computes_its_statistics_before_normalising_and_writes_them_last() {
    let attrs = LayerNormAttrs {
        kind: LayerNormKind::Layer(NormOutputs::WithStatistics),
        axis: Axis::new(1),
        epsilon: EPSILON,
        stash: StashType::Bf16,
    };
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::F16, &[2, 3]));
    let scale = builder.add_input(tensor_type(DType::F16, &[3]));
    let bias = builder.add_input(tensor_type(DType::F16, &[1]));
    let outputs = builder.add_node(Op::LayerNorm(attrs), vec![x, scale, bias]).unwrap();
    let graph = builder.finish(outputs).unwrap();

    let definitions: String =
        stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect();

    golden::check("norm/layer_norm.txt", &definitions).unwrap();
}

#[test]
fn rms_norm_skips_the_mean_and_takes_the_scale_type() {
    let attrs = LayerNormAttrs {
        kind: LayerNormKind::Rms,
        axis: Axis::new(0),
        epsilon: EPSILON,
        stash: StashType::F32,
    };
    let mut builder = GraphBuilder::new();
    let x = builder.add_input(tensor_type(DType::Bf16, &[4]));
    let scale = builder.add_input(tensor_type(DType::F32, &[4]));
    let out = add_one(&mut builder, Op::LayerNorm(attrs), &[x, scale]);
    let graph = builder.finish(vec![out]).unwrap();

    let definitions: String =
        stages(graph).unwrap().iter().map(|stage| print_compute(stage.compute())).collect();

    golden::check("norm/rms_norm.txt", &definitions).unwrap();
}

#[test]
fn batch_and_instance_norm_read_parameters_per_channel() {
    let batch = lowered(
        Op::BatchNorm(BatchNormAttrs { epsilon: EPSILON }),
        &[
            (DType::F16, &[2, 3, 2]),
            (DType::F32, &[3]),
            (DType::F32, &[3]),
            (DType::F16, &[3]),
            (DType::F16, &[3]),
        ],
    );
    let instance = lowered(
        Op::InstanceNorm(InstanceNormAttrs { epsilon: EPSILON }),
        &[(DType::F32, &[1, 2, 4]), (DType::F32, &[2]), (DType::F32, &[2])],
    );

    golden::check("norm/batch_instance.txt", &[batch, instance].join("\n")).unwrap();
}
