use crate::elementwise::lowered;
use ptxc_core::{Axis, DType, Rank};
use ptxc_graph::Op;
use ptxc_testkit::golden;

#[test]
fn gathers_read_through_loaded_indices_wrapped_into_the_axis() {
    let definitions = [
        lowered(Op::Gather(Axis::new(1)), &[(DType::F32, &[2, 5, 3]), (DType::I64, &[4])]),
        lowered(Op::GatherElements(Axis::new(0)), &[(DType::I8, &[3, 2]), (DType::I32, &[4, 2])]),
        lowered(Op::GatherNd(Rank::new(1)), &[(DType::F16, &[2, 3, 4]), (DType::I64, &[2, 5, 2])]),
        lowered(Op::Gather(Axis::new(0)), &[(DType::F32, &[0, 2]), (DType::I64, &[0])]),
    ];

    golden::check("gather/gathers.txt", &definitions.join("\n")).unwrap();
}
