use crate::elementwise::lowered;
use ptxc_core::DType;
use ptxc_graph::{F32Attr, GemmAttrs, Op, Orientation};
use ptxc_testkit::golden;

#[test]
fn matmul_broadcasts_batches_and_drops_promoted_axes() {
    let definitions = [
        lowered(Op::MatMul, &[(DType::F16, &[2, 1, 2, 3]), (DType::F16, &[4, 3, 5])]),
        lowered(Op::MatMul, &[(DType::I32, &[3]), (DType::I32, &[2, 3, 4])]),
        lowered(Op::MatMul, &[(DType::F32, &[2, 3]), (DType::F32, &[3])]),
    ];

    golden::check("matmul/matmul.txt", &definitions.join("\n")).unwrap();
}

#[test]
fn gemm_reads_its_operands_in_their_orientations_and_scales_around_the_sum() {
    let transposed = GemmAttrs {
        a: Orientation::Transposed,
        b: Orientation::AsIs,
        alpha: F32Attr::new(0.5),
        beta: F32Attr::new(1.0),
    };
    let unit = GemmAttrs {
        a: Orientation::AsIs,
        b: Orientation::Transposed,
        alpha: F32Attr::new(1.0),
        beta: F32Attr::new(1.0),
    };
    let scaled = GemmAttrs { beta: F32Attr::new(2.0), ..transposed };
    let bias_scaled = GemmAttrs { beta: F32Attr::new(2.0), ..unit };
    let definitions = [
        lowered(
            Op::Gemm(transposed),
            &[(DType::Bf16, &[3, 2]), (DType::Bf16, &[3, 4]), (DType::Bf16, &[4])],
        ),
        lowered(
            Op::Gemm(unit),
            &[(DType::I64, &[2, 3]), (DType::I64, &[4, 3]), (DType::I64, &[2, 1])],
        ),
        lowered(Op::Gemm(scaled), &[(DType::I32, &[3, 2]), (DType::I32, &[3, 4])]),
        lowered(
            Op::Gemm(bias_scaled),
            &[(DType::I32, &[2, 3]), (DType::I32, &[4, 3]), (DType::I32, &[4])],
        ),
    ];

    golden::check("matmul/gemm.txt", &definitions.join("\n")).unwrap();
}
