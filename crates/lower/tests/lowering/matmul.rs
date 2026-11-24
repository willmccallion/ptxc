use crate::elementwise::lowered;
use ptxc_core::{Axis, DType, ZeroPoint};
use ptxc_graph::{F32Attr, GemmAttrs, MatMulIntegerAttrs, Op, Orientation, ZeroPoints};
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

const fn per_tensor(point: i32) -> ZeroPoints {
    ZeroPoints::PerTensor(ZeroPoint::new(point))
}

fn per_axis(axis: usize, points: &[i32]) -> ZeroPoints {
    let points = points.iter().copied().map(ZeroPoint::new).collect();
    ZeroPoints::PerAxis { axis: Axis::new(axis), points }
}

/// The definitions of `MatMulInteger` of `i8[2, 3]` by `u8[3, 2]` with
/// zero points `a` and `b`.
fn matmul_integer(a: ZeroPoints, b: ZeroPoints) -> String {
    let operands: [(DType, &[u64]); 2] = [(DType::I8, &[2, 3]), (DType::U8, &[3, 2])];
    lowered(Op::MatMulInteger(MatMulIntegerAttrs { a, b }), &operands)
}

#[test]
fn matmul_integer_sums_each_operand_only_where_the_others_zero_points_need_it() {
    let definitions = [
        matmul_integer(per_tensor(0), per_axis(1, &[0, 0])),
        matmul_integer(per_tensor(-3), per_tensor(0)),
        matmul_integer(per_axis(0, &[1, 2]), per_axis(1, &[7, 0])),
    ];

    golden::check("matmul/matmul_integer.txt", &definitions.join("\n")).unwrap();
}
