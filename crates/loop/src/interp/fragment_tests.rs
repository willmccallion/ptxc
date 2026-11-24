//! The fragment statements: every mma instruction of the target against
//! products computed by hand, and the warps they need whole and agreeing.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use super::tensors::{data, values};
use super::value::Value;
use super::{RunError, execute, run};
use crate::compute::{BinaryOp, VarName};
use crate::compute::{Scalar, TensorName};
use crate::expr::VarId;
use crate::stmt::{
    Access, Buffer, BufferId, FragmentShape, FragmentTile, KernelParam, KernelVar, LoweredKernel,
    MemScope, Stmt, TileAccess, TileOrder, ValueExpr,
};
use crate::test_kernels::{mma_kernel, mma_kernel_storing, store_each, warp_kernel};
use crate::verify::VerifyError;
use ptxc_core::convert::int_to_f64;
use ptxc_core::{DType, Dim, ElementCount, HostTensor, Idx, Shape};
use ptxc_target::{MmaInstruction, MmaShape, SmVersion, ThreadsPerBlock, preset};

/// The shapes and types of the `sm_89` mma table.
const INSTRUCTIONS: [([u64; 3], DType, DType); 6] = [
    ([16, 8, 16], DType::F16, DType::F32),
    ([16, 8, 16], DType::Bf16, DType::F32),
    ([16, 8, 32], DType::F8E4M3Fn, DType::F32),
    ([16, 8, 32], DType::F8E5M2, DType::F32),
    ([16, 8, 32], DType::I8, DType::I32),
    ([16, 8, 32], DType::U8, DType::I32),
];

fn instruction(
    ([m, n, k], multiplicands, accumulator): ([u64; 3], DType, DType),
) -> MmaInstruction {
    let shape = MmaShape { m: Dim::new(m), n: Dim::new(n), k: Dim::new(k) };
    preset(SmVersion::SM_89).unwrap().mma_instruction(shape, multiplicands, accumulator).unwrap()
}

/// A small integer at row `row`, column `column`: in `-4..=4`, or `0..=8`
/// for an unsigned type, so every multiplicand type holds it and every sum
/// of products is exact.
fn entry(dtype: DType, row: u64, column: u64, salt: u64) -> i64 {
    let value = i64::try_from((3 * row + 5 * column + salt) % 9).unwrap();
    if dtype == DType::U8 { value } else { value - 4 }
}

/// `entries` as values of `dtype`.
fn as_values(dtype: DType, entries: &[i64]) -> Vec<Value> {
    let value = |&entry: &i64| {
        if dtype.is_float() { Value::Float(int_to_f64(entry)) } else { Value::Int(entry) }
    };
    entries.iter().map(value).collect()
}

fn tensor(dtype: DType, dims: [u64; 2], entries: &[i64]) -> HostTensor {
    let shape = Shape::new(dims.map(Dim::new).to_vec()).unwrap();
    HostTensor::new(shape, data(&as_values(dtype, entries), dtype)).unwrap()
}

/// `rows × columns` entries of `dtype`, row by row, the one at (r, c)
/// from `at(r, c)`.
fn matrix(rows: u64, columns: u64, at: impl Fn(u64, u64) -> i64) -> Vec<i64> {
    (0..rows)
        .flat_map(|row| (0..columns).map(move |column| (row, column)))
        .map(|(row, column)| at(row, column))
        .collect()
}

#[test]
fn each_mma_instruction_multiplies_its_fragments_as_stored_or_transposed() {
    for entry_types in INSTRUCTIONS {
        let instruction = instruction(entry_types);
        let shape = instruction.shape;
        let [rows, columns, depth] = [shape.m, shape.n, shape.k].map(Dim::get);
        let dtype = instruction.multiplicands;
        let lhs_at = |row, step| entry(dtype, row, step, 0);
        let rhs_at = |step, column| entry(dtype, step, column, 4);
        let product = matrix(rows, columns, |row, column| {
            (0..depth).map(|step| lhs_at(row, step) * rhs_at(step, column)).sum()
        });
        let lhs = tensor(dtype, [rows, depth], &matrix(rows, depth, lhs_at));
        for order in [TileOrder::RowMajor, TileOrder::Transposed] {
            let rhs = match order {
                TileOrder::RowMajor => {
                    tensor(dtype, [depth, columns], &matrix(depth, columns, rhs_at))
                }
                TileOrder::Transposed => tensor(
                    dtype,
                    [columns, depth],
                    &matrix(columns, depth, |column, step| rhs_at(step, column)),
                ),
            };

            let outputs = run(&mma_kernel(instruction, order), &[lhs.clone(), rhs]).unwrap();

            let expected = as_values(instruction.accumulator, &product);
            assert_eq!(values(outputs[0].data()), expected, "{instruction:?} {order:?}");
        }
    }
}

/// A one-warp kernel whose body is `body`, given a shared `tile` of 256
/// f16s and an f16 `fragment` of 16 by 16, both allocated around it.
fn with_fragment(
    body: impl FnOnce(&mut LoweredKernel, BufferId, FragmentTile) -> Stmt,
) -> LoweredKernel {
    let mut kernel = warp_kernel();
    let mut buffer = |name: &str, scope| {
        let (name, size) = (TensorName::new(name), ElementCount::new(256));
        kernel.buffers.push(Buffer { name, dtype: DType::F16, scope, size })
    };
    let shape = FragmentShape { rows: Dim::new(16), columns: Dim::new(16) };
    let tile = buffer("tile", MemScope::Shared);
    let fragment = buffer("fragment", MemScope::Fragment(shape));
    let whole = FragmentTile { buffer: fragment, tile: kernel.exprs.constant(0) };
    let inner = body(&mut kernel, tile, whole);
    let in_fragment = Stmt::Alloc { buffer: fragment, body: Box::new(inner) };
    kernel.body = Stmt::Alloc { buffer: tile, body: Box::new(in_fragment) };
    kernel
}

#[test]
fn a_fragment_statement_in_a_warp_the_block_does_not_fill_is_refused() {
    let mut kernel = with_fragment(|_, _, fragment| Stmt::FragmentFill {
        fragment,
        value: Scalar::zero(DType::F16),
    });
    kernel.launch.block[0] = ThreadsPerBlock::new(48);

    let unverified = execute(&kernel, &[]).unwrap_err();

    assert_eq!(unverified, RunError::PartialWarp);
    assert_eq!(run(&kernel, &[]).unwrap_err(), RunError::Invalid(VerifyError::PartialWarp));
}

#[test]
fn lanes_that_disagree_on_where_a_tile_is_are_refused() {
    let kernel = with_fragment(|kernel, tile, fragment| {
        let lane = kernel.exprs.var(VarId::new(0));
        let at = TileAccess { origin: lane, stride: Dim::new(16), order: TileOrder::RowMajor };
        Stmt::FragmentLoad { fragment, source: tile, at }
    });

    let error = execute(&kernel, &[]).unwrap_err();

    assert_eq!(error, RunError::WarpDisagrees);
}

#[test]
fn a_fragment_kernel_prints_its_tiles_and_mma() {
    let kernel = mma_kernel(instruction(INSTRUCTIONS[0]), TileOrder::Transposed);

    let text = crate::print_kernel(&kernel);

    ptxc_testkit::golden::check("lower/mma_fragments.txt", &text).unwrap();
}

/// Two warps, each loading its own 16 × 16 tile of a shared copy of `x`
/// and storing it to its own part of `y`.
fn two_warps() -> LoweredKernel {
    let mut kernel = warp_kernel();
    kernel.launch.block[0] = ThreadsPerBlock::new(64);
    let mut buffer = |name: &str, scope| {
        let (name, size) = (TensorName::new(name), ElementCount::new(512));
        kernel.buffers.push(Buffer { name, dtype: DType::F16, scope, size })
    };
    let x = buffer("x", MemScope::Global);
    let y = buffer("y", MemScope::Global);
    let tile = buffer("tile", MemScope::Shared);
    let shape = FragmentShape { rows: Dim::new(16), columns: Dim::new(16) };
    let fragment = kernel.buffers.push(Buffer {
        name: TensorName::new("fragment"),
        dtype: DType::F16,
        scope: MemScope::Fragment(shape),
        size: ElementCount::new(256),
    });
    kernel.params = vec![
        KernelParam { buffer: x, access: Access::Read },
        KernelParam { buffer: y, access: Access::Write },
    ];
    let exprs = &mut kernel.exprs;
    let thread = exprs.var(VarId::new(0));
    let (warp_size, tile_size) = (exprs.constant(32), exprs.constant(256));
    let warp = exprs.floor_div(thread, warp_size);
    let origin = exprs.mul(warp, tile_size);
    let round = kernel.vars.push(crate::stmt::KernelVar::Loop(VarName::new("round")));
    let (round, block) = (kernel.exprs.var(round), kernel.exprs.constant(64));
    let scaled = kernel.exprs.mul(round, block);
    let element = kernel.exprs.add(scaled, thread);
    let at = TileAccess { origin, stride: Dim::new(16), order: TileOrder::RowMajor };
    let copy = Stmt::For {
        var: VarId::new(1),
        extent: Dim::new(8),
        kind: crate::stmt::ForKind::Serial,
        body: Box::new(Stmt::Store {
            buffer: tile,
            index: element,
            value: crate::stmt::ValueExpr::Load { buffer: x, index: element },
        }),
    };
    let whole = FragmentTile { buffer: fragment, tile: kernel.exprs.constant(0) };
    let per_warp = Stmt::Seq(vec![
        Stmt::FragmentLoad { fragment: whole, source: tile, at },
        store_each(&mut kernel, whole, (y, at)),
    ]);
    let body = Stmt::Seq(vec![
        copy,
        Stmt::Barrier,
        Stmt::Alloc { buffer: fragment, body: Box::new(per_warp) },
    ]);
    kernel.body = Stmt::Alloc { buffer: tile, body: Box::new(body) };
    kernel
}

#[test]
fn each_warp_holds_its_own_fragments() {
    let entries: Vec<i64> = (0..512).map(|entry| entry % 97).collect();
    let x = HostTensor::new(
        Shape::new(vec![Dim::new(512)]).unwrap(),
        data(&as_values(DType::F16, &entries), DType::F16),
    )
    .unwrap();

    let outputs = run(&two_warps(), &[x]).unwrap();

    assert_eq!(values(outputs[0].data()), as_values(DType::F16, &entries));
}

#[test]
fn an_epilogue_runs_once_for_each_element_of_the_accumulator() {
    let instruction = instruction(INSTRUCTIONS[0]);
    let kernel = mma_kernel_storing(instruction, TileOrder::RowMajor, |kernel, accumulator, c| {
        let row = kernel.vars.push(KernelVar::Loop(VarName::new("row")));
        let column = kernel.vars.push(KernelVar::Loop(VarName::new("column")));
        let exprs = &mut kernel.exprs;
        let (row_expr, column_expr, columns) =
            (exprs.var(row), exprs.var(column), exprs.constant(8));
        let scaled = exprs.mul(row_expr, columns);
        let index = exprs.add(scaled, column_expr);
        let one = ValueExpr::Const(Scalar::one(DType::F32));
        let value = ValueExpr::Binary(
            BinaryOp::Add,
            Box::new(ValueExpr::Element(accumulator.buffer)),
            Box::new(one),
        );
        let store = Stmt::Store { buffer: c, index, value };
        Stmt::FragmentEach { fragment: accumulator, row, column, body: Box::new(store) }
    });
    let lhs = matrix(16, 16, |row, step| entry(DType::F16, row, step, 0));
    let rhs = matrix(16, 8, |step, column| entry(DType::F16, step, column, 4));
    let inputs = [tensor(DType::F16, [16, 16], &lhs), tensor(DType::F16, [16, 8], &rhs)];
    let plus_one = matrix(16, 8, |row, column| {
        1 + (0..16)
            .map(|step| entry(DType::F16, row, step, 0) * entry(DType::F16, step, column, 4))
            .sum::<i64>()
    });

    let outputs = run(&kernel, &inputs).unwrap();

    assert_eq!(values(outputs[0].data()), as_values(DType::F32, &plus_one));
}

#[test]
fn a_fragment_buffer_holds_one_tile_per_index() {
    let mut kernel = warp_kernel();
    let shape = FragmentShape { rows: Dim::new(16), columns: Dim::new(8) };
    let size = ElementCount::new(2 * 128);
    let pair = kernel.buffers.push(Buffer {
        name: TensorName::new("pair"),
        dtype: DType::F32,
        scope: MemScope::Fragment(shape),
        size,
    });
    let out = kernel.buffers.push(Buffer {
        name: TensorName::new("out"),
        dtype: DType::F32,
        scope: MemScope::Global,
        size: ElementCount::new(128),
    });
    kernel.params = vec![KernelParam { buffer: out, access: Access::Write }];
    let (first, second) = (kernel.exprs.constant(0), kernel.exprs.constant(1));
    let tile = |index| FragmentTile { buffer: pair, tile: index };
    let at = TileAccess { origin: first, stride: Dim::new(8), order: TileOrder::RowMajor };
    let fill = |value| Scalar::float(DType::F32, value).unwrap();
    let store = store_each(&mut kernel, tile(second), (out, at));
    let body = Stmt::Seq(vec![
        Stmt::FragmentFill { fragment: tile(first), value: fill(1.0) },
        Stmt::FragmentFill { fragment: tile(second), value: fill(2.0) },
        store,
    ]);
    kernel.body = Stmt::Alloc { buffer: pair, body: Box::new(body) };

    let outputs = run(&kernel, &[]).unwrap();

    assert_eq!(values(outputs[0].data()), vec![Value::Float(2.0); 128]);
}
