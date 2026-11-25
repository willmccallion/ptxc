//! Kernels built for crate-internal tests.

use crate::compute::{
    BinaryOp, ComputeBuilder, ComputeName, Scalar, TensorExpr, TensorName, VarName,
};
use crate::expr::{ExprArena, VarId};
use crate::schedule::{Primitive, Schedule, SplitFactor, ThreadAxis};
use crate::stmt::{
    Access, Buffer, BufferId, Condition, ForKind, FragmentShape, FragmentTile, KernelParam,
    KernelVar, LaunchShape, LoweredKernel, MemScope, Stmt, TileAccess, TileOrder, ValueExpr,
};
use ptxc_core::{DType, Dim, ElementCount, Idx, IndexVec, KernelName, Shape};
use ptxc_target::{
    BlockCount, MmaInstruction, MmaShape, SmVersion, SmemBytes, ThreadsPerBlock, preset,
};
use std::sync::Arc;

/// `y[i] = x[i] + 1` over 8 elements, the sum split by 4 with the inner
/// loop bound to `threadIdx.x`: buffers 0 (`x`) and 1 (`y`).
pub(crate) fn kernel() -> LoweredKernel {
    kernel_with(None)
}

/// As `kernel`, with the outer loop bound to `blockIdx.x`, so two blocks
/// of four threads run.
pub(crate) fn two_block_kernel() -> LoweredKernel {
    kernel_with(Some(ThreadAxis::BlockX))
}

fn kernel_with(outer: Option<ThreadAxis>) -> LoweredKernel {
    let mut compute = ComputeBuilder::new(ComputeName::new("inc"));
    let x = compute.input(TensorName::new("x"), DType::F32, Shape::new(vec![Dim::new(8)]).unwrap());
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(8));
    let one = TensorExpr::Const(Scalar::float(DType::F32, 1.0).unwrap());
    let load = TensorExpr::load(x, vec![i]);
    compute.output(
        TensorName::new("y"),
        TensorExpr::Binary(BinaryOp::Add, Box::new(load), Box::new(one)),
    );
    let mut schedule = Schedule::new(Arc::new(compute.finish().unwrap()));
    let nest = schedule.loops().nest().to_vec();
    let parts = schedule.split(nest[0], vec![SplitFactor::new(4).unwrap()]).unwrap();
    schedule.apply(Primitive::Bind { loop_: parts[1], axis: ThreadAxis::ThreadX }).unwrap();
    if let Some(axis) = outer {
        schedule.apply(Primitive::Bind { loop_: parts[0], axis }).unwrap();
    }
    schedule.validate(&preset(SmVersion::SM_89).unwrap()).unwrap().lower()
}

/// One warp computing `c = a · b` with fragments for `instruction`: it
/// copies `a` (`m × k`, row-major) and `b` (`k × n`, stored as `b_order`
/// says) into shared tiles, loads both into fragments, fills an accumulator
/// with zero, multiplies, and stores `c` (`m × n`, row-major). Buffers 0
/// (`a`), 1 (`b`) and 2 (`c`) are its parameters.
pub(crate) fn mma_kernel(instruction: MmaInstruction, b_order: TileOrder) -> LoweredKernel {
    let row_major = TileOrder::RowMajor;
    mma_kernel_storing(instruction, b_order, |kernel, accumulator, c| {
        let origin = kernel.exprs.constant(0);
        let at = TileAccess { origin, stride: instruction.shape.n, order: row_major };
        store_each(kernel, accumulator, (c, at))
    })
}

/// Stores each element of `fragment`'s tile to `target`, where `at`
/// places the tile, through a `FragmentEach`.
pub(crate) fn store_each(
    kernel: &mut LoweredKernel,
    fragment: FragmentTile,
    (target, at): (BufferId, TileAccess),
) -> Stmt {
    let row = kernel.vars.push(KernelVar::Loop(VarName::new("row")));
    let column = kernel.vars.push(KernelVar::Loop(VarName::new("column")));
    let exprs = &mut kernel.exprs;
    let (row_expr, column_expr) = (exprs.var(row), exprs.var(column));
    let (outer, inner) = match at.order {
        TileOrder::RowMajor => (row_expr, column_expr),
        TileOrder::Transposed => (column_expr, row_expr),
    };
    let stride = exprs.constant(i64::try_from(at.stride.get()).unwrap());
    let scaled = exprs.mul(outer, stride);
    let within = exprs.add(scaled, inner);
    let index = exprs.add(at.origin, within);
    let store = Stmt::Store { buffer: target, index, value: ValueExpr::Element(fragment.buffer) };
    Stmt::FragmentEach { fragment, row, column, body: Box::new(store) }
}

/// As `mma_kernel`, with `store` making the statement that stores the
/// accumulator's tile to `c`.
pub(crate) fn mma_kernel_storing(
    instruction: MmaInstruction,
    b_order: TileOrder,
    store: impl FnOnce(&mut LoweredKernel, FragmentTile, BufferId) -> Stmt,
) -> LoweredKernel {
    let MmaShape { m, n, k } = instruction.shape;
    let (multiplicands, accumulator) = (instruction.multiplicands, instruction.accumulator);
    let mut kernel = warp_kernel();
    let [rows, columns, depth] = [m, n, k].map(Dim::get);
    let fragment = |rows, columns| MemScope::Fragment(FragmentShape { rows, columns });
    let mut buffer = |name: &str, dtype, scope, size: u64| {
        let name = TensorName::new(name);
        kernel.buffers.push(Buffer { name, dtype, scope, size: ElementCount::new(size) })
    };
    let a = buffer("a", multiplicands, MemScope::Global, rows * depth);
    let b = buffer("b", multiplicands, MemScope::Global, depth * columns);
    let c = buffer("c", accumulator, MemScope::Global, rows * columns);
    let a_tile = buffer("a_tile", multiplicands, MemScope::Shared { swizzle: None }, rows * depth);
    let b_tile =
        buffer("b_tile", multiplicands, MemScope::Shared { swizzle: None }, depth * columns);
    let a_fragment = buffer("a_fragment", multiplicands, fragment(m, k), rows * depth);
    let b_fragment = buffer("b_fragment", multiplicands, fragment(k, n), depth * columns);
    let c_fragment = buffer("c_fragment", accumulator, fragment(m, n), rows * columns);
    kernel.params = vec![
        KernelParam { buffer: a, access: Access::Read },
        KernelParam { buffer: b, access: Access::Read },
        KernelParam { buffer: c, access: Access::Write },
    ];
    let zero = kernel.exprs.constant(0);
    let at = |stride: Dim, order| TileAccess { origin: zero, stride, order };
    let [a_tile_0, b_tile_0, c_tile_0] =
        [a_fragment, b_fragment, c_fragment].map(|buffer| FragmentTile { buffer, tile: zero });
    let b_stride = if b_order == TileOrder::Transposed { k } else { n };
    let fragments = Stmt::Seq(vec![
        Stmt::FragmentLoad { fragment: a_tile_0, source: a_tile, at: at(k, TileOrder::RowMajor) },
        Stmt::FragmentLoad { fragment: b_tile_0, source: b_tile, at: at(b_stride, b_order) },
        Stmt::FragmentFill { fragment: c_tile_0, value: Scalar::zero(accumulator) },
        Stmt::Mma { instruction, a: a_tile_0, b: b_tile_0, accumulator: c_tile_0 },
        store(&mut kernel, c_tile_0, c),
    ]);
    let staged = Stmt::Seq(vec![
        warp_copy(&mut kernel, a, a_tile, rows * depth),
        warp_copy(&mut kernel, b, b_tile, depth * columns),
        Stmt::Barrier,
        allocated(&[a_fragment, b_fragment, c_fragment], fragments),
    ]);
    kernel.body = allocated(&[a_tile, b_tile], staged);
    kernel
}

/// A kernel of one block of one warp, its lane bound to `threadIdx.x`
/// (variable 0), with no buffers and an empty body.
pub(crate) fn warp_kernel() -> LoweredKernel {
    let mut vars = IndexVec::new();
    let _lane: VarId =
        vars.push(KernelVar::Thread { name: VarName::new("lane"), axis: ThreadAxis::ThreadX });
    LoweredKernel {
        name: KernelName::new("warp"),
        params: Vec::new(),
        buffers: IndexVec::new(),
        vars,
        launch: LaunchShape {
            grid: [BlockCount::new(1); 3],
            block: [ThreadsPerBlock::new(32), ThreadsPerBlock::new(1), ThreadsPerBlock::new(1)],
        },
        smem: SmemBytes::new(0),
        body: Stmt::Seq(Vec::new()),
        exprs: ExprArena::new(),
    }
}

/// The warp copying `size` elements of `from` to `to`, element `e` by lane
/// `e mod 32`.
fn warp_copy(kernel: &mut LoweredKernel, from: BufferId, to: BufferId, size: u64) -> Stmt {
    let round = kernel.vars.push(KernelVar::Loop(VarName::new("round")));
    let lane = VarId::new(0);
    let exprs = &mut kernel.exprs;
    let (round_expr, lane_expr, warp) = (exprs.var(round), exprs.var(lane), exprs.constant(32));
    let scaled = exprs.mul(round_expr, warp);
    let element = exprs.add(scaled, lane_expr);
    let bound = exprs.constant(i64::try_from(size).unwrap());
    let copy = Stmt::Store {
        buffer: to,
        index: element,
        value: ValueExpr::Load { buffer: from, index: element },
    };
    Stmt::For {
        var: round,
        extent: Dim::new(size.div_ceil(32)),
        kind: ForKind::Serial,
        body: Box::new(Stmt::If {
            condition: Condition { index: element, bound },
            then: Box::new(copy),
        }),
    }
}

/// `body` with each of `buffers` allocated around it, the first outermost.
fn allocated(buffers: &[BufferId], body: Stmt) -> Stmt {
    buffers.iter().rev().fold(body, |body, &buffer| Stmt::Alloc { buffer, body: Box::new(body) })
}
