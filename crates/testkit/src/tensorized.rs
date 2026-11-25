//! Products tensorized onto `mma`, and inputs whose every sum of products
//! is exact in any order, so a tensorized kernel matches the serial one bit
//! for bit (D58). The loop, codegen and GPU tests share them.
//!
//! NOTE: these are test code shared as a library. A builder handed a case
//! it cannot build has a bug, so it panics rather than returning an error
//! (D52).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    reason = "test builders panic on cases they build wrong (D52)"
)]

use ptxc_core::float::{Bf16, F8E4M3Fn, F8E5M2, F16, Overflow};
use ptxc_core::{DType, Dim, HostTensor, Idx, Shape, TensorData};
use ptxc_loop::{
    BinaryOp, ComputeBuilder, ComputeDef, ComputeName, ExprId, LoopId, Open, Primitive, ReduceOp,
    Schedule, SplitFactor, TensorExpr, TensorName, TensorParamId, ThreadAxis, TileOrder, VarName,
};
use ptxc_target::MmaInstruction;
use std::fmt;
use std::sync::Arc;

/// The loads a product multiplies, from its inputs `a` and `b` and its
/// `i`, `j` and `k`, in the order it multiplies them.
pub type Loads = dyn FnOnce(
    &mut ComputeBuilder,
    [TensorParamId; 2],
    [ExprId; 3],
) -> [(TensorParamId, Vec<ExprId>); 2];

/// The inputs of a product, by their shapes, and the loads it multiplies.
pub struct Operands {
    /// The shapes of `a` and `b`.
    pub shapes: [Vec<u64>; 2],
    /// The loads.
    pub loads: Box<Loads>,
}

impl fmt::Debug for Operands {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Operands").field("shapes", &self.shapes).finish_non_exhaustive()
    }
}

impl Operands {
    /// `a[i, k]` and `b[k, j]` over `extents`, `b` holding its `k × n`
    /// matrix in `b_order`.
    pub fn matmul([rows, columns, depth]: [u64; 3], b_order: TileOrder) -> Self {
        let b_shape = match b_order {
            TileOrder::RowMajor => vec![depth, columns],
            TileOrder::Transposed => vec![columns, depth],
        };
        let loads =
            move |_: &mut ComputeBuilder, [a, b]: [TensorParamId; 2], [i, j, k]: [ExprId; 3]| {
                let b_index = match b_order {
                    TileOrder::RowMajor => vec![k, j],
                    TileOrder::Transposed => vec![j, k],
                };
                [(a, vec![i, k]), (b, b_index)]
            };
        Self { shapes: [vec![rows, depth], b_shape], loads: Box::new(loads) }
    }
}

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// `out[i, j] = sum over k of x * y` over `extents`, `x` and `y` the loads
/// of `operands`, of `instruction`'s multiplicand type, each widened to its
/// accumulator's.
pub fn product(
    instruction: MmaInstruction,
    [rows, columns, depth]: [u64; 3],
    operands: Operands,
) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("product"));
    let [a_shape, b_shape] = &operands.shapes;
    let multiplicands = instruction.multiplicands;
    let inputs = [("a", a_shape), ("b", b_shape)]
        .map(|(name, dims)| compute.input(TensorName::new(name), multiplicands, shape(dims)));
    let (_, row) = compute.spatial(VarName::new("i"), Dim::new(rows));
    let (_, column) = compute.spatial(VarName::new("j"), Dim::new(columns));
    let (k_axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(depth));
    let [x, y] = (operands.loads)(&mut compute, inputs, [row, column, k]).map(|(tensor, index)| {
        Box::new(TensorExpr::Cast {
            to: instruction.accumulator,
            overflow: Overflow::Saturate,
            value: Box::new(TensorExpr::load(tensor, index)),
        })
    });
    let body = Box::new(TensorExpr::Binary(BinaryOp::Mul, x, y));
    let sum = TensorExpr::Reduce { op: ReduceOp::Sum, axes: vec![k_axis], body };
    compute.output(TensorName::new("out"), sum);
    Arc::new(compute.finish().unwrap())
}

/// Whether a tensorized nest's block loops are bound to the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blocks {
    /// On `blockIdx.y` (rows) and `blockIdx.x` (columns).
    Bound,
    /// Run one after another in a single block.
    Serial,
}

/// How a warp's array of tiles is looped over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileLoops {
    /// As counted loops.
    Rolled,
    /// Unrolled, so each tile's fragments are at constant places.
    Unrolled,
}

/// How a tensorized product's work is laid out: tiles per warp along rows
/// and columns, warps per block along rows, the block loops and the loops
/// over each warp's tiles.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Tiles per warp along rows and columns.
    pub tiles: [u64; 2],
    /// Warps per block, along rows.
    pub warps: u64,
    /// The block loops.
    pub blocks: Blocks,
    /// The loops over each warp's tiles.
    pub tile_loops: TileLoops,
}

/// One warp of one tile per block.
pub const ONE_TILE: Layout =
    Layout { tiles: [1, 1], warps: 1, blocks: Blocks::Bound, tile_loops: TileLoops::Rolled };

fn split(schedule: &mut Schedule<Open>, loop_: LoopId, factors: &[u64]) -> Vec<LoopId> {
    let factors = factors.iter().map(|&factor| SplitFactor::new(factor).unwrap()).collect();
    schedule.split(loop_, factors).unwrap()
}

/// `compute`, a product, tensorized with `instruction`.
///
/// Rows split into blocks, warps on `threadIdx.y`, tiles and the
/// instruction's `m`; columns into blocks, tiles and its `n`; the
/// reduction into steps of its `k` staging both operands; the tile's loops
/// innermost.
pub fn tensorized(
    compute: &Arc<ComputeDef>,
    layout: Layout,
    instruction: MmaInstruction,
) -> Schedule<Open> {
    let mut schedule = Schedule::new(Arc::clone(compute));
    let nest = schedule.loops().nest().to_vec();
    let shape = instruction.shape;
    let rows = split(&mut schedule, nest[0], &[layout.warps, layout.tiles[0], shape.m.get()]);
    let columns = split(&mut schedule, nest[1], &[layout.tiles[1], shape.n.get()]);
    let depth = split(&mut schedule, nest[2], &[shape.k.get()]);
    let order = vec![
        rows[0], columns[0], rows[1], depth[0], rows[2], columns[1], rows[3], columns[2], depth[1],
    ];
    schedule.apply(Primitive::Reorder { order }).unwrap();
    let mut binds = vec![(rows[1], ThreadAxis::ThreadY)];
    if layout.blocks == Blocks::Bound {
        binds.extend([(rows[0], ThreadAxis::BlockY), (columns[0], ThreadAxis::BlockX)]);
    }
    for (loop_, axis) in binds {
        schedule.apply(Primitive::Bind { loop_, axis }).unwrap();
    }
    if layout.tile_loops == TileLoops::Unrolled {
        for loop_ in [rows[2], columns[1]] {
            schedule.apply(Primitive::Unroll { loop_ }).unwrap();
        }
    }
    for tensor in [0, 1].map(TensorParamId::new) {
        schedule.apply(Primitive::CacheRead { tensor, at: depth[0] }).unwrap();
    }
    let tile = Primitive::Tensorize { m: rows[3], n: columns[2], k: depth[1], instruction };
    schedule.apply(tile).unwrap();
    schedule
}

/// The inputs of `compute`: integers from -4 to 4, or 0 to 8 if unsigned,
/// salted per input, so every sum of products is exact in any order, and
/// no float is subnormal.
pub fn exact_inputs(compute: &ComputeDef) -> Vec<HostTensor> {
    compute
        .inputs()
        .iter()
        .zip(0_u64..)
        .map(|(input, salt)| {
            let count = input.shape.element_count().get();
            let values = (0..count).map(|n| i64::try_from((3 * n + 7 * salt) % 9).unwrap());
            let data = small_integers(input.dtype, values);
            HostTensor::new(input.shape.clone(), data).unwrap()
        })
        .collect()
}

/// `values`, each from 0 to 8, as `dtype`: shifted to -4 to 4 unless
/// `dtype` is unsigned.
fn small_integers(dtype: DType, values: impl Iterator<Item = i64>) -> TensorData {
    let centred = values.map(|value| value - 4);
    match dtype {
        DType::F16 => centred.map(F16::from_i64).collect::<Vec<_>>().into(),
        DType::Bf16 => centred.map(Bf16::from_i64).collect::<Vec<_>>().into(),
        DType::F8E4M3Fn => centred
            .map(|value| F8E4M3Fn::from_i64(value, Overflow::Saturate))
            .collect::<Vec<_>>()
            .into(),
        DType::F8E5M2 => centred
            .map(|value| F8E5M2::from_i64(value, Overflow::Saturate))
            .collect::<Vec<_>>()
            .into(),
        DType::I8 => centred.map(|value| i8::try_from(value).unwrap()).collect::<Vec<_>>().into(),
        DType::U8 => {
            centred.map(|value| u8::try_from(value + 4).unwrap()).collect::<Vec<_>>().into()
        }
        other => panic!("{other:?} is no mma multiplicand"),
    }
}
