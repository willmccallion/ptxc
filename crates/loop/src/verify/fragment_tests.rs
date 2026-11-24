//! The rules on fragment statements, each broken by hand.

#![allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]

use super::{VerifyError, verify};
use crate::compute::{Scalar, TensorName, VarName};
use crate::expr::ExprId;
use crate::stmt::{
    Access, Buffer, BufferId, Condition, FragmentShape, FragmentTile, KernelParam, KernelVar,
    LoweredKernel, MemScope, Stmt, TileAccess, TileOrder, ValueExpr,
};
use crate::test_kernels::{store_each, warp_kernel};
use ptxc_core::{DType, Dim, ElementCount, Idx};
use ptxc_target::{MmaShape, SmVersion, ThreadsPerBlock, preset};

/// The buffers of a one-warp f16 product: a global input and output, a
/// shared tile, and fragments `a` (16 × 16), `b` (16 × 8) and `c` (16 × 8,
/// f32).
struct Product {
    kernel: LoweredKernel,
    input: BufferId,
    output: BufferId,
    tile: BufferId,
    a: FragmentTile,
    b: FragmentTile,
    c: FragmentTile,
}

fn product() -> Product {
    let mut kernel = warp_kernel();
    let fragment = |rows, columns| {
        MemScope::Fragment(FragmentShape { rows: Dim::new(rows), columns: Dim::new(columns) })
    };
    let mut buffer = |name: &str, dtype, scope, size| {
        let (name, size) = (TensorName::new(name), ElementCount::new(size));
        kernel.buffers.push(Buffer { name, dtype, scope, size })
    };
    let input = buffer("input", DType::F16, MemScope::Global, 256);
    let output = buffer("output", DType::F32, MemScope::Global, 128);
    let tile = buffer("tile", DType::F16, MemScope::Shared { swizzle: None }, 256);
    let a = buffer("a", DType::F16, fragment(16, 16), 256);
    let b = buffer("b", DType::F16, fragment(16, 8), 128);
    let c = buffer("c", DType::F32, fragment(16, 8), 128);
    kernel.params = vec![
        KernelParam { buffer: input, access: Access::Read },
        KernelParam { buffer: output, access: Access::Write },
    ];
    let zero = kernel.exprs.constant(0);
    let [a, b, c] = [a, b, c].map(|buffer| FragmentTile { buffer, tile: zero });
    Product { kernel, input, output, tile, a, b, c }
}

impl Product {
    /// The kernel with `body` inside the tile's and fragments'
    /// allocations.
    fn with(mut self, body: Stmt) -> LoweredKernel {
        self.kernel.body = [self.c.buffer, self.b.buffer, self.a.buffer, self.tile]
            .into_iter()
            .fold(body, |body, buffer| Stmt::Alloc { buffer, body: Box::new(body) });
        self.kernel
    }

    fn at(&mut self, origin: i64, stride: u64) -> TileAccess {
        let origin = self.kernel.exprs.constant(origin);
        TileAccess { origin, stride: Dim::new(stride), order: TileOrder::RowMajor }
    }
}

/// The verdict on `product`'s kernel with the body `build` makes.
fn verified(build: impl FnOnce(&mut Product) -> Stmt) -> Result<(), VerifyError> {
    let mut product = product();
    let body = build(&mut product);
    verify(&product.with(body))
}

/// `c += a · b` at m16n8k16 on f16, accumulating in f32.
fn mma([a, b, accumulator]: [FragmentTile; 3]) -> Stmt {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    let target = preset(SmVersion::SM_89).unwrap();
    let instruction = target.mma_instruction(shape, DType::F16, DType::F32).unwrap();
    Stmt::Mma { instruction, a, b, accumulator }
}

#[test]
fn a_whole_product_verifies() {
    let whole = verified(|product| {
        let (whole, half) = (product.at(0, 16), product.at(0, 8));
        Stmt::Seq(vec![
            Stmt::FragmentLoad { fragment: product.a, source: product.tile, at: whole },
            Stmt::FragmentLoad { fragment: product.b, source: product.tile, at: half },
            Stmt::FragmentFill { fragment: product.c, value: Scalar::zero(DType::F32) },
            mma([product.a, product.b, product.c]),
            store_each(&mut product.kernel, product.c, (product.output, half)),
        ])
    });

    assert_eq!(whole, Ok(()));
}

#[test]
fn an_mma_takes_fragments_of_its_shapes_and_types() {
    let shape = |rows, columns| FragmentShape { rows: Dim::new(rows), columns: Dim::new(columns) };

    let swapped = verified(|product| mma([product.b, product.a, product.c]));
    let shared = verified(|product| {
        let tile = FragmentTile { buffer: product.tile, tile: product.a.tile };
        mma([tile, product.b, product.c])
    });
    let narrow_accumulator = verified(|product| mma([product.a, product.b, product.b]));

    let expected = shape(16, 16);
    let actual = shape(16, 8);
    assert_eq!(
        swapped,
        Err(VerifyError::FragmentShape { fragment: TensorName::new("b"), expected, actual })
    );
    assert_eq!(shared, Err(VerifyError::NotFragment(TensorName::new("tile"))));
    assert!(matches!(
        narrow_accumulator,
        Err(VerifyError::WrongType { expected: DType::F32, actual: DType::F16, .. })
    ));
}

#[test]
fn fragments_move_only_between_shared_tiles_and_memory_in_bounds() {
    let from_global = verified(|product| {
        let at = product.at(0, 16);
        Stmt::FragmentLoad { fragment: product.a, source: product.input, at }
    });
    let past_the_end = verified(|product| {
        let at = product.at(1, 16);
        Stmt::FragmentLoad { fragment: product.a, source: product.tile, at }
    });
    let read_directly = verified(|product| {
        let index = product.kernel.exprs.constant(0);
        let value = ValueExpr::Load { buffer: product.a.buffer, index };
        Stmt::Store { buffer: product.output, index, value }
    });
    let into_input = verified(|product| {
        let at = product.at(0, 16);
        store_each(&mut product.kernel, product.a, (product.input, at))
    });

    assert_eq!(from_global, Err(VerifyError::TileNotShared(TensorName::new("input"))));
    assert_eq!(past_the_end, Err(VerifyError::OutOfBounds(TensorName::new("tile"))));
    assert_eq!(read_directly, Err(VerifyError::FragmentAccess(TensorName::new("a"))));
    assert_eq!(into_input, Err(VerifyError::StoreToInput(TensorName::new("input"))));
}

#[test]
fn every_lane_of_a_warp_reaches_a_fragment_statement() {
    let half_the_lanes = verified(|product| {
        let fill = Stmt::FragmentFill { fragment: product.c, value: Scalar::zero(DType::F32) };
        let lane = product.kernel.exprs.var(crate::expr::VarId::new(0));
        let half = product.kernel.exprs.constant(16);
        Stmt::If { condition: Condition { index: lane, bound: half }, then: Box::new(fill) }
    });
    let mut partial = product();
    partial.kernel.launch.block[0] = ThreadsPerBlock::new(48);
    let fill = Stmt::FragmentFill { fragment: partial.c, value: Scalar::zero(DType::F32) };

    assert_eq!(half_the_lanes, Err(VerifyError::DivergentWarp));
    assert_eq!(verify(&partial.with(fill)), Err(VerifyError::PartialWarp));
}

#[test]
fn a_fragment_is_used_inside_its_allocation_with_room_for_its_tile() {
    let mut outside = product();
    outside.kernel.body =
        Stmt::FragmentFill { fragment: outside.c, value: Scalar::zero(DType::F32) };
    let mut small = product();
    small.kernel.buffers[small.c.buffer].size = ElementCount::new(64);
    let fill = Stmt::FragmentFill { fragment: small.c, value: Scalar::zero(DType::F32) };

    assert_eq!(verify(&outside.kernel), Err(VerifyError::NotAllocated(TensorName::new("c"))));
    assert_eq!(verify(&small.with(fill)), Err(VerifyError::FragmentSize(TensorName::new("c"))));
}

/// A loop over the elements of `product`'s accumulator running `body`,
/// which `make` builds given the element's flat position in the output.
fn each_element(product: &mut Product, make: impl FnOnce(&mut Product, ExprId) -> Stmt) -> Stmt {
    let row = product.kernel.vars.push(KernelVar::Loop(VarName::new("row")));
    let column = product.kernel.vars.push(KernelVar::Loop(VarName::new("column")));
    let exprs = &mut product.kernel.exprs;
    let (row_expr, column_expr, columns) = (exprs.var(row), exprs.var(column), exprs.constant(8));
    let scaled = exprs.mul(row_expr, columns);
    let position = exprs.add(scaled, column_expr);
    let body = make(product, position);
    Stmt::FragmentEach { fragment: product.c, row, column, body: Box::new(body) }
}

#[test]
fn an_element_is_read_only_inside_a_loop_over_its_fragment() {
    let store_element = |product: &mut Product, index| Stmt::Store {
        buffer: product.output,
        index,
        value: ValueExpr::Element(product.c.buffer),
    };

    let inside = verified(|product| each_element(product, store_element));
    let outside = verified(|product| {
        let index = product.kernel.exprs.constant(0);
        store_element(product, index)
    });

    assert_eq!(inside, Ok(()));
    assert_eq!(outside, Err(VerifyError::ElementOutsideEach(TensorName::new("c"))));
}

#[test]
fn a_loop_over_elements_runs_each_in_one_lane() {
    let with_barrier = verified(|product| each_element(product, |_, _| Stmt::Barrier));
    let with_fill = verified(|product| {
        each_element(product, |product, _| Stmt::FragmentFill {
            fragment: product.a,
            value: Scalar::zero(DType::F16),
        })
    });

    assert_eq!(with_barrier, Err(VerifyError::DivergentBarrier));
    assert_eq!(with_fill, Err(VerifyError::DivergentWarp));
}

#[test]
fn a_tile_index_stays_within_its_fragment_buffer() {
    let past_the_last = verified(|product| {
        let tile = product.kernel.exprs.constant(1);
        let fragment = FragmentTile { tile, ..product.c };
        Stmt::FragmentFill { fragment, value: Scalar::zero(DType::F32) }
    });

    assert_eq!(past_the_last, Err(VerifyError::OutOfBounds(TensorName::new("c"))));
}
