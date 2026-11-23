//! Kernels covering every operation, type and statement codegen selects.
//! Each is assembled by ptxas. The selection tables record, per operation
//! and type, the instructions a rank-0 kernel runs between its parameter
//! loads and its `ret`, so a review reads the selection case by case.

use crate::ptxas::assemble;
use crate::{emitted, target};
use ptxc_codegen::generate;
use ptxc_core::float::Overflow;
use ptxc_core::{DType, Dim, Shape};
use ptxc_loop::{
    ArgKind, BinaryOp, ComputeBuilder, ComputeDef, ComputeName, InfSign, IterVarId, ReduceOp,
    Scalar, TensorExpr, TensorName, TieBreak, UnaryOp, VarName,
};
use ptxc_testkit::golden;
use std::fmt::Write;
use std::sync::Arc;

fn shape(dims: &[u64]) -> Shape {
    Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap()
}

/// `y = expr(x0, x1, ...)` over tensors of `shape`, one input per dtype,
/// or `None` if loop IR's type rules reject `expr` on these types.
fn elementwise(
    inputs: &[DType],
    shape: &Shape,
    expr: impl FnOnce(&[TensorExpr]) -> TensorExpr,
) -> Option<Arc<ComputeDef>> {
    let mut compute = ComputeBuilder::new(ComputeName::new("elementwise"));
    let tensors: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(n, &dtype)| compute.input(TensorName::new(format!("x{n}")), dtype, shape.clone()))
        .collect();
    let index: Vec<_> = shape
        .dims()
        .iter()
        .enumerate()
        .map(|(axis, &extent)| compute.spatial(VarName::new(format!("i{axis}")), extent).1)
        .collect();
    let loads: Vec<TensorExpr> = tensors
        .into_iter()
        .map(|tensor| TensorExpr::Load { tensor, index: index.clone() })
        .collect();
    compute.output(TensorName::new("y"), expr(&loads));
    compute.finish().ok().map(Arc::new)
}

/// A rank-0 elementwise kernel.
fn scalar(
    inputs: &[DType],
    expr: impl FnOnce(&[TensorExpr]) -> TensorExpr,
) -> Option<Arc<ComputeDef>> {
    elementwise(inputs, &Shape::scalar(), expr)
}

fn unary(op: UnaryOp, x: &TensorExpr) -> TensorExpr {
    TensorExpr::Unary(op, Box::new(x.clone()))
}

fn binary(op: BinaryOp, x: &TensorExpr, y: &TensorExpr) -> TensorExpr {
    TensorExpr::Binary(op, Box::new(x.clone()), Box::new(y.clone()))
}

/// A selection table's entry for `compute`, under `label`: its assembled
/// instructions after the parameters' addresses, or what codegen does not
/// emit yet.
fn entry(label: &str, compute: &Arc<ComputeDef>) -> String {
    let target = target();
    let kernel = ptxc_schedule::default_schedule(Arc::clone(compute), &target).unwrap().lower();
    let module = match generate(&kernel, &target) {
        Ok(module) => module,
        Err(error) => return format!("{label}: {error}\n"),
    };
    assemble(&module);
    let text = module.to_string();
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.iter().rposition(|line| line.contains("cvta.to.global")).unwrap() + 1;
    let end = lines.iter().position(|line| line.trim() == "ret;").unwrap();
    let mut entry = format!("{label}\n");
    for line in &lines[start..end] {
        writeln!(entry, "{line}").unwrap();
    }
    entry
}

#[test]
fn an_elementwise_add_emits_one_thread_per_element() {
    let add = |x: &[TensorExpr]| binary(BinaryOp::Add, &x[0], &x[1]);
    let compute = elementwise(&[DType::F32, DType::F32], &shape(&[37]), add).unwrap();

    let module = emitted(compute);

    assemble(&module);
    golden::check("add.ptx", &module.to_string()).unwrap();
}

/// Every cast, saturating and not where the target is FP8, and a cast to
/// bool feeding a boolean operation.
#[test]
fn casts_select_per_type_pair() {
    let mut table = String::new();
    for from in DType::ALL {
        for to in DType::ALL {
            let modes: &[Overflow] = if matches!(to, DType::F8E4M3Fn | DType::F8E5M2) {
                &[Overflow::Saturate, Overflow::NonFinite]
            } else {
                &[Overflow::NonFinite]
            };
            for &overflow in modes {
                let cast = |x: &[TensorExpr]| TensorExpr::Cast {
                    to,
                    overflow,
                    value: Box::new(x[0].clone()),
                };
                let label = format!("{} -> {} ({overflow:?})", from.name(), to.name());
                table += &entry(&label, &scalar(&[from], cast).unwrap());
            }
        }
    }

    let not_of_cast = |x: &[TensorExpr]| {
        let cast = TensorExpr::Cast {
            to: DType::Bool,
            overflow: Overflow::NonFinite,
            value: Box::new(x[0].clone()),
        };
        unary(UnaryOp::Not, &cast)
    };
    table += &entry("not (bool -> bool)", &scalar(&[DType::Bool], not_of_cast).unwrap());
    golden::check("casts.txt", &table).unwrap();
}

#[test]
fn unary_operations_select_per_type() {
    let ops = [
        UnaryOp::Abs,
        UnaryOp::Neg,
        UnaryOp::Exp,
        UnaryOp::Expm1,
        UnaryOp::Log,
        UnaryOp::Log1p,
        UnaryOp::Sqrt,
        UnaryOp::Reciprocal,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Round,
        UnaryOp::Sign,
        UnaryOp::Erf,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tanh,
        UnaryOp::Not,
        UnaryOp::IsNaN,
        UnaryOp::IsInf(InfSign::Either),
        UnaryOp::IsInf(InfSign::Positive),
        UnaryOp::IsInf(InfSign::Negative),
    ];
    let mut table = String::new();
    for op in ops {
        for dtype in DType::ALL {
            if let Some(compute) = scalar(&[dtype], |x| unary(op, &x[0])) {
                table += &entry(&format!("{op:?} {}", dtype.name()), &compute);
            }
        }
    }

    golden::check("unary.txt", &table).unwrap();
}

#[test]
fn binary_operations_select_per_type() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::TruncatedRem,
        BinaryOp::FlooredRem,
        BinaryOp::Equal,
        BinaryOp::Less,
        BinaryOp::LessOrEqual,
        BinaryOp::Greater,
        BinaryOp::GreaterOrEqual,
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Xor,
    ];
    let mut table = String::new();
    for op in ops {
        for dtype in DType::ALL {
            if let Some(compute) = scalar(&[dtype, dtype], |x| binary(op, &x[0], &x[1])) {
                table += &entry(&format!("{} {}", op.name(), dtype.name()), &compute);
            }
        }
    }

    golden::check("binary.txt", &table).unwrap();
}

#[test]
fn selects_and_constants_select_per_type() {
    let mut table = String::new();
    for dtype in DType::ALL {
        let select = |x: &[TensorExpr]| TensorExpr::Select {
            condition: Box::new(x[0].clone()),
            then: Box::new(x[1].clone()),
            otherwise: Box::new(x[2].clone()),
        };
        let compute = scalar(&[DType::Bool, dtype, dtype], select).unwrap();
        table += &entry(&format!("select {}", dtype.name()), &compute);
    }
    for dtype in DType::ALL {
        let constant = |_: &[TensorExpr]| TensorExpr::Const(Scalar::one(dtype));
        table += &entry(&format!("one {}", dtype.name()), &scalar(&[], constant).unwrap());
    }

    golden::check("select_constant.txt", &table).unwrap();
}

/// `y[i, j] = x[floordiv(i * 5 + j - 3, 4), floormod(i * 5 + j - 3, 4)]`,
/// clamped, which exercises floored division of negative indices.
#[test]
fn floored_index_arithmetic_assembles() {
    let mut compute = ComputeBuilder::new(ComputeName::new("floored"));
    let x = compute.input(TensorName::new("x"), DType::F16, shape(&[8, 4]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(5));
    let (_, j) = compute.spatial(VarName::new("j"), Dim::new(5));
    let exprs = compute.exprs();
    let five = exprs.constant(5);
    let row = exprs.mul(i, five);
    let flat = exprs.add(row, j);
    let minus_three = exprs.constant(-3);
    let shifted = exprs.add(flat, minus_three);
    let four = exprs.constant(4);
    let quotient = exprs.floor_div(shifted, four);
    let zero = exprs.constant(0);
    let seven = exprs.constant(7);
    let raised = exprs.max(quotient, zero);
    let clamped = exprs.min(raised, seven);
    let column = exprs.floor_mod(shifted, four);
    let load = TensorExpr::Load { tensor: x, index: vec![clamped, column] };
    compute.output(TensorName::new("y"), load);

    let module = emitted(Arc::new(compute.finish().unwrap()));

    assemble(&module);
    golden::check("floored.ptx", &module.to_string()).unwrap();
}

/// `y[i] = x[idx[i]]`: a gather reads its index, then the data.
#[test]
fn a_gather_loads_its_index_then_the_element() {
    let mut compute = ComputeBuilder::new(ComputeName::new("gather"));
    let x = compute.input(TensorName::new("x"), DType::Bf16, shape(&[9]));
    let positions = compute.input(TensorName::new("idx"), DType::I32, shape(&[6]));
    let (_, i) = compute.spatial(VarName::new("i"), Dim::new(6));
    let loaded = compute.loaded_index(VarName::new("j"), positions, vec![i]);
    compute.output(TensorName::new("y"), TensorExpr::Load { tensor: x, index: vec![loaded] });

    let module = emitted(Arc::new(compute.finish().unwrap()));

    assemble(&module);
    golden::check("gather.ptx", &module.to_string()).unwrap();
}

/// A row reduction of `dtype` values, a block per row whose threads
/// combine through shared memory.
fn row_reduction(
    dtype: DType,
    reduce: impl FnOnce(IterVarId, TensorExpr) -> TensorExpr,
) -> Arc<ComputeDef> {
    let mut compute = ComputeBuilder::new(ComputeName::new("rows"));
    let x = compute.input(TensorName::new("x"), dtype, shape(&[3, 300]));
    let (_, row) = compute.spatial(VarName::new("row"), Dim::new(3));
    let (axis, k) = compute.reduce_axis(VarName::new("k"), Dim::new(300));
    let load = TensorExpr::Load { tensor: x, index: vec![row, k] };
    compute.output(TensorName::new("y"), reduce(axis, load));
    Arc::new(compute.finish().unwrap())
}

#[test]
fn a_row_sum_combines_threads_in_a_shared_memory_tree() {
    let sum = |axis, body| TensorExpr::Reduce {
        op: ReduceOp::Sum,
        axes: vec![axis],
        body: Box::new(body),
    };

    let module = emitted(row_reduction(DType::F32, sum));

    assemble(&module);
    golden::check("row_sum.ptx", &module.to_string()).unwrap();
}

#[test]
fn a_row_argmax_assembles() {
    let argmax = |axis, body| TensorExpr::ArgReduce {
        kind: ArgKind::Max,
        tie: TieBreak::First,
        axis,
        body: Box::new(body),
    };

    assemble(&emitted(row_reduction(DType::F16, argmax)));
}
