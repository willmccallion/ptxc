//! Schedules as text: the loop nest, each axis's index, the guards and the
//! trace.

use super::index::write_index;
use crate::compute::ComputeDef;
use crate::schedule::{LoopAnnotation, LoopId, LoopKind, LoopTree, Primitive, Schedule};
use ptxc_core::Idx;
use std::fmt::Write;

/// The text of `schedule`:
///
/// ```text
/// schedule add
///   for %i.0 in 4 bind(blockIdx.x)
///     for %i.1 in 256 bind(threadIdx.x)
///   %i = %i.0 * 256 + %i.1
///   guard %i.0 * 256 + %i.1 < 1000
/// trace
///   split %i [256]
/// ```
pub fn print_schedule<S>(schedule: &Schedule<S>) -> String {
    let loops = schedule.loops();
    let compute = schedule.compute();
    let name = |var: crate::expr::VarId| loops.loops()[LoopId::new(var.index())].name.to_string();
    let mut out = format!("schedule {}\n", compute.name());
    for (depth, &id) in loops.nest().iter().enumerate() {
        write_loop(&mut out, loops, id, depth);
    }
    for (axis, iter) in compute.iter_vars().iter_enumerated() {
        let _infallible = write!(out, "  %{} = ", iter.name);
        write_index(&mut out, loops.exprs(), loops.axis_index(axis), &name);
        out.push('\n');
    }
    for guard in loops.guards() {
        out.push_str("  guard ");
        write_index(&mut out, loops.exprs(), guard.index, &name);
        let _infallible = writeln!(out, " < {}", guard.extent.get());
    }
    out.push_str("trace\n");
    for step in schedule.trace().steps() {
        out.push_str("  ");
        write_step(&mut out, (compute, loops), step);
        out.push('\n');
    }
    out
}

fn write_loop(out: &mut String, loops: &LoopTree, id: LoopId, depth: usize) {
    let found = &loops.loops()[id];
    let _infallible = write!(
        out,
        "{:indent$}for %{} in {}",
        "",
        found.name,
        found.extent.get(),
        indent = 2 * depth + 2
    );
    if found.kind == LoopKind::Reduce {
        out.push_str(" reduce");
    }
    match found.annotation {
        LoopAnnotation::Serial => {}
        LoopAnnotation::Unrolled => out.push_str(" unroll"),
        LoopAnnotation::Vectorized => out.push_str(" vectorize"),
        LoopAnnotation::Bound(axis) => {
            let _infallible = write!(out, " bind({})", axis.name());
        }
    }
    out.push('\n');
}

fn write_step(out: &mut String, (compute, loops): (&ComputeDef, &LoopTree), step: &Primitive) {
    let name = |id: LoopId| format!("%{}", loops.loops()[id].name);
    let list = |ids: &[LoopId]| ids.iter().map(|&id| name(id)).collect::<Vec<_>>().join(", ");
    let _infallible = match step {
        Primitive::Split { loop_, factors } => {
            let factors: Vec<String> =
                factors.iter().map(|factor| factor.get().to_string()).collect();
            write!(out, "split {} [{}]", name(*loop_), factors.join(", "))
        }
        Primitive::Reorder { order } => write!(out, "reorder {}", list(order)),
        Primitive::Fuse { loops } => write!(out, "fuse {}", list(loops)),
        Primitive::Bind { loop_, axis } => write!(out, "bind {} {}", name(*loop_), axis.name()),
        Primitive::Unroll { loop_ } => write!(out, "unroll {}", name(*loop_)),
        Primitive::Vectorize { loop_ } => write!(out, "vectorize {}", name(*loop_)),
        Primitive::CacheRead { tensor, at } => {
            write!(out, "cache_read {} {}", compute.inputs()[*tensor].name, name(*at))
        }
    };
}
