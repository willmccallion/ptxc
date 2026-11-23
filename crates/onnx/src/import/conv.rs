//! Imports `Conv`, and reads the window attributes `Conv` and pooling share.

use crate::context::NodeContext;
use crate::error::{ImportError, ImportErrorKind};
use crate::lowered::Lowered;
use ptxc_core::Dim;
use ptxc_graph::{ConvAttrs, Dilation, Groups, Op, Padding, Stride, WindowAxis};

pub(super) fn lower(
    op: &str,
    context: &mut NodeContext<'_>,
) -> Option<Result<Lowered, ImportError>> {
    (op == "Conv").then(|| conv(context))
}

fn conv(context: &mut NodeContext<'_>) -> Result<Lowered, ImportError> {
    let input = context.ty(0)?.shape().dims().to_vec();
    let weights = context.ty(1)?.shape().dims().to_vec();
    let kernel = weights.get(2..).unwrap_or_default().to_vec();
    check_kernel_shape(context, &kernel)?;
    let axes = read_window(context, input.get(2..).unwrap_or_default(), &kernel)?;
    let groups = context.attrs.int_or("group", 1)?;
    let groups = u64::try_from(groups)
        .ok()
        .and_then(Groups::new)
        .ok_or_else(|| invalid("Conv group must be positive"))?;
    let operands = if context.has_input(2) { 3 } else { 2 };
    Ok(Lowered::on_inputs(Op::Conv(ConvAttrs { axes, groups }), operands))
}

/// Checks an optional `kernel_shape` against the weights' spatial dims,
/// which are the kernel ptxc uses.
pub(super) fn check_kernel_shape(
    context: &mut NodeContext<'_>,
    kernel: &[Dim],
) -> Result<(), ImportError> {
    match context.attrs.ints("kernel_shape")? {
        Some(declared)
            if declared
                .iter()
                .copied()
                .map(u64::try_from)
                .ne(kernel.iter().map(|dim| Ok(dim.get()))) =>
        {
            Err(invalid("kernel_shape differs from the weights' spatial dims"))
        }
        _ => Ok(()),
    }
}

fn invalid(problem: &str) -> ImportError {
    ImportErrorKind::Invalid { field: "window attribute", problem: problem.to_owned() }.into()
}

/// Per-axis windows from `strides`, `dilations`, `pads` and `auto_pad`,
/// resolving `auto_pad` to explicit pads with ONNX's own formula
/// (`convPoolShapeInference`), which static shapes allow (D3).
pub(super) fn read_window(
    context: &mut NodeContext<'_>,
    spatial: &[Dim],
    kernel: &[Dim],
) -> Result<Vec<WindowAxis>, ImportError> {
    let count = spatial.len();
    let strides = per_axis(context.attrs.ints("strides")?, count)?
        .into_iter()
        .map(|value| value.and_then(Stride::new).ok_or_else(|| invalid("a stride is not positive")))
        .collect::<Result<Vec<_>, _>>()?;
    let dilations = per_axis(context.attrs.ints("dilations")?, count)?
        .into_iter()
        .map(|value| {
            value.and_then(Dilation::new).ok_or_else(|| invalid("a dilation is not positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pads = context.attrs.ints("pads")?.map(<[i64]>::to_vec);
    let auto_pad = context.attrs.string_or("auto_pad", "NOTSET")?;
    if kernel.len() != count {
        return Err(invalid("the kernel's rank differs from the input's spatial rank"));
    }
    let explicit = match (auto_pad.as_str(), pads) {
        ("NOTSET", Some(pads)) if pads.len() == 2 * count => pads,
        ("NOTSET" | "VALID", None) => vec![0; 2 * count],
        ("SAME_UPPER" | "SAME_LOWER", None) => {
            same_pads(&auto_pad, spatial, kernel, &strides, &dilations)?
        }
        ("NOTSET", Some(_)) => return Err(invalid("pads must have two entries per spatial axis")),
        ("VALID" | "SAME_UPPER" | "SAME_LOWER", Some(_)) => {
            return Err(invalid("pads cannot be given with auto_pad"));
        }
        _ => return Err(context.attrs.unsupported("auto_pad")),
    };
    (0..count)
        .map(|axis| {
            let pad = |value: i64| {
                u64::try_from(value).map(Padding::new).map_err(|_| invalid("negative pads"))
            };
            Ok(WindowAxis {
                stride: strides[axis],
                dilation: dilations[axis],
                pad_begin: pad(explicit[axis])?,
                pad_end: pad(explicit[axis + count])?,
            })
        })
        .collect()
}

/// One value per spatial axis, ONNX's default of 1 when absent. `None`
/// marks a negative value, which no window newtype accepts.
fn per_axis(values: Option<&[i64]>, count: usize) -> Result<Vec<Option<u64>>, ImportError> {
    match values {
        None => Ok(vec![Some(1); count]),
        Some(values) if values.len() == count => {
            Ok(values.iter().map(|&value| u64::try_from(value).ok()).collect())
        }
        Some(_) => Err(invalid("a window attribute needs one entry per spatial axis")),
    }
}

/// ONNX's SAME padding: the total keeps `ceil(input / stride)` outputs, split
/// with the larger half at the end (`SAME_UPPER`) or the start.
fn same_pads(
    mode: &str,
    spatial: &[Dim],
    kernel: &[Dim],
    strides: &[Stride],
    dilations: &[Dilation],
) -> Result<Vec<i64>, ImportError> {
    let mut begins = Vec::with_capacity(spatial.len());
    let mut ends = Vec::with_capacity(spatial.len());
    for axis in 0..spatial.len() {
        let effective = (kernel[axis].get().saturating_sub(1)) * dilations[axis].get() + 1;
        let stride = strides[axis].get();
        let residual = spatial[axis].get() % stride;
        let total = if residual == 0 {
            effective.saturating_sub(stride)
        } else {
            effective.saturating_sub(residual)
        };
        let small = i64::try_from(total / 2).map_err(|_| invalid("pads too large"))?;
        let big = i64::try_from(total - total / 2).map_err(|_| invalid("pads too large"))?;
        let (begin, end) = if mode == "SAME_UPPER" { (small, big) } else { (big, small) };
        begins.push(begin);
        ends.push(end);
    }
    begins.extend(ends);
    Ok(begins)
}
