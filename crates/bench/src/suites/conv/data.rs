//! A problem's inputs, and the check of an output against them. Every
//! input and weight is a multiple of 1/8 in [-1, 1], which f16 holds
//! exactly; their products are multiples of 1/64, which f32 sums exactly
//! over the at most 4608 a layer's output sums, so an output differs from
//! the host's only by its own type's rounding. Each problem's values are
//! its own, salted by its name.

use super::ConvError;
use super::layers::{ConvProblem, ConvType, Layer, id};
use crate::host::{data, salt_of, values, whole_value};
use ptxc_core::{Dim, ElementCount, HostTensor, Shape, TensorData};
use ptxc_testkit::compare::{Tolerance, compare};

/// How many output elements a check recomputes on the host.
const SAMPLES: u64 = 256;

/// Each operand's salt, so the input and the weights differ.
const X_SALT: u64 = 1;
const W_SALT: u64 = 2;

/// The input and the weights, NCHW and OIHW, on the host.
#[derive(Clone, Debug)]
pub(super) struct Inputs {
    pub(super) x: HostTensor,
    pub(super) w: HostTensor,
}

/// One problem's input and weight values by position, salted by it.
#[derive(Clone, Copy, Debug)]
struct Values<'p> {
    layer: &'p Layer,
    x_salt: u64,
    w_salt: u64,
}

impl<'p> Values<'p> {
    fn new(problem: &'p ConvProblem, dtype: ConvType) -> Self {
        let hash = salt_of(&id(problem, dtype));
        Self { layer: &problem.layer, x_salt: hash ^ X_SALT, w_salt: hash ^ W_SALT }
    }

    /// The input's element at image `n`, channel `c`, row `h`, column `v`.
    fn x(self, [n, c, h, v]: [u64; 4]) -> f64 {
        let (channels, size) = (u64::from(self.layer.channels), u64::from(self.layer.size));
        whole_value(((n * channels + c) * size + h) * size + v, self.x_salt) / 8.0
    }

    /// The weights' element at output channel `m`, channel `c`, tap row
    /// `r`, tap column `s`.
    fn w(self, [m, c, r, s]: [u64; 4]) -> f64 {
        let (channels, kernel) = (u64::from(self.layer.channels), u64::from(self.layer.kernel));
        whole_value(((m * channels + c) * kernel + r) * kernel + s, self.w_salt) / 8.0
    }

    /// The output at image `n`, output channel `m`, row `p`, column `q`:
    /// the sum of every tap inside the input, padding being zero.
    fn output(self, [n, m, p, q]: [u64; 4]) -> f64 {
        let layer = self.layer;
        let (kernel, stride) = (u64::from(layer.kernel), u64::from(layer.stride));
        let (pad, size) = (u64::from(layer.pad), u64::from(layer.size));
        let inside = |out: u64, tap: u64| {
            (out * stride + tap).checked_sub(pad).filter(|&position| position < size)
        };
        let mut sum = 0.0;
        for c in 0..u64::from(layer.channels) {
            for r in 0..kernel {
                for s in 0..kernel {
                    if let (Some(h), Some(v)) = (inside(p, r), inside(q, s)) {
                        sum = self.x([n, c, h, v]).mul_add(self.w([m, c, r, s]), sum);
                    }
                }
            }
        }
        sum
    }
}

impl Inputs {
    /// `problem`'s input and weights, of `dtype`.
    pub(super) fn new(problem: &ConvProblem, dtype: ConvType) -> Result<Self, ConvError> {
        let values = Values::new(problem, dtype);
        let layer = &problem.layer;
        let x_dims = [problem.batch, layer.channels, layer.size, layer.size];
        let w_dims = [layer.features, layer.channels, layer.kernel, layer.kernel];
        Ok(Self {
            x: tensor(dtype, x_dims, |at| values.x(at))?,
            w: tensor(dtype, w_dims, |at| values.w(at))?,
        })
    }
}

/// A tensor of `dims` whose element at each index is `at(index)`.
fn tensor(
    dtype: ConvType,
    dims: [u32; 4],
    at: impl Fn([u64; 4]) -> f64,
) -> Result<HostTensor, ConvError> {
    let [a, b, c, d] = dims.map(u64::from);
    let indices = (0..a).flat_map(move |i| {
        (0..b).flat_map(move |j| (0..c).flat_map(move |k| (0..d).map(move |l| [i, j, k, l])))
    });
    let elements: Vec<f64> = indices.map(at).collect();
    let shape = Shape::new(dims.map(|dim| Dim::new(u64::from(dim))).to_vec())?;
    Ok(HostTensor::new(shape, data(dtype.dtype(), &elements)?)?)
}

/// Whether `output`, the convolution as a framework computed it, NCHW,
/// matches the host's at evenly spaced elements within its type's
/// tolerance for the layer's sum; if not, how it differs.
pub(super) fn check(
    problem: &ConvProblem,
    dtype: ConvType,
    output: &TensorData,
) -> Result<(), String> {
    let layer = &problem.layer;
    let (features, out) = (u64::from(layer.features), u64::from(layer.output()));
    let total = u64::from(problem.batch) * features * out * out;
    let count = SAMPLES.min(total);
    let all = values(output).map_err(|error| error.to_string())?;
    let values_of = Values::new(problem, dtype);
    let (mut actual, mut expected) = (Vec::new(), Vec::new());
    for sample in 0..count {
        let position = sample * total / count.max(1);
        let index = usize::try_from(position).map_err(|_| "the output is too large to index")?;
        actual.push(*all.get(index).ok_or("the output is smaller than the problem")?);
        let (n, rest) = (position / (features * out * out), position % (features * out * out));
        let (m, rest) = (rest / (out * out), rest % (out * out));
        expected.push(values_of.output([n, m, rest / out, rest % out]));
    }
    let as_tensor = |elements: &[f64]| -> Result<HostTensor, String> {
        Ok(HostTensor::vector(data(dtype.dtype(), elements).map_err(|error| error.to_string())?))
    };
    let depth = ElementCount::new(u64::from(layer.depth()).max(1));
    let tolerance = Tolerance::for_dtype(dtype.dtype()).for_reduction(depth);
    compare(&as_tensor(&actual)?, &as_tensor(&expected)?, tolerance)
        .map_err(|mismatch| mismatch.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{Inputs, Values, check};
    use crate::host::{data, values, whole_value};
    use crate::suites::conv::layers::{ConvProblem, ConvType, Layer};
    use ptxc_core::TensorData;
    use ptxc_core::float::F16;

    /// Two images of two 5 by 5 channels into three, a 3 by 3 kernel at
    /// stride 2 and padding 1: a 3 by 3 output whose edges read padding.
    fn problem() -> ConvProblem {
        let layer = Layer {
            name: "tiny".to_owned(),
            channels: 2,
            features: 3,
            size: 5,
            kernel: 3,
            stride: 2,
            pad: 1,
        };
        ConvProblem { layer, batch: 2 }
    }

    /// The convolution of the stored input and weights, NCHW, worked out
    /// from their flat positions alone.
    fn convolved(inputs: &Inputs) -> Vec<f64> {
        let (x, w) = (values(inputs.x.data()).unwrap(), values(inputs.w.data()).unwrap());
        let read = |out: usize, tap: usize| (2 * out + tap).checked_sub(1).filter(|&at| at < 5);
        positions([2, 3, 3, 3])
            .map(|[n, m, p, q]| {
                let taps = positions([1, 2, 3, 3]).map(|[_, c, r, s]| (c, r, s));
                taps.fold(0.0, |sum, (c, r, s)| match (read(p, r), read(q, s)) {
                    (Some(h), Some(v)) => x[((n * 2 + c) * 5 + h) * 5 + v]
                        .mul_add(w[((m * 2 + c) * 3 + r) * 3 + s], sum),
                    _ => sum,
                })
            })
            .collect()
    }

    /// Every index below `extents`, row-major.
    fn positions([a, b, c, d]: [usize; 4]) -> impl Iterator<Item = [usize; 4]> {
        (0..a).flat_map(move |i| {
            (0..b).flat_map(move |j| (0..c).flat_map(move |k| (0..d).map(move |l| [i, j, k, l])))
        })
    }

    fn as_f16(elements: &[f64]) -> Vec<F16> {
        let TensorData::F16(halves) = data(ConvType::F16.dtype(), elements).unwrap() else {
            unreachable!()
        };
        halves
    }

    #[test]
    fn each_input_holds_its_salted_values_at_their_flat_positions() {
        let problem = problem();
        let inputs = Inputs::new(&problem, ConvType::F16).unwrap();
        let at = Values::new(&problem, ConvType::F16);

        let x = values(inputs.x.data()).unwrap();
        let w = values(inputs.w.data()).unwrap();

        let salted = |salt, count: u64| -> Vec<f64> {
            (0..count).map(|index| whole_value(index, salt) / 8.0).collect()
        };
        assert_eq!(x, salted(at.x_salt, 2 * 2 * 5 * 5));
        assert_eq!(w, salted(at.w_salt, 3 * 2 * 3 * 3));
        assert!(x.iter().all(|v| (-1.0..=1.0).contains(v) && (v * 8.0).fract() == 0.0));
        assert_ne!(x[..18], w[..]);
    }

    #[test]
    fn the_host_output_is_the_convolution_of_the_stored_inputs() {
        let problem = problem();
        let inputs = Inputs::new(&problem, ConvType::F16).unwrap();
        let at = Values::new(&problem, ConvType::F16);

        let host: Vec<f64> = positions([2, 3, 3, 3])
            .map(|at_output| at.output(at_output.map(|i| u64::try_from(i).unwrap())))
            .collect();

        assert_eq!(host, convolved(&inputs));
    }

    #[test]
    fn the_true_convolution_passes_and_a_wrong_element_anywhere_fails() {
        let problem = problem();
        let truth = as_f16(&convolved(&Inputs::new(&problem, ConvType::F16).unwrap()));
        let wrong_at = |index: usize| {
            let mut wrong = truth.clone();
            wrong[index] = F16::from_f64(wrong[index].to_f64() + 1.0);
            check(&problem, ConvType::F16, &TensorData::F16(wrong))
        };

        assert_eq!(check(&problem, ConvType::F16, &TensorData::F16(truth.clone())), Ok(()));
        for index in [0, 31, truth.len() - 1] {
            assert!(wrong_at(index).is_err(), "{index}");
        }
        assert!(check(&problem, ConvType::F16, &TensorData::F16(truth[..53].to_vec())).is_err());
    }
}
