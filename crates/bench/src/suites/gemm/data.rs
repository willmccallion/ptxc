//! A problem's inputs, and the check of an output against them. Every input
//! is a multiple of 1/8 in [-1, 1] (an integer in [-8, 8] for int8), which
//! every type holds exactly; their products are multiples of 1/64, which
//! f32 sums exactly up to K = 2^17, so an output differs from the host's
//! only by its own type's rounding. Each problem's values are its own, so
//! no two problems share a product, and a C left in reused device memory by
//! one problem cannot pass another's check.

use super::SweepError;
use super::problems::id;
use crate::host::{data, salt_of, values, whole_value};
use crate::problem::{GemmProblem, GemmShape, GemmType};
use ptxc_core::{Dim, ElementCount, HostTensor, Shape, TensorData};
use ptxc_testkit::compare::{Tolerance, compare};

/// How many output elements a check recomputes on the host.
const SAMPLES: u64 = 256;

/// Each operand's salt, so A and B differ.
const A_SALT: u64 = 1;
const B_SALT: u64 = 2;

/// A and B as stored, on the host.
#[derive(Clone, Debug)]
pub(super) struct Inputs {
    pub(super) a: HostTensor,
    pub(super) b: HostTensor,
}

/// The value at row-major position `index` of the operand salted `salt`.
fn value(index: u64, salt: u64, dtype: GemmType) -> f64 {
    let whole = whole_value(index, salt);
    if dtype == GemmType::I8 { whole } else { whole / 8.0 }
}

/// One problem's operand values by row and column, salted by the problem.
#[derive(Clone, Copy, Debug)]
struct OperandValues {
    problem: GemmProblem,
    a_salt: u64,
    b_salt: u64,
}

impl OperandValues {
    fn new(problem: GemmProblem) -> Self {
        let hash = salt_of(&id(problem));
        Self { problem, a_salt: hash ^ A_SALT, b_salt: hash ^ B_SALT }
    }

    /// A's element at row `i`, column `p`.
    fn a(self, i: u64, p: u64) -> f64 {
        value(i * u64::from(self.problem.shape.k) + p, self.a_salt, self.problem.dtype)
    }

    /// B's element at row `p`, column `j`.
    fn b(self, p: u64, j: u64) -> f64 {
        value(p * u64::from(self.problem.shape.n) + j, self.b_salt, self.problem.dtype)
    }
}

impl Inputs {
    /// `problem`'s A and B, each stored as its layout says.
    pub(super) fn new(problem: GemmProblem) -> Result<Self, SweepError> {
        let GemmShape { m, n, k } = problem.shape;
        let (dtype, layout) = (problem.dtype, problem.layout);
        let operands = OperandValues::new(problem);
        let a = stored([m, k], layout.a_transposed(), |i, p| operands.a(i, p));
        let b = stored([k, n], layout.b_transposed(), |p, j| operands.b(p, j));
        Ok(Self { a: tensor(dtype, a)?, b: tensor(dtype, b)? })
    }
}

/// A matrix of `[rows, columns]` whose element at (r, c) is `at(r, c)`,
/// row-major, stored transposed if `transposed`: its dims and values.
fn stored(
    [rows, columns]: [u32; 2],
    transposed: bool,
    at: impl Fn(u64, u64) -> f64,
) -> ([u32; 2], Vec<f64>) {
    let (outer, inner) = if transposed { (columns, rows) } else { (rows, columns) };
    let values = (0..u64::from(outer))
        .flat_map(|o| (0..u64::from(inner)).map(move |i| (o, i)))
        .map(|(o, i)| if transposed { at(i, o) } else { at(o, i) })
        .collect();
    ([outer, inner], values)
}

fn tensor(
    dtype: GemmType,
    ([rows, columns], values): ([u32; 2], Vec<f64>),
) -> Result<HostTensor, SweepError> {
    let shape = Shape::new(vec![Dim::new(u64::from(rows)), Dim::new(u64::from(columns))])?;
    Ok(HostTensor::new(shape, data(dtype.dtype(), &values)?)?)
}

/// Whether `output`, C as a framework computed it, matches the host's at
/// evenly spaced elements within its type's tolerance for a K-long sum; if
/// not, how it differs.
pub(super) fn check(problem: GemmProblem, output: &TensorData) -> Result<(), String> {
    let GemmShape { m, n, k } = problem.shape;
    let total = u64::from(m) * u64::from(n);
    let count = SAMPLES.min(total);
    let positions: Vec<u64> = (0..count).map(|s| s * total / count.max(1)).collect();
    let all = values(output).map_err(|error| error.to_string())?;
    let operands = OperandValues::new(problem);
    let mut actual = Vec::new();
    let mut expected = Vec::new();
    for &position in &positions {
        let (i, j) = (position / u64::from(n), position % u64::from(n));
        let index = usize::try_from(position).map_err(|_| "C is too large to index".to_owned())?;
        actual.push(*all.get(index).ok_or("C is smaller than the problem")?);
        let sum: f64 = (0..u64::from(k)).map(|p| operands.a(i, p) * operands.b(p, j)).sum();
        expected.push(sum);
    }
    let output_type = problem.dtype.output();
    let as_tensor = |values: &[f64]| -> Result<HostTensor, String> {
        let data = data(output_type, values).map_err(|error| error.to_string())?;
        Ok(HostTensor::vector(data))
    };
    let tolerance =
        Tolerance::for_dtype(output_type).for_reduction(ElementCount::new(u64::from(k).max(1)));
    compare(&as_tensor(&actual)?, &as_tensor(&expected)?, tolerance)
        .map_err(|mismatch| mismatch.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{Inputs, OperandValues, check, stored, value};
    use crate::host::{data, values};
    use crate::problem::{GemmProblem, GemmShape, GemmType, Layout};
    use ptxc_core::TensorData;

    const SHAPE: GemmShape = GemmShape { m: 3, n: 4, k: 5 };

    fn problem(dtype: GemmType, layout: Layout) -> GemmProblem {
        GemmProblem { shape: SHAPE, dtype, layout }
    }

    /// C computed on the host from the stored inputs, in its output type.
    fn product(problem: GemmProblem) -> TensorData {
        let inputs = Inputs::new(problem).unwrap();
        let (lhs, rhs) = (values(inputs.a.data()).unwrap(), values(inputs.b.data()).unwrap());
        let (rows, columns, depth) = (3, 4, 5);
        let lhs_at = |i: usize, p: usize| {
            if problem.layout.a_transposed() { lhs[p * rows + i] } else { lhs[i * depth + p] }
        };
        let rhs_at = |p: usize, j: usize| {
            if problem.layout.b_transposed() { rhs[j * depth + p] } else { rhs[p * columns + j] }
        };
        let product: Vec<f64> = (0..rows)
            .flat_map(|i| (0..columns).map(move |j| (i, j)))
            .map(|(i, j)| (0..depth).map(|p| lhs_at(i, p) * rhs_at(p, j)).sum())
            .collect();
        data(problem.dtype.output(), &product).unwrap()
    }

    #[test]
    fn values_are_eighths_in_unit_range_or_small_integers() {
        let floats: Vec<f64> = (0..1000).map(|i| value(i, 1, GemmType::F16)).collect();
        let integers: Vec<f64> = (0..1000).map(|i| value(i, 1, GemmType::I8)).collect();

        assert!(floats.iter().all(|v| (-1.0..=1.0).contains(v) && (v * 8.0).fract() == 0.0));
        assert!(integers.iter().all(|v| (-8.0..=8.0).contains(v) && v.fract() == 0.0));
        assert!(floats.iter().any(|&v| v < 0.0) && floats.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn a_transposed_operand_stores_the_same_matrix_transposed() {
        let at = |r: u64, c: u64| f64::from(u32::try_from(r * 10 + c).unwrap());

        let (dims, plain) = stored([2, 3], false, at);
        let (transposed_dims, transposed) = stored([2, 3], true, at);

        assert_eq!((dims, plain), ([2, 3], vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]));
        assert_eq!((transposed_dims, transposed), ([3, 2], vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]));
    }

    #[test]
    fn the_true_product_passes_the_check_in_every_type_and_layout() {
        for dtype in [GemmType::F32, GemmType::F16, GemmType::Bf16, GemmType::E4m3, GemmType::I8] {
            for layout in [Layout::Nn, Layout::Nt, Layout::Tn] {
                let problem = problem(dtype, layout);

                assert_eq!(check(problem, &product(problem)), Ok(()), "{problem:?}");
            }
        }
    }

    #[test]
    fn each_operand_holds_its_own_values_at_their_row_major_positions() {
        let problem = problem(GemmType::F32, Layout::Nn);
        let inputs = Inputs::new(problem).unwrap();
        let (a, b) = (values(inputs.a.data()).unwrap(), values(inputs.b.data()).unwrap());

        let operands = OperandValues::new(problem);
        let salted = |salt, count| (0..count).map(move |index| value(index, salt, GemmType::F32));
        let expected_a: Vec<f64> = salted(operands.a_salt, 15).collect();
        let expected_b: Vec<f64> = salted(operands.b_salt, 20).collect();
        assert_eq!(a, expected_a);
        assert_eq!(b, expected_b);
        assert_ne!(a, b[..15]);
    }

    #[test]
    fn problems_differing_only_in_layout_or_type_have_different_products() {
        let products: Vec<Vec<f64>> = [
            problem(GemmType::F32, Layout::Nn),
            problem(GemmType::F32, Layout::Nt),
            problem(GemmType::E4m3, Layout::Nn),
        ]
        .map(|problem| values(&product(problem)).unwrap())
        .into();

        assert_ne!(products[0], products[1]);
        assert_ne!(products[0], products[2]);
    }

    #[test]
    fn a_wrong_last_element_fails_the_check() {
        let problem = problem(GemmType::F32, Layout::Nn);
        let TensorData::F32(mut c) = product(problem) else { unreachable!() };
        let last = c.len() - 1;
        c[last] += 1.0;

        assert!(check(problem, &TensorData::F32(c)).is_err());
    }

    #[test]
    fn a_wrong_product_fails_the_check() {
        let problem = problem(GemmType::F32, Layout::Nn);
        let TensorData::F32(mut c) = product(problem) else { unreachable!() };
        c[0] += 1.0;

        assert!(check(problem, &TensorData::F32(c)).is_err());
        assert!(check(problem, &TensorData::F32(vec![0.0; 2])).is_err());
    }
}
