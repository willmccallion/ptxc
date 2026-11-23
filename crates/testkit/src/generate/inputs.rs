//! Graph inputs, and the moderate values the connectors give theirs: small
//! enough that a graph of continuous ops on them stays within tolerance of
//! the reference.

use super::shapes::Rng;
use crate::families::values::{floats, integers, shape};
use ptxc_core::convert::int_to_f64;
use ptxc_core::{DType, HostTensor};

/// A graph input: its values, and where they came from.
#[derive(Clone, Debug)]
pub struct Input {
    /// The values.
    pub tensor: HostTensor,
    /// Where they came from.
    pub source: Source,
}

/// Where an input's values came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    /// Drawn for a connector: eighths in `[-2, 2]` for floats, whole
    /// numbers in `[-8, 8]` for integers (`[0, 8]` unsigned), any bools.
    Moderate,
    /// Drawn by the named op family, reaching that op's edge cases.
    Family(&'static str),
}

/// A moderate input of `dtype` and `dims`.
pub(crate) fn moderate(rng: &mut Rng, dtype: DType, dims: &[u64]) -> Input {
    let count = shape(dims).element_count().get();
    let mut draw = |low, high| (0..count).map(|_| rng.between(low, high)).collect::<Vec<_>>();
    let data = match dtype {
        DType::Bool => integers(dtype, &draw(0, 1)),
        DType::U8 => integers(dtype, &draw(0, 8)),
        float if float.is_float() => {
            let eighths: Vec<f64> =
                draw(-16, 16).into_iter().map(|d| int_to_f64(d) / 8.0).collect();
            floats(float, &eighths)
        }
        signed => integers(signed, &draw(-8, 8)),
    };
    let tensor = HostTensor::new(shape(dims), data).unwrap();
    Input { tensor, source: Source::Moderate }
}
