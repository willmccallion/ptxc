//! `Microseconds`: the unit ptxc measures time in.

/// A duration in microseconds.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Microseconds(f64);

impl Microseconds {
    /// Wraps a duration.
    pub const fn new(microseconds: f64) -> Self {
        Self(microseconds)
    }

    /// A duration given in milliseconds, as CUDA events report them.
    pub fn from_millis(milliseconds: f32) -> Self {
        Self(f64::from(milliseconds) * 1000.0)
    }

    /// The duration.
    pub const fn get(self) -> f64 {
        self.0
    }
}
