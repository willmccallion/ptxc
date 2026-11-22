//! Attribute values shared by several op families.

use std::hash::{Hash, Hasher};

/// A float attribute. Equality and hashing compare bits, so ops holding one
/// stay `Eq + Hash` (D2).
#[derive(Clone, Copy, Debug)]
pub struct F32Attr(f32);

impl F32Attr {
    /// Wraps a value.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// The value.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for F32Attr {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F32Attr {}

impl Hash for F32Attr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}
