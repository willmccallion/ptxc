//! Names in compute definitions: of the definition, its tensors and its
//! variables. They label printed text; nothing is looked up by name.

use std::fmt;

macro_rules! names {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(String);

        impl $name {
            /// Wraps a name.
            pub fn new(name: impl Into<String>) -> Self {
                Self(name.into())
            }

            /// The name.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    )*};
}

names! {
    /// A compute definition's name.
    ComputeName,
    /// An input or output tensor's name.
    TensorName,
    /// An iteration or index variable's name.
    VarName,
}
