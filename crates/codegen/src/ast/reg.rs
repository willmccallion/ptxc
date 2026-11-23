//! Virtual registers typed by PTX type; ptxas allocates physical registers.

use std::marker::PhantomData;

/// The declared type of a register, which fixes its size: predicates,
/// 16, 32 and 64 bits, and f32.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum RegClass {
    Pred,
    B16,
    B32,
    B64,
    F32,
}

impl RegClass {
    /// Every class, in declaration order.
    pub(crate) const ALL: [Self; 5] = [Self::Pred, Self::B16, Self::B32, Self::B64, Self::F32];

    /// The position of this class in `ALL`.
    pub(crate) const fn slot(self) -> usize {
        match self {
            Self::Pred => 0,
            Self::B16 => 1,
            Self::B32 => 2,
            Self::B64 => 3,
            Self::F32 => 4,
        }
    }

    /// The register name prefix.
    pub(crate) const fn prefix(self) -> &'static str {
        match self {
            Self::Pred => "%p",
            Self::B16 => "%h",
            Self::B32 => "%r",
            Self::B64 => "%rd",
            Self::F32 => "%f",
        }
    }

    /// The type a register of this class is declared with.
    pub(crate) const fn declared(self) -> &'static str {
        match self {
            Self::Pred => ".pred",
            Self::B16 => ".b16",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::F32 => ".f32",
        }
    }
}

/// A register of any class, as instructions store it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RegId {
    pub(crate) class: RegClass,
    pub(crate) index: u32,
}

/// A marker for a register class, so a typed register knows its class.
pub(crate) trait Class: Copy {
    const CLASS: RegClass;
}

macro_rules! classes {
    ($($(#[$doc:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$doc])*
            #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
            pub(crate) struct $name;

            impl Class for $name {
                const CLASS: RegClass = RegClass::$name;
            }
        )+
    };
}

classes!(
    /// A predicate.
    Pred,
    /// 16 bits: f16 and bf16 in memory form.
    B16,
    /// 32-bit integers, and 8-bit ones widened.
    B32,
    /// 64-bit integers and addresses.
    B64,
    /// f32, and every narrower float widened.
    F32,
);

/// A register of class `C`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Reg<C> {
    index: u32,
    class: PhantomData<C>,
}

impl<C: Class> Reg<C> {
    pub(crate) const fn new(index: u32) -> Self {
        Self { index, class: PhantomData }
    }

    pub(crate) const fn id(self) -> RegId {
        RegId { class: C::CLASS, index: self.index }
    }
}
