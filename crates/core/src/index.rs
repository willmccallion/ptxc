//! Typed indices: the `define_index!` macro and `IndexVec`, a vector indexed
//! by a typed id, as rustc's `newtype_index!` and `IndexVec`.

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A typed position in an `IndexVec`.
pub trait Idx: Copy + Eq + Ord + Hash + Debug {
    /// The id at position `index`.
    fn new(index: usize) -> Self;

    /// The position this id names.
    fn index(self) -> usize;
}

/// Declares a typed id: a `usize` newtype implementing [`Idx`].
///
/// ```
/// ptxc_core::define_index!(
///     /// A node in a graph.
///     pub NodeId
/// );
/// ```
#[macro_export]
macro_rules! define_index {
    ($(#[$doc:meta])* $vis:vis $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        $vis struct $name(usize);

        impl $crate::Idx for $name {
            fn new(index: usize) -> Self {
                Self(index)
            }

            fn index(self) -> usize {
                self.0
            }
        }
    };
}

/// A vector whose elements are addressed by the id type `I` rather than by
/// `usize`, so an id from one arena cannot index another.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IndexVec<I, T> {
    raw: Vec<T>,
    id: PhantomData<fn(&I)>,
}

impl<I: Idx, T> IndexVec<I, T> {
    /// An empty vector.
    pub const fn new() -> Self {
        Self { raw: Vec::new(), id: PhantomData }
    }

    /// Appends `element` and returns its id.
    pub fn push(&mut self, element: T) -> I {
        let id = self.next_id();
        self.raw.push(element);
        id
    }

    /// The id the next `push` will return.
    pub fn next_id(&self) -> I {
        I::new(self.raw.len())
    }

    /// The element `id` names, if it is in range.
    pub fn get(&self, id: I) -> Option<&T> {
        self.raw.get(id.index())
    }

    /// The element `id` names, if it is in range.
    pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
        self.raw.get_mut(id.index())
    }

    /// The number of elements.
    pub const fn len(&self) -> usize {
        self.raw.len()
    }

    /// Whether the vector has no elements.
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// The elements in id order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.raw.iter()
    }

    /// The elements in id order, each with its id.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (I, &T)> {
        self.raw.iter().enumerate().map(|(index, element)| (I::new(index), element))
    }

    /// Every id in range, in order.
    pub fn ids(&self) -> impl Iterator<Item = I> + use<I, T> {
        (0..self.raw.len()).map(I::new)
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(elements: It) -> Self {
        Self { raw: elements.into_iter().collect(), id: PhantomData }
    }
}

impl<I: Idx, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.raw.into_iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<I: Idx, T> Index<I> for IndexVec<I, T> {
    type Output = T;

    fn index(&self, id: I) -> &T {
        &self.raw[id.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IndexVec<I, T> {
    fn index_mut(&mut self, id: I) -> &mut T {
        &mut self.raw[id.index()]
    }
}
