use ptxc_core::{Idx, IndexVec};

ptxc_core::define_index!(
    /// An id used only by these tests.
    TestId
);

#[test]
fn push_returns_consecutive_ids_from_zero() {
    let mut elements = IndexVec::<TestId, char>::new();

    let ids = [elements.push('a'), elements.push('b'), elements.push('c')];

    assert_eq!(ids, [TestId::new(0), TestId::new(1), TestId::new(2)]);
}

#[test]
fn next_id_is_the_id_push_returns() {
    let mut elements = IndexVec::<TestId, char>::new();
    let _ = elements.push('a');

    let predicted = elements.next_id();

    assert_eq!(elements.push('b'), predicted);
}

#[test]
fn indexing_returns_the_element_pushed_under_that_id() {
    let mut elements = IndexVec::<TestId, char>::new();
    let a = elements.push('a');
    let b = elements.push('b');

    elements[a] = 'x';

    assert_eq!((elements[a], elements[b]), ('x', 'b'));
}

#[test]
fn get_is_none_past_the_end() {
    let mut elements = IndexVec::<TestId, char>::new();
    let _ = elements.push('a');

    assert_eq!(elements.get(TestId::new(1)), None);
}

#[test]
fn iter_enumerated_pairs_each_element_with_its_id() {
    let elements: IndexVec<TestId, char> = "ab".chars().collect();

    let pairs: Vec<_> = elements.iter_enumerated().collect();

    assert_eq!(pairs, [(TestId::new(0), &'a'), (TestId::new(1), &'b')]);
}

#[test]
fn ids_lists_every_id_in_order() {
    let elements: IndexVec<TestId, char> = "abc".chars().collect();

    let ids: Vec<_> = elements.ids().map(Idx::index).collect();

    assert_eq!(ids, [0, 1, 2]);
}

#[test]
fn new_vector_is_empty() {
    assert!(IndexVec::<TestId, char>::new().is_empty());
}
