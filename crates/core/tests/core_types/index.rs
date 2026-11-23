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
fn into_iter_yields_the_elements_in_id_order() {
    let elements: IndexVec<TestId, char> = "abc".chars().collect();

    let owned: String = elements.into_iter().collect();

    assert_eq!(owned, "abc");
}

#[test]
fn new_vector_is_empty() {
    assert!(IndexVec::<TestId, char>::new().is_empty());
}

#[test]
fn get_returns_the_element_under_an_id_in_range() {
    let elements: IndexVec<TestId, char> = "ab".chars().collect();

    assert_eq!(elements.get(TestId::new(1)), Some(&'b'));
}

#[test]
fn get_mut_changes_the_element_in_place() {
    let mut elements: IndexVec<TestId, char> = "ab".chars().collect();

    *elements.get_mut(TestId::new(0)).unwrap() = 'z';

    assert_eq!(elements[TestId::new(0)], 'z');
    assert_eq!(elements.get_mut(TestId::new(2)), None);
}

#[test]
fn a_vector_with_elements_is_not_empty() {
    let elements: IndexVec<TestId, char> = "a".chars().collect();

    assert!(!elements.is_empty());
}

#[test]
fn iterating_by_reference_visits_each_element_in_id_order() {
    let elements: IndexVec<TestId, char> = "abc".chars().collect();

    let mut visited = String::new();
    for &element in &elements {
        visited.push(element);
    }

    assert_eq!(visited, "abc");
}

#[test]
fn the_length_counts_pushed_elements() {
    let mut elements = IndexVec::<TestId, char>::new();
    assert_eq!(elements.len(), 0);

    let _first = elements.push('a');
    let _second = elements.push('b');

    assert_eq!(elements.len(), 2);
}
