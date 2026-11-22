use ptxc_core::{Axis, Dim, ElementCount, Rank, Shape, ShapeError};

fn shape(extents: &[u64]) -> Shape {
    Shape::new(extents.iter().copied().map(Dim::new).collect()).unwrap()
}

fn resolve(axis: i64, rank: usize) -> Result<usize, ShapeError> {
    Axis::from_signed(axis, Rank::new(rank)).map(Axis::index)
}

#[test]
fn element_count_is_the_product_of_the_dims() {
    assert_eq!(shape(&[2, 3, 5]).element_count(), ElementCount::new(30));
}

#[test]
fn scalar_has_rank_zero_and_one_element() {
    let scalar = Shape::scalar();

    assert_eq!((scalar.rank(), scalar.element_count()), (Rank::new(0), ElementCount::new(1)));
    assert_eq!(scalar, shape(&[]));
}

#[test]
fn a_zero_dim_makes_the_shape_empty() {
    assert_eq!(shape(&[4, 0, 3]).element_count(), ElementCount::new(0));
}

#[test]
fn element_count_at_u64_max_is_accepted() {
    assert_eq!(shape(&[u64::MAX, 1]).element_count(), ElementCount::new(u64::MAX));
}

#[test]
fn element_count_past_u64_max_is_rejected() {
    let dims = vec![Dim::new(1 << 32), Dim::new(1 << 32)];

    let error = Shape::new(dims.clone()).unwrap_err();

    assert_eq!(error, ShapeError::ElementCountOverflow { dims });
}

#[test]
fn dim_returns_the_extent_of_an_axis_and_none_past_the_rank() {
    let shape = shape(&[7, 9]);
    let [outer, inner] = [0, 1].map(|axis| Axis::from_signed(axis, shape.rank()).unwrap());

    assert_eq!((shape.dim(outer), shape.dim(inner)), (Some(Dim::new(7)), Some(Dim::new(9))));
    assert_eq!(Shape::scalar().dim(outer), None);
}

#[test]
fn non_negative_axes_below_the_rank_are_kept() {
    assert_eq!((resolve(0, 3).unwrap(), resolve(2, 3).unwrap()), (0, 2));
}

#[test]
fn negative_axes_count_from_the_innermost() {
    assert_eq!((resolve(-1, 3).unwrap(), resolve(-3, 3).unwrap()), (2, 0));
}

#[test]
fn axes_outside_minus_rank_to_rank_are_rejected() {
    for axis in [3, -4, i64::MAX, i64::MIN] {
        let error = resolve(axis, 3).unwrap_err();

        assert_eq!(error, ShapeError::AxisOutOfRange { axis, rank: Rank::new(3) });
    }
}

#[test]
fn a_scalar_has_no_axes() {
    assert!(resolve(0, 0).is_err());
    assert!(resolve(-1, 0).is_err());
    assert_eq!(Rank::new(0).axes().count(), 0);
}

#[test]
fn rank_axes_lists_each_axis_outermost_first() {
    let indices: Vec<_> = Rank::new(3).axes().map(Axis::index).collect();

    assert_eq!(indices, [0, 1, 2]);
}

#[test]
fn dim_get_returns_its_extent() {
    assert_eq!(Dim::new(7).get(), 7);
}

#[test]
fn rank_get_returns_its_number_of_dimensions() {
    assert_eq!(Rank::new(3).get(), 3);
}

#[test]
fn dims_lists_the_extents_outermost_first() {
    assert_eq!(shape(&[2, 5]).dims(), [Dim::new(2), Dim::new(5)]);
}

#[test]
fn axis_new_keeps_its_index() {
    assert_eq!(Axis::new(5).index(), 5);
}
