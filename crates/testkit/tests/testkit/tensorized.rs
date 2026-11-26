//! The inputs of tensorized products: small integers, so every sum of
//! their products is exact in any order.

use ptxc_core::{DType, Dim, HostTensor, Shape, TensorData};
use ptxc_testkit::tensorized::exact_tensor;

fn six(dtype: DType, salt: u64) -> HostTensor {
    exact_tensor(dtype, Shape::new(vec![Dim::new(6)]).unwrap(), salt)
}

#[test]
fn exact_tensors_hold_small_integers_varied_by_their_salt() {
    let first = six(DType::F32, 0);
    let second = six(DType::F32, 1);
    let unsigned = six(DType::U8, 2);

    let (TensorData::F32(first), TensorData::F32(second), TensorData::U8(unsigned)) =
        (first.data(), second.data(), unsigned.data())
    else {
        panic!("{first:?} {second:?} {unsigned:?}")
    };
    assert_eq!(first, &[-4.0, -1.0, 2.0, -4.0, -1.0, 2.0]);
    assert_eq!(second, &[3.0, -3.0, 0.0, 3.0, -3.0, 0.0]);
    assert_eq!(unsigned, &[5, 8, 2, 5, 8, 2]);
}
