//! Durations.

use ptxc_core::Microseconds;

#[test]
fn milliseconds_convert_to_microseconds() {
    assert_eq!(Microseconds::from_millis(1.5), Microseconds::new(1500.0));
    assert_eq!(Microseconds::from_millis(0.0).get(), 0.0);
}
