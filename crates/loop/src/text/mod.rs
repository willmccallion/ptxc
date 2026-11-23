//! A text format for loop IR, used by golden tests.

mod index;
mod print;
mod schedule;

pub use print::print_compute;
pub use schedule::print_schedule;
