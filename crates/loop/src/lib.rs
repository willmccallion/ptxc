//! Loop IR: compute definitions, schedules as replayable traces, lowered loop nests and a CPU interpreter for them.

mod compute;
mod expr;
mod interp;
mod lower;
mod schedule;
mod stmt;
mod text;
mod verify;
