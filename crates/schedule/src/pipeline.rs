//! cp.async software pipelining: how many stages a default schedule fills
//! its staged loop ahead by.

use ptxc_core::ByteSize;
use ptxc_loop::StageCount;
use ptxc_target::Target;

/// `wanted` stages for a loop staging `step` bytes a step, or as many as
/// `target`'s shared memory holds if fewer, if that is at least two.
pub(crate) fn stages_for(
    step: ByteSize,
    wanted: StageCount,
    target: &Target,
) -> Option<StageCount> {
    let limit = u64::from(target.limits().block.smem.get());
    let fitting = limit.checked_div(step.get())?;
    let stages = fitting.min(u64::from(wanted.get()));
    StageCount::new(u32::try_from(stages).ok()?)
}

#[cfg(test)]
mod tests {
    use super::stages_for;
    use ptxc_core::ByteSize;
    use ptxc_loop::StageCount;
    use ptxc_target::{SmVersion, preset};

    #[test]
    fn a_loop_takes_the_stages_it_wants_or_as_many_as_fit_if_at_least_two() {
        let target = preset(SmVersion::SM_89).unwrap();
        let four = StageCount::new(4).unwrap();
        let kib = |count: u64| ByteSize::new(count * 1024);

        let stages = [4, 16, 24, 25].map(|step| stages_for(kib(step), four, &target));

        let expected = [StageCount::new(4), StageCount::new(3), StageCount::new(2), None];
        assert_eq!(stages, expected);
    }
}
