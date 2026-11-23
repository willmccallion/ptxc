//! `ptxc fuzz`: differential fuzzing (D13, D53). Each seed's random graph
//! runs on every executor; a mismatch is shrunk and written to the output
//! directory with its seed.

mod check;

use super::runner::Runner;
use crate::args::{Backend as BackendArg, FuzzArgs};
use crate::ui;
use anyhow::Context;
use check::{Backend, Failure, check};
use ptxc_graph::Graph;
use ptxc_testkit::generate::{Generator, Recipe, Seed};
use ptxc_testkit::shrink::shrink;
use std::ops::Range;
use std::path::Path;
use std::process::ExitCode;

/// The exit code when a case mismatches.
const MISMATCHED: u8 = 1;

/// Runs `args.cases` seeds from `args.seed`, shrinking and writing every
/// mismatch, and fails if there was one.
pub(crate) fn run(args: &FuzzArgs) -> anyhow::Result<ExitCode> {
    ui::section("Fuzz");
    ui::header("Backend", args.backend.name());
    ui::header("Seeds", &format!("{} from {}", args.cases, args.seed));
    let end = args.seed.checked_add(args.cases).context("the seeds run past u64::MAX")?;
    let runner = Runner::new(args.backend)?;
    let on_runner = |graph: &Graph, inputs: &[_]| runner.evaluate(graph, inputs);
    let backend: Option<&Backend<'_>> = match args.backend {
        BackendArg::Interp => None,
        BackendArg::Gpu => Some(&on_runner),
    };
    let tally = fuzz(args.seed..end, backend, &args.out)?;
    ui::stat("agreed", &tally.agreed.to_string());
    ui::stat("of them against the reference", &tally.against_reference.to_string());
    ui::stat("rejected by the loop interpreter", &tally.rejected.to_string());
    ui::stat("mismatched", &tally.mismatched.len().to_string());
    let passed = tally.mismatched.is_empty();
    Ok(if passed { ExitCode::SUCCESS } else { ExitCode::from(MISMATCHED) })
}

/// What a run of seeds found.
#[derive(Debug, Default, PartialEq, Eq)]
struct Tally {
    agreed: u64,
    /// Agreed cases also within tolerance of the reference interpreter.
    against_reference: u64,
    rejected: u64,
    /// The seeds that mismatched.
    mismatched: Vec<u64>,
}

/// Every seed's case checked, each mismatch shrunk and written to `out`.
fn fuzz(seeds: Range<u64>, backend: Option<&Backend<'_>>, out: &Path) -> anyhow::Result<Tally> {
    let generator = Generator::default();
    let mut tally = Tally::default();
    for seed in seeds {
        let recipe = generator.generate(Seed(seed));
        match check(&recipe, backend) {
            Ok(()) => {
                tally.agreed += 1;
                tally.against_reference += u64::from(recipe.compares_with_reference());
            }
            Err(Failure::Rejected(_)) => tally.rejected += 1,
            Err(Failure::Mismatch(_)) => {
                report(seed, recipe, backend, out)?;
                tally.mismatched.push(seed);
            }
        }
    }
    Ok(tally)
}

/// Shrinks `recipe`, which mismatched, and writes the smallest failing case
/// with its seed and the mismatch.
fn report(
    seed: u64,
    recipe: Recipe,
    backend: Option<&Backend<'_>>,
    out: &Path,
) -> anyhow::Result<()> {
    let still_mismatches =
        |candidate: &Recipe| matches!(check(candidate, backend), Err(Failure::Mismatch(_)));
    let shrunk = shrink(recipe, still_mismatches);
    let Err(Failure::Mismatch(message)) = check(&shrunk, backend) else {
        anyhow::bail!("seed {seed}'s shrunk case no longer fails");
    };
    let case = shrunk.build().context("a shrunk case always builds")?;
    let graph = ptxc_graph::text::print(&case.graph);
    let text = format!("seed {seed}\n{message}\n\n{graph}\nrecipe: {shrunk:?}\n");
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let path = out.join(format!("seed-{seed}.txt"));
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    ui::stat(&format!("seed {seed}"), &format!("mismatch, shrunk to {}", path.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check::{Failure, check};
    use super::fuzz;
    use ptxc_core::HostTensor;
    use ptxc_graph::{BinaryOp, Graph, Op};
    use ptxc_testkit::generate::{Generator, Input, Node, Operand, Recipe, Seed, Source};
    use ptxc_testkit::lowered::run_lowered;

    fn on_loop_interpreter(
        graph: &Graph,
        inputs: &[HostTensor],
    ) -> Result<Vec<HostTensor>, String> {
        run_lowered(graph.clone(), inputs, ptxc_loop::run).map_err(|error| error.to_string())
    }

    fn integers(values: Vec<i32>) -> Input {
        Input { tensor: HostTensor::vector(values.into()), source: Source::Moderate }
    }

    #[test]
    fn a_backend_matching_the_loop_interpreter_agrees_on_every_seed() {
        let dir = tempfile::tempdir().unwrap();

        let tally = fuzz(0..40, Some(&on_loop_interpreter), dir.path()).unwrap();

        assert_eq!(tally.mismatched, Vec::<u64>::new());
        assert!(tally.agreed > tally.rejected, "{tally:?}");
        assert!(tally.against_reference > 0 && tally.against_reference < tally.agreed);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_failing_backend_mismatches_and_each_seed_is_written_shrunk() {
        let dir = tempfile::tempdir().unwrap();
        let broken = |_: &Graph, _: &[HostTensor]| Err("broken".to_owned());

        let tally = fuzz(0..6, Some(&broken), dir.path()).unwrap();

        assert_eq!(tally.agreed, 0);
        assert_eq!(u64::try_from(tally.mismatched.len()).unwrap() + tally.rejected, 6);
        let seed = tally.mismatched[0];
        let text = std::fs::read_to_string(dir.path().join(format!("seed-{seed}.txt"))).unwrap();
        assert!(text.starts_with(&format!("seed {seed}\nthe backend failed: broken\n")), "{text}");
        assert_eq!(text.matches("Node {").count(), 1, "{text}");
    }

    #[test]
    fn a_backend_returning_the_wrong_outputs_mismatches() {
        let generator = Generator::default();
        let recipe = (0..40)
            .map(|seed| generator.generate(Seed(seed)))
            .find(|recipe| check(recipe, None).is_ok())
            .unwrap();
        let none = |_: &Graph, _: &[HostTensor]| Ok(Vec::new());

        let verdict = check(&recipe, Some(&none));

        let Err(Failure::Mismatch(message)) = verdict else { panic!("{verdict:?}") };
        assert!(message.starts_with("the backend against the loop interpreter: 0 outputs"));
    }

    #[test]
    fn an_integer_division_by_zero_is_rejected_before_any_backend_runs() {
        let recipe = Recipe {
            inputs: vec![integers(vec![6, 4]), integers(vec![3, 0])],
            nodes: vec![Node {
                op: Op::Binary(BinaryOp::Div),
                operands: vec![Operand::Input(0), Operand::Input(1)],
            }],
        };
        let unreachable = |_: &Graph, _: &[HostTensor]| Err("the backend ran".to_owned());

        let verdict = check(&recipe, Some(&unreachable));

        assert_eq!(verdict, Err(Failure::Rejected("integer division by zero".into())));
    }
}
