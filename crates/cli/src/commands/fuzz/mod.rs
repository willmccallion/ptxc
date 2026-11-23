//! `ptxc fuzz`: differential fuzzing (D13). Each seed's random graph runs on
//! every executor; a mismatch is shrunk and written to the output directory
//! with its seed.

mod check;

use super::runner::Runner;
use crate::args::{Backend as BackendArg, FuzzArgs};
use crate::ui;
use anyhow::Context;
use check::{Backend, check};
use ptxc_graph::Graph;
use ptxc_testkit::generate::{Recipe, Seed, generate};
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
    let mismatched = fuzz(args.seed..end, backend, &args.out)?;
    let agreed = args.cases - u64::try_from(mismatched.len())?;
    ui::stat("agreed", &agreed.to_string());
    ui::stat("mismatched", &mismatched.len().to_string());
    Ok(if mismatched.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(MISMATCHED) })
}

/// The seeds whose cases mismatched, each shrunk and written to `out`.
fn fuzz(seeds: Range<u64>, backend: Option<&Backend<'_>>, out: &Path) -> anyhow::Result<Vec<u64>> {
    let mut mismatched = Vec::new();
    for seed in seeds {
        let recipe = generate(Seed(seed));
        if check(&recipe, backend).is_err() {
            report(seed, recipe, backend, out)?;
            mismatched.push(seed);
        }
    }
    Ok(mismatched)
}

/// Shrinks `recipe`, which mismatched, and writes the smallest failing case
/// with its seed and the mismatch.
fn report(
    seed: u64,
    recipe: Recipe,
    backend: Option<&Backend<'_>>,
    out: &Path,
) -> anyhow::Result<()> {
    let shrunk = shrink(recipe, |candidate| check(candidate, backend).is_err());
    let Err(message) = check(&shrunk, backend) else {
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
    use super::check::check;
    use super::fuzz;
    use ptxc_core::HostTensor;
    use ptxc_graph::Graph;
    use ptxc_testkit::generate::{Seed, generate};
    use ptxc_testkit::lowered::run_lowered;

    fn on_loop_interpreter(
        graph: &Graph,
        inputs: &[HostTensor],
    ) -> Result<Vec<HostTensor>, String> {
        run_lowered(graph.clone(), inputs, ptxc_loop::run).map_err(|error| error.to_string())
    }

    #[test]
    fn a_backend_matching_the_loop_interpreter_agrees_on_every_seed() {
        let dir = tempfile::tempdir().unwrap();

        let mismatched = fuzz(0..40, Some(&on_loop_interpreter), dir.path()).unwrap();

        assert_eq!(mismatched, Vec::<u64>::new());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_failing_backend_mismatches_and_each_seed_is_written_shrunk() {
        let dir = tempfile::tempdir().unwrap();
        let broken = |_: &Graph, _: &[HostTensor]| Err("broken".to_owned());

        let mismatched = fuzz(5..8, Some(&broken), dir.path()).unwrap();

        assert_eq!(mismatched, [5, 6, 7]);
        let text = std::fs::read_to_string(dir.path().join("seed-6.txt")).unwrap();
        assert!(text.starts_with("seed 6\nthe backend failed: broken\n"), "{text}");
        assert_eq!(text.matches("Node {").count(), 1, "{text}");
    }

    #[test]
    fn a_backend_returning_the_wrong_outputs_mismatches() {
        let none = |_: &Graph, _: &[HostTensor]| Ok(Vec::new());

        let verdict = check(&generate(Seed(0)), Some(&none));

        assert_eq!(
            verdict,
            Err("the backend against the loop interpreter: 0 outputs, expected 1".into())
        );
    }
}
