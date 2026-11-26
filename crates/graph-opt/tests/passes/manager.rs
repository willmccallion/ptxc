use crate::graph;
use ptxc_core::Idx;
use ptxc_graph::{CycleError, Graph, NodeId, NodeOutputIndex, VerifyError};
use ptxc_graph_opt::{
    Canonicalize, ConstFold, Cse, Dce, Pass, PassError, PassManager, PassReport, PipelineError,
    Simplify,
};

/// Makes the first node's output read the second node's, closing a cycle,
/// which the rewriter allows and the verifier refuses.
#[derive(Debug)]
struct CloseACycle;

impl Pass for CloseACycle {
    fn name(&self) -> &'static str {
        "close-a-cycle"
    }

    fn run(&self, graph: &mut Graph) -> Result<PassReport, PassError> {
        let outputs: Vec<_> =
            graph.nodes().map(|(_, node)| node.outputs()[NodeOutputIndex::new(0)]).collect();
        graph.rewriter().replace_all_uses(outputs[0], outputs[1])?;
        Ok(PassReport { changed: true })
    }
}

/// Fails as soon as it runs.
#[derive(Debug)]
struct Refuse;

impl Pass for Refuse {
    fn name(&self) -> &'static str {
        "refuse"
    }

    fn run(&self, _: &mut Graph) -> Result<PassReport, PassError> {
        Err(PassError::Cycle(CycleError(NodeId::new(0))))
    }
}

const CHAIN: &str = "\
graph(%x: f32[4]) {
  %0 = neg %x : f32[4]
  %1 = abs %0 : f32[4]
  return %1
}
";

#[test]
fn passes_run_in_order_each_reporting() {
    let mut chain = graph(CHAIN);
    let manager = PassManager::new(vec![Box::new(Dce), Box::new(Dce)]).verifying_each_pass();

    let reports = manager.run(&mut chain).unwrap();

    assert_eq!(reports, [PassReport { changed: false }; 2]);
}

#[test]
fn a_pass_that_leaves_an_invalid_graph_is_named() {
    let mut chain = graph(CHAIN);
    let manager = PassManager::new(vec![Box::new(CloseACycle)]).verifying_each_pass();

    let error = manager.run(&mut chain).unwrap_err();

    let PipelineError::Invalid { pass, source: VerifyError::Cycle(_) } = error else {
        panic!("{error:?}")
    };
    assert_eq!(pass, "close-a-cycle");
}

#[test]
fn a_pass_that_fails_is_named_and_stops_the_rest() {
    let mut chain = graph(CHAIN);
    let manager = PassManager::new(vec![Box::new(Refuse), Box::new(CloseACycle)]);

    let error = manager.run(&mut chain).unwrap_err();

    let PipelineError::Failed { pass: "refuse", source: PassError::Cycle(CycleError(node)) } =
        error
    else {
        panic!("{error:?}")
    };
    assert_eq!(node, NodeId::new(0));
    assert_eq!(chain.verify(), Ok(()));
}

#[test]
fn a_pass_on_a_graph_with_a_cycle_fails_under_its_own_name() {
    let passes: [(Box<dyn Pass>, &str); 5] = [
        (Box::new(Simplify), "simplify"),
        (Box::new(ConstFold), "const-fold"),
        (Box::new(Canonicalize), "canonicalize"),
        (Box::new(Cse), "cse"),
        (Box::new(Dce), "dce"),
    ];
    for (refusing, name) in passes {
        let mut chain = graph(CHAIN);
        let _invalid = PassManager::new(vec![Box::new(CloseACycle)]).run(&mut chain);

        let error = PassManager::new(vec![refusing]).run(&mut chain).unwrap_err();

        let PipelineError::Failed { pass, source: PassError::Cycle(_) } = error else {
            panic!("{error:?}")
        };
        assert_eq!(pass, name);
    }
}

/// Release builds verify only when asked; debug builds always do, so this
/// runs only under `--release`.
#[cfg(not(debug_assertions))]
#[test]
fn release_builds_verify_after_each_pass_only_when_asked() {
    let unverified = PassManager::new(vec![Box::new(CloseACycle)]);
    let verified = PassManager::new(vec![Box::new(CloseACycle)]).verifying_each_pass();

    let unchecked = unverified.run(&mut graph(CHAIN));
    let checked = verified.run(&mut graph(CHAIN));

    assert_eq!(unchecked.unwrap(), [PassReport { changed: true }]);
    assert!(
        matches!(checked, Err(PipelineError::Invalid { pass: "close-a-cycle", .. })),
        "{checked:?}"
    );
}

/// Debug builds verify after every pass unasked; this runs only without
/// `--release`.
#[cfg(debug_assertions)]
#[test]
fn debug_builds_verify_after_each_pass_unasked() {
    let unasked = PassManager::new(vec![Box::new(CloseACycle)]);

    let checked = unasked.run(&mut graph(CHAIN));

    assert!(
        matches!(checked, Err(PipelineError::Invalid { pass: "close-a-cycle", .. })),
        "{checked:?}"
    );
}
