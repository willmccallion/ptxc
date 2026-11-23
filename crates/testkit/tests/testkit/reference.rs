//! Input sets and cached ONNX Runtime references.

use ptxc_core::Dim;
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::data::{DataError, workspace_root};
use ptxc_testkit::reference::{
    ImageSource, InputSet, ReferenceError, Seed, input_set_id, inputs, ort_reference,
    ort_reference_under,
};
use std::path::{Path, PathBuf};

/// The id `ptxc-baselines`' tests also check for this fixture.
const FIXTURE_ID: &str = "1b45c6b27a40be0bd568351f3ad0c40074fa259d4d14b810edc0abfd59e87379";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/input-set")
}

#[test]
fn the_shared_fixture_has_the_id_python_computes() {
    assert_eq!(input_set_id(&fixture()).unwrap(), FIXTURE_ID);
}

#[test]
fn a_directory_without_inputs_has_no_id() {
    let empty = tempfile::tempdir().unwrap();

    assert!(matches!(input_set_id(empty.path()), Err(ReferenceError::Io { .. })));
}

#[test]
fn input_sets_read_in_input_order() {
    let tensors = inputs(&fixture()).unwrap();

    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].dtype(), ptxc_core::DType::F32);
    assert_eq!(tensors[1].dtype(), ptxc_core::DType::I64);
}

#[test]
fn input_sets_are_named_and_made_as_the_baselines_cli_does() {
    let images = InputSet::Images { source: ImageSource::ImageNetV2, count: Dim::new(16) };
    let validation = InputSet::Images { source: ImageSource::ImageNet, count: Dim::new(8) };
    let random = InputSet::Random {
        model: "gpt2",
        batch: Dim::new(2),
        sequence: Dim::new(64),
        seed: Seed::new(7),
    };
    let command =
        |arguments: &str| format!("uv run --project baselines ptxc-baselines {arguments}");

    assert_eq!(images.name(), "imagenetv2-16");
    assert_eq!(images.command(), command("images --source imagenetv2 --count 16"));
    assert_eq!(validation.name(), "imagenet-val-8");
    assert_eq!(validation.command(), command("images --source imagenet --count 8"));
    assert_eq!(random.name(), "gpt2-batch2-sequence64-seed7");
    assert_eq!(random.command(), command("random --model gpt2 --batch 2 --sequence 64 --seed 7"));
}

/// SHA-256 of `model bytes`, the stand-in model file's contents.
const MODEL_SHA256: &str = "9cb7487000bc86ac36ce83c4acfabe8878552be99572a6770f65ab1d048a5c48";

/// A model file and the directory its references for the fixture live in.
fn cached_model(root: &Path) -> (PathBuf, PathBuf) {
    let model = root.join("model.onnx");
    std::fs::write(&model, b"model bytes").unwrap();
    (model, root.join("reference").join(MODEL_SHA256).join(FIXTURE_ID))
}

#[test]
fn the_reference_is_read_from_its_tool_directory() {
    let temp = tempfile::tempdir().unwrap();
    let (model, directory) = cached_model(temp.path());
    let tool = directory.join("ort-1.32.0");
    std::fs::create_dir_all(&tool).unwrap();
    let _bytes: u64 =
        std::fs::copy(fixture().join("input_1.pb"), tool.join("output_0.pb")).unwrap();

    let outputs = ort_reference_under(&temp.path().join("reference"), &model, &fixture()).unwrap();

    let [output] = outputs.as_slice() else { panic!("{} outputs", outputs.len()) };
    assert_eq!(compare(output, &inputs(&fixture()).unwrap()[1], Tolerance::Exact), Ok(()));
}

#[test]
fn a_missing_reference_names_the_command_that_caches_it() {
    let temp = tempfile::tempdir().unwrap();
    let (model, _) = cached_model(temp.path());

    let error = ort_reference_under(&temp.path().join("reference"), &model, &fixture());

    let Err(ReferenceError::Missing(DataError::NotFetched { fetch, .. })) = error else {
        panic!("{error:?}")
    };
    assert!(fetch.starts_with("uv run --project baselines ptxc-baselines reference --model "));
    assert!(fetch.contains(&format!("--inputs {}", fixture().display())), "{fetch}");
}

#[test]
fn several_cached_versions_are_ambiguous() {
    let temp = tempfile::tempdir().unwrap();
    let (model, directory) = cached_model(temp.path());
    for version in ["ort-1.31.0", "ort-1.32.0"] {
        std::fs::create_dir_all(directory.join(version)).unwrap();
    }

    let error = ort_reference_under(&temp.path().join("reference"), &model, &fixture());

    let Err(ReferenceError::Ambiguous { mut found, .. }) = error else { panic!("{error:?}") };
    found.sort();
    assert_eq!(found, ["ort-1.31.0", "ort-1.32.0"]);
}

#[test]
fn a_missing_input_set_names_the_command_that_makes_it() {
    let unmade = InputSet::Random {
        model: "mlp",
        batch: Dim::new(3),
        sequence: Dim::new(5),
        seed: Seed::new(987_654_321),
    };

    let Err(DataError::NotFetched { path, fetch }) = unmade.directory() else { panic!() };

    assert!(path.ends_with("data/inputs/mlp-batch3-sequence5-seed987654321"), "{}", path.display());
    assert_eq!(fetch, unmade.command());
}

#[test]
fn a_reference_directory_without_an_ort_version_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let (model, directory) = cached_model(temp.path());
    std::fs::create_dir_all(directory.join("torch-2.14.1")).unwrap();

    let error = ort_reference_under(&temp.path().join("reference"), &model, &fixture());

    assert!(
        matches!(error, Err(ReferenceError::Missing(DataError::NotFetched { .. }))),
        "{error:?}"
    );
}

#[test]
fn the_workspace_reference_is_looked_up_under_data_reference() {
    let temp = tempfile::tempdir().unwrap();
    let (model, _) = cached_model(temp.path());

    let error = ort_reference(&model, &fixture());

    let Err(ReferenceError::Missing(DataError::NotFetched { path, .. })) = error else {
        panic!("{error:?}")
    };
    let expected = workspace_root().unwrap().join("data/reference").join(MODEL_SHA256);
    assert!(path.starts_with(&expected), "{}", path.display());
}
