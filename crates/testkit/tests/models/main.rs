//! The target models against ONNX Runtime: each imports, and the reference
//! interpreter matches ONNX Runtime's cached outputs on a fixed input set.
//! The models, input sets and references come from `baselines/`; a missing
//! one fails with the command that makes it (TESTING.md lists them). Run
//! with `--release`: the interpreter is naive, and ResNet-50 on 16 images
//! takes minutes.

#![allow(clippy::unwrap_used, clippy::panic, reason = "test code: a panic is a test failure")]

use ptxc_core::{DType, Dim, ElementCount, Evaluate};
use ptxc_interp::Interpreter;
use ptxc_onnx::{ImportOptions, import_model};
use ptxc_testkit::compare::{Tolerance, compare};
use ptxc_testkit::data::model;
use ptxc_testkit::reference::{ImageSource, InputSet, Seed, inputs, ort_reference};
use std::collections::HashMap;

/// A target model, the input set it is checked on, and its longest
/// reduction, which widens the f32 tolerance by its square root.
struct Check {
    model: &'static str,
    inputs: InputSet,
    longest_reduction: ElementCount,
}

const SEQUENCE: Dim = Dim::new(128);

const fn random(model: &'static str, batch: u64) -> InputSet {
    InputSet::Random { model, batch: Dim::new(batch), sequence: SEQUENCE, seed: Seed::new(0) }
}

const MLP: Check =
    Check { model: "mlp", inputs: random("mlp", 4), longest_reduction: ElementCount::new(784) };

const RESNET50: Check = Check {
    model: "resnet50",
    inputs: InputSet::Images { source: ImageSource::ImageNetV2, count: Dim::new(16) },
    longest_reduction: ElementCount::new(3 * 3 * 512),
};

const BERT_BASE: Check = Check {
    model: "bert-base",
    inputs: random("bert-base", 1),
    longest_reduction: ElementCount::new(3072),
};

const GPT2: Check =
    Check { model: "gpt2", inputs: random("gpt2", 1), longest_reduction: ElementCount::new(3072) };

/// The exported models' symbolic dims, `batch` and `sequence`, bound to the
/// input set's.
fn import_options(inputs: &InputSet) -> ImportOptions {
    let dims = match inputs {
        InputSet::Images { count, .. } => vec![("batch", *count)],
        InputSet::Random { batch, sequence, .. } => {
            vec![("batch", *batch), ("sequence", *sequence)]
        }
    };
    let symbolic_dims: HashMap<String, Dim> =
        dims.into_iter().map(|(name, dim)| (name.to_owned(), dim)).collect();
    ImportOptions { symbolic_dims, ..ImportOptions::default() }
}

fn import(check: &Check) -> ptxc_graph::Graph {
    let path = model(check.model).unwrap_or_else(|error| panic!("{error}"));
    import_model(&std::fs::read(&path).unwrap(), &import_options(&check.inputs))
        .unwrap_or_else(|error| panic!("{}: {error}", check.model))
}

#[test]
#[ignore = "needs the exported models: uv run --project baselines ptxc-baselines export --model <name>"]
fn every_target_model_imports_and_verifies() {
    for check in [&MLP, &RESNET50, &BERT_BASE, &GPT2] {
        let graph = import(check);

        assert!(graph.verify().is_ok(), "{}: {:?}", check.model, graph.verify());
    }
}

/// Runs `check` on the interpreter and compares every output with ONNX
/// Runtime's.
fn matches_onnx_runtime(check: &Check) {
    let path = model(check.model).unwrap_or_else(|error| panic!("{error}"));
    let directory = check.inputs.directory().unwrap_or_else(|error| panic!("{error}"));
    let expected = ort_reference(&path, &directory).unwrap_or_else(|error| panic!("{error}"));
    let graph = import(check);

    let actual = Interpreter::new(&graph).evaluate(&inputs(&directory).unwrap()).unwrap();

    assert_eq!(actual.len(), expected.len());
    let tolerance = Tolerance::for_dtype(DType::F32).for_reduction(check.longest_reduction);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        if let Err(mismatch) = compare(actual, expected, tolerance) {
            panic!("{} output {index}: {mismatch}", check.model);
        }
    }
}

#[test]
#[ignore = "needs the MLP export, its input set and its ONNX Runtime reference"]
fn mlp_matches_onnx_runtime() {
    matches_onnx_runtime(&MLP);
}

#[test]
#[ignore = "needs the ResNet-50 export, 16 ImageNetV2 images and their ONNX Runtime reference"]
fn resnet50_matches_onnx_runtime_on_16_images() {
    matches_onnx_runtime(&RESNET50);
}

#[test]
#[ignore = "needs the BERT-base export, its input set and its ONNX Runtime reference"]
fn bert_base_matches_onnx_runtime() {
    matches_onnx_runtime(&BERT_BASE);
}

#[test]
#[ignore = "needs the GPT-2 export, its input set and its ONNX Runtime reference"]
fn gpt2_matches_onnx_runtime() {
    matches_onnx_runtime(&GPT2);
}
