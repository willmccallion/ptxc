# Testing

Correctness comes from outside: the ONNX backend test suite, ONNX Runtime and
PyTorch outputs, and ptxas. Inside the repository, the reference interpreter
(`ptxc-interp`) is the golden model every other executor is compared with.
It plays the role spike plays for rvsim.

## Layers

| Layer | What it proves | Needs | Runs in |
|---|---|---|---|
| Unit tests | each function's contract, including its error paths | nothing | `cargo test` |
| Golden IR tests | passes, lowering and printing produce the expected IR text | nothing | `cargo test` |
| Float format tests | every encoding round-trips; every f32 rounds to nearest-even | nothing | `cargo test` (exhaustive: `--release -- --ignored`) |
| Loop-interpreter checks | scheduled kernels compute what the graph interpreter computes, without races | nothing | `cargo test` |
| ptxas checks | every emitted kernel assembles for its target | ptxas (flake) | `cargo test` |
| GPU tests | kernels and runtime are correct on the device | GPU | `cargo test --features gpu` |
| ONNX conformance | interpreter and GPU agree with the official node tests | `data/onnx` | `ptxc conformance` |
| Differential fuzzing | random well-typed graphs agree across executors | GPU for `--backend gpu` | `ptxc fuzz` |
| Whole models | accuracy on real data matches PyTorch | models, datasets, cached references | `ptxc check-model` |

## Commands

```sh
cargo test                                   # everything that needs no GPU and no downloads
cargo test --features gpu                    # also the device tests (crates: cuda, runtime, autotune)
cargo test --release -p ptxc-core -- --ignored   # exhaustive float-format checks
cargo xtask fetch onnx-tests                 # pinned ONNX node tests into data/onnx/
ptxc conformance --backend interp            # M1
ptxc conformance --backend gpu               # M2 onwards
ptxc fuzz --backend gpu --seed 1 --cases 10000
ptxc check-model resnet50 --dtype f16        # M4 onwards
```

GPU tests are compiled only with the `gpu` feature. Without it, they don't
exist; with it, a missing GPU is a test failure, never a silent skip.

## Directory layout

```
crates/<crate>/tests/            integration tests through the crate's public API
crates/<crate>/tests/golden/     expected outputs (IR text, reports)
crates/<crate>/tests/fixtures/   small hand-written inputs (graph IR text, tiny .onnx files)
data/onnx/<tag>/node/            ONNX backend node tests (gitignored, fetched)
data/models/<name>/              exported ONNX models (gitignored, produced by baselines/)
data/datasets/                   ImageNet validation subset, ImageNetV2 (D35), WikiText-2 (gitignored)
data/inputs/<set>/               input sets: input_k.pb and input_set.json (gitignored, produced by baselines/)
data/reference/                  cached reference outputs (gitignored)
data/fuzz/                       shrunk failing cases (gitignored; promoted to fixtures by hand)
```

## Golden files

`ptxc_testkit::golden::check(path, actual)` compares with
`tests/golden/<path>`. `PTXC_BLESS=1 cargo test` rewrites the golden files.
Bless only on unchanged code, and review every golden diff in the commit
that changes it.

## ONNX backend node tests

- Each case is a directory holding `model.onnx` and
  `test_data_set_N/{input,output}_K.pb`. The `.pb` files are `TensorProto`s,
  read with the same importer as model weights; tests of sequences and
  optionals store other protobufs there, and count by their model's reason.
- `ptxc conformance --backend interp [--tests <dir>] [--out <dir>]` runs
  every case through the chosen backend and writes:
  - `report.json`: per test, one of `pass`, `fail { failure }` (the worst
    mismatch, or an import, evaluation or loading error) or
    `unsupported { reason }`, where the reason is an op, an element type,
    an attribute value, an opset, control flow, a dynamic shape, or a
    shape computed from values no input fixes;
  - `report.md`: totals by status and by ONNX op (`composite` for the
    multi-node `_expanded` tests), every failure, and the unsupported tests
    by reason.
- It exits with 1 if any test fails, and 2 if it cannot run at all.
- The tolerance is the suite's own: rtol 1e-3, atol 1e-7, compared as
  `|actual - expected| <= atol + rtol * |expected|`.
- The GPU report is diffed against the interpreter report. Any test that
  passes on one backend and not the other is a bug.

## Differential fuzzing

- `ptxc_testkit::generate` builds a random well-typed graph from a `Seed`.
  It picks ops whose type rules accept the values available, picks shapes
  that favour edge cases (dimension 1, primes, non-multiples of tile sizes,
  empty tensors where ONNX allows them), and picks a dtype per subgraph.
- Every case runs on at least two executors through the `Evaluate` trait:
  - the reference interpreter;
  - the loop-IR interpreter;
  - the GPU runtime;
  - cached ONNX Runtime outputs, for cases exported to ONNX.
- On a mismatch, `shrink.rs` tries, until nothing smaller still fails:
  - removing nodes;
  - replacing subgraphs by their inputs;
  - halving dimensions;
  - lowering the precision.
- The shrunk case is written to `data/fuzz/` as graph IR text with its seed.
  When fixed, it becomes a fixture and a regression test.
- Fuzzing is deterministic per seed and runs on stable Rust. It is not
  cargo-fuzz, which needs nightly (D13).

### Tolerances

`compare.rs` uses `|a - r| <= atol + rtol * |r|` per element and reports the
worst element by both measures, with its index.

| DType | atol | rtol | Notes |
|---|---|---|---|
| f32 | 1e-5 | 1e-4 | scaled by √K for reductions of length K |
| f16 | 1e-3 | 1e-2 | scaled by √K for reductions |
| bf16 | 1e-2 | 2e-2 | scaled by √K for reductions |
| E4M3FN / E5M2 | ± 1 step of the output format | — | compared in the encoded domain |
| i8 / u8 (quantized outputs) | ± 1 step | — | rounding ties can legitimately differ after rescaling |
| integer, bool | exact | exact | |

These are starting values, to be calibrated against the observed error of
ORT and PyTorch on the same graphs during M2 and recorded in DECISIONS.md.
A tolerance is never loosened to make a single test pass.

## Whole-model checks

| Model | Data | Metric | Reference | Pass |
|---|---|---|---|---|
| ResNet-50 | ImageNet validation subset (fixed list of 5,000 images) | top-1 | PyTorch, same weights and preprocessing | within 0.1 points (f16); within the budget for each quantized dtype |
| BERT-base | fixed set of 256 sequences | final hidden states and pooled output | PyTorch | per-dtype tolerance |
| GPT-2 124M | WikiText-2 test split | perplexity | PyTorch | within 0.5% |
| MLP | random inputs | outputs | interpreter and PyTorch | per-dtype tolerance |

Accuracy budgets for the quantized models are set when QDQ models land (M6).

## Reference outputs and caching

- `baselines/` (Python, uv) exports models to ONNX and computes reference
  outputs with ORT and PyTorch.
- Outputs are written as ONNX `TensorProto` `.pb` files, so Rust reads them
  with the importer it already has.
- Cache key: `data/reference/<sha256 of model>/<input set id>/<tool>-<tool version>/`,
  plus `manifest.json` (tool versions, device, command line). A changed
  model, input set or tool version is a cache miss, never a stale hit.
- An input set's id is the SHA-256, in hex, over each `input_k.pb` in
  order: its length as 8 little-endian bytes, then its bytes. Python and
  Rust both check it against `crates/testkit/tests/fixtures/input-set/`.
- ORT runs on the CPU with graph optimisations disabled, so it computes the
  model as written.
- Rust never runs Python. If a cached reference is missing, the Rust side
  fails with the exact `uv run` command that produces it.

The target models are checked against ORT by ignored tests in
`ptxc-testkit`, after their data is made:

```
cargo xtask fetch imagenetv2
uv run --project baselines ptxc-baselines export --model resnet50     # also mlp, bert-base, gpt2
uv run --project baselines ptxc-baselines images --source imagenetv2 --count 16
uv run --project baselines ptxc-baselines random --model gpt2 --batch 1 --sequence 128 --seed 0
uv run --project baselines ptxc-baselines reference --model resnet50 --inputs imagenetv2-16
cargo test --release -p ptxc-testkit --test models -- --ignored
```

`random` makes MLP inputs with `--batch 4` and BERT and GPT-2 inputs with
`--batch 1`, as the tests expect. `uv run pytest` in `baselines/`
tests the baselines themselves.

## Test quality

Passing tests show only that the code does what the tests check. Two tools
show what they don't check.

- **Mutation testing** (`p quality mutants`, which runs cargo-mutants) makes
  small edits to the code, such as replacing a function body, flipping a
  comparison or deleting a statement. It then lists every edit that all the
  tests still pass.
  - Run it before opening each pull request. After the first run it tests only
    the previous survivors and changed code.
  - Every surviving mutant is either killed by a new test, or excluded in
    `.cargo/mutants.toml` with a comment saying why no test can see it (for
    example, code that needs the network).
  - A mutant that makes the tests hang (a lexer that stops advancing, say)
    is reported as a timeout. That counts as caught, but `--iterate` tests
    it again on every run, so the run exits non-zero for timeouts alone.
- **Coverage** (`p quality cover`, which runs cargo llvm-cov) lists lines no
  test runs, worst file first. It is a map of untested code, not a target to
  reach: a line can run without anything checking its result, which is what
  mutation testing catches. The flake points cargo llvm-cov at the LLVM tools
  of rustc's own LLVM version.

## Rules

- Every bug fix comes with a test that fails without it: a fixture from the
  fuzzer, a conformance case, or a unit test.
- Tests are deterministic. Seeds are explicit; there is no sleeping and no
  network in `cargo test`.
- Before refactoring a path, pin its behaviour with golden tests first and
  commit them.
- No mutant survives a pull request unless `.cargo/mutants.toml` excludes it
  with a reason.
