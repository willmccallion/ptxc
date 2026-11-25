# Decisions

Each entry gives the options considered, the choice, and the reason. A
decision is changed by adding a new entry that supersedes it, not by editing
the old one.

## D1. Stable Rust, pinned to the flake's rustc

- **Options:** stable or nightly.
- **Choice:** stable. `rust-toolchain.toml` pins 1.99.1, the rustc in the
  flake's nixpkgs.
- **Reason:** nothing planned needs nightly. The interpreter is scalar, PTX
  is text, and FFI is plain `extern "C"`. Pinning keeps the deny-level
  pedantic and nursery lints from drifting.

## D2. Graph IR: SSA in typed arenas, ops as a typed enum

- **Options:** `Rc<RefCell<Node>>` graphs; arenas with typed ids; an e-graph
  as the main representation.
- **Choice:**
  - arenas (`IndexVec<NodeId, _>`, `IndexVec<ValueId, _>`) in SSA form;
  - an `Op` enum whose variants carry typed attribute structs;
  - all mutation through `Rewriter`;
  - erased nodes are tombstoned and removed by `compact()`.
- **Reason:**
  - ONNX is already SSA;
  - arenas give cheap copies, stable ids for side tables, and no borrow
    tangles;
  - typed attributes are parsed once at the boundary.
- **E-graph path:**
  - `Op` is `Eq + Hash`, with float attributes stored as bits;
  - rewrite rules are data (`Pattern`, `Template`, `Guard`) consumed through
    the `RewriteEngine` trait;
  - an e-graph engine can be added beside the greedy one without touching
    `Op` or any rule.

## D3. Static shapes in v0.1.0

- **Options:** static only; symbolic dimensions with shape functions;
  bucketed specialisation.
- **Choice:** every dimension is known at compile time. A model with a
  symbolic batch dimension is compiled once per requested batch size
  (`--batch`).
- **Reason:**
  - static shapes make tiling, memory planning and CUDA graphs exact;
  - the benchmarks fix their batch sizes anyway.
- `Dim` is a newtype, so a symbolic variant can be added later without
  changing signatures that take `Dim`.

## D4. Loop IR: compute, schedule as a trace, lowered tree

- **Options:** a TVM-style tensor-expression and schedule split; Triton-style
  block programs; an affine/polyhedral IR.
- **Choice:**
  - Halide and TVM's split between `ComputeDef` (what) and schedule (how);
  - the schedule is recorded as a replayable `Trace` of `Primitive`s (TVM
    MetaSchedule);
  - lowering produces an owned `Stmt` tree (TIR-like) with index expressions
    hash-consed in an `ExprArena`.
- **Reason:**
  - the trace is the tuning record and the search space at once;
  - an owned tree is the simplest structure for a loop nest;
  - hash-consing makes the simplifier and CSE on index arithmetic cheap.
- Index expressions and value expressions are separate types.

## D5. Data types in v0.1.0

- **Choice:** f32, f16, bf16, FP8 E4M3FN and E5M2, i8, u8, i32, i64 and
  bool.
  - ONNX's FNUZ FP8 variants and 4-bit types are rejected at import with a
    typed error.
  - Quantization enters as QDQ ONNX models: `QuantizeLinear`/`DequantizeLinear`
    nodes carry the scales.
  - QDQ chains fold into quantized compute ops.
  - Calibration, choosing the scales ourselves, is M7.
- **Reason:**
  - Ada's tensor cores support FP8 and int8, and TensorRT's quantized modes
    are where it is strongest, so the comparison needs them;
  - designing them in from the start costs little: dtype enum variants, mma
    table rows, interpreter evaluation rules;
  - QDQ input gives ptxc and TensorRT identical scales.

## D6. Our own float formats, no `half` crate

- **Options:** `half` for f16 and bf16 with our own FP8; our own for all.
- **Choice:** our own. One `FloatFormat` (exponent bits, mantissa bits,
  bias, overflow behaviour) and one encoder and decoder cover f16, bf16,
  E4M3FN and E5M2.
- **Reason:**
  - the four formats are one algorithm with four parameter sets;
  - one implementation means one set of rounding and saturation rules,
    tested once;
  - decoding is exact, and encoding is checked exhaustively over every f32
    input against the property "nearest representable value, ties to even".

## D7. ONNX protobuf types: prost, generated code committed

- **Options:** `prost-build` in `build.rs`; `prost-build` run by hand with
  the output committed; `rust-protobuf`; a hand-written decoder.
- **Choice:** `cargo xtask proto` runs `prost-build` on a pinned
  `onnx.proto` and writes `crates/onnx-proto/src/generated.rs`, which is
  committed. CI regenerates it and fails on a difference.
- **Reason:**
  - an ordinary build needs no protoc;
  - the generated code is reviewable;
  - lint exemptions for generated code are confined to one crate.

## D8. CUDA: our own driver API bindings, loaded with dlopen

- **Options:** `cudarc`; bindgen against `cuda.h`; hand-written bindings
  with dlopen.
- **Choice:** hand-written bindings for the roughly 40 driver functions ptxc
  uses, resolved from `libcuda.so.1` at runtime, behind a small RAII API.
- **Reason:**
  - small, readable surface;
  - exact control of streams, events and graph capture;
  - builds without CUDA installed, so CI and GPU-free tests need nothing;
  - consistent with building the stack from scratch.
- `cudarc` would have saved about a week.

## D9. Memory planning at compile time; `Program` is the IR-free boundary

- **Options:** a runtime allocator or caching allocator; offsets planned at
  compile time.
- **Choice:** the compiler assigns every intermediate buffer an offset in
  one arena (XLA's BufferAssignment). The runtime consumes a `Program` that
  contains no IR.
- **Reason:**
  - planning is a pure algorithm and is tested without a GPU;
  - fixed addresses are what CUDA graph replay needs;
  - a serialised `Program` is comparable with a TensorRT engine;
  - the runtime does not depend on any IR crate.

## D10. A CPU interpreter for lowered loop IR

- **Choice:** `ptxc-loop` interprets `LoweredKernel`s:
  - threads run in lockstep between barriers;
  - shared-memory races are detected;
  - `mma`/`ldmatrix` fragments are modelled from the PTX ISA.
- **Reason:** schedules, which are where GPU compilers break, become
  testable in `cargo test` without a GPU.

## D11. ptxas as a test-only checker

- **Choice:** the flake provides ptxas (`cudaPackages.cuda_nvcc`, the only
  unfree package allowed). Tests assemble every emitted module. At runtime
  the driver JIT-compiles PTX.
- **Reason:** an outside assembler catches malformed PTX early, with a
  precise message. The compiler path still depends only on the driver.
- ptxas (13.5) is newer than the driver (CUDA 13.2). ptxas accepting a
  module is therefore necessary but not sufficient: the `.version`
  directive ptxc emits is capped by `Target`'s `PtxIsaVersion`, which
  `ptxc-runtime` derives from the driver's reported version.
- nixpkgs needs every CUDA package nvcc pulls in to be allowed, so the
  flake allows exactly the packages under the CUDA EULA licence.

## D12. Shape inference in op type rules; constant folding by the interpreter

- **Choice:**
  - each op's `TypeRule` infers its output types, and `GraphBuilder`
    applies the rule as nodes are added;
  - constant folding evaluates constant subgraphs with `ptxc-interp` (as
    XLA does with HloEvaluator).
- **Reason:**
  - every pass that creates nodes gets inference, not just the importer;
  - folding cannot disagree with the golden model.

## D13. Differential fuzzing on stable, without cargo-fuzz

- **Choice:** a seeded, type-directed graph generator and a shrinker in
  `ptxc-testkit`, driven by `ptxc fuzz`.
- **Reason:**
  - the inputs are structured graphs, not byte strings;
  - cargo-fuzz needs nightly;
  - a fixed seed reproduces a case exactly.
- A cargo-fuzz target for the protobuf parser could be added later as a
  nightly-only extra.

## D14. Crate layout and enforced dependency rules

- **Choice:** the 21 crates and the dependency rules in ARCHITECTURE.md.
  `cargo xtask check-deps` enforces the rules:
  - the GPU-free core;
  - cuBLAS only in `bench`;
  - unsafe code only in `cuda`;
  - `interp` independent of `lower`.
- **Reason:** most tests run without a GPU, and the boundaries hold because
  a check fails when they are crossed.

## D15. Code structure conventions

- **Typestate** for objects whose phases the code drives:
  - `Compilation<S>`;
  - `Schedule<S>`;
  - `GraphCapture<S>`;
  - `Executor<S>`;
  - the launch builder.
- **Distinct types** when a phase changes the data's shape:
  - `Graph` to `Partitioned`;
  - `ComputeDef` to `LoweredKernel`.
- **Enums** for state known only at runtime.
- **Newtypes** for every quantity with a meaning: ids, sizes, offsets in
  each memory space, dimensions, axes, versions, tiles, quantization
  parameters, times.
- **Traits** only where the roadmap commits to at least two implementations
  (the table in ARCHITECTURE.md).
- **Files:** one concept each, with a soft cap of about 300 lines. Op families
  are mirrored file for file across `onnx/import`, `graph/ops`, `interp/eval`
  and `lower/ops`.
- **Functions:** clippy's `too_many_lines` at 60 as a hard ceiling, as in
  PARE. Most functions should be well under it.

## D16. Python only for baselines

- **Choice:**
  - `baselines/` is a uv project (nix-ld is set up on this machine);
  - it exports models, produces reference outputs, and runs PyTorch,
    torch.compile, ORT and TensorRT;
  - Rust never invokes Python; a missing reference fails with the command
    that produces it.
- **Reason:** keeps the compiler and its tests free of a Python runtime.

## D17. Repository tasks in Rust (`xtask`)

- **Choice:** `cargo xtask proto | fetch | check-deps`.
- **Reason:** the task scripts stay in the same language and lints as the
  code, and Python stays confined to baselines (D16).

## D18. No control flow in v0.1.0

- **Choice:** `If`, `Loop` and `Scan` are rejected at import. The
  conformance report counts them as unsupported (`ControlFlow`).
- **Reason:** none of the target models need them, and with static shapes
  they would need a different execution model.

## D19. Interpreter numerics

- **Choice:**
  - floats accumulate in f64 and integers in i64;
  - narrow float types are widened to f32 per op and rounded to nearest-even
    on output;
  - quantization follows the ONNX definitions exactly.
- **Reason:** the golden model should be at least as accurate as everything
  it judges, so tolerances measure the compiled code's error.

## D20. Error types with `thiserror`

- **Choice:** each crate has its own error enums, derived with `thiserror`.
  The CLI alone uses `anyhow` to add context at the top level.
- **Reason:** typed errors at every library boundary, as in rvsim and PARE.

## D21. Serialisation with `serde`

- **Choice:**
  - `serde` + `serde_json` for tuning records (JSON lines), reports and
    benchmark results;
  - a `Program` file is a JSON header followed by the raw constant blob.
- **Reason:** the header can be read and diffed by hand, and the weights are
  not inflated by text encoding.

## D22. CLI in the style of PARE

- **Choice:**
  - `clap` derive with one subcommand per command and a global
    `--report <path>` for a JSON report of the run;
  - styled output on stderr (`yansi`, with TTY detection) through `ui.rs`;
  - timed phases as an enum in `phase.rs`;
  - `env_logger` in `logger.rs`.
- **Reason:** the same look and behaviour as PARE. stdout stays clean for
  piping.

## D23. Repository and licence

- **Choice:** `github.com/willmccallion/ptxc`, dual-licensed MIT OR
  Apache-2.0, as rvsim and PARE are.

## D24. Our own `dlopen`/`dlsym` declarations

- **Options:** the `libloading` crate; our own `extern "C"` declarations of
  `dlopen`, `dlsym` and `dlerror`.
- **Choice:** our own, in `ptxc-cuda/src/sys/library.rs`, reused by the
  cuBLAS baseline in `ptxc-bench`.
- **Reason:** libcuda is loaded once and never unloaded, so `libloading`'s
  lifetime-bound symbols buy nothing. Our own is about 50 lines, has no
  dependency, and keeps every unsafe line in one auditable file.

## D25. ImageNet validation set for ResNet-50 accuracy

- **Choice:** the official ILSVRC2012 validation set, downloaded by hand
  after registering (its terms forbid redistribution) into
  `data/datasets/imagenet-val/`. The repository commits only the list of
  5,000 file names used.
- **Reason:** it is the number people recognise. The rate at which ptxc and
  PyTorch pick the same class is reported alongside, and needs no labels.

## D26. Clock locking by hand for now

- **Choice:** published series are run after the clock commands in
  BENCHMARKING.md, typed by hand with sudo. A sudoers rule may replace this
  later. The harness records the clocks either way.

## D27. CI: GitHub-hosted, GPU-free

- **Choice:** `.github/workflows/ci.yml` runs fmt, clippy, `cargo test`
  and `cargo doc` on GitHub's runners, as PARE's `check` job does.
  `check-deps` and the protobuf diff join it when those tasks exist (M1,
  M2).
- GPU suites run locally. A local pre-merge GPU check, or a self-hosted
  runner, can be added if GPU-only bugs start getting through.

## D28. ONNX pin and minimum opset

- **Options:** for the pin, the latest ONNX release or an older one that
  more exporters target; for the minimum opset, 7 (the oldest ORT still
  runs), 13, or 17.
- **Choice:**
  - pin the latest ONNX release at the time the `onnx-proto` task (M1
    task 3) starts. `onnx.proto` and the node tests come from the same
    tag, recorded in `xtask` with the archive's SHA-256;
  - accept models whose default-domain opset is 13 or later. Older models
    are rejected at import with a typed error, and the conformance report
    counts them as unsupported (`Opset`).
- **Reason:**
  - opset 13 moved `axes` and `split` from attributes to inputs (`Squeeze`,
    `Unsqueeze`, `ReduceSum`, `Split`), added per-axis
    `QuantizeLinear`/`DequantizeLinear`, and added bf16 to most ops.
    Starting there means one canonical form per op, with fewer per-version
    import paths;
  - current PyTorch exporters and ONNX Runtime's quantizer emit opset 13 or
    later, so none of the target models is lost.

## D29. TVM/Ansor comparison deferred to M5

- **Choice:** whether to build TVM for the tuning-time comparison is
  decided at M5, when the autotuner exists to compare.
- **Reason:** the comparison only means something against a working
  tuner, and building TVM costs time that M1 to M4 don't need to spend.

## D30. onnx.proto committed and parsed by protox

- **Refines:** D7, which stays otherwise: prost-build generates the code,
  and `generated.rs` is committed and checked by CI.
- **Options:** for the source, download `onnx.proto` at the pinned tag or
  commit a copy; for the parser, protoc or protox, a protobuf compiler
  written in Rust.
- **Choice:**
  - `crates/onnx-proto/proto/onnx.proto` is ONNX v1.23.2's file, committed
    unchanged. An ONNX upgrade replaces it and the tag in `xtask`;
  - `cargo xtask proto` parses it with protox and passes the descriptors to
    prost-build. protoc is no longer in the flake.
- **Reason:**
  - regenerating needs no network, and an upgrade shows up as a diff of the
    `.proto` itself;
  - protox and prost-build are both locked in `Cargo.lock`, so local and CI
    regeneration produce the same bytes, and CI installs nothing.

## D31. ONNX pinned at v1.22.0, the last release with node-test files

- **Supersedes:** D28's "the latest ONNX release", and the version in D30.
  The minimum opset of 13 stands.
- **Context:** ONNX v1.23.0 deleted the node-test files from its repository
  and its packages (ONNX PR #7959). They are now generated in Python at test
  time, so v1.23 has no archive of fixed test data to pin and checksum.
- **Options:**
  - pin v1.22.0 for `onnx.proto` and the node tests;
  - pin `onnx.proto` at v1.23.2 and the node tests at v1.22.0;
  - keep v1.23.2 and generate the node tests with `baselines/`.
- **Choice:** v1.22.0 for both, from ONNX's source distribution on PyPI.
  `cargo xtask fetch onnx-tests` downloads it, checks its SHA-256 before
  unpacking anything, and unpacks the 1,765 node tests into
  `data/onnx/v1.22.0/node/`.
- **Reason:**
  - the node tests stay fixed data from outside, checked by hash, rather
    than files we generate. ONNX seeds its generators, but the expected
    outputs come from numpy, so regenerated files need not be bit-identical
    across machines;
  - one tag for the proto and the tests, as D28 intended;
  - PyPI never replaces a published file and publishes its SHA-256.
    GitHub's tag archives are generated on demand and changed once (2023);
  - v1.22.0 covers default-domain opsets up to 27. v1.23 adds opset 28, so
    only operator versions introduced in opset 28 lack node tests here.

## D32. Canonical forms in the graph IR

- **Context:** ONNX often has several ways to say one thing: optional
  operands, defaulted attributes, tensors that are constants in practice,
  and ops that differ only in how they compute a shape. Every pass, the
  interpreter and lowering would otherwise handle each variant.
- **Choice:** the importer (M1 task 8) resolves each op to one form, and
  type rules accept only that form:
  - shape-like operands become attributes, since static shapes (D3) make
    them constants: `Reshape` and `Expand` targets, `Slice` bounds, `Pad`
    widths, `Tile` repeats, `Split` sizes, `Trilu`'s `k`, `Resize` scales
    and sizes, and quantization scales and zero points;
  - `Squeeze`, `Unsqueeze` and `Flatten` become `Reshape`, as XLA's HLO has
    only reshape;
  - derived values are resolved once:
    - `auto_pad` into explicit pads;
    - reduction axes into an explicit list;
    - `Slice` into a start, step and length per axis;
    - `Split`'s `num_outputs` into sizes;
    - negative `Pad` widths into a `Slice`;
  - optional operands get their defaults: `Clip` always has both bounds, and
    constant `Pad` always has its value;
  - attributes that only matter in one case are normalised elsewhere, so
    equal ops compare equal (`Cast` and `Quantize` keep `saturate` for FP8
    targets only);
  - `Attention` follows ONNX's `Attention` op, which the M6 matcher also
    produces, rather than a separate ptxc op.
- **Reason:** one form per op means the interpreter, lowering and every
  rewrite rule handle one case, and structural equality (CSE) sees equal
  ops as equal.

## D33. The node tests v0.1.0 leaves unsupported

- **Context:** M1's exit criteria ask for the unsupported node tests to be
  reviewed. `ptxc conformance --backend interp` on the 1765 ONNX v1.22.0
  node tests gives 979 passed, 0 failed and 786 unsupported.
- **Choice:** these stay unsupported for v0.1.0:
  - 345 tests of ops outside the v0.1.0 set: training losses
    (`NegativeLogLikelihoodLoss`, `SoftmaxCrossEntropyLoss`) and optimisers,
    recurrent ops (`RNN`, `LSTM`, `GRU`), signal processing (`DFT`, `STFT`,
    window functions), image ops (`GridSample`, `RoiAlign`,
    `ImageDecoder`, `Col2Im`, `DeformConv`), scatter and `GatherND`,
    `TopK`, `ConvTranspose`, `LpPool`, the newest attention variants
    (`LinearAttention`, `FlexAttention`, `RotaryEmbedding`), random ops,
    bitwise ops, the less common unary ops (`Tan`, `Asin`, `Cosh`, …) and
    the `ai.onnx.ml` domain;
  - 289 tests of element types outside `DType`: f64, strings, the 16-bit,
    unsigned 32- and 64-bit and sub-byte integers, FP8 FNUZ, FP4 and E8M0,
    sequence and optional types, and `Range` on f16 and bf16;
  - 107 tests at opsets below 13 (D28);
  - 24 tests of control flow (`If`, `Loop`, `Scan`, D18);
  - 15 tests of attribute values ptxc does not support: training-mode
    `Dropout` and `BatchNormalization`, `Dropout`'s mask and `MaxPool`'s
    indices outputs, and blocked quantization;
  - 6 tests whose shapes are computed through float ops from inputs, which
    the importer cannot resolve even with the inputs bound.
- **Reason:** v0.1.0's target models (MLP, ResNet-50, BERT-base, GPT-2)
  are not expected to need any of these; M1 task 12 exports them, and the
  exit criteria check that they import. Each category maps to one importer
  error, so a model that needs one fails at import with its reason rather
  than running wrongly.

## D34. `GatherND` and `CumSum` join the v0.1.0 op set

- **Context:** BERT-base and GPT-2, exported from transformers 5 with
  `torch.onnx.export(dynamo=True)`, build their attention masks with
  `GatherND` (one in BERT, four in GPT-2) and `CumSum` (one in GPT-2).
  D33 left `GatherND` unsupported on the expectation that the target models
  would not need it, so both models failed to import.
- **Options:** support both ops; export with fixed shapes and fold the mask
  construction away, which works for GPT-2 but not BERT, whose mask reads
  the `attention_mask` input; drop BERT and GPT-2 from the v0.1.0 checks.
- **Choice:** support both, as `Op::GatherNd(batch_dims)` in the gather
  family and `Op::CumSum` in the reduce family. `CumSum`'s axis input must
  be known at import, as other shape-like operands are (D32).
- **Reason:** the models should be checked as users export them. Every
  `GatherND` node test and both `CumSum` tests on supported element types
  pass; the other seven `CumSum` tests are f64, which stays unsupported.
  Conformance on the v1.22.0 node tests gives 984 passed, 0 failed and 781
  unsupported.
- **Supersedes:** `GatherND` in D33's list of ops outside the v0.1.0 set.

## D35. ImageNetV2 stands in for ImageNet until it can be downloaded

- **Context:** M1's ResNet-50 check compares the interpreter with ONNX
  Runtime on 16 preprocessed images, and D25 takes them from the ImageNet
  validation set. image-net.org's downloads were unreachable when the
  baselines were built.
- **Options:** wait for ImageNet; Kaggle's copy of ImageNet, which needs an
  account; Imagenette, ten ImageNet classes; ImageNetV2's matched-frequency
  set, 10,000 images of all 1,000 classes under the MIT licence.
- **Choice:** ImageNetV2, fetched by `cargo xtask fetch imagenetv2` from a
  pinned Hugging Face revision with its SHA-256. `ptxc-baselines images
  --source imagenetv2` takes evenly spaced images, ordered by class and then
  name; `--source imagenet` keeps D25's list for when ImageNet is in place.
- **Reason:** the interpreter-against-ORT check needs only realistic inputs,
  not labels, and ImageNetV2 comes from the same distribution ResNet-50 was
  trained on. M4's top-1 accuracy still uses ImageNet itself, as D25 says.

## D36. Targets declare their SM's minimum PTX ISA version

- **Context:** D11 caps the `.version` ptxc emits by `Target`'s
  `PtxIsaVersion`, derived from the driver. The GPU-free crates and their
  tests need a version before any driver is involved, so the presets in
  `ptxc-target` carry one.
- **Options:** the version that introduced the SM (7.8 for `sm_89`, CUDA
  11.8); the newest version the pinned ptxas accepts, lowered at runtime to
  what the driver supports.
- **Choice:** the SM's minimum. `Target::from_device` checks that the
  driver supports at least the preset's version and fails otherwise; it
  does not raise it.
- **Reason:** what M2 and M3 emit (`mma.sync`, `ldmatrix`, `cp.async`)
  exists at 7.8, so modules load on any driver that knows `sm_89`, and the
  emitted text does not depend on which driver and ptxas produced it. A
  feature that needs a newer version is a new decision.
- **Evidence:** ptxas 13.5 rejects `.version 7.7` with `.target sm_89` ("PTX
  .version 7.7 does not support .target sm_89") and accepts 7.8. The
  preset's limits are what the RTX 4060 Ti's driver reports through
  `cuDeviceGetAttribute`, and agree with the CUDA programming guide's
  tables for compute capability 8.9; registers per thread (255) is from the
  guide, as no attribute gives it.
- **Supersedes:** D11's "which `ptxc-runtime` derives from the driver's
  reported version".

## D37. CUDA handles: shared contexts, byte buffers, no device-visible host memory

- **Context:** `ptxc-cuda` is the only crate allowed unsafe code, so
  everything above it, including kernel launches, must be a safe API.
- **Options and choices:**
  - *Ownership:* handles borrowing the context with lifetimes, or holding
    an `Arc` of it. **`Arc`:** `ptxc-runtime` can then own modules,
    buffers and streams together, without self-referential lifetimes. The
    function table is a process-wide `&'static`, as libcuda is never
    unloaded (D24).
  - *Buffers:* typed `DeviceBuffer<T>` or bytes. **Bytes:** the memory plan
    is a byte arena at offsets, and `HostTensor` already knows its element
    type.
  - *Launch soundness:* Rust cannot check what a kernel does. **No host
    memory is mapped into the device's address space** (no pinned, mapped
    or managed memory), so a wrong kernel corrupts only device memory and
    fails with a CUDA error; the launch builder also checks the arguments
    against the kernel's parameter sizes (`cuFuncGetParamInfo`). Pinned
    staging buffers for CUDA graphs (M4) need a new decision.
  - *Current context:* an explicit guard, or setting it on every call.
    **Every call** makes its context current (`cuCtxSetCurrent`), so
    handles work from any thread, as `cargo test`'s threads need.
- **Consequence:** copies are synchronous and run on the legacy default
  stream, so ptxc's streams are created blocking (`CU_STREAM_DEFAULT`) to
  keep copies ordered after the kernels queued before them.

## D38. A range-aware index simplifier, property-tested with proptest

- **Context:** split and fuse produce index arithmetic such as
  `floordiv(io*4 + ii, 4)`, which is `io` only because `0 <= ii < 4`.
- **Options:** algebraic simplification only (folding, identities, like
  terms), or also TVM's floor division and modulo rules, which need each
  variable's range; for the property tests, `proptest` or a seeded
  generator of our own, as D13 has for graphs.
- **Choice:** range-aware simplification, with ranges passed in as
  `VarRanges`; random expressions from `proptest`, a dev-dependency of
  `ptxc-loop` only.
- **Reason:** without the range rules every split and fuse leaves a
  division and a modulo in the index, which M3's kernels pay for in every
  thread. proptest generates expression trees from a strategy and shrinks a
  failure to a minimal expression, which a hand-written generator would
  have to reimplement; D13's generator stays for graphs, whose shrinking is
  domain-specific.
- **Properties:** for every expression and every assignment within the
  ranges under which it evaluates, the simplified expression evaluates to
  the same value; and simplifying twice changes nothing.

## D39. Schedule semantics: TVM's primitives, guards for uneven splits

- **Context:** M2's schedules need split, reorder, fuse, bind, unroll and
  vectorize, and M3's tiled GEMM must be expressible with them.
- **Choice:**
  - a split rounds its outer loop up when its factors do not divide the
    extent, and records a guard `index < extent`, as TVM's `split` does;
    lowering emits it as an `If`;
  - reorder may put reduction loops between spatial loops, as a tiled GEMM
    does with `k` between its block and thread tiles; lowering then keeps
    an accumulator per inner spatial point;
  - a reduction loop may be bound to a thread axis, whose threads combine
    their partial results, but not to a block axis;
  - each axis's index is an expression over the loop variables, simplified
    after every primitive; a rejected primitive changes nothing, so a trace
    holds only steps that applied.
- **Reason:** these are TVM's semantics, which the M3 rules are written
  against. The property tests check that any sequence of splits, fuses and
  reorders visits every point of the iteration domain exactly once.

## D40. A compute definition is one stage with at most one reduction

- **Context:** a schedule is one loop nest. It says how one reduction is
  tiled, reordered and bound to threads, but not how a second reduction
  nested in the first would be. Softmax, layer norm and log-sum-exp each
  need several reductions in sequence.
- **Options:** any number of reductions per definition, each computed
  where it is used with its own serial loops; or TVM's model, one reduction
  per definition (a stage), with multi-reduction ops as chains of stages.
- **Choice:** one reduction per definition, and then exactly one output;
  `ComputeBuilder::finish` rejects more. Lowering an op family gives a list
  of stages joined by intermediate buffers. In M2 each stage is a kernel;
  M3's `compute_at` places stages in one kernel.
- **Also:** every load whose index does not come from a gather must stay
  within its tensor over the iteration domain, which `finish` proves with
  the index simplifier's interval bounds. Schedules visit exactly the
  domain (D39), so lowered loads are in bounds by construction, and ops
  that read outside a tensor (padding) must clamp the index rather than
  rely on a select, whose branches are both evaluated.
- **Reason:** every reduction stays schedulable, so M3's fast reduction and
  GEMM schedules apply to each stage, and lowering handles one reduction
  shape well instead of many shapes slowly. It is TVM's model, which the
  M3 and M5 rules follow.

## D41. The loop interpreter computes as a GPU does

- **Context:** `ptxc-interp` is the reference: it widens floats and
  accumulates in f64 (D19). The loop interpreter checks schedules and,
  from M2 on, that it agrees with the GPU on every conformance kernel.
- **Options:** reuse the reference's numerics; or round every operation to
  its element type, as the GPU does.
- **Choice:** round every operation: floats to their type (f32 results are
  exact through f64, whose precision suffices), integers wrapped to their
  width, casts from floats to integers truncated and saturated as PTX's
  `cvt.rzi.sat` does. Threads run in lockstep with masks; memory starts
  unset; races are errors.
- **Reason:** a schedule's reassociation (cross-thread trees, split
  reductions) changes f32 results exactly as it will on the device, so the
  interpreter predicts the GPU's output rather than the reference's, and
  the GPU agreement check in M2's exit criteria can be exact for integer and
  arg-reduction kernels. Agreement with the reference stays a tolerance
  check, as it is for the GPU.

## D42. Lowering binds stages to graph values and computes narrow floats in f32

- **Context:** a `ComputeDef` names its tensors for printing only. The
  compiler must know which graph value or intermediate buffer each tensor
  is, and arithmetic on f16, bf16 and FP8 values needs a rounding rule.
- **Options:** encode value ids in tensor names, or return a `Stage` that
  binds each tensor to a value; compute in each value's own type, or widen
  narrow floats to f32 and round back once.
- **Choice:** lowering returns `IndexVec<GroupId, Vec<Stage>>`, each stage a
  `ComputeDef` with a `ValueId` per input tensor and per output (this
  supersedes the `Vec<ComputeDef>` of D40). Float arithmetic on types
  narrower than f32 widens to f32 and rounds once at the end;
  comparisons, selects and integer arithmetic stay in their own types.
  `Pow` of an integer base and a float exponent runs in f32 (a `NOTE` in
  `elementwise.rs`): exact while the base and result stay below 2^24,
  where the reference computes in f64.
- **Reason:** bindings are typed, so a stage cannot be wired to the wrong
  buffer through a string. Widening matches `ptxc-interp` (D19) and what
  GPU kernels do outside tensor cores, costs nothing in memory-bound
  elementwise kernels, and avoids rounding every intermediate of a
  composed op (a mean, a GELU) to f16. Loop IR has no f64, and f64
  throughput on sm_89 is 1/64 of f32, which one rare `Pow` form does not
  justify.

## D43. Reductions accumulate in f32, and groups own their intermediates

- **Context:** `ptxc-interp` accumulates float reductions in f64 (D19).
  `ReduceLogSumExp` needs a maximum before its sum, which D40 makes two
  stages joined by a buffer that is no graph value. Loop IR has no scan for
  `CumSum`.
- **Options:** for accumulation, f64 or f32; for the joining buffer, a graph
  value added by lowering, or a buffer the group owns; for `CumSum`, a new
  scan construct, or a reduction masked to the elements the scan has
  reached.
- **Choice:**
  - Float reductions accumulate in f32 and round once to the output type.
    Integer `Sum`, `Prod`, `SumSquare` and `L1` wrap in their own type, and
    integer `Mean` divides an exact `i64` sum. Integer `L2`, `LogSum` and
    `LogSumExp` run in f32 (a `NOTE` in `reduce.rs`).
  - Lowering returns `IndexVec<GroupId, LoweredGroup>`. A group lists its
    intermediates' types, and a stage's `Binding` is a graph value or one
    of its group's intermediates (this supersedes D42's `Vec<Stage>`).
  - `CumSum` is a sum along the axis masked by comparing positions, read as
    values through the new `TensorExpr::Index`. It is O(n²) along the axis.
  - An arg-reduction over an empty axis gives 0, as the reference does, and
    an empty loop cannot be bound to a thread or block axis, which would
    launch nothing.
- **Reason:**
  - f32 accumulation is what ONNX Runtime and every GPU library do, and f64
    on sm_89 runs at 1/64 of f32's rate. The difference is rounding, which
    testkit's per-length reduction tolerance covers, and intermediate
    overflow past f32's range (an `L2` or `Prod` of values near 1e30),
    which ORT shares.
  - An intermediate is not a graph value: graph passes never see it, and
    the memory plan can place it in the group's scratch.
  - The masked sum needs nothing new in schedules or codegen. The
    sequences ptxc meets are short, such as positions along a sequence
    axis. A scan schedule comes when a model needs long scans.

## D44. Softmax and normalisation are chains of f32 statistics stages

- **Context:** softmax and the normalisations each need two or three
  reductions in sequence. D40 makes each reduction its own stage, and D43
  keeps statistics in f32.
- **Options:** one-pass statistics (Welford's variance, the online
  softmax), or two-pass statistics, a stage per statistic; an f32 mean, or
  a compensated one.
- **Choice:**
  - `Softmax` and `LogSoftmax` are three stages: the row maximum, the sum of
    `e^(x - max)`, then the elementwise result.
  - Layer and instance normalisation take the mean, then the inverse
    deviation `1 / sqrt(variance + epsilon)` from squared deviations about
    that mean, then the normalised tensor. RMS normalisation skips the mean.
  - A statistics stage writes `LayerNormalization`'s mean and inverse
    standard deviation outputs in their stash type, only when they are read.
  - Inference batch normalisation is one elementwise stage.
- **Reason:**
  - Each stage is a plain reduction or elementwise kernel, which M2's
    default schedules handle. Two passes are the reference's algorithm and
    the most accurate uncompensated one.
  - The online softmax and Welford fusion arrive with M3's `compute_at` and
    M6's fused attention, where they save memory passes.
  - An f32 mean rounds by half an ulp of the inputs, and normalising
    multiplies that by up to `1 / sqrt(epsilon)` when a block barely varies.
    That is f32's conditioning, shared by ONNX Runtime and PyTorch.
    Compensating would cost another pass of every block for inputs that
    are ill-conditioned anyway. The differential tests draw normalisation
    inputs from whole numbers, which keeps them well-conditioned.

## D45. A gather wraps its indices into the axis

- **Context:** ONNX makes a gather index outside `[-extent, extent)` an
  error. A GPU kernel cannot return one, and an out-of-bounds read can
  fault, which leaves the CUDA context unusable for every later call.
- **Options:** read at the raw index (undefined behaviour on bad input);
  clamp the index into the axis; or wrap it, as `floormod(index, extent)`.
- **Choice:** wrap. For every valid index, wrapping is exactly ONNX's
  resolution of negative indices. An invalid one reads some element of the
  axis instead of faulting. A gather over an empty axis reads nothing.
- **Reason:** one remainder both resolves negative indices and keeps every
  read in bounds, so a bad input can never take down the process's GPU
  context. Clamping would still need a select to resolve negative indices.
  Invalid indices give wrong values rather than an error. A later runtime
  check could report them with a device-side flag, without changing the
  kernels' reads.

## D46. Lowered groups own constant tables, and quantized products rescale in f32

- **Context:** quantization parameters are op attributes (D19), not graph
  tensors. A per-axis map has one scale and zero point per channel, and a
  stage can only read tensors bound to it. `QLinearMatMul` and
  `QLinearConv` rescale an exact integer sum by
  `a_scale * b_scale / y_scale`.
- **Options:** for per-axis maps, a chain of selects on the channel index,
  growing with the channel count, or small tables the group owns; for the
  rescale, the reference's f64 or f32.
- **Choice:**
  - A `LoweredGroup` lists constant tables built from attributes, and
    `Binding::Table` binds one to a stage input. The runtime uploads tables
    once with the program. One map stays an inlined constant.
  - Quantized products sum exactly in `i64` and rescale in f32. FP8
    operands of a quantized product are an `UnsupportedDType` error, as the
    reference does not take them either.
- **Reason:**
  - A table read costs one load per element whatever the channel count;
    select chains cost a comparison per channel. Tables also keep the graph
    free of tensors no model has.
  - f32 epilogues are what int8 GPU kernels use. A rounding tie can land
    one step apart once a sum passes 2^24, which testkit's `QUANTIZED`
    tolerance of one step already allows for.

## D47. Resize reads selected elements directly and interpolates from tables

- **Context:** ONNX's reference resize multiplies every tap by its weight,
  zero weights included, so a NaN or an infinity beside a selected element
  spreads to it. Following it literally reads every tap of every axis: the
  batch and channel axes of an image tensor, which resizing leaves
  unchanged, would multiply a 2-D linear resize's 4 taps to 16.
- **Options:** follow the reference tap for tap; or read an axis directly
  where every output position selects one element, as ONNX Runtime and
  PyTorch do.
- **Choice:** lowering computes each axis's taps once, in f64, from ONNX's
  definitions. An axis whose every position selects one element with
  weight 1 is a direct read: the identity, or an index table. Any other axis
  interpolates over its taps, read from index and weight tables (D46).
  `tf_crop_and_resize` positions outside the region read a flag table and
  give the extrapolation value. Integer results interpolate in f32 and
  truncate (a `NOTE`).
- **Reason:** direct reads cost one load and no arithmetic, where the
  reference's form multiplies the work by two per unchanged axis. The
  results differ only where a zero-weight neighbour is NaN or infinite,
  which no model relies on; the differential test uses finite inputs and
  says why.

## D48. Attention lowers unfused in M2

- **Context:** ONNX `Attention` is a softmax between two matrix products,
  with masks, causality, a key-value cache, softcapping and an optional
  intermediate output. M6 fuses it as FlashAttention-2 does; M2 needs it
  correct on every option first.
- **Options:** a fused kernel now, or a chain of plain stages.
- **Choice:** four stages over f32 intermediates:
  - the scores, `softcap(scale * Q K^T)` plus the bias;
  - each row's maximum;
  - each row's sum of `exp(score - max)`;
  - `Y = sum of exp(score - max) / sum * V`, laid out as `Y` is.

  The bias is negative infinity where the mask, causality or a non-pad
  length hides a key, the additive mask's value, or nothing. Keys and
  values after a past are read through a select between the past and
  current operands, each clamped (D40). Grouped heads read key head
  `head / (q_heads / kv_heads)`. Present keys and values and the QK
  intermediate are stages of their own, built only when read; its matmul
  and softcapped forms recompute the scores.
- **Reason:**
  - Each stage is a plain reduction or elementwise kernel that M2's
    default schedules and codegen handle, and two-pass softmax statistics
    are the reference's algorithm.
  - The scores take memory quadratic in the sequence, as an unfused
    attention does everywhere. M6's fused kernel removes it, and with it
    the recomputation for the QK output.

## D49. M2's default schedules are two fixed rules

- **Context:** every stage needs a schedule before codegen. The tuned
  schedules come later: tensor-core GEMM and convolution in M3, search over
  sketches in M5.
- **Options:** the `ScheduleRule` trait now, with sketches whose knobs are
  fixed, or plain functions until there are knobs to search.
- **Choice:** `default_schedule` picks one of two plain functions.
  - An injective stage fuses its loops and splits them into blocks of 256
    threads, or fewer for a smaller output, so consecutive threads write
    consecutive elements.
  - A reduction binds its fused output rows to blocks. It fuses its
    reduction loops and splits them so `threadIdx.x` takes consecutive
    elements, with a shared-memory tree combining the threads.
  - An empty output stays serial, as does an empty row's reduction: an
    empty loop cannot be bound.
- **Reason:** a trait whose only implementations have no knobs would fix
  an interface before the search that shapes it. These two rules make every
  stage a valid GPU kernel. The differential tests run every lowered op
  family under them in the loop interpreter, races and barriers checked.

## D50. sm_89 targets PTX ISA 8.4, and FP8 converts in hardware

- **Context:** codegen converts every FP8 load, store, cast and constant.
  sm_89's `cvt.rn.satfinite.e4m3x2.f32` and `cvt.rn.f16x2.e4m3x2` need PTX
  ISA 8.1. Its FP8 `mma.sync`, which M3 uses, needs 8.4. D36 fixed the
  preset at 7.8.
- **Options:**
  - keep 7.8 and convert FP8 in software, about twenty integer
    instructions per element;
  - declare each module's version from the features it uses, so a target
    carries a minimum and a maximum;
  - raise the preset to 8.4.
- **Choice:** the sm_89 preset declares 8.4.
  - FP8 converts with one `cvt` each way.
  - A cast or store that overflows FP8 without saturating fixes up the
    saturated result. E4M3FN overflows to NaN above 464, and E5M2 to
    infinity from 61440: the ties where rounding to nearest even leaves
    the largest finite value.
  - An integer cast to a float narrower than f32 goes through an f32
    rounded to odd. sm_89 has no direct `cvt` from an integer to bf16
    (it needs sm_90), and rounding twice through f32 differs from
    rounding once for some integers above 2^24.
- **Reason:** FP8 is sm_89's headline datatype, and software conversion
  would tax every FP8 tensor to support drivers older than CUDA 12.4
  (2024). Per-module versions add a second version to every target to
  serve those drivers. The emitted text still does not depend on which
  driver or ptxas produced it.
- **Evidence:** ptxas 13.5, `--gpu-name sm_89`:
  - FP8 `cvt` at 8.0: "requires .target sm_90 or higher"; accepted at 8.1.
  - FP8 `mma.sync.m16n8k32` at 8.3: "requires PTX ISA .version 8.4 or
    later"; accepted at 8.4.
  - `cvt.rn.bf16.s32`: "requires .target sm_90 or higher".
- **Supersedes:** D36's choice of the SM's minimum version. Its check that
  the driver supports the preset's version stands.

## D51. Transcendental functions are math programs both executors run

- **Context:** M2's exit criteria ask the loop interpreter to agree with the
  GPU on every conformance kernel.
  - The interpreter computed exp, log and the other transcendental
    functions in f64 through libm.
  - PTX has no correctly rounded transcendental instructions. Its `.approx`
    instructions are specified only by error bounds, which the
    interpreter cannot reproduce.
- **Options:**
  - libm in the interpreter, approximate instructions on the GPU, and a
    tolerance between them;
  - a GPU implementation in codegen only, with the interpreter keeping
    libm;
  - one program per function that both run.
- **Choice:** a function is a `MathProgram` in `ptxc_loop::math`.
  - It is straight-line code over f32, 32-bit integer and predicate slots.
  - Every step is exact IEEE arithmetic (add, sub, mul, div and fma
    rounding to nearest even with subnormals kept, rounding to an
    integer, conversions) or integer arithmetic, which PTX without `.ftz`
    computes the same way.
  - The loop interpreter evaluates programs, and codegen emits each step
    as one instruction, so the two agree bit for bit, NaN payloads aside.
  - exp uses Cody and Waite's reduction, a degree-7 Taylor polynomial and
    a scale in two steps, so a subnormal result rounds once: within 1.06
    ulps of the exact value over every f32.
  - log uses fdlibm's formulation with five terms of the atanh series:
    within 0.84 ulps.
  - Coefficients are Taylor coefficients computed in code, not fitted
    constants.
- **Reason:**
  - Exact agreement makes the loop interpreter an oracle for codegen: a
    mismatch is a codegen bug, never accuracy noise.
  - Accuracy is checked on the host, over every f32.
  - Programs are data, so a better algorithm replaces one without
    touching either executor.
  - The graph interpreter keeps f64 libm as the reference both are
    compared with.
- **Evidence:** `cargo test --release -p ptxc-loop --test math -- --ignored`
  checks every f32 for both functions in about ten minutes on 28 threads.

## D52. Test generators shared as a library keep their panics

- **Context:** the op-family generators and the stage-by-stage runner lived
  in testkit's lowering tests. The GPU tests in `ptxc-runtime` need the
  same cases, and the fuzz commands will too, so they moved into the
  testkit library, where the workspace denies `unwrap` and `panic`.
- **Options:**
  - return errors from every generator;
  - keep the panics in the generators, under an `expect` that names the
    reason.
- **Choice:**
  - `ptxc_testkit::families` keeps its panics. A generator that builds an
    invalid shape or graph has a bug, and a proptest strategy has no way to
    report an error.
  - The fuzzer's graph generator, `ptxc_testkit::generate`, does the same
    for its own invariants, such as drawing from a non-empty table.
  - `ptxc_testkit::lowered::run_lowered`, which callers drive with their
    own executors, returns typed errors.
- **Reason:** the panics guard invariants the generators establish
  themselves, so threading `Result` through roughly forty strategies would
  add noise without catching anything. In test code a panic is already a
  failure with its message.

## D53. The fuzzer splices op-family cases into its graphs

- **Context:** the first graph generator drew only continuous ops, with
  one element type per graph, so that its graphs could be compared with
  the reference within tolerance. That left most of the v0.1.0 op set, and
  every change of element type inside a graph, out of the fuzzer.
- **Options:**
  - a second catalogue of every op in the generator, deriving attributes
    from the operand types;
  - splicing cases drawn from the op-family generators, which already
    produce valid attributes and inputs that reach each op's edge cases.
- **Choice:** splice family cases, drawn through proptest with a ChaCha
  stream seeded from the fuzz seed. An operand reads an earlier value of
  the same type half the time one exists. Continuous connectors join the
  values up, and the reference comparison applies only to graphs of
  continuous ops on moderate inputs; the GPU is always compared with the
  loop interpreter exactly.
- **Reason:** one description of each op's valid cases, kept by the
  families, serves the op tests, the GPU tests and the fuzzer. Exact
  comparison needs no tolerance, so the discontinuous ops are fuzzed as
  thoroughly as the rest.
- A recipe holds its input tensors rather than a seed for them, so a
  shrunk case is self-contained. A seed reproduces its recipe only while
  proptest's sampling stays the same: an upgrade may change which case a
  seed draws, but never a case written to `data/fuzz/`.

## D54. Fuzz cases are checked against ONNX Runtime through an ONNX export

- **Context:** M2 asks for 1,000 fuzz cases checked against cached ONNX
  Runtime outputs. Rust never runs Python, so the cases have to reach ONNX
  Runtime as files, and ptxc had no way to write a graph as ONNX.
- **Options:**
  - export every op in the v0.1.0 set;
  - export only the ops of the cases ONNX Runtime can usefully check.
- **Choice:** `ptxc_onnx::model_to_bytes` writes a graph at opset 21 for
  the ops the fuzzer's connectors draw, and names any other op in an
  error.
  - `ptxc fuzz-export` writes cases the reference can check;
    `ptxc-baselines fuzz-reference` caches ONNX Runtime's outputs;
    `ptxc fuzz-ort` compares them, reading each model back through ptxc's
    own importer.
  - Cases with bf16 values are not exported: ONNX Runtime's CPU provider
    has no bf16 kernels for these ops, and opset 21 does not define
    `Softsign` on bf16.
  - Cases that take the mean of no elements are not exported. ONNX leaves
    that result undefined. ptxc computes NaN, as numpy does; ONNX Runtime
    computes 0.
- **Reason:** another implementation can only be compared within
  tolerance, so like the reference it checks graphs of continuous ops on
  moderate inputs, and those use only the connectors' ops. The exact GPU
  comparison already covers the rest of the op set. Writing every op
  would add about forty mappings that nothing reads.

## D55. The cuBLAS baseline's bindings may use unsafe code

- **Context:** D24 has `ptxc-bench` load cuBLAS with ptxc-cuda's `dlopen`
  wrapper, but calling the functions it loads is unsafe, and the rules
  allowed unsafe code only in `ptxc-cuda` while letting only `ptxc-bench`
  mention cuBLAS.
- **Options:**
  - allow unsafe code in one `ptxc-bench` file;
  - put the cuBLAS bindings in `ptxc-cuda` behind a feature only
    `ptxc-bench` enables.
- **Choice:** `crates/bench/src/cublas/sys/`, one file for cuBLAS and one
  for cuBLASLt, may use unsafe code. It holds the function tables and the
  calls, each with a `// SAFETY:` line. Everything else in
  `ptxc-bench` stays safe. `ptxc-cuda` exposes its `dlopen` wrapper, a
  buffer's device address and a stream's handle for it.
- **Reason:** cuBLAS stays wholly out of the CUDA crate the runtime builds
  on, and `ptxc-cuda` stays the only unsafe crate on the compiler and
  runtime path. `cargo xtask check-deps` names the one file, so no other
  unsafe code can join it unnoticed.

## D56. Matrix products are tiled on CUDA cores by a third fixed rule

- **Context:** D49's reduction rule gives a matrix product one block per
  output row, its threads splitting the sum, so every block rereads both
  operands from global memory. A 256³ f32 GEMM takes about 1 ms against
  cuBLAS's 7 µs. Tensor cores come later in M3, the search in M5.
- **Options:**
  - recognise MatMul and Gemm stages by the op that made them;
  - recognise a matrix product from the compute definition alone, as
    Ansor's multi-level tiling recognises the data reuse that pays for
    tiling.
- **Choice:** `default_schedule` recognises the structure.
  - A product is a sum over one axis of two loads multiplied, each reading
    an output axis the other does not. The innermost such axis is the row
    or column and any other a broadcast batch. Every index part is a plain
    axis or a constant, each operand is read by its one load alone, and no
    axis is empty.
  - The rule tiles it as CUTLASS's SIMT GEMM does. A block of 16 × 16
    threads computes a 64 × 64 output tile, each thread a 4 × 4 tile in
    registers. Each 8-deep step of the sum stages both operands' slices
    with `cache_read`, and the step's depth and the thread's tile are
    unrolled. The batch and block loops fuse onto `blockIdx.x`, which no
    grid limit binds.
  - Like D49's rules, it is a plain function with fixed tiles until the
    search has knobs to tune.
- **Reason:** the structure, not the op, is what tiling depends on, so any
  stage of that shape benefits, and the tensor-core rule reuses the same
  recognition. Every output still sums its axis in order, so a tiled
  kernel matches the serial schedule bit for bit, which the tests check in
  the interpreter and on the GPU.

## D57. An f32 sum of products is fused into multiply-adds

- **Context:** CUDA-core kernels issued a multiply and an add for every
  product they summed, keeping the GPU bit-exact with the loop
  interpreter, which rounded each. An FFMA does both at the throughput of
  one, and nvcc contracts `acc + a * b` into it by default.
- **Options:**
  - keep separate operations, and so half the arithmetic throughput;
  - contract in codegen alone, so the GPU no longer matches the
    interpreter bit for bit;
  - contract in the loop IR, which both executors follow.
- **Choice:** `ValueExpr::MulAdd` computes `lhs * rhs + addend` on f32,
  rounded once. Lowering combines a reduction's f32 sum with a product
  body into it. Codegen selects `fma.rn.f32`, and the interpreter uses
  `f32::mul_add`, IEEE's fused operation, rather than its f64 arithmetic,
  which would round twice. Other sums, and products outside a sum, keep
  their separate operations.
- **Reason:** both executors compute the same rounding, so the GPU still
  matches the interpreter bit for bit, while the hot loop of every matrix
  product, convolution and attention stage issues one instruction per
  term. A fused sum is at least as accurate as a separate one, and the
  reference comparisons already allow a sum's rounding.

## D58. Tensor-core fragments are logical tiles in the loop IR

- **Context:** a warp's `mma.sync` operands live in registers spread over
  its 32 lanes, in layouts the PTX ISA fixes per shape and type, and
  `ldmatrix` has its own x1, x2, x4 and transposed forms. M3's plan had the
  loop IR and its interpreter model those per-lane layouts, so
  tensor-core schedules could be checked without a GPU, and codegen encode
  them again.
- **Options:**
  - per-lane fragments in the loop IR, the interpreter following the PTX
    layouts and codegen repeating them;
  - fragments as logical tiles one warp owns, with codegen the only place
    that maps them to lanes, as Triton's block-level dot and TVM's first
    wmma fragments do.
- **Choice:** a `Fragment` memory scope carries its tile's rows and
  columns. Four statements, each executed by a whole warp at once, touch
  fragments: fill, load a tile from shared memory (stored or transposed),
  `Mma` for an instruction of the target's table, and store a tile. The
  interpreter runs each once per warp on the logical tile; its `Mma` adds
  the exact products to the accumulator in `k` order and rounds once. The
  verifier checks shapes and types against the instruction, that tiles
  stay in their buffers, and that whole warps reach the statements.
- **Reason:** what CPU verification protects is the schedule: which tiles
  are staged, loaded and multiplied, and that, at the logical level, still
  runs in the interpreter. The lane layouts belong to instruction
  selection, like every other register assignment, and the GPU tests check
  them. One encoding of them instead of two is less code and one fewer
  place to be wrong, and the PTX emitted is the same. NVIDIA does not
  document how tensor cores round a sum, so GPU and interpreter are
  compared bit for bit on inputs whose sums are exact in any order, and
  within the type's tolerance otherwise.

## D59. Fragments leave a warp only through `FragmentEach`

- **Context:** D58 gave fragments a store statement, writing a tile to
  memory at a `TileAccess`. Tensorizing then added `FragmentEach`, which
  runs a body for each element of a tile with its row and column bound, so
  the output's own store, guards and epilogue apply to an accumulator. No
  schedule emits the tile store, and on `sm_89` it would select to the
  same per-element stores a `FragmentEach` does: `stmatrix` needs `sm_90`.
- **Options:** keep both, or keep only `FragmentEach`.
- **Choice:** only `FragmentEach`. A tile store is a `FragmentEach` whose
  body stores the element at the tile's position.
- **Reason:** one way out of a fragment, which every executor, the
  verifier and codegen implement and every schedule exercises, rather than
  two, one of them untested. A target with `stmatrix` can select it for a
  `FragmentEach` whose body is a plain store into shared memory.
- **Supersedes:** the tile store among D58's fragment statements.

## D60. Tensor-core stages match the interpreter within tolerance

- **Context:** every check of the GPU against the loop interpreter was
  exact. Tensor cores round a sum their own way, which NVIDIA does not
  document, so under D58 a kernel with `mma` matches bit for bit only on
  inputs whose sums are exact in any order. Once the default schedule
  multiplies f16 and bf16 matrices on tensor cores (#106), random inputs
  in the family tests, fuzzing and the ONNX suite would differ in the last
  bits of those sums.
- **Options:**
  - model a GPU's tensor-core rounding in the interpreter, which ties the
    interpreter to one architecture's undocumented behaviour;
  - keep tensor cores off by default, which leaves every f16 model on CUDA
    cores;
  - hold only tensor-core stages to a tolerance.
- **Choice:** a GPU run whose lowered stages include one that multiplies
  on tensor cores is compared with the loop interpreter within each
  output type's tolerance from TESTING.md, widened for the longest axis
  of those stages' arguments, which bounds their reductions. Integer
  sums stay exact. Everything else is still compared exactly. Fuzzing
  compares such a case only where it is continuous with moderate inputs,
  as it already does with the reference, since a last-bit difference can
  flip a later floor or comparison. Tests of tensor-core codegen keep
  exact-sum inputs and compare exactly.
- **Reason:** the interpreter stays the target-independent meaning of a
  kernel, and every kernel without `mma` keeps the bit-for-bit check. A
  wrong fragment layout, tile or step is far outside the tolerance, and
  the exact-sum tests still pin tensor-core codegen bit for bit.

## D61. `MatMulInteger` subtracts its zero points through operand sums

- **Context:** ONNX `MatMulInteger` multiplies 8-bit integer operands less
  their zero points into `i32`. Subtracting the zero points inside the sum
  makes each term a product of two differences, which no `mma` takes. Real
  int8 GEMMs have nonzero activation zero points, so that would keep them
  off tensor cores.
- **Options:** sum `(a - za)(b - zb)` directly; or sum `ab` and correct
  it, as int8 GEMM libraries do:
  `sum((a - za)(b - zb)) = sum(ab) - zb sum(a) - za sum(b) + k za zb`.
- **Choice:**
  - The product stage sums `ab` in wrapping `i32`, which the default
    schedule multiplies on tensor cores when both operands have one 8-bit
    type.
  - Stages of their own sum `a` along its rows and `b` along its columns,
    only where the other operand's zero points are not all 0. The
    product's epilogue subtracts them and adds `k za zb`.
  - Zero points are attributes (D32): one per tensor, or one per row of
    `a` or column of `b`, read from a table (D46). Zero points that also
    vary over a batch are unsupported.
- **Reason:** in wrapping `i32` the identity is exact, so the result is
  ONNX's bit for bit, including where ONNX lets the sum overflow. The
  sums cost one pass over each operand, against a product that reads each
  of them once per tile of the other.

## D62. Codegen holds constant-indexed local arrays in registers

- **Context:** codegen declares accumulators and `mma` fragments as
  `.local` arrays and relied on ptxas to keep them in registers, which it
  does where every access is at a constant offset. ptxas 13.5 stops doing
  so in any function with a `cp.async`, whatever its cache operator or
  address form; a 40-line kernel reproduces it. Every staged kernel is to
  fill its tiles with `cp.async` (#108), which would move every fragment
  to the stack.
- **Options:** keep synchronous fills and stage through registers, as
  pre-Ampere GEMMs do; or hold the arrays in registers in codegen, as
  LLVM's SROA does before NVVM emits PTX, so ptxas never sees them as
  memory.
- **Choice:** codegen holds a local or fragment array of elements of at
  most 4 bytes in registers when every access to it is at a constant index
  after unrolling. Each 4 bytes is one 32-bit register; `ldmatrix` and
  `mma` take the words, and a narrower element is shifted and masked out of
  its word, or merged into it, as a load or store of its type would.
  Arrays with a computed index stay in local memory.
- **Reason:** NVCC's code never depends on ptxas promoting local memory,
  because NVVM hands it registers; doing the same keeps register residency
  under ptxc's control and independent of what else a kernel contains.
  Fragments are already accessed at constant indices for ptxas's sake, so
  the rule holds every fragment the default schedules make.

## D63. Staged tiles fill with `cp.async`

- **Context:** a staged fill loaded each run into registers and stored it
  to shared memory, 16 bytes at a time where the tile allowed (#146) and
  one element otherwise. Multi-stage pipelining (#109) needs copies that
  land later than they are issued, which only `cp.async` gives on sm_80
  and later.
- **Options:** keep synchronous fills and add `cp.async` only with
  pipelining; or fill with `cp.async` now, one stage deep, so the pipeline
  only moves waits.
- **Choice:**
  - A fill copies each run of 16, 8 or 4 bytes with one `AsyncCopy`, the
    longest the tile's rows allow. It then commits once and waits for
    every group before its closing barrier.
  - A copy carries its input bounds and copies zeros where they fail, so a
    tensorized tile overhanging its input needs no separate clear.
  - Tiles whose rows fit no such run, for example 8- or 16-bit elements at
    an odd offset, still copy one element at a time.
  - The interpreter lands a copy only at its wait and rejects any access
    before it, so a missing or misplaced wait fails on the CPU.
- **Reason:** this is the copy CUTLASS's Ampere GEMMs make, and a pipeline
  then only reorders waits, which the interpreter checks. One stage deep,
  it is 3 to 5% slower than loads and stores on large GEMMs whose tiles
  fit (`.ca` instead of `.cg` changes nothing), and up to 1.8 times faster
  where tiles overhang their input or the product is skinny. The f16 NN
  sweep's shapes of at least 1024 go from 70% to 78% of cuBLAS. It
  depends on codegen holding fragments in registers (D62).

## D64. Pipelining fills a loop's tiles `S - 1` iterations ahead

- **Context:** with one stage, each reduction step waits for its own
  copies before computing (D63), so the copies' latency is never hidden.
- **Options:**
  - Double buffering only, as TVM's `double_buffer` scope.
  - An `S`-stage pipeline, as CUTLASS's multistage mainloop.
  - A pipeline annotation per stage, as TVM's `software_pipeline_stage`.
- **Choice:**
  - `Primitive::Pipeline { loop_, stages }` marks a serial loop with at
    least one input staged at it `Pipelined(stages)`, for any `stages` of
    2 or more. Like the other annotations, it keeps the loop from being
    split, fused or annotated again.
  - Lowering gives each tile `stages` slots. Before the loop it starts the
    first `stages - 1` iterations' copies, a group each.
  - Each iteration waits until at most `stages - 2` groups are in flight,
    passes one barrier, starts iteration `i + stages - 1`'s copies into
    the slot iteration `i - 1` read, commits them, and computes on slot
    `i mod stages`.
  - A barrier before the prologue keeps the loop safe inside an outer one.
- **Reason:** the structure is the one CUTLASS's Ampere GEMMs use, and two
  stages give double buffering as a case of it. The interpreter checks every wait and barrier, so a schedule
  with too few of either fails on the CPU. The tuner can search the stage
  count against shared memory.

## D65. Default schedules pipeline every tile but the 128 by 128 one

- **Context:** #153 added pipelining (D64), and the default matmul rules
  had to choose stage counts. Swept on the RTX 4060 Ti against one stage
  (D63):
  - the SIMT tile and the 64 by 64 tensor-core tile gain at 2, 3 or 4
    stages;
  - the 128 by 128 tile loses 3 to 10% at 2 and at 3. Registers fall with
    pipelining and occupancy does not change, and preferring the most
    shared memory per SM changes nothing.
  - Its loop has 255 instructions an iteration against 151. With one copy
    of each tile, ptxas folds every `ldmatrix` address into a register
    and an immediate outside the loop; with a slot that rotates, it
    recomputes the 24 addresses every iteration.
  - A pipelined loop's slot, `floormod(k, S)`, was also divided again in
    every guarded load of a SIMT body, which made 3 stages up to 4 times
    slower than 1.
- **Choice:**
  - Codegen computes each loop body's dividing index expressions that the
    body's own loops and elements do not change once, at its start, as it
    already did for the kernel's grid-only ones (#145).
  - The SIMT tile and the 64 by 64 tile pipeline over four stages, or as
    many as fit shared memory if fewer: a power of two makes the slot a
    mask, and four was as fast as two or faster everywhere measured.
  - The 128 by 128 tile is not pipelined until its addresses rotate
    cheaply.
- **Reason:** each choice is the faster one measured, and none gives up
  the large tile's speed. Pipelining the large tile profitably needs
  `ldmatrix` addresses that advance with the slot by an immediate or one
  add, as CUTLASS's iterators do, which is codegen work of its own (#155).

## D66. A swizzle is a shared buffer's layout

- **Context:**
  - The 128 by 128 tensor-core tile is bound by shared memory. Nsight
    Compute on 3000³ shows 5.2M `ldmatrix` taking 83.2M wavefronts, 69.3M
    of them bank-conflict replays.
  - The tiles are row-major. The 8 rows an `ldmatrix` reads lie 64 or 256
    bytes apart and share 2 or 1 of the 8 groups of 16-byte banks a
    128-byte line spans.
- **Options:**
  - An XOR in the index expressions, which would make them non-affine for
    the simplifier, bounds, alignment and run analyses.
  - Padding the rows, which wastes shared memory and still leaves `cp.async`
    runs to place.
  - A permutation of the buffer's bytes that indices never see, as
    CUTLASS's swizzled layouts are.
- **Choice:**
  - `MemScope::Shared { swizzle: Option<Swizzle> }`: a swizzle is part of
    a shared buffer's scope and nothing else's.
  - `Primitive::Swizzle` gives a staged tile one, and `Swizzle::for_tile`
    picks CUTLASS's: 16-byte units permuted by the row's place among 8.
  - The interpreter stores through the permutation and codegen applies it
    to every byte offset into the buffer.
  - The default schedule swizzles the 64 by 64 tensor-core tile: shapes
    using it gain up to 19%, and none is slower beyond noise.
  - It does not swizzle the 128 by 128 tile yet: its swizzled addresses
    cost a 64-bit shift, mask, shift and exclusive-or each. At 215
    registers ptxas recomputes them every iteration, and 3000³ becomes 10%
    slower though 4096³ becomes 6% faster. Cheap addresses are #157.
- **Reason:** indices stay affine, so every analysis on them stands; a
  swizzle that loses elements fails on the CPU; and a unit of 16 bytes
  keeps every copy and `ldmatrix` row whole.

## D67. Swizzled addresses fold their constants in apart

- **Context:**
  - Under D66 every swizzled address costs a 64-bit shift, mask, shift and
    exclusive-or. Yet the `ldmatrix` reads and `cp.async` writes of an
    unrolled step differ only by constants: tile origins and steps along
    the reduction.
  - Computing shared addresses in 32 bits first made the plain 128 by 128
    tile 10% slower at 4096³. Truncating before scaling hides each tile's
    constant from ptxas, which then keeps a register per copy instead of
    one base and immediates (236 registers, from 228).
- **Options:**
  - Per-thread base pointers in the loop IR, which needs exclusive-or in
    index expressions (rejected in D66).
  - In codegen, the per-thread part of an address swizzled once and each
    constant folded in after, as CUTLASS's iterators do. This needs to know
    which bits the per-thread part can have.
- **Choice:**
  - A swizzled buffer's index is split into its constant addend and the
    rest. With the swizzle's period `T` (`2^(M+S+B)` bytes) and the
    constant `kT + r`:
    - where the rest's possible bits miss `r`'s, the address is
      `(swz(rest) ^ swz(r)) + kT`, with `kT` the immediate offset;
    - otherwise the whole sum is swizzled, as before.
  - `possible_bits` is a known-bits analysis over the emitted PTX, as
    LLVM's `computeKnownBits`, following each register to the one
    instruction that defines it. The launch bounds lane, thread and block
    indices.
  - A register set more than once is a `Var`: only
    `FunctionBuilder::variable` makes one, and only `assign` and
    `increment` write one. It is known to have several definitions from its
    creation, so the analysis never takes a loop counter's first value for
    all of its values.
  - Plain buffers' addresses are unchanged: ptxas folds their constants
    itself.
  - The 64 by 64 tile, swizzled and pipelined, becomes 9 to 20% faster on
    every f16 NN shape that uses it, measured at a locked 2700 MHz.
  - The 128 by 128 tile stays plain:
    - Its swizzled loop is now the same 151 instructions as the plain one.
      Nsight Compute on 1024³ shows its shared-load bank conflicts gone
      (2.6M before) and shared wavefronts down from 3.1M to 0.5M.
    - But with no pipeline, the two blocks an SM holds start together and
      stay in step: they wait on their loads and contend for the tensor
      cores at the same times (`math_pipe_throttle` up from 11.5 to 13.8
      cycles an instruction).
    - An SM with one block finishes 13% sooner, one with two 3% later, so
      one-wave grids lose 5 to 10%. Grids of many waves, whose blocks fall
      out of step, gain 5 to 8%.
    - Pipelining the tile comes first.
- **Supersedes:** D66's reason for leaving the 128 by 128 tile plain.
- **Reason:** a swizzle is linear over exclusive-or (`swz(x ^ y) =
  swz(x) ^ swz(y)`) and commutes with adding a multiple of its period, and
  a sum of values with disjoint bits is their exclusive-or. The analysis
  runs on what codegen emits, so it sees the lane arithmetic that the loop
  IR does not express.

## D68. The 128 by 128 tile pipelines over two stages, swizzled

- **Context:**
  - Alone, neither change pays on this tile. Pipelined but plain, it is
    3 to 12% slower from 1024³ to 4096³ (D65). Swizzled but unpipelined,
    grids of one wave lose 5 to 10% (D67).
  - Together they cover each other's loss. The pipeline keeps each
    block's next loads in flight while it computes, so the two blocks an
    SM holds no longer wait in step. The swizzle removes the bank
    conflicts that otherwise bind the pipelined loop.
- **Options:** two or three stages, plain or swizzled. Four stages need
  64 KiB a block, which leaves one block an SM.
- **Measured:** f16 NN on the RTX 4060 Ti locked at 2700 MHz, two
  interleaved rounds that agree within 0.4%, as a share of cuBLAS:

  | shape | unpipelined, plain | 2 stages, swizzled | 3 stages, swizzled |
  |---|---|---|---|
  | 1024³ | 74.7% | 76.3% | 75.8% |
  | 2048³ | 79.2% | 82.7% | 82.9% |
  | 3000³ | 82.4% | 85.3% | 85.5% |
  | 4096³ | 80.2% | 87.5% | 87.2% |
  | 8192³ | 71.8% | 75.9% | 72.3% |
  | geomean | 77.5% | 81.4% | 80.5% |

  - Of the other shapes using the tile, two stages make 4096×2304×768
    12% faster, 4096×3072×768 11%, 1024×3072×768 5%, 4096×768×3072 4.5%
    and 1000³ 2%.
  - 1024×768×3072 is unchanged (+0.5%). 1024×768×768 is 2.8% slower: a
    grid of one wave and 24 steps, whose fill the pipeline adds to.
- **Choice:**
  - The 128 by 128 tile pipelines over two stages and is swizzled.
  - Every tensor-core tile is now swizzled where its rows have a swizzle,
    so the schedule no longer carries a layout choice.
- **Supersedes:** D65's leaving the 128 by 128 tile unpipelined, and D67's
  leaving it plain.
- **Reason:** it is the fastest measured, and the one loss is under 3%
  on the smallest grid. The pipelined loop still does much more integer
  work an iteration than the unpipelined one: 287 instructions against
  152 at 4096³, and 322 against 286 at 3000³. That is #155's to remove.

## D69. Addresses are summed outermost loop first, their invariant part before the loop

- **Context:** the 128 by 128 tile's pipelined loop (D68) ran 287
  instructions an iteration at 4096³, against 152 for the unpipelined one.
  - A swizzled address swizzled its whole offset, though a pipeline slot
    (8 KiB), a warp's tile origin and a copy loop's chunk (2 KiB) are
    whole multiples of the swizzle's period (1 or 2 KiB).
  - An index was emitted in the order its expression's tree held it. A
    refill's global address, `((inv₁ + k·32) + inv₂) + …`, then had no
    partial sum the loop did not change, so ptxas recomputed all of it
    every iteration, before and after pipelining.
  - ptxas does not hoist out of a branch the body runs only sometimes, as
    it runs the refill.
  - Each refill copy checked bounds its guard already implied.
- **Choice:**
  - An index is taken apart into terms, constant factors distributed over
    sums. It is summed outermost loop first, as LLVM's Reassociate ranks
    operands: a term's rank is the depth of the innermost loop whose
    variable it reads.
  - An address is the buffer's start plus each rank's part in bytes,
    outermost first, with the constant as its immediate offset.
  - In a swizzled buffer, terms that are whole multiples of the period are
    added after the swizzle, since `swz(x + mT) = swz(x) + mT`. Only the
    rest is swizzled, and the constant folds in as in D67.
  - 64-bit index operations are value-numbered within a block, except
    those reading a `Var`, whose value changes.
  - Before each serial loop, codegen computes the part of every address in
    its body that reads no variable of the loop or of a loop inside it, as
    LLVM's LICM does. Value numbering hands the body those registers.
  - Lowering builds a refill's copies with the loop variable's range
    narrowed to what their guard allows, so they check no bound the guard
    implies.
- **Result:**
  - The 128 by 128 tile's loop has 150 instructions an iteration at 4096³
    (from 287) and 210 at 3000³ (from 322), fewer than unpipelined (152
    and 286), at 212 and 229 registers.
  - The golden kernels' PTX shrinks: the row argmax from 449 instructions
    to 404, and the row sum from 146 to 124.
  - Against main, f16 NN on the RTX 4060 Ti locked at 2700 MHz, two
    interleaved rounds within 0.5%:
    - the 64 by 64 tile's shapes are 13 to 27% faster: 128×768×768 from
      10.95 to 7.95 µs;
    - the 128 by 128 tile's are 3 to 5% faster, but 8192³, which is
      bound elsewhere, is unchanged;
    - the M3 exit geomean goes from 81.4% to 84.1% of cuBLAS.
  - f32 NN on CUDA cores is 2 to 12% faster.
- **Reason:** these are the standard passes for this shape of loop. ptxas
  does not reassociate a sum by loop depth or hoist out of a branch, so
  codegen does both.

## D70. A convolution is an implicit GEMM by its schedule, its tiles reindexed

- **Context:** a convolution is a sum over taps and input channels of an
  input element times a weight. On a GPU it runs fastest as an implicit
  GEMM, as cuDNN's and CUTLASS's kernels run it: output positions `n·p·q`
  are the rows, output channels the columns and `r·s·c` the depth, and
  each step stages a rows-by-depth slice of the input.
  - That slice is not a box of the input: a row's window moves with `p`
    and `q`, and a step's depth crosses taps and channels. `CacheRead`
    stages a box per input axis, so it cannot stage it.
  - A compute definition rewritten as the GEMM, its axes the fused ones,
    would make the interpreter, the default rules and every other
    schedule see that layout, and the output's shape its own.
- **Choice:**
  - The compute definition stays the direct convolution. A schedule fuses
    `(n, p, q)` and `(r, s, c)` with `Fuse` and tiles the result as a GEMM,
    as TVM's convolution schedules do.
  - `Reindex` makes a staged tile a box in the iteration space, as
    MetaSchedule's `ReIndex` stages a block's read: each coordinate is a
    group of axes fused row-major. The input reindexed by
    `[n, p, q] [r, s, c]` stages as the GEMM's left matrix, the weights by
    `[m] [r, s, c]` as its right.
  - The loop tree carries each reindexed coordinate through later fuses
    and splits, as it carries axis indices, so a group fused into one loop
    stays that loop's index and the tile stays affine in the loops it
    spans. Reindexing comes before the groups' loops are fused and split,
    as MetaSchedule reindexes before tiling: rebuilding the coordinate
    from its axes after a split needs a canonical form for fused and split
    iterators (TVM's `IterMapDetector`) that the simplifier does not have.
  - The fill places each element by splitting its coordinates back into
    their axes and evaluating the input's index there. A run of elements
    copies as one `cp.async` where the innermost group's innermost axis
    steps the input's innermost index by one, as channels do in NHWC; the
    copy checks each coordinate against its group's extent and the index
    against the input.
  - Validation refuses a reindexed tile whose coordinates miss an axis
    the input's index reads and the tile spans.
- **Result:** a padded, strided, dilated convolution in NHWC or NCHW,
  staged both ways and pipelined, matches the default schedule in the
  loop interpreter. In NHWC, with four 32-bit channels, both tiles copy
  16 bytes at a time; in NCHW, element by element.
- **Reason:** the convolution's meaning stays in one place, the compute
  definition, and implicit GEMM is a schedule like any tiling, open to
  the tuner. MetaSchedule reaches tensor cores on convolutions the same
  way.

## D71. A window reads its padding through the load

- **Context:** a convolution's or pool's tap that reads padding was handled
  after the load. The index was clamped into the input and a select took
  the padding value: zero in place of a convolution's product, the lowest
  value for a max pool, zero for an average.
  - A tile staged for `mma` (D70) is multiplied as it lies, so padding
    has to be in the tile. A select over the product is not a value the
    fill can copy, and `cp.async` copies or zero-fills, nothing between.
  - The reference interpreter skipped a convolution's padded taps. ONNX,
    ONNX Runtime and cuDNN pad the input with zeros and multiply them in,
    so an infinite weight at a padded tap gives NaN there.
- **Choice:**
  - `TensorExpr::Load` carries `padding: Option<Scalar>`: the value it
    reads where its index lies outside the tensor, as Halide's
    `BoundaryConditions::constant_exterior` and TVM's inlined padding
    stage read one. Without padding, the index lies inside wherever the
    load runs, as `ComputeBuilder::finish` checks.
  - Unstaged, a padded load lowers to the load clamped into the tensor,
    selected against the padding wherever the index can leave it.
  - Staged, the fill already checks each copy's bounds and zero-fills what
    lies outside the input, as CUTLASS's implicit-GEMM iterators do, so a
    tile holds zero padding with no select. A non-zero padding is stored
    before an element copy, which then copies only what lies inside.
  - A convolution pads its input with zero, a quantized convolution with
    its input's zero point, a max pool with the lowest value and an
    average pool with zero. The reference convolution multiplies padding
    as ONNX does.
- **Result:**
  - The implicit GEMM's inner loop is a multiply-add with no select, and
    its input tile copies with bounds on the window's taps.
  - An empty input is all padding, so a padded convolution of one sums
    zeros times its weights rather than returning zero early.
- **Reason:** padding is a property of how a window reads its input, so
  it belongs with the read, where staging, lowering and the interpreter all
  see it, rather than in a mask around whatever uses the value.

## D72. A quantized matmul may give a float result, from FP8 operands too

- **Context:** the GEMM sweep's FP8 rows take E4M3 operands to an f32
  result. No ONNX op does that directly: models write `DequantizeLinear`
  on FP8 operands feeding `MatMul`, which QDQ folding (M4) is to fold into
  a quantized matmul. But `QMatMul` always requantized its result, and D46
  made FP8 operands of a quantized product an error.
- **Choice:** this supersedes D46's FP8 sentence.
  - `QMatMulAttrs::output` is a `QMatMulOutput`. `Quantized` is the grid
    the product is requantized onto, as `QLinearMatMul` gives it, from
    operands on integer grids. `Float` is a float type the dequantized
    product is given in, from operands both on integer grids or both on
    FP8 ones, whose zero points are 0.
  - Lowered, integer operands keep the exact zero-centred sum. FP8 operands
    sum their products widened to f32, the plain product of two loads that
    the tensor-core rule tiles on FP8 `mma`. A float result is the sum in
    f32 times both scales, converted to its type; per-row and per-column
    scales are tables, as D46 has them.
  - The reference gives a float result as the exact sum times both scales
    in f64, rounded once: the dequantized product as real numbers. The
    lowering's f32 sum is within f32's tolerance for the inner extent.
  - A requantized result keeps integer operands and an integer grid, which
    is all `QLinearMatMul` imports. FP8 operands with a requantized result
    are refused by the type rule.
- **Reason:** a float result is how FP8 GEMMs are used, by TensorRT and
  Transformer Engine alike: the scales go into the epilogue and the
  product stays in f32 or f16, not back on a grid. Keeping the operands'
  dequantization out of the sum lets the FP8 `mma` multiply them as
  stored.

