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
