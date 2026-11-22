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

## D3. Static shapes in v1

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

## D5. Data types in v1

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

## D18. No control flow in v1

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
