# ptxc

A deep-learning compiler in Rust: ONNX → graph IR → loop IR → PTX → CUDA
driver API. Correctness first, proven by outside suites (ONNX node tests,
ONNX Runtime, PyTorch, ptxas). Then speed, compared honestly against cuBLAS,
PyTorch, torch.compile, ORT and TensorRT. `docs/` holds the architecture,
roadmap, testing, benchmarking and decision log; read the relevant one
before changing a stage.

## Borrowed designs and divergences

- Doc comments name the system a design follows (`as XLA's BufferAssignment`,
  `Ansor's sketch rules`, `FlashAttention-2's online softmax`).
- **Every intentional divergence from the ONNX spec, or from the system a
  design follows, carries a `// NOTE:` comment** saying what the original
  does and why ptxc does otherwise.
- A decision that changes the architecture is added to `docs/DECISIONS.md`
  as a new entry that supersedes the old one.

## Boundaries

- The compiler path never uses cuBLAS, cuDNN, CUTLASS or NVRTC. Only
  `ptxc-bench` loads cuBLAS.
- Crates other than `cuda`, `runtime`, `autotune` and `bench` never depend
  on a GPU. `cargo xtask check-deps` enforces the rules in ARCHITECTURE.md.
- Only `ptxc-cuda`, and `ptxc-bench`'s cuBLAS bindings in
  `crates/bench/src/cublas/sys/`, contain unsafe code (D55). Every unsafe
  block has a `// SAFETY:` line naming its invariant.
- `ptxc-interp` stays naive and shares no code with `ptxc-lower`.
- Python lives only in `baselines/`, managed with uv. Rust never runs it.

## Types

- No bare `usize`, `u64`, `i64` or `f32` crosses a function boundary
  carrying a meaning.
  - Ids come from `define_index!` (`NodeId`, `ValueId`, `ExprId`,
    `BufferId`, ...).
  - Sizes and offsets are `ElementCount`, `ByteSize`, `ByteOffset`, and
    `SmemOffset` separate from `GlobalOffset`.
  - Tensor geometry uses `Dim`, `Axis` and `Rank`.
  - The target uses `SmVersion`, `PtxIsaVersion`, `ThreadsPerBlock`,
    `SmemBytes`; tiles use `BlockTile`, `WarpTile`, `MmaShape`.
  - Quantization uses `Scale` and `ZeroPoint`; time uses `Microseconds`.
- Conversions between units are methods on the newtypes, never casts at call
  sites. Cast lints stay denied.
- Phases the code drives are typestates: `Compilation<S>`, `Schedule<S>`,
  `GraphCapture<S>`, `Executor<S>`, the launch builder.
  - When a phase changes the data's shape, it gets a distinct type
    (`Graph` → `Partitioned`, `ComputeDef` → `LoweredKernel`).
  - State known only at runtime is an enum.
- Absence is `Option`; one-of-N is an enum, never a bool pair or a flag
  argument.
- Data is parsed once at a boundary (protobuf, CLI, `Program` files, tuning
  records) and trusted after.
- A trait exists only where the roadmap commits to at least two
  implementations (see the table in ARCHITECTURE.md).

## Files and functions

- One concept per file, with a soft cap of about 300 lines; split by concept
  past it.
- Op families are mirrored file for file in `onnx/src/import`,
  `graph/src/ops`, `interp/src/eval` and `lower/src/ops`. A new op touches
  the same family file in each.
- Functions: 60 lines is clippy's hard ceiling; aim far below.
- Workspace lints are pedantic, nursery and cargo at deny. Fix the code
  rather than adding `#[allow]`.

## Tests

- `cargo test` needs no GPU and no downloads. Device tests are compiled
  under `--features gpu` and fail without a GPU; they never skip.
- Golden files live in a crate's `tests/golden/`. `PTXC_BLESS=1 cargo test`
  rewrites them. Bless only on unchanged code, and review every golden diff.
- Every bug fix comes with a test that fails without it. Shrunk fuzz cases
  become fixtures.
- Before refactoring a path, pin its behaviour with tests and commit them.
- Tolerances are per dtype, in `ptxc-testkit`. Never loosen one to pass a
  single test.

## Workflow

- `fish -c 'p check'` runs fmt, clippy, tests and `cargo doc`.
- `rust-toolchain.toml` pins the flake's rustc.
- `crates/onnx-proto/src/generated.rs` is generated: change the pin and run
  `cargo xtask proto`.
- README results change only at milestones, from committed code, through
  `baselines/ptxc_baselines/render_tables.py`.

## Measuring

- The GPU and CPU are shared with other timing runs. Ask before builds,
  benchmarks, fuzzing campaigns or anything else heavy, and never run GPU
  work during someone else's timing series.
- Never `pkill -f`. Stop only processes you started, by PID.
- Published benchmark series follow docs/BENCHMARKING.md: locked clocks,
  warmup, median with p10 and p90, clocks recorded before and after.
