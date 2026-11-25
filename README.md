# ptxc

A deep-learning compiler written from scratch in Rust. It takes an ONNX
model, optimises and fuses its graph, schedules each fused kernel for the
GPU, emits PTX directly, and runs the result through the CUDA driver API
with a statically planned memory arena and CUDA graphs.

**Status:** M2, GPU code generation correct on everything, is done. ptxc
compiles ONNX models to PTX it emits itself and runs them through the CUDA
driver API.
- On the GPU, 984 of the 1,765 ONNX v1.22.0 node tests pass, test for test
  the same as the reference interpreter. The rest are reported unsupported
  with a reason (D33, D34).
- Every kernel matches the loop interpreter bit for bit, NaN payloads aside.
- 10,000 random graphs over the whole op set agree across the interpreters
  and the GPU, and 1,200 agree with ONNX Runtime.
- The kernels are naive. Tiling, tensor cores and the first comparison
  with cuBLAS come in M3.

See [the roadmap](docs/ROADMAP.md).

## Goals

- **Correct, checked from outside.**
  - Pass the official ONNX backend node tests.
  - Agree with a reference interpreter and ONNX Runtime on randomly
    generated graphs.
  - Match PyTorch on ResNet-50 ImageNet accuracy, GPT-2 perplexity and BERT
    outputs.
- **Fast, compared honestly.**
  - Matrix multiply against cuBLAS.
  - Whole models against PyTorch eager, torch.compile, the ONNX Runtime CUDA
    execution provider and TensorRT, in f16, int8 and FP8.
  - Tuning time against TVM/Ansor, and build time against TensorRT.
  - Published with the methodology in
    [docs/BENCHMARKING.md](docs/BENCHMARKING.md), including the cases where
    ptxc loses.
- **Self-contained.** The compiler path uses no cuBLAS, cuDNN, CUTLASS or
  NVRTC; those appear only as benchmark baselines. Kernels go from loop IR
  to PTX inside ptxc, with tensor cores used through `mma.sync`, `ldmatrix`
  and `cp.async`.

## Results

M3's sweeps on an RTX 4060 Ti with its SM locked at 2700 MHz, timed as
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) describes and rendered from
the committed results by `ptxc-baselines render`. Percentages are of the
baseline's throughput, as a geometric mean over the problems.

### GEMM against cuBLAS

<!-- bench:gemm:begin -->
<!-- bench:gemm:end -->

### ResNet-50's convolutions against cuDNN

ptxc takes each layer NCHW, as ONNX gives it, so its input tiles copy
element by element; channels-last comes with M4's layout pass.

<!-- bench:conv:begin -->
<!-- bench:conv:end -->

## Pipeline

```
ONNX → graph IR → rewrites, folding, layout, fusion → loop IR (compute + schedule)
     → GPU schedules (tiling, tensor cores, pipelining; autotuned) → PTX → Program → runtime
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes each stage, its data
structures and the crate boundaries.

## Repository

| Path | Contents |
|---|---|
| `crates/` | The Rust workspace: 21 crates, IRs independent of the CUDA runtime |
| `baselines/` | Python (uv): model export, reference outputs, baseline benchmarks |
| `bench/results/` | Benchmark results as JSON, committed at milestones |
| `docs/` | Architecture, roadmap, testing, benchmarking, decisions |
| `data/` | Downloaded models, datasets and test data (not committed) |

## Development

The dev shell (`direnv allow`, or `nix develop`) provides the pinned Rust
toolchain, uv, ptxas and cuBLAS (for `ptxc-bench` only).

```sh
cargo build
cargo test                    # needs no GPU
cargo test --features gpu     # device tests
cargo xtask check-deps        # crate dependency rules
```

Target hardware: NVIDIA RTX 4060 Ti (Ada, sm_89). Architectures are
described as data in `ptxc-target`.

## Licence

MIT or Apache-2.0, at your option.
