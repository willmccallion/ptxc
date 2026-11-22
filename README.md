# ptxc

A deep-learning compiler written from scratch in Rust. It takes an ONNX
model, optimises and fuses its graph, schedules each fused kernel for the
GPU, emits PTX directly, and runs the result through the CUDA driver API
with a statically planned memory arena and CUDA graphs.

**Status:** planning (M0). The workspace contains documented stub crates
only; no compiler logic is implemented yet. See [the roadmap](docs/ROADMAP.md).

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
toolchain, protoc, uv and ptxas.

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
