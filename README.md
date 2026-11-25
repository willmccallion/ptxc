# ptxc

A deep-learning compiler written from scratch in Rust. It takes an ONNX
model, optimises and fuses its graph, schedules each fused kernel for the
GPU, emits PTX directly, and runs the result through the CUDA driver API
with a statically planned memory arena and CUDA graphs.

**Status:** M3, tensor-core GEMM and convolution benchmarked against
cuBLAS, is done. ptxc tiles matrix products and convolutions onto tensor
cores in f16, bf16, E4M3 and int8, with `ldmatrix`, `mma.sync`, multi-stage
`cp.async` pipelines, swizzled shared memory and split-K, in PTX it emits
itself.
- f16 GEMMs with M, N and K at least 1024 run at 90% of cuBLAS's
  throughput as a geometric mean (NN 89%, NT 89%, TN 91%), with
  hand-picked tiles before any autotuning.
- Every swept GEMM and convolution is checked at full size against an f64
  reference. On tensor cores, every `mma` type at the sweep's kinds of
  shape, made small enough to interpret, and all 23 of ResNet-50's
  convolution layers match the loop interpreter bit for bit.
- Where ptxc loses, the [results](#results) show it:
  - ResNet-50's convolutions run at 26–32% of cuDNN. ptxc takes them NCHW,
    as ONNX gives them; channels-last comes with M4's layout pass.
  - int8 and E4M3 run at 22–25% of cuBLAS in NT, the layout its fast
    kernels take, where ptxc builds the transposed operand with a kernel
    of its own ([#176](https://github.com/willmccallion/ptxc/issues/176)).
  - f32 on CUDA cores runs at 55–56% of cuBLAS in NN and TN, 16% with B
    transposed ([#189](https://github.com/willmccallion/ptxc/issues/189)),
    and least with few rows
    ([#180](https://github.com/willmccallion/ptxc/issues/180)).
- M2's correctness checks, from the ONNX node tests to fuzzing against ONNX
  Runtime, are described in [docs/TESTING.md](docs/TESTING.md).

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

M3's sweeps on an RTX 4060 Ti with its SM locked at 2355 MHz, timed as
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) describes and rendered from
the committed results by `ptxc-baselines render`. Percentages are of the
baseline's throughput, as a geometric mean over the problems.

### GEMM against cuBLAS

On this GPU cuBLASLt runs E4M3 only as NT, so the FP8 NN and TN rows have
no baseline. Its int8 NN and TN kernels are several times slower than its
NT ones, so ptxc's int8 percentages are highest where cuBLAS is slowest.

<!-- bench:gemm:begin -->
| Type | Layout | Problems | % of cuBLAS | % of best of 8 | Faster than cuBLAS | Slowest against cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 | NN | 32 | 80% | 77% | 3 of 32 | 1x768x768 (53%) |
| f16 | NT | 32 | 77% | 73% | 2 of 32 | 4x2304x768 (50%) |
| f16 | TN | 32 | 78% | 75% | 4 of 32 | 1x2304x768 (38%) |
| bf16 | NN | 32 | 80% | 77% | 3 of 32 | 1x768x768 (53%) |
| bf16 | NT | 32 | 76% | 73% | 2 of 32 | 4x2304x768 (50%) |
| bf16 | TN | 32 | 78% | 75% | 5 of 32 | 1x2304x768 (39%) |
| e4m3 | NN | 32 | – | – | 0 of 0 | – |
| e4m3 | NT | 32 | 22% | 22% | 0 of 30 | 1x3072x768 (6%) |
| e4m3 | TN | 32 | – | – | 0 of 0 | – |
| i8 | NN | 32 | 126% | 121% | 26 of 32 | 1x2304x768 (48%) |
| i8 | NT | 32 | 25% | 22% | 0 of 32 | 1x3072x768 (8%) |
| i8 | TN | 32 | 113% | 106% | 17 of 32 | 1x2304x768 (43%) |
| f32 | NN | 32 | 56% | 53% | 1 of 32 | 4x2304x768 (18%) |
| f32 | NT | 32 | 16% | 14% | 0 of 32 | 1x3072x768 (5%) |
| f32 | TN | 32 | 55% | 51% | 0 of 32 | 1x2304x768 (18%) |

f16 GEMMs with M, N and K at least 1024, geometric mean of cuBLAS's throughput: 90% (NN 89%, NT 89%, TN 91%).

† The GPU reported a power or thermal cap during the series; its clocks held.

Every case: [bench/results/2026-11-25-8ef7be1/gemm.md](bench/results/2026-11-25-8ef7be1/gemm.md).
<!-- bench:gemm:end -->

### ResNet-50's convolutions against cuDNN

ptxc takes each layer NCHW, as ONNX gives it, so its input tiles copy
element by element; channels-last comes with M4's layout pass.

<!-- bench:conv:begin -->
| Batch | Layers | % of cuDNN NCHW | % of cuDNN channels-last | Faster than NCHW | Slowest against NCHW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 23 | 26% | 19% | 0 of 23 | layer4-reduce (11%) |
| 8 | 23 | 27% | 21% | 0 of 23 | layer3-3x3 (14%) |
| 32 | 23 | 32% | 23% | 0 of 23 | layer1-reduce (15%) |

Every case: [bench/results/2026-11-25-8ef7be1/conv.md](bench/results/2026-11-25-8ef7be1/conv.md).
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
