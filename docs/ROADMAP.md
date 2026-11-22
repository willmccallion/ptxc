# Roadmap

Each milestone ends with a demo that can be run and exit criteria that are
measured, not asserted. README numbers change only at milestones, from
committed code.

The percentage targets for M3 to M6 are goals to calibrate against once M2
gives us real measurements. They will be revised in DECISIONS.md, with the
reason, rather than quietly lowered.

## M0: Plan and skeleton (this commit)

- The docs in `docs/`, a workspace of stub crates, lints, toolchain pin,
  flake.
- **Exit:**
  - `cargo build`, `cargo test`, `cargo clippy --all-targets` and
    `cargo doc` pass with zero warnings;
  - the flake provides protoc, uv and ptxas.

## M1: Frontend and reference interpreter

**Demo:** `ptxc conformance --backend interp` runs the ONNX backend node
tests and writes `report.json` and `report.md`. Every test is counted as
passed, failed, or unsupported (with the reason: op, dtype, attribute,
opset, control flow).

**Tasks**

1. **core: types.** `DType`, `Dim`, `Shape`, `Rank`, `Axis`, the size
   newtypes, `define_index!` and `IndexVec`, `HostTensor`, `QuantParams`.
2. **core: float formats.** One `FloatFormat` description and one encoder and
   decoder for f16, bf16, E4M3FN and E5M2, following ONNX's saturating and
   non-saturating cast rules. Exhaustive tests:
   - every encoding decodes and re-encodes to itself;
   - every f32 encodes to the nearest representable value, ties to even
     (all 2³² inputs; a release-mode, `#[ignore]` test).
3. **onnx-proto.** Pin an `onnx.proto` version and write `cargo xtask proto`.
   Commit the generated code. CI fails if regenerating changes it.
4. **xtask fetch.** Download the ONNX node tests at the pinned tag into
   `data/onnx/<tag>/node/`, and check a recorded SHA-256 of the archive.
5. **graph: IR.** Arenas, `Value`/`Node`, `Rewriter`, the verifier, and
   topological traversal.
6. **graph: ops.** The `Op` enum and `TypeRule` for the v1 op families:
   elementwise, activation, matmul, conv, pool, reduce, norm, shape, gather,
   quant.
7. **graph: text format.** Printer and parser, with round-trip tests.
8. **onnx: import.** Model-level import, tensor import (raw, typed fields,
   external data), the typed attribute reader, opset dispatch, and one
   importer file per family. Typed `ImportError`s.
9. **interp.** One evaluator file per family; the numerics in `numeric.rs`;
   exact quantize and dequantize.
10. **testkit.** `Tolerance` per dtype and the mismatch report, the
    `PTXC_BLESS` golden helper, and the ONNX test-case loader.
11. **cli.** `ptxc conformance` with JSON and Markdown reports.
12. **baselines.** A uv project with `onnxruntime` pinned, and a script that
    caches ORT outputs for any model and input set (used by M2's fuzzer and
    M4's checks).

**Exit criteria**

- Every node test whose ops and dtypes are in the v1 set passes on the
  interpreter, at the suite's own tolerance (rtol 1e-3, atol 1e-7).
- Every test outside the set is reported as unsupported with a specific
  reason. The list of unsupported tests is reviewed and recorded in
  DECISIONS.md.
- MLP, ResNet-50, BERT-base and GPT-2 (exported by `baselines/`) all import.
  The interpreter's ResNet-50 output matches ORT on 16 images.
- Graph-IR text round-trips for every node test model.

## M2: Naive GPU code generation, correct on everything

**Demo:** `ptxc conformance --backend gpu` passes everything the interpreter
passes, and `ptxc fuzz --backend gpu --cases 10000` finds no mismatch. MLP
runs end to end on the GPU from the CLI.

**Tasks**

1. **target.** `SmVersion`, `PtxIsaVersion`, the limits, an sm_89 preset,
   and the mma table (empty until M3).
2. **cuda.**
   - dlopen loader and function table.
   - `CudaError`.
   - Device, primary context, module, memory, stream, event, launch builder.
   - A GPU smoke test running a hand-written PTX vector add.
3. **loop: expressions.** The hash-consed `ExprArena` and the simplifier,
   with property tests against direct evaluation.
4. **loop: compute and schedule.** `ComputeDef`, then `Schedule<Open>` with
   split, reorder, fuse, bind, unroll and vectorize; `validate`; lowering to
   `LoweredKernel`; the verifier; the text printer.
5. **loop: interpreter.** Lockstep execution between barriers, memory
   spaces, and the race detector.
6. **lower.** One `ComputeDef` per op family. Each op is its own group, from
   a trivial partitioner.
7. **schedule.** Default rules: injective (flatten, then bind to blocks and
   threads) and reduction (one block per output row, tree reduction in shared
   memory).
8. **codegen.** The AST, instruction selection for expressions, memory and
   control flow, and the printer. Every module is checked by ptxas in tests.
9. **program and compiler.** The `Program` format, the
   `Compilation<S>` pipeline with no graph optimisation yet, and a trivial
   memory plan (one allocation per buffer).
10. **runtime.** Load, `Executor<Eager>`, and `Evaluate`.
11. **testkit: generator.** The seeded random graph generator over the v1 op
    set and the shrinker.
12. **cli.** `ptxc compile`, `ptxc run`, `ptxc fuzz` and
    `ptxc conformance --backend gpu`.
13. **xtask check-deps.** Enforce the dependency rules in ARCHITECTURE.md.

**Exit criteria**

- GPU conformance equals interpreter conformance, test for test.
- 10,000 fuzz cases (seeds recorded) with no mismatch against the
  interpreter, and 1,000 against cached ORT outputs.
- The loop interpreter agrees with the GPU on every conformance kernel.
- ptxas accepts every emitted kernel.
- `cargo test` with no GPU still covers lowering, scheduling and PTX
  printing.

## M3: Tensor-core GEMM and convolution, benchmarked against cuBLAS

**Demo:** `ptxc-bench gemm` produces the GEMM sweep table against cuBLAS,
and the convolution sweep table.

- Shared-memory tiling, `ldmatrix`, `mma.sync` for f16 and bf16 (f32
  accumulate), FP8 E4M3/E5M2 and int8 (i32 accumulate).
- cp.async multi-stage pipelines, XOR swizzles, vectorised global loads,
  split-K for tall-skinny shapes.
- Implicit-GEMM convolution (NHWC).
- The loop interpreter models `ldmatrix` and `mma` fragments, so tensor-core
  schedules are verified on the CPU.
- **Exit:**
  - every swept shape and dtype is correct against the interpreter;
  - geometric-mean throughput of at least 80% of cuBLAS on f16 GEMMs with
    M, N, K ≥ 1024, using hand-picked tiles before any autotuning;
  - the full table is published, including the shapes where we lose.

## M4: Fusion and memory planning; ResNet-50 and BERT end to end

**Demo:** ResNet-50 and BERT-base compile from ONNX and run, with accuracy
checked against PyTorch and latency compared with PyTorch eager and the ORT
CUDA execution provider.

- The graph passes: CSE, DCE, constant folding, algebraic rewrites, layout,
  QDQ folding.
- The greedy fusion partitioner and epilogue fusion.
- Greedy-by-size and linear-scan memory planners, with the overlap verifier.
- `Executor<Graphed>`.
- **Exit:**
  - ResNet-50 top-1 on the ImageNet validation subset within 0.1 points of
    PyTorch (f16);
  - BERT-base outputs within tolerance;
  - the peak arena is within 10% of the lower bound (the maximum over time of
    the live bytes);
  - the latency table is published against PyTorch eager and ORT.

## M5: Autotuner and cost model

**Demo:** `ptxc tune resnet50` improves end-to-end latency over the default
schedules. The tuning-time comparison with TVM/Ansor is published.

- Sketch generation from the rules, random and evolutionary search, the
  roofline and learned cost models, GPU measurement with correctness checks,
  the task scheduler, the records database.
- **Exit:**
  - the cost model's rank correlation (Kendall τ) on held-out records is
    reported;
  - tuned ResNet-50 and BERT beat the M4 defaults;
  - tuning time and the result are compared with Ansor on the same models.

## M6: Attention fusion, GPT-2, full benchmark table

**Demo:** GPT-2 124M runs with fused attention. The README table covers MLP,
ResNet-50, BERT-base and GPT-2 at several batch sizes against PyTorch eager,
torch.compile, the ORT CUDA EP and TensorRT, with f16, int8 and FP8 rows.

- The attention matcher and a FlashAttention-2-style schedule rule (online
  softmax, K/V tiles streamed through shared memory).
- QDQ int8 and FP8 models from ONNX Runtime's quantizer, compared with
  TensorRT's explicit-quantization mode on the same models.
- **Exit:**
  - GPT-2 perplexity on WikiText-2 within 0.5% of PyTorch;
  - every table cell is reproducible from `bench/results/` with one command;
  - TensorRT's engine build time is compared with `ptxc compile`.

## M7: Calibration (stretch)

- A `Calibrator` trait with min-max, percentile and entropy implementations,
  so ptxc quantizes float models itself.
- **Exit:** the int8 accuracy of ResNet-50 and BERT, calibrated by ptxc, is
  within 0.5 points of ORT's quantizer on the same calibration set.

## Later

Dynamic shapes, ViT, the Stable Diffusion UNet, an e-graph rewrite engine,
more architectures (sm_80, sm_90).
