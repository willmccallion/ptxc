# Benchmarking

Every published number is reproducible from committed code, a recorded
environment, and one command. Baselines get their best documented
configuration. Where ptxc loses, the table says so.

## Machine

Recorded 2026-11-22:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, compute capability 8.9 (Ada) |
| Memory | 8188 MiB |
| Max clocks | SM 3105 MHz, memory 9001 MHz |
| Power limit | 160 W |
| PCIe | gen 4 ×8 |
| Driver | 595.112.07 (CUDA 13.2 driver API) |
| CUDA toolkit | none installed system-wide; ptxas 13.5.37 comes from the flake, test-only |
| Rust | rustc 1.99.1, cargo 1.99.0 |

Each results file records these values again at run time, so a driver
upgrade shows up in the data.

## Methodology

- **Clocks.** Locked by hand around a published series (see Clock
  locking below). The harness records the actual SM and memory clocks,
  temperature and throttle reasons (`nvidia-smi --query-gpu`) before and
  after each timed series, once its warmup has run, and marks a series
  invalid if the clocks moved.
- **Timing.**
  - A warmup of at least 20 iterations and 1 s, each between CUDA events on
    one stream.
  - Then at least 100 timed samples, or 2 s, whichever is longer. A sample
    is a batch of back-to-back launches between two events, its size the
    smallest power of two the warmup's median says lasts 1 ms, and records
    the time per launch. Short kernels are timed in batches, so the events'
    0.5 µs resolution and the gaps between launches stay small against
    each sample.
  - Reported as the median, with p10 and p90 alongside, interpolated
    linearly between the nearest ranks as numpy's default percentile is.
    Raw samples are kept in the JSON.
- **Whole models.** All frameworks are timed the same way:
  - from inputs already on the device to outputs on the device;
  - one stream, with CUDA graphs where the framework supports them (ptxc
    `Executor<Graphed>`, torch.compile `mode="max-autotune"`, ORT with CUDA
    graphs enabled, TensorRT enqueue);
  - with host-to-device and device-to-host copies reported separately.
- **Fairness.**
  - The same ONNX file feeds ptxc, ORT and TensorRT. PyTorch runs the module
    the ONNX file was exported from.
  - Same precision: f16 means f16 storage with f32 accumulation everywhere.
    TF32 is off for f32 baselines unless a row says otherwise.
  - Quantized rows use the same QDQ model for ptxc and TensorRT's explicit
    quantization.
  - Every library version is recorded.
- **Noise.** Desktop processes are idle. No other GPU work runs during a
  series; the harness checks `nvidia-smi` for other compute processes and
  refuses to start if there are any.

## Clock locking

Run these by hand before and after a published series. They need root.

```sh
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40            # no sudo: list supported clocks
sudo nvidia-smi --persistence-mode=1                    # keep the driver loaded between runs
sudo nvidia-smi --lock-gpu-clocks=<MHz>,<MHz>           # pin the SM clock (min = max)
sudo nvidia-smi --lock-memory-clocks=<MHz>,<MHz>        # may be unsupported on GeForce; the harness records it either way

# afterwards
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi --reset-memory-clocks
sudo nvidia-smi --persistence-mode=0
```

The SM frequency is chosen once at M3: the highest clock this card holds
through a 60 s tensor-core GEMM without power or thermal throttling.

Recorded 2026-11-24: **2700 MHz**. Locked there, back-to-back f16
8192³ GEMMs, ptxc's and cuBLAS's, ran for 80 s with `nvidia-smi` sampling
every 0.5 s. Every busy sample read 2700 MHz with no throttle reason, at
up to 147 W of the 160 W limit and 68 °C.

The same load under a lock of 2775, 2850, 2925 or 3000 MHz ran below the
lock in every busy sample. It never fell under 2715 MHz, and it never
passed 152 W or 69 °C. With the SM clock locked, the memory clock reads
8751 MHz.

## Suites

### GEMM against cuBLAS (`ptxc-bench gemm`, M3)

- **Shapes:**
  - squares M = N = K in {256, 512, 1024, 2048, 4096, 8192};
  - the transformer shapes of BERT-base and GPT-2: M = batch × sequence
    length in {128, 1024, 4096} (batch 1, 8 and 32 at sequence length 128),
    with (N, K) in {(768, 768), (3072, 768), (768, 3072), (2304, 768)};
  - skinny shapes with M in {1, 4, 16}, as in decoding, with the same N
    and K;
  - squares no power-of-two tile divides: 1000 and 3000.
- **Data types:** f16→f32, bf16→f32, E4M3→f32, int8→i32, f32 (SIMT).
- **Layouts:** NN, NT, TN.
- **Baseline:** cuBLAS (cuBLASLt for FP8 and int8), loaded with dlopen by
  `ptxc-bench` only, with its default heuristic algorithm. The best of its
  top 8 algorithms is also recorded, as an upper bound.
  - On sm_89, cuBLASLt runs E4M3 only as NT: A transposed and B as is, in
    its column-major terms. The FP8 NN and TN rows have no baseline.
- **Metrics:** TFLOP/s (or TOP/s) and the percentage of cuBLAS. Every result
  is checked against a reference before it is timed.
  - Inputs are multiples of 1/8 in [-1, 1] (integers in [-8, 8] for int8),
    which every type holds exactly. Their products sum exactly in f32, so
    256 evenly spaced output elements recomputed on the host match within
    the output type's tolerance for a K-long sum.
  - A result that fails its check is reported and not timed, and the run
    fails.
- **Frameworks:** each writes its own file: `gemm-ptxc.json`,
  `gemm-cublas.json` (the default heuristic: `GemmEx`, or cuBLASLt's first
  algorithm for E4M3 and int8) and `gemm-cublas-best-of-8.json`. A cuBLAS
  file records cuBLAS's version; its `commit` is the harness's.
- ptxc compiles each problem from an ONNX graph: a `Gemm` with A and B
  transposed as the layout stores them for the float types; a
  `MatMulInteger` for int8 to i32; and, for E4M3, the quantized matmul
  with unit scales and an f32 result that `DequantizeLinear` feeding
  `MatMul` folds into (D72). Neither quantized product takes a transpose,
  so a transposed A or B is a `Transpose` node before it, which runs as a
  kernel of its own, and is timed with the product, until fusion (M4)
  folds it into the product's reads (#176).

### Convolution (`ptxc-bench conv`, M3)

ResNet-50's distinct convolution layers at batch sizes 1, 8 and 32, against
PyTorch with cuDNN, with `cudnn.benchmark = True`.

- **Layers:** ResNet-50 v1.5's 23 distinct convolutions at a 224 by 224
  input: the stem, and each stage's 1 by 1, 3 by 3, strided 3 by 3 and
  strided projection layers. `bench/conv.json` lists them with the batch
  sizes and the element type, and both frameworks read it.
- **Data type:** f16 input, weights and output, summed in f32. No bias.
- **Layout:** NCHW input and output, OIHW weights, as ONNX's `Conv` takes
  them; ptxc compiles each layer as one.
- **Check:** inputs and weights are multiples of 1/8 in [-1, 1], whose
  products f32 sums exactly. 256 evenly spaced output elements recomputed
  on the host match within f16's tolerance for the layer's sum. A result
  that fails its check is reported and not timed, and the run fails.
- **Baseline:** PyTorch with cuDNN, `cudnn.benchmark = True`, run from
  `baselines/` on the same layers. Each layer is captured in a CUDA graph
  after cuDNN has chosen its algorithm and replayed, so a sample times the
  GPU's work rather than Python's launch overhead. It runs NCHW, as ptxc
  takes the layer, and channels-last, cuDNN's documented best layout for
  tensor cores. Its result is checked as ptxc's is, against a float64
  convolution, within the same tolerance.
- **Metric:** time per layer, and TFLOP/s at two operations per
  multiply-add of the direct convolution.
- **Files:** `conv-ptxc.json`, `conv-pytorch.json` (NCHW) and
  `conv-pytorch-channels-last.json`. A PyTorch file's version names
  PyTorch's and cuDNN's; its `commit` is the harness's.

### Whole models (`ptxc-bench models`, M4 to M6)

| Model | Batch sizes | Baselines |
|---|---|---|
| MLP | 1, 64, 1024 | PyTorch eager, torch.compile, ORT CUDA EP, TensorRT |
| ResNet-50 | 1, 8, 32, 64 | same |
| BERT-base, sequence length 128 | 1, 8, 32 | same |
| GPT-2 124M, prefill at 128 and 512 tokens | 1, 8 | same |
| Later: ViT-B/16, SD UNet | | |

Reported: latency (median, p10, p90) and throughput (samples per second).

### Tuning time and build time (M5, M6)

- **Tuning:** ptxc against TVM/Ansor on the same models, each given the same
  trial budget (1,000, 5,000 and 20,000 trials). Reported: wall-clock tuning
  time and the latency reached.
- **Build:** `ptxc compile` (with an existing tuning database) against
  TensorRT's engine build (`trtexec`), as wall-clock time.

## Layout and commands

```
crates/bench/                    Rust harness (ptxc and cuBLAS)
baselines/ptxc_baselines/        PyTorch, torch.compile, ORT, TensorRT runners; render_tables.py
bench/results/<date>-<commit>/   committed at milestones: one JSON file per suite and framework
bench/results/scratch/           gitignored, for runs during development
```

```sh
ptxc-bench gemm --out bench/results/scratch/              # --only f16/NN for a subset
ptxc-bench conv --out bench/results/scratch/              # --only b32 or --only layer1 for a subset
uv run --project baselines ptxc-baselines bench-conv --out bench/results/scratch/
ptxc-bench models --model resnet50 --batch 1,8,32 --out bench/results/scratch/
uv run --project baselines ptxc-baselines bench --framework trt --model resnet50 --batch 1,8,32 --out bench/results/scratch/
uv run --project baselines ptxc-baselines render --results bench/results/<run>/ --readme README.md
```

## Results schema

One file per suite and framework:

```json
{
  "schema": 1,
  "suite": "gemm",
  "framework": { "name": "ptxc", "version": "0.1.0", "commit": "abc1234def56", "dirty": false },
  "environment": { "gpu": "NVIDIA GeForce RTX 4060 Ti", "driver": "595.112.07" },
  "cases": [
    {
      "id": "gemm/f16/NN/4096x4096x4096",
      "params": { "m": 4096, "n": 4096, "k": 4096, "dtype": "f16", "layout": "NN" },
      "verified": true,
      "iterations_per_sample": 256,
      "clocks": {
        "before": { "sm_mhz": 2520, "mem_mhz": 9001, "temperature_c": 52, "events": [] },
        "after": { "sm_mhz": 2520, "mem_mhz": 9001, "temperature_c": 61, "events": ["sw_power_cap"] },
        "held": true
      },
      "samples_us": [...],
      "median_us": 0.0, "p10_us": 0.0, "p90_us": 0.0
    }
  ]
}
```

- `dirty` records uncommitted changes in the working tree; published
  results come only from committed code.
- `clocks` is the GPU's state just before and just after the case's
  timed series, after its warmup. `events` are the reasons `nvidia-smi`
  gives for holding the clocks down, named as its `clocks_event_reasons`
  fields are. `held` is whether the SM and memory clocks were the same
  before and after.
- Whether clocks were locked is not recorded: the harness cannot read it.
  Clocks that held are what locking is for.

`render_tables.py` joins a run's files on `id` (D77).
- It rewrites each suite's summary in the README, between
  `<!-- bench:<suite>:begin -->` and `<!-- bench:<suite>:end -->` markers:
  - GEMM: ptxc's geometric-mean percentage of cuBLAS's throughput, against
    its default algorithm and its best of 8, for each type and layout, with
    the problems ptxc wins and its slowest, then the M3 exit's metric;
  - convolution: the same against cuDNN in NCHW and channels-last, for
    each batch size.
- It writes every case, with its time, its throughput and each baseline's,
  to `<suite>.md` beside the results, under the GPU, driver, SM clock and
  each framework's version and commit, and the summary links to it.
- It refuses the whole run if any file comes from uncommitted changes or
  another schema, or holds a case that is unverified or whose clocks moved
  (`held` false). A series during which the GPU reported a power or
  thermal cap, its clocks held, is marked †.
