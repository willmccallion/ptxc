"""`ptxc-baselines bench-conv`: PyTorch with cuDNN on ResNet-50's convolution
layers, the baseline of `ptxc-bench conv` (BENCHMARKING.md). Both read
`bench/conv.json`, so they time the same problems.

cuDNN picks each layer's algorithm with `cudnn.benchmark = True`. Each layer
is captured in a CUDA graph and replayed, so a sample times the GPU's work,
not Python's launch overhead. It runs with the input and weights NCHW, as
ptxc takes them, and in channels-last memory, cuDNN's documented best for
tensor cores; each writes its own file. Every result is checked before it is
timed: inputs and weights are multiples of 1/8 in [-1, 1], whose products
f32 sums exactly, and 256 evenly spaced outputs must match a float64
convolution within f16's tolerance for the layer's sum, the tolerance
`ptxc-testkit` holds ptxc to."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import gpu, paths, results, timing

#: The suite's definition, which `ptxc-bench conv` reads too.
DEFINITION = paths.ROOT / "bench" / "conv.json"

#: How many output elements a check compares.
SAMPLES = 256

#: f16's tolerance before a reduction widens it, as `ptxc-testkit` has it.
ATOL = 1e-3
RTOL = 1e-2

#: The seed of each problem's input and weights.
SEED = 0


@dataclass(frozen=True)
class Layer:
    """One NCHW convolution layer, square: input channels, output channels
    (`features`), input size, kernel size, stride and padding on each side."""

    name: str
    channels: int
    features: int
    size: int
    kernel: int
    stride: int
    pad: int

    @property
    def output(self) -> int:
        return (self.size + 2 * self.pad - self.kernel) // self.stride + 1

    @property
    def depth(self) -> int:
        """The products each output element sums."""
        return self.channels * self.kernel * self.kernel


@dataclass(frozen=True)
class Problem:
    """A layer at a batch size, in the suite's element type."""

    layer: Layer
    batch: int
    dtype: str

    @property
    def id(self) -> str:
        return f"conv/{self.dtype}/{self.layer.name}/b{self.batch}"

    def params(self) -> dict:
        """The parameters a result records, as `ptxc-bench conv` does."""
        layer = self.layer
        return {
            "layer": layer.name,
            "batch": self.batch,
            "channels": layer.channels,
            "features": layer.features,
            "size": layer.size,
            "kernel": layer.kernel,
            "stride": layer.stride,
            "pad": layer.pad,
            "dtype": self.dtype,
        }


def problems(definition: Path = DEFINITION) -> list[Problem]:
    """Each layer of `definition` at each of its batch sizes."""
    suite = json.loads(definition.read_text())
    if suite["dtype"] != "f16":
        raise ValueError(f"the suite runs f16, not {suite['dtype']}")
    layers = [Layer(**layer) for layer in suite["layers"]]
    return [Problem(layer, batch, suite["dtype"]) for layer in layers for batch in suite["batches"]]


def positions(total: int) -> list[int]:
    """`SAMPLES` evenly spaced flat positions below `total`, or all of them."""
    count = min(SAMPLES, total)
    return [sample * total // count for sample in range(count)]


def check(problem: Problem, actual, expected) -> str | None:
    """How the NCHW output `actual` differs from the float64 `expected` at
    evenly spaced elements, beyond f16's tolerance for the layer's sum; or
    `None` if it does not."""
    import torch

    at = torch.tensor(positions(expected.numel()), dtype=torch.long)
    got = actual.reshape(-1).cpu()[at].double()
    want = expected.reshape(-1)[at].to(torch.float16).double()
    widening = math.sqrt(max(problem.layer.depth, 1))
    allowed = ATOL * widening + RTOL * widening * want.abs()
    misses = ((got - want).abs() > allowed) | got.isnan()
    if not bool(misses.any()):
        return None
    worst = int(((got - want).abs() - allowed).argmax())
    return (
        f"{int(misses.sum())} of {len(at)} elements differ; at flat position {int(at[worst])}: "
        f"{float(got[worst])} vs expected {float(want[worst])}"
    )


def operands(problem: Problem):
    """The input and weights as float64 multiples of 1/8 in [-1, 1]."""
    import torch

    layer = problem.layer
    generator = torch.Generator().manual_seed(SEED)

    def eighths(*shape: int):
        return torch.randint(-8, 9, shape, generator=generator).double() / 8

    x = eighths(problem.batch, layer.channels, layer.size, layer.size)
    w = eighths(layer.features, layer.channels, layer.kernel, layer.kernel)
    return x, w


#: Each layout timed: the framework's name in the results, and the memory
#: format of the input, weights and output.
LAYOUTS = ["pytorch", "pytorch-channels-last"]


def captured(problem: Problem, x, w):
    """The layer's convolution of `x` by `w` on the GPU, captured in a CUDA
    graph after cuDNN has chosen its algorithm, and the graph's output."""
    import torch
    import torch.nn.functional as functional

    layer = problem.layer

    def convolve():
        return functional.conv2d(x, w, stride=layer.stride, padding=layer.pad)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            convolve()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = convolve()
    return graph, out


def run_problem(problem: Problem, layout: str, reference) -> dict | str:
    """`problem` in `layout`, replayed once and checked against `reference`,
    then timed; or how its check failed."""
    import torch

    x64, w64, expected = reference
    memory = torch.channels_last if layout == "pytorch-channels-last" else torch.contiguous_format
    x = x64.to("cuda", torch.float16).contiguous(memory_format=memory)
    w = w64.to("cuda", torch.float16).contiguous(memory_format=memory)
    graph, out = captured(problem, x, w)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    failure = check(problem, out, expected)
    if failure is not None:
        return failure
    before = gpu.state()
    series = timing.time(graph.replay)
    after = gpu.state()
    return results.case(problem.id, problem.params(), (before, after), series)


def run(out: Path, only: str | None) -> list[str]:
    """Times every selected problem in each layout and writes each layout's
    results under `out`; the checks that failed."""
    gpu.ensure_idle()
    import torch
    import torch.nn.functional as functional

    torch.backends.cudnn.benchmark = True
    version = f"{torch.__version__} cudnn {torch.backends.cudnn.version()}"
    selected = [problem for problem in problems() if only is None or only in problem.id]
    cases: dict[str, list[dict]] = {layout: [] for layout in LAYOUTS}
    failed = []
    for index, problem in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] {problem.id}", flush=True)
        x64, w64 = operands(problem)
        layer = problem.layer
        expected = functional.conv2d(x64, w64, stride=layer.stride, padding=layer.pad)
        for layout in LAYOUTS:
            ran = run_problem(problem, layout, (x64, w64, expected))
            if isinstance(ran, str):
                failed.append(f"{problem.id} {layout}: {ran}")
            else:
                cases[layout].append(ran)
    environment = gpu.identity()
    for layout in LAYOUTS:
        framework = results.framework(layout, version, paths.ROOT)
        results.write(out / f"conv-{layout}.json", "conv", framework, environment, cases[layout])
    return failed
