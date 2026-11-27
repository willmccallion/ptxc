use crate::{graph, run};
use ptxc_core::{DType, Dim, Evaluate, HostTensor, Shape, TensorData};
use ptxc_graph::Orientation;
use ptxc_graph::text::print;
use ptxc_graph_opt::{FoldQdq, GreedyPartitioner, Partitioner, PassManager, PassReport};
use ptxc_interp::Interpreter;
use ptxc_testkit::compare::{Tolerance, compare};

const CHANGED: PassReport = PassReport { changed: true };
const UNCHANGED: PassReport = PassReport { changed: false };

/// A quantized tensor's element type and grid, as text.
#[derive(Clone, Copy)]
struct Grid {
    dtype: &'static str,
    params: &'static str,
}

/// Constant quantized values and their grid, as text.
#[derive(Clone, Copy)]
struct Constant {
    grid: Grid,
    values: &'static str,
}

const U8_INPUT: Grid = Grid { dtype: "u8", params: "[u8, [0.05, 128]]" };
const FP8_INPUT: Grid = Grid { dtype: "f8e4m3fn", params: "[f8e4m3fn, [0.5, 0]]" };
const PRODUCT_GRID: Grid = Grid { dtype: "u8", params: "[u8, [0.1, 120]]" };
const BY_COLUMN: &str = "[i8, 1, [[0.02, 0], [0.03, 0], [0.01, 0]]]";
const BY_ROW: &str = "[i8, 0, [[0.02, 0], [0.03, 0], [0.01, 0]]]";
const I8_VALUES: &str = "[12, -7, 33, -50, 18, 4, 91, -2, -64, 7, 25, -11]";
const I8_WEIGHTS: Constant =
    Constant { grid: Grid { dtype: "i8", params: BY_COLUMN }, values: I8_VALUES };
const FP8_WEIGHTS: Constant = Constant {
    grid: Grid { dtype: "f8e4m3fn", params: "[f8e4m3fn, [0.25, 0]]" },
    values: "[1.0, -2.0, 0.5, 3.0, -1.5, 2.0, 4.0, -0.25, 1.0, -3.0, 0.75, 6.0]",
};
const F32: Tolerance = Tolerance::for_dtype(DType::F32);

/// A product of the input `%a`, `[2, 4]`, and constant `[4, 3]` weights,
/// both dequantized to f32 and multiplied as `%3`, then `tail`, the lines
/// after it.
fn product(a: Grid, b: Constant, tail: &str) -> String {
    format!(
        "\
graph(%a: {a_type}[2, 4]) {{
  %0 = const {b_type}[4, 3] {values}
  %1 = dequantize<params={a_grid}, output=f32> %a : f32[2, 4]
  %2 = dequantize<params={b_grid}, output=f32> %0 : f32[4, 3]
  %3 = matmul %1, %2 : f32[2, 3]
{tail}}}
",
        a_type = a.dtype,
        b_type = b.grid.dtype,
        values = b.values,
        a_grid = a.params,
        b_grid = b.grid.params,
    )
}

/// `product`'s tail quantizing the product onto `PRODUCT_GRID`, then
/// returning `returned`.
fn quantized(returned: &str) -> String {
    format!(
        "  %4 = quantize<params={grid}> %3 : u8[2, 3]\n  return {returned}\n",
        grid = PRODUCT_GRID.params
    )
}

/// Checks that the folded graph gives what the graph `text` gives on
/// `inputs` in the interpreter, within `tolerance`.
fn assert_matches_unfolded(folded: &str, text: &str, inputs: &[HostTensor], tolerance: Tolerance) {
    let evaluate = |text| Interpreter::new(&graph(text)).evaluate(inputs).unwrap();
    for (actual, expected) in evaluate(folded).iter().zip(&evaluate(text)) {
        compare(actual, expected, tolerance).unwrap();
    }
}

/// Checks that folding leaves the graph `text` as it was.
fn assert_unchanged(text: &str) {
    let (after, report) = run(&FoldQdq, text);

    assert_eq!(after, print(&graph(text)), "{text}");
    assert_eq!(report, UNCHANGED, "{text}");
}

fn tensor(dims: &[u64], data: TensorData) -> HostTensor {
    let shape = Shape::new(dims.iter().copied().map(Dim::new).collect()).unwrap();
    HostTensor::new(shape, data).unwrap()
}

fn activations() -> HostTensor {
    tensor(&[2, 4], vec![130_u8, 200, 50, 128, 255, 0, 100, 140].into())
}

#[test]
fn a_dequantize_matmul_quantize_chain_folds_into_one_requantized_product() {
    let text = product(U8_INPUT, I8_WEIGHTS, &quantized("%4"));

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: u8[2, 4]) {{
  %0 = const i8[4, 3] {I8_VALUES}
  %1 = qmatmul<a={a}, b={BY_COLUMN}, output={y}> %a, %0 : u8[2, 3]
  return %1
}}
",
        a = U8_INPUT.params,
        y = PRODUCT_GRID.params,
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
    assert_matches_unfolded(&after, &text, &[activations()], Tolerance::QUANTIZED);
}

#[test]
fn a_dequantized_product_folds_into_a_quantized_product_with_a_float_result() {
    let text = product(U8_INPUT, I8_WEIGHTS, "  return %3\n");

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: u8[2, 4]) {{
  %0 = const i8[4, 3] {I8_VALUES}
  %1 = qmatmul<a={a}, b={BY_COLUMN}, result=f32> %a, %0 : f32[2, 3]
  return %1
}}
",
        a = U8_INPUT.params,
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
    assert_matches_unfolded(&after, &text, &[activations()], F32);
}

#[test]
fn a_product_read_beside_its_quantization_keeps_its_float_result() {
    let text = product(U8_INPUT, I8_WEIGHTS, &quantized("%4, %3"));

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: u8[2, 4]) {{
  %0 = const i8[4, 3] {I8_VALUES}
  %1 = qmatmul<a={a}, b={BY_COLUMN}, result=f32> %a, %0 : f32[2, 3]
  %2 = quantize<params={y}> %1 : u8[2, 3]
  return %2, %1
}}
",
        a = U8_INPUT.params,
        y = PRODUCT_GRID.params,
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn fp8_operands_fold_into_a_float_product_and_leave_its_quantization() {
    let text = product(FP8_INPUT, FP8_WEIGHTS, &quantized("%4"));

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: f8e4m3fn[2, 4]) {{
  %0 = const f8e4m3fn[4, 3] {values}
  %1 = qmatmul<a={a}, b={b}, result=f32> %a, %0 : f32[2, 3]
  %2 = quantize<params={y}> %1 : u8[2, 3]
  return %2
}}
",
        values = FP8_WEIGHTS.values,
        a = FP8_INPUT.params,
        b = FP8_WEIGHTS.grid.params,
        y = PRODUCT_GRID.params,
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
}

#[test]
fn operands_on_an_integer_grid_and_an_fp8_grid_stay_unfolded() {
    assert_unchanged(&product(U8_INPUT, FP8_WEIGHTS, "  return %3\n"));
}

#[test]
fn a_left_grid_varying_by_row_folds_and_one_varying_along_the_sum_stays() {
    let by_row = Grid { dtype: "u8", params: "[u8, 0, [[0.05, 128], [0.1, 120]]]" };
    let along_sum = Grid {
        dtype: "u8",
        params: "[u8, 1, [[0.05, 128], [0.1, 120], [0.05, 120], [0.02, 128]]]",
    };

    let (_, report) = run(&FoldQdq, &product(by_row, I8_WEIGHTS, "  return %3\n"));

    assert_eq!(report, CHANGED);
    assert_unchanged(&product(along_sum, I8_WEIGHTS, "  return %3\n"));
}

/// A `Gemm` of the dequantized input `%a`, on `a_grid`, and constant `i8`
/// weights on a grid by output feature, `[2, 4]` and `[4, 3]` as the
/// product reads them; with `C`, an f32 vector, if `with_c`.
#[derive(Clone, Copy)]
struct Gemm {
    a: Orientation,
    b: Orientation,
    alpha: &'static str,
    beta: &'static str,
    a_grid: &'static str,
    with_c: bool,
}

impl Gemm {
    /// An unscaled `Gemm` reading `a` and `b` in their orientations.
    const fn oriented(a: Orientation, b: Orientation, a_grid: &'static str, with_c: bool) -> Self {
        Self { a, b, alpha: "1.0", beta: "1.0", a_grid, with_c }
    }

    fn text(&self) -> String {
        let word = |orientation| match orientation {
            Orientation::AsIs => "as_is",
            Orientation::Transposed => "transposed",
        };
        let a_dims = if self.a == Orientation::Transposed { "[4, 2]" } else { "[2, 4]" };
        let (b_dims, b_grid) = match self.b {
            Orientation::AsIs => ("[4, 3]", BY_COLUMN),
            Orientation::Transposed => ("[3, 4]", BY_ROW),
        };
        let (c_constant, c_operand) = if self.with_c {
            ("  %1 = const f32[3] [0.5, -1.0, 0.25]\n", ", %1")
        } else {
            ("", "")
        };
        format!(
            "\
graph(%a: u8{a_dims}) {{
  %0 = const i8{b_dims} {I8_VALUES}
{c_constant}  %2 = dequantize<params={a_grid}, output=f32> %a : f32{a_dims}
  %3 = dequantize<params={b_grid}, output=f32> %0 : f32{b_dims}
  %4 = gemm<a={a}, b={b}, alpha={alpha}, beta={beta}> %2, %3{c_operand} : f32[2, 3]
  return %4
}}
",
            a_grid = self.a_grid,
            a = word(self.a),
            b = word(self.b),
            alpha = self.alpha,
            beta = self.beta,
        )
    }
}

#[test]
fn a_gemm_transposing_its_weights_folds_through_a_transpose_and_adds_c_after() {
    let gemm = Gemm::oriented(Orientation::AsIs, Orientation::Transposed, U8_INPUT.params, true);
    let text = gemm.text();

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: u8[2, 4]) {{
  %0 = const i8[3, 4] {I8_VALUES}
  %1 = const f32[3] [0.5, -1.0, 0.25]
  %2 = transpose<permutation=[1, 0]> %0 : i8[4, 3]
  %3 = qmatmul<a={a}, b={BY_COLUMN}, result=f32> %a, %2 : f32[2, 3]
  %4 = add %3, %1 : f32[2, 3]
  return %4
}}
",
        a = U8_INPUT.params,
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
    assert_matches_unfolded(&after, &text, &[activations()], F32);
}

#[test]
fn a_gemm_transposing_both_operands_reads_both_through_a_transpose() {
    let by_column = "[u8, 1, [[0.05, 128], [0.1, 120]]]";
    let gemm = Gemm::oriented(Orientation::Transposed, Orientation::Transposed, by_column, false);
    let text = gemm.text();

    let (after, report) = run(&FoldQdq, &text);

    let expected = format!(
        "\
graph(%a: u8[4, 2]) {{
  %0 = const i8[3, 4] {I8_VALUES}
  %1 = transpose<permutation=[1, 0]> %a : u8[2, 4]
  %2 = transpose<permutation=[1, 0]> %0 : i8[4, 3]
  %3 = qmatmul<a=[u8, 0, [[0.05, 128], [0.1, 120]]], b={BY_COLUMN}, result=f32> %1, %2 : f32[2, 3]
  return %3
}}
"
    );
    assert_eq!(after, expected);
    assert_eq!(report, CHANGED);
    let transposed = tensor(&[4, 2], vec![130_u8, 255, 200, 0, 50, 100, 128, 140].into());
    assert_matches_unfolded(&after, &text, &[transposed], F32);
}

#[test]
fn a_gemm_folds_unless_it_scales_its_product_or_its_c() {
    let plain = Gemm::oriented(Orientation::AsIs, Orientation::AsIs, U8_INPUT.params, false);

    let (_, report) = run(&FoldQdq, &Gemm { beta: "0.5", ..plain }.text());

    assert_eq!(report, CHANGED);
    assert_unchanged(&Gemm { beta: "0.5", with_c: true, ..plain }.text());
    assert_unchanged(&Gemm { alpha: "2.0", ..plain }.text());
}

#[test]
fn a_folded_gemm_fuses_with_the_transposes_it_reads_and_the_c_it_adds() {
    let text = "\
graph(%a: u8[4, 2], %b: i8[3, 4], %c: f32[3]) {
  %0 = dequantize<params=[u8, [0.05, 128]], output=f32> %a : f32[4, 2]
  %1 = dequantize<params=[i8, 0, [[0.02, 0], [0.03, 0], [0.01, 0]]], output=f32> %b : f32[3, 4]
  %2 = gemm<a=transposed, b=transposed, alpha=1.0, beta=1.0> %0, %1, %c : f32[2, 3]
  return %2
}
";
    let mut folded = graph(text);
    let _reports = PassManager::standard().run(&mut folded).unwrap();

    let partitioned = GreedyPartitioner.partition(folded).unwrap();

    let groups: Vec<usize> = partitioned.groups().iter().map(|group| group.nodes().len()).collect();
    assert_eq!(groups, [4]);
}

/// A convolution, 2 by 2 and two channels to two, of the input `%x`,
/// `[1, 2, 3, 3]`, by constant `weights`, with a constant bias `[40, -25]`
/// on `bias` if any, all dequantized to f32, then quantized onto
/// `output`; returning `returned`.
#[derive(Clone, Copy)]
struct Conv {
    input: Grid,
    weights: Constant,
    bias: Option<Grid>,
    output: Grid,
    returned: &'static str,
}

const CONV_INPUT: Grid = Grid { dtype: "u8", params: "[u8, [0.0625, 128]]" };
const CONV_WEIGHTS: Constant = Constant {
    grid: Grid { dtype: "i8", params: "[i8, 0, [[0.03125, 0], [0.015625, 0]]]" },
    values: "[3, -5, 7, 2, -4, 6, 1, -8, 5, 2, -3, 9, -6, 4, 8, -1]",
};
const BIAS_ON_GRID: Grid =
    Grid { dtype: "i32", params: "[i32, 0, [[0.001953125, 0], [0.0009765625, 0]]]" };
const CONV_OUTPUT: Grid = Grid { dtype: "u8", params: "[u8, [0.05, 100]]" };
const CONV_AXES: &str = "axes=[[1, 1, 0, 0], [1, 1, 0, 0]]";

impl Conv {
    /// The chain on the usual grids, with `bias` if any.
    const fn on_grids(bias: Option<Grid>) -> Self {
        Self { input: CONV_INPUT, weights: CONV_WEIGHTS, bias, output: CONV_OUTPUT, returned: "%6" }
    }

    fn text(&self) -> String {
        let (bias_constant, bias_dequantized, bias_operand) = self.bias.map_or_else(
            || (String::new(), String::new(), ""),
            |bias| {
                (
                    format!("  %1 = const {}[2] [40, -25]\n", bias.dtype),
                    format!("  %4 = dequantize<params={}, output=f32> %1 : f32[2]\n", bias.params),
                    ", %4",
                )
            },
        );
        format!(
            "\
graph(%x: {x_type}[1, 2, 3, 3]) {{
  %0 = const {w_type}[2, 2, 2, 2] {values}
{bias_constant}  %2 = dequantize<params={x_grid}, output=f32> %x : f32[1, 2, 3, 3]
  %3 = dequantize<params={w_grid}, output=f32> %0 : f32[2, 2, 2, 2]
{bias_dequantized}  %5 = conv<groups=1, {CONV_AXES}> %2, %3{bias_operand} : f32[1, 2, 2, 2]
  %6 = quantize<params={y_grid}> %5 : {y_type}[1, 2, 2, 2]
  return {returned}
}}
",
            x_type = self.input.dtype,
            w_type = self.weights.grid.dtype,
            values = self.weights.values,
            x_grid = self.input.params,
            w_grid = self.weights.grid.params,
            y_grid = self.output.params,
            y_type = self.output.dtype,
            returned = self.returned,
        )
    }
}

fn image() -> HostTensor {
    let values =
        vec![130_u8, 200, 50, 128, 255, 0, 100, 140, 90, 60, 170, 128, 30, 220, 110, 145, 75, 190];
    tensor(&[1, 2, 3, 3], values.into())
}

#[test]
fn a_dequantize_conv_quantize_chain_folds_into_one_quantized_convolution() {
    for (bias, bias_constant, bias_operand, output) in [
        (Some(BIAS_ON_GRID), "  %1 = const i32[2] [40, -25]\n", ", %1", "%2"),
        (None, "", "", "%1"),
    ] {
        let text = Conv::on_grids(bias).text();

        let (after, report) = run(&FoldQdq, &text);

        let expected = format!(
            "\
graph(%x: u8[1, 2, 3, 3]) {{
  %0 = const i8[2, 2, 2, 2] {values}
{bias_constant}  {output} = qconv<groups=1, {CONV_AXES}, input={x}, weights={w}, output={y}> %x, %0{bias_operand} : u8[1, 2, 2, 2]
  return {output}
}}
",
            values = CONV_WEIGHTS.values,
            x = CONV_INPUT.params,
            w = CONV_WEIGHTS.grid.params,
            y = CONV_OUTPUT.params,
        );
        assert_eq!(after, expected);
        assert_eq!(report, CHANGED);
        assert_matches_unfolded(&after, &text, &[image()], Tolerance::QUANTIZED);
    }
}

#[test]
fn a_bias_off_the_grid_of_the_integer_sum_stays_unfolded() {
    for bias in [
        Grid { dtype: "i32", params: "[i32, 0, [[0.002, 0], [0.0009765625, 0]]]" },
        Grid { dtype: "i32", params: "[i32, 0, [[0.001953125, 0], [0.0009765625, 1]]]" },
        Grid { dtype: "i32", params: "[i32, [0.001953125, 0]]" },
        Grid { dtype: "i8", params: "[i8, 0, [[0.001953125, 0], [0.0009765625, 0]]]" },
    ] {
        assert_unchanged(&Conv::on_grids(Some(bias)).text());
    }
}

#[test]
fn a_convolution_on_or_onto_an_fp8_grid_stays_unfolded() {
    let fp8 = Grid { dtype: "f8e4m3fn", params: "[f8e4m3fn, [0.5, 0]]" };
    let fp8_weights = Constant {
        grid: Grid { dtype: "f8e4m3fn", params: "[f8e4m3fn, [0.25, 0]]" },
        values: "[1.0, -2.0, 0.5, 3.0, -1.5, 2.0, 4.0, -0.25, 1.0, -3.0, 0.75, 6.0, 0.5, -1.0, 2.0, -0.5]",
    };
    let usual = Conv::on_grids(None);

    for conv in [
        Conv { input: fp8, ..usual },
        Conv { weights: fp8_weights, ..usual },
        Conv { output: fp8, ..usual },
    ] {
        assert_unchanged(&conv.text());
    }
}

#[test]
fn a_convolution_read_beside_its_quantization_stays_unfolded() {
    assert_unchanged(&Conv { returned: "%6, %5", ..Conv::on_grids(None) }.text());
}

#[test]
fn the_standard_passes_fold_quantization_before_constant_folding_dequantizes_the_weights() {
    for (text, folded_op) in [
        (product(U8_INPUT, I8_WEIGHTS, &quantized("%4")), "qmatmul"),
        (Conv::on_grids(Some(BIAS_ON_GRID)).text(), "qconv"),
    ] {
        let mut folded = graph(&text);

        let _reports = PassManager::standard().run(&mut folded).unwrap();

        let ops: Vec<&str> = folded.nodes().map(|(_, node)| node.op().mnemonic()).collect();
        assert_eq!(ops, [folded_op]);
    }
}
