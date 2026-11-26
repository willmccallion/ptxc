//! The sweep's choice of the fastest algorithm and, on a GPU, a problem
//! run end to end and the check that catches a run writing nothing.

#![allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]

use super::fastest;
use super::problems::{GemmParams, params, problems};
use crate::clocks::{Celsius, GpuState, Megahertz};
use crate::results::Case;
use crate::timing::Series;
use ptxc_core::Microseconds;

/// A case of cuBLASLt's `algorithm` whose one sample is `median_us`.
fn case(algorithm: u8, median_us: f64) -> Case<GemmParams> {
    let state = GpuState {
        sm: Megahertz(2520),
        memory: Megahertz(9001),
        temperature: Celsius(60),
        events: Vec::new(),
    };
    let series = Series { samples: vec![Microseconds::new(median_us)], batch: 1 };
    let params = params(problems()[0], Some(algorithm));
    Case::new("gemm".to_owned(), params, true, (state.clone(), state), &series).unwrap()
}

fn algorithm(case: Option<Case<GemmParams>>) -> Option<u8> {
    case.and_then(|case| case.params.algorithm)
}

#[test]
fn the_fastest_case_has_the_lowest_median() {
    let cases = vec![case(0, 9.0), case(1, 4.0), case(2, 6.0)];

    assert_eq!(algorithm(fastest(cases)), Some(1));
}

#[test]
fn the_first_of_tied_cases_is_the_fastest() {
    let cases = vec![case(0, 5.0), case(1, 5.0)];

    assert_eq!(algorithm(fastest(cases)), Some(0));
}

#[test]
fn no_cases_have_no_fastest() {
    assert_eq!(fastest(Vec::new()), None);
}

#[test]
fn each_type_compiles_its_own_product_transposing_what_its_layout_stores_transposed() {
    use super::data::Inputs;
    use super::frameworks::graph;
    use super::problems::id;
    use ptxc_core::DType;
    use ptxc_graph::Op;

    let built = |name: &str| {
        let problem = problems().into_iter().find(|&problem| id(problem) == name).unwrap();
        let graph = graph(problem, &Inputs::new(problem).unwrap()).unwrap();
        let ops: Vec<&'static str> = graph
            .nodes()
            .map(|(_, node)| match node.op() {
                Op::Transpose(_) => "transpose",
                Op::QMatMul(_) => "qmatmul",
                Op::MatMulInteger(_) => "matmul_integer",
                Op::Gemm(_) => "gemm",
                _ => "other",
            })
            .collect();
        let output = graph.value(*graph.outputs().iter().next().unwrap()).ty().dtype();
        (ops, output)
    };

    assert_eq!(built("gemm/e4m3/NN/256x256x256"), (vec!["qmatmul"], DType::F32));
    assert_eq!(built("gemm/e4m3/NT/256x256x256"), (vec!["transpose", "qmatmul"], DType::F32));
    assert_eq!(built("gemm/i8/TN/256x256x256"), (vec!["transpose", "matmul_integer"], DType::I32));
    assert_eq!(built("gemm/f16/NT/256x256x256"), (vec!["gemm"], DType::F16));
}

#[test]
fn a_quantized_product_reads_its_transposed_operand_on_tensor_cores_in_one_kernel() {
    use super::data::Inputs;
    use super::frameworks::graph;
    use super::problems::id;
    use ptxc_compiler::Compilation;
    use ptxc_target::{SmVersion, preset};

    let layouts = ["e4m3/NT", "e4m3/TN", "i8/NT", "i8/TN"];
    for name in layouts.map(|layout| format!("gemm/{layout}/256x256x256")) {
        let problem = problems().into_iter().find(|&problem| id(problem) == name).unwrap();
        let graph = graph(problem, &Inputs::new(problem).unwrap()).unwrap();
        let target = preset(SmVersion::SM_89).unwrap();

        let compiled =
            Compilation::from_graph(graph, target).optimize().unwrap().partition().unwrap();
        let program = compiled.lower().unwrap().schedule().unwrap().emit().unwrap();

        let [kernel] = &program.kernels().iter().collect::<Vec<_>>()[..] else {
            panic!("{name}: {} kernels", program.kernels().len())
        };
        assert!(kernel.ptx.contains("mma.sync"), "{name} is not tiled on tensor cores");
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use crate::cublas::{Handle, lt};
    use crate::suites::gemm::data::Inputs;
    use crate::suites::gemm::frameworks::{Default, Operands};
    use crate::suites::gemm::problems::{id, problems};
    use crate::suites::gemm::{GemmArgs, checked_run, run};
    use crate::suites::testing::repository;
    use ptxc_cuda::{DeviceOrdinal, Driver};

    #[test]
    fn one_problem_is_checked_timed_and_written_for_every_framework() {
        let _gpu = crate::timing::exclusive_gpu();
        let dir = tempfile::tempdir().unwrap();
        repository(dir.path());
        let out = dir.path().join("results");
        let args = GemmArgs { out: out.clone(), only: Some("f32/NN/256x256x256".to_owned()) };
        let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        let stream = device.primary_context().stream().unwrap();
        let cublas = Handle::new(&stream).unwrap().version().unwrap();

        run(&args, dir.path()).unwrap();

        for (file, framework, version) in [
            ("gemm-ptxc.json", "ptxc", env!("CARGO_PKG_VERSION")),
            ("gemm-cublas.json", "cublas", cublas.as_str()),
            ("gemm-cublas-best-of-8.json", "cublas-best-of-8", cublas.as_str()),
        ] {
            let text = std::fs::read_to_string(out.join(file)).unwrap();
            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(json["framework"]["name"], framework);
            assert_eq!(json["framework"]["version"], version, "{file}");
            let cases = json["cases"].as_array().unwrap();
            assert_eq!(cases.len(), 1, "{file}");
            assert_eq!(cases[0]["id"], "gemm/f32/NN/256x256x256");
            assert_eq!(cases[0]["verified"], true);
            assert!(cases[0]["median_us"].as_f64().unwrap() > 0.0);
        }
    }

    #[test]
    fn ptxcs_e4m3_and_int8_rows_pass_their_checks_in_every_layout() {
        use crate::suites::gemm::data::check;
        use crate::suites::gemm::frameworks::ptxc;

        let _gpu = crate::timing::exclusive_gpu();
        let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        let stream = device.primary_context().stream().unwrap();
        let quantized = problems().into_iter().filter(|&problem| {
            let name = id(problem);
            (name.contains("/e4m3/") || name.contains("/i8/")) && name.ends_with("/256x256x256")
        });

        for problem in quantized {
            let inputs = Inputs::new(problem).unwrap();
            let executor = ptxc(problem, &device, &inputs).unwrap();
            let bound = executor.bind(&[inputs.a.clone(), inputs.b.clone()]).unwrap();
            bound.enqueue(&stream).unwrap();
            stream.synchronize().unwrap();
            let outputs = bound.outputs().unwrap();

            assert_eq!(check(problem, outputs[0].data()), Ok(()), "{}", id(problem));
        }
    }

    #[test]
    fn a_cublas_run_that_writes_nothing_fails_its_check() {
        let _gpu = crate::timing::exclusive_gpu();
        let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        let context = device.primary_context();
        let stream = context.stream().unwrap();
        let problem = problems().into_iter().find(|&p| id(p) == "gemm/f32/NN/256x256x256").unwrap();
        let inputs = Inputs::new(problem).unwrap();
        let mut operands = Operands::new(&device, problem, &inputs).unwrap();
        let workspace = context.alloc(lt::MAX_WORKSPACE).unwrap();
        let mut default = Default::new(&stream, problem).unwrap().unwrap();
        let mut writes = |operands: &Operands| default.enqueue(problem, operands, &workspace);

        let written = checked_run(problem, &stream, &mut operands, &mut writes).unwrap();
        let unwritten = checked_run(problem, &stream, &mut operands, &mut |_: &Operands| Ok(()));

        assert_eq!(written, Ok(()));
        assert!(unwritten.unwrap().is_err());
    }
}
