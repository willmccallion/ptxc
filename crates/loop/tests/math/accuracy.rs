//! Each program's largest error, in ulps of the f32 result.

use ptxc_core::convert::f64_to_f32;
use ptxc_loop::math::{self, MathProgram};
use std::num::NonZeroUsize;
use std::thread;

const SAMPLE_SEED: u64 = 0x0E8B_10C5;
const SAMPLE_COUNT: usize = 1_000_000;

/// A program, its f64 reference, and the most ulps it may be off: the
/// largest error over every f32, rounded up.
struct Function {
    name: &'static str,
    program: fn() -> &'static MathProgram,
    reference: fn(f64) -> f64,
    bound_ulps: f64,
}

const FUNCTIONS: [Function; 8] = [
    Function { name: "exp", program: math::exp, reference: f64::exp, bound_ulps: 1.06 },
    Function { name: "log", program: math::log, reference: f64::ln, bound_ulps: 0.84 },
    Function { name: "expm1", program: math::expm1, reference: f64::exp_m1, bound_ulps: 1.65 },
    Function { name: "log1p", program: math::log1p, reference: f64::ln_1p, bound_ulps: 1.44 },
    Function { name: "tanh", program: math::tanh, reference: f64::tanh, bound_ulps: 2.42 },
    Function { name: "erf", program: math::erf, reference: libm::erf, bound_ulps: 1.03 },
    Function { name: "sin", program: math::sin, reference: f64::sin, bound_ulps: 2.25 },
    Function { name: "cos", program: math::cos, reference: f64::cos, bound_ulps: 2.26 },
];

/// The distance from `actual` to the exact `reference`, in ulps of f32 at
/// `reference`; zero when both are the same infinity or both NaN.
fn ulps(actual: f32, reference: f64) -> f64 {
    let rounded = f64_to_f32(reference);
    if actual.is_nan() || rounded.is_nan() {
        return if actual.is_nan() && rounded.is_nan() { 0.0 } else { f64::INFINITY };
    }
    if actual.is_infinite() || rounded.is_infinite() {
        return if actual == rounded { 0.0 } else { f64::INFINITY };
    }
    let exponent = reference.abs().log2().floor().max(-126.0);
    (f64::from(actual) - reference).abs() / (exponent - 23.0).exp2()
}

/// The largest error over `inputs`, with the input that gives it.
fn worst(
    program: &MathProgram,
    reference: fn(f64) -> f64,
    inputs: impl Iterator<Item = u32>,
) -> (f64, f32) {
    inputs
        .map(f32::from_bits)
        .map(|x| (ulps(program.eval(x), reference(f64::from(x))), x))
        .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a })
}

/// xorshift64 over every f32 bit pattern.
fn sample(seed: u64, count: usize) -> impl Iterator<Item = u32> {
    let mut state = seed;
    std::iter::repeat_with(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        u32::try_from(state >> 32).unwrap()
    })
    .take(count)
}

fn check(function: &Function, worst_case: (f64, f32)) {
    let (error, input) = worst_case;
    let name = function.name;
    eprintln!("{name}: {error:.3} ulps at {input:e} ({:#010x})", input.to_bits());
    assert!(error <= function.bound_ulps, "{name} is {error} ulps off at {input:e}");
}

#[test]
fn sampled_inputs_are_within_each_bound() {
    for function in &FUNCTIONS {
        let program = (function.program)();
        check(function, worst(program, function.reference, sample(SAMPLE_SEED, SAMPLE_COUNT)));
    }
}

/// `function`'s largest error over every f32, on every core.
fn check_every_input(function: &Function) {
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let total = 1_u64 << 32;
    let chunk = total.div_ceil(u64::try_from(threads).unwrap());
    let program = (function.program)();
    let worst_case = thread::scope(|scope| {
        let mut workers = Vec::new();
        for start in (0..total).step_by(usize::try_from(chunk).unwrap()) {
            let end = (start + chunk).min(total);
            let bits = (start..end).map(|n| u32::try_from(n).unwrap());
            workers.push(scope.spawn(move || worst(program, function.reference, bits)));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a })
    });
    check(function, worst_case);
}

/// One ignored test per function, so a run can name the ones that changed:
/// `cargo test --release -p ptxc-loop --test math -- --ignored every_sin`.
macro_rules! exhaustive {
    ($($test:ident: $index:expr),+ $(,)?) => {$(
        #[test]
        #[ignore = "exhaustive over every f32; run with --release -- --ignored"]
        fn $test() {
            check_every_input(&FUNCTIONS[$index]);
        }
    )+};
}

exhaustive!(
    every_exp_input: 0,
    every_log_input: 1,
    every_expm1_input: 2,
    every_log1p_input: 3,
    every_tanh_input: 4,
    every_erf_input: 5,
    every_sin_input: 6,
    every_cos_input: 7,
);
