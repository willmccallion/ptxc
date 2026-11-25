//! Times a launch with CUDA events, as BENCHMARKING.md's methodology says:
//! a warmup of at least 20 iterations and 1 s, then a series of at least
//! 100 samples and 2 s. A sample is a batch of back-to-back launches between
//! two events on one stream, sized in the warmup to last at least 1 ms, so
//! the events' resolution and the gaps between launches are a small part
//! of it; it records the time per launch. The GPU's state is read just
//! before and just after the series, once the warmup has brought it to the
//! state the series runs in.

use ptxc_core::Microseconds;
use ptxc_cuda::{Context, CudaError, Stream};
use std::time::{Duration, Instant};

/// How long a sample lasts at the least.
const SAMPLE_TIME: Microseconds = Microseconds::new(1000.0);

/// The most launches a sample batches, however short they are.
const MAX_BATCH: u32 = 100_000;

/// How many iterations, and for how long, a phase runs at the least.
#[derive(Clone, Copy, Debug)]
struct Budget {
    iterations: usize,
    time: Duration,
}

impl Budget {
    /// The warmup: at least 20 iterations and 1 s.
    const WARMUP: Self = Self { iterations: 20, time: Duration::from_secs(1) };

    /// The timed series: at least 100 samples and 2 s.
    const TIMED: Self = Self { iterations: 100, time: Duration::from_secs(2) };

    /// Whether `iterations` run over `elapsed` meet the budget.
    const fn is_met(self, iterations: usize, elapsed: Duration) -> bool {
        iterations >= self.iterations && elapsed.as_nanos() >= self.time.as_nanos()
    }
}

/// A timed series: each sample's time per launch, in order, and how many
/// launches each sample batched.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Series {
    pub(crate) samples: Vec<Microseconds>,
    pub(crate) batch: u32,
}

/// A timed series, and the GPU's state read just before and just after it.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Timed<S> {
    pub(crate) series: Series,
    pub(crate) before: S,
    pub(crate) after: S,
}

/// A series' median and spread.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Summary {
    pub(crate) median: Microseconds,
    pub(crate) p10: Microseconds,
    pub(crate) p90: Microseconds,
}

impl Series {
    /// The median, p10 and p90, interpolated linearly between the nearest
    /// ranks as numpy's default percentile is; `None` for an empty series.
    pub(crate) fn summary(&self) -> Option<Summary> {
        let mut sorted: Vec<f64> = self.samples.iter().map(|sample| sample.get()).collect();
        sorted.sort_by(f64::total_cmp);
        Some(Summary {
            median: percentile(&sorted, 50)?,
            p10: percentile(&sorted, 10)?,
            p90: percentile(&sorted, 90)?,
        })
    }
}

/// The `percent`th percentile of `sorted`, between the two nearest ranks.
fn percentile(sorted: &[f64], percent: u8) -> Option<Microseconds> {
    let scaled_rank = sorted.len().checked_sub(1)?.checked_mul(usize::from(percent))?;
    let (index, remainder) = (scaled_rank / 100, scaled_rank % 100);
    let lower = *sorted.get(index)?;
    if remainder == 0 {
        return Some(Microseconds::new(lower));
    }
    let upper = *sorted.get(index + 1)?;
    let fraction = f64::from(u8::try_from(remainder).ok()?) / 100.0;
    Some(Microseconds::new((upper - lower).mul_add(fraction, lower)))
}

/// Times `launch`, which queues one iteration's work on `stream`: a warmup
/// of single launches, then a timed series of batches sized from it, with
/// the GPU's state `read` just before and just after the series.
pub(crate) fn time<S, E: From<CudaError>>(
    context: &Context,
    stream: &Stream,
    mut launch: impl FnMut() -> Result<(), E>,
    mut read: impl FnMut() -> Result<S, E>,
) -> Result<Timed<S>, E> {
    let (start, end) = (context.event()?, context.event()?);
    let mut sample = |batch: u32| -> Result<Microseconds, E> {
        start.record(stream)?;
        for _ in 0..batch {
            launch()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        Ok(Microseconds::new(end.elapsed_since(&start)?.get() / f64::from(batch)))
    };
    let warmup = run(Budget::WARMUP, &mut || sample(1))?;
    let batch = batch_size(&Series { samples: warmup, batch: 1 });
    let before = read()?;
    let samples = run(Budget::TIMED, &mut || sample(batch))?;
    let after = read()?;
    Ok(Timed { series: Series { samples, batch }, before, after })
}

/// The launches a sample batches to last `SAMPLE_TIME`, from `warmup`'s
/// median time per launch: the smallest power of two that does, at most
/// `MAX_BATCH`.
fn batch_size(warmup: &Series) -> u32 {
    let Some(summary) = warmup.summary() else { return 1 };
    let per_launch = summary.median.get();
    if per_launch >= SAMPLE_TIME.get() {
        return 1;
    }
    let needed = (SAMPLE_TIME.get() / per_launch.max(f64::MIN_POSITIVE)).ceil();
    let mut batch = 1;
    while batch < MAX_BATCH && f64::from(batch) < needed {
        batch = (batch * 2).min(MAX_BATCH);
    }
    batch
}

/// Each time `iteration` gives, run until `budget` is met.
fn run<E>(
    budget: Budget,
    iteration: &mut impl FnMut() -> Result<Microseconds, E>,
) -> Result<Vec<Microseconds>, E> {
    let began = Instant::now();
    let mut samples = Vec::new();
    while !budget.is_met(samples.len(), began.elapsed()) {
        samples.push(iteration()?);
    }
    Ok(samples)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod tests {
    use super::{Budget, MAX_BATCH, Series, batch_size, percentile, run};
    use ptxc_core::Microseconds;
    use std::time::Duration;

    fn series(samples: &[f64]) -> Series {
        Series { samples: samples.iter().copied().map(Microseconds::new).collect(), batch: 1 }
    }

    #[test]
    fn percentiles_interpolate_between_the_nearest_ranks_as_numpy_does() {
        let sorted: Vec<f64> = (1..=10).map(f64::from).collect();

        let at = |percent| percentile(&sorted, percent).unwrap().get();

        assert!((at(50) - 5.5).abs() < 1e-12);
        assert!((at(10) - 1.9).abs() < 1e-12);
        assert!((at(90) - 9.1).abs() < 1e-12);
        assert!((at(0) - 1.0).abs() < 1e-12 && (at(100) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn the_summary_sorts_the_samples_first() {
        let summary = series(&[30.0, 10.0, 20.0]).summary().unwrap();

        assert!((summary.median.get() - 20.0).abs() < 1e-12);
        assert!((summary.p10.get() - 12.0).abs() < 1e-12);
        assert!((summary.p90.get() - 28.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_sample_is_every_percentile_and_no_samples_have_none() {
        let one = series(&[7.0]).summary().unwrap();

        assert_eq!(
            (one.median, one.p10, one.p90),
            (Microseconds::new(7.0), Microseconds::new(7.0), Microseconds::new(7.0))
        );
        assert_eq!(series(&[]).summary(), None);
    }

    #[test]
    fn a_batch_is_the_smallest_power_of_two_lasting_a_millisecond() {
        assert_eq!(batch_size(&series(&[7.0, 7.0, 9.0])), 256);
        assert_eq!(batch_size(&series(&[250.0])), 4);
        assert_eq!(batch_size(&series(&[1000.0])), 1);
        assert_eq!(batch_size(&series(&[5000.0])), 1);
    }

    #[test]
    fn a_batch_is_never_empty_nor_past_its_limit() {
        assert_eq!(batch_size(&series(&[])), 1);
        assert_eq!(batch_size(&series(&[0.0])), MAX_BATCH);
        assert_eq!(batch_size(&series(&[1e-6])), MAX_BATCH);
    }

    #[test]
    fn a_budget_needs_both_its_iterations_and_its_time() {
        let budget = Budget { iterations: 20, time: Duration::from_secs(1) };

        assert!(!budget.is_met(19, Duration::from_secs(5)));
        assert!(!budget.is_met(500, Duration::from_millis(999)));
        assert!(budget.is_met(20, Duration::from_secs(1)));
    }

    #[test]
    fn the_methodology_fixes_the_warmup_and_the_series() {
        assert!(!Budget::WARMUP.is_met(19, Duration::from_secs(60)));
        assert!(!Budget::TIMED.is_met(99, Duration::from_secs(60)));
        assert!(!Budget::TIMED.is_met(10_000, Duration::from_millis(1999)));
        assert!(Budget::TIMED.is_met(100, Duration::from_secs(2)));
    }

    #[test]
    fn a_run_keeps_every_sample_in_order_and_stops_at_its_budget() {
        let budget = Budget { iterations: 4, time: Duration::ZERO };
        let mut next = 0.0;

        let samples = run::<()>(budget, &mut || {
            next += 1.0;
            Ok(Microseconds::new(next))
        })
        .unwrap();

        assert_eq!(series(&[1.0, 2.0, 3.0, 4.0]).samples, samples);
    }

    #[test]
    fn a_failing_iteration_stops_the_run_with_its_error() {
        let budget = Budget { iterations: 4, time: Duration::ZERO };

        let outcome = run(budget, &mut || Err("launch failed"));

        assert_eq!(outcome, Err("launch failed"));
    }
}

/// Held by every test that times the GPU or needs no other process on it,
/// and by the test whose child process holds it, so none of them overlap.
#[cfg(all(test, feature = "gpu"))]
pub(crate) fn exclusive_gpu() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(all(test, feature = "gpu"))]
#[allow(clippy::unwrap_used, reason = "test code: a panic is a test failure")]
mod gpu {
    use super::{Budget, Timed, exclusive_gpu, time};
    use ptxc_core::KernelName;
    use ptxc_cuda::{BlockDim, CudaError, DeviceOrdinal, Driver, GridDim};
    use std::cell::Cell;

    /// Waits until `nanoseconds` of the global timer have passed.
    const WAIT: &str = "
.version 8.4
.target sm_89
.address_size 64
.visible .entry wait(.param .u64 nanoseconds)
{
    .reg .pred %p<1>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [nanoseconds];
    mov.u64 %rd1, %globaltimer;
LOOP:
    mov.u64 %rd2, %globaltimer;
    sub.s64 %rd3, %rd2, %rd1;
    setp.lt.u64 %p0, %rd3, %rd0;
    @%p0 bra LOOP;
    ret;
}
";

    /// The wait timed, in microseconds.
    const WAIT_MICROSECONDS: f64 = 50.0;

    /// The wait kernel's launches timed, `launched` called at each launch
    /// and `read` for the GPU's state.
    fn timed_waits<S>(
        mut launched: impl FnMut(),
        read: impl FnMut() -> Result<S, CudaError>,
    ) -> Timed<S> {
        let device = Driver::load().unwrap().device(DeviceOrdinal::new(0)).unwrap();
        let context = device.primary_context();
        let module = context.load_ptx(WAIT).unwrap();
        let function = module.function(&KernelName::new("wait")).unwrap();
        let stream = context.stream().unwrap();
        let nanoseconds = 50_000_u64;

        let launch = || -> Result<(), CudaError> {
            launched();
            function
                .launch()
                .grid(GridDim::new(1, 1, 1))
                .block(BlockDim::new(1, 1, 1))
                .arg(nanoseconds)
                .launch(&stream)
        };
        time(&context, &stream, launch, read).unwrap()
    }

    #[test]
    fn a_kernel_waiting_50_microseconds_times_at_about_50() {
        let _gpu = exclusive_gpu();

        let series = timed_waits(|| {}, || Ok(())).series;

        let summary = series.summary().unwrap();
        assert!(series.samples.len() >= 100, "{}", series.samples.len());
        assert!((16..=32).contains(&series.batch), "{}", series.batch);
        assert!(summary.p10.get() >= WAIT_MICROSECONDS, "{summary:?}");
        assert!(summary.median.get() < 2.0 * WAIT_MICROSECONDS, "{summary:?}");
        assert!(summary.p10 <= summary.median && summary.median <= summary.p90, "{summary:?}");
    }

    #[test]
    fn the_gpu_is_read_after_the_warmup_just_before_and_just_after_the_series() {
        let _gpu = exclusive_gpu();
        let launches = Cell::new(0_usize);

        let timed = timed_waits(|| launches.set(launches.get() + 1), || Ok(launches.get()));

        let series = &timed.series;
        let timed_launches = series.samples.len() * usize::try_from(series.batch).unwrap();
        assert!(timed.before >= Budget::WARMUP.iterations, "{}", timed.before);
        assert_eq!(timed.after - timed.before, timed_launches);
        assert_eq!(timed.after, launches.get());
    }
}
