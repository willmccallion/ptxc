"""The README's result tables, rendered from one run's results."""

import json

import pytest

from ptxc_baselines import conv_tables, gemm_tables, render_tables, tables

GPU = {"gpu": "NVIDIA GeForce RTX 4060 Ti", "driver": "595.112.07"}


def clocks(*, held=True, events=()):
    before = {"sm_mhz": 2700, "mem_mhz": 8751, "temperature_c": 50, "events": []}
    after = {"sm_mhz": 2700 if held else 2610, "mem_mhz": 8751, "temperature_c": 60}
    return {"before": before, "after": {**after, "events": list(events)}, "held": held}


def case(case_id, params, median_us, **state):
    return {
        "id": case_id,
        "params": params,
        "verified": state.get("verified", True),
        "clocks": clocks(held=state.get("held", True), events=state.get("events", ())),
        "median_us": median_us,
        "p10_us": median_us * 0.99,
        "p90_us": median_us * 1.01,
    }


def document(suite, name, cases, *, dirty=False):
    framework = {"name": name, "version": "1.0", "commit": "abc123def456", "dirty": dirty}
    return {"schema": 1, "suite": suite, "framework": framework, "environment": GPU, "cases": cases}


def gemm(m, n, k, dtype="f16", layout="NN"):
    params = {"m": m, "n": n, "k": k, "dtype": dtype, "layout": layout}
    return f"gemm/{dtype}/{layout}/{m}x{n}x{k}", params


def conv(layer, batch):
    params = {
        "layer": layer,
        "batch": batch,
        "channels": 64,
        "features": 64,
        "size": 56,
        "kernel": 3,
        "stride": 1,
        "pad": 1,
        "dtype": "f16",
    }
    return f"conv/f16/{layer}/b{batch}", params


def write(directory, documents):
    for doc in documents:
        name = f"{doc['suite']}-{doc['framework']['name']}.json"
        (directory / name).write_text(json.dumps(doc))


def test_results_from_uncommitted_code_unverified_or_with_moving_clocks_are_refused():
    cases = [
        case(*gemm(256, 256, 256), 10.0, verified=False),
        case(*gemm(512, 512, 512), 20.0, held=False),
    ]

    found = tables.problems(document("gemm", "ptxc", cases, dirty=True), "gemm-ptxc.json")

    assert found == [
        "gemm-ptxc.json: run from uncommitted changes",
        "gemm-ptxc.json: gemm/f16/NN/256x256x256 was not verified",
        "gemm-ptxc.json: gemm/f16/NN/512x512x512's clocks moved during its series",
    ]


def test_loading_refuses_every_file_that_may_not_be_published(tmp_path):
    ours = document("gemm", "ptxc", [case(*gemm(256, 256, 256), 10.0)], dirty=True)
    theirs = document("gemm", "cublas", [case(*gemm(256, 256, 256), 8.0, held=False)])
    write(tmp_path, [ours, theirs])

    with pytest.raises(tables.RenderError) as refused:
        tables.load(tmp_path, "gemm")

    assert "gemm-ptxc.json: run from uncommitted changes" in str(refused.value)
    assert "gemm-cublas.json: gemm/f16/NN/256x256x256's clocks" in str(refused.value)


def loaded_gemm(timings):
    """Each framework's results from `(problem, ptxc, cuBLAS, best of 8)`
    times; `None` where cuBLAS does not run the problem."""
    columns = {"ptxc": [], "cublas": [], "cublas-best-of-8": []}
    for problem, *times in timings:
        for name, time in zip(columns, times, strict=True):
            if time is not None:
                columns[name].append(case(*problem, time))
    return {name: tables.results(document("gemm", name, cases)) for name, cases in columns.items()}


def test_the_gemm_summary_gives_each_type_and_layout_its_geometric_mean_and_extremes():
    rows = gemm_tables.rows(
        loaded_gemm(
            [
                (gemm(1024, 1024, 1024), 10.0, 8.0, 5.0),
                (gemm(2048, 2048, 2048), 40.0, 50.0, 40.0),
                (gemm(256, 256, 256, "e4m3"), 2.0, None, None),
            ]
        )
    )

    text = gemm_tables.summary(rows)

    # 0.8 and 1.25 have a geometric mean of 1; against the best, 0.5 and 1.
    assert "| f16 | NN | 2 | 100% | 71% | 1 of 2 | 1024x1024x1024 (80%) |" in text
    none = tables.NONE
    assert f"| e4m3 | NN | 1 | {none} | {none} | 0 of 0 | {none} |" in text


def test_the_exit_metric_takes_f16_problems_of_at_least_1024_by_layout():
    rows = gemm_tables.rows(
        loaded_gemm(
            [
                (gemm(1024, 1024, 1024), 10.0, 9.0, 9.0),
                (gemm(1024, 1024, 1024, layout="NT"), 10.0, 6.0, 6.0),
                (gemm(1024, 768, 1024), 10.0, 1.0, 1.0),
                (gemm(1024, 1024, 1024, "bf16"), 10.0, 1.0, 1.0),
            ]
        )
    )

    metric = gemm_tables.exit_metric(rows)

    # NN at 90% and NT at 60%: their geometric mean is 73%.
    assert metric.endswith("throughput: 73% (NN 90%, NT 60%).")


def test_a_gemm_rate_counts_two_operations_per_multiply_add():
    row = gemm_tables.rows(loaded_gemm([(gemm(1000, 1000, 1000), 2000.0, None, None)]))[0]

    assert row.rate() == pytest.approx(1.0)


def loaded_conv(timings):
    columns = {"ptxc": [], "pytorch": [], "pytorch-channels-last": []}
    for problem, *times in timings:
        for name, time in zip(columns, times, strict=True):
            columns[name].append(case(*problem, time))
    return {name: tables.results(document("conv", name, cases)) for name, cases in columns.items()}


def test_the_conv_summary_compares_each_batch_with_both_cudnn_layouts():
    rows = conv_tables.rows(
        loaded_conv(
            [
                (conv("layer1", 1), 10.0, 5.0, 4.0),
                (conv("layer2", 1), 10.0, 20.0, 5.0),
                (conv("layer1", 8), 30.0, 15.0, 15.0),
            ]
        )
    )

    text = conv_tables.summary(rows)

    assert "| 1 | 2 | 100% | 45% | 1 of 2 | layer1 (50%) |" in text
    assert "| 8 | 1 | 50% | 50% | 0 of 1 | layer1 (50%) |" in text


def test_a_conv_rate_counts_the_direct_convolutions_multiply_adds():
    row = conv_tables.rows(loaded_conv([(conv("layer1", 1), 1.0, 1.0, 1.0)]))[0]

    # 64 output channels of 56 by 56, each summing 64 channels of 3 by 3.
    operations = 2 * 64 * 56 * 56 * 64 * 9
    assert row.rate() == pytest.approx(operations / 1e6)


def test_only_the_text_between_a_suites_markers_is_replaced():
    text = "a\n<!-- bench:gemm:begin -->\nold\n<!-- bench:gemm:end -->\nb"

    replaced = tables.replace_between(text, "gemm", "new")

    assert replaced == "a\n<!-- bench:gemm:begin -->\nnew\n<!-- bench:gemm:end -->\nb"
    with pytest.raises(tables.RenderError):
        tables.replace_between(text, "conv", "new")


def test_a_run_renders_the_readme_and_every_case_beside_its_results(tmp_path):
    run = tmp_path / "bench" / "results" / "2026-11-25-abc123"
    run.mkdir(parents=True)
    write(
        run,
        [
            document(
                "gemm", "ptxc", [case(*gemm(1024, 1024, 1024), 10.0, events=["sw_power_cap"])]
            ),
            document("gemm", "cublas", [case(*gemm(1024, 1024, 1024), 9.0)]),
        ],
    )
    readme = tmp_path / "README.md"
    markers = "<!-- bench:{0}:begin -->\n<!-- bench:{0}:end -->"
    readme.write_text(markers.format("gemm") + "\n" + markers.format("conv") + "\n")

    rendered = render_tables.render(run, readme)

    text = readme.read_text()
    assert rendered == ["gemm"]
    assert "| f16 | NN | 1 | 90% |" in text
    assert (
        "[bench/results/2026-11-25-abc123/gemm.md](bench/results/2026-11-25-abc123/gemm.md)" in text
    )
    assert "<!-- bench:conv:begin -->\n<!-- bench:conv:end -->" in text
    page = (run / "gemm.md").read_text()
    assert "NVIDIA GeForce RTX 4060 Ti, driver 595.112.07, SM at 2700 MHz." in page
    none = tables.NONE
    assert f"| 1024x1024x1024 | 10.0† | 214.75 | 9.0 | 90% | {none} | {none} |" in page
    assert "† The GPU reported a power or thermal cap" in page


def test_results_in_another_schema_are_refused():
    later = {**document("gemm", "ptxc", []), "schema": 2}

    assert tables.problems(later, "gemm-ptxc.json") == ["gemm-ptxc.json: schema 2, not 1"]
