from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_ROOT
    / "chapter_8_algorithmic_trading_is_good_for_body_and_soul/src/run_chapter8_analysis.py"
)


@pytest.fixture(scope="module")
def chapter8():
    spec = importlib.util.spec_from_file_location("chapter8_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(chapter8):
    return chapter8.load_data()


@pytest.fixture(scope="module")
def results(chapter8, data):
    return chapter8.run_experiments(data)


def test_chapter8_archive_absence_and_reused_data_contract(chapter8, data):
    chapter8.validate_offline_assets()
    manifest = chapter8.chapter8_manifest()
    assert manifest["official_archive_available"] is False
    assert manifest["reused_archive_chapter"] == 5
    assert data.source_sha256 == manifest["data_sha256"]
    assert len(data.dates) == 2_415
    assert data.dates[0] == 20060426
    assert data.dates[-1] == 20151125
    assert "GLD" in data.stocks and "USO" in data.stocks
    assert "GDX" not in data.stocks


def test_cross_book_data_identity_is_independently_checked(data):
    assert data.cross_book_source_present is True
    assert data.cross_book_data_present is True
    assert data.cross_book_values_match is True


def test_rolling_hedge_ratio_does_not_use_next_observation(chapter8):
    gld = np.arange(1.0, 26.0)
    uso = 2.0 + 3.0 * gld
    baseline = chapter8.rolling_hedge_ratio(gld, uso, lookback=20)
    changed = uso.copy()
    changed[20] = 10_000.0
    perturbed = chapter8.rolling_hedge_ratio(gld, changed, lookback=20)
    assert baseline[19] == pytest.approx(3.0)
    assert perturbed[19] == pytest.approx(baseline[19])
    assert perturbed[20] != pytest.approx(baseline[20])


def test_position_state_machine_exposes_source_long_exit_typo(chapter8):
    zscore = np.array([np.nan, -1.2, -0.8, -0.1, 0.0, 1.2, 0.8, 0.1, 0.0, -0.1])
    corrected = chapter8.positions_from_zscore(zscore)
    typo = chapter8.positions_from_zscore(zscore, source_long_exit_bug=True)
    assert corrected.tolist() == [0, 1, 1, 1, 1, -1, -1, -1, -1, 0]
    assert typo.tolist() == [0, 1, 0, 0, 0, -1, -1, -1, -1, 0]


def test_cross_book_active_day_metrics_exact_replay(chapter8, results):
    strategy = results["gld_uso"]["original_period"]
    assert len(strategy["dates"]) == 1_500
    assert strategy["source_active_day"]["periods"] == 1_146
    for metric, expected in chapter8.SOURCE_ACTIVE_DAY_RESULTS.items():
        assert strategy["source_active_day"][metric] == pytest.approx(
            expected, abs=1e-12
        )
    assert all(row["matches_source"] for row in chapter8.source_comparisons(results))


def test_calendar_clock_cost_and_extension_are_not_overclaimed(results):
    original = results["gld_uso"]["original_period"]
    extension = results["gld_uso"]["post_book_extension"]
    assert original["source_active_day"]["annual_return"] > original["calendar_gross"]["annual_return"]
    assert original["calendar_gross"]["annual_return"] > original["calendar_net_10bps"]["annual_return"]
    assert extension["dates"][0] == 20120410
    assert extension["dates"][-1] == 20151125
    assert len(extension["dates"]) == 915
    assert extension["calendar_gross"]["annual_return"] > 0
    assert extension["calendar_net_10bps"]["annual_return"] > 0


def test_strategy_lifecycle_is_deterministic_lagged_conceptual_fixture(results):
    lifecycle = results["strategy_lifecycle"]
    assert lifecycle["contract"]["random_seed"] == 20260718
    assert lifecycle["contract"]["weights_use_information_through_previous_day"]
    assert lifecycle["contract"]["conceptual_simulation_not_backtest"]
    assert np.allclose(lifecycle["adaptive_weights"].sum(axis=1), 1.0)
    assert (
        lifecycle["performance"]["equal_weight_live_pool"]["maximum_drawdown"]
        > lifecycle["performance"]["concentrated_dying_strategy"]["maximum_drawdown"]
    )


def test_managed_account_integer_granularity_is_monotone(results):
    experiment = results["managed_account_granularity"]
    rows = experiment["rows"]
    assert [row["strategies_with_nonzero_allocation"] for row in rows] == [0, 4, 10, 10]
    assert rows[0]["l1_weight_error"] == pytest.approx(1.0)
    assert rows[-1]["l1_weight_error"] == pytest.approx(0.018)
    assert experiment["contract"]["illustrative_margin_not_market_quote"]


def test_all_chapter8_verifications_pass(chapter8, data, results):
    checks = chapter8.verify_results(data, results)
    assert len(checks) == 24
    assert all(checks.values())
    references = chapter8.reference_only_results()
    assert len(references) == 3
    assert all(row["compared"] is False and row["reason"] for row in references)


def test_generated_notebook_and_metrics_are_clean(chapter8):
    if not chapter8.NOTEBOOK_PATH.exists():
        pytest.skip("notebook is generated by the chapter entrypoint")
    notebook = nbformat.read(chapter8.NOTEBOOK_PATH, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(
        output.output_type == "error" for cell in code for output in cell.outputs
    )
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code
        for output in cell.outputs
    ) == 5
    assert [cell.id for cell in notebook.cells] == [
        f"ch8-{index:02d}-{cell.cell_type}"
        for index, cell in enumerate(notebook.cells)
    ]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            chapter8.assert_clean_markdown_math(cell.source, f"notebook cell {index}")
    metrics = json.loads(
        (chapter8.REPORT_DIR / "metrics.json").read_text(encoding="utf-8")
    )
    assert all(
        type(value) is bool for value in metrics["verification"]["checks"].values()
    )
    assert metrics["source_limitations"]["no_official_chapter_8_code_or_data_archive"]
