from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "chapter_7_bitcoins/src/run_chapter7_analysis.py"


@pytest.fixture(scope="module")
def chapter7():
    spec = importlib.util.spec_from_file_location("chapter7_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(chapter7):
    return chapter7.load_data()


@pytest.fixture(scope="module")
def results(chapter7, data):
    return chapter7.run_experiments(data)


def test_official_archive_and_data_contract(chapter7, data):
    chapter7.validate_offline_assets()
    manifest = json.loads(chapter7.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 12
    assert sum(row["kind"] == "original_source" for row in manifest["members"]) == 9
    assert sum(row["kind"] == "research_data" for row in manifest["members"]) == 3
    assert len(data.mid) == 479_535
    assert len(data.daily_close) == 345
    assert len(data.trades) == 119_914
    assert data.hhmm_rows == 13_438
    assert data.crossed_quote_rows == 517


def test_btc_daily_risk_replay(results):
    risk = results["risk"]
    assert risk["annualized_volatility"] == pytest.approx(0.67356891126152)
    assert risk["worst_daily_return"] == pytest.approx(-0.24488986784140976)
    assert risk["worst_date"] == 20150114
    assert risk["best_daily_return"] == pytest.approx(0.19682466798530673)
    assert risk["best_date"] == 20140303
    assert risk["maximum_drawdown"] == pytest.approx(-0.7923083447431873)


def test_ar_approximate_replay_and_lookahead_disclosure(chapter7, results):
    ar = results["ar"]
    assert ar["yule_walker_selected_lag"] == 15
    assert ar["source_selected_lag"] == 16
    assert ar["adf"]["statistic"] == pytest.approx(-3.0009983648270886)
    assert ar["contract"]["source_fit_uses_test_prices"] is True
    assert not np.array_equal(
        ar["source_full_fit"]["coefficients"],
        ar["train_only"]["coefficients"],
    )
    comparisons = [
        row
        for row in chapter7.source_comparisons(results)
        if row["topic"] == "buildARp_BTCUSD.m"
    ]
    assert len(comparisons) == 20
    assert all(row["matches_source"] for row in comparisons)
    assert all("approximate" in row["classification"] for row in comparisons)


def test_bollinger_exact_replay(chapter7, results):
    bollinger = results["bollinger"]
    assert bollinger["contract"]["rolling_standard_deviation_ddof"] == 1
    assert bollinger["test"]["cumulative_return"] == pytest.approx(
        chapter7.SOURCE_RESULTS["bollinger_test_cumulative_return"], abs=1e-12
    )
    assert bollinger["test"]["annual_return"] == pytest.approx(
        chapter7.SOURCE_RESULTS["bollinger_test_annual_return"], abs=1e-12
    )
    assert bollinger["test_net_10bps"]["cumulative_return"] == pytest.approx(
        chapter7.SOURCE_RESULTS["bollinger_cost_test_cumulative_return"],
        abs=1e-12,
    )


def test_nanosecond_rolling_window_boundary(chapter7):
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:30",
                "2020-01-01 00:01:00",
                "2020-01-01 00:01:01",
            ]
        )
    )
    flow = chapter7.rolling_order_flow(
        timestamps, np.array([1.0, 2.0, 4.0, 8.0]), seconds=60
    )
    assert np.isnan(flow[:2]).all()
    assert flow[2] == 6.0  # exactly 60 seconds ago is excluded
    assert flow[3] == 14.0


def test_order_flow_all_daily_and_aggregate_outputs_match(chapter7, results):
    order = results["order_flow"]
    observed = [
        (row["date"], round(row["gross_pl_usd"], 2), row["num_trades"])
        for row in order["daily"]
    ]
    assert observed == chapter7.SOURCE_ORDER_DAILY
    assert order["gross_pl_usd"] == pytest.approx(32.54, abs=1e-9)
    assert order["pl_per_trade_usd"] == pytest.approx(0.096845, abs=5e-7)
    assert order["num_trades"] == 336
    assert order["ending_position"] == 0
    assert order["cost_sensitivity"]["0.06_one_way"] == pytest.approx(12.38)
    assert order["cost_sensitivity"]["0.25_one_way"] == pytest.approx(-51.46)


def test_random_forest_is_deterministic_adaptation_not_exact(results):
    forest = results["random_forest"]
    assert forest["contract"]["random_seed"] == 1
    assert forest["contract"]["trees"] == 5
    assert forest["contract"]["minimum_leaf_size"] == 100
    assert forest["contract"]["not_numerically_equivalent_to_matlab_treebagger"]
    assert forest["train"]["indices"][-1] < forest["test"]["indices"][0]
    assert (
        forest["test"]["performance_net_10bps"]["cumulative_return_log"]
        < forest["test"]["performance"]["cumulative_return_log"]
    )


def test_historical_arbitrage_is_not_called_risk_free(results):
    arbitrage = results["arbitrage"]
    assert arbitrage["gross_spread_usd_per_btc"] == pytest.approx(5.644)
    assert arbitrage["illustrative_net_usd_per_btc"] == pytest.approx(2.644)
    assert arbitrage["credit_and_transfer_risk_quantified"] is False
    assert arbitrage["historical_example_not_current_recommendation"] is True


def test_all_chapter7_verifications_pass(chapter7, data, results):
    checks = chapter7.verify_results(data, results)
    assert len(checks) == 25
    assert all(checks.values())
    assert all(row["matches_source"] for row in chapter7.source_comparisons(results))
    references = chapter7.reference_only_results()
    assert len(references) == 5
    assert all(row["compared"] is False and row["reason"] for row in references)


def test_generated_notebook_and_metrics_are_clean(chapter7):
    if not chapter7.NOTEBOOK_PATH.exists():
        pytest.skip("notebook is generated by the chapter entrypoint")
    notebook = nbformat.read(chapter7.NOTEBOOK_PATH, as_version=4)
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
        f"ch7-{index:02d}-{cell.cell_type}"
        for index, cell in enumerate(notebook.cells)
    ]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            chapter7.assert_clean_markdown_math(cell.source, f"notebook cell {index}")
    metrics = json.loads(
        (chapter7.REPORT_DIR / "metrics.json").read_text(encoding="utf-8")
    )
    assert all(
        type(value) is bool for value in metrics["verification"]["checks"].values()
    )
    assert metrics["source_limitations"]["ar_source_full_sample_fit_is_lookahead"]
