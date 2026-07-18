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
    / "chapter_6_intraday_trading_and_market_microstructure/src/run_chapter6_analysis.py"
)


@pytest.fixture(scope="module")
def chapter6():
    spec = importlib.util.spec_from_file_location("chapter6_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(chapter6):
    return chapter6.load_data()


@pytest.fixture(scope="module")
def results(chapter6, data):
    return chapter6.run_experiments(data)


def test_official_archive_contract(chapter6, data):
    chapter6.validate_offline_assets()
    manifest = json.loads(chapter6.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 7
    assert sum(row["kind"] == "original_source" for row in manifest["members"]) == 4
    assert sum(row["kind"] == "research_data" for row in manifest["members"]) == 3
    assert len(data.tick_dn) == 77_390
    assert len(data.volume_dn) == 3_639
    assert len(data.executions) == 246_060


def test_tick_rule_tie_propagation_and_rolling_window(chapter6):
    prices = np.array([100.0, 101.0, 101.0, 100.0, 100.0])
    volume = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    signed = chapter6.tick_rule_order_flow(prices, volume)
    assert signed.tolist() == [0.0, 3.0, 5.0, -7.0, -11.0]
    datenum = np.array([0.0, 10.0, 20.0, 70.0, 80.0]) / 86_400.0
    rolling = chapter6.rolling_flow(datenum, signed, 60.0)
    assert np.isnan(rolling[:3]).all()
    assert rolling[3:].tolist() == [1.0, -18.0]


def test_tick_rule_bidask_exact_replay(chapter6, results):
    actual = results["tick_rule"]["bid_ask"]
    for key, expected in chapter6.SOURCE_TICK_BIDASK.items():
        tolerance = 0 if key == "num_trades" else chapter6.STRICT_TOLERANCE
        assert actual[key] == pytest.approx(expected, abs=tolerance)
    assert actual["net_pl_usd"] == pytest.approx(-6800.55, abs=1e-10)
    assert actual["ending_position"] == 0


def test_midpoint_comment_contradiction_is_not_hidden(chapter6, results):
    tick = results["tick_rule"]
    assert tick["midpoint_current_code"]["num_trades"] == 1190
    assert tick["bid_ask"]["num_trades"] == 1190
    assert tick["midpoint_current_code"]["gross_pl_usd"] == pytest.approx(887.5)
    reference = chapter6.reference_only_results()[0]
    assert reference["compared"] is False
    assert reference["book_output"]["num_trades"] == 1168
    assert "cannot change signals or trade count" in reference["reason"]


def test_volume_bar_replay_is_bounded_and_not_overclaimed(results):
    volume = results["volume_bar"]
    finite = volume["buy_fraction"][np.isfinite(volume["buy_fraction"])]
    assert len(finite) > 0
    assert np.all((finite >= 0) & (finite <= 1))
    assert volume["lookback_bars"] == 100
    assert volume["bar_volume_contracts"] == 500
    assert volume["contract"]["entry_threshold"] == 0.95
    assert volume["contract"]["exit_threshold"] == 0.5
    assert volume["strategy"]["gross_pl_usd"] == pytest.approx(-3975.0)
    assert volume["strategy"]["num_trades"] == 688
    assert volume["source_has_no_published_numeric_output"] is True


def test_buy_fraction_strategy_preserves_source_hold_band(chapter6):
    buy_fraction = np.array([np.nan, 0.96, 0.80, 0.51, 0.50, 0.04, 0.20, 0.49, 0.50])
    bid = np.full(len(buy_fraction), 99.0)
    ask = np.full(len(buy_fraction), 101.0)
    strategy = chapter6.execute_buy_fraction_strategy(buy_fraction, bid, ask)
    assert strategy["positions"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert strategy["num_trades"] == 4


def test_order_book_state_transitions(chapter6):
    buy: dict[float, int] = {}
    sell: dict[float, int] = {}
    assert chapter6.apply_book_event(
        buy,
        sell,
        side="BUY",
        action="WORKING_CONFIRMED",
        price=100.0,
        amount=5,
    ) == (False, False)
    assert chapter6.apply_book_event(
        buy,
        sell,
        side="BUY",
        action="PARTIAL_FILL_CONFIRMED",
        price=100.0,
        amount=2,
    ) == (False, False)
    assert buy == {100.0: 3}
    assert chapter6.apply_book_event(
        buy,
        sell,
        side="SELL",
        action="CANCEL_CONFIRMED",
        price=101.0,
        amount=1,
    ) == (True, False)


def test_order_book_quality_failures_are_disclosed(results):
    book = results["order_book"]
    assert book["events"] == 246_060
    assert book["sequence_is_strictly_increasing"] is True
    assert book["valid_two_sided_bars"] > 0
    assert book["crossed_event_states"] > 0
    assert book["median_spread"] < 0
    assert book["invalid_missing_reductions"] > 0
    assert book["invalid_oversize_reductions"] > 0


def test_all_chapter6_verifications_pass(chapter6, data, results):
    checks = chapter6.verify_results(data, results)
    assert len(checks) == 21
    assert all(checks.values())
    comparisons = chapter6.source_comparisons(results)
    assert len(comparisons) == 4
    assert all(row["matches_source"] for row in comparisons)
    references = chapter6.reference_only_results()
    assert all(row["compared"] is False and row["reason"] for row in references)


def test_generated_notebook_is_fully_executed(chapter6):
    if not chapter6.NOTEBOOK_PATH.exists():
        pytest.skip("notebook is generated by the chapter entrypoint")
    notebook = nbformat.read(chapter6.NOTEBOOK_PATH, as_version=4)
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
        f"ch6-{index:02d}-{cell.cell_type}"
        for index, cell in enumerate(notebook.cells)
    ]


def test_generated_markdown_math_and_metrics_are_clean(chapter6):
    report = chapter6.REPORT_DIR / "chapter6_report.md"
    if not report.exists():
        pytest.skip("report is generated by the chapter entrypoint")
    chapter6.assert_clean_markdown_math(report.read_text(encoding="utf-8"), "report")
    notebook = nbformat.read(chapter6.NOTEBOOK_PATH, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            chapter6.assert_clean_markdown_math(cell.source, f"notebook cell {index}")
    metrics = json.loads((chapter6.REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert all(type(value) is bool for value in metrics["verification"]["checks"].values())
    assert metrics["source_limitations"][
        "coinsetter_invalid_or_crossed_states_preserved_as_diagnostics"
    ] is True
