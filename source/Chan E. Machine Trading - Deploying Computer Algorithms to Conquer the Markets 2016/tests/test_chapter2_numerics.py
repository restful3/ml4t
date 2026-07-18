from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = PROJECT_ROOT / "chapter_2_factor_models"
CHAPTER_SRC = CHAPTER_DIR / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAPTER_SRC))

from book3_common import sha256_file  # noqa: E402
from run_chapter2_analysis import (  # noqa: E402
    BOOK_RESULTS,
    ChapterData,
    assert_clean_markdown_math,
    audit_current_notebook,
    fama_position_row,
    fit_pca_adaptation,
    implied_moment_example,
)


def load_metrics() -> dict:
    path = CHAPTER_SRC / "reports/metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_official_archive_is_materialized_and_hashed() -> None:
    path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 20
    assert sum(item["kind"] == "original_source" for item in manifest["members"]) == 17
    assert sum(item["kind"] == "research_data" for item in manifest["members"]) == 3
    for item in manifest["members"]:
        extracted = PROJECT_ROOT / item["destination"]
        assert extracted.is_file()
        assert sha256_file(extracted) == item["sha256"]


def test_exact_book_comparisons_and_cost_contract_are_materialized() -> None:
    metrics = load_metrics()
    mapping = {
        "cross27": "cross_sectional_27",
        "roe_bm": "roe_bm",
        "roe_only": "roe_only",
    }
    for result_key, book_key in mapping.items():
        result = metrics["strategies"][result_key]["test"]
        expected = BOOK_RESULTS[book_key]
        assert np.isclose(result["cagr"], expected["test_cagr"], atol=1e-5)
        assert np.isclose(result["sharpe"], expected["test_sharpe"], atol=1e-5)

    corrected = metrics["strategies"]["fama_corrected"]
    assert corrected["metadata"]["long_count"] == 50
    assert (
        corrected["test_with_2bps_one_way_cost"]["cumulative_return"]
        <= corrected["test"]["cumulative_return"]
    )


def test_official_fama_source_preserves_the_diagnosed_indexing_bug() -> None:
    source = (
        CHAPTER_DIR / "original_matlab/FamaFrenchFactors_predictive.m"
    ).read_text(encoding="utf-8", errors="replace")
    assert "I(1:topN)" in source
    assert "I(end-topN+1)))=1" in source.replace(" ", "")
    assert "I(end-topN+1:end)" not in source


def test_python_fama_bug_path_selects_the_fiftieth_highest_stock() -> None:
    predictions = np.arange(120, dtype=float)
    bug_positions, ordered = fama_position_row(predictions, "matlab_bug")
    corrected_positions, _ = fama_position_row(predictions, "corrected_50_50")
    assert np.flatnonzero(bug_positions == 1.0).tolist() == [ordered[-50]]
    assert np.flatnonzero(corrected_positions == 1.0).tolist() == ordered[-50:].tolist()
    assert bug_positions.sum() == -49.0
    assert corrected_positions.sum() == 0.0


def test_option_moment_formula_example() -> None:
    moments = implied_moment_example()
    assert np.isclose(moments["implied_volatility"], 0.23)
    assert np.isclose(moments["implied_skewness_proxy"], -0.12)
    assert np.isclose(moments["implied_kurtosis_proxy"], 0.04)


def test_pca_adaptation_is_forward_only_and_dollar_neutral() -> None:
    generator = np.random.default_rng(7)
    rows, stocks = 75, 120
    returns = generator.normal(0.0, 0.01, size=(rows, stocks))
    returns[0] = np.nan
    mid = 100.0 * np.cumprod(1.0 + np.nan_to_num(returns), axis=0)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    data = ChapterData(
        dates=dates,
        date_codes=dates.strftime("%Y%m%d").astype(int).to_numpy(),
        symbols=tuple(f"S{index:03d}" for index in range(stocks)),
        mid=mid,
        returns=returns,
        factor_names=(),
        factors={},
    )
    result = fit_pca_adaptation(
        data, lookback=30, components=5, rebalance_every=10
    )
    assert not np.any(result.positions[:30])
    assert result.metadata["rebalances"] > 0
    active = np.any(result.positions != 0, axis=1)
    assert np.allclose(result.positions[active].sum(axis=1), 0.0)


def test_report_and_executed_notebook_contract() -> None:
    report = (CHAPTER_SRC / "reports/chapter2_report.md").read_text(encoding="utf-8")
    forbidden = [
        character
        for character in report
        if ord(character) < 32 and character != "\n"
    ]
    assert not forbidden
    assert_clean_markdown_math(report, "Chapter 2 generated report")

    notebook_path = CHAPTER_SRC / "chapter2_full_report.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"Chapter 2 notebook cell {index}")
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    benchmark = (CHAPTER_SRC / "reports/quality_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert f"셀 {len(notebook.cells)}개" in benchmark
    assert f"code 실행 {len(code_cells)}/{len(code_cells)}" in benchmark


def test_metrics_are_strict_json_and_mark_empty_pca_train_as_unavailable() -> None:
    path = CHAPTER_SRC / "reports/metrics.json"

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(value)

    metrics = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonstandard_constant,
    )
    assert metrics["strategies"]["pca"]["train"] is None
    assert metrics["strategies"]["pca"]["metadata"]["train_segment"] == (
        "no positions (lookback covers train)"
    )
    assert metrics["verification"]["checks"]["fama_train_sharpe_matches_book"]


def test_missing_repository_audit_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        audit_current_notebook.__globals__,
        "AUDIT_SCRIPT_PATH",
        tmp_path / "missing.py",
    )
    with pytest.warns(RuntimeWarning, match="quality audit unavailable"):
        assert audit_current_notebook() is None
