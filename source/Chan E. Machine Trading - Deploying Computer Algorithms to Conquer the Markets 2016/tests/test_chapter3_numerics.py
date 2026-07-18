from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = PROJECT_ROOT / "chapter_3_time_series_analysis"
CHAPTER_SRC = CHAPTER_DIR / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAPTER_SRC))

from book3_common import sha256_file  # noqa: E402
from run_chapter3_analysis import (  # noqa: E402
    AR10_BOOK_COEFFICIENTS,
    BOOK_RESULTS,
    PAIR_B,
    PAIR_D,
    assert_clean_markdown_math,
    audit_current_notebook,
    fit_ar_conditional,
)


def load_metrics() -> dict:
    path = CHAPTER_SRC / "reports/metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_official_archive_is_materialized_and_hashed() -> None:
    path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 33
    assert sum(item["kind"] == "original_source" for item in manifest["members"]) == 28
    assert sum(item["kind"] == "research_data" for item in manifest["members"]) == 5
    for item in manifest["members"]:
        extracted = PROJECT_ROOT / item["destination"]
        assert extracted.is_file()
        assert sha256_file(extracted) == item["sha256"]


def test_sufficient_statistic_ar_matches_independent_lstsq() -> None:
    generator = np.random.default_rng(42)
    prices = np.empty(250)
    prices[:3] = [100.0, 100.2, 99.9]
    for row in range(3, len(prices)):
        prices[row] = (
            0.2
            + 0.91 * prices[row - 1]
            + 0.05 * prices[row - 2]
            - 0.01 * prices[row - 3]
            + generator.normal(0, 0.03)
        )
    actual = fit_ar_conditional(prices, lag=3, end=200)
    rows = np.arange(3, 200)
    design = np.column_stack(
        [np.ones(len(rows)), prices[rows - 1], prices[rows - 2], prices[rows - 3]]
    )
    expected, _, _, _ = np.linalg.lstsq(design, prices[rows], rcond=None)
    assert np.allclose(actual, expected, atol=1e-9, rtol=0)


def test_ar_book_reproduction_and_cost_contract() -> None:
    metrics = load_metrics()
    univariate = metrics["univariate"]
    assert univariate["selected_ar_lag"] == 10
    assert np.allclose(
        univariate["ar10_full_coefficients"],
        AR10_BOOK_COEFFICIENTS,
        atol=5e-6,
        rtol=0,
    )
    ar10_full = metrics["strategies"]["ar10_full"]
    arma25 = metrics["strategies"]["arma25"]
    assert np.isclose(
        ar10_full["test"]["annual_return"],
        BOOK_RESULTS["ar10_annual_return"],
        atol=0.005,
    )
    assert np.isclose(
        arma25["test"]["annual_return"],
        BOOK_RESULTS["arma25_annual_return"],
        atol=0.005,
    )
    corrected = metrics["strategies"]["ar10_train"]
    assert (
        corrected["test_with_cost_sensitivity"]["cumulative_return"]
        <= corrected["test"]["cumulative_return"]
    )


def test_source_lookahead_is_preserved_and_corrected_path_differs() -> None:
    source = (CHAPTER_DIR / "original_matlab/buildARp_AUDUSD.m").read_text(
        encoding="utf-8"
    )
    assert "trainset" in source
    assert "fit=estimate(model, mid);" in source
    metrics = load_metrics()
    assert not np.allclose(
        metrics["univariate"]["ar10_full_coefficients"],
        metrics["univariate"]["ar10_train_coefficients"],
        atol=1e-7,
        rtol=0,
    )
    assert metrics["verification"]["checks"]["source_ar10_lookahead_detected"]


def test_var_vec_and_inactive_train_contracts() -> None:
    metrics = load_metrics()
    phi = np.asarray(metrics["multivariate"]["var_phi"])
    vec_c = np.asarray(metrics["multivariate"]["vec_c"])
    assert np.allclose(vec_c, phi - np.eye(5))
    assert metrics["strategies"]["var"]["metadata"]["lag"] == 1
    assert metrics["strategies"]["var"]["train"] is None
    assert metrics["verification"]["checks"]["var_positions_are_sector_neutral"]


def test_pair_replay_pins_published_parameters_and_split() -> None:
    metrics = load_metrics()
    pair = metrics["strategies"]["pair_kalman"]
    assert pair["metadata"]["process_loading"] == PAIR_B.tolist()
    assert pair["metadata"]["measurement_loading"] == PAIR_D
    assert pair["metadata"]["test_start"] == 1250
    assert pair["metadata"]["state_changes"] == 578
    assert np.isclose(pair["train"]["annual_return"], 0.0957120144185839)
    assert np.isclose(pair["test"]["annual_return"], -0.005262470428524901)


def test_metrics_are_strict_json_and_all_checks_pass() -> None:
    path = CHAPTER_SRC / "reports/metrics.json"

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(value)

    metrics = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonstandard_constant,
    )
    summary = metrics["verification"]["summary"]
    assert summary["total"] == 15
    assert summary["passed"] == 15
    assert all(metrics["verification"]["checks"].values())


def test_report_and_executed_notebook_contract() -> None:
    report = (CHAPTER_SRC / "reports/chapter3_report.md").read_text(encoding="utf-8")
    assert not [
        character
        for character in report
        if ord(character) < 32 and character != "\n"
    ]
    assert_clean_markdown_math(report, "regenerated report")
    assert "look-ahead" in report
    assert "방법론적 적응" in report

    notebook_path = CHAPTER_SRC / "chapter3_full_report.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"executed notebook cell {index}")
    assert len(notebook.cells) >= 24
    assert len(code_cells) >= 10
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    embedded_png = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert embedded_png >= 4
    benchmark = (CHAPTER_SRC / "reports/quality_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert "**100**" in benchmark
    assert f"code 실행 {len(code_cells)}/{len(code_cells)}" in benchmark


def test_markdown_math_guard_rejects_bare_commands_and_tabs() -> None:
    with pytest.raises(AssertionError, match="bare LaTeX"):
        assert_clean_markdown_math("$$P_t = phi_0 + epsilon_t$$", "bare formula")
    with pytest.raises(AssertionError, match="control characters"):
        assert_clean_markdown_math("$$P_t = \\theta_t$$\t", "tabbed formula")


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
