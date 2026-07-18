from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_SRC = (
    PROJECT_ROOT
    / "chapter_1_the_basics_of_algorithmic_trading"
    / "src"
)
sys.path.insert(0, str(CHAPTER_SRC))

from run_chapter1_analysis import (  # noqa: E402
    REPORT_DIR,
    _solve_long_only_qp,
    assert_clean_markdown_math,
    audit_current_notebook,
    calculate_efficient_frontier,
    load_etf_data,
    summarize_net_log_seed_distribution,
)


def test_active_set_qp_matches_independent_slsqp() -> None:
    frontier = calculate_efficient_frontier(load_etf_data())
    for target in frontier.targets[1:-1]:
        weights, variance = _solve_long_only_qp(
            frontier.mean_returns, frontier.covariance, float(target)
        )
        result = minimize(
            lambda candidate: float(
                candidate @ frontier.covariance @ candidate
            ),
            x0=np.full(len(frontier.symbols), 1.0 / len(frontier.symbols)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(frontier.symbols),
            constraints=(
                {"type": "eq", "fun": lambda candidate: np.sum(candidate) - 1.0},
                {
                    "type": "eq",
                    "fun": lambda candidate, target=target: (
                        candidate @ frontier.mean_returns - target
                    ),
                },
            ),
            options={"ftol": 1e-14, "maxiter": 2_000},
        )
        assert result.success, result.message
        assert np.isclose(result.fun, variance, rtol=1e-8, atol=1e-13)
        assert np.isclose(weights @ frontier.mean_returns, target, atol=1e-10)


def test_multiseed_net_log_error_has_expected_convergence() -> None:
    summaries = summarize_net_log_seed_distribution(
        sample_sizes=(100, 1_000, 10_000), seeds=range(1, 31)
    )
    medians = np.array([item.median_error for item in summaries])
    assert np.all(np.diff(medians) < 0)
    slope = np.polyfit(np.log10([100, 1_000, 10_000]), np.log10(medians), 1)[0]
    assert -2.5 < slope < -1.5


def test_generated_report_has_no_control_characters() -> None:
    report = (REPORT_DIR / "chapter1_report.md").read_text(encoding="utf-8")
    forbidden = [character for character in report if ord(character) < 32 and character != "\n"]
    assert not forbidden
    assert_clean_markdown_math(report, "Chapter 1 generated report")


def test_quality_benchmark_matches_executed_notebook_counts() -> None:
    notebook_path = CHAPTER_SRC / "chapter1_full_report.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            assert_clean_markdown_math(source, f"Chapter 1 notebook cell {index}")
    code_count = sum(cell["cell_type"] == "code" for cell in notebook["cells"])
    benchmark = (REPORT_DIR / "quality_benchmark.md").read_text(encoding="utf-8")
    assert f"총 {len(notebook['cells'])}셀" in benchmark
    assert f"code {code_count}" in benchmark


def test_missing_repository_audit_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        audit_current_notebook.__globals__, "AUDIT_SCRIPT_PATH", tmp_path / "missing.py"
    )
    with pytest.warns(RuntimeWarning, match="quality audit is unavailable"):
        assert audit_current_notebook() is None
