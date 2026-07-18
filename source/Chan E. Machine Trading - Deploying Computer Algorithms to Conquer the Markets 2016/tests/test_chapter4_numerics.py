from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_ROOT
    / "chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py"
)


@pytest.fixture(scope="module")
def chapter4():
    spec = importlib.util.spec_from_file_location("chapter4_test_module", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(chapter4):
    return chapter4.load_chapter_data()


@pytest.fixture(scope="module")
def results(chapter4, data):
    return chapter4.run_experiments(data)


def test_official_data_contract(chapter4, data):
    chapter4.validate_offline_assets()
    assert len(data.dates) == 2357
    assert data.split == 1178
    assert str(data.dates[0].date()) == "2004-12-22"
    assert str(data.dates[-1].date()) == "2014-06-02"
    assert str(data.dates[data.split - 1].date()) == "2009-09-15"
    assert str(data.dates[data.split].date()) == "2009-09-16"
    assert np.all(np.isfinite(data.close_prices))
    assert np.all(data.close_prices > 0)


def test_return_and_target_alignment(chapter4, data):
    expected = (data.close_prices[1] - data.close_prices[0]) / data.close_prices[0]
    assert np.isnan(data.returns[0])
    assert data.returns[1] == pytest.approx(expected, abs=1e-15)
    assert data.target[20] == data.returns[21]
    assert np.isnan(data.target[-1])


def test_linear_replay_matches_matlab(chapter4, results):
    linear = results["linear"]
    assert linear["observations"] == 1158
    assert np.allclose(
        linear["coefficients"], chapter4.BOOK_LINEAR_COEFFICIENTS, atol=2e-14, rtol=0
    )
    assert linear["train"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["linear_train_cagr"], abs=5e-7
    )
    assert linear["test"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["linear_test_cagr"], abs=5e-7
    )


def test_stepwise_complete_case_replay(chapter4, results):
    stepwise = results["stepwise"]
    assert stepwise["observations"] == 1158
    assert stepwise["selected"] == ["ret2"]
    assert stepwise["candidate_p_values"]["ret2"] == min(
        stepwise["candidate_p_values"].values()
    )
    assert np.allclose(
        stepwise["coefficients"], chapter4.BOOK_STEPWISE_COEFFICIENTS, atol=5e-10, rtol=0
    )
    assert stepwise["train"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["stepwise_train_cagr"], abs=5e-7
    )
    assert stepwise["test"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["stepwise_test_cagr"], abs=5e-7
    )


def test_tree_rule_and_hmm_replays(chapter4, results):
    assert results["tree_rule"]["train"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["tree_rule_train_cagr"], abs=5e-7
    )
    assert results["tree_rule"]["test"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["tree_rule_test_cagr"], abs=5e-7
    )
    assert results["hmm"]["train"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["hmm_train_cagr"], abs=5e-7
    )
    assert results["hmm"]["test"].gross.annual_return == pytest.approx(
        chapter4.BOOK_RESULTS["hmm_test_cagr"], abs=5e-7
    )
    assert np.allclose(chapter4.HMM_TRANSITION.sum(axis=1), 1)
    assert np.allclose(chapter4.HMM_EMISSION.sum(axis=1), 1)


def test_causal_position_and_ordered_validation(results):
    assert results["linear"]["test"].positions[0] == 0
    assert results["stepwise"]["test"].positions[0] == 0
    assert all(
        boundary["train_last"] < boundary["validation_first"]
        for boundary in results["python_models"]["cv_boundaries"]
    )


def test_source_boundary_leakage_is_detected_and_corrected(chapter4, data, results):
    assert data.target[data.split - 1] == data.returns[data.split]
    assert results["linear"]["test"].metadata["source_boundary_target_leakage"] is True
    assert results["stepwise"]["test"].metadata["source_boundary_target_leakage"] is True
    x_train, y_train, training_index = chapter4.valid_training_arrays(data)
    assert len(x_train) == len(y_train) == 1157
    assert training_index[-1] == data.split - 2
    assert training_index[-1] + 1 < data.split
    assert results["linear"]["causal"]["observations"] == 1157
    assert results["stepwise"]["causal"]["observations"] == 1157
    assert results["linear"]["causal"]["test"].metadata[
        "source_boundary_target_leakage"
    ] is False
    assert results["stepwise"]["causal"]["test"].metadata[
        "source_boundary_target_leakage"
    ] is False


def test_costs_are_monotone_and_forest_is_deterministic(chapter4, results):
    strategies = [
        results["linear"]["train"],
        results["linear"]["test"],
        results["linear"]["causal"]["test"],
        results["stepwise"]["train"],
        results["stepwise"]["test"],
        results["stepwise"]["causal"]["test"],
        results["tree_rule"]["train"],
        results["tree_rule"]["test"],
        results["hmm"]["train"],
        results["hmm"]["test"],
        *chapter4.python_model_results(results).values(),
    ]
    assert all(
        result.net.cumulative_return <= result.gross.cumulative_return + 1e-12
        for result in strategies
    )
    assert results["python_models"]["random_forest_repeat_equal"] is True


def test_unavailable_cross_sectional_results_are_null(results):
    for record in results["unavailable"].values():
        assert record["status"] == "unavailable_data"
        assert record["test"] is None
        assert record["reason"]
        assert record["book_output"]["annual_return"] is not None


def test_noncomparable_outputs_are_explicit_references(chapter4, results):
    references = chapter4.reference_only_comparisons(results)
    assert {record["topic"] for record in references} == {
        "bagging/random subspace",
        "classification tree",
        "support vector machine",
        "neural-network average",
    }
    assert all(record["compared"] is False and record["reason"] for record in references)
    classification = next(
        record for record in references if record["topic"] == "classification tree"
    )
    assert "duplicated in five MATLAB scripts" in classification["source_file"]
    assert results["python_models"]["mlp_ensemble"].metadata[
        "internal_validation"
    ] == "none; fixed 500-iteration training budget"


def test_all_chapter4_verifications_pass(chapter4, data, results):
    checks = chapter4.verify_results(data, results)
    assert len(checks) == len(chapter4.VERIFICATION_CLASSES) == 23
    assert all(checks.values())
    summary = chapter4.verification_summary(checks)
    assert summary["passed"] == 23
    assert summary["independent_or_empirical"] == 13
    assert summary["contract_invariant"] == 10


def test_generated_markdown_math_is_clean(chapter4):
    report = chapter4.REPORT_DIR / "chapter4_report.md"
    if not report.exists():
        pytest.skip("report is generated by the chapter entrypoint")
    chapter4.assert_clean_markdown_math(report.read_text(encoding="utf-8"), "report")
    if chapter4.NOTEBOOK_PATH.exists():
        notebook = nbformat.read(chapter4.NOTEBOOK_PATH, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "markdown":
                chapter4.assert_clean_markdown_math(cell.source, f"notebook cell {index}")
