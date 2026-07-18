#!/usr/bin/env python3
"""Reproduce and audit Machine Trading Chapter 4 AI examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)

CHAPTER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from book3_common import (  # noqa: E402
    assert_clean_markdown_math,
    chapter_manifest as load_chapter_manifest,
    download_verified_archive,
    environment_versions,
    execute_notebook,
    materialize_chapter_archive,
    sha256_file,
    validate_chapter_extraction,
    write_text_if_changed,
)


PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"
RAW_DIR = PROJECT_ROOT / "data/raw/book3/chapter_4"
SPY_PATH = RAW_DIR / "inputData_SPY.mat"
REPORT_DIR = CHAPTER_DIR / "src/reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = CHAPTER_DIR / "src/chapter4_full_report.ipynb"
SOURCE_DIR = CHAPTER_DIR / "original_matlab"
SOURCE_MANIFEST_PATH = SOURCE_DIR / "SOURCE_MANIFEST.json"
AUDIT_SCRIPT_PATH = (
    PROJECT_ROOT.parents[1]
    / ".codex/skills/create-chan-chapter-analysis/scripts/audit_chapter_artifacts.py"
)

FEATURE_NAMES = ("ret1", "ret2", "ret5", "ret20")
ONE_WAY_COST_BPS = 2.0
RANDOM_SEED = 1

BOOK_RESULTS = {
    "linear_train_cagr": 0.343339,
    "linear_train_sharpe": 1.365304,
    "linear_test_cagr": 0.003582,
    "linear_test_sharpe": 0.103952,
    "stepwise_train_cagr": 0.436383,
    "stepwise_train_sharpe": 1.649068,
    "stepwise_test_cagr": 0.105685,
    "stepwise_test_sharpe": 0.695228,
    "tree_rule_train_cagr": 0.287519,
    "tree_rule_train_sharpe": 1.529458,
    "tree_rule_test_cagr": 0.038665,
    "tree_rule_test_sharpe": 0.505706,
    "bagging_test_cagr": 0.071967,
    "bagging_test_sharpe": 0.505925,
    "classification_tree_test_cagr": 0.047740,
    "classification_tree_test_sharpe": 0.366381,
    "svm_test_cagr": 0.133530,
    "svm_test_sharpe": 0.847489,
    "hmm_train_cagr": 0.086644,
    "hmm_train_sharpe": 0.467205,
    "hmm_test_cagr": -0.010173,
    "hmm_test_sharpe": 0.019724,
    "nn_average_test_cagr": 0.078454,
    "nn_average_test_sharpe": 0.542809,
    "technical_stock_test_cagr": 0.023153,
    "technical_stock_test_sharpe": 0.872566,
    "fundamental_stock_test_cagr": 0.039567,
    "fundamental_stock_test_sharpe": 1.117113,
}

BOOK_LINEAR_COEFFICIENTS = np.array(
    [
        4.55250560811905e-05,
        -0.0249732825555171,
        -0.130976974703952,
        0.0139640149617444,
        0.00173116222712215,
    ]
)
BOOK_STEPWISE_COEFFICIENTS = np.array([4.15245367e-05, -0.130063585])
HMM_PRIOR = np.array([0.001229270954050, 0.998770729045950])
HMM_TRANSITION = np.array(
    [
        [0.595799945653891, 0.404200054346109],
        [0.748746357609356, 0.251253642390644],
    ]
)
HMM_EMISSION = np.array(
    [
        [0.190513239994937, 0.809486760005063],
        [0.973037767548789, 0.026962232451211],
    ]
)

CHAPTER_COVERAGE = (
    ("AI와 자동화된 패턴 탐색", "개념 설명", "예측보다 검증 설계가 우선"),
    ("선형회귀와 과적합", "정확 수치 재현", "lr.m, SPY 공식 MAT"),
    ("stepwise regression", "정확 수치 재현", "완전관측 행 고정과 ret2 선택"),
    ("regression tree", "정확 규칙 재현 + Python 적응", "극단 leaf 규칙과 sklearn 전체 tree"),
    ("cross validation", "실행된 방법론 수정", "random K-fold 대신 TimeSeriesSplit"),
    ("bagging/random subspace", "Python 적응 + output-only 원본 참조", "RandomForestRegressor, seed=1; 직접 비교 금지"),
    ("boosting", "Python 적응 + 원본 한계 진단", "훈련 CV로 iteration 선택"),
    ("classification tree", "Python 적응 + provenance-uncertain 원본 참조", "같은 주석이 5개 MATLAB 파일에 중복"),
    ("support vector machine", "Python 적응 + output-only 원본 참조", "train-only kernel 선택"),
    ("hidden Markov model", "published-parameter 정확 replay", "온라인 filtering, 행렬 재추정 아님"),
    ("neural network", "결정적 다중 seed 적응 + output-only 원본 참조", "MLP 10개와 seed 분산"),
    ("aggregation과 normalization", "실행 진단", "StandardScaler와 ensemble 평균"),
    ("technical stock selection", "output-only", "fundamentalData가 공식 ZIP에 없음"),
    ("fundamental stock selection", "output-only", "저자 로컬 Dropbox 경로만 존재"),
    ("look-ahead/survivorship/selection bias", "개념 + 자동 검증", "시계열 split과 테스트 격리"),
    ("거래비용과 위험", "실행 백테스트", "2bps 편도 민감도, MDD와 duration"),
    ("연습문제·요약·endnotes", "커버리지 매핑", "모형 확장과 검증 질문"),
)

VERIFICATION_CLASSES = {
    "archive_manifest_matches": "independent_or_empirical",
    "data_dates_are_strictly_increasing": "independent_or_empirical",
    "linear_coefficients_match_source": "independent_or_empirical",
    "linear_train_matches_source": "independent_or_empirical",
    "linear_test_matches_source": "independent_or_empirical",
    "stepwise_selects_ret2": "independent_or_empirical",
    "stepwise_coefficients_match_source": "independent_or_empirical",
    "stepwise_train_matches_source": "independent_or_empirical",
    "stepwise_test_matches_source": "independent_or_empirical",
    "tree_rule_train_matches_source": "independent_or_empirical",
    "tree_rule_test_matches_source": "independent_or_empirical",
    "hmm_train_matches_source": "independent_or_empirical",
    "hmm_test_matches_source": "independent_or_empirical",
    "hmm_probabilities_are_stochastic": "contract_invariant",
    "signals_are_applied_one_day_later": "contract_invariant",
    "chronological_split_is_disjoint": "contract_invariant",
    "source_boundary_target_leakage_detected": "contract_invariant",
    "python_training_targets_precede_test": "contract_invariant",
    "time_series_cv_respects_order": "contract_invariant",
    "transaction_costs_do_not_improve_results": "contract_invariant",
    "random_forest_is_seed_deterministic": "contract_invariant",
    "cross_sectional_data_is_not_claimed_available": "contract_invariant",
    "incomplete_matlab_scripts_are_disclosed": "contract_invariant",
}


@dataclass(frozen=True)
class Performance:
    annual_return: float
    sharpe: float | None
    maximum_drawdown: float
    drawdown_duration: int
    cumulative_return: float
    mean_daily_return: float
    annual_turnover: float


@dataclass
class StrategyResult:
    name: str
    signals: np.ndarray
    positions: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    gross: Performance
    net: Performance
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChapterData:
    dates: pd.DatetimeIndex
    open_prices: np.ndarray
    close_prices: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    target: np.ndarray
    split: int


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=4)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    return materialize_chapter_archive(PROJECT_ROOT, manifest, payload, force=force)


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def simple_returns(prices: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    result = np.full(values.shape, np.nan)
    result[lag:] = (values[lag:] - values[:-lag]) / values[:-lag]
    return result


def load_chapter_data() -> ChapterData:
    payload = loadmat(SPY_PATH, squeeze_me=True)
    close_prices = np.asarray(payload["cl"], dtype=float)
    open_prices = np.asarray(payload["op"], dtype=float)
    dates = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(payload["tday"], dtype=int).astype(str), format="%Y%m%d")
    )
    if not (len(dates) == len(close_prices) == len(open_prices)):
        raise ValueError("SPY arrays have inconsistent lengths")
    if np.any(~np.isfinite(close_prices)) or np.any(close_prices <= 0):
        raise ValueError("SPY closes must be finite and positive")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("SPY dates must be strictly increasing")
    returns = simple_returns(close_prices, 1)
    features = np.column_stack(
        [simple_returns(close_prices, lag) for lag in (1, 2, 5, 20)]
    )
    target = np.r_[returns[1:], np.nan]
    return ChapterData(
        dates=dates,
        open_prices=open_prices,
        close_prices=close_prices,
        returns=returns,
        features=features,
        target=target,
        split=len(dates) // 2,
    )


def calculate_performance(returns: np.ndarray, positions: np.ndarray) -> Performance:
    clean = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[:-1]
    peaks = np.maximum(peaks, wealth)
    drawdown = wealth / peaks - 1.0
    duration = 0
    longest = 0
    for value in drawdown:
        duration = duration + 1 if value < 0 else 0
        longest = max(longest, duration)
    standard_deviation = float(np.std(clean, ddof=1))
    sharpe = (
        float(np.sqrt(252.0) * np.mean(clean) / standard_deviation)
        if standard_deviation > 0
        else None
    )
    annual_return = float(wealth[-1] ** (252.0 / len(clean)) - 1.0)
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    return Performance(
        annual_return=annual_return,
        sharpe=sharpe,
        maximum_drawdown=float(np.min(drawdown)),
        drawdown_duration=int(longest),
        cumulative_return=float(wealth[-1] - 1.0),
        mean_daily_return=float(np.mean(clean)),
        annual_turnover=float(np.mean(turnover) * 252.0),
    )


def backtest_signals(
    name: str,
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    one_way_cost_bps: float = ONE_WAY_COST_BPS,
) -> StrategyResult:
    raw_signals = np.nan_to_num(np.asarray(signals, dtype=float), nan=0.0)
    positions = np.r_[0.0, raw_signals[:-1]]
    gross_returns = positions * np.nan_to_num(returns, nan=0.0)
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    net_returns = gross_returns - turnover * one_way_cost_bps / 10_000.0
    return StrategyResult(
        name=name,
        signals=raw_signals,
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        gross=calculate_performance(gross_returns, positions),
        net=calculate_performance(net_returns, positions),
        metadata={"one_way_cost_bps": one_way_cost_bps, **(metadata or {})},
    )


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    degrees_of_freedom = len(y) - design.shape[1]
    variance = float(residuals @ residuals / degrees_of_freedom)
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    return coefficients, p_values


def linear_regression_replay(data: ChapterData) -> dict[str, Any]:
    train_index = np.arange(data.split)
    complete = np.all(np.isfinite(data.features[train_index]), axis=1) & np.isfinite(
        data.target[train_index]
    )
    coefficients, p_values = ols_fit(
        data.features[train_index][complete], data.target[train_index][complete]
    )
    output: dict[str, Any] = {
        "coefficients": coefficients,
        "p_values": p_values,
        "observations": int(np.count_nonzero(complete)),
    }
    for label, index in (
        ("train", train_index),
        ("test", np.arange(data.split, len(data.dates))),
    ):
        prediction = np.full(len(index), np.nan)
        finite = np.all(np.isfinite(data.features[index]), axis=1)
        prediction[finite] = (
            np.column_stack([np.ones(np.count_nonzero(finite)), data.features[index][finite]])
            @ coefficients
        )
        output[label] = backtest_signals(
            f"linear_{label}",
            np.sign(prediction),
            data.returns[index],
            metadata={
                "reproduction": "source-faithful exact replay",
                "fit_end": str(data.dates[data.split - 1].date()),
                "last_fit_predictor_date": str(data.dates[data.split - 1].date()),
                "last_fit_target_realization_date": str(data.dates[data.split].date()),
                "source_boundary_target_leakage": True,
                "segment_start": str(data.dates[index[0]].date()),
                "segment_end": str(data.dates[index[-1]].date()),
            },
        )
    causal_x, causal_y, causal_index = valid_training_arrays(data)
    causal_coefficients, causal_p_values = ols_fit(causal_x, causal_y)
    test_index = np.arange(data.split, len(data.dates))
    causal_prediction = (
        np.column_stack([np.ones(len(test_index)), data.features[test_index]])
        @ causal_coefficients
    )
    output["causal"] = {
        "coefficients": causal_coefficients,
        "p_values": causal_p_values,
        "observations": len(causal_x),
        "test": backtest_signals(
            "linear_causal_test",
            np.sign(causal_prediction),
            data.returns[test_index],
            metadata={
                "reproduction": "boundary-corrected Python adaptation",
                "last_fit_predictor_date": str(data.dates[causal_index[-1]].date()),
                "last_fit_target_realization_date": str(
                    data.dates[causal_index[-1] + 1].date()
                ),
                "test_start": str(data.dates[data.split].date()),
                "source_boundary_target_leakage": False,
            },
        ),
    }
    return output


def select_stepwise_predictor(
    x: np.ndarray, y: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, dict[str, float]]:
    candidate_p_values: dict[str, float] = {}
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        _, p_values = ols_fit(x[:, [feature_index]], y)
        candidate_p_values[feature_name] = float(p_values[-1])
    selected_index = min(
        range(len(FEATURE_NAMES)), key=lambda index: candidate_p_values[FEATURE_NAMES[index]]
    )
    if candidate_p_values[FEATURE_NAMES[selected_index]] >= 0.05:
        raise AssertionError("No stepwise predictor passes the 5% entry threshold")
    coefficients, p_values = ols_fit(x[:, [selected_index]], y)
    return selected_index, coefficients, p_values, candidate_p_values


def forward_stepwise_fit(data: ChapterData) -> dict[str, Any]:
    train_index = np.arange(data.split)
    complete = np.all(np.isfinite(data.features[train_index]), axis=1) & np.isfinite(
        data.target[train_index]
    )
    x = data.features[train_index][complete]
    y = data.target[train_index][complete]
    selected_index, coefficients, p_values, candidate_p_values = (
        select_stepwise_predictor(x, y)
    )
    result: dict[str, Any] = {
        "selected": [FEATURE_NAMES[selected_index]],
        "selected_index": selected_index,
        "coefficients": coefficients,
        "p_values": p_values,
        "candidate_p_values": candidate_p_values,
        "observations": int(np.count_nonzero(complete)),
        "complete_case_rule": "all four candidate predictors and next-day target",
    }
    for label, index in (
        ("train", train_index),
        ("test", np.arange(data.split, len(data.dates))),
    ):
        prediction = coefficients[0] + coefficients[1] * data.features[index, selected_index]
        result[label] = backtest_signals(
            f"stepwise_{label}",
            np.sign(prediction),
            data.returns[index],
            metadata={
                "reproduction": "source-faithful exact replay",
                "selected_predictors": [FEATURE_NAMES[selected_index]],
                "fit_end": str(data.dates[data.split - 1].date()),
                "last_fit_predictor_date": str(data.dates[data.split - 1].date()),
                "last_fit_target_realization_date": str(data.dates[data.split].date()),
                "source_boundary_target_leakage": True,
            },
        )
    causal_x, causal_y, causal_index = valid_training_arrays(data)
    (
        causal_selected_index,
        causal_coefficients,
        causal_p_values,
        causal_candidate_p_values,
    ) = select_stepwise_predictor(causal_x, causal_y)
    test_index = np.arange(data.split, len(data.dates))
    causal_prediction = (
        causal_coefficients[0]
        + causal_coefficients[1] * data.features[test_index, causal_selected_index]
    )
    result["causal"] = {
        "selected": [FEATURE_NAMES[causal_selected_index]],
        "coefficients": causal_coefficients,
        "p_values": causal_p_values,
        "candidate_p_values": causal_candidate_p_values,
        "observations": len(causal_x),
        "test": backtest_signals(
            "stepwise_causal_test",
            np.sign(causal_prediction),
            data.returns[test_index],
            metadata={
                "reproduction": "boundary-corrected Python adaptation",
                "selected_predictors": [FEATURE_NAMES[causal_selected_index]],
                "last_fit_predictor_date": str(data.dates[causal_index[-1]].date()),
                "last_fit_target_realization_date": str(
                    data.dates[causal_index[-1] + 1].date()
                ),
                "test_start": str(data.dates[data.split].date()),
                "source_boundary_target_leakage": False,
            },
        ),
    }
    return result


def tree_rule_replay(data: ChapterData) -> dict[str, StrategyResult]:
    output: dict[str, StrategyResult] = {}
    segments = {
        "train": (np.arange(data.split), 0.01531, -0.01392),
        "test": (np.arange(data.split, len(data.dates)), 0.015314, -0.0139236),
    }
    for label, (index, ret2_threshold, ret1_threshold) in segments.items():
        ret1 = data.features[index, 0]
        ret2 = data.features[index, 1]
        signals = np.zeros(len(index))
        signals[ret2 >= ret2_threshold] = -1.0
        signals[(ret2 < ret2_threshold) & (ret1 < ret1_threshold)] = 1.0
        output[label] = backtest_signals(
            f"tree_rule_{label}",
            signals,
            data.returns[index],
            metadata={
                "reproduction": "exact published extreme-leaf rule replay",
                "ret2_short_threshold": ret2_threshold,
                "ret1_long_threshold": ret1_threshold,
                "zero_signal_means_flat": True,
                "thresholds_inherit_source_boundary_target_leakage": True,
            },
        )
    return output


def hmm_signals(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations = (np.asarray(returns) >= 0).astype(int)
    emission_forecast = np.full((2, len(observations)), np.nan)
    alpha = HMM_PRIOR * HMM_EMISSION[:, observations[0]]
    alpha = alpha / alpha.sum()
    for index in range(len(observations) - 1):
        if index > 0:
            alpha = (alpha @ HMM_TRANSITION) * HMM_EMISSION[:, observations[index]]
            alpha = alpha / alpha.sum()
        emission_forecast[:, index + 1] = HMM_EMISSION.T @ HMM_TRANSITION.T @ alpha
    signals = np.where(
        emission_forecast[0] <= emission_forecast[1], 1.0, -1.0
    )
    return signals, emission_forecast


def hmm_replay(data: ChapterData) -> dict[str, Any]:
    signals, probabilities = hmm_signals(data.returns)
    output: dict[str, Any] = {
        "prior": HMM_PRIOR,
        "transition": HMM_TRANSITION,
        "emission": HMM_EMISSION,
        "probabilities": probabilities,
        "reproduction": "published-parameter exact replay; no HMM re-estimation",
    }
    for label, index in (
        ("train", np.arange(data.split)),
        ("test", np.arange(data.split, len(data.dates))),
    ):
        output[label] = backtest_signals(
            f"hmm_{label}",
            signals[index],
            data.returns[index],
            metadata={
                "reproduction": "published-parameter exact replay",
                "fit_scope": "matrices printed by hmm_train.m",
            },
        )
    return output


def valid_training_arrays(data: ChapterData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # target[t] is realized at t+1.  The source MATLAB includes t=split-1,
    # whose target is the first test return.  Modern adaptations must stop at
    # split-2 so every fitted target is observable no later than train_end.
    train_index = np.arange(data.split - 1)
    complete = np.all(np.isfinite(data.features[train_index]), axis=1) & np.isfinite(
        data.target[train_index]
    )
    return data.features[train_index][complete], data.target[train_index][complete], train_index[complete]


def chronological_cv_regression(
    estimator: Any, x: np.ndarray, y: np.ndarray, splits: int = 5
) -> tuple[float, list[dict[str, int]]]:
    errors: list[float] = []
    boundaries: list[dict[str, int]] = []
    for train, validation in TimeSeriesSplit(n_splits=splits).split(x):
        fitted = clone(estimator).fit(x[train], y[train])
        prediction = fitted.predict(x[validation])
        errors.append(float(np.sqrt(mean_squared_error(y[validation], prediction))))
        boundaries.append(
            {
                "train_last": int(train[-1]),
                "validation_first": int(validation[0]),
                "validation_last": int(validation[-1]),
            }
        )
    return float(np.mean(errors)), boundaries


def chronological_cv_classification(
    estimator: Any, x: np.ndarray, labels: np.ndarray, splits: int = 5
) -> tuple[float, list[dict[str, int]]]:
    accuracies: list[float] = []
    boundaries: list[dict[str, int]] = []
    for train, validation in TimeSeriesSplit(n_splits=splits).split(x):
        fitted = clone(estimator).fit(x[train], labels[train])
        prediction = fitted.predict(x[validation])
        accuracies.append(float(accuracy_score(labels[validation], prediction)))
        boundaries.append(
            {
                "train_last": int(train[-1]),
                "validation_first": int(validation[0]),
                "validation_last": int(validation[-1]),
            }
        )
    return float(np.mean(accuracies)), boundaries


def model_strategy(
    name: str,
    estimator: Any,
    data: ChapterData,
    *,
    classification: bool,
    metadata: dict[str, Any],
) -> StrategyResult:
    x_train, y_train, training_index = valid_training_arrays(data)
    target = y_train >= 0 if classification else y_train
    fitted = clone(estimator).fit(x_train, target)
    test_index = np.arange(data.split, len(data.dates))
    prediction = fitted.predict(data.features[test_index])
    signals = np.where(prediction, 1.0, -1.0) if classification else np.sign(prediction)
    return backtest_signals(
        name,
        signals,
        data.returns[test_index],
        metadata={
            "reproduction": "Python methodological adaptation",
            "fit_rows": len(x_train),
            "fit_end": str(data.dates[data.split - 1].date()),
            "last_fit_predictor_date": str(data.dates[training_index[-1]].date()),
            "last_fit_target_realization_date": str(
                data.dates[training_index[-1] + 1].date()
            ),
            "test_start": str(data.dates[data.split].date()),
            **metadata,
        },
    )


def run_python_models(data: ChapterData) -> dict[str, Any]:
    x_train, y_train, training_index = valid_training_arrays(data)
    labels = y_train >= 0
    test_index = np.arange(data.split, len(data.dates))

    tree = DecisionTreeRegressor(min_samples_leaf=100, random_state=RANDOM_SEED)
    tree_rmse, cv_boundaries = chronological_cv_regression(tree, x_train, y_train)

    classification_tree = DecisionTreeClassifier(
        min_samples_leaf=100, random_state=RANDOM_SEED
    )
    tree_accuracy, _ = chronological_cv_classification(
        classification_tree, x_train, labels
    )

    forest = RandomForestRegressor(
        n_estimators=5,
        min_samples_leaf=100,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    forest_rmse, _ = chronological_cv_regression(forest, x_train, y_train)
    forest_a = clone(forest).fit(x_train, y_train).predict(data.features[test_index])
    forest_b = clone(forest).fit(x_train, y_train).predict(data.features[test_index])

    boosting_scores: dict[int, float] = {}
    for estimators in (5, 20, 80, 160):
        candidate = GradientBoostingRegressor(
            n_estimators=estimators,
            min_samples_leaf=100,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
        )
        boosting_scores[estimators], _ = chronological_cv_regression(
            candidate, x_train, y_train
        )
    selected_boosters = min(boosting_scores, key=boosting_scores.get)
    boosting = GradientBoostingRegressor(
        n_estimators=selected_boosters,
        min_samples_leaf=100,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )

    svm_candidates = {
        "linear": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
        "polynomial": make_pipeline(
            StandardScaler(), SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
        ),
        "rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    }
    svm_scores: dict[str, float] = {}
    for name, candidate in svm_candidates.items():
        svm_scores[name], _ = chronological_cv_classification(candidate, x_train, labels)
    selected_svm = max(svm_scores, key=svm_scores.get)

    target_mean = float(np.mean(y_train))
    target_scale = float(np.std(y_train, ddof=1))
    mlp_test_predictions: list[np.ndarray] = []
    mlp_seed_cagrs: list[float] = []
    for seed in range(1, 11):
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(8, 4),
                activation="tanh",
                solver="adam",
                alpha=0.01,
                max_iter=500,
                early_stopping=False,
                random_state=seed,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(x_train, (y_train - target_mean) / target_scale)
        prediction = mlp.predict(data.features[test_index]) * target_scale + target_mean
        mlp_test_predictions.append(prediction)
        seed_result = backtest_signals(
            f"mlp_seed_{seed}", np.sign(prediction), data.returns[test_index]
        )
        mlp_seed_cagrs.append(seed_result.gross.annual_return)
    mlp_prediction = np.mean(np.vstack(mlp_test_predictions), axis=0)

    return {
        "tree": model_strategy(
            "python_regression_tree",
            tree,
            data,
            classification=False,
            metadata={"cv_rmse": tree_rmse, "min_samples_leaf": 100},
        ),
        "classification_tree": model_strategy(
            "python_classification_tree",
            classification_tree,
            data,
            classification=True,
            metadata={"cv_accuracy": tree_accuracy, "min_samples_leaf": 100},
        ),
        "random_forest": model_strategy(
            "python_random_forest",
            forest,
            data,
            classification=False,
            metadata={
                "cv_rmse": forest_rmse,
                "n_estimators": 5,
                "max_features": "sqrt",
            },
        ),
        "boosting": model_strategy(
            "python_gradient_boosting",
            boosting,
            data,
            classification=False,
            metadata={
                "selected_estimators": selected_boosters,
                "cv_rmse_by_estimators": boosting_scores,
            },
        ),
        "svm": model_strategy(
            "python_svm",
            svm_candidates[selected_svm],
            data,
            classification=True,
            metadata={
                "selected_kernel": selected_svm,
                "cv_accuracy_by_kernel": svm_scores,
            },
        ),
        "mlp_ensemble": backtest_signals(
            "python_mlp_ensemble",
            np.sign(mlp_prediction),
            data.returns[test_index],
            metadata={
                "reproduction": "deterministic 10-seed Python adaptation",
                "seeds": list(range(1, 11)),
                "hidden_layers": [8, 4],
                "internal_validation": "none; fixed 500-iteration training budget",
                "seed_cagrs": mlp_seed_cagrs,
                "fit_end": str(data.dates[data.split - 1].date()),
                "last_fit_predictor_date": str(data.dates[training_index[-1]].date()),
                "last_fit_target_realization_date": str(
                    data.dates[training_index[-1] + 1].date()
                ),
            },
        ),
        "cv_boundaries": cv_boundaries,
        "random_forest_repeat_equal": bool(np.array_equal(forest_a, forest_b)),
    }


def run_experiments(data: ChapterData) -> dict[str, Any]:
    return {
        "linear": linear_regression_replay(data),
        "stepwise": forward_stepwise_fit(data),
        "tree_rule": tree_rule_replay(data),
        "hmm": hmm_replay(data),
        "python_models": run_python_models(data),
        "unavailable": {
            "technical_stock_selection": {
                "status": "unavailable_data",
                "test": None,
                "reason": "rTree_SPX.m loads a private fundamentalData file absent from the official ZIP",
                "book_output": {
                    "annual_return": BOOK_RESULTS["technical_stock_test_cagr"],
                    "sharpe": BOOK_RESULTS["technical_stock_test_sharpe"],
                },
            },
            "fundamental_stock_selection": {
                "status": "unavailable_data",
                "test": None,
                "reason": "stepwiseLR_SPX.m loads the author's local Dropbox fundamentalData file",
                "book_output": {
                    "annual_return": BOOK_RESULTS["fundamental_stock_test_cagr"],
                    "sharpe": BOOK_RESULTS["fundamental_stock_test_sharpe"],
                },
            },
        },
    }


def wealth_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))


def create_model_map_figure() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.axis("off")
    boxes = [
        (0.03, "Observed data\nSPY close"),
        (0.22, "Lagged features\n1, 2, 5, 20 days"),
        (0.43, "Train-only fit\nchronological CV"),
        (0.64, "Next-day signal\nlong / flat / short"),
        (0.83, "Lagged P&L\ngross and net risk"),
    ]
    for x, label in boxes:
        axis.text(
            x,
            0.52,
            label,
            ha="left",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#e9f2ff", "edgecolor": "#24527a"},
            transform=axis.transAxes,
        )
    for start, end in zip((0.16, 0.37, 0.58, 0.78), (0.21, 0.42, 0.63, 0.82)):
        axis.annotate(
            "",
            xy=(end, 0.52),
            xytext=(start, 0.52),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.6},
        )
    axis.text(
        0.5,
        0.1,
        "Modern adaptations exclude all test observations from fitting and model selection",
        ha="center",
        color="#a02323",
        weight="bold",
        transform=axis.transAxes,
    )
    figure.suptitle("Chapter 4 causal model-to-trade contract")
    figure.tight_layout()
    return figure


def create_data_diagnostics_figure(data: ChapterData) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    normalized = data.close_prices / data.close_prices[0]
    axes[0, 0].plot(data.dates, normalized, color="#24527a", lw=1.2)
    axes[0, 0].axvline(data.dates[data.split], color="#a02323", ls="--", label="test start")
    axes[0, 0].set_title("Normalized SPY close and fixed split")
    axes[0, 0].legend()
    finite_returns = data.returns[np.isfinite(data.returns)]
    axes[0, 1].hist(finite_returns, bins=70, color="#5a8f5a", alpha=0.85)
    axes[0, 1].set_title("Daily simple-return distribution")
    correlations = [
        pd.Series(finite_returns).autocorr(lag=lag) for lag in range(1, 21)
    ]
    axes[1, 0].bar(np.arange(1, 21), correlations, color="#8b6f47")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_title("Return autocorrelation")
    feature_frame = pd.DataFrame(data.features, columns=FEATURE_NAMES)
    image = axes[1, 1].imshow(feature_frame.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 1].set_xticks(range(4), FEATURE_NAMES)
    axes[1, 1].set_yticks(range(4), FEATURE_NAMES)
    axes[1, 1].set_title("Overlapping-feature correlation")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    figure.tight_layout()
    return figure


def create_book_replay_figure(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    train_dates = data.dates[: data.split]
    test_dates = data.dates[data.split :]
    for name, result in (
        ("Linear", results["linear"]["train"]),
        ("Stepwise", results["stepwise"]["train"]),
        ("Tree rule", results["tree_rule"]["train"]),
        ("HMM", results["hmm"]["train"]),
    ):
        axes[0].plot(train_dates, wealth_curve(result.gross_returns), label=name, lw=1.1)
    for name, result in (
        ("Linear", results["linear"]["test"]),
        ("Stepwise", results["stepwise"]["test"]),
        ("Tree rule", results["tree_rule"]["test"]),
        ("HMM", results["hmm"]["test"]),
    ):
        axes[1].plot(test_dates, wealth_curve(result.gross_returns), label=name, lw=1.1)
    axes[0].set_title("Book-faithful train replay")
    axes[1].set_title("Book-faithful out-of-sample replay")
    for axis in axes:
        axis.axhline(1.0, color="black", lw=0.7)
        axis.set_ylabel("Growth of $1")
        axis.legend()
    figure.tight_layout()
    return figure


def python_model_results(results: dict[str, Any]) -> dict[str, StrategyResult]:
    models = results["python_models"]
    return {
        key: models[key]
        for key in (
            "tree",
            "classification_tree",
            "random_forest",
            "boosting",
            "svm",
            "mlp_ensemble",
        )
    }


def create_python_model_figure(results: dict[str, Any]) -> plt.Figure:
    models = python_model_results(results)
    names = list(models)
    gross = [models[name].gross.annual_return for name in names]
    net = [models[name].net.annual_return for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(names))
    width = 0.38
    axes[0].bar(x - width / 2, gross, width, label="gross", color="#3677a8")
    axes[0].bar(x + width / 2, net, width, label="net at 2 bps", color="#b05a47")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_title("Python adaptations: OOS CAGR")
    axes[0].legend()
    for name, result in models.items():
        axes[1].plot(wealth_curve(result.gross_returns), label=name, lw=1.0)
    axes[1].axhline(1.0, color="black", lw=0.7)
    axes[1].set_title("Python adaptations: OOS gross wealth")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    return figure


def create_seed_stability_figure(results: dict[str, Any]) -> plt.Figure:
    seed_cagrs = results["python_models"]["mlp_ensemble"].metadata["seed_cagrs"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(range(1, 11), seed_cagrs, color="#8064a2")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("OOS CAGR")
    axes[0].set_title("MLP seed sensitivity")
    cv = results["python_models"]["svm"].metadata["cv_accuracy_by_kernel"]
    axes[1].bar(list(cv), list(cv.values()), color="#4f8b6d")
    axes[1].set_ylim(0.45, max(0.65, max(cv.values()) + 0.03))
    axes[1].set_ylabel("Mean time-series CV accuracy")
    axes[1].set_title("SVM kernel selection on train only")
    figure.tight_layout()
    return figure


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], plt.Figure]] = {
        "model_map.png": create_model_map_figure,
        "data_diagnostics.png": lambda: create_data_diagnostics_figure(data),
        "book_replays.png": lambda: create_book_replay_figure(data, results),
        "python_models.png": lambda: create_python_model_figure(results),
        "seed_stability.png": lambda: create_seed_stability_figure(results),
    }
    for filename, factory in figures.items():
        figure = factory()
        figure.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)


def check_close(actual: float, expected: float, tolerance: float = 5e-7) -> bool:
    return bool(abs(actual - expected) <= tolerance)


def source_incomplete_markers() -> dict[str, str]:
    expected = {
        "hmm.m": "INCOMPLETE!",
        "hmm_train2.m": "Does not work",
        "nn_retrain.m": "INCOMPLETE!",
        "stepwiseLR_SPX_fillFwd.m": "TODO",
    }
    return {
        filename: marker
        for filename, marker in expected.items()
        if marker in (SOURCE_DIR / filename).read_text(encoding="utf-8", errors="replace")
    }


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    extraction = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    linear = results["linear"]
    stepwise = results["stepwise"]
    tree = results["tree_rule"]
    hmm = results["hmm"]
    models = python_model_results(results)
    first_test_position = linear["test"].positions[0]
    all_costs_non_improving = all(
        result.net.cumulative_return <= result.gross.cumulative_return + 1e-12
        for result in (
            linear["train"],
            linear["test"],
            linear["causal"]["test"],
            stepwise["train"],
            stepwise["test"],
            stepwise["causal"]["test"],
            tree["train"],
            tree["test"],
            hmm["train"],
            hmm["test"],
            *models.values(),
        )
    )
    cv_ordered = all(
        boundary["train_last"] < boundary["validation_first"]
        for boundary in results["python_models"]["cv_boundaries"]
    )
    _, _, causal_training_index = valid_training_arrays(data)
    fundamental_present = any(RAW_DIR.glob("*fundamentalData*"))
    return {
        "archive_manifest_matches": (
            extraction["archive_sha256"] == chapter_manifest()["sha256"]
            and len(extraction["members"]) == 24
        ),
        "data_dates_are_strictly_increasing": bool(
            data.dates.is_monotonic_increasing and not data.dates.has_duplicates
        ),
        "linear_coefficients_match_source": bool(
            np.allclose(linear["coefficients"], BOOK_LINEAR_COEFFICIENTS, atol=2e-14, rtol=0)
        ),
        "linear_train_matches_source": check_close(
            linear["train"].gross.annual_return, BOOK_RESULTS["linear_train_cagr"]
        ) and check_close(
            linear["train"].gross.sharpe, BOOK_RESULTS["linear_train_sharpe"]
        ),
        "linear_test_matches_source": check_close(
            linear["test"].gross.annual_return, BOOK_RESULTS["linear_test_cagr"]
        ) and check_close(
            linear["test"].gross.sharpe, BOOK_RESULTS["linear_test_sharpe"]
        ),
        "stepwise_selects_ret2": stepwise["selected"] == ["ret2"],
        "stepwise_coefficients_match_source": bool(
            np.allclose(stepwise["coefficients"], BOOK_STEPWISE_COEFFICIENTS, atol=5e-10, rtol=0)
        ),
        "stepwise_train_matches_source": check_close(
            stepwise["train"].gross.annual_return, BOOK_RESULTS["stepwise_train_cagr"]
        ) and check_close(
            stepwise["train"].gross.sharpe, BOOK_RESULTS["stepwise_train_sharpe"]
        ),
        "stepwise_test_matches_source": check_close(
            stepwise["test"].gross.annual_return, BOOK_RESULTS["stepwise_test_cagr"]
        ) and check_close(
            stepwise["test"].gross.sharpe, BOOK_RESULTS["stepwise_test_sharpe"]
        ),
        "tree_rule_train_matches_source": check_close(
            tree["train"].gross.annual_return, BOOK_RESULTS["tree_rule_train_cagr"]
        ) and check_close(
            tree["train"].gross.sharpe, BOOK_RESULTS["tree_rule_train_sharpe"]
        ),
        "tree_rule_test_matches_source": check_close(
            tree["test"].gross.annual_return, BOOK_RESULTS["tree_rule_test_cagr"]
        ) and check_close(
            tree["test"].gross.sharpe, BOOK_RESULTS["tree_rule_test_sharpe"]
        ),
        "hmm_train_matches_source": check_close(
            hmm["train"].gross.annual_return, BOOK_RESULTS["hmm_train_cagr"]
        ) and check_close(
            hmm["train"].gross.sharpe, BOOK_RESULTS["hmm_train_sharpe"]
        ),
        "hmm_test_matches_source": check_close(
            hmm["test"].gross.annual_return, BOOK_RESULTS["hmm_test_cagr"]
        ) and check_close(
            hmm["test"].gross.sharpe, BOOK_RESULTS["hmm_test_sharpe"]
        ),
        "hmm_probabilities_are_stochastic": bool(
            np.allclose(HMM_TRANSITION.sum(axis=1), 1)
            and np.allclose(HMM_EMISSION.sum(axis=1), 1)
            and np.isclose(HMM_PRIOR.sum(), 1)
        ),
        "signals_are_applied_one_day_later": bool(first_test_position == 0),
        "chronological_split_is_disjoint": bool(
            data.dates[data.split - 1] < data.dates[data.split]
        ),
        "source_boundary_target_leakage_detected": bool(
            data.target[data.split - 1] == data.returns[data.split]
            and linear["test"].metadata["source_boundary_target_leakage"]
            and stepwise["test"].metadata["source_boundary_target_leakage"]
        ),
        "python_training_targets_precede_test": bool(
            causal_training_index[-1] + 1 < data.split
            and data.dates[causal_training_index[-1] + 1] < data.dates[data.split]
            and not linear["causal"]["test"].metadata[
                "source_boundary_target_leakage"
            ]
            and not stepwise["causal"]["test"].metadata[
                "source_boundary_target_leakage"
            ]
        ),
        "time_series_cv_respects_order": cv_ordered,
        "transaction_costs_do_not_improve_results": all_costs_non_improving,
        "random_forest_is_seed_deterministic": results["python_models"][
            "random_forest_repeat_equal"
        ],
        "cross_sectional_data_is_not_claimed_available": bool(
            not fundamental_present
            and results["unavailable"]["technical_stock_selection"]["test"] is None
            and results["unavailable"]["fundamental_stock_selection"]["test"] is None
        ),
        "incomplete_matlab_scripts_are_disclosed": len(source_incomplete_markers()) == 4,
    }


def verification_summary(checks: dict[str, bool]) -> dict[str, Any]:
    counts = {"independent_or_empirical": 0, "contract_invariant": 0}
    for name, passed in checks.items():
        if name not in VERIFICATION_CLASSES:
            raise KeyError(f"Missing verification class: {name}")
        if passed:
            counts[VERIFICATION_CLASSES[name]] += 1
    return {
        "total": len(checks),
        "passed": int(sum(checks.values())),
        "failed": [name for name, passed in checks.items() if not passed],
        **counts,
        "classification": VERIFICATION_CLASSES,
    }


def strategy_metrics(result: StrategyResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "gross": asdict(result.gross),
        "net": asdict(result.net),
        "metadata": result.metadata,
        "observations": len(result.gross_returns),
        "active_fraction": float(np.mean(np.abs(result.positions) > 0)),
        "long_fraction": float(np.mean(result.positions > 0)),
        "short_fraction": float(np.mean(result.positions < 0)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    # bool is a subclass of int in Python, so preserve JSON boolean semantics
    # before handling integer scalars.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def book_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = (
        ("linear train CAGR", results["linear"]["train"].gross.annual_return, "linear_train_cagr"),
        ("linear train Sharpe", results["linear"]["train"].gross.sharpe, "linear_train_sharpe"),
        ("linear test CAGR", results["linear"]["test"].gross.annual_return, "linear_test_cagr"),
        ("linear test Sharpe", results["linear"]["test"].gross.sharpe, "linear_test_sharpe"),
        ("stepwise train CAGR", results["stepwise"]["train"].gross.annual_return, "stepwise_train_cagr"),
        ("stepwise train Sharpe", results["stepwise"]["train"].gross.sharpe, "stepwise_train_sharpe"),
        ("stepwise test CAGR", results["stepwise"]["test"].gross.annual_return, "stepwise_test_cagr"),
        ("stepwise test Sharpe", results["stepwise"]["test"].gross.sharpe, "stepwise_test_sharpe"),
        ("tree-rule train CAGR", results["tree_rule"]["train"].gross.annual_return, "tree_rule_train_cagr"),
        ("tree-rule train Sharpe", results["tree_rule"]["train"].gross.sharpe, "tree_rule_train_sharpe"),
        ("tree-rule test CAGR", results["tree_rule"]["test"].gross.annual_return, "tree_rule_test_cagr"),
        ("tree-rule test Sharpe", results["tree_rule"]["test"].gross.sharpe, "tree_rule_test_sharpe"),
        ("HMM train CAGR", results["hmm"]["train"].gross.annual_return, "hmm_train_cagr"),
        ("HMM train Sharpe", results["hmm"]["train"].gross.sharpe, "hmm_train_sharpe"),
        ("HMM test CAGR", results["hmm"]["test"].gross.annual_return, "hmm_test_cagr"),
        ("HMM test Sharpe", results["hmm"]["test"].gross.sharpe, "hmm_test_sharpe"),
    )
    return [
        {
            "metric": label,
            "python": actual,
            "book_or_source": BOOK_RESULTS[key],
            "absolute_error": abs(actual - BOOK_RESULTS[key]),
            "tolerance": 5e-7,
            "passed": check_close(actual, BOOK_RESULTS[key]),
        }
        for label, actual, key in pairs
    ]


def reference_only_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose non-comparable MATLAB outputs without claiming numerical replication."""
    models = results["python_models"]
    return [
        {
            "topic": "bagging/random subspace",
            "source_file": "rTreeBagger.m",
            "book_or_source": {
                "annual_return": BOOK_RESULTS["bagging_test_cagr"],
                "sharpe": BOOK_RESULTS["bagging_test_sharpe"],
            },
            "python_adaptation": {
                "annual_return": models["random_forest"].gross.annual_return,
                "sharpe": models["random_forest"].gross.sharpe,
            },
            "compared": False,
            "reason": (
                "MATLAB TreeBagger split and random-subspace defaults differ from "
                "scikit-learn; values are output-only references"
            ),
        },
        {
            "topic": "classification tree",
            "source_file": "cTree.m adjacent comment; duplicated in five MATLAB scripts",
            "book_or_source": {
                "annual_return": BOOK_RESULTS["classification_tree_test_cagr"],
                "sharpe": BOOK_RESULTS["classification_tree_test_sharpe"],
            },
            "python_adaptation": {
                "annual_return": models["classification_tree"].gross.annual_return,
                "sharpe": models["classification_tree"].gross.sharpe,
            },
            "compared": False,
            "reason": (
                "The identical output comment appears in rTree.m, cTree.m, hmm.m, "
                "hmm_VXX.m, and hmm2.m, so provenance is uncertain"
            ),
        },
        {
            "topic": "support vector machine",
            "source_file": "svm.m",
            "book_or_source": {
                "annual_return": BOOK_RESULTS["svm_test_cagr"],
                "sharpe": BOOK_RESULTS["svm_test_sharpe"],
            },
            "python_adaptation": {
                "annual_return": models["svm"].gross.annual_return,
                "sharpe": models["svm"].gross.sharpe,
            },
            "compared": False,
            "reason": "random-fold MATLAB estimator differs from train-only ordered-CV adaptation",
        },
        {
            "topic": "neural-network average",
            "source_file": "nn_feedfwd_avg.m",
            "book_or_source": {
                "annual_return": BOOK_RESULTS["nn_average_test_cagr"],
                "sharpe": BOOK_RESULTS["nn_average_test_sharpe"],
            },
            "python_adaptation": {
                "annual_return": models["mlp_ensemble"].gross.annual_return,
                "sharpe": models["mlp_ensemble"].gross.sharpe,
            },
            "compared": False,
            "reason": "network architecture, solver, validation, and RNG differ",
        },
    ]


def build_metrics(
    data: ChapterData, results: dict[str, Any], checks: dict[str, bool]
) -> dict[str, Any]:
    extraction = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = chapter_manifest()
    metrics = {
        "chapter": 4,
        "title": "Artificial Intelligence Techniques",
        "provenance": {
            "source_page": manifest["source_page"],
            "archive_url": manifest["url"],
            "archive_sha256": manifest["sha256"],
            "archive_size_bytes": manifest["size_bytes"],
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
            "member_count": len(extraction["members"]),
            "input_mat_sha256": next(
                member["sha256"]
                for member in extraction["members"]
                if member["archive_path"].endswith("inputData_SPY.mat")
            ),
            "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
        },
        "environment": environment_versions(
            (
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "scikit-learn",
                "nbformat",
                "nbclient",
            )
        ),
        "data": {
            "file": str(SPY_PATH.relative_to(PROJECT_ROOT)),
            "observations": len(data.dates),
            "start": str(data.dates[0].date()),
            "end": str(data.dates[-1].date()),
            "train_observations": data.split,
            "test_observations": len(data.dates) - data.split,
            "train_end": str(data.dates[data.split - 1].date()),
            "test_start": str(data.dates[data.split].date()),
            "missing_close": int(np.count_nonzero(~np.isfinite(data.close_prices))),
            "nonpositive_close": int(np.count_nonzero(data.close_prices <= 0)),
            "complete_model_rows_train": int(
                np.count_nonzero(
                    np.all(np.isfinite(data.features[: data.split]), axis=1)
                    & np.isfinite(data.target[: data.split])
                )
            ),
            "causal_model_rows_train": len(valid_training_arrays(data)[0]),
            "feature_names": FEATURE_NAMES,
        },
        "reproduction_classification": {
            "linear_regression": "source-faithful exact replay with disclosed one-row boundary leakage",
            "stepwise_regression": "source-faithful exact replay with disclosed one-row boundary leakage",
            "extreme_tree_rule": "source-faithful exact rule replay; thresholds inherit source fit",
            "hmm": "published-parameter exact replay",
            "sklearn_models": "methodological adaptation",
            "cross_sectional_models": "output-only; unavailable official input",
        },
        "book_comparisons": book_comparisons(results),
        "reference_only_comparisons": reference_only_comparisons(results),
        "strategies": {
            "linear": {
                "coefficients": results["linear"]["coefficients"],
                "p_values": results["linear"]["p_values"],
                "train": strategy_metrics(results["linear"]["train"]),
                "test": strategy_metrics(results["linear"]["test"]),
                "causal": {
                    "coefficients": results["linear"]["causal"]["coefficients"],
                    "observations": results["linear"]["causal"]["observations"],
                    "test": strategy_metrics(results["linear"]["causal"]["test"]),
                },
            },
            "stepwise": {
                "selected": results["stepwise"]["selected"],
                "coefficients": results["stepwise"]["coefficients"],
                "candidate_p_values": results["stepwise"]["candidate_p_values"],
                "train": strategy_metrics(results["stepwise"]["train"]),
                "test": strategy_metrics(results["stepwise"]["test"]),
                "causal": {
                    "selected": results["stepwise"]["causal"]["selected"],
                    "coefficients": results["stepwise"]["causal"]["coefficients"],
                    "observations": results["stepwise"]["causal"]["observations"],
                    "test": strategy_metrics(results["stepwise"]["causal"]["test"]),
                },
            },
            "tree_rule": {
                "train": strategy_metrics(results["tree_rule"]["train"]),
                "test": strategy_metrics(results["tree_rule"]["test"]),
            },
            "hmm": {
                "prior": HMM_PRIOR,
                "transition": HMM_TRANSITION,
                "emission": HMM_EMISSION,
                "train": strategy_metrics(results["hmm"]["train"]),
                "test": strategy_metrics(results["hmm"]["test"]),
            },
            "python_models": {
                name: strategy_metrics(result)
                for name, result in python_model_results(results).items()
            },
            "unavailable": results["unavailable"],
        },
        "source_limitations": {
            "incomplete_scripts": source_incomplete_markers(),
            "missing_fundamental_data": True,
            "one_row_target_boundary_leakage": (
                "lr.m, stepwiseLR.m, and fitted tree thresholds include the first "
                "test return as the target of the final training predictor row"
            ),
            "duplicated_classification_output_comment": (
                "0.047740/0.366381 appears in five different MATLAB scripts; "
                "treated as a provenance-uncertain output-only reference"
            ),
            "matlab_cross_validation_note": (
                "Original scripts select one fitted fold; Python adaptation uses ordered CV "
                "for train-only model choice and refits on all training rows"
            ),
        },
        "coverage": [
            {"topic": topic, "status": status, "evidence": evidence}
            for topic, status, evidence in CHAPTER_COVERAGE
        ],
        "verification": {"checks": checks, "summary": verification_summary(checks)},
    }
    return json_safe(metrics)


def coverage_table() -> str:
    rows = ["| 주제 | 재현 상태 | 근거 |", "|---|---|---|"]
    rows.extend(f"| {topic} | {status} | {evidence} |" for topic, status, evidence in CHAPTER_COVERAGE)
    return "\n".join(rows)


def result_table(results: dict[str, Any]) -> str:
    rows = [
        "| 실험 | 구간 | Python CAGR | 책/원본 CAGR | Python Sharpe | 책/원본 Sharpe | 분류 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    entries = (
        ("Full linear", "train", results["linear"]["train"], "linear_train_cagr", "linear_train_sharpe", "원본 충실 재현(경계 누수 포함)"),
        ("Full linear", "test", results["linear"]["test"], "linear_test_cagr", "linear_test_sharpe", "원본 충실 재현(경계 누수 포함)"),
        ("Stepwise", "train", results["stepwise"]["train"], "stepwise_train_cagr", "stepwise_train_sharpe", "원본 충실 재현(경계 누수 포함)"),
        ("Stepwise", "test", results["stepwise"]["test"], "stepwise_test_cagr", "stepwise_test_sharpe", "원본 충실 재현(경계 누수 포함)"),
        ("Extreme tree rule", "train", results["tree_rule"]["train"], "tree_rule_train_cagr", "tree_rule_train_sharpe", "원본 threshold 재현"),
        ("Extreme tree rule", "test", results["tree_rule"]["test"], "tree_rule_test_cagr", "tree_rule_test_sharpe", "원본 threshold 재현"),
        ("HMM", "train", results["hmm"]["train"], "hmm_train_cagr", "hmm_train_sharpe", "계수 replay"),
        ("HMM", "test", results["hmm"]["test"], "hmm_test_cagr", "hmm_test_sharpe", "계수 replay"),
    )
    for name, segment, result, cagr_key, sharpe_key, classification in entries:
        rows.append(
            f"| {name} | {segment} | {result.gross.annual_return:.6f} | "
            f"{BOOK_RESULTS[cagr_key]:.6f} | {result.gross.sharpe:.6f} | "
            f"{BOOK_RESULTS[sharpe_key]:.6f} | {classification} |"
        )
    return "\n".join(rows)


def reference_table(results: dict[str, Any]) -> str:
    rows = [
        "| 원본 주제 | 원본 CAGR / Sharpe | Python 적응 CAGR / Sharpe | 비교 상태 |",
        "|---|---:|---:|---|",
    ]
    for record in reference_only_comparisons(results):
        source = record["book_or_source"]
        python = record["python_adaptation"]
        rows.append(
            f"| {record['topic']} | {source['annual_return']:.6f} / "
            f"{source['sharpe']:.6f} | {python['annual_return']:.6f} / "
            f"{python['sharpe']:.6f} | output-only reference; 직접 비교 안 함 |"
        )
    return "\n".join(rows)


def write_report(data: ChapterData, results: dict[str, Any], metrics: dict[str, Any]) -> None:
    summary = metrics["verification"]["summary"]
    model_rows = []
    for name, result in python_model_results(results).items():
        model_rows.append(
            f"| {name} | {result.gross.annual_return:.4f} | {result.net.annual_return:.4f} | "
            f"{result.gross.sharpe:.3f} | {result.gross.maximum_drawdown:.3f} |"
        )
    report = f"""# Chapter 4 — Artificial Intelligence Techniques 재현 보고서

## 1. 개요와 문제 정의

이 장의 질문은 “어떤 AI 모형이 가장 높은 백테스트를 만드는가”가 아니다. 같은 네 개의
중첩 수익률 피처로 복잡도를 높였을 때 훈련 성능과 표본 외 성능이 어떻게 갈라지는지,
그리고 검증·시차·비용 계약이 그 차이를 얼마나 정직하게 드러내는지가 핵심이다.

공식 SPY 일봉 {len(data.dates):,}개를 {data.dates[0].date()}부터
{data.dates[-1].date()}까지 사용한다. 달력상 앞 {data.split:,}개는 훈련,
뒤 {len(data.dates)-data.split:,}개는 테스트다. 다만 원본 회귀는 마지막 훈련일의
다음 날 target으로 첫 테스트 수익을 한 행 사용한다. exact replay는 이 경계 누수까지
보존하고, Python 적응은 마지막 predictor 행을 빼 테스트를 완전히 격리한다.

## 2. 핵심 수식과 수식-코드 지도

단순수익률과 다음 날 목표는 다음처럼 정의한다.

$$ r_t(k) = (P_t - P_{{t-k}}) / P_{{t-k}} $$

$$ y_t = r_{{t+1}}(1) $$

| 수식/계약 | 구현 함수 | 검증 |
|---|---|---|
| lagged simple return | `simple_returns` | MATLAB `calculateReturns.m`과 동일 |
| OLS와 coefficient p-value | `ols_fit` | 원본 계수와 수치 비교 |
| stepwise complete-case selection | `forward_stepwise_fit` | ret2 선택과 계수 비교 |
| signal(t) → position(t+1) | `backtest_signals` | 첫 포지션 0 assert |
| HMM one-step observation forecast | `hmm_signals` | 행렬 확률합과 원본 CAGR 비교 |
| ordered cross validation | `chronological_cv_*` | 모든 train 마지막 < validation 첫 행 |

## 3. 구현 범위와 재현 상태

{coverage_table()}

원본 충실 재현은 같은 공식 입력, 같은 식, 같은 분할과 원본의 경계 누수까지 보존해
수치를 다시 계산했다는 뜻이다. 누수를 올바른 설계로 승인한다는 뜻은 아니다.
Python 적응은 라이브러리와 검증 설계를 현대화한 별도 실험이며 MATLAB 숫자와 직접
일치한다고 주장하지 않는다. output-only는 책/소스 주석값만 보존하고 전략 지표는
`null`로 둔다.

## 4. 출처·데이터·환경

- 저자 페이지: {metrics['provenance']['source_page']}
- 직접 ZIP: {metrics['provenance']['archive_url']}
- ZIP SHA-256: `{metrics['provenance']['archive_sha256']}`
- 입력 MAT SHA-256: `{metrics['provenance']['input_mat_sha256']}`
- `uv.lock` SHA-256: `{metrics['provenance']['uv_lock_sha256']}`
- 공식 멤버: {metrics['provenance']['member_count']}개, close 결측 {metrics['data']['missing_close']}개,
  비양수 {metrics['data']['nonpositive_close']}개

`SOURCE_MANIFEST.json`은 MATLAB 22개와 데이터 2개의 개별 checksum을 고정한다.
offline 실행은 네트워크 없이 전 멤버를 재검증한다. `SPY.xls`는 원본 보존용이며 계산은
구조가 명확한 `inputData_SPY.mat`을 사용한다.

## 5. 선형회귀: 복잡한 모델이 자동으로 낫지 않다

{result_table(results)}

네 피처 전체 회귀는 훈련 CAGR {results['linear']['train'].gross.annual_return:.1%}에서
테스트 {results['linear']['test'].gross.annual_return:.1%}로 거의 소멸한다. 반면 stepwise는
`ret2` 하나만 남겨 테스트 {results['stepwise']['test'].gross.annual_return:.1%}를 기록한다.
이는 stepwise가 항상 우월하다는 증거가 아니라 이 한 표본에서 분산 감소가 유리했다는
결과다. 후보와 진입 규칙을 테스트를 본 뒤 바꾸면 selection bias가 된다.

### 5.1 원본의 1행 경계 누수와 수정본

`retFut1=fwdshift(1, ret1)` 뒤 `trainset=1:floor(N/2)`를 그대로 쓰면 훈련 마지막
predictor 날짜 {data.dates[data.split-1].date()}의 target이 첫 테스트일
{data.dates[data.split].date()} 수익이다. 따라서 위 표의 exact test는 엄밀한 완전 격리
OOS가 아니라 **source-defined test replay**다. 수정본은 predictor를
{data.dates[data.split-2].date()}까지만 적합해 마지막 target이 훈련 종료일에 실현되게 한다.

| 모델 | source-defined test CAGR | boundary-corrected test CAGR | source fit rows | corrected fit rows |
|---|---:|---:|---:|---:|
| Full linear | {results['linear']['test'].gross.annual_return:.6f} | {results['linear']['causal']['test'].gross.annual_return:.6f} | {results['linear']['observations']} | {results['linear']['causal']['observations']} |
| Stepwise | {results['stepwise']['test'].gross.annual_return:.6f} | {results['stepwise']['causal']['test'].gross.annual_return:.6f} | {results['stepwise']['observations']} | {results['stepwise']['causal']['observations']} |

모든 sklearn 적응, ordered CV, scaling, MLP target normalization은 1,157개 corrected row만
사용한다. `source_boundary_target_leakage_detected`와
`python_training_targets_precede_test`가 두 경로를 각각 검증한다.

## 6. Regression tree와 extreme-leaf 규칙

원본 tree 전체 예측은 훈련 수익이 매우 높지만 테스트에서 음수가 된다. 책이 별도로
제시한 극단 leaf 두 개만 쓰는 규칙은 flat 상태를 허용해 테스트 CAGR
{results['tree_rule']['test'].gross.annual_return:.1%}, Sharpe
{results['tree_rule']['test'].gross.sharpe:.2f}를 정확 재현한다. leaf threshold가 학습자료에서
선택됐다는 사실은 유지되며, 테스트에 맞춘 재최적화는 하지 않았다.

## 7. Cross validation 교정

MATLAB 예제는 무작위 5-fold를 만든 뒤 loss가 가장 낮은 한 fold의 fitted model을 꺼내 쓴다.
금융 시계열에서는 미래 행이 과거 행의 모델 선택에 섞일 수 있고, 최종 훈련 전체를 다시
적합하지 않는 문제도 있다. Python 적응은 `TimeSeriesSplit`으로 항상 과거→미래 순서를
지키고, 첫 테스트 target을 제외한 훈련 행 전체에 재적합한다. 따라서 아래 수치는 원본 수치 재현이 아니라
검증 설계를 바꾼 방법론적 적응이다.

## 8. Bagging, random subspace, boosting, classification, SVM, NN

| Python 적응 | gross CAGR | net CAGR (2bps) | gross Sharpe | max drawdown |
|---|---:|---:|---:|---:|
{chr(10).join(model_rows)}

Random forest는 MATLAB `TreeBagger(K=5)`와 같은 tree 수와 seed를 쓰지만 split 규칙과
random-subspace 기본값이 달라 직접 수치 비교하지 않는다. classification tree 원본값은
같은 주석이 서로 다른 MATLAB 5개에 복제돼 provenance도 확정할 수 없다. 아래 값은
정확성 판정에 쓰지 않는 output-only reference다.

{reference_table(results)}

Boosting iteration과 SVM kernel은 훈련 구간의 ordered CV만으로 선택한다. MLP는 내부
무작위 validation을 쓰지 않고 고정 500 iteration으로 seed 1~10을 모두 실행한 뒤 평균
예측을 사용한다. seed별 성능 분산은 한 번의 좋은 초기화를 일반화 성능으로 오인하면
안 된다는 진단이다.

## 9. HMM replay와 상태 해석

HMM은 `hmm_train.m`이 출력한 prior, transition, emission을 고정해 온라인 filtering을
replay한다. 훈련 CAGR {results['hmm']['train'].gross.annual_return:.1%}와 테스트
{results['hmm']['test'].gross.annual_return:.1%}가 원본과 일치하지만, 이는 Python에서 EM
최적화를 독립 재현했다는 뜻이 아니다. `hmm.m`에는 `INCOMPLETE!`, `hmm_train2.m`에는
작동하지 않는다는 assert가 있어 published-parameter replay로 분류했다.

## 10. 누락된 cross-sectional 데이터

`rTree_SPX.m`과 `stepwiseLR_SPX.m`은 공식 ZIP에 없는 저자 로컬
`C:/Users/Ernest/Dropbox/AI_WS/fundamentalData`를 읽는다. 따라서 기술 피처 주식선택
2.3%/0.9와 fundamental 주식선택 4.0%/1.1은 output-only다. 다른 최신 S&P 데이터로
바꾸면 survivorship, point-in-time fundamentals, universe가 달라지므로 정확 재현으로
표시할 수 없다. `metrics.json`의 두 `test` 값은 0이 아니라 `null`이며 이유가 함께 있다.

## 11. 거래비용·위험·적용 범위

모든 전략은 신호를 하루 뒤 포지션에 적용한다. 편도 2bps × 포지션 변화량을 차감한
민감도와 gross를 함께 저장한다. 이는 spread, slippage, borrow fee, market impact를 모두
포함한 실거래 견적이 아니며 일봉 회전율의 최소 마찰 진단이다. CAGR과 Sharpe 외에
maximum drawdown, drawdown duration, annual turnover를 기록한다.

이 결과는 SPY 한 종목, 한 번의 고정 분할에 대한 백테스트다. 모델 우열의 보편적 증거나
현재 시장 추천이 아니다. test 결과를 보고 seed, leaf, kernel, 비용을 다시 선택하면 그
test도 훈련자료가 되므로 새 표본 외 기간이 필요하다.

## 12. 편향과 원본 코드 감사

- look-ahead: 원본의 첫 테스트 target 1행 사용을 exact replay에 공개하고 Python 적응에서는 제외했으며, random K-fold도 ordered split으로 교체했다.
- survivorship: SPY 자체에는 구성종목 생존 편향이 없지만 cross-sectional 결과는 원자료가 없어 검증 불가다.
- selection bias: model family, boosting iteration, SVM kernel은 train CV 안에서만 선택했다.
- data snooping: 책에 인쇄된 좋은 모델을 이미 알고 있다는 사후 지식을 분류에 명시했다.
- source completeness: {', '.join(source_incomplete_markers())}의 중단 표식을 보존했다.

## 13. 자동 검증과 결론

검증 {summary['passed']}/{summary['total']}개가 통과했다. 독립/경험 비교
{summary['independent_or_empirical']}개와 계산 계약 invariant
{summary['contract_invariant']}개를 구분했다.

Chapter 4의 가장 재사용 가능한 결론은 “더 복잡한 AI”가 아니라 “더 엄격한 정보 경계”다.
공식 OLS·stepwise·tree rule·HMM 수치와 경계 누수를 함께 보존하면서도 Python 실험은
첫 테스트 target을 제외한 시간순 CV, corrected-train 재적합, seed ensemble, 비용 전후를
별도 계약으로 둔다. 데이터가 없는 주식선택 예제는
숫자를 꾸며내지 않고 output-only로 남긴다. 다른 책에서도 동일하게 원본 수치 replay와
현대적 방법론 적응을 한 표의 다른 행으로 분리해야 한다.
"""
    assert_clean_markdown_math(report, "Chapter 4 report")
    write_text_if_changed(REPORT_DIR / "chapter4_report.md", report)


def write_readme() -> None:
    content = """# Chapter 4 Python experiments

Run from the Machine Trading project directory:

```bash
uv run python chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py
uv run python chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py --offline
```

The first command verifies the official author ZIP and materializes source/data. The offline
command revalidates every SHA-256 without network access, regenerates metrics, five figures,
the Korean report, and the executed notebook. Raw `.mat` and `.xls` inputs stay ignored;
the tracked `SOURCE_MANIFEST.json` pins their hashes.

Source-faithful numerical replays preserve and disclose the MATLAB one-row target-boundary
leakage; corrected linear/stepwise and all scikit-learn adaptations exclude that row. Extreme
tree rules and published HMM parameters are replayed separately. Non-comparable MATLAB outputs
are output-only references, including an ambiguous comment duplicated across five scripts.
Cross-sectional stock examples are output-only because the
official archive does not include the author's private `fundamentalData` input.
"""
    write_text_if_changed(CHAPTER_DIR / "README.md", content)


def build_and_execute_notebook() -> None:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# Machine Trading Chapter 4: Artificial Intelligence Techniques

이 노트북은 공식 SPY 데이터로 책의 선형회귀·stepwise·tree rule·HMM 결과를
수치 재현하고, tree ensemble·SVM·신경망을 시간순 검증으로 다시 구현한다. 목표는
가장 예쁜 CAGR을 고르는 것이 아니라 모델 복잡도, 검증 설계, 거래 시차, 비용이
결론에 미치는 영향을 추적하는 것이다. 원본의 첫 테스트 target 1행 경계 누수도
숨기지 않고 source replay와 corrected adaptation을 분리한다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 1. 학습 질문과 백테스트 범위

1. 네 피처 전체 회귀가 왜 훈련에서는 좋아도 표본 외에서 사라지는가?
2. stepwise와 tree가 분산을 줄이는 방식은 무엇이며 선택 편향은 어디에 생기는가?
3. random K-fold 대신 ordered CV를 쓰면 어떤 정보 경계가 보존되는가?
4. HMM과 NN의 잠재 상태·초기화 불확실성을 어떻게 보고해야 하는가?
5. 편도 2bps와 drawdown을 포함해도 결론이 유지되는가?
6. 원본 `fwdshift`와 절반 분할 사이의 1행 경계 누수를 어떻게 보존하고 수정하는가?

여기서 전략 결과는 하루 lag를 적용한 시간순 백테스트다. 모형 구조 그림, 행렬 확률합,
누락 데이터 설명은 백테스트가 아닌 수치·개념 진단이다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 2. 구현 범위와 재현 상태

원본 충실 재현, published-parameter replay, boundary-corrected Python 방법론 적응,
output-only를 섞지 않는다.
공식 입력이 없는 cross-sectional 예제는 책 숫자만 보존하고 실행 결과는 `null`로 둔다.

""" + coverage_table()
        ),
        nbformat.v4.new_code_cell(
            """from io import BytesIO
from pathlib import Path
import importlib.util
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

def find_project_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "book3_common.py").exists():
            return candidate
    raise FileNotFoundError("Machine Trading project root not found")

PROJECT = find_project_root()
RUNNER = PROJECT / "chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py"
spec = importlib.util.spec_from_file_location("chapter4_analysis", RUNNER)
chapter4 = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = chapter4
spec.loader.exec_module(chapter4)
data = chapter4.load_chapter_data()
results = chapter4.run_experiments(data)

def show_figure(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    display(Image(data=buffer.getvalue()))

print("loaded", len(data.dates), "SPY rows; split", data.split)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 3. 수식과 formula-to-code 지도

단순수익률과 예측 목표는 다음과 같다.

$$ r_t(k) = (P_t - P_{t-k}) / P_{t-k} $$

$$ y_t = r_{t+1}(1) $$

`simple_returns`가 첫 식, `ols_fit`이 선형 추정, `hmm_signals`가 one-step emission,
`backtest_signals`가 signal(t) → position(t+1) → P&L을 구현한다. 수식의 행 번호와
거래 체결 시점을 같은 것으로 착각하면 look-ahead가 생긴다."""
        ),
        nbformat.v4.new_code_cell(
            """formula_map = pd.DataFrame([
    ("simple return", "simple_returns", "lags 1/2/5/20"),
    ("OLS", "ols_fit", "book coefficients"),
    ("stepwise", "forward_stepwise_fit", "ret2 selection"),
    ("HMM forecast", "hmm_signals", "stochastic matrices"),
    ("source/corrected boundary", "valid_training_arrays", "last target precedes test"),
    ("causal P&L", "backtest_signals", "first position is zero"),
], columns=["formula or contract", "implementation", "verification"])
display(formula_map)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 4. 공식 출처, checksum, 잠금 환경

저자 페이지와 직접 ZIP URL, archive SHA-256, 각 멤버 checksum, `uv.lock` SHA-256을
함께 기록한다. URL만 있으면 같은 이름의 파일이 교체돼도 감지하지 못한다. random seed는
PCG64를 직접 쓰는 시뮬레이션 대신 각 sklearn estimator의 `random_state=1`에 고정하며,
MLP는 seed 1~10 전체를 보고한다."""
        ),
        nbformat.v4.new_code_cell(
            """manifest = chapter4.chapter_manifest()
provenance = pd.Series({
    "source_page": manifest["source_page"],
    "archive_url": manifest["url"],
    "archive_sha256": manifest["sha256"],
    "input_sha256": next(m["sha256"] for m in json.loads(chapter4.SOURCE_MANIFEST_PATH.read_text())["members"] if m["archive_path"].endswith("inputData_SPY.mat")),
    "uv.lock SHA-256": chapter4.sha256_file(chapter4.UV_LOCK_PATH),
})
display(provenance.to_frame("value"))
display(pd.Series(chapter4.environment_versions(("numpy","pandas","scipy","matplotlib","scikit-learn","nbformat","nbclient"))).to_frame("environment version"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 5. 데이터 구조·결측·분할 진단

공식 MAT에는 SPY open, close, YYYYMMDD가 있다. close 결측과 비양수 가격을 먼저
거부하고 날짜 중복도 거부한다. 피처는 중첩수익률이라 서로 강하게 상관될 수 있고 첫
20행은 ret20 때문에 비어 있다. MATLAB stepwise는 최종 ret2 모델이라도 네 후보의
완전관측 1,158행을 고정한다. 이 미묘한 행 규칙을 지켜야 원본 계수가 재현되지만,
마지막 행의 target은 첫 테스트 수익이다. corrected 모델은 1,157행만 쓴다."""
        ),
        nbformat.v4.new_code_cell(
            """diagnostics = pd.Series({
    "rows": len(data.dates),
    "start": data.dates[0].date(),
    "end": data.dates[-1].date(),
    "train rows": data.split,
    "test rows": len(data.dates) - data.split,
    "train end": data.dates[data.split-1].date(),
    "test start": data.dates[data.split].date(),
    "missing close": int(np.count_nonzero(~np.isfinite(data.close_prices))),
    "nonpositive close": int(np.count_nonzero(data.close_prices <= 0)),
    "complete train model rows": int(np.count_nonzero(np.all(np.isfinite(data.features[:data.split]), axis=1) & np.isfinite(data.target[:data.split]))),
    "causal train model rows": len(chapter4.valid_training_arrays(data)[0]),
})
display(diagnostics.to_frame("value"))
assert diagnostics["missing close"] == 0 and diagnostics["nonpositive close"] == 0
show_figure(chapter4.create_data_diagnostics_figure(data))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 6. 모형에서 거래까지의 인과 지도

관측 데이터 → lagged feature → boundary-corrected train-only fit → 다음 날 signal →
하루 뒤 P&L의 순서를 고정한다. 테스트 시작일의 position은 반드시 0이다. 검증 fold도
train 마지막 행보다 validation 첫 행이 뒤에 있어야 한다. 이 지도는 modern adaptation
계약이며 source exact replay의 경계 누수는 별도로 보존한다."""
        ),
        nbformat.v4.new_code_cell(
            """show_figure(chapter4.create_model_map_figure())
assert results["linear"]["test"].positions[0] == 0
assert all(b["train_last"] < b["validation_first"] for b in results["python_models"]["cv_boundaries"])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 7. Full linear regression: 훈련 성공의 붕괴

네 피처를 모두 쓴 OLS 계수는 `lr.m`과 거의 machine precision으로 일치한다. 하지만
훈련 CAGR 약 34%가 테스트 약 0.4%로 줄어든다. ret1, ret2, ret5, ret20이 중첩돼
다중공선성이 있고 약한 예측 신호에 비해 추정 자유도가 많다. 훈련 적합도가 높다는 사실은
표본 외 경제적 가치와 같지 않다. 이 exact 숫자는 원본처럼 훈련 마지막 predictor의
target으로 첫 테스트 수익을 사용한다. 아래에 그 1행을 제외한 corrected 결과를 함께 둔다."""
        ),
        nbformat.v4.new_code_cell(
            """linear_table = pd.DataFrame({
    "Python coefficient": results["linear"]["coefficients"],
    "book/source coefficient": chapter4.BOOK_LINEAR_COEFFICIENTS,
    "p-value": results["linear"]["p_values"],
}, index=["intercept", *chapter4.FEATURE_NAMES])
display(linear_table)
display(pd.DataFrame({
    "train": results["linear"]["train"].gross.__dict__,
    "source-defined test": results["linear"]["test"].gross.__dict__,
    "boundary-corrected test": results["linear"]["causal"]["test"].gross.__dict__,
}).loc[["annual_return","sharpe","maximum_drawdown","drawdown_duration"]])
display(pd.Series({
    "source fit rows": results["linear"]["observations"],
    "corrected fit rows": results["linear"]["causal"]["observations"],
    "source last target": results["linear"]["test"].metadata["last_fit_target_realization_date"],
    "corrected last target": results["linear"]["causal"]["test"].metadata["last_fit_target_realization_date"],
    "test start": results["linear"]["causal"]["test"].metadata["test_start"],
}).to_frame("value"))
assert np.allclose(results["linear"]["coefficients"], chapter4.BOOK_LINEAR_COEFFICIENTS, atol=2e-14, rtol=0)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 8. Stepwise regression: 분산 감소와 선택 편향

forward entry p-value가 가장 작은 ret2만 선택된다. 훈련 CAGR 약 44%, 테스트 약 11%로
full model보다 낫지만 이 표본 하나가 stepwise의 일반적 우월성을 증명하지 않는다.
후보 lag나 진입 문턱을 테스트 결과를 보며 바꾸면 selection bias다. 여기에 저장된
candidate p-value는 선택 과정 자체를 감사할 수 있게 한다. source-defined와 corrected
적합 모두 ret2를 선택하지만 계수와 결과는 별도로 기록한다."""
        ),
        nbformat.v4.new_code_cell(
            """display(pd.Series(results["stepwise"]["candidate_p_values"]).sort_values().to_frame("entry p-value"))
comparison = pd.DataFrame({
    "Python CAGR": [results["linear"]["test"].gross.annual_return, results["stepwise"]["test"].gross.annual_return],
    "book/source CAGR": [chapter4.BOOK_RESULTS["linear_test_cagr"], chapter4.BOOK_RESULTS["stepwise_test_cagr"]],
    "Python Sharpe": [results["linear"]["test"].gross.sharpe, results["stepwise"]["test"].gross.sharpe],
}, index=["full linear", "stepwise ret2"])
display(comparison)
display(pd.DataFrame({
    "source-defined": [results["stepwise"]["test"].gross.annual_return, results["stepwise"]["test"].gross.sharpe],
    "boundary-corrected": [results["stepwise"]["causal"]["test"].gross.annual_return, results["stepwise"]["causal"]["test"].gross.sharpe],
}, index=["test CAGR", "test Sharpe"]))
assert results["stepwise"]["selected"] == ["ret2"]"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 9. Regression tree: 전체 tree와 extreme leaves

tree는 비선형 조건을 쉽게 표현하지만 작은 표본의 분할을 기억하기도 쉽다. 책의 전체
tree 예측은 훈련에서 높고 테스트에서 음수다. 극단 leaf 두 개만 거래하는 규칙은 flat을
허용해 더 보수적이다. 아래 곡선은 선형·stepwise·극단 tree·HMM을 동일 train/test
구간으로 분리해 보여준다. 서로 다른 구간을 한 wealth curve로 이어붙이지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """show_figure(chapter4.create_book_replay_figure(data, results))
tree_compare = pd.Series({
    "Python test CAGR": results["tree_rule"]["test"].gross.annual_return,
    "book/source test CAGR": chapter4.BOOK_RESULTS["tree_rule_test_cagr"],
    "active fraction": np.mean(np.abs(results["tree_rule"]["test"].positions) > 0),
})
display(tree_compare.to_frame("value"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 10. Bagging, random subspace, boosting

Bagging은 bootstrap 표본의 tree를 평균해 분산을 낮추고, random subspace는 tree마다
피처 일부만 보여 상관을 낮춘다. Boosting은 앞 learner의 오차를 다음 learner가 보완한다.
원본은 여러 iteration의 test 성능을 그려 사실상 test를 모델 선택에 노출한다. 여기서는
boosting iteration을 train ordered CV로만 선택하고 test는 한 번 평가한다. 원본 bagging,
classification tree, SVM, NN 숫자는 estimator·split·RNG가 달라 output-only reference로
나란히 표시하되 일치 여부를 판정하지 않는다. classification 값은 같은 주석이 MATLAB
5개에 중복돼 provenance가 특히 불확실하다."""
        ),
        nbformat.v4.new_code_cell(
            """models = chapter4.python_model_results(results)
model_table = pd.DataFrame({
    "gross CAGR": {k: v.gross.annual_return for k, v in models.items()},
    "net CAGR at 2bps": {k: v.net.annual_return for k, v in models.items()},
    "gross Sharpe": {k: v.gross.sharpe for k, v in models.items()},
    "max drawdown": {k: v.gross.maximum_drawdown for k, v in models.items()},
})
display(model_table)
display(pd.DataFrame(chapter4.reference_only_comparisons(results)).set_index("topic")[["book_or_source","python_adaptation","compared","reason"]])
show_figure(chapter4.create_python_model_figure(results))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 11. Classification tree와 SVM

회귀는 다음 날 수익률 크기를 예측하지만 classification은 부호를 직접 예측한다. SVM은
margin을 최대화하며 kernel로 비선형 경계를 만든다. 스케일이 거리와 margin에 영향을
주므로 `StandardScaler`를 train pipeline 안에 둔다. kernel은 ordered CV accuracy로만
고르고, 책의 random-fold SVM 숫자는 원본 출력 비교이지 같은 estimator라고 주장하지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """svm_meta = results["python_models"]["svm"].metadata
display(pd.Series(svm_meta["cv_accuracy_by_kernel"]).to_frame("mean ordered-CV accuracy"))
print("selected kernel:", svm_meta["selected_kernel"])
print("book default-SVM OOS CAGR/Sharpe:", chapter4.BOOK_RESULTS["svm_test_cagr"], chapter4.BOOK_RESULTS["svm_test_sharpe"])
print("Python adaptation OOS CAGR/Sharpe:", results["python_models"]["svm"].gross.annual_return, results["python_models"]["svm"].gross.sharpe)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 12. HMM: 관측 부호 뒤의 잠재 상태

두 hidden state와 양/음 return observation을 둔다. 아래 transition·emission의 각 행은
1이어야 한다. Python은 원본에 인쇄된 행렬을 replay하므로 CAGR은 정확히 맞지만 EM
재추정의 독립 재현은 아니다. `hmm.m`과 `hmm_train2.m`의 중단 assert 때문에 이 경계를
명시하지 않으면 재현 수준을 과장하게 된다."""
        ),
        nbformat.v4.new_code_cell(
            """display(pd.DataFrame(chapter4.HMM_TRANSITION, index=["state 1","state 2"], columns=["next 1","next 2"]))
display(pd.DataFrame(chapter4.HMM_EMISSION, index=["state 1","state 2"], columns=["negative","nonnegative"]))
hmm_compare = pd.DataFrame({
    "Python": [results["hmm"]["train"].gross.annual_return, results["hmm"]["test"].gross.annual_return],
    "book/source": [chapter4.BOOK_RESULTS["hmm_train_cagr"], chapter4.BOOK_RESULTS["hmm_test_cagr"]],
}, index=["train CAGR","test CAGR"])
display(hmm_compare)
assert np.allclose(chapter4.HMM_TRANSITION.sum(axis=1), 1)
assert np.allclose(chapter4.HMM_EMISSION.sum(axis=1), 1)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 13. Neural network, normalization, aggregation

신경망은 초기 weight와 validation 분할에 민감하다. 한 seed의 좋은 결과를 고르는 대신
동일 구조를 seed 1~10으로 모두 실행하고 예측을 평균한다. 입력은 train 안에서 표준화하고
목표도 train 평균·표준편차로만 정규화한다. seed별 OOS CAGR 분산은 model uncertainty의
간단한 진단이지 신뢰구간이나 미래 성능 보장이 아니다. 무작위 내부 validation은 쓰지
않고 모든 seed에 같은 500-iteration 예산을 적용해 시간순 계약을 흐리지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """show_figure(chapter4.create_seed_stability_figure(results))
seed_cagrs = results["python_models"]["mlp_ensemble"].metadata["seed_cagrs"]
display(pd.Series({"minimum": min(seed_cagrs), "median": float(np.median(seed_cagrs)), "maximum": max(seed_cagrs), "ensemble": results["python_models"]["mlp_ensemble"].gross.annual_return}).to_frame("OOS CAGR"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 14. Cross-sectional stock selection은 왜 output-only인가

기술 피처와 fundamental 피처의 주식선택 스크립트는 저자 로컬 Dropbox의
`fundamentalData`를 읽는다. 공식 ZIP에는 그 파일, point-in-time universe, delisted
종목, 당시 fundamentals가 없다. 현재 구성종목 데이터로 대체하면 survivorship bias와
시점 불일치가 생긴다. 그래서 책의 2.3%/0.9와 4.0%/1.1은 보존하되 실행 test는 `null`이다."""
        ),
        nbformat.v4.new_code_cell(
            """unavailable = pd.DataFrame(results["unavailable"]).T
display(unavailable[["status","test","reason"]])
assert unavailable["test"].isna().all()
display(pd.DataFrame(chapter4.source_incomplete_markers().items(), columns=["source file","disclosed stop marker"]))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 15. 거래비용, slippage, 위험지표

편도 2bps는 포지션 변화량에 곱한다. -1에서 +1 전환은 두 단위 turnover라 4bps다.
net CAGR은 어떤 전략에서도 gross보다 좋아질 수 없다. 다만 이 민감도에는 실제 spread,
slippage, borrow fee, latency, market impact가 빠져 있으므로 실현 가능성의 하한이 아니라
낙관적 진단이다. Sharpe와 CAGR 외에 maximum drawdown, drawdown duration, annual
turnover를 함께 본다."""
        ),
        nbformat.v4.new_code_cell(
            """cost_table = pd.DataFrame({
    "gross CAGR": {k: v.gross.annual_return for k, v in models.items()},
    "net CAGR": {k: v.net.annual_return for k, v in models.items()},
    "annual turnover": {k: v.gross.annual_turnover for k, v in models.items()},
    "drawdown duration": {k: v.gross.drawdown_duration for k, v in models.items()},
})
display(cost_table)
assert (cost_table["net CAGR"] <= cost_table["gross CAGR"] + 1e-12).all()"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 16. Look-ahead, survivorship, selection bias 체크

- look-ahead: source replay의 마지막 train target이 첫 test 수익인 사실을 검출한다. corrected fit은 feature t, target t+1 모두 test 시작 전이고, signal t → position/return t+1이다.
- survivorship: SPY 단일 ETF에는 구성종목 필터를 적용하지 않지만, 누락된 주식선택은 재현 불가다.
- selection bias: boosting iteration과 SVM kernel은 train ordered CV에서만 선택한다.
- test reuse: 책의 좋은 결과를 이미 안다는 사후 정보 때문에 exact replay와 새 적응을 분리한다.
- random seed: seed 하나를 숨기지 않고 forest 반복 동일성과 MLP 10개를 공개한다.

이 체크는 미래 성능을 보장하지 않는다. 모델군 자체를 이 test를 보고 결정했다면 독립된 새
표본 외 기간이 필요하다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 17. 자동 검증

checksum, 날짜, 계수, 책 수익률 같은 독립/경험 비교와 시차·확률합·CV 순서·비용 단조성
같은 invariant를 분리한다. assert 하나라도 실패하면 노트북은 성공 산출물로 기록되지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """checks = chapter4.verify_results(data, results)
summary = chapter4.verification_summary(checks)
display(pd.Series(checks).to_frame("passed"))
display(pd.Series(summary).drop("classification").to_frame("value"))
assert all(checks.values())
assert summary["total"] == len(chapter4.VERIFICATION_CLASSES)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 18. 결론

이 장에서 복잡한 AI가 자동으로 더 좋은 표본 외 전략을 만들지 않았다. full linear의 훈련
성과는 테스트에서 거의 사라졌고, 작은 stepwise와 flat을 허용한 tree rule이 이 표본에서는
더 안정적이었다. HMM은 published matrix replay에는 성공했지만 재추정 소스가 완전하지
않다. exact 회귀 수치는 원본의 1행 경계 누수까지 보존했고 corrected 결과를 별도로 냈다.
Python ensemble·SVM·MLP는 첫 test target을 제외한 ordered-CV 적응으로 분리했다.

다른 책에도 적용할 원칙은 명확하다. 공식 archive와 SHA-256을 잠그고, 식을 구현 함수에
연결하며, train/model-selection/test 경계를 시간순으로 assert하고, seed와 비용 전후를
모두 공개한다. 필요한 원자료가 없으면 0을 만들지 말고 `null + reason`으로 남긴다.
좋은 AI 연구의 출발점은 더 큰 모델이 아니라 더 감사 가능한 실험 계약이다."""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch4-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"Chapter 4 notebook cell {index}")
    execute_notebook(notebook, NOTEBOOK_PATH, workdir=PROJECT_ROOT, timeout=600)


def audit_current_notebook() -> dict[str, Any] | None:
    if not AUDIT_SCRIPT_PATH.exists():
        warnings.warn(
            f"Optional quality audit unavailable: {AUDIT_SCRIPT_PATH}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    spec = importlib.util.spec_from_file_location("chapter4_audit", AUDIT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {AUDIT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_notebook(NOTEBOOK_PATH)


def write_quality_benchmark(metrics: dict[str, Any]) -> None:
    audit = audit_current_notebook()
    if audit is None:
        return
    counts = audit["counts"]
    scores = audit["scores"]
    summary = metrics["verification"]["summary"]
    content = f"""# Chapter 4 품질 벤치마크

실행된 노트북과 저장소 감사 rubric에서 자동 생성한 결과다.

| Chapter | 실행 | 재현성 | 엄밀성 | 교육 | 트레이딩 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch4** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

- 셀 {counts['cells']}개: Markdown {counts['markdown_cells']}, code {counts['code_cells']}.
- code 실행 {counts['executed_code_cells']}/{counts['code_cells']}, 오류 {counts['error_outputs']}, 인라인 PNG {counts['embedded_png_outputs']}.
- Markdown {counts['markdown_characters']:,}자, 한국어 {counts['korean_characters']:,}자, 헤딩 {counts['headings']}.
- 검증 {summary['total']}개: 독립/경험 {summary['independent_or_empirical']}, invariant {summary['contract_invariant']}.
- exact replay, Python adaptation, output-only를 분리하고 누락 입력은 `null + reason`으로 기록했다.
"""
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", content)


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_chapter_data()
    results = run_experiments(data)
    checks = verify_results(data, results)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Chapter 4 verification failed: {failed}")
    metrics = build_metrics(data, results, checks)
    save_figures(data, results)
    write_text_if_changed(
        REPORT_DIR / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_report(data, results, metrics)
    write_readme()
    if execute_notebook_flag:
        build_and_execute_notebook()
        write_quality_benchmark(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        validate_offline_assets()
        statuses = ["offline assets verified"]
    else:
        statuses = download_official_assets(force=args.force_download)
    for status in statuses:
        print(status)
    if args.download_only:
        return
    metrics = run_analysis(execute_notebook_flag=not args.skip_notebook)
    print("Chapter 4 analysis complete")
    print(
        "Stepwise OOS CAGR:",
        metrics["strategies"]["stepwise"]["test"]["gross"]["annual_return"],
    )
    print("Checks:", metrics["verification"]["summary"]["passed"])
    print("Report:", REPORT_DIR / "chapter4_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
