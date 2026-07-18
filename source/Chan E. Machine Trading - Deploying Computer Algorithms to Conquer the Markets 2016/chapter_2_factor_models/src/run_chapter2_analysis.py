#!/usr/bin/env python3
"""Reproduce and audit Machine Trading Chapter 2 factor-model examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.io import loadmat


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
RAW_DIR = PROJECT_ROOT / "data/raw/book3/chapter_2"
FUNDAMENTAL_DATA_PATH = RAW_DIR / "fundamentalData.mat"
FUNDAMENTALS_PATH = RAW_DIR / "fundamentals.mat"
FAMA_FRENCH_PATH = RAW_DIR / "F-F_Research_Data_Factors_daily.CSV"
REPORT_DIR = CHAPTER_DIR / "src/reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = CHAPTER_DIR / "src/chapter2_full_report.ipynb"
AUDIT_SCRIPT_PATH = (
    PROJECT_ROOT.parents[1]
    / ".codex/skills/create-chan-chapter-analysis/scripts/audit_chapter_artifacts.py"
)

BOOK_RESULTS = {
    "fama_french_official": {
        "train_cagr": 1.035627,
        "train_sharpe": 2.464065,
        "test_cagr": -0.070378,
        "test_sharpe": -0.643982,
    },
    "cross_sectional_27": {
        "train_cagr": 0.043255,
        "train_sharpe": 0.391330,
        "test_cagr": 0.123549,
        "test_sharpe": 1.675774,
    },
    "roe_bm": {
        "train_cagr": 0.050343,
        "train_sharpe": 0.370628,
        "test_cagr": 0.093199,
        "test_sharpe": 1.010660,
    },
    "roe_only": {
        "train_cagr": 0.014004,
        "train_sharpe": 0.177303,
        "test_cagr": 0.201661,
        "test_sharpe": 1.305036,
    },
    "pca_daily_book": {"cagr": 0.155643, "sharpe": 1.378843},
}

CHAPTER_COVERAGE = (
    ("Factor risk, alpha, and predictive regression", "conceptual explanation", "Sections 1–2"),
    ("Example 2.1 Fama–French next-day model", "executed + approximate book comparison", "faithful I(end-49) bug and corrected 50/50"),
    ("Example 2.2 27 cross-sectional fundamentals", "executed + book comparison", "quarter holding"),
    ("Example 2.3 ROE/BM and ROE-only", "executed + book comparison", "month holding"),
    ("Option-implied moments", "formula illustration; backtest unavailable", "archive has source but no option panel"),
    ("Monthly implied-volatility change", "conceptual explanation", "licensed option panel absent"),
    ("Put–call IV spread", "conceptual explanation", "licensed option panel absent"),
    ("OTM put–ATM call smirk", "conceptual explanation", "licensed option panel absent"),
    ("Market-implied-volatility change", "conceptual explanation", "licensed option panel absent"),
    ("Short interest", "conceptual explanation", "short-interest panel absent"),
    ("Liquidity", "conceptual explanation", "volume and shares-outstanding inputs absent"),
    ("Example 2.5 statistical PCA factors", "lower-frequency executed adaptation", "21-day rebalance vs book daily"),
    ("Combining factors and rank robustness", "conceptual explanation", "collinearity and train-only selection"),
    ("Exercises and endnotes", "out of scope", "questions retained in chapter source"),
)

VERIFICATION_CLASSES = {
    "archive_manifest_matches": "independent_or_empirical",
    "fama_train_cagr_matches_book": "independent_or_empirical",
    "fama_train_sharpe_matches_book": "independent_or_empirical",
    "fama_test_cagr_matches_book": "independent_or_empirical",
    "fama_test_sharpe_matches_book": "independent_or_empirical",
    "cross27_test_metrics_match_book": "independent_or_empirical",
    "roe_bm_test_metrics_match_book": "independent_or_empirical",
    "roe_only_test_metrics_match_book": "independent_or_empirical",
    "corrected_fama_is_dollar_neutral": "contract_invariant",
    "official_fama_exposes_indexing_bug": "contract_invariant",
    "costs_do_not_improve_corrected_fama": "contract_invariant",
    "pca_starts_after_lookback": "contract_invariant",
    "pca_uses_five_components": "contract_invariant",
    "option_moment_formula_consistent": "contract_invariant",
    "oos_split_is_chronological": "contract_invariant",
}


@dataclass(frozen=True)
class Performance:
    cagr: float
    sharpe: float | None
    maximum_drawdown: float
    drawdown_duration: int
    cumulative_return: float
    mean_daily_return: float


@dataclass
class StrategyResult:
    name: str
    dates: pd.DatetimeIndex
    daily_returns: np.ndarray
    positions: np.ndarray
    train_end: int
    train: Performance
    test: Performance
    test_with_cost: Performance | None = None
    coefficients: np.ndarray | None = None
    predictions: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChapterData:
    dates: pd.DatetimeIndex
    date_codes: np.ndarray
    symbols: tuple[str, ...]
    mid: np.ndarray
    returns: np.ndarray
    factor_names: tuple[str, ...]
    factors: dict[str, np.ndarray]


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=2)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    return materialize_chapter_archive(
        PROJECT_ROOT, manifest, payload, force=force
    )


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def net_returns(prices: np.ndarray, lag: int = 1) -> np.ndarray:
    result = np.full_like(prices, np.nan, dtype=float)
    result[lag:] = prices[lag:] / prices[:-lag] - 1.0
    return result


def forward_shift(values: np.ndarray, steps: int) -> np.ndarray:
    shifted = np.roll(values, -steps, axis=0)
    shifted[-steps:] = np.nan
    return shifted


def fill_forward(values: np.ndarray) -> np.ndarray:
    row_indices = np.where(
        np.isfinite(values), np.arange(len(values))[:, None], 0
    )
    np.maximum.accumulate(row_indices, axis=0, out=row_indices)
    return np.take_along_axis(values, row_indices, axis=0)


def load_chapter_data() -> ChapterData:
    prices = loadmat(FUNDAMENTAL_DATA_PATH, squeeze_me=True)
    fundamentals = loadmat(FUNDAMENTALS_PATH, squeeze_me=True)
    date_codes = np.asarray(prices["tday"], dtype=int)
    symbols = tuple(np.asarray(prices["syms"]).astype(str).tolist())
    mid = np.asarray(prices["mid"], dtype=float)
    if mid.shape != (1762, 743):
        raise ValueError(f"Unexpected mid-price shape: {mid.shape}")
    if not np.array_equal(date_codes, np.asarray(fundamentals["tday"], dtype=int)):
        raise ValueError("Price and fundamental dates differ")
    if symbols != tuple(np.asarray(fundamentals["syms"]).astype(str).tolist()):
        raise ValueError("Price and fundamental symbols differ")
    factor_names = tuple(
        [f"ARQ_{name}" for name in np.atleast_1d(prices["indQ"]).astype(str)]
        + [f"ART_{name}" for name in np.atleast_1d(prices["indT"]).astype(str)]
    )
    factor_arrays = {
        name: np.asarray(prices[name], dtype=float) for name in factor_names
    }
    factor_arrays.update(
        {
            "ARQ_EPS": np.asarray(fundamentals["ARQ_EPS"], dtype=float),
            "ARQ_BVPS": np.asarray(fundamentals["ARQ_BVPS"], dtype=float),
        }
    )
    return ChapterData(
        dates=pd.DatetimeIndex(pd.to_datetime(date_codes.astype(str), format="%Y%m%d")),
        date_codes=date_codes,
        symbols=symbols,
        mid=mid,
        returns=net_returns(mid),
        factor_names=factor_names,
        factors=factor_arrays,
    )


def performance(returns: np.ndarray) -> Performance:
    clean = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    max_drawdown = float(np.min(drawdowns))
    duration = 0
    longest = 0
    for value in drawdowns:
        duration = duration + 1 if value < 0 else 0
        longest = max(longest, duration)
    standard_deviation = float(np.std(clean, ddof=1))
    return Performance(
        cagr=float(wealth[-1] ** (252.0 / len(clean)) - 1.0),
        sharpe=(
            float(np.sqrt(252.0) * np.mean(clean) / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        maximum_drawdown=max_drawdown,
        drawdown_duration=longest,
        cumulative_return=float(wealth[-1] - 1.0),
        mean_daily_return=float(np.mean(clean)),
    )


def daily_returns_from_positions(
    positions: np.ndarray,
    returns: np.ndarray,
    one_way_cost: float = 0.0,
) -> np.ndarray:
    previous = np.roll(positions, 1, axis=0)
    previous[0] = 0.0
    pnl = np.nansum(previous * returns, axis=1)
    gross = np.sum(np.abs(previous), axis=1)
    daily = np.divide(pnl, gross, out=np.zeros_like(pnl), where=gross > 0)
    if one_way_cost:
        changes = np.sum(
            np.abs(positions - np.roll(positions, 1, axis=0)), axis=1
        )
        changes[0] = np.sum(np.abs(positions[0]))
        charged = np.roll(changes, 1)
        charged[0] = 0.0
        daily -= np.divide(
            one_way_cost * charged,
            gross,
            out=np.zeros_like(daily),
            where=gross > 0,
        )
    return daily


def load_fama_french() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(FAMA_FRENCH_PATH, skiprows=3)
    date_values = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    valid = date_values.notna()
    dates = date_values.loc[valid].astype(int).to_numpy()
    factors = (
        frame.loc[valid]
        .iloc[:, 1:4]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    return dates, factors


def fama_position_row(
    predictions: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """Translate one prediction row into the book-bug or corrected positions."""
    positions = np.zeros_like(predictions, dtype=float)
    available = np.flatnonzero(np.isfinite(predictions))
    if len(available) < 100:
        return positions, available
    ordered = available[np.argsort(predictions[available])]
    positions[ordered[:50]] = -1.0
    if variant == "matlab_bug":
        positions[ordered[-50]] = 1.0
    elif variant == "corrected_50_50":
        positions[ordered[-50:]] = 1.0
    else:
        raise ValueError(f"Unknown Fama–French position variant: {variant}")
    return positions, ordered


def fit_fama_french(data: ChapterData, variant: str) -> StrategyResult:
    ff_dates, ff_factors = load_fama_french()
    _, price_index, factor_index = np.intersect1d(
        data.date_codes, ff_dates, return_indices=True
    )
    aligned_returns = data.returns[price_index]
    future_returns = forward_shift(data.returns, 1)[price_index]
    design = np.column_stack(
        (np.ones(len(price_index)), ff_factors[factor_index])
    )
    train_end = len(design) // 2
    predictions = np.full_like(aligned_returns, np.nan)
    coefficients = np.full((aligned_returns.shape[1], design.shape[1]), np.nan)
    for stock in range(aligned_returns.shape[1]):
        response = future_returns[:train_end, stock]
        valid = np.isfinite(response) & np.all(
            np.isfinite(design[:train_end]), axis=1
        )
        if valid.sum() < design.shape[1] + 1:
            continue
        coefficient = np.linalg.lstsq(
            design[:train_end][valid], response[valid], rcond=None
        )[0]
        coefficients[stock] = coefficient
        predictions[:, stock] = design @ coefficient

    positions = np.zeros_like(aligned_returns)
    for row in range(len(positions)):
        positions[row], _ = fama_position_row(predictions[row], variant)
    daily = daily_returns_from_positions(positions, aligned_returns)
    daily_cost = daily_returns_from_positions(
        positions, aligned_returns, one_way_cost=2e-4
    )
    is_bug = variant == "matlab_bug"
    label = "official 50-short/I(end-49)-long" if is_bug else "corrected 50/50"
    return StrategyResult(
        name=f"Fama–French {label}",
        dates=data.dates[price_index],
        daily_returns=daily,
        positions=positions,
        train_end=train_end,
        train=performance(daily[:train_end]),
        test=performance(daily[train_end:]),
        test_with_cost=performance(daily_cost[train_end:]),
        coefficients=coefficients,
        predictions=predictions,
        metadata={
            "long_count": 1 if is_bug else 50,
            "short_count": 50,
            "long_selection": "50th-highest scalar" if is_bug else "top 50",
            "long_rank_from_top": 50 if is_bug else list(range(1, 51)),
            "book_source_line": "I(end-topN+1) without :end",
            "aligned_rows": len(price_index),
        },
    )


def factor_design(
    data: ChapterData, names: Iterable[str], row_ids: slice | np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [data.factors[name][row_ids].ravel(order="F") for name in names]
    )


def overlapping_positions(predictions: np.ndarray, holding_days: int) -> np.ndarray:
    signal = np.sign(predictions)
    signal = np.roll(signal, 1, axis=0)
    signal[0] = 0.0
    positions = np.zeros_like(signal)
    for lag_count in range(holding_days):
        lagged = np.roll(signal, lag_count, axis=0)
        if lag_count:
            lagged[:lag_count] = 0.0
        positions += np.nan_to_num(lagged, nan=0.0)
    return positions


def fit_cross_sectional_model(
    data: ChapterData,
    names: tuple[str, ...],
    holding_days: int,
    label: str,
) -> StrategyResult:
    future_return = forward_shift(
        net_returns(data.mid, holding_days), holding_days + 1
    )
    train_end = len(data.dates) // 2
    train_rows = np.arange(train_end)
    design = factor_design(data, names, train_rows)
    response = future_return[:train_end].ravel(order="F")
    valid = np.isfinite(response) & np.all(np.isfinite(design), axis=1)
    coefficient = np.linalg.lstsq(
        np.column_stack((np.ones(valid.sum()), design[valid])),
        response[valid],
        rcond=None,
    )[0]

    segment_returns: list[np.ndarray] = []
    segment_positions: list[np.ndarray] = []
    segment_cost_returns: list[np.ndarray] = []
    for row_ids in (np.arange(train_end), np.arange(train_end, len(data.dates))):
        segment_design = factor_design(data, names, row_ids)
        good = np.all(np.isfinite(segment_design), axis=1)
        flat_prediction = np.full(len(segment_design), np.nan)
        flat_prediction[good] = np.column_stack(
            (np.ones(good.sum()), segment_design[good])
        ) @ coefficient
        predictions = flat_prediction.reshape(
            len(row_ids), len(data.symbols), order="F"
        )
        positions = overlapping_positions(predictions, holding_days)
        realized = data.returns[row_ids]
        segment_positions.append(positions)
        segment_returns.append(daily_returns_from_positions(positions, realized))
        segment_cost_returns.append(
            daily_returns_from_positions(positions, realized, one_way_cost=2e-4)
        )

    daily = np.concatenate(segment_returns)
    cost_daily = np.concatenate(segment_cost_returns)
    return StrategyResult(
        name=label,
        dates=data.dates,
        daily_returns=daily,
        positions=np.vstack(segment_positions),
        train_end=train_end,
        train=performance(segment_returns[0]),
        test=performance(segment_returns[1]),
        test_with_cost=performance(segment_cost_returns[1]),
        coefficients=coefficient,
        metadata={
            "holding_days": holding_days,
            "training_observations": int(valid.sum()),
            "factor_names": list(names),
        },
    )


def prepare_roe_bm(data: ChapterData) -> ChapterData:
    book_value = fill_forward(data.factors["ARQ_BVPS"])
    lagged_book_value = np.roll(book_value, 1, axis=0)
    lagged_book_value[0] = np.nan
    roe = 1.0 + data.factors["ARQ_EPS"] / lagged_book_value
    roe[roe <= 0] = np.nan
    book_to_market = 1.0 / data.factors["ARQ_PB"]
    book_to_market[book_to_market <= 0] = np.nan
    factors = dict(data.factors)
    factors["LOG_ROE"] = np.log(roe)
    factors["LOG_BM"] = np.log(book_to_market)
    return ChapterData(
        dates=data.dates,
        date_codes=data.date_codes,
        symbols=data.symbols,
        mid=data.mid,
        returns=data.returns,
        factor_names=data.factor_names,
        factors=factors,
    )


def fit_pca_adaptation(
    data: ChapterData,
    lookback: int = 252,
    components: int = 5,
    rebalance_every: int = 21,
) -> StrategyResult:
    positions = np.zeros_like(data.returns)
    explained: list[float] = []
    rebalances = 0
    for row in range(lookback, len(data.dates) - 1, rebalance_every):
        window = data.returns[row - lookback + 1 : row + 1]
        eligible = np.flatnonzero(np.all(np.isfinite(window), axis=0))
        if len(eligible) < 101:
            continue
        matrix = window[:, eligible]
        centered = matrix - np.mean(matrix, axis=0)
        time_covariance = centered @ centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(time_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        factors = eigenvectors[:, order[:components]] * np.sqrt(
            eigenvalues[:components]
        )
        response = matrix[1:]
        design = np.column_stack((np.ones(lookback - 1), factors[:-1]))
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        prediction = np.r_[1.0, factors[-1]] @ coefficients
        ordered = eligible[np.argsort(prediction)]
        selected = np.zeros(len(data.symbols))
        selected[ordered[:50]] = -1.0
        selected[ordered[-50:]] = 1.0
        positions[row : min(row + rebalance_every, len(positions))] = selected
        explained.append(
            float(np.sum(eigenvalues[:components]) / np.sum(eigenvalues))
        )
        rebalances += 1
    daily = daily_returns_from_positions(positions, data.returns)
    cost_daily = daily_returns_from_positions(
        positions, data.returns, one_way_cost=2e-4
    )
    return StrategyResult(
        name="PCA 5-factor, 21-day rebalance adaptation",
        dates=data.dates,
        daily_returns=daily,
        positions=positions,
        train_end=lookback,
        train=performance(daily[:lookback]),
        test=performance(daily[lookback:]),
        test_with_cost=performance(cost_daily[lookback:]),
        metadata={
            "lookback": lookback,
            "components": components,
            "rebalance_every": rebalance_every,
            "rebalances": rebalances,
            "mean_explained_variance": float(np.mean(explained)),
            "reproduction_classification": "lower-frequency methodological adaptation",
        },
    )


def implied_moment_example() -> dict[str, float]:
    atm_call, atm_put = 0.22, 0.24
    otm_call, otm_put = 0.19, 0.31
    return {
        "implied_volatility": (atm_call + atm_put) / 2.0,
        "implied_skewness_proxy": otm_call - otm_put,
        "implied_kurtosis_proxy": (otm_call + otm_put) - (atm_call + atm_put),
        "atm_call": atm_call,
        "atm_put": atm_put,
        "otm_call": otm_call,
        "otm_put": otm_put,
    }


def run_experiments(data: ChapterData) -> dict[str, StrategyResult]:
    enriched = prepare_roe_bm(data)
    return {
        "fama_official": fit_fama_french(data, variant="matlab_bug"),
        "fama_corrected": fit_fama_french(data, variant="corrected_50_50"),
        "cross27": fit_cross_sectional_model(
            data, data.factor_names, 63, "27 fundamental factors"
        ),
        "roe_bm": fit_cross_sectional_model(
            enriched, ("LOG_BM", "LOG_ROE"), 21, "log(BM) + log(ROE)"
        ),
        "roe_only": fit_cross_sectional_model(
            enriched, ("LOG_ROE",), 21, "log(ROE) only"
        ),
        "pca": fit_pca_adaptation(data),
    }


def create_equity_figure(results: dict[str, StrategyResult]) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    for key in ("fama_official", "fama_corrected"):
        result = results[key]
        start = result.train_end
        axes[0].plot(
            result.dates[start:],
            np.cumprod(1 + result.daily_returns[start:]) - 1,
            label=result.name,
        )
    axes[0].axhline(0, color="0.4", linewidth=0.8)
    axes[0].set_title("Example 2.1: official indexing bug vs corrected 50/50")
    axes[0].set_ylabel("out-of-sample cumulative return")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    for key in ("cross27", "roe_bm", "roe_only"):
        result = results[key]
        start = result.train_end
        axes[1].plot(
            result.dates[start:],
            np.cumprod(1 + result.daily_returns[start:]) - 1,
            label=result.name,
        )
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set_title("Examples 2.2–2.3: common out-of-sample window")
    axes[1].set_ylabel("cumulative return")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    pca = results["pca"]
    start = pca.train_end
    axes[2].plot(
        pca.dates[start:],
        np.cumprod(1 + pca.daily_returns[start:]) - 1,
        label=pca.name,
        color="#d62728",
    )
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    axes[2].set_title("Example 2.5: longer forward-only PCA adaptation window")
    axes[2].set_ylabel("cumulative return")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def create_diagnostics_figure(
    data: ChapterData, results: dict[str, StrategyResult]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    missing = np.mean(~np.isfinite(data.mid), axis=0)
    axes[0].hist(missing, bins=30, color="#2563eb", alpha=0.8)
    axes[0].set_title("Per-stock missing-price fraction")
    axes[0].set_xlabel("missing fraction")
    axes[0].set_ylabel("stocks")

    fama = results["fama_official"]
    coefficient = fama.coefficients
    finite = np.all(np.isfinite(coefficient), axis=1)
    axes[1].boxplot(
        coefficient[finite, 1:], tick_labels=["Mkt-RF", "SMB", "HML"], showfliers=False
    )
    axes[1].set_title("Predictive loadings across stocks")
    axes[1].axhline(0, color="0.5", linewidth=0.8)

    labels = ["27 factors", "ROE+BM", "ROE", "PCA"]
    values = [
        results[key].test.cagr for key in ("cross27", "roe_bm", "roe_only", "pca")
    ]
    costs = [
        results[key].test_with_cost.cagr
        for key in ("cross27", "roe_bm", "roe_only", "pca")
    ]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.18, values, 0.36, label="gross")
    axes[2].bar(x + 0.18, costs, 0.36, label="2 bps one-way")
    axes[2].set_xticks(x, labels, rotation=20)
    axes[2].set_title("Cost sensitivity")
    axes[2].set_ylabel("CAGR")
    axes[2].legend(fontsize=8)
    axes[2].axhline(0, color="0.4", linewidth=0.8)
    fig.tight_layout()
    return fig


def create_factor_map_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    boxes = {
        "Time-series\nMkt/HML/SMB": (0.22, 0.78),
        "Cross-sectional\nROE/BM/27 fundamentals": (0.22, 0.58),
        "Statistical\nPCA": (0.22, 0.38),
        "Option/positioning\nIV moments, short interest": (0.22, 0.18),
        "Predict next-period\nreturn": (0.78, 0.48),
    }
    for text, (x, y) in boxes.items():
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.6", "fc": "#eff6ff", "ec": "#2563eb"},
        )
    for start in ((0.31, 0.78), (0.35, 0.58), (0.28, 0.38), (0.35, 0.18)):
        ax.annotate(
            "",
            xy=(0.68, 0.48),
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5},
        )
    ax.set_title("Chapter 2 factor taxonomy and prediction target", fontsize=14)
    return fig


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def check_close(actual: float | None, expected: float, tolerance: float) -> bool:
    return bool(actual is not None and abs(actual - expected) <= tolerance)


def verify_results(
    results: dict[str, StrategyResult], option_example: dict[str, float]
) -> dict[str, bool]:
    extraction_path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    official = results["fama_official"]
    corrected = results["fama_corrected"]
    cross27 = results["cross27"]
    roe_bm = results["roe_bm"]
    roe_only = results["roe_only"]
    pca = results["pca"]
    book_fama = BOOK_RESULTS["fama_french_official"]
    faithful_long_selection = True
    for row in np.flatnonzero(np.any(official.positions != 0, axis=1)):
        ordered = np.flatnonzero(np.isfinite(official.predictions[row]))
        ordered = ordered[np.argsort(official.predictions[row, ordered])]
        selected_long = np.flatnonzero(official.positions[row] == 1.0)
        faithful_long_selection &= bool(
            len(selected_long) == 1 and selected_long[0] == ordered[-50]
        )
    checks = {
        "archive_manifest_matches": extraction["archive_sha256"] == chapter_manifest()["sha256"],
        "fama_train_cagr_matches_book": check_close(official.train.cagr, book_fama["train_cagr"], 0.03),
        "fama_train_sharpe_matches_book": check_close(official.train.sharpe, book_fama["train_sharpe"], 0.08),
        "fama_test_cagr_matches_book": check_close(official.test.cagr, book_fama["test_cagr"], 0.02),
        "fama_test_sharpe_matches_book": check_close(official.test.sharpe, book_fama["test_sharpe"], 0.08),
        "cross27_test_metrics_match_book": (
            check_close(cross27.test.cagr, BOOK_RESULTS["cross_sectional_27"]["test_cagr"], 1e-5)
            and check_close(cross27.test.sharpe, BOOK_RESULTS["cross_sectional_27"]["test_sharpe"], 1e-5)
        ),
        "roe_bm_test_metrics_match_book": (
            check_close(roe_bm.test.cagr, BOOK_RESULTS["roe_bm"]["test_cagr"], 1e-5)
            and check_close(roe_bm.test.sharpe, BOOK_RESULTS["roe_bm"]["test_sharpe"], 1e-5)
        ),
        "roe_only_test_metrics_match_book": (
            check_close(roe_only.test.cagr, BOOK_RESULTS["roe_only"]["test_cagr"], 1e-5)
            and check_close(roe_only.test.sharpe, BOOK_RESULTS["roe_only"]["test_sharpe"], 1e-5)
        ),
        "corrected_fama_is_dollar_neutral": bool(
            np.allclose(np.sum(corrected.positions, axis=1)[1:], 0.0)
        ),
        "official_fama_exposes_indexing_bug": bool(
            official.metadata["long_count"] == 1
            and official.metadata["long_rank_from_top"] == 50
            and faithful_long_selection
            and np.min(np.sum(official.positions, axis=1)) == -49
        ),
        "costs_do_not_improve_corrected_fama": bool(
            corrected.test_with_cost.cumulative_return <= corrected.test.cumulative_return
        ),
        "pca_starts_after_lookback": bool(
            not np.any(pca.positions[: pca.metadata["lookback"]])
        ),
        "pca_uses_five_components": pca.metadata["components"] == 5,
        "option_moment_formula_consistent": bool(
            np.isclose(option_example["implied_volatility"], 0.23)
            and np.isclose(option_example["implied_skewness_proxy"], -0.12)
            and np.isclose(option_example["implied_kurtosis_proxy"], 0.04)
        ),
        "oos_split_is_chronological": bool(
            official.dates[official.train_end - 1] < official.dates[official.train_end]
            and cross27.dates[cross27.train_end - 1] < cross27.dates[cross27.train_end]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Chapter 2 verification failed: {', '.join(failed)}")
    return checks


def strategy_metrics(result: StrategyResult) -> dict[str, Any]:
    has_train_positions = bool(np.any(result.positions[: result.train_end]))
    metadata = dict(result.metadata or {})
    if not has_train_positions:
        metadata["train_segment"] = "no positions (lookback covers train)"
    return {
        "name": result.name,
        "train": asdict(result.train) if has_train_positions else None,
        "test": asdict(result.test),
        "test_with_2bps_one_way_cost": (
            asdict(result.test_with_cost) if result.test_with_cost else None
        ),
        "metadata": metadata,
        "coefficients": (
            result.coefficients.tolist()
            if result.coefficients is not None and result.coefficients.ndim == 1
            else None
        ),
    }


def verification_summary(checks: dict[str, bool]) -> dict[str, Any]:
    independent = [
        name for name in checks if VERIFICATION_CLASSES[name] == "independent_or_empirical"
    ]
    invariants = [
        name for name in checks if VERIFICATION_CLASSES[name] == "contract_invariant"
    ]
    return {
        "total": len(checks),
        "passed": sum(checks.values()),
        "independent_or_empirical": len(independent),
        "contract_invariant": len(invariants),
        "classification": {
            "independent_or_empirical": independent,
            "contract_invariant": invariants,
        },
    }


def build_metrics(
    data: ChapterData,
    results: dict[str, StrategyResult],
    option_example: dict[str, float],
    checks: dict[str, bool],
) -> dict[str, Any]:
    extraction_path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    return {
        "chapter": 2,
        "chapter_coverage": [
            {"topic": topic, "status": status, "evidence": evidence}
            for topic, status, evidence in CHAPTER_COVERAGE
        ],
        "random_seed": None,
        "randomness_note": "No stochastic estimator; NumPy deterministic linear algebra",
        "provenance": {
            "source_page": chapter_manifest()["source_page"],
            "archive_url": chapter_manifest()["url"],
            "archive_sha256": chapter_manifest()["sha256"],
            "extraction_manifest_sha256": sha256_file(extraction_path),
            "input_sha256": {
                Path(member["destination"]).name: member["sha256"]
                for member in extraction["members"]
                if member["kind"] == "research_data"
            },
            "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
            "environment": environment_versions(),
        },
        "data": {
            "start_date": data.dates[0].date().isoformat(),
            "end_date": data.dates[-1].date().isoformat(),
            "shape": list(data.mid.shape),
            "symbols": len(data.symbols),
            "missing_prices": int(np.count_nonzero(~np.isfinite(data.mid))),
            "nonpositive_finite_prices": int(
                np.count_nonzero(np.isfinite(data.mid) & (data.mid <= 0))
            ),
            "factor_count": len(data.factor_names),
        },
        "strategies": {key: strategy_metrics(value) for key, value in results.items()},
        "option_moment_formula_illustration": option_example,
        "book_results": BOOK_RESULTS,
        "verification": {
            "checks": checks,
            "summary": verification_summary(checks),
        },
    }


def coverage_table() -> str:
    rows = ["| Topic | Status | Evidence |", "|---|---|---|"]
    rows.extend(f"| {a} | {b} | {c} |" for a, b, c in CHAPTER_COVERAGE)
    return "\n".join(rows)


def comparison_table(results: dict[str, StrategyResult]) -> str:
    mapping = (
        ("fama_official", "fama_french_official", "근사 재현", "CAGR 0.02 / Sharpe 0.08"),
        ("cross27", "cross_sectional_27", "수치 재현", "1e-5"),
        ("roe_bm", "roe_bm", "수치 재현", "1e-5"),
        ("roe_only", "roe_only", "수치 재현", "1e-5"),
    )
    rows = [
        "| Experiment | 등급 | 허용오차 | Python OOS CAGR | Book OOS CAGR | Python OOS Sharpe | Book OOS Sharpe |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result_key, book_key, grade, tolerance in mapping:
        result = results[result_key]
        book = BOOK_RESULTS[book_key]
        rows.append(
            f"| {result.name} | {grade} | {tolerance} | {result.test.cagr:.6f} | {book['test_cagr']:.6f} | "
            f"{result.test.sharpe:.6f} | {book['test_sharpe']:.6f} |"
        )
    return "\n".join(rows)


def write_report(
    data: ChapterData,
    results: dict[str, StrategyResult],
    option_example: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    corrected = results["fama_corrected"]
    official = results["fama_official"]
    pca = results["pca"]
    verification = metrics["verification"]["summary"]
    report = fr"""# Chapter 2 Factor Models — Python 재현 리포트

## 1. 문제와 재현 범위

요인 모델은 고유 잡음이 아니라 여러 자산에 함께 남는 체계적 위험과 예측 신호를
분리한다. 이 장의 핵심 질문은 과거에 관측한 요인 적재량이 **다음 기간** 수익률을
예측하는가이다. 동시 회귀의 설명력과 예측 백테스트를 혼동하지 않는다.

{coverage_table()}

옵션·공매도·유동성 절은 공식 ZIP에 MATLAB 소스는 있지만 필요한 OptionMetrics,
Compustat short-interest/volume·shares 패널이 없다. 편리한 현대 데이터로 바꿔
수치 재현이라고 부르지 않고, 공식과 데이터 요건만 설명한다.

![요인 지도](figures/factor_map.png)

## 2. 공식 데이터와 진단

- 공식 페이지: {chapter_manifest()['source_page']}
- 공식 ZIP: `{chapter_manifest()['url']}`
- ZIP SHA-256: `{chapter_manifest()['sha256']}`
- 기간: {data.dates[0].date()} ~ {data.dates[-1].date()}
- 가격 shape: {data.mid.shape}; 종목 {len(data.symbols)}개
- 결측 가격: {metrics['data']['missing_prices']:,}; 유한 비양수 가격: {metrics['data']['nonpositive_finite_prices']}
- 27개 요인: 분기 ARQ {len(np.atleast_1d(loadmat(FUNDAMENTAL_DATA_PATH, squeeze_me=True)['indQ']))}개 + trailing ART {len(np.atleast_1d(loadmat(FUNDAMENTAL_DATA_PATH, squeeze_me=True)['indT']))}개

결측은 상장·상장폐지와 펀더멘털 발표 주기에서 온다. 회귀는 각 학습 구간에서
응답과 모든 설명변수가 유한한 행만 사용하며, 미래 값으로 결측을 채우지 않는다.

## 3. 수식에서 코드로

시계열 예측 모델은

$$R_{{t+1,s}}-r_F=\alpha_s+\beta_{{s}}^T f_t+\epsilon_{{t+1,s}}$$

이고 `fit_fama_french`가 종목별 OLS를 수행한다. 횡단면 모델은 같은 시점의 여러
종목을 쌓아

$$R_{{t+h,s}}=\alpha+\gamma^T x_{{t,s}}+\epsilon_{{t+h,s}}$$

를 한 번 학습하며 `fit_cross_sectional_model`이 구현한다. 신호는 하루 늦춰
포지션으로 바꾸고 21일 또는 63일 겹침 보유를 구성한다. PCA 적응은 과거 252일만
사용해 5개 성분을 만들고 21일마다 재학습한다.

## 4. Example 2.1 — Fama–French와 원본 인덱싱 버그

공식 `FamaFrenchFactors_predictive.m`은 숏에 `I(1:topN)`을 쓰지만 롱에는
`I(end-topN+1)`만 써 콜론이 빠졌다. `topN=50`이므로 이는 최상위가 아니라
**위에서 50번째 종목 하나**를 롱한다. 따라서 책 출력은 50개 숏과 이 1개 롱의
순숏 포트폴리오다. Python의 MATLAB 충실 경로도 같은 순위를 선택한다.

학습 CAGR/Sharpe는 `{official.train.cagr:.6f}` / `{official.train.sharpe:.6f}`, OOS는
`{official.test.cagr:.6f}` / `{official.test.sharpe:.6f}`다. 책 비교 허용오차는 각각
CAGR `0.03`/`0.02`, Sharpe `0.08`/`0.08`이며 이는 **근사 재현**이다. 잔여 격차는
MATLAB과 NumPy 회귀·연산 순서 및 `smartsum` 의미 차이가 후보지만 원인을 입증하지
못했다. 이 일치는 전략 타당성의 증거가 아니라 버그 진단의 증거다.

콜론을 고쳐 50/50으로 만들면 OOS CAGR/Sharpe는
`{corrected.test.cagr:.6f}` / `{corrected.test.sharpe:.6f}`다. 편도 2bps를
차감하면 CAGR은 `{corrected.test_with_cost.cagr:.6f}`다. 원본과 수정본을 섞어
책 비교를 하지 않는다.

## 5. Examples 2.2–2.3 — 횡단면 펀더멘털

{comparison_table(results)}

27요인과 ROE/BM 모델은 전반부에서만 계수를 추정하고 후반부에 고정 적용한다.
`log(ROE)` 단일요인이 2요인보다 강한 책의 결과도 재현된다. 다만 이 데이터는
과거 SPX 구성종목을 포함한다고 설명되어 있어도, 공급자 point-in-time 지연과
회계 정정 이력을 별도로 검증하지 못했다. 발표일이 아닌 보고값의 가용일을 잘못
쓰면 look-ahead bias가 생긴다.

## 6. Example 2.5 — 통계적 PCA 요인

책은 매일 252일 창을 다시 적합해 CAGR `{BOOK_RESULTS['pca_daily_book']['cagr']:.1%}`,
Sharpe `{BOOK_RESULTS['pca_daily_book']['sharpe']:.2f}`를 보고한다. 이 구현은 계산량과
회전율을 명시하기 위해 21일마다만 재적합한 **방법론적 적응**이며 직접 수치 재현이
아니다. {pca.metadata['rebalances']}회 재학습, 평균 상위 5성분 설명분산
`{pca.metadata['mean_explained_variance']:.1%}`, 비용 전/후 CAGR은
`{pca.test.cagr:.6f}` / `{pca.test_with_cost.cagr:.6f}`다.
초기 252일은 포지션 없는 lookback이므로 `metrics.json`의 PCA train은 측정치가
아닌 `null`이며, 더 긴 forward-only 기간은 펀더멘털 OOS와 별도 패널에 그린다.

![전략 결과](figures/factor_equity.png)

## 7. 옵션·공매도·유동성 요인의 구현 경계

30일 ATM/OTM 변동성으로 정의한 예시에서 IV, skew proxy, kurtosis proxy는 각각
`{option_example['implied_volatility']:.3f}`,
`{option_example['implied_skewness_proxy']:.3f}`,
`{option_example['implied_kurtosis_proxy']:.3f}`다. 이는 수식 점검이지 옵션 전략
백테스트가 아니다. 만기 보간·델타 보간·bid/ask와 생존한 옵션만 쓰는 선택을
처리할 원시 패널이 없기 때문이다. short interest와 liquidity도 보고 지연,
shares-outstanding 단위, 거래량 조정이 필요하다.

![진단과 비용](figures/factor_diagnostics.png)

## 8. 비용·편향·표본 외 한계

- **Look-ahead:** 학습/테스트는 시간 순서로 나눴지만 회계 정보의 실제 공개시각은 검증하지 못했다.
- **Survivorship/selection:** 역사적 CRSP 유니버스라는 책의 설명에 의존하며 독립 구성종목 파일이 없다.
- **거래비용:** 편도 2bps 민감도만 계산했고 borrow fee, locate, market impact, bid-ask spread는 없다.
- **다중검정:** 27개 요인·여러 보유기간·정렬순서를 시도하면 data snooping이 커진다.
- **노출:** 50/50 종목 수가 beta·산업·시가총액 중립을 보장하지 않는다.
- **PCA:** 부호와 성분은 창마다 회전하며 경제적 의미가 고정되지 않는다.

## 9. 검증과 결론

총 {verification['total']}개 검증 통과: 책·공식 해시의 독립/경험 검증
{verification['independent_or_empirical']}개, 계산 계약 invariant
{verification['contract_invariant']}개다. `metrics.json`에 분류를 보존한다.

결론은 “요인이 수익을 보장한다”가 아니다. 원본 버그 하나가 시장중립 모델을
순숏 모델로 바꾸고, 비용·가용시점·중립화 선택이 성과를 바꾼다. 재사용할 원칙은
요인 수식, 정보 가용시점, 포지션 구성, 비용, 표본 외 적용을 따로 검증하는 것이다.
"""
    assert_clean_markdown_math(report, "Chapter 2 report")
    write_text_if_changed(REPORT_DIR / "chapter2_report.md", report)


def write_readme() -> None:
    content = """# Chapter 2 Factor Models 실험 환경

공식 Book 3 ZIP의 MATLAB 소스 전체를 `original_matlab/`에 보존하고 데이터는
Git에서 제외된 `data/raw/book3/chapter_2/`에 둔다. 예제 2.1·2.2·2.3은 책 수치와
비교하고, 예제 2.5는 21일 재학습 적응으로 명확히 구분한다.

```bash
uv run python chapter_2_factor_models/src/run_chapter2_analysis.py
uv run python chapter_2_factor_models/src/run_chapter2_analysis.py --offline
```

`--skip-notebook`은 계산·리포트·그림만 갱신한다. 저장소 `.codex` 감사 도구가
없으면 핵심 실행은 유지하고 품질 벤치마크만 경고와 함께 생략한다.
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
            """# Machine Trading — Chapter 2 Factor Models

이 노트북은 요인 위험·알파·예측 회귀를 연결하고 공식 예제의 실제 코드를
재현한다. 특히 책 수치를 만든 MATLAB 인덱싱 버그를 숨기지 않는다.

학습 질문은 다음과 같다.

1. 설명적 요인 모델과 다음 기간 예측 모델은 어떻게 다른가?
2. Fama–French 원본이 정말 시장중립 포트폴리오를 만드는가?
3. 27개 펀더멘털, ROE/BM, ROE 단일요인은 표본 외에서 책과 일치하는가?
4. PCA 요인의 경제적 의미와 재학습 비용은 무엇인가?
5. 옵션·short-interest·liquidity 데이터 부재를 어떻게 정직하게 표시하는가?

결과는 일부는 진짜 시간순 백테스트지만, 옵션 공식 예시는 백테스트가 아니다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 1. Chapter coverage / 재현 상태

| 주제 | 상태 | 근거 |
|---|---|---|
| 요인 위험·알파·예측 회귀 | 개념 설명 | 식 2.1/2.2 구분 |
| 예제 2.1 Fama–French | 근사 재현 | 원본 50숏/위에서 50번째 1롱과 수정 50/50 |
| 예제 2.2 27개 요인 | 수치 재현 | 63일 겹침 보유, 허용오차 1e-5 |
| 예제 2.3 ROE/BM·ROE | 수치 재현 | 21일 겹침 보유, 허용오차 1e-5 |
| 옵션 내재 모멘트 5종 | 수식 예시·백테스트 불가 | licensed option panel 부재 |
| 공매도 잔고·유동성 | 개념 설명·데이터 부재 | 보고 지연/주식수 입력 필요 |
| 예제 2.5 PCA | 저빈도 방법 적응 | 252일 창, 21일 재학습 |
| 결합·다중정렬 | 개념 설명 | 공선성·train-only 선택 |
| 연습문제 | 범위 밖 | 장 원문에 보존 |

**재현 분류:** exact numerical reproduction, approximate comparison,
methodological adaptation, unavailable licensed data를 구분한다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 2. 핵심 수식과 formula-to-code

시계열 예측:

$$R_{t+1,s}-r_F=\alpha_s+\beta_s^T f_t+\epsilon_{t+1,s}$$

횡단면 예측:

$$R_{t+h,s}=\alpha+\gamma^T x_{t,s}+\epsilon_{t+h,s}$$

| 수식 | 구현 함수 |
|---|---|
| $P_t/P_{t-h}-1$ | `net_returns` |
| 종목별 시계열 OLS | `fit_fama_french` |
| pooled 횡단면 OLS | `fit_cross_sectional_model` |
| 겹침 보유 | `overlapping_positions` |
| 과거 252일 PCA | `fit_pca_adaptation` |
| 비용 차감 | `daily_returns_from_positions` |

random_seed = None이다. 난수는 없고 같은 BLAS 환경의 결정론적 선형대수만 쓴다."""
        ),
        nbformat.v4.new_code_cell(
            """from io import BytesIO
from pathlib import Path
import sys
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_project_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("project root not found")

PROJECT_ROOT = find_project_root()
SRC = PROJECT_ROOT / "chapter_2_factor_models/src"
sys.path.insert(0, str(SRC))
from run_chapter2_analysis import (
    AUDIT_SCRIPT_PATH, BOOK_RESULTS, CHAPTER_COVERAGE, FAMA_FRENCH_PATH,
    FUNDAMENTAL_DATA_PATH, FUNDAMENTALS_PATH, PYPROJECT_PATH, UV_LOCK_PATH,
    VERIFICATION_CLASSES,
    chapter_manifest, create_diagnostics_figure, create_equity_figure,
    create_factor_map_figure, environment_versions, implied_moment_example,
    load_chapter_data, run_experiments, sha256_file, validate_offline_assets,
    verify_results, verification_summary,
)

validate_offline_assets()

def show_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buffer.getvalue()))

print("official chapter archive members: checksum verified")"""
        ),
        nbformat.v4.new_code_cell("""show_figure(create_factor_map_figure())"""),
        nbformat.v4.new_markdown_cell(
            """## 3. Provenance와 잠금 환경

공식 페이지·직접 ZIP·아카이브 SHA-256·개별 추출 파일 SHA-256·`uv.lock`
SHA-256을 함께 둔다. 원시 데이터가 바뀌면 offline 검증이 즉시 실패한다."""
        ),
        nbformat.v4.new_code_cell(
            """manifest = chapter_manifest()
provenance = pd.Series({
    "source_page": manifest["source_page"],
    "archive_url": manifest["url"],
    "archive_sha256": manifest["sha256"],
    "fundamentalData_sha256": sha256_file(FUNDAMENTAL_DATA_PATH),
    "fundamentals_sha256": sha256_file(FUNDAMENTALS_PATH),
    "fama_french_sha256": sha256_file(FAMA_FRENCH_PATH),
    "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
})
display(provenance.to_frame("value"))
display(pd.Series(environment_versions()).to_frame("version"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 4. 데이터 구조와 결측 진단

가격은 2007–2013년 743개 역사적 SPX 종목의 mid quote다. 결측을 0으로
바꾸지 않는다. 회귀 적합은 유한한 행만 쓰고, 포트폴리오 P&L은 해당 날짜에
가격 수익률이 없는 종목을 합산에서 제외한다. 이는 원본 `smartsum` 의미를
따르지만 denominator에는 선택 포지션이 남는다는 점도 기억해야 한다."""
        ),
        nbformat.v4.new_code_cell(
            """data = load_chapter_data()
diagnostics = pd.Series({
    "start": data.dates[0].date(), "end": data.dates[-1].date(),
    "rows": len(data.dates), "stocks": len(data.symbols),
    "missing_prices": int(np.count_nonzero(~np.isfinite(data.mid))),
    "nonpositive_finite": int(np.count_nonzero(np.isfinite(data.mid) & (data.mid <= 0))),
    "cross_sectional_factors": len(data.factor_names),
})
display(diagnostics.to_frame("value"))
display(pd.Series(np.mean(~np.isfinite(data.mid), axis=0)).describe().to_frame("missing fraction"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 5. 모든 실행 실험

다음 셀은 예제 2.1의 원본/수정본, 예제 2.2의 27요인, 예제 2.3의 두 모델,
예제 2.5의 저빈도 PCA 적응을 실제로 계산한다. 전반부 학습·후반부 테스트를
고정하며 비용 민감도는 편도 2bps다."""
        ),
        nbformat.v4.new_code_cell(
            """results = run_experiments(data)
rows=[]
for key, result in results.items():
    rows.append({
        "experiment": key, "train_cagr": result.train.cagr,
        "train_sharpe": result.train.sharpe, "test_cagr": result.test.cagr,
        "test_sharpe": result.test.sharpe,
        "test_cagr_2bps": result.test_with_cost.cagr,
        "max_drawdown": result.test.maximum_drawdown,
    })
result_table=pd.DataFrame(rows).set_index("experiment")
display(result_table)
show_figure(create_equity_figure(results))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 6. Example 2.1 — 버그 재현과 수정

원본 숏 선택은 50개지만 롱 선택은 `I(end-topN+1)`, 즉 위에서 50번째 하나뿐이다.
책 값과의 비교는 CAGR 0.03/0.02, Sharpe 0.08 허용오차의 **근사 재현**이다.
수정 50/50은 각 날짜 종목 수 기준 dollar neutral이지만 beta·산업 중립은 아니다.
원본 수치와 수정 전략을 같은 열에서 비교해 “Python이 개선했다”고 과장하지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """book_rows=[]
mapping={"fama_official":"fama_french_official","cross27":"cross_sectional_27","roe_bm":"roe_bm","roe_only":"roe_only"}
for key, book_key in mapping.items():
    r=results[key]; b=BOOK_RESULTS[book_key]
    grade="근사 재현" if key=="fama_official" else "수치 재현"
    book_rows.append({"experiment":key,"grade":grade,"python_cagr":r.test.cagr,"book_cagr":b["test_cagr"],"python_sharpe":r.test.sharpe,"book_sharpe":b["test_sharpe"]})
display(pd.DataFrame(book_rows).set_index("experiment"))
official=results["fama_official"].positions
corrected=results["fama_corrected"].positions
assert results["fama_official"].metadata["long_rank_from_top"] == 50
assert abs(results["fama_official"].train.sharpe-BOOK_RESULTS["fama_french_official"]["train_sharpe"]) <= 0.08
display(pd.Series({"official_long_rank_from_top":50,"official_min_net_count":np.min(official.sum(axis=1)),"corrected_max_abs_net_count":np.max(np.abs(corrected.sum(axis=1)))},name="position diagnostic").to_frame())"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 7. Examples 2.2–2.3 — 왜 ROE 단일요인이 더 강했나

27요인은 많은 변수를 한꺼번에 쓰므로 공선성과 high-leverage point에 민감하다.
ROE/BM 모델도 두 계수의 부호가 경제 직관과 항상 같지 않다. 책과 동일하게
ROE 단일요인의 OOS CAGR이 약 20%로 더 높지만, 이는 한 역사적 표본의 결과다.
회계 발표시각과 수정 이력을 point-in-time으로 검증하지 못했으므로 실전 알파
증거가 아니라 코드·수치 재현으로 한정한다."""
        ),
        nbformat.v4.new_code_cell(
            """exact_checks={
    "cross27_cagr": abs(results["cross27"].test.cagr-BOOK_RESULTS["cross_sectional_27"]["test_cagr"]) <= 1e-5,
    "cross27_sharpe": abs(results["cross27"].test.sharpe-BOOK_RESULTS["cross_sectional_27"]["test_sharpe"]) <= 1e-5,
    "roe_bm_cagr": abs(results["roe_bm"].test.cagr-BOOK_RESULTS["roe_bm"]["test_cagr"]) <= 1e-5,
    "roe_only_cagr": abs(results["roe_only"].test.cagr-BOOK_RESULTS["roe_only"]["test_cagr"]) <= 1e-5,
}
display(pd.Series(exact_checks, name="수치 재현 assertion").to_frame())
assert all(exact_checks.values())"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 8. Example 2.5 — PCA 통계적 요인

PCA는 공분산의 서로 직교하는 방향을 찾지만 부호·순서·경제 의미가 창마다
바뀔 수 있다. 책은 매일 재학습하지만 여기서는 21일마다 재학습한다. 따라서
책의 CAGR 15.6%와 직접 수치 비교하지 않는 methodological adaptation이다.
과거 252일 바깥의 미래 수익률은 학습 행렬에 들어가지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """pca=results["pca"]
assert pca.metadata["reproduction_classification"] == "lower-frequency methodological adaptation"
assert not np.any(pca.positions[:pca.metadata["lookback"]])
display(pd.Series(pca.metadata).to_frame("value"))
show_figure(create_diagnostics_figure(data, results))"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 9. 옵션 내재 모멘트 — 수식 점검만

ATM call/put IV를 $C_A,P_A$, OTM을 $C_O,P_O$라 두면 예시 proxy는

$$IV=(C_A+P_A)/2,\quad Skew=C_O-P_O,$$
$$Kurt=(C_O+P_O)-(C_A+P_A).$$

30일 만기·목표 델타로 보간한 실제 option panel이 ZIP에 없으므로 이 셀은
산술 identity만 검증한다. 이는 옵션 수익률 백테스트가 아니다."""
        ),
        nbformat.v4.new_code_cell(
            """option_example=implied_moment_example()
assert np.isclose(option_example["implied_volatility"], 0.23)
assert np.isclose(option_example["implied_skewness_proxy"], -0.12)
display(pd.Series(option_example).to_frame("value"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 10. 공매도 잔고·유동성·요인 결합

short interest는 공표 지연과 대차 가능성, liquidity는 거래량 조정과 당시
shares outstanding이 필요하다. 원본 `liquidity.m`도 외부 Compustat 경로를
요구한다. 데이터 부재를 현대 대체 데이터로 조용히 덮지 않는다.

여러 요인을 결합할 때는 AIC/BIC나 rank/multisort를 **학습셋 안에서만** 선택해야
한다. 산업·시가총액별 요인 부호가 달라질 수 있고, 단순 50/50은 각 요인 노출을
0으로 만들지 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 11. Bias, 비용, OOS 한계

- look-ahead: 회계값의 실제 공개일을 검증하지 못했다.
- survivorship/selection bias: 책의 CRSP 설명에 의존한다.
- transaction cost: 편도 2bps 외 borrow fee·locate·impact·slippage가 없다.
- data snooping: 요인·보유기간·topN·정렬순서 최적화가 성과를 부풀릴 수 있다.
- OOS: 시간분할은 했지만 단 하나의 holdout이고 walk-forward 반복 검증이 아니다.
- risk metrics: CAGR·Sharpe·maximum drawdown은 tail liquidity를 완전히 설명하지 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 12. 자동 검증

책 수치·아카이브 hash 대조와 포지션/수식 invariant를 분리한다. invariant를
외부 재현 증거로 세지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """checks=verify_results(results, option_example)
summary=verification_summary(checks)
verification=pd.DataFrame({"passed":checks,"class":{k:VERIFICATION_CLASSES[k] for k in checks}})
display(verification)
display(pd.Series(summary).drop("classification").to_frame("count"))
assert verification["passed"].all()
print(f"{summary['total']} checks passed")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 13. 결론

예측 요인의 성능은 수식만이 아니라 정보 시점, 종목 선택, 포지션 대칭성,
보유기간, 비용에 달려 있다. 예제 2.1의 콜론 하나가 시장중립을 순숏으로 바꾼
사실이 가장 강한 교훈이다. 예제 2.2·2.3의 책 수치는 재현했지만 미래에도 같은
알파가 있다는 뜻은 아니다. PCA는 데이터만으로 요인을 만들 수 있으나 경제적
안정성과 비용을 별도 검증해야 한다.

따라서 이 장의 재사용 가능한 절차는 source/checksum → 가용시점 → train/OOS →
포지션 계약 → 비용 → 책 비교 → 편향 경고의 순서다. 각 단계의 실패는 다음
단계의 높은 Sharpe보다 먼저 설명하고 수정해야 한다."""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch2-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"Chapter 2 notebook cell {index}")
    execute_notebook(
        notebook, NOTEBOOK_PATH, workdir=CHAPTER_DIR / "src", timeout=900
    )


def audit_current_notebook() -> dict[str, Any] | None:
    if not AUDIT_SCRIPT_PATH.exists():
        warnings.warn(
            "Repository quality audit unavailable; benchmark not refreshed",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    spec = importlib.util.spec_from_file_location("chapter2_audit", AUDIT_SCRIPT_PATH)
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
    content = f"""# Chapter 2 품질 벤치마크

이 문서는 실행된 노트북과 저장소 감사 rubric에서 자동 생성한다.

| Chapter | 실행 | 재현성 | 엄밀성 | 교육 | 트레이딩 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch2** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

- 셀 {counts['cells']}개: Markdown {counts['markdown_cells']}, code {counts['code_cells']}.
- code 실행 {counts['executed_code_cells']}/{counts['code_cells']}, 오류 {counts['error_outputs']}, 인라인 PNG {counts['embedded_png_outputs']}.
- Markdown {counts['markdown_characters']:,}자, 한국어 {counts['korean_characters']:,}자, 헤딩 {counts['headings']}.
- 검증 {summary['total']}개: 독립/경험 {summary['independent_or_empirical']}, invariant {summary['contract_invariant']}.
- 공식 수치 재현과 원본 버그/수정본 분리, unavailable licensed data 표시를 수동 확인했다.
"""
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", content)


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_chapter_data()
    results = run_experiments(data)
    option_example = implied_moment_example()
    checks = verify_results(results, option_example)
    metrics = build_metrics(data, results, option_example, checks)
    save_figure(create_factor_map_figure(), FIGURE_DIR / "factor_map.png")
    save_figure(create_equity_figure(results), FIGURE_DIR / "factor_equity.png")
    save_figure(
        create_diagnostics_figure(data, results),
        FIGURE_DIR / "factor_diagnostics.png",
    )
    write_text_if_changed(
        REPORT_DIR / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_report(data, results, option_example, metrics)
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
    print("Chapter 2 analysis complete")
    print(
        "Official Fama OOS CAGR:",
        metrics["strategies"]["fama_official"]["test"]["cagr"],
    )
    print("Report:", REPORT_DIR / "chapter2_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
