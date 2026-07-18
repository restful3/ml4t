#!/usr/bin/env python3
"""Reproduce the numerical experiments in Machine Trading, Chapter 1.

The module downloads and checksum-verifies the official Book 3 archive, keeps
the original MATLAB sources, runs deterministic Python implementations of Box
1.1 and Box 1.2, and writes figures, metrics, a Markdown report, and an executed
notebook.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
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
    materialize_verified_members,
    safe_project_path as resolve_project_path,
    sha256_file,
    validate_manifest_members,
    write_text_if_changed,
)


PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"
DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/book3/chapter_1/inputDataOHLCDaily_ETF_20150417.mat"
)
REPORT_DIR = CHAPTER_DIR / "src/reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = CHAPTER_DIR / "src/chapter1_full_report.ipynb"
AUDIT_SCRIPT_PATH = (
    PROJECT_ROOT.parents[1]
    / ".codex/skills/create-chan-chapter-analysis/scripts/audit_chapter_artifacts.py"
)
BOOK_TANGENCY_WEIGHTS = np.array(
    [0.451338068065785, 0.263444604411169, 0.000019328104256,
     0.000004913963978, 0.000007497342085, 0.285185588112728]
)
BOOK_MINIMUM_VARIANCE_WEIGHTS = np.array(
    [0.381518036613685, 0.000033166607068, 0.604011066007258,
     0.000133505575004, 0.014299849485899, 0.000004375711085]
)
BOOK_NET_LOG_RESULTS = {
    100: {
        "log_mean": 2.990083640786517e-04,
        "log_std": 0.199020662430273,
        "net_mean": 0.020062858930938,
        "net_std": 0.203490759432704,
        "approximation": -6.413856563117180e-04,
    },
    10_000: {
        "log_mean": 2.298891894512627e-05,
        "log_std": 0.020073619161881,
        "net_mean": 2.244844190541103e-04,
        "net_std": 0.020082451920644,
        "approximation": 2.283198148161179e-05,
    },
    1_000_000: {
        "log_mean": 2.651130365101439e-06,
        "log_std": 0.002002010442023,
        "net_mean": 4.655164389720140e-06,
        "net_std": 0.002002025199583,
        "approximation": 2.651111939837329e-06,
    },
}
CHAPTER_COVERAGE = (
    ("Algorithmic trading loop", "conceptual explanation", "system workflow figure"),
    ("Historical market data", "historical context", "data-quality checklist"),
    ("Live data, platforms, brokers", "historical context", "operational checklist"),
    (
        "Performance metrics",
        "executed formula illustration + concept",
        "CAGR, Sharpe, MDD/duration, Calmar/MAR, scalar Kelly",
    ),
    ("Box 1.1 net vs log returns", "executed and compared", "official MATLAB output"),
    ("Box 1.2 efficient frontier", "executed and compared", "official data and weights"),
    ("Box 1.3 Sharpe/Kelly", "executed formula diagnostic", "analytical vs long-only"),
    ("Estimation risk and risk parity", "executed diagnostic + concept", "split samples"),
    ("Exercises and endnotes", "out of scope", "questions retained in chapter source"),
)

VERIFICATION_CLASSES = {
    "official_data_checksum_matches": "independent_or_empirical",
    "data_has_no_missing_values": "independent_or_empirical",
    "data_prices_are_positive": "independent_or_empirical",
    "net_log_error_decreases": "independent_or_empirical",
    "net_log_final_error_below_1e_9": "independent_or_empirical",
    "tangency_weights_match_book": "independent_or_empirical",
    "minimum_variance_no_worse_than_book": "independent_or_empirical",
    "unconstrained_sharpe_no_lower_than_long_only": "independent_or_empirical",
    "frontier_weights_nonnegative": "contract_invariant",
    "frontier_weights_sum_to_one": "contract_invariant",
    "frontier_meets_target_returns": "contract_invariant",
    "box13_weights_sum_to_one": "contract_invariant",
    "box13_first_order_condition": "contract_invariant",
    "split_sample_weights_are_long_only": "contract_invariant",
    "split_sample_weights_sum_to_one": "contract_invariant",
}


@dataclass(frozen=True)
class NetLogResult:
    sample_size: int
    log_mean: float
    log_std: float
    net_mean: float
    net_std: float
    net_mean_minus_half_variance: float
    absolute_approximation_error: float


@dataclass(frozen=True)
class NetLogSeedSummary:
    sample_size: int
    seed_count: int
    median_error: float
    p05_error: float
    p95_error: float
    seed_one_error: float
    seed_one_percentile: float


@dataclass(frozen=True)
class ETFData:
    symbols: tuple[str, ...]
    dates: pd.DatetimeIndex
    close: np.ndarray


@dataclass(frozen=True)
class FrontierResult:
    symbols: tuple[str, ...]
    mean_returns: np.ndarray
    covariance: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    variances: np.ndarray
    tangency_index: int
    minimum_variance_index: int

    @property
    def standard_deviations(self) -> np.ndarray:
        return np.sqrt(self.variances)

    @property
    def sharpes(self) -> np.ndarray:
        return self.targets / self.standard_deviations

    @property
    def tangency_weights(self) -> np.ndarray:
        return self.weights[self.tangency_index]

    @property
    def minimum_variance_weights(self) -> np.ndarray:
        return self.weights[self.minimum_variance_index]


@dataclass(frozen=True)
class PortfolioTheoryResult:
    unconstrained_weights: np.ndarray
    unconstrained_daily_sharpe: float
    unconstrained_gross_exposure: float
    first_half_weights: np.ndarray
    second_half_weights: np.ndarray
    first_half_daily_sharpe: float
    second_half_daily_sharpe: float
    split_weight_l1_distance: float
    split_date: pd.Timestamp


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=1)


def safe_project_path(relative_path: str) -> Path:
    return resolve_project_path(PROJECT_ROOT, relative_path)


def download_official_assets(force: bool = False) -> list[str]:
    """Download the official archive and materialize verified Ch1 members."""
    manifest = chapter_manifest()
    archive_payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    statuses = materialize_verified_members(
        PROJECT_ROOT, manifest, archive_payload, force=force
    )

    source_note = f"""# Official MATLAB source

Source: {manifest['source_page']}  
Archive: `{manifest['url']}`  
Archive SHA-256: `{manifest['sha256']}`

These files are preserved as the official reference implementation. The
Python experiment fixes paths, makes random seeds explicit, and uses a
deterministic active-set quadratic-program solver. Raw research data are stored under the ignored
project-level `data/` directory.
"""
    write_text_if_changed(CHAPTER_DIR / "original_matlab/SOURCE.md", source_note)
    return statuses


def validate_offline_assets() -> None:
    validate_manifest_members(PROJECT_ROOT, chapter_manifest())


def simulate_net_vs_log_returns(
    sample_sizes: Iterable[int] = (100, 10_000, 1_000_000),
    seed: int = 1,
) -> list[NetLogResult]:
    """Numerically verify mu ~= m - s^2 / 2 using deterministic simulations."""
    results: list[NetLogResult] = []
    for sample_size in sample_sizes:
        if sample_size <= 1:
            raise ValueError("Each sample size must be greater than one")
        rng = np.random.default_rng(seed)
        log_returns = (
            1.0 / sample_size
            + 2.0 / np.sqrt(sample_size) * rng.standard_normal(sample_size)
        )
        net_returns = np.exp(log_returns) - 1.0
        log_mean = float(np.mean(log_returns))
        log_std = float(np.std(log_returns, ddof=1))
        net_mean = float(np.mean(net_returns))
        net_std = float(np.std(net_returns, ddof=1))
        approximation = net_mean - net_std**2 / 2.0
        results.append(
            NetLogResult(
                sample_size=sample_size,
                log_mean=log_mean,
                log_std=log_std,
                net_mean=net_mean,
                net_std=net_std,
                net_mean_minus_half_variance=approximation,
                absolute_approximation_error=abs(log_mean - approximation),
            )
        )
    return results


def summarize_net_log_seed_distribution(
    sample_sizes: Iterable[int] = (100, 10_000, 1_000_000),
    seeds: Iterable[int] = range(1, 201),
) -> list[NetLogSeedSummary]:
    """Quantify sampling luck instead of presenting seed 1 as representative."""
    sizes = tuple(sample_sizes)
    seed_values = tuple(seeds)
    if not sizes or not seed_values or min(sizes) <= 1:
        raise ValueError("Positive seeds and sample sizes greater than one are required")
    error_rows: list[list[float]] = []
    maximum_size = max(sizes)
    for seed in seed_values:
        standard_normals = np.random.default_rng(seed).standard_normal(maximum_size)
        row: list[float] = []
        for sample_size in sizes:
            log_returns = (
                1.0 / sample_size
                + 2.0 / np.sqrt(sample_size)
                * standard_normals[:sample_size]
            )
            net_returns = np.exp(log_returns) - 1.0
            approximation = float(
                np.mean(net_returns)
                - np.std(net_returns, ddof=1) ** 2 / 2.0
            )
            row.append(abs(float(np.mean(log_returns)) - approximation))
        error_rows.append(row)

    errors = np.asarray(error_rows)
    seed_one_index = seed_values.index(1) if 1 in seed_values else 0
    summaries: list[NetLogSeedSummary] = []
    for column, sample_size in enumerate(sizes):
        values = errors[:, column]
        seed_one_error = float(values[seed_one_index])
        summaries.append(
            NetLogSeedSummary(
                sample_size=sample_size,
                seed_count=len(seed_values),
                median_error=float(np.median(values)),
                p05_error=float(np.quantile(values, 0.05)),
                p95_error=float(np.quantile(values, 0.95)),
                seed_one_error=seed_one_error,
                seed_one_percentile=float(100.0 * np.mean(values < seed_one_error)),
            )
        )
    return summaries


def calculate_performance_metric_examples() -> dict[str, float]:
    """Return executable examples for CAGR and scalar Kelly leverage warnings."""
    daily_return = 0.01
    periods = 252
    cagr = (1.0 + daily_return) ** periods - 1.0
    simple_annualization = daily_return * periods
    black_monday_return = -0.20
    leverage = 5.0
    remaining_wealth = 1.0 + leverage * black_monday_return
    return {
        "daily_return": daily_return,
        "periods": float(periods),
        "cagr": cagr,
        "simple_annualized_return": simple_annualization,
        "black_monday_return": black_monday_return,
        "illustrative_leverage": leverage,
        "remaining_wealth_after_shock": remaining_wealth,
    }


def excluded_universe_diagnostics(path: Path = DATA_PATH) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True)
    symbols = np.asarray(payload["stocks"]).astype(str)
    close = np.asarray(payload["cl"], dtype=float)
    return {
        symbol: {
            "column": int(index),
            "missing_values": int(np.count_nonzero(~np.isfinite(close[:, index]))),
        }
        for index, symbol in enumerate(symbols)
        if symbol in {"EWZ", "FXI"}
    }


def load_etf_data(path: Path = DATA_PATH) -> ETFData:
    """Load the official MATLAB v5 dataset and retain the six book ETFs."""
    payload = loadmat(path, squeeze_me=True)
    all_symbols = np.asarray(payload["stocks"]).astype(str)
    keep = ~np.isin(all_symbols, ("EWZ", "FXI"))
    symbols = tuple(all_symbols[keep].tolist())
    if symbols != ("EWC", "EWG", "EWJ", "EWQ", "EWU", "EWY"):
        raise ValueError(f"Unexpected ETF universe: {symbols}")

    close = np.asarray(payload["cl"], dtype=float)[:, keep]
    raw_dates = np.asarray(payload["tday"]).astype(np.int64).astype(str)
    dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, format="%Y%m%d"))
    if close.shape != (len(dates), len(symbols)):
        raise ValueError("Price matrix dimensions do not match dates and symbols")
    return ETFData(symbols=symbols, dates=dates, close=close)


def calculate_net_returns(close: np.ndarray) -> np.ndarray:
    if close.ndim != 2 or close.shape[0] < 2:
        raise ValueError("Close prices must be a two-dimensional time series")
    if not np.all(np.isfinite(close)) or np.any(close <= 0):
        raise ValueError("Close prices must be finite and positive")
    return close[1:] / close[:-1] - 1.0


def _solve_long_only_qp(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    target: float,
) -> tuple[np.ndarray, float]:
    """Solve one six-asset long-only QP by enumerating active asset sets.

    For each possible set of nonzero weights, the equality-constrained minimum
    variance solution has a closed form. Enumerating the 63 nonempty subsets
    avoids optimizer tolerance changes at the boundary and is exact for this
    small Chapter 1 universe.
    """
    asset_count = len(mean_returns)
    best_weights: np.ndarray | None = None
    best_variance = np.inf

    for active_count in range(1, asset_count + 1):
        for active_tuple in combinations(range(asset_count), active_count):
            active = np.asarray(active_tuple)
            active_means = mean_returns[active]
            if active_count == 1:
                if not np.isclose(active_means[0], target, atol=1e-14, rtol=0.0):
                    continue
                active_weights = np.ones(1)
            else:
                active_covariance = covariance[np.ix_(active, active)]
                constraints = np.vstack((np.ones(active_count), active_means))
                right_hand_side = np.array((1.0, target))
                try:
                    covariance_inverse_constraints = np.linalg.solve(
                        active_covariance, constraints.T
                    )
                    multipliers = np.linalg.solve(
                        constraints @ covariance_inverse_constraints,
                        right_hand_side,
                    )
                    active_weights = covariance_inverse_constraints @ multipliers
                except np.linalg.LinAlgError:
                    covariance_inverse_constraints = (
                        np.linalg.pinv(active_covariance) @ constraints.T
                    )
                    active_weights = covariance_inverse_constraints @ np.linalg.pinv(
                        constraints @ covariance_inverse_constraints
                    ) @ right_hand_side

            if np.min(active_weights) < -1e-9:
                continue
            weights = np.zeros(asset_count)
            weights[active] = np.maximum(active_weights, 0.0)
            if abs(np.sum(weights) - 1.0) > 1e-8:
                continue
            if abs(weights @ mean_returns - target) > 1e-10:
                continue
            variance = float(weights @ covariance @ weights)
            if variance < best_variance:
                best_variance = variance
                best_weights = weights

    if best_weights is None:
        raise RuntimeError(f"No feasible portfolio for target return {target}")
    return best_weights, best_variance


def solve_global_long_only_minimum_variance(
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the true long-only global minimum, without the 21-point grid."""
    asset_count = covariance.shape[0]
    best_weights: np.ndarray | None = None
    best_variance = np.inf
    for active_count in range(1, asset_count + 1):
        for active_tuple in combinations(range(asset_count), active_count):
            active = np.asarray(active_tuple)
            active_covariance = covariance[np.ix_(active, active)]
            ones = np.ones(active_count)
            try:
                raw = np.linalg.solve(active_covariance, ones)
            except np.linalg.LinAlgError:
                raw = np.linalg.pinv(active_covariance) @ ones
            normalizer = float(ones @ raw)
            if normalizer <= 0:
                continue
            active_weights = raw / normalizer
            if np.min(active_weights) < -1e-10:
                continue
            weights = np.zeros(asset_count)
            weights[active] = np.maximum(active_weights, 0.0)
            weights /= np.sum(weights)
            variance = float(weights @ covariance @ weights)
            if variance < best_variance:
                best_weights = weights
                best_variance = variance
    if best_weights is None:
        raise RuntimeError("No feasible global minimum-variance portfolio")
    return best_weights, best_variance


def calculate_efficient_frontier(
    data: ETFData,
    target_count: int = 21,
) -> FrontierResult:
    """Solve the book's long-only, fully invested quadratic programs."""
    if target_count < 3:
        raise ValueError("The efficient frontier needs at least three targets")
    returns = calculate_net_returns(data.close)
    mean_returns = np.mean(returns, axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    targets = np.linspace(mean_returns.min(), mean_returns.max(), target_count)

    all_weights: list[np.ndarray] = []
    variances: list[float] = []
    for target in targets:
        weights, variance = _solve_long_only_qp(
            mean_returns, covariance, float(target)
        )
        all_weights.append(weights)
        variances.append(variance)

    weight_matrix = np.vstack(all_weights)
    variance_array = np.asarray(variances)
    standard_deviations = np.sqrt(variance_array)
    sharpes = targets / standard_deviations
    return FrontierResult(
        symbols=data.symbols,
        mean_returns=mean_returns,
        covariance=covariance,
        targets=targets,
        weights=weight_matrix,
        variances=variance_array,
        tangency_index=int(np.argmax(sharpes)),
        minimum_variance_index=int(np.argmin(variance_array)),
    )


def calculate_normalized_kelly_weights(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Return the fully invested analytical C^-1 M allocation from Box 1.3."""
    raw_weights = np.linalg.solve(covariance, mean_returns)
    normalizer = float(np.sum(raw_weights))
    if np.isclose(normalizer, 0.0):
        raise ValueError("Kelly weights cannot be normalized to unit net exposure")
    return raw_weights / normalizer


def calculate_portfolio_theory_diagnostics(
    data: ETFData,
    full_frontier: FrontierResult,
) -> PortfolioTheoryResult:
    """Compare Box 1.3 with long-only and split-sample allocations."""
    unconstrained = calculate_normalized_kelly_weights(
        full_frontier.mean_returns, full_frontier.covariance
    )
    unconstrained_variance = float(
        unconstrained @ full_frontier.covariance @ unconstrained
    )
    unconstrained_sharpe = float(
        (unconstrained @ full_frontier.mean_returns)
        / np.sqrt(unconstrained_variance)
    )

    midpoint = len(data.dates) // 2
    first_data = ETFData(
        symbols=data.symbols,
        dates=data.dates[:midpoint],
        close=data.close[:midpoint],
    )
    second_data = ETFData(
        symbols=data.symbols,
        dates=data.dates[midpoint:],
        close=data.close[midpoint:],
    )
    first_frontier = calculate_efficient_frontier(first_data)
    second_frontier = calculate_efficient_frontier(second_data)
    first_weights = first_frontier.tangency_weights
    second_weights = second_frontier.tangency_weights

    return PortfolioTheoryResult(
        unconstrained_weights=unconstrained,
        unconstrained_daily_sharpe=unconstrained_sharpe,
        unconstrained_gross_exposure=float(np.sum(np.abs(unconstrained))),
        first_half_weights=first_weights,
        second_half_weights=second_weights,
        first_half_daily_sharpe=float(
            first_frontier.sharpes[first_frontier.tangency_index]
        ),
        second_half_daily_sharpe=float(
            second_frontier.sharpes[second_frontier.tangency_index]
        ),
        split_weight_l1_distance=float(np.sum(np.abs(first_weights - second_weights))),
        split_date=data.dates[midpoint],
    )


def create_net_log_figure(
    results: list[NetLogResult],
    seed_summary: list[NetLogSeedSummary] | None = None,
) -> plt.Figure:
    """Construct the Box 1.1 figure for files or inline notebook output."""
    log_means = np.array([item.log_mean for item in results])
    approximations = np.array(
        [item.net_mean_minus_half_variance for item in results]
    )
    sample_sizes = np.array([item.sample_size for item in results])
    errors = np.array([item.absolute_approximation_error for item in results])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lower = min(log_means.min(), approximations.min())
    upper = max(log_means.max(), approximations.max())
    axes[0].plot([lower, upper], [lower, upper], "--", color="0.5", label="equality")
    axes[0].scatter(log_means, approximations, s=70, color="#2563eb")
    label_offsets = ((5, 5), (5, -14), (8, 8))
    for size, x_value, y_value, offset in zip(
        sample_sizes, log_means, approximations, label_offsets
    ):
        axes[0].annotate(
            f"N={size:,}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_xlabel("sample mean of log returns, μ")
    axes[0].set_ylabel("m - s²/2")
    axes[0].set_title("Net-return approximation")
    axes[0].legend()

    if seed_summary is not None:
        medians = np.array([item.median_error for item in seed_summary])
        p05 = np.array([item.p05_error for item in seed_summary])
        p95 = np.array([item.p95_error for item in seed_summary])
        axes[1].fill_between(
            sample_sizes,
            p05,
            p95,
            color="#93c5fd",
            alpha=0.45,
            label="200-seed p05–p95",
        )
        axes[1].loglog(
            sample_sizes,
            medians,
            marker="s",
            color="#2563eb",
            label="200-seed median",
        )
    axes[1].loglog(
        sample_sizes,
        errors,
        marker="o",
        color="#dc2626",
        label="seed 1",
    )
    axes[1].set_xlabel("number of subperiods N")
    axes[1].set_ylabel("|μ - (m - s²/2)|")
    axes[1].set_title("Sampling distribution and convergence")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_net_log_results(
    results: list[NetLogResult],
    seed_summary: list[NetLogSeedSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_net_log_figure(results, seed_summary)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_efficient_frontier_figure(result: FrontierResult) -> plt.Figure:
    """Construct the Box 1.2 frontier and allocation figure."""
    scale = np.sqrt(252.0)
    annual_returns = result.targets * 252.0
    annual_volatility = result.standard_deviations * scale
    tangency = result.tangency_index
    minimum = result.minimum_variance_index

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(annual_volatility, annual_returns, marker="o", ms=4,
                 color="#2563eb", label="efficient frontier")
    axes[0].scatter(annual_volatility[tangency], annual_returns[tangency],
                    s=90, color="#dc2626", label="tangency")
    axes[0].scatter(annual_volatility[minimum], annual_returns[minimum],
                    s=90, color="#16a34a", label="minimum variance")
    axes[0].set_xlabel("annualized volatility")
    axes[0].set_ylabel("annualized expected return")
    axes[0].set_title("Long-only efficient frontier")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    x_positions = np.arange(len(result.symbols))
    width = 0.36
    axes[1].bar(x_positions - width / 2, result.tangency_weights, width,
                label="tangency", color="#dc2626")
    axes[1].bar(x_positions + width / 2, result.minimum_variance_weights, width,
                label="minimum variance", color="#16a34a")
    axes[1].set_xticks(x_positions, result.symbols)
    axes[1].set_ylabel("portfolio weight")
    axes[1].set_title("Optimized allocations")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_efficient_frontier(result: FrontierResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_efficient_frontier_figure(result)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_data_diagnostics_figure(data: ETFData) -> plt.Figure:
    """Show normalized prices and daily-return correlation before modeling."""
    prices = pd.DataFrame(data.close, index=data.dates, columns=data.symbols)
    returns = prices.pct_change(fill_method=None).dropna()
    normalized = prices / prices.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    normalized.plot(ax=axes[0], linewidth=1.1)
    axes[0].set_title("Normalized ETF close prices")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("growth of $1")
    axes[0].grid(alpha=0.2)

    correlation = returns.corr().to_numpy()
    image = axes[1].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(len(data.symbols)), data.symbols)
    axes[1].set_yticks(range(len(data.symbols)), data.symbols)
    axes[1].set_title("Daily net-return correlation")
    for row in range(len(data.symbols)):
        for column in range(len(data.symbols)):
            axes[1].text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(correlation[row, column]) > 0.55 else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_data_diagnostics(data: ETFData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_data_diagnostics_figure(data)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_trading_system_figure() -> plt.Figure:
    """Visualize the chapter's shared research-to-execution architecture."""
    fig, axis = plt.subplots(figsize=(13, 5.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = {
        "Historical data": (0.11, 0.72, "#dbeafe"),
        "Live data": (0.11, 0.30, "#dbeafe"),
        "Shared strategy\nlogic": (0.39, 0.51, "#fef3c7"),
        "Backtest engine\nand metrics": (0.67, 0.72, "#dcfce7"),
        "Orders / broker API": (0.67, 0.30, "#fee2e2"),
        "Fills, status,\npositions": (0.90, 0.30, "#f3e8ff"),
    }
    for label, (x_value, y_value, color) in boxes.items():
        axis.text(
            x_value,
            y_value,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": color,
                "edgecolor": "#334155",
                "linewidth": 1.2,
            },
        )

    arrow_pairs = (
        ((0.18, 0.72), (0.32, 0.55)),
        ((0.18, 0.30), (0.32, 0.47)),
        ((0.47, 0.56), (0.59, 0.69)),
        ((0.47, 0.46), (0.59, 0.33)),
        ((0.75, 0.30), (0.83, 0.30)),
        ((0.90, 0.22), (0.46, 0.22)),
    )
    for start, end in arrow_pairs:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.6},
        )
    axis.text(
        0.65,
        0.13,
        "feedback must update the same strategy state and logic",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    axis.set_title(
        "One strategy core for research, backtesting, and live execution",
        fontsize=15,
        pad=18,
    )
    fig.tight_layout()
    return fig


def plot_trading_system(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_trading_system_figure()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_portfolio_theory_figure(
    symbols: tuple[str, ...],
    frontier: FrontierResult,
    theory: PortfolioTheoryResult,
) -> plt.Figure:
    """Show Box 1.3 constraints and instability across sample periods."""
    positions = np.arange(len(symbols))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    width = 0.36
    axes[0].bar(
        positions - width / 2,
        frontier.tangency_weights,
        width,
        label="long-only grid",
        color="#2563eb",
    )
    axes[0].bar(
        positions + width / 2,
        theory.unconstrained_weights,
        width,
        label="normalized C⁻¹M",
        color="#f97316",
    )
    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_xticks(positions, symbols)
    axes[0].set_ylabel("net portfolio weight")
    axes[0].set_title("Box 1.3: constraints change the allocation")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    split_width = 0.25
    axes[1].bar(
        positions - split_width,
        theory.first_half_weights,
        split_width,
        label="first half",
        color="#0f766e",
    )
    axes[1].bar(
        positions,
        theory.second_half_weights,
        split_width,
        label="second half",
        color="#a855f7",
    )
    axes[1].bar(
        positions + split_width,
        frontier.tangency_weights,
        split_width,
        label="full sample",
        color="#64748b",
    )
    axes[1].set_xticks(positions, symbols)
    axes[1].set_ylabel("long-only tangency weight")
    axes[1].set_title("Estimated tangency weights are sample-sensitive")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_portfolio_theory(
    symbols: tuple[str, ...],
    frontier: FrontierResult,
    theory: PortfolioTheoryResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = create_portfolio_theory_figure(symbols, frontier, theory)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def weights_dict(symbols: tuple[str, ...], weights: np.ndarray) -> dict[str, float]:
    return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}


def verify_results(
    data: ETFData,
    seed_summary: list[NetLogSeedSummary],
    frontier: FrontierResult,
    theory: PortfolioTheoryResult,
) -> dict[str, bool]:
    errors = np.array([item.median_error for item in seed_summary])
    tangency_difference = np.max(
        np.abs(frontier.tangency_weights - BOOK_TANGENCY_WEIGHTS)
    )
    observed_minimum_variance = frontier.variances[frontier.minimum_variance_index]
    book_minimum_variance = float(
        BOOK_MINIMUM_VARIANCE_WEIGHTS
        @ frontier.covariance
        @ BOOK_MINIMUM_VARIANCE_WEIGHTS
    )
    manifest = chapter_manifest()
    data_member = next(
        member for member in manifest["members"] if member["kind"] == "research_data"
    )
    covariance_times_weights = frontier.covariance @ theory.unconstrained_weights
    proportionality = float(
        (frontier.mean_returns @ covariance_times_weights)
        / (frontier.mean_returns @ frontier.mean_returns)
    )
    first_order_residual = float(
        np.max(
            np.abs(
                covariance_times_weights
                - proportionality * frontier.mean_returns
            )
        )
    )
    checks = {
        "official_data_checksum_matches": sha256_file(DATA_PATH)
        == data_member["sha256"],
        "data_has_no_missing_values": bool(np.all(np.isfinite(data.close))),
        "data_prices_are_positive": bool(np.all(data.close > 0)),
        "net_log_error_decreases": bool(np.all(np.diff(errors) < 0)),
        "net_log_final_error_below_1e_9": bool(errors[-1] < 1e-9),
        "frontier_weights_nonnegative": bool(np.min(frontier.weights) >= -1e-10),
        "frontier_weights_sum_to_one": bool(
            np.allclose(np.sum(frontier.weights, axis=1), 1.0, atol=1e-8)
        ),
        "frontier_meets_target_returns": bool(
            np.allclose(
                frontier.weights @ frontier.mean_returns,
                frontier.targets,
                atol=1e-10,
                rtol=0.0,
            )
        ),
        "tangency_weights_match_book": bool(tangency_difference < 2e-4),
        "minimum_variance_no_worse_than_book": bool(
            observed_minimum_variance <= book_minimum_variance * 1.000001
        ),
        "box13_weights_sum_to_one": bool(
            np.isclose(np.sum(theory.unconstrained_weights), 1.0, atol=1e-12)
        ),
        "box13_first_order_condition": bool(first_order_residual < 1e-12),
        "unconstrained_sharpe_no_lower_than_long_only": bool(
            theory.unconstrained_daily_sharpe
            >= frontier.sharpes[frontier.tangency_index] - 1e-12
        ),
        "split_sample_weights_are_long_only": bool(
            np.min(theory.first_half_weights) >= -1e-10
            and np.min(theory.second_half_weights) >= -1e-10
        ),
        "split_sample_weights_sum_to_one": bool(
            np.isclose(np.sum(theory.first_half_weights), 1.0, atol=1e-8)
            and np.isclose(np.sum(theory.second_half_weights), 1.0, atol=1e-8)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Chapter 1 verification failed: {', '.join(failed)}")
    return checks


def verification_summary(checks: dict[str, bool]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {
        "independent_or_empirical": [],
        "contract_invariant": [],
    }
    for name in checks:
        grouped[VERIFICATION_CLASSES[name]].append(name)
    return {
        "total": len(checks),
        "passed": sum(checks.values()),
        "independent_or_empirical": len(grouped["independent_or_empirical"]),
        "contract_invariant": len(grouped["contract_invariant"]),
        "classification": grouped,
    }


def build_metrics(
    data: ETFData,
    net_log_results: list[NetLogResult],
    seed_summary: list[NetLogSeedSummary],
    frontier: FrontierResult,
    theory: PortfolioTheoryResult,
    checks: dict[str, bool],
) -> dict[str, Any]:
    tangency = frontier.tangency_index
    minimum = frontier.minimum_variance_index
    tangency_difference = np.abs(frontier.tangency_weights - BOOK_TANGENCY_WEIGHTS)
    minimum_difference = np.abs(
        frontier.minimum_variance_weights - BOOK_MINIMUM_VARIANCE_WEIGHTS
    )
    book_min_variance = float(
        BOOK_MINIMUM_VARIANCE_WEIGHTS
        @ frontier.covariance
        @ BOOK_MINIMUM_VARIANCE_WEIGHTS
    )
    global_min_weights, global_min_variance = (
        solve_global_long_only_minimum_variance(frontier.covariance)
    )
    manifest = chapter_manifest()
    data_member = next(
        member for member in manifest["members"] if member["kind"] == "research_data"
    )
    returns = calculate_net_returns(data.close)
    return {
        "chapter": 1,
        "chapter_coverage": [
            {"topic": topic, "status": status, "evidence": evidence}
            for topic, status, evidence in CHAPTER_COVERAGE
        ],
        "random_seed": 1,
        "random_generator": "NumPy PCG64",
        "provenance": {
            "source_page": manifest["source_page"],
            "archive_url": manifest["url"],
            "archive_sha256": manifest["sha256"],
            "input_archive_member": data_member["archive_path"],
            "input_sha256": sha256_file(DATA_PATH),
            "pyproject_sha256": sha256_file(PYPROJECT_PATH),
            "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
            "environment": environment_versions(),
        },
        "data": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "start_date": data.dates[0].date().isoformat(),
            "end_date": data.dates[-1].date().isoformat(),
            "price_observations": len(data.dates),
            "return_observations": len(data.dates) - 1,
            "symbols": list(data.symbols),
            "excluded_symbols": ["EWZ", "FXI"],
            "excluded_symbol_diagnostics": excluded_universe_diagnostics(),
            "missing_values": int(np.size(data.close) - np.count_nonzero(np.isfinite(data.close))),
            "nonpositive_values": int(np.count_nonzero(data.close <= 0)),
            "minimum_price": float(np.min(data.close)),
            "maximum_price": float(np.max(data.close)),
            "annualized_mean_net_returns": weights_dict(
                data.symbols, np.mean(returns, axis=0) * 252.0
            ),
            "annualized_volatility": weights_dict(
                data.symbols, np.std(returns, axis=0, ddof=1) * np.sqrt(252.0)
            ),
        },
        "net_vs_log_returns": [asdict(item) for item in net_log_results],
        "net_vs_log_seed_distribution": [asdict(item) for item in seed_summary],
        "performance_metric_examples": calculate_performance_metric_examples(),
        "efficient_frontier": {
            "target_count": len(frontier.targets),
            "tangency": {
                "target_daily_return": float(frontier.targets[tangency]),
                "daily_volatility": float(frontier.standard_deviations[tangency]),
                "daily_sharpe": float(frontier.sharpes[tangency]),
                "weights": weights_dict(data.symbols, frontier.tangency_weights),
                "book_weights": weights_dict(data.symbols, BOOK_TANGENCY_WEIGHTS),
                "max_absolute_weight_difference": float(tangency_difference.max()),
            },
            "minimum_variance": {
                "target_daily_return": float(frontier.targets[minimum]),
                "daily_volatility": float(frontier.standard_deviations[minimum]),
                "variance": float(frontier.variances[minimum]),
                "weights": weights_dict(
                    data.symbols, frontier.minimum_variance_weights
                ),
                "book_weights": weights_dict(
                    data.symbols, BOOK_MINIMUM_VARIANCE_WEIGHTS
                ),
                "max_absolute_weight_difference": float(minimum_difference.max()),
                "book_weights_variance_on_same_data": book_min_variance,
                "global_minimum_weights": weights_dict(
                    data.symbols, global_min_weights
                ),
                "global_minimum_variance": global_min_variance,
                "grid_variance_gap": float(
                    frontier.variances[minimum] - global_min_variance
                ),
            },
        },
        "portfolio_theory": {
            "box_1_3": {
                "normalized_kelly_weights": weights_dict(
                    data.symbols, theory.unconstrained_weights
                ),
                "daily_sharpe_rf0": theory.unconstrained_daily_sharpe,
                "gross_exposure": theory.unconstrained_gross_exposure,
                "allows_short_sales": bool(np.any(theory.unconstrained_weights < 0)),
            },
            "split_sample_stability": {
                "split_date": theory.split_date.date().isoformat(),
                "first_half_weights": weights_dict(
                    data.symbols, theory.first_half_weights
                ),
                "second_half_weights": weights_dict(
                    data.symbols, theory.second_half_weights
                ),
                "first_half_daily_sharpe_rf0": theory.first_half_daily_sharpe,
                "second_half_daily_sharpe_rf0": theory.second_half_daily_sharpe,
                "weight_l1_distance": theory.split_weight_l1_distance,
            },
        },
        "verification": {
            "checks": checks,
            "summary": verification_summary(checks),
        },
    }


def markdown_weight_table(
    symbols: tuple[str, ...],
    observed: np.ndarray,
    expected: np.ndarray,
) -> str:
    lines = ["| ETF | Python | Book | Difference |", "|---|---:|---:|---:|"]
    for symbol, actual, reference in zip(symbols, observed, expected):
        lines.append(
            f"| {symbol} | {actual:.6f} | {reference:.6f} | "
            f"{actual - reference:+.6f} |"
        )
    return "\n".join(lines)


def markdown_coverage_table() -> str:
    lines = [
        "| Chapter topic | Reproduction status | Evidence |",
        "|---|---|---|",
    ]
    for topic, status, evidence in CHAPTER_COVERAGE:
        lines.append(f"| {topic} | {status} | {evidence} |")
    return "\n".join(lines)


def write_report(
    data: ETFData,
    net_log_results: list[NetLogResult],
    seed_summary: list[NetLogSeedSummary],
    frontier: FrontierResult,
    theory: PortfolioTheoryResult,
    metrics: dict[str, Any],
) -> None:
    net_lines = [
        "| N | μ | m | s | m - s²/2 | absolute error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in net_log_results:
        net_lines.append(
            f"| {item.sample_size:,} | {item.log_mean:.8e} | "
            f"{item.net_mean:.8e} | {item.net_std:.8e} | "
            f"{item.net_mean_minus_half_variance:.8e} | "
            f"{item.absolute_approximation_error:.3e} |"
        )

    comparison_lines = [
        "| N | Python μ | Book μ | Python m-s²/2 | Book m-s²/2 | "
        "Python identity error | Book identity error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in net_log_results:
        book = BOOK_NET_LOG_RESULTS[item.sample_size]
        comparison_lines.append(
            f"| {item.sample_size:,} | {item.log_mean:.8e} | "
            f"{book['log_mean']:.8e} | "
            f"{item.net_mean_minus_half_variance:.8e} | "
            f"{book['approximation']:.8e} | "
            f"{item.absolute_approximation_error:.3e} | "
            f"{abs(book['log_mean'] - book['approximation']):.3e} |"
        )

    seed_lines = [
        "| N | seeds | median error | p05 | p95 | seed 1 | seed 1 percentile |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in seed_summary:
        seed_lines.append(
            f"| {item.sample_size:,} | {item.seed_count} | "
            f"{item.median_error:.3e} | {item.p05_error:.3e} | "
            f"{item.p95_error:.3e} | {item.seed_one_error:.3e} | "
            f"{item.seed_one_percentile:.1f}% |"
        )

    tangency = frontier.tangency_index
    minimum = frontier.minimum_variance_index
    manifest = chapter_manifest()
    provenance = metrics["provenance"]
    minimum_metrics = metrics["efficient_frontier"]["minimum_variance"]
    performance = metrics["performance_metric_examples"]
    verification = metrics["verification"]["summary"]
    excluded = metrics["data"]["excluded_symbol_diagnostics"]
    theory_lines = [
        "| ETF | Long-only tangency | Normalized C⁻¹M | First half | Second half |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, symbol in enumerate(data.symbols):
        theory_lines.append(
            f"| {symbol} | {frontier.tangency_weights[index]:.6f} | "
            f"{theory.unconstrained_weights[index]:.6f} | "
            f"{theory.first_half_weights[index]:.6f} | "
            f"{theory.second_half_weights[index]:.6f} |"
        )

    report = fr"""# Chapter 1 Python 종합 분석 리포트

## 1. 장의 목표와 구현 범위

이 장의 핵심은 특정 전략 하나가 아니라 알고리즘 트레이딩의 전체 흐름이다.
시장 데이터가 같은 전략 로직으로 들어가 백테스트 또는 주문을 만들고, 브로커
API의 체결·상태가 다시 포지션 상태로 돌아와야 한다. 수치 부분에서는 순수익률과
로그수익률, 성과지표, 효율적 투자선, Sharpe/Kelly 최적화를 연결한다.

### Chapter coverage / 재현 상태

{markdown_coverage_table()}

공식 코드가 있는 Box 1.1과 1.2는 계산·책 비교까지 수행한다. Box 1.3은 책의
`C⁻¹M` 해를 공식 ETF 데이터에 적용한다. 데이터·플랫폼·브로커 목록은 2016년
책의 역사적 맥락이므로 현재 제품 추천으로 해석하지 않는다.

![알고리즘 트레이딩 시스템 흐름](figures/trading_system_workflow.png)

## 2. 데이터, 플랫폼, 브로커의 실무 원칙

### 과거·실시간 데이터

| 영역 | 장에서 강조하는 검증 질문 |
|---|---|
| 주식/ETF | 분할·배당 조정, 상장폐지 종목, 당시 지수 구성종목이 포함되는가? |
| 시가/종가 | 합성 종가 대신 주 거래소 경매가 또는 BBO 중간값을 쓸 수 있는가? |
| 선물 | 연속선물 롤 규칙이 신호에 미래 정보를 넣지 않는가? |
| 옵션 | 마지막 체결가보다 종가 bid/ask, Greeks, 변동성 표면이 필요한가? |
| 실시간 피드 | 타임스탬프, 지연, 누락·중복, 거래소 직접/통합 피드 차이를 통제하는가? |

책은 CSI, Quandl, CRSP, Bloomberg 등 당시 서비스를 예로 들지만, 재사용할
원칙은 데이터 출처와 조정 규칙을 고정하고 survivorship/look-ahead bias를
검사하는 것이다.

### 백테스트·실거래·브로커

백테스트와 실거래가 서로 다른 전략 구현을 쓰면 연구 결과를 배포하는 순간
논리가 달라질 수 있다. 신호와 포지션 상태 전이는 공유하고, 데이터 어댑터,
시뮬레이션 체결기, 실제 broker API만 교체하는 구조가 핵심이다. 주문 거부,
부분 체결, 재접속, 중복 주문, 마진과 계좌 보호 범위도 전략 수익률과 별개의
운영 리스크다.

## 3. 성과지표와 포트폴리오 수식

가격 `P_t`의 순수익률과 로그수익률은

$$R_t=\frac{{P_t}}{{P_{{t-1}}}}-1, \qquad r_t=\log\left(\frac{{P_t}}{{P_{{t-1}}}}\right)$$

이고, 로그수익률이 정규분포라는 가정 아래

$$\mu \approx m-\frac{{s^2}}{{2}}.$$

초기 자산 $V_0$, 마지막 자산 $V_T$, 연수 $Y$일 때 CAGR은

$$\mathrm{{CAGR}}=\left(\frac{{V_T}}{{V_0}}\right)^{{1/Y}}-1$$

이다. 일 1%를 252일 복리화하면 CAGR은
`{performance['cagr']:.1%}`지만 단순 연환산은
`{performance['simple_annualized_return']:.1%}`에 불과하다. 책이 백테스트의
레버리지를 1로 두라고 한 이유도 이 비교 가능성 때문이다. 일별 P&L을 전일
포지션의 총 gross market value로 나눠야 이 계약을 지킬 수 있다.

연환산 Sharpe는 대략 `sqrt(252) × mean(excess return) / std(return)`이다.
MDD는 누적자산이 직전 고점에서 얼마나 하락했는지, drawdown duration은 고점을
회복하지 못한 기간을 측정한다. 책의 정의에서 Calmar는 CAGR을 최근 36개월
MDD로 나누고, MAR은 inception 이후 CAGR을 inception 이후 MDD로 나눈다.
긴 백테스트일수록 MDD가 커지기 쉬워 MAR가 표본 길이에 민감하다는 점이 책이
Calmar를 선호한 이유다. 변동성만으로 tail risk를 다 설명할 수 없으므로
Sharpe와 MDD를 함께 봐야 한다.

초과수익률의 평균과 분산으로 계산하는 스칼라 Kelly 레버리지는

$$f^*=\frac{{\operatorname{{E}}[R-R_f]}}{{\operatorname{{Var}}(R-R_f)}}$$

이다. 추정오차와 fat tail을 무시하면 레버리지를 과대평가한다. 예를 들어
20% 급락에 5배 노출이면 비용 전 자산이 초기 1.0배에서
`{performance['remaining_wealth_after_shock']:.1f}`배가 되어 청산된다. 따라서
Kelly 비중은 상한, stress test, margin 규칙과 함께 써야 한다.

포트폴리오 분산 최소화 문제는

$$\min_F F^TCF,\quad F^TM=m_p,\quad F^T\mathbf{{1}}=1,\quad F\geq0$$

이며 Box 1.3의 공매도 허용 분석해는 `F ∝ C⁻¹M`이다.

## 4. 공식 데이터와 provenance

| 항목 | 값 |
|---|---|
| 공식 배포 페이지 | [{manifest['source_page']}]({manifest['source_page']}) |
| 공식 ZIP | `{manifest['url']}` |
| 입력 파일 | `{metrics['data']['path']}` |
| 기간 | {data.dates[0].date()} ~ {data.dates[-1].date()} |
| 가격/수익률 관측치 | {len(data.dates):,} / {len(data.dates) - 1:,} |
| 사용 ETF | {', '.join(data.symbols)} |
| 제외 ETF | EWZ (결측 {excluded['EWZ']['missing_values']:,}), FXI (결측 {excluded['FXI']['missing_values']:,}) |
| 결측/비양수 가격 | {metrics['data']['missing_values']} / {metrics['data']['nonpositive_values']} |

공식 `ef.m`은 `stocks`를 먼저 축소한 뒤 이미 축소된 배열로 `cl`의 열 마스크를
다시 만드는 순서 문제가 있다. Python은 원래 8개 심볼에서 마스크를 한 번
계산하여 책의 주석과 6개 ETF 출력이 의도한 동작을 구현한다. 다만 공식 MAT의
심볼 순서가 `EWC, EWG, EWJ, EWQ, EWU, EWY, EWZ, FXI`여서 원본의 잘못된
`cl(:, 1:6)`도 우연히 같은 여섯 열을 고른다. 즉 이 데이터에서는 버그가
책 수치에 영향을 주지 않는다. EWZ 제외는 2,500행 중
{excluded['EWZ']['missing_values']:,}개 결측이라는 데이터 가용성 근거가 있지만,
FXI 제외는 결측이 없는 사후 선택이므로 selection bias 가능성을 별도로 남긴다.

![데이터 진단](figures/data_diagnostics.png)

## 5. Box 1.1 — 순수익률과 로그수익률

### 목적·방법·Python 결과

`N=100, 10,000, 1,000,000`으로 기간을 나누고 각 `N`에서 평균 `1/N`,
표준편차 `2/sqrt(N)`인 로그수익률을 만든다. `R=exp(r)-1`로 변환한 뒤 표본
`μ`와 `m-s²/2`의 차이를 계산한다. NumPy `PCG64`, seed 1을 명시했다.

{chr(10).join(net_lines)}

### 책과 비교·해석

{chr(10).join(comparison_lines)}

### 난수 표본운 공개

{chr(10).join(seed_lines)}

seed 1의 `N=100` 오차는 200개 시드 중
`{seed_summary[0].seed_one_percentile:.1f}` 퍼센타일로, 중앙값보다 약
`{seed_summary[0].median_error / seed_summary[0].seed_one_error:.0f}`배 작게 나온
비대표적 표본이다. 책의 같은 오차는
`{abs(BOOK_NET_LOG_RESULTS[100]['log_mean'] - BOOK_NET_LOG_RESULTS[100]['approximation']):.3e}`로
200시드 중앙값과 비슷하다. 그림은 seed 1만 잇지 않고 200시드 중앙값과
p05–p95 범위를 함께 그린다. 책의 MATLAB `rng(1)`과 NumPy PCG64는 난수열이
다르므로 이 실험은 **근사 재현**이며, 검증 대상은 개별 표본의 우열이 아니라
다중 시드에서 `N` 증가에 따라 항등식 오차가 수렴하는 성질이다. 세 `N`은 각
시드에서 같은 난수열의 prefix를 써 책의 재시드 동작을 재현하므로 서로 독립인
표본은 아니다.

![순수익률과 로그수익률](figures/net_vs_log_returns.png)

## 6. Box 1.2 — 롱온리 효율적 투자선

공식 종가의 평균 순수익률과 공분산으로 21개 목표수익률을 만든다. 가능한
63개 활성 자산 집합을 열거하고 KKT 선형식을 풀어, 최적화기 버전에 따른 경계
허용오차를 피한다.

- Tangency 일수익률: {frontier.targets[tangency]:.8f}
- Tangency 일변동성: {frontier.standard_deviations[tangency]:.8f}
- 무위험수익률 0 가정 일별 Sharpe: {frontier.sharpes[tangency]:.6f}
- 최소분산 지점 일변동성: {frontier.standard_deviations[minimum]:.8f}

### Tangency 비중 — 책과 수치 비교

{markdown_weight_table(data.symbols, frontier.tangency_weights, BOOK_TANGENCY_WEIGHTS)}

최대 절대 차이는
`{metrics['efficient_frontier']['tangency']['max_absolute_weight_difference']:.3e}`로
허용오차 `2e-4` 안이다. 이는 **수치 재현**이다.

### 최소분산 격자점 — 책과 수치 비교

{markdown_weight_table(data.symbols, frontier.minimum_variance_weights, BOOK_MINIMUM_VARIANCE_WEIGHTS)}

Python 분산은 `{minimum_metrics['variance']:.12e}`, 책 비중을 같은 데이터에
적용한 분산은 `{minimum_metrics['book_weights_variance_on_same_data']:.12e}`이다.
Python 해가 낮고 두 해의 목표수익률은 같다. 역사적 MATLAB `quadprog`가 0
근처 비중을 남긴 허용오차 차이이며, 결과를 억지로 책 숫자에 맞추지 않았다.
여기서 “최소분산”은 21개 목표수익률 격자 중 최소다. 목표수익률 제약을 없앤
진짜 롱온리 global minimum variance는
`{minimum_metrics['global_minimum_variance']:.12e}`이고 격자점과의 분산 차이는
`{minimum_metrics['grid_variance_gap']:.3e}`다.

![효율적 투자선](figures/efficient_frontier.png)

## 7. Box 1.3 — Sharpe/Kelly와 추정 불안정성

공매도 제약이 없을 때 기대 로그성장률을 최대화하는 Kelly 해와 고정 순노출의
Sharpe 최대화 해는 비례하며, 완전투자 정규화는

$$F_{{analytic}}=\frac{{C^{{-1}}M}}{{\mathbf{{1}}^TC^{{-1}}M}}$$

이다. 이 해는 Box 1.2의 롱온리 제약을 포함하지 않는다.

{chr(10).join(theory_lines)}

- 분석해 일별 Sharpe: `{theory.unconstrained_daily_sharpe:.6f}`
- 롱온리 격자 tangency 일별 Sharpe: `{frontier.sharpes[tangency]:.6f}`
- 분석해 gross exposure: `{theory.unconstrained_gross_exposure:.3f}` (net exposure 1)
- 전·후반 롱온리 비중 L1 거리: `{theory.split_weight_l1_distance:.3f}`

분석해는 공매도로 더 높은 표본 Sharpe를 얻지만 gross exposure가 크다. 더
중요하게, `{theory.split_date.date()}` 전후로 추정한 롱온리 tangency 비중은
서로 거의 겹치지 않는다. 이는 과거 평균수익률의 작은 변화가 최적 비중을 크게
바꾼다는 책의 경고를 실제 데이터로 보여준다. 최소분산이나 수축 추정, 위험예산
방식이 실무에서 선호되는 이유다. Risk parity는 공분산 최적화와 같지 않으며,
단순 역변동성 방식은 상관관계와 tail risk를 놓칠 수 있다. 책은 등레버리지·
risk parity에서 위기 손실과 청산이 다른 구성요소로 전염될 수 있다고 경고하고,
그 대안으로 각 구성요소의 MDD를 같은 목표로 두는 접근을 제시한다. 이 노트북의
추가 경고는 MDD 타깃도 추정오차와 상관관계 급변에서 자유롭지 않다는 점이다.
All Weather의 2015년 약 7% 손실은 risk parity가 tail-risk 면역 전략이 아님을
보여주는 역사적 예시다.

![Box 1.3과 표본 민감도](figures/portfolio_theory_diagnostics.png)

## 8. 포트폴리오 수치 실험 — 백테스트가 아님

이 결과는 전체 또는 두 하위 표본의 평균·공분산으로 같은 표본의 비중을 구한
정적 진단이다. 시간 순서대로 학습하고 다음 기간에 적용하지 않았으므로 성과
백테스트나 표본 외 검증이 아니다. 실제 전략에는 rolling/expanding window,
다음 기간 체결, turnover, 거래비용, bid-ask spread, slippage가 필요하다.

### 한계와 편향

- 전체 기간 최적 비중을 미래 성과로 읽으면 look-ahead bias다.
- 사후 ETF 선택과 EWZ·FXI 제외에는 selection/survivorship bias 가능성이 있다.
- 공식 파일의 가격 조정 여부를 외부 공급자 데이터로 독립 검증하지 않았다.
- 무위험수익률을 0으로 두었고 세금·시장 충격·마진·차입비용을 계산하지 않았다.
- 표본 평균과 공분산을 점추정치로 사용해 통계적 불확실성을 직접 모델링하지 않았다.

## 9. 결론과 재현성

Box 1.1의 수렴, Box 1.2의 책 비중, Box 1.3의 일차조건을 재현했다. 동시에
Box 1.3 분석해의 큰 gross exposure와 반기별 tangency 비중의 급변은 “표본 내
최적화”가 곧 실전 최적화가 아니라는 반례다. 따라서 이 장의 실전 결론은 가장
높은 표본 Sharpe를 택하는 것이 아니라 데이터 계보, 동일한 연구/실거래 로직,
다중 위험지표, 견고한 비중, 표본 외 검증을 함께 관리하는 것이다.

```bash
uv sync --locked
uv run python chapter_1_the_basics_of_algorithmic_trading/src/run_chapter1_analysis.py --offline
```

| 재현 정보 | 값 |
|---|---|
| 공식 ZIP SHA-256 | `{provenance['archive_sha256']}` |
| 입력 MAT SHA-256 | `{provenance['input_sha256']}` |
| `uv.lock` SHA-256 | `{provenance['uv_lock_sha256']}` |
| Python | {provenance['environment']['python']} |
| NumPy / pandas / SciPy | {provenance['environment']['numpy']} / {provenance['environment']['pandas']} / {provenance['environment']['scipy']} |

총 {verification['total']}개 검증이 모두 통과했다. 이 중 책 수치·공식 해시·
경험적 성질을 독립 대조하는 검증은 {verification['independent_or_empirical']}개,
계산 함수가 자신의 계약을 다시 확인하는 invariant는
{verification['contract_invariant']}개다. invariant를 외부 증거로 과장하지 않는다.
상세 분류와 수치는 [`metrics.json`](metrics.json)에 저장된다.
"""
    assert_clean_markdown_math(report, "Chapter 1 report")
    write_text_if_changed(REPORT_DIR / "chapter1_report.md", report)


def write_chapter_readme() -> None:
    content = """# Chapter 1 실험 환경

*Machine Trading* 1장의 시스템·데이터·성과지표·포트폴리오 주제를 한국어로
연결하고 Box 1.1~1.3을 실행하는 Python 프로젝트다. 공식 코드가 있는 Box
1.1·1.2는 책과 수치 비교하고, Box 1.3은 공식 ETF 데이터에 분석해를 적용한다.
포트폴리오 결과는 전략 성과 백테스트가 아니라 정적 수치 실험이다.

## 실행

책 프로젝트 디렉터리에서 실행한다.

```bash
uv sync --locked
uv run python chapter_1_the_basics_of_algorithmic_trading/src/run_chapter1_analysis.py --offline
```

최초 다운로드가 필요하면 `--offline`을 빼고 실행한다. 공식 ZIP과 추출 파일의
SHA-256을 확인하며, 원시 데이터는 Git에서 제외된 `data/raw/` 아래에 둔다.
다운로드 상태만 검증하려면 다음을 실행한다.

```bash
uv run python chapter_1_the_basics_of_algorithmic_trading/src/run_chapter1_analysis.py --download-only
```

## 산출물

- `src/chapter1_full_report.ipynb`: 실행 결과가 내장된 한국어 학습 노트북
- `src/reports/chapter1_report.md`: 공식 데이터·수식·책 비교·한계 리포트
- `src/reports/metrics.json`: 해시와 자동 검증을 포함한 기계 판독 결과
- `src/reports/figures/`: 재생성되는 도표
- `original_matlab/`: 체크섬으로 검증한 공식 원본 소스와 그림

노트북은 CLI와 같은 계산 함수를 쓰되, 도표는 실행 중 직접 만들어 셀 출력에
내장한다. 따라서 미리 생성된 PNG가 없어도 노트북 내용 자체는 재실행 가능하다.
장 전체 coverage 표는 코드가 없는 주제, 역사적 제품 정보, 실행한 실험을
구분하며, 수치 결론은 `metrics.json`의 자동 검증과 연결한다.

전체 저장소에는 `.codex/skills/create-chan-chapter-analysis/` 품질 감사 도구가
함께 버전관리된다. 전체 실행은 이 도구가 있을 때 `quality_benchmark.md`도
갱신한다. 책 디렉터리만 독립 복사해 감사 도구가 없으면 경고 후 벤치마크
갱신만 건너뛰며, 핵심 리포트·metrics·노트북·그림 생성은 실패하지 않는다.
`--skip-notebook`은 노트북과 품질 벤치마크를 모두 갱신하지 않아 서로 다른
실행 시점의 계측과 수치를 한 문서에 섞지 않는다.
"""
    write_text_if_changed(CHAPTER_DIR / "README.md", content)


def audit_current_notebook() -> dict[str, Any] | None:
    """Run the repository quality rubric so benchmark evidence cannot drift."""
    if not AUDIT_SCRIPT_PATH.exists():
        warnings.warn(
            "Repository quality audit is unavailable; core chapter artifacts were "
            f"generated but quality_benchmark.md was not refreshed: {AUDIT_SCRIPT_PATH}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    spec = importlib.util.spec_from_file_location(
        "chan_chapter_quality_audit", AUDIT_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audit script: {AUDIT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_notebook(NOTEBOOK_PATH)


def write_quality_benchmark(metrics: dict[str, Any]) -> bool:
    """Regenerate benchmark claims from the executed notebook and metrics."""
    audit = audit_current_notebook()
    if audit is None:
        return False
    counts = audit["counts"]
    scores = audit["scores"]
    verification = metrics["verification"]["summary"]
    theory = metrics["portfolio_theory"]
    quality_report = f"""# Chapter 1 품질 벤치마크

## 평가 목적과 재생성 계약

현재 Ch1을 다른 Chan 책의 실행 노트북과 같은 정적 rubric 및 수동 수치 검토로
비교한다. 이 파일은 수동으로 고치지 않는다. `run_chapter1_analysis.py`가 실행된
노트북, `metrics.json`, 아래 감사 스크립트에서 매번 재생성한다.

- 감사 스크립트: `{AUDIT_SCRIPT_PATH.relative_to(PROJECT_ROOT.parents[1])}`
- 감사 스크립트 SHA-256: `{sha256_file(AUDIT_SCRIPT_PATH)}`
- 실행 셀·그림·문자 수: 현재 `chapter1_full_report.ipynb`에서 계산
- 수치 검증 수: 현재 `metrics.json`에서 계산

## 비교군 정적 감사

| 책 / Chapter | 실행 | 재현성 | 엄밀성 | 교육 | 트레이딩 맥락 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Algorithmic Trading 2013 Ch3 | 20 | 5 | 5 | 17 | 12 | 59 |
| Algorithmic Trading 2013 Ch5 | 20 | 5 | 9 | 17 | 9 | 60 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch7 | 20 | 5 | 2 | 15 | 12 | 54 |
| **Machine Trading 2016 Ch1 — 현재** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

정적 점수는 품질의 하한이다. 글자 수나 헤딩을 늘리는 것으로 수치·결론 모순을
가릴 수 없으므로, 공식 MATLAB 결과·독립 솔버·표본운·편향도 별도로 검토한다.

## 현재 실행 산출물 계측

- 총 {counts['cells']}셀: Markdown {counts['markdown_cells']}, code {counts['code_cells']}.
- code {counts['executed_code_cells']}/{counts['code_cells']} 실행, 오류 출력 {counts['error_outputs']}.
- 실행 중 인라인 생성 PNG {counts['embedded_png_outputs']}개; 사전 생성 PNG 읽기 없음.
- Markdown {counts['markdown_characters']:,}자, 한국어 문자 {counts['korean_characters']:,}자, 헤딩 {counts['headings']}개.
- 데이터 {metrics['data']['price_observations']:,}행, 결측 {metrics['data']['missing_values']}, 비양수 {metrics['data']['nonpositive_values']}.
- 검증 {verification['total']}개 통과: 독립/경험 검증 {verification['independent_or_empirical']}개, 계약 invariant {verification['contract_invariant']}개.
- normalized `C^-1 M` gross exposure `{theory['box_1_3']['gross_exposure']:.4f}`.
- 전·후반 롱온리 tangency 비중 L1 거리 `{theory['split_sample_stability']['weight_l1_distance']:.4f}`.

## 반복 개선 근거

초기 구현도 공식 해시·잠금 환경·책 비중 비교와 15개 검증을 갖췄지만, 장 전체
성과지표·운영 맥락과 검증 강도 분류가 부족했다. 현재 버전은 CAGR·scalar Kelly,
200시드 분포, global minimum variance, 제외 자산 결측 근거, 검증 8+7 분류,
공통 재현 계층과 pytest 교차검증을 추가했다.

## 판정

현재 Ch1은 실행·재현성·엄밀성·교육·트레이딩 맥락에서 {audit['total']}/100이다.
진짜 성과 백테스트가 없는 것은 1장 공식 예제의 범위이며, 노트북은 정적
포트폴리오 결과를 표본 외 성과로 과장하지 않는다. 이 문서의 계측값은 파이프라인
재실행 때 함께 갱신되므로 노트북과 한 리비전 어긋나지 않는다.
"""
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", quality_report)
    return True


def build_and_execute_notebook() -> None:
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# Machine Trading — Chapter 1 재현 실험

Ernest P. Chan의 *Machine Trading* 1장 전체를 학습용으로 정리하고, 공식
동반 자료가 있는 Box 1.1·1.2와 수식으로 제시된 Box 1.3을 실행한다.

> **범위:** 데이터·플랫폼·브로커 설명은 2016년 책의 역사적 맥락과 재사용할
> 원칙을 구분한다. 포트폴리오 계산은 정적 수치 실험이며 시간 순서가 있는
> 매매 성과 **백테스트는 아니다**.

확인할 질문은 다섯 가지다.

1. 연구와 실거래가 같은 전략 로직을 써야 하는 이유는 무엇인가?
2. 데이터와 성과지표에서 어떤 편향과 tail risk를 확인해야 하는가?
3. 기간을 잘게 나눌수록 `평균 로그수익률 ≈ 평균 순수익률 - 분산/2`가
   정확해지는가?
4. 공식 ETF 데이터에서 롱온리 효율적 투자선과 tangency 포트폴리오가
   책의 결과와 일치하는가?
5. Box 1.3의 `C⁻¹M` 해와 표본 분할은 최적 비중의 어떤 위험을 드러내는가?"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Chapter coverage / 구현 범위와 재현 상태

| 장의 주제 | 재현 상태 | 노트북 증거 |
|---|---|---|
| 알고리즘 트레이딩 흐름 | 개념 설명 | 연구→백테스트/실거래 시스템 도표 |
| 과거 시장 데이터 | 역사적 맥락 | 조정·상장폐지·경매가·선물 롤 체크리스트 |
| 실시간 데이터·플랫폼·브로커 | 역사적 맥락 | 지연·API·체결 상태·운영 리스크 설명 |
| 성과지표 | 실행 수식 예제+개념 | CAGR, Sharpe, MDD/duration, Calmar/MAR, scalar Kelly |
| Box 1.1 | 근사 재현·책 비교 | PCG64 시뮬레이션과 MATLAB 표본 표 |
| Box 1.2 | 수치 재현·책 비교 | 공식 ETF, 효율적 투자선, 비중 허용오차 |
| Box 1.3 | 수식 실행 진단 | 정규화 `C⁻¹M`과 롱온리 해 비교 |
| 추정 위험·risk parity | 실행 진단+개념 | 전·후반 표본 비중과 gross exposure |
| 연습문제·주석 | 범위 밖 | 원문/한국어 장 문서에 보존 |

코드가 없는 주제를 조용히 빼지 않고 개념 설명 또는 역사적 맥락으로 표시했다.
연습문제 풀이를 본 실험에 억지로 포함하지 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 1. 하나의 전략 코어: 연구에서 주문까지

책의 Figure 1.1에서 가장 오래 남는 설계 원칙은 백테스트와 실거래가 같은
모델을 써야 한다는 것이다. 데이터 어댑터와 체결 어댑터는 달라도 신호,
포지션 상태 전이, 위험 제한은 공유해야 한다. 그래야 연구 코드와 실거래 코드의
미세한 차이 때문에 생기는 배포 오류를 줄일 수 있다.

아래 도표에서 historical data는 모의 체결·성과지표로, live data는 broker
API 주문으로 이어진다. 체결, 거부, 부분 체결과 현재 포지션은 다시 공유 상태로
돌아와야 한다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 2. 데이터·플랫폼·브로커 체크리스트

### 과거·실시간 데이터

- 주식/ETF는 분할·배당 조정뿐 아니라 상장폐지 종목과 당시 지수 구성종목이
  있어야 survivorship bias를 줄일 수 있다.
- 일별 시가·종가는 합성 체결가보다 주 거래소 경매가나 BBO 중간값이 실행
  가능성을 더 잘 반영할 수 있다.
- 연속선물은 롤 규칙 자체가 결과를 바꾸며 미래 롤 정보를 쓰면 look-ahead
  bias가 된다.
- 비유동 옵션은 마지막 체결가보다 bid/ask와 보조 정보가 중요하다.
- 실시간 데이터는 가격뿐 아니라 타임스탬프, 지연, 누락·중복을 검증해야 한다.

책이 열거한 CSI, Quandl, CRSP, Bloomberg 및 당시 플랫폼은 **2016년의 역사적
맥락**이다. 이 노트북은 현재 공급자 추천으로 갱신하지 않고, 공급자가 바뀌어도
유효한 데이터 품질 질문만 사용한다.

### 플랫폼과 브로커

연구 생산성과 운영 안정성은 다르다. 실거래에서는 주문 거부, 부분 체결,
재접속, 중복 주문, 마진, 계좌 보호 범위까지 다뤄야 한다. 전략의 높은 Sharpe가
이 운영 리스크를 상쇄하지는 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 3. 성과지표: 평균뿐 아니라 tail risk

초기 자산 $V_0$, 마지막 자산 $V_T$, 연수 $Y$일 때

$$\mathrm{CAGR}=\left(\frac{V_T}{V_0}\right)^{1/Y}-1.$$

일 1%를 252일 복리화한 CAGR과 단순 연환산 252%는 전혀 다르다. 책의
레버리지 1 비교 원칙을 지키려면 일별 P&L을 전일 gross market value로
나눈 수익률을 써야 한다.

일별 초과수익률 $R_t-r_f$의 연환산 Sharpe는

$$SR=\sqrt{252}\frac{\overline{R-r_f}}{\sigma(R-r_f)}$$

로 쓸 수 있다. 최대 낙폭은 누적 자산 $V_t$와 이전 고점 $H_t$에서

$$DD_t=\frac{V_t}{H_t}-1,\qquad MDD=\min_t DD_t$$

로 구하며, drawdown duration은 고점을 회복하지 못한 기간이다. 책에서 Calmar는
CAGR/최근 36개월 MDD, MAR은 inception 이후 CAGR/inception 이후 MDD다. 긴
백테스트일수록 MDD가 커져 MAR가 표본 길이에 민감하다는 것이 책이 Calmar를
선호한 이유다.

스칼라 Kelly 최적 레버리지는

$$f^*=\frac{\operatorname{E}[R-R_f]}{\operatorname{Var}(R-R_f)}$$

이지만 fat tail과 추정오차를 반영하지 않는다. 20% 급락에 5배 노출이면 비용
전 자산이 초기 1.0에서 0.0이 되므로 leverage cap, stress test, margin 규칙이 필요하다.
변동성은 Gaussian한 흔들림을 요약할 뿐 tail risk와 유동성 위험을 모두
보여주지 않으므로 Sharpe 하나로 전략을 평가하면 안 된다.

이 장의 ETF 최적화는 시간 순서가 있는 백테스트가 아니므로 APR·MDD를 계산해
성과처럼 꾸미지 않는다. 대신 제약 충족, 책 수치, 표본 민감도를 검증한다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 4. 재현 환경과 실행 방식

환경은 프로젝트의 `pyproject.toml`과 `uv.lock`으로 고정한다. 원시 데이터가
없을 때만 공식 ZIP을 내려받고, 이후에는 파일 SHA-256을 검증한다. 아래 셀은
미리 만들어 둔 그림을 읽지 않는다. 모든 도표는 현재 실행에서 계산해 PNG
바이트로 셀 출력에 내장된다."""
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
    raise RuntimeError("pyproject.toml이 있는 프로젝트 루트를 찾지 못했습니다")

PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "chapter_1_the_basics_of_algorithmic_trading/src"
sys.path.insert(0, str(SRC_DIR))

from run_chapter1_analysis import (
    BOOK_MINIMUM_VARIANCE_WEIGHTS,
    BOOK_NET_LOG_RESULTS,
    BOOK_TANGENCY_WEIGHTS,
    CHAPTER_COVERAGE,
    DATA_PATH,
    PYPROJECT_PATH,
    UV_LOCK_PATH,
    VERIFICATION_CLASSES,
    calculate_efficient_frontier,
    calculate_net_returns,
    calculate_performance_metric_examples,
    calculate_portfolio_theory_diagnostics,
    chapter_manifest,
    create_data_diagnostics_figure,
    create_efficient_frontier_figure,
    create_net_log_figure,
    create_portfolio_theory_figure,
    create_trading_system_figure,
    download_official_assets,
    environment_versions,
    excluded_universe_diagnostics,
    load_etf_data,
    sha256_file,
    simulate_net_vs_log_returns,
    solve_global_long_only_minimum_variance,
    summarize_net_log_seed_distribution,
    validate_offline_assets,
    verify_results,
    verification_summary,
)

if not DATA_PATH.exists():
    download_official_assets()
validate_offline_assets()

def show_figure(fig):
    # 현재 실행에서 만든 그림을 노트북 출력에 직접 내장한다.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buffer.getvalue()))

print(f"project: {PROJECT_ROOT}")
print("official assets: checksum verified")"""
        ),
        nbformat.v4.new_code_cell(
            """show_figure(create_trading_system_figure())"""
        ),
        nbformat.v4.new_code_cell(
            """performance_example = pd.Series(
    calculate_performance_metric_examples(), name="computed value"
).to_frame()
display(performance_example)
assert performance_example.loc["cagr", "computed value"] > 11
assert performance_example.loc["remaining_wealth_after_shock", "computed value"] == 0
print("CAGR 복리 예제와 5배 tail-shock 경고 계산 통과")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 5. 공식 출처와 환경 provenance

공식 배포 URL, 원본 ZIP 해시, 실제 입력 MAT 해시, 잠금 파일 해시를 함께
남긴다. 같은 파일과 같은 환경을 썼는지 나중에 확인할 수 있는 최소 조건이다."""
        ),
        nbformat.v4.new_code_cell(
            """manifest = chapter_manifest()
data_member = next(
    member for member in manifest["members"] if member["kind"] == "research_data"
)
provenance = pd.DataFrame(
    {
        "value": {
            "source_page": manifest["source_page"],
            "archive_url": manifest["url"],
            "archive_sha256": manifest["sha256"],
            "input_member": data_member["archive_path"],
            "input_sha256": sha256_file(DATA_PATH),
            "pyproject_sha256": sha256_file(PYPROJECT_PATH),
            "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
        }
    }
)
display(provenance)
display(pd.Series(environment_versions(), name="version").to_frame())"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 6. 데이터 구조와 기본 진단

가격 $P_t$에서 일별 순수익률은 다음과 같다.

$$R_t = \frac{P_t}{P_{t-1}} - 1$$

책의 의도대로 EWZ와 FXI를 제외하고 EWC, EWG, EWJ, EWQ, EWU, EWY를 쓴다.
공식 `ef.m`은 심볼 배열을 먼저 줄인 뒤 종가 배열을 필터링하는 순서 문제가
있다. 하지만 공식 심볼 순서에서 EWZ·FXI가 마지막 두 열이어서 원본도 우연히
같은 6개 열을 선택한다. 여기서는 원래 8개 심볼에서 마스크를 한 번 계산한다.
EWZ와 FXI의 결측 수를 따로 공개해 데이터 가용성 제외와 사후 선택을 구분한다."""
        ),
        nbformat.v4.new_code_cell(
            """etf_data = load_etf_data(DATA_PATH)
prices = pd.DataFrame(etf_data.close, index=etf_data.dates, columns=etf_data.symbols)
returns = pd.DataFrame(
    calculate_net_returns(etf_data.close),
    index=etf_data.dates[1:],
    columns=etf_data.symbols,
)

data_diagnostics = pd.Series(
    {
        "start": prices.index.min().date(),
        "end": prices.index.max().date(),
        "rows": len(prices),
        "assets": prices.shape[1],
        "missing_values": int(prices.isna().sum().sum()),
        "nonpositive_values": int((prices <= 0).sum().sum()),
        "minimum_price": float(prices.min().min()),
        "maximum_price": float(prices.max().max()),
    },
    name="value",
)
display(data_diagnostics.to_frame())
display(pd.DataFrame(excluded_universe_diagnostics()).T)
display(prices.head())
display(prices.describe().T[["min", "mean", "std", "max"]])"""
        ),
        nbformat.v4.new_code_cell(
            """annualized_statistics = pd.DataFrame(
    {
        "annualized_mean_net_return": returns.mean() * 252,
        "annualized_volatility": returns.std(ddof=1) * np.sqrt(252),
        "minimum_daily_return": returns.min(),
        "maximum_daily_return": returns.max(),
    }
)
display(annualized_statistics)
show_figure(create_data_diagnostics_figure(etf_data))"""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 7. 수식에서 코드로

| 개념 | 수식·제약 | 구현 |
|---|---|---|
| 순수익률 | $R_t=P_t/P_{t-1}-1$ | `calculate_net_returns` |
| Box 1.1 변환 | $R=e^r-1$ | `simulate_net_vs_log_returns` |
| 연속시간 근사 | $\mu\approx m-s^2/2$ | 각 표본의 `net_mean_minus_half_variance` |
| 포트폴리오 수익 | $F^T M$ | `weights @ mean_returns` |
| 포트폴리오 분산 | $F^T C F$ | `weights @ covariance @ weights` |
| 제약 | $F^T1=1, F^TM=m_p, F\ge0$ | `_solve_long_only_qp`의 활성집합/KKT 풀이 |
| Tangency | $\arg\max(m_p/\sigma_p)$ | 21개 격자의 `argmax(sharpes)` |
| Box 1.3 | $F=C^{-1}M/(1^TC^{-1}M)$ | `calculate_normalized_kelly_weights` |
| 표본 민감도 | 전반/후반의 같은 최적화 | `calculate_portfolio_theory_diagnostics` |

책의 역사적 MATLAB `quadprog` 대신, 자산이 6개라는 점을 이용해 63개
비어 있지 않은 활성집합을 모두 조사한다. 이 선택은 경계 비중의 최적화기
허용오차 차이를 줄인다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 8. Box 1.1 — 순수익률 평균과 로그수익률 평균

로그수익률이 정규분포일 때

$$\mu \approx m-\frac{s^2}{2}$$

를 확인한다. `N`이 커질수록 각 하위 기간의 평균은 `1/N`, 표준편차는
`2/sqrt(N)`으로 작아진다. 책은 MATLAB `rng(1)`, Python은 명시적으로
NumPy PCG64 seed 1을 사용하므로 개별 표본값이 아니라 근사의 수렴 성질을
비교한다. 같은 seed에서 세 `N`은 같은 난수열의 prefix이므로 서로 독립 표본은
아니다. seed 1의 표본운을 숨기지 않기 위해 200개 seed 분포도 계산한다."""
        ),
        nbformat.v4.new_code_cell(
            """net_log = simulate_net_vs_log_returns()
seed_summary = summarize_net_log_seed_distribution()
net_log_table = pd.DataFrame([item.__dict__ for item in net_log])
seed_summary_table = pd.DataFrame([item.__dict__ for item in seed_summary])

book_rows = []
for sample_size, book in BOOK_NET_LOG_RESULTS.items():
    book_rows.append(
        {
            "sample_size": sample_size,
            "book_log_mean": book["log_mean"],
            "book_approximation": book["approximation"],
            "book_identity_error": abs(book["log_mean"] - book["approximation"]),
        }
    )
comparison = net_log_table.merge(pd.DataFrame(book_rows), on="sample_size")
display(net_log_table)
display(seed_summary_table)
display(
    comparison[
        [
            "sample_size",
            "log_mean",
            "book_log_mean",
            "net_mean_minus_half_variance",
            "book_approximation",
            "absolute_approximation_error",
            "book_identity_error",
        ]
    ]
)
show_figure(create_net_log_figure(net_log, seed_summary))"""
        ),
        nbformat.v4.new_markdown_cell(
            """### Box 1.1 해석

위 표의 `seed_one_percentile`이 seed 1의 대표성을 직접 보여준다. 특히 작은
`N`에서 seed 1은 200개 중 극단적으로 오차가 작은 표본이므로, 그 한 점을
Python 구현의 우월성처럼 읽으면 안 된다. 책과 Python의 `μ` 값 차이는 난수
생성기 차이다. 판단 근거는 200시드 중앙값과 p05–p95 밴드가 `N` 증가에 따라
함께 감소하는지 여부다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 9. Box 1.2 — 롱온리 효율적 투자선

각 목표 일수익률 $m_p$에서 다음 문제를 푼다.

$$\min_F F^TCF \quad \text{s.t.}\quad F^TM=m_p,\;F^T1=1,\;F\ge0$$

21개 목표의 해 중 `m_p / 표준편차`가 최대인 점을 tangency, 분산이 최소인
점을 최소분산 포트폴리오로 표시한다. 무위험수익률은 책과 같이 0으로 둔다."""
        ),
        nbformat.v4.new_code_cell(
            """frontier = calculate_efficient_frontier(etf_data)
tangency = frontier.tangency_index
minimum = frontier.minimum_variance_index

frontier_summary = pd.DataFrame(
    {
        "daily_return": [frontier.targets[tangency], frontier.targets[minimum]],
        "daily_volatility": [
            frontier.standard_deviations[tangency],
            frontier.standard_deviations[minimum],
        ],
        "daily_sharpe_rf0": [frontier.sharpes[tangency], frontier.sharpes[minimum]],
    },
    index=["tangency", "minimum_variance"],
)
display(frontier_summary)

weight_comparison = pd.DataFrame(
    {
        "python_tangency": frontier.tangency_weights,
        "book_tangency": BOOK_TANGENCY_WEIGHTS,
        "python_minimum_variance": frontier.minimum_variance_weights,
        "book_minimum_variance": BOOK_MINIMUM_VARIANCE_WEIGHTS,
    },
    index=frontier.symbols,
)
weight_comparison["abs_diff_tangency"] = abs(
    weight_comparison["python_tangency"] - weight_comparison["book_tangency"]
)
weight_comparison["abs_diff_minimum"] = abs(
    weight_comparison["python_minimum_variance"]
    - weight_comparison["book_minimum_variance"]
)
display(weight_comparison)

book_min_variance = float(
    BOOK_MINIMUM_VARIANCE_WEIGHTS
    @ frontier.covariance
    @ BOOK_MINIMUM_VARIANCE_WEIGHTS
)
global_min_weights, global_min_variance = solve_global_long_only_minimum_variance(
    frontier.covariance
)
display(
    pd.Series(
        {
            "max_abs_tangency_weight_diff": weight_comparison["abs_diff_tangency"].max(),
            "python_minimum_variance": frontier.variances[minimum],
            "book_weights_variance_same_data": book_min_variance,
            "true_global_minimum_variance": global_min_variance,
            "grid_minus_global_variance": (
                frontier.variances[minimum] - global_min_variance
            ),
        },
        name="comparison",
    ).to_frame()
)
show_figure(create_efficient_frontier_figure(frontier))"""
        ),
        nbformat.v4.new_markdown_cell(
            """### Box 1.2 해석

위 비교표의 최대 절대 차이가 허용오차 안이면 tangency 비중은 수치상
재현된다. 최소분산 비중은 더 다르지만 Python 해의 목적함수(분산)가 책의
출력 비중보다 낮다. 역사적 `quadprog`가 0 근처 자산에 작은 비중을 남긴
허용오차 차이이며, Python의 활성집합 해는 같은 목표수익률·완전 투자·롱온리
제약을 모두 만족한다. 21개 목표 격자의 최소와 목표 제약이 없는 진짜 global
minimum도 위 표에서 구분한다.

투자 관점에서 더 중요한 점은 평균수익률 추정이 불안정하다는 것이다.
Tangency 포트폴리오는 평균의 작은 변화에도 크게 움직일 수 있지만,
최소분산 포트폴리오는 상대적으로 추정이 쉬운 공분산에 더 의존한다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 10. Box 1.3 — Sharpe/Kelly 해와 추정 위험

공매도 제약이 없고 무위험수익률을 0으로 두면 Kelly 성장 최적화와 Sharpe
최대화 해는 $C^{-1}M$에 비례한다. 순비중을 1로 정규화하면

$$F_{analytic}=\frac{C^{-1}M}{1^TC^{-1}M}.$$

Box 1.2는 $F\ge0$을 강제하지만 Box 1.3의 분석해는 음수 비중과 큰 gross
exposure를 허용한다. 또 전체 기간의 앞·뒤 절반에서 각각 같은 롱온리 문제를
풀어 평균수익률 추정에 대한 민감도를 진단한다. 이 두 절반은 각각 자기 표본을
최적화하므로 표본 외 백테스트가 아니다."""
        ),
        nbformat.v4.new_code_cell(
            """theory = calculate_portfolio_theory_diagnostics(etf_data, frontier)
box13_weights = pd.DataFrame(
    {
        "long_only_full": frontier.tangency_weights,
        "normalized_C_inv_M": theory.unconstrained_weights,
        "long_only_first_half": theory.first_half_weights,
        "long_only_second_half": theory.second_half_weights,
    },
    index=frontier.symbols,
)
display(box13_weights)
display(
    pd.Series(
        {
            "long_only_daily_sharpe_rf0": frontier.sharpes[tangency],
            "unconstrained_daily_sharpe_rf0": theory.unconstrained_daily_sharpe,
            "unconstrained_gross_exposure": theory.unconstrained_gross_exposure,
            "split_date": theory.split_date.date(),
            "first_half_daily_sharpe_rf0": theory.first_half_daily_sharpe,
            "second_half_daily_sharpe_rf0": theory.second_half_daily_sharpe,
            "first_vs_second_weight_L1": theory.split_weight_l1_distance,
        },
        name="diagnostic",
    ).to_frame()
)
show_figure(create_portfolio_theory_figure(frontier.symbols, frontier, theory))"""
        ),
        nbformat.v4.new_markdown_cell(
            """### Box 1.3 해석과 risk parity

분석해는 허용된 공매도와 레버리지로 표본 Sharpe를 높이지만 net exposure 1에
비해 gross exposure가 매우 크다. 전반과 후반의 롱온리 tangency 비중도 거의
겹치지 않는다. 이는 기대수익률 점추정치가 조금만 바뀌어도 최적해가 급변한다는
책의 경고를 수치로 확인한 것이다.

최소분산은 불안정한 평균 추정에 덜 의존한다. Risk parity는 또 다른 대안이지만
단순 역변동성 비중은 자산 간 상관관계와 tail risk를 직접 통제하지 않는다.
책은 등레버리지·risk parity에서 위기 손실과 청산이 다른 구성요소로 전염될
수 있다고 경고하고, 그 대안으로 각 구성요소의 MDD를 같은 목표로 두는 방식을
제시한다. 이 노트북의 추가 경고는 MDD 타깃도 추정오차와 상관 급변에서 자유롭지
않다는 점이다. 2015년 All Weather의 약 7% 손실은 risk parity의 tail-risk
한계를 보여주는 역사적 예시다. 어떤 방식이든 표본 외 안정성, turnover,
차입·거래비용을 별도로 검증해야 한다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 11. 이 결과를 실전 성과로 해석하면 안 되는 이유

- **Look-ahead:** 전체 2005–2015 표본으로 평균·공분산과 비중을 동시에
  구했다. 미래 성과가 아니다.
- **선택·생존 편향:** 책이 정한 ETF 집합을 사후에 사용하고 EWZ·FXI를
  제외한다.
- **가격 메타데이터:** 공식 파일의 가격 조정 여부를 별도 공급자 데이터로
  독립 검증하지 않았다.
- **비용 누락:** 수수료, bid-ask spread, 슬리피지, 세금, 리밸런싱 회전율을
  계산하지 않았다.
- **모형 가정:** 무위험수익률 0, 과거 평균·공분산이 대표적이라는 가정을 쓴다.
- **표본외 검증 없음:** 실전 확장 시 롤링 창으로 추정하고 다음 기간에만
  비중을 적용한 뒤 비용을 차감해야 한다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 12. 실행 검증

마지막으로 데이터 해시, 결측·양수 조건, Box 1.1 수렴, 모든 효율적 투자선
비중·목표수익률 제약, 책의 tangency 허용오차, 최소분산 목적함수를 한 번에
검사한다."""
        ),
        nbformat.v4.new_code_cell(
            """checks = verify_results(etf_data, seed_summary, frontier, theory)
verification = pd.DataFrame(
    {
        "passed": checks,
        "class": {name: VERIFICATION_CLASSES[name] for name in checks},
    }
)
display(verification)
assert verification["passed"].all()
summary = verification_summary(checks)
display(pd.Series(summary).drop("classification").to_frame("count"))
print(
    f"{summary['total']}개 통과: 독립/경험 검증 "
    f"{summary['independent_or_empirical']}개, 계약 invariant "
    f"{summary['contract_invariant']}개"
)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 결론

1장의 운영 원칙은 시장 데이터부터 주문·체결 상태까지 동일한 전략 코어와
검증 가능한 데이터 계보를 유지하는 것이다. 성과는 Sharpe 하나가 아니라 MDD,
duration, 비용과 운영 리스크를 함께 봐야 한다.

수치적으로 Box 1.1의 연속시간 근사, Box 1.2의 책 tangency 비중, Box 1.3의
`C⁻¹M` 일차조건을 재현했다. 동시에 분석해의 큰 gross exposure와 전·후반의
서로 다른 tangency 비중은 표본 내 최고 Sharpe가 실전 최적해라는 믿음을
반박한다.

따라서 이 노트북은 장 전체의 개념과 공식 예제를 연결한 **수학·수치 검증**이지
거래 전략의 수익성 증거가 아니다. 진짜 백테스트에는 시점별 추정, 표본 외
적용, 리밸런싱 규칙, 거래비용과 체결 모델이 필요하다."""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch1-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"Chapter 1 notebook cell {index}")
    execute_notebook(
        notebook,
        NOTEBOOK_PATH,
        workdir=CHAPTER_DIR / "src",
        timeout=600,
    )


def run_analysis(execute_notebook: bool = True) -> dict[str, Any]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    net_log_results = simulate_net_vs_log_returns()
    seed_summary = summarize_net_log_seed_distribution()
    data = load_etf_data()
    frontier = calculate_efficient_frontier(data)
    theory = calculate_portfolio_theory_diagnostics(data, frontier)
    checks = verify_results(data, seed_summary, frontier, theory)
    metrics = build_metrics(
        data, net_log_results, seed_summary, frontier, theory, checks
    )

    plot_trading_system(FIGURE_DIR / "trading_system_workflow.png")
    plot_data_diagnostics(data, FIGURE_DIR / "data_diagnostics.png")
    plot_net_log_results(
        net_log_results,
        seed_summary,
        FIGURE_DIR / "net_vs_log_returns.png",
    )
    plot_efficient_frontier(frontier, FIGURE_DIR / "efficient_frontier.png")
    plot_portfolio_theory(
        data.symbols,
        frontier,
        theory,
        FIGURE_DIR / "portfolio_theory_diagnostics.png",
    )
    write_text_if_changed(
        REPORT_DIR / "metrics.json",
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
    )
    write_report(data, net_log_results, seed_summary, frontier, theory, metrics)
    write_chapter_readme()
    if execute_notebook:
        build_and_execute_notebook()
        write_quality_benchmark(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not access the network; verify already-downloaded assets instead",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace downloaded assets whose checksums differ from the manifest",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Verify and extract the official archive without running experiments",
    )
    parser.add_argument(
        "--skip-notebook",
        action="store_true",
        help="Run calculations and reports without executing the notebook",
    )
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

    metrics = run_analysis(execute_notebook=not args.skip_notebook)
    tangency = metrics["efficient_frontier"]["tangency"]
    minimum = metrics["efficient_frontier"]["minimum_variance"]
    print("Chapter 1 analysis complete")
    print(f"Tangency weights: {tangency['weights']}")
    print(f"Minimum-variance weights: {minimum['weights']}")
    print(f"Report: {REPORT_DIR / 'chapter1_report.md'}")
    if not args.skip_notebook:
        print(f"Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
