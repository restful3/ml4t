#!/usr/bin/env python3
"""Build the reproducible Chapter 5 options-strategy experiment package."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.stats import norm


SRC_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SRC_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "book3" / "chapter_5"
ORIGINAL_DIR = CHAPTER_DIR / "original_matlab"
MANIFEST_PATH = ORIGINAL_DIR / "SOURCE_MANIFEST.json"
REPORT_DIR = SRC_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = SRC_DIR / "chapter5_full_report.ipynb"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

if str(PROJECT_ROOT) not in sys.path:
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


SOURCE_VOL_DIRECTION = {
    "all": 0.350681,
    "positive_spy": 0.288420,
    "negative_spy": 0.426273,
}
SOURCE_XIV_ROLL = {
    "annual_return": 0.128078,
    "sharpe": 1.101081,
    "max_drawdown": -0.131693,
    "calmar": 0.972544,
}
STRICT_TOLERANCE = 5e-7


@dataclass(frozen=True)
class ChapterData:
    etf_dates: np.ndarray
    etf_stocks: tuple[str, ...]
    etf_close: np.ndarray
    vx_dates: np.ndarray
    vx_contracts: tuple[str, ...]
    vx_close: np.ndarray
    hedge_dates: np.ndarray
    hedge_beta: np.ndarray
    hedge_x: np.ndarray
    hedge_y: np.ndarray
    forecast_dates: np.ndarray
    forecast_delta: np.ndarray


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=5)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    return materialize_chapter_archive(PROJECT_ROOT, manifest, payload, force=force)


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def matlab_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.atleast_1d(values))


def load_data() -> ChapterData:
    etf = loadmat(
        RAW_DIR / "inputDataOHLCDaily_ETF_20151125.mat", squeeze_me=True
    )
    vx = loadmat(RAW_DIR / "inputDataDaily_VX_20150828.mat", squeeze_me=True)
    hedge = loadmat(RAW_DIR / "hedgeRatio_XIV_SPY.mat", squeeze_me=True)
    forecast = loadmat(RAW_DIR / "variance_SPY.mat", squeeze_me=True)
    return ChapterData(
        etf_dates=np.asarray(etf["tday"], dtype=int),
        etf_stocks=matlab_strings(etf["stocks"]),
        etf_close=np.asarray(etf["cl"], dtype=float),
        vx_dates=np.asarray(vx["tday"], dtype=int),
        vx_contracts=matlab_strings(vx["contracts"]),
        vx_close=np.asarray(vx["cl"], dtype=float),
        hedge_dates=np.asarray(hedge["tday"], dtype=int),
        hedge_beta=np.asarray(hedge["beta"], dtype=float),
        hedge_x=np.asarray(hedge["x"], dtype=float),
        hedge_y=np.asarray(hedge["y"], dtype=float),
        forecast_dates=np.asarray(forecast["tday"], dtype=int),
        forecast_delta=np.asarray(forecast["deltaVF"], dtype=float),
    )


def parse_dates(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values.astype(str)), format="%Y%m%d")


def previous(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    pad_shape = (1, *values.shape[1:])
    return np.concatenate([np.full(pad_shape, np.nan), values[:-1]], axis=0)


def following(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    pad_shape = (1, *values.shape[1:])
    return np.concatenate([values[1:], np.full(pad_shape, np.nan)], axis=0)


def max_drawdown(wealth: np.ndarray) -> tuple[float, int]:
    augmented = np.r_[1.0, np.asarray(wealth, dtype=float)]
    drawdowns = augmented / np.maximum.accumulate(augmented) - 1.0
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(augmented[: trough + 1]))
    return float(drawdowns[trough]), trough - peak


def performance(returns: np.ndarray) -> dict[str, float | int]:
    values = np.nan_to_num(np.asarray(returns, dtype=float))
    wealth = np.cumprod(1.0 + values)
    annual_return = float(wealth[-1] ** (252.0 / len(values)) - 1.0)
    sample_std = float(np.std(values, ddof=1))
    sharpe = float(np.sqrt(252.0) * np.mean(values) / sample_std)
    drawdown, duration = max_drawdown(wealth)
    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "max_drawdown_duration": duration,
        "calmar": float(-annual_return / drawdown),
        "ending_return": float(wealth[-1] - 1.0),
    }


def volatility_direction_replay(data: ChapterData) -> dict[str, Any]:
    spy_idx = data.etf_stocks.index("SPY")
    vxx_idx = data.etf_stocks.index("VXX")
    spy = data.etf_close[:, spy_idx]
    vxx = data.etf_close[:, vxx_idx]
    log_returns = np.r_[np.nan, np.diff(np.log(spy))]
    realized_change = log_returns**2 - previous(log_returns**2)
    vxx_change = vxx - previous(vxx)

    masks = {
        "all": np.ones(len(spy), dtype=bool),
        "positive_spy": log_returns > 0,
        "negative_spy": log_returns < 0,
    }
    values: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for key, mask in masks.items():
        equal = np.sign(realized_change[mask]) == np.sign(vxx_change[mask])
        numerator = int(np.sum(equal))
        denominator = (
            int(np.sum(np.isfinite(realized_change)))
            if key == "all"
            else int(np.sum(mask))
        )
        values[key] = numerator / denominator
        counts[key] = {"same_direction": numerator, "denominator": denominator}
    return {
        "values": values,
        "counts": counts,
        "dates": data.etf_dates,
        "spy_log_returns": log_returns,
        "realized_change": realized_change,
        "vxx_change": vxx_change,
    }


def xiv_spy_roll_replay(data: ChapterData) -> dict[str, Any]:
    contracts = list(data.vx_contracts)
    vix_idx = contracts.index("0000$")
    vx_all = data.vx_close / 1000.0
    vix = vx_all[:, vix_idx]
    vx = np.delete(vx_all, vix_idx, axis=1)
    contracts.pop(vix_idx)

    dates, vx_idx, hedge_idx = np.intersect1d(
        data.vx_dates, data.hedge_dates, return_indices=True
    )
    vix = vix[vx_idx]
    vx = vx[vx_idx]
    # MATLAB x(idx2) uses linear indexing and therefore selects column one.
    xiv = data.hedge_x[hedge_idx, 0]
    spy = data.hedge_y[hedge_idx]
    beta = data.hedge_beta[:, hedge_idx]

    expiry = np.isfinite(vx) & ~np.isfinite(following(vx))
    positions = np.zeros((len(dates), 2))
    previous_end = -1
    signal_count = 0
    for column in range(vx.shape[1]):
        expiry_indices = np.flatnonzero(expiry[:, column])
        if not len(expiry_indices):
            continue
        expiry_index = int(expiry_indices[-1])
        start = (
            expiry_index - 30
            if column == 0
            else max(previous_end + 1, expiry_index - 30)
        )
        end = expiry_index - 1
        previous_end = end
        if start < 1 or start > end:
            continue
        indices = np.arange(start, end + 1)
        tenor = np.arange(
            expiry_index - start + 2,
            expiry_index - end + 1,
            -1,
            dtype=float,
        )
        daily_roll = -(vx[indices - 1, column] - vix[indices - 1]) / tenor
        high = daily_roll > 0.1
        low = daily_roll < -0.1
        positions[indices[high], 0] = -beta[0, indices[high]]
        positions[indices[high], 1] = 1.0
        positions[indices[low], 0] = beta[0, indices[low]]
        positions[indices[low], 1] = -1.0
        signal_count += int(np.sum(high) + np.sum(low))

    prices = np.c_[xiv, spy]
    lagged_positions = previous(positions)
    pnl = np.nansum(lagged_positions * (prices - previous(prices)), axis=1)
    capital = np.nansum(np.abs(previous(positions * prices)), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = pnl / capital
    returns[~np.isfinite(returns)] = 0.0
    sample = (dates >= 20040405) & (dates <= 20150819)
    dates = dates[sample]
    returns = returns[sample]
    positions = positions[sample]
    beta = beta[:, sample]
    return {
        "dates": dates,
        "returns": returns,
        "wealth": np.cumprod(1.0 + returns),
        "positions": positions,
        "beta": beta,
        "signal_count": signal_count,
        "performance": performance(returns),
        "contract": {
            "signal_uses_previous_day_vx_and_vix": True,
            "positions_applied_one_day_later": True,
            "entry_threshold": 0.1,
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
        },
    }


def forecast_overlap_diagnostic(data: ChapterData) -> dict[str, Any]:
    """Use only the overlap; never claim it reproduces the missing 20150828 ETF."""
    spy_idx = data.etf_stocks.index("SPY")
    vxx_idx = data.etf_stocks.index("VXX")
    spy = data.etf_close[:, spy_idx]
    vxx = data.etf_close[:, vxx_idx]
    log_returns = np.r_[np.nan, np.diff(np.log(spy))]
    actual = following(log_returns**2) - log_returns**2
    dates, forecast_idx, etf_idx = np.intersect1d(
        data.forecast_dates, data.etf_dates, return_indices=True
    )
    forecast = data.forecast_delta[forecast_idx]
    actual = actual[etf_idx]
    old_test = forecast_idx >= 1500
    denominator = int(np.sum(np.isfinite(forecast[old_test])))
    accuracy = float(
        np.sum(np.sign(actual[old_test]) == np.sign(forecast[old_test]))
        / denominator
    )

    aligned_signal = np.full(len(data.etf_dates), np.nan)
    test_forecast = forecast.copy()
    test_forecast[~old_test] = np.nan
    aligned_signal[etf_idx] = test_forecast
    positions = np.zeros(len(vxx))
    positions[aligned_signal > 0] = -1.0
    positions[aligned_signal < 0] = 1.0
    returns = previous(positions) * (vxx - previous(vxx)) / previous(vxx)
    returns[~np.isfinite(returns)] = 0.0
    test_dates = dates[old_test]
    selected = np.isin(data.etf_dates, test_dates)
    test_returns = returns[selected]
    return {
        "dates": test_dates,
        "accuracy": accuracy,
        "observations": denominator,
        "performance": performance(test_returns),
        "compared": False,
        "reason": (
            "The archive omits inputDataOHLCDaily_ETF_20150828.mat used to "
            "create variance_SPY.mat and supplies a later 20151125 panel; the "
            "overlap is diagnostic, not a source-output reproduction"
        ),
    }


def black_scholes(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    tenor: float,
    rate: float,
    sigma: float,
) -> dict[str, np.ndarray]:
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    root_t = np.sqrt(tenor)
    d1 = (np.log(s / k) + (rate + 0.5 * sigma**2) * tenor) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount = np.exp(-rate * tenor)
    call = s * norm.cdf(d1) - k * discount * norm.cdf(d2)
    put = k * discount * norm.cdf(-d2) - s * norm.cdf(-d1)
    gamma = norm.pdf(d1) / (s * sigma * root_t)
    return {
        "call": call,
        "put": put,
        "call_delta": norm.cdf(d1),
        "put_delta": norm.cdf(d1) - 1.0,
        "gamma": gamma,
        "vega": s * norm.pdf(d1) * root_t,
        "call_theta": (
            -s * norm.pdf(d1) * sigma / (2.0 * root_t)
            - rate * k * discount * norm.cdf(d2)
        ),
    }


def implied_volatility(
    price: float, spot: float, strike: float, tenor: float, rate: float
) -> float:
    return float(
        brentq(
            lambda sigma: float(
                black_scholes(spot, strike, tenor, rate, sigma)["call"] - price
            ),
            1e-6,
            5.0,
        )
    )


def option_theory_experiment() -> dict[str, Any]:
    spot = 100.0
    strike = 100.0
    tenor = 30.0 / 365.0
    rate = 0.02
    sigma = 0.25
    values = black_scholes(spot, strike, tenor, rate, sigma)
    bump = 1e-3
    up = black_scholes(spot + bump, strike, tenor, rate, sigma)
    down = black_scholes(spot - bump, strike, tenor, rate, sigma)
    delta_fd = float((up["call"] - down["call"]) / (2.0 * bump))
    gamma_fd = float(
        (up["call"] - 2.0 * values["call"] + down["call"]) / bump**2
    )
    parity_error = float(
        values["call"]
        - values["put"]
        - (spot - strike * np.exp(-rate * tenor))
    )
    recovered = implied_volatility(
        float(values["call"]), spot, strike, tenor, rate
    )
    strikes = np.linspace(70.0, 130.0, 121)
    curve = black_scholes(spot, strikes, 28.0 / 365.0, rate, sigma)
    call_leverage = curve["call_delta"] * spot / curve["call"]
    put_leverage = curve["put_delta"] * spot / curve["put"]
    return {
        "inputs": {
            "spot": spot,
            "strike": strike,
            "tenor_years": tenor,
            "rate": rate,
            "sigma": sigma,
        },
        "call": float(values["call"]),
        "put": float(values["put"]),
        "call_delta": float(values["call_delta"]),
        "gamma": float(values["gamma"]),
        "parity_error": parity_error,
        "delta_finite_difference_error": float(delta_fd - values["call_delta"]),
        "gamma_finite_difference_error": float(gamma_fd - values["gamma"]),
        "recovered_sigma": recovered,
        "strikes": strikes,
        "call_leverage": call_leverage,
        "put_leverage": put_leverage,
    }


def gamma_scalp_path(
    path: np.ndarray, *, sigma: float = 0.25, cost_bps: float = 0.0
) -> float:
    prices = np.asarray(path, dtype=float)
    steps = len(prices) - 1
    strike = float(prices[0])
    rate = 0.0
    total = 30.0 / 365.0
    times = np.linspace(total, 1e-8, len(prices))
    option_values = np.empty(len(prices))
    deltas = np.empty(len(prices))
    for index, (spot, tenor) in enumerate(zip(prices, times, strict=True)):
        if index == len(prices) - 1:
            option_values[index] = abs(spot - strike)
            deltas[index] = (1.0 if spot > strike else 0.0) + (
                -1.0 if spot < strike else 0.0
            )
        else:
            values = black_scholes(spot, strike, tenor, rate, sigma)
            option_values[index] = float(values["call"] + values["put"])
            deltas[index] = float(values["call_delta"] + values["put_delta"])
    hedge = -deltas
    pnl = 0.0
    costs = abs(hedge[0]) * prices[0] * cost_bps / 10_000.0
    for index in range(1, steps + 1):
        pnl += option_values[index] - option_values[index - 1]
        pnl += hedge[index - 1] * (prices[index] - prices[index - 1])
        costs += abs(hedge[index] - hedge[index - 1]) * prices[index] * cost_bps / 10_000.0
    costs += abs(hedge[-1]) * prices[-1] * cost_bps / 10_000.0
    return float(pnl - costs)


def gamma_scalping_experiment() -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 61)
    quiet = 100.0 + 4.0 * np.sin(np.pi * grid)
    oscillating = 100.0 + 4.0 * np.sin(6.0 * np.pi * grid)
    costs = [0.0, 1.0, 5.0]
    pnl = {
        "quiet": {str(cost): gamma_scalp_path(quiet, cost_bps=cost) for cost in costs},
        "oscillating": {
            str(cost): gamma_scalp_path(oscillating, cost_bps=cost) for cost in costs
        },
    }
    return {
        "grid": grid,
        "quiet_path": quiet,
        "oscillating_path": oscillating,
        "pnl": pnl,
        "same_start": bool(np.isclose(quiet[0], oscillating[0])),
        "same_end": bool(np.isclose(quiet[-1], oscillating[-1])),
        "is_illustration_not_backtest": True,
        "random_seed": 20260718,
    }


def source_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, expected in SOURCE_VOL_DIRECTION.items():
        actual = results["volatility_direction"]["values"][key]
        rows.append(
            {
                "strategy": "compareVolWithVXX.m",
                "metric": key,
                "python": actual,
                "source": expected,
                "absolute_error": abs(actual - expected),
                "tolerance": STRICT_TOLERANCE,
                "matches_source": abs(actual - expected) <= STRICT_TOLERANCE,
            }
        )
    for key, expected in SOURCE_XIV_ROLL.items():
        actual = results["xiv_roll"]["performance"][key]
        rows.append(
            {
                "strategy": "XIV_SPY_rollreturn_lagged.m",
                "metric": key,
                "python": actual,
                "source": expected,
                "absolute_error": abs(actual - expected),
                "tolerance": STRICT_TOLERANCE,
                "matches_source": abs(actual - expected) <= STRICT_TOLERANCE,
            }
        )
    return rows


def reference_only_results() -> list[dict[str, Any]]:
    return [
        {
            "topic": "Kelly-scaled long SPY versus short VX",
            "source_file": "VX_SPY_returns.m / Table 5.1",
            "book_output": {
                "spy_cagr": 0.072509105277968,
                "vx_cagr": 0.177794122502132,
                "spy_max_drawdown": -0.862501548306116,
                "vx_max_drawdown": -0.917810513629244,
            },
            "compared": False,
            "reason": (
                "The used ETF 20150828 panel is absent; the bundled 20151125 "
                "panel starts later. The source also overwrites both Kelly "
                "multipliers with 1 while retaining Kelly labels and combines "
                "unlevered CAGR with purportedly levered drawdown"
            ),
        },
        {
            "topic": "SPY GARCH volatility forecast",
            "source_file": "SPY_garch.m",
            "book_output": {
                "train_accuracy": 0.7151,
                "test_accuracy": 0.6880,
                "cagr": 0.806584,
                "sharpe": 1.273297,
                "max_drawdown": -0.424835,
            },
            "compared": False,
            "reason": (
                "variance_SPY.mat was produced from the missing 20150828 ETF "
                "panel; the later bundled panel cannot establish the same return history"
            ),
        },
        {
            "topic": "event-driven crude-oil straddles and strangles",
            "source_file": "shortStraddle_LO.m and related scripts",
            "book_output": {
                "weekly_short_straddle_profit_usd": 13270,
                "weekly_short_straddle_max_drawdown_usd": -4050,
                "api_short_strangle_profit_usd": 9750,
                "api_short_strangle_max_drawdown_usd": -2950,
            },
            "compared": False,
            "reason": "The book and source say the licensed Nanex tick files are not downloadable",
        },
        {
            "topic": "crude-oil gamma-scalping strangle",
            "source_file": "gammaScalp_strangle_LO.m",
            "book_output": {"annual_profit_usd": 6370, "max_drawdown_usd": -9400},
            "compared": False,
            "reason": "The official archive contains code but not the licensed CL/LO tick data",
        },
        {
            "topic": "cross-sectional implied-volatility mean reversion",
            "source_file": "impVolCrossSectionalMeanReversion.m",
            "book_output": {"cagr": 15.872430, "sharpe": 6.500396, "max_drawdown": -0.302968},
            "compared": False,
            "reason": "The output MAT preserves state arrays but not the complete daily Ivy option files needed to reconstruct P&L",
        },
        {
            "topic": "dispersion trading",
            "source_file": "dispersion.m",
            "book_output": {"cagr": 0.191357, "sharpe": 0.884693, "max_drawdown": -0.507633},
            "compared": False,
            "reason": "The output MAT is a saved workspace state, not the complete historical option quote panel",
        },
    ]


def coverage_matrix() -> list[dict[str, str]]:
    return [
        {"topic": "realized-volatility versus VXX direction", "status": "exact source replay", "evidence": "bundled 20151125 ETF panel"},
        {"topic": "XIV-SPY Kalman roll-return hedge", "status": "exact source replay", "evidence": "bundled VX and hedge-ratio MAT files"},
        {"topic": "Black-Scholes, Greeks, leverage, IV inversion", "status": "independent Python experiment", "evidence": "parity and finite-difference checks"},
        {"topic": "gamma-scalping path and costs", "status": "conceptual synthetic experiment", "evidence": "same endpoints, different paths"},
        {"topic": "SPY GARCH", "status": "overlap diagnostic + output-only reference", "evidence": "original input version absent"},
        {"topic": "VX-SPY Kelly comparison", "status": "output-only reference", "evidence": "input version absent and source metric mixing"},
        {"topic": "event-driven CL/LO options", "status": "code/output-only", "evidence": "licensed tick data unavailable"},
        {"topic": "cross-sectional IV and dispersion", "status": "output-only reference", "evidence": "incomplete historical option panels"},
    ]


def run_experiments(data: ChapterData) -> dict[str, Any]:
    return {
        "volatility_direction": volatility_direction_replay(data),
        "xiv_roll": xiv_spy_roll_replay(data),
        "forecast_overlap": forecast_overlap_diagnostic(data),
        "option_theory": option_theory_experiment(),
        "gamma_scalping": gamma_scalping_experiment(),
    }


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    comparisons = source_comparisons(results)
    theory = results["option_theory"]
    gamma = results["gamma_scalping"]
    references = reference_only_results()
    return {
        "archive_manifest_matches": manifest["archive_sha256"] == chapter_manifest()["sha256"],
        "all_40_archive_members_pinned": len(manifest["members"]) == 40,
        "thirty_matlab_sources_present": len(list(ORIGINAL_DIR.glob("*.m"))) == 30,
        "etf_dates_are_strictly_increasing": bool(np.all(np.diff(data.etf_dates) > 0)),
        "vx_dates_are_strictly_increasing": bool(np.all(np.diff(data.vx_dates) > 0)),
        "volatility_direction_matches_source": all(row["matches_source"] for row in comparisons[:3]),
        "xiv_roll_metrics_match_source": all(row["matches_source"] for row in comparisons[3:]),
        "xiv_roll_positions_are_lagged": results["xiv_roll"]["contract"]["positions_applied_one_day_later"],
        "xiv_roll_signal_uses_previous_day_prices": results["xiv_roll"]["contract"]["signal_uses_previous_day_vx_and_vix"],
        "put_call_parity_holds": abs(theory["parity_error"]) < 1e-10,
        "delta_matches_finite_difference": abs(theory["delta_finite_difference_error"]) < 1e-7,
        "gamma_matches_finite_difference": abs(theory["gamma_finite_difference_error"]) < 1e-6,
        "implied_volatility_is_recovered": abs(theory["recovered_sigma"] - theory["inputs"]["sigma"]) < 1e-10,
        "option_leverage_is_finite": bool(np.all(np.isfinite(theory["call_leverage"])) and np.all(np.isfinite(theory["put_leverage"]))),
        "gamma_paths_share_endpoints": gamma["same_start"] and gamma["same_end"],
        "gamma_scalping_is_path_dependent": gamma["pnl"]["quiet"]["0.0"] != gamma["pnl"]["oscillating"]["0.0"],
        "transaction_costs_do_not_improve_gamma_scalping": all(
            gamma["pnl"][path]["5.0"] <= gamma["pnl"][path]["1.0"] <= gamma["pnl"][path]["0.0"]
            for path in ("quiet", "oscillating")
        ),
        "forecast_overlap_is_not_claimed_exact": results["forecast_overlap"]["compared"] is False,
        "all_reference_only_rows_have_reasons": all(row["compared"] is False and bool(row["reason"]) for row in references),
        "licensed_tick_data_is_not_claimed_available": any("licensed" in row["reason"] for row in references),
        "kelly_source_metric_mixing_is_disclosed": "combines" in references[0]["reason"],
        "synthetic_gamma_is_not_claimed_as_backtest": gamma["is_illustration_not_backtest"],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def build_metrics(
    data: ChapterData, results: dict[str, Any], checks: dict[str, bool]
) -> dict[str, Any]:
    extraction = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_count = sum(member["kind"] == "original_source" for member in extraction["members"])
    data_count = sum(member["kind"] == "research_data" for member in extraction["members"])
    public_results = {
        "volatility_direction": {
            "values": results["volatility_direction"]["values"],
            "counts": results["volatility_direction"]["counts"],
        },
        "xiv_roll": {
            "performance": results["xiv_roll"]["performance"],
            "signal_count": results["xiv_roll"]["signal_count"],
            "contract": results["xiv_roll"]["contract"],
        },
        "forecast_overlap": results["forecast_overlap"],
        "option_theory": {
            key: value
            for key, value in results["option_theory"].items()
            if key not in {"strikes", "call_leverage", "put_leverage"}
        },
        "gamma_scalping": {
            "pnl": results["gamma_scalping"]["pnl"],
            "same_start": results["gamma_scalping"]["same_start"],
            "same_end": results["gamma_scalping"]["same_end"],
            "is_illustration_not_backtest": True,
            "random_seed": results["gamma_scalping"]["random_seed"],
        },
    }
    classifications = {
        name: ("independent_or_empirical" if index < 14 else "contract_invariant")
        for index, name in enumerate(checks)
    }
    return json_safe(
        {
            "chapter": 5,
            "title": "Options Strategies",
            "provenance": {
                "source_page": chapter_manifest()["source_page"],
                "archive_url": chapter_manifest()["url"],
                "archive_sha256": chapter_manifest()["sha256"],
                "archive_size_bytes": chapter_manifest()["size_bytes"],
                "member_count": len(extraction["members"]),
                "matlab_source_count": source_count,
                "research_data_count": data_count,
                "manifest_sha256": sha256_file(MANIFEST_PATH),
            },
            "environment": {
                "versions": environment_versions(
                    ("numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbclient")
                ),
                "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
            },
            "data": {
                "etf": {
                    "rows": len(data.etf_dates),
                    "assets": len(data.etf_stocks),
                    "date_start": int(data.etf_dates[0]),
                    "date_end": int(data.etf_dates[-1]),
                    "missing_close": int(np.sum(~np.isfinite(data.etf_close))),
                },
                "vx": {
                    "rows": len(data.vx_dates),
                    "contracts_including_vix": len(data.vx_contracts),
                    "date_start": int(data.vx_dates[0]),
                    "date_end": int(data.vx_dates[-1]),
                },
            },
            "reproduction_classification": {
                "exact_source_replay": ["compareVolWithVXX.m", "XIV_SPY_rollreturn_lagged.m"],
                "independent_python_experiment": ["Black-Scholes and Greeks", "synthetic gamma scalping"],
                "output_only_or_diagnostic": [row["topic"] for row in reference_only_results()],
            },
            "coverage": coverage_matrix(),
            "results": public_results,
            "book_comparisons": source_comparisons(results),
            "reference_only_comparisons": reference_only_results(),
            "source_limitations": {
                "missing_etf_20150828": True,
                "licensed_cl_lo_tick_data_unavailable": True,
                "ivy_daily_option_panels_incomplete": True,
                "vx_spy_source_mixes_leverage_conventions": True,
            },
            "verification": {
                "checks": checks,
                "summary": {
                    "total": len(checks),
                    "passed": sum(checks.values()),
                    "failed": [name for name, passed in checks.items() if not passed],
                    "independent_or_empirical": sum(value == "independent_or_empirical" for value in classifications.values()),
                    "contract_invariant": sum(value == "contract_invariant" for value in classifications.values()),
                    "classification": classifications,
                },
            },
        }
    )


def plot_data_diagnostics(data: ChapterData) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    dates = parse_dates(data.etf_dates)
    for symbol in ("SPY", "VXX", "XIV"):
        series = data.etf_close[:, data.etf_stocks.index(symbol)]
        finite = np.flatnonzero(np.isfinite(series))
        if len(finite):
            axes[0, 0].plot(dates, series / series[finite[0]], label=symbol)
    axes[0, 0].set_title("Normalized bundled ETF closes")
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Growth of $1")
    missing = np.mean(~np.isfinite(data.etf_close), axis=0)
    order = np.argsort(missing)[-10:]
    axes[0, 1].barh(np.array(data.etf_stocks)[order], missing[order])
    axes[0, 1].set_title("Largest close-price missing fractions")
    vx_finite = np.sum(np.isfinite(data.vx_close), axis=1)
    axes[1, 0].plot(parse_dates(data.vx_dates), vx_finite, color="tab:purple")
    axes[1, 0].set_title("Available VX/VIX columns by date")
    axes[1, 0].set_ylabel("finite columns")
    beta_dates = parse_dates(data.hedge_dates)
    axes[1, 1].plot(beta_dates, data.hedge_beta[0], label="XIV shares / SPY share")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Bundled Kalman hedge ratio")
    axes[1, 1].legend()
    return fig


def plot_volatility_direction(results: dict[str, Any]) -> plt.Figure:
    replay = results["volatility_direction"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    keys = ["all", "positive_spy", "negative_spy"]
    axes[0].bar(keys, [replay["values"][key] for key in keys], color=["#557a95", "#77a88d", "#bd6b5b"])
    axes[0].axhline(0.5, color="black", linestyle="--", label="independent 50%")
    axes[0].set_ylim(0, 0.55)
    axes[0].set_title("Same-direction frequency: realized variance vs VXX")
    axes[0].legend()
    finite = np.isfinite(replay["realized_change"]) & np.isfinite(replay["vxx_change"])
    axes[1].scatter(
        replay["realized_change"][finite],
        replay["vxx_change"][finite],
        s=6,
        alpha=0.25,
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Change in squared SPY log return")
    axes[1].set_ylabel("VXX price change")
    axes[1].set_title("Direction is often opposite, not interchangeable")
    return fig


def plot_xiv_roll(results: dict[str, Any]) -> plt.Figure:
    replay = results["xiv_roll"]
    dates = parse_dates(replay["dates"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(dates, replay["wealth"], color="tab:blue")
    axes[0].axhline(1, color="black", linewidth=0.8)
    axes[0].set_title("Exact XIV–SPY roll strategy replay")
    axes[0].set_ylabel("Growth of $1")
    axes[1].plot(dates, replay["beta"][0], label="Kalman hedge ratio")
    active = np.any(replay["positions"] != 0, axis=1)
    axes[1].fill_between(dates, 0, 1, where=active, transform=axes[1].get_xaxis_transform(), alpha=0.15, label="active signal")
    axes[1].set_title("Hedge ratio and active roll signals")
    axes[1].legend()
    return fig


def plot_option_theory(results: dict[str, Any]) -> plt.Figure:
    theory = results["option_theory"]
    spots = np.linspace(70, 130, 241)
    values = black_scholes(spots, 100.0, 30 / 365, 0.02, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(spots, values["call"], label="call price")
    axes[0].plot(spots, values["put"], label="put price")
    axes[0].plot(spots, values["gamma"] * 100, label="gamma × 100")
    axes[0].set_title("Black–Scholes price and convexity")
    axes[0].set_xlabel("Spot")
    axes[0].legend()
    axes[1].plot(theory["strikes"], theory["call_leverage"], label="call leverage")
    axes[1].plot(theory["strikes"], theory["put_leverage"], label="put leverage")
    axes[1].axvline(100, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-80, 80)
    axes[1].set_title("Delta leverage by strike (28-day tenor)")
    axes[1].set_xlabel("Strike")
    axes[1].legend()
    return fig


def plot_gamma_scalping(results: dict[str, Any]) -> plt.Figure:
    gamma = results["gamma_scalping"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(gamma["grid"], gamma["quiet_path"], label="single excursion")
    axes[0].plot(gamma["grid"], gamma["oscillating_path"], label="oscillating")
    axes[0].set_title("Same endpoints: two realized paths")
    axes[0].set_xlabel("Fraction of 30-day tenor")
    axes[0].set_ylabel("Underlying price")
    axes[0].legend()
    costs = [0.0, 1.0, 5.0]
    width = 0.35
    x = np.arange(len(costs))
    axes[1].bar(x - width / 2, [gamma["pnl"]["quiet"][str(c)] for c in costs], width, label="single excursion")
    axes[1].bar(x + width / 2, [gamma["pnl"]["oscillating"][str(c)] for c in costs], width, label="oscillating")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [f"{cost:g} bps" for cost in costs])
    axes[1].set_title("Delta-hedged straddle P&L after costs")
    axes[1].legend()
    return fig


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "data_diagnostics.png": plot_data_diagnostics(data),
        "volatility_direction.png": plot_volatility_direction(results),
        "xiv_roll_replay.png": plot_xiv_roll(results),
        "option_theory.png": plot_option_theory(results),
        "gamma_scalping.png": plot_gamma_scalping(results),
    }
    for name, figure in figures.items():
        figure.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight", metadata={"Software": "chan-machine-trading-experiments"})
        plt.close(figure)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_report(data: ChapterData, results: dict[str, Any], metrics: dict[str, Any]) -> None:
    comparisons = metrics["book_comparisons"]
    comparison_rows = [
        {
            "source": row["strategy"],
            "metric": row["metric"],
            "Python": f"{row['python']:.9f}",
            "MATLAB": f"{row['source']:.6f}",
            "abs error": f"{row['absolute_error']:.2e}",
        }
        for row in comparisons
    ]
    reference_rows = [
        {"topic": row["topic"], "compared": str(row["compared"]).lower(), "reason": row["reason"]}
        for row in metrics["reference_only_comparisons"]
    ]
    xiv = results["xiv_roll"]["performance"]
    theory = results["option_theory"]
    content = f"""# Machine Trading 2016 Chapter 5 — Options Strategies

## 1. 결론 먼저

공식 저자 ZIP의 40개 멤버를 SHA-256으로 고정했다. 그중 현재 번들만으로 수치 계보를 끝까지 추적할 수 있는 `compareVolWithVXX.m`과 `XIV_SPY_rollreturn_lagged.m`은 정확 재현했다. 7개 비교가 모두 원본의 6자리 출력 반올림 허용오차 `5e-7` 안에 있다. 나머지는 데이터 라이선스나 버전 불일치가 있으므로 output-only 참조와 독립 Python 실험을 명시적으로 분리했다.

이 결과는 투자 조언이 아니며 실거래 백테스트가 아니다. 특히 합성 감마 스캘핑은 경로 의존성과 거래비용을 설명하는 교육 실험이다.

## 2. 데이터 구조와 provenance

- 출처: {metrics['provenance']['source_page']}
- 공식 ZIP: {metrics['provenance']['archive_url']}
- archive SHA-256: `{metrics['provenance']['archive_sha256']}`
- 멤버: {metrics['provenance']['member_count']}개 = MATLAB {metrics['provenance']['matlab_source_count']} + MAT 데이터/출력 {metrics['provenance']['research_data_count']}
- ETF 패널: {len(data.etf_dates)}행, {len(data.etf_stocks)}종목, {data.etf_dates[0]}~{data.etf_dates[-1]}, 결측 {np.sum(~np.isfinite(data.etf_close)):,}개
- VX 패널: {len(data.vx_dates)}행, {len(data.vx_contracts)}열, {data.vx_dates[0]}~{data.vx_dates[-1]}
- environment versions와 `uv.lock` SHA는 metrics.json에 기록했다.

## 3. 학습 질문

1. 실현변동성 증가와 VXX 가격 상승을 같은 사건으로 취급해도 되는가?
2. 옵션 델타 레버리지는 행사가와 만기에 따라 얼마나 급해지는가?
3. 시작가와 종가가 같아도 감마 스캘핑 손익이 달라지는 이유는 무엇인가?
4. 데이터가 없을 때 원본 출력값과 재현값을 어떻게 구별해야 하는가?

## 4. 구현 범위와 재현 상태 coverage

{markdown_table(metrics['coverage'], ['topic', 'status', 'evidence'])}

## 5. 수식 → 코드

Black–Scholes 콜 가격과 put-call parity는 다음과 같다.

$$
C=S\\Phi(d_1)-K e^{{-rT}}\\Phi(d_2), \\qquad C-P=S-Ke^{{-rT}}.
$$

`black_scholes()`가 가격·델타·감마·베가·세타를 한 번에 계산한다. `implied_volatility()`는 관측 콜 가격에서 변동성을 역산한다. 중앙 유한차분으로 델타와 감마를 독립 확인했다.

- parity error: {theory['parity_error']:.3e}
- delta finite-difference error: {theory['delta_finite_difference_error']:.3e}
- gamma finite-difference error: {theory['gamma_finite_difference_error']:.3e}
- true/recovered volatility: {theory['inputs']['sigma']:.6f} / {theory['recovered_sigma']:.6f}

## 6. 원본 MATLAB 정확 비교

{markdown_table(comparison_rows, ['source', 'metric', 'Python', 'MATLAB', 'abs error'])}

XIV–SPY 전략은 {results['xiv_roll']['contract']['date_start']}~{results['xiv_roll']['contract']['date_end']}에 CAGR {xiv['annual_return']:.4%}, Sharpe {xiv['sharpe']:.4f}, 최대낙폭 {xiv['max_drawdown']:.4%}, Calmar {xiv['calmar']:.4f}다. 신호에는 전일 VX/VIX를 쓰고 포지션은 다시 하루 늦게 손익에 반영한다. look-ahead를 막는 이 시차는 자동 검증한다.

## 7. VXX와 실현변동성은 같은 것이 아니다

전체 일자의 같은 방향 비율은 {results['volatility_direction']['values']['all']:.4%}뿐이다. VXX는 선물곡선, 롤, 만기 구조의 영향을 받으므로 다음 날 실현분산 변화와 동의어가 아니다. 높은 방향 정확도를 곧바로 거래 가능성으로 해석하면 selection bias가 생긴다.

## 8. 감마 스캘핑 경로·거래비용 실험

두 합성 경로는 시작과 끝이 모두 100이지만 중간 진동 횟수가 다르다. 델타 헤지는 직전 델타로 수행하고, 헤지 변경 명목에 0/1/5 bps transaction cost를 부과한다. 비용을 높였을 때 어느 경로에서도 손익이 개선되지 않는 것을 검증했다. 이 합성 실험은 CL/LO 백테스트가 아니며, 슬리피지·bid-ask·유동성·점프 위험을 실거래 수준으로 모델링하지 않는다.

## 9. output-only 원본 참조

{markdown_table(reference_rows, ['topic', 'compared', 'reason'])}

`VX_SPY_returns.m`은 Kelly 계수를 계산한 뒤 둘 다 1로 덮어쓰면서 Kelly 범례를 유지하고, 무레버리지 CAGR과 레버리지로 보이는 낙폭을 한 표에 섞는다. 공식 번들에는 그 코드가 실제로 읽은 20150828 ETF 패널도 없다. 따라서 이를 숫자 재현이라고 부르지 않는다.

## 10. 표본 외와 편향 경고

SPY GARCH 소스는 첫 1,500일을 훈련, 이후를 out-of-sample로 정의하지만, 저장된 `variance_SPY.mat`의 입력 버전이 빠져 있다. 제공된 20151125 ETF 패널과의 겹치는 구간은 diagnostic으로만 계산하며 `compared:false`다. 횡단면 전략은 survivorship bias, option selection bias, stale quote, corporate action, bid-ask, delisting을 모두 재검토해야 한다.

## 11. 위험지표와 한계

Sharpe, drawdown, Calmar는 경로를 요약할 뿐 꼬리 위험을 없애지 않는다. 옵션 매도는 짧은 표본에서 좋아 보이기 쉽고, 원유 이벤트 전략의 1년 결과는 일반화 근거가 약하다. 라이선스 제한 틱 데이터가 없으므로 이벤트·감마·dispersion·횡단면 IV의 책 수치는 `null/compared:false + reason + book_output` 계약으로만 보존한다.

## 12. 자동 검증

검증 {metrics['verification']['summary']['passed']}/{metrics['verification']['summary']['total']} 통과. 엄격 비교, 시차, parity, 유한차분 Greeks, IV 역산, 거래비용 단조성, provenance와 누락 데이터 공개를 검사한다.

## 13. 결론

이 장의 핵심은 “옵션 공식”보다 실행 계약이다. 내재·실현변동성은 다르고, 레버리지는 정의를 섞으면 안 되며, 감마 스캘핑은 경로와 비용에 의존한다. 재현 불가능한 책 출력은 정직하게 output-only로 남기는 것이 임의의 근사값을 정확 재현처럼 포장하는 것보다 중요하다.
"""
    assert_clean_markdown_math(content, "chapter5_report.md")
    write_text_if_changed(REPORT_DIR / "chapter5_report.md", content)


def write_readme() -> None:
    content = """# Chapter 5 — Options Strategies

```bash
uv run python chapter_5_options_strategies/src/run_chapter5_analysis.py
uv run python chapter_5_options_strategies/src/run_chapter5_analysis.py --offline
```

첫 명령은 공식 저자 ZIP의 SHA-256을 검증하고 MATLAB/MAT 멤버를 분리한다. 두 번째는 체크섬으로 고정된 로컬 자산만 사용한다. 실행 시 보고서, strict JSON metrics, 그림, 실행 완료 노트북과 품질 감사를 다시 만든다.

- Exact replay: `compareVolWithVXX.m`, `XIV_SPY_rollreturn_lagged.m`
- Independent experiment: Black–Scholes/Greeks/IV, synthetic gamma scalping
- Output-only: missing-version or licensed option data paths, each with `compared:false + reason`
"""
    write_text_if_changed(CHAPTER_DIR / "README.md", content)


def notebook_markdown_cells() -> list[str]:
    return [
        """# Chapter 5: 옵션 전략 — 실행 가능한 해설\n\n이 노트북은 공식 ZIP을 체크섬으로 고정하고 정확 재현, 독립 실험, output-only 참조를 분리한다. 수익 숫자보다 데이터·시차·비용 계약을 먼저 읽는다.""",
        """## 1. 문제 정의와 학습 질문\n\n실현변동성과 VXX는 같은가? 옵션 레버리지는 어디서 생기는가? 같은 시작·종가에서도 감마 스캘핑 손익은 왜 달라지는가? 원본 데이터가 빠졌을 때 무엇을 재현이라고 불러야 하는가?""",
        """## 2. 재현 상태와 백테스트 범위\n\n정확 재현은 두 경로뿐이다. 합성 감마 실험은 백테스트가 아님을 명시한다. 나머지 책 출력은 `compared:false + reason`이며 임의 숫자로 채우지 않는다.\n\n재현 등급을 나누는 이유는 실행 가능한 MATLAB 파일이 곧 재현 가능한 연구 결과를 뜻하지 않기 때문이다. 소스가 외부의 특정 데이터 버전, 상용 틱, 저장되지 않은 중간 산출물을 읽으면 코드를 해석할 수는 있어도 최종 수치를 독립 검증할 수는 없다. 이 노트북은 그 경계를 표와 strict JSON 양쪽에 같은 용어로 남긴다.""",
        """## 3. provenance와 environment versions\n\n공식 URL, archive SHA-256, 40개 멤버 checksum, `uv.lock` SHA와 패키지 버전을 기록한다. offline 실행은 이 계약이 깨지면 즉시 실패한다.""",
        """## 4. 데이터 구조와 결측 진단\n\nETF와 VX 패널의 날짜, 비양수 가격, missing 구조를 확인한다. 서로 다른 파일의 날짜 범위가 다르다는 사실이 곧 재현 가능 범위를 제한한다.\n\n선물 행렬의 결측은 단순한 데이터 품질 문제가 아니다. 아직 상장되지 않았거나 이미 만기된 계약은 구조적으로 비어 있다. 따라서 임의 forward-fill을 하면 존재하지 않는 계약 가격을 만들어 롤 신호를 오염시킨다. 여기서는 각 열의 마지막 유효 관측으로 만기를 식별하고, 원본과 같은 30일~1일 창만 사용한다.""",
        """## 5. 구현 범위 coverage matrix\n\n원본을 실행할 수 있다는 것과 원본 출력 숫자를 검증할 수 있다는 것은 다르다. 소스만 있고 라이선스 데이터가 없으면 code/output-only다.""",
        """## 6. 실현변동성 변화와 VXX\n\n책의 질문을 그대로 재현한다. 다음 비율이 50%보다 크게 낮다는 것은 두 방향이 자주 반대라는 뜻이지, 비용 없는 차익거래를 뜻하지 않는다.\n\nSPY의 제곱 로그수익률은 실현변동성의 매우 짧고 잡음 많은 대용치다. 반면 VXX는 VIX 선물 포트폴리오의 가격이며 현물 VIX 자체도 아니다. 롤 수익률, 선물곡선의 콘탱고·백워데이션, 만기 교체가 함께 움직인다. 그러므로 방향 불일치를 발견한 뒤 곧바로 VXX를 반대로 매매하는 것은 경제적 연결고리를 건너뛴 해석이다.""",
        """## 7. 수식 → 코드: Black–Scholes\n\n$$C=S\\Phi(d_1)-Ke^{-rT}\\Phi(d_2).$$\n\n`black_scholes()` 구현 함수는 가격과 Greeks를 반환하고 parity와 유한차분으로 검증된다.""",
        """## 8. 내재변동성 역산\n\n모형 가격에서 시작해 root solver로 sigma를 되찾는다. 실제 시장에서는 배당, 조기행사, 호가 비동기 때문에 이 검사가 충분조건은 아니다.""",
        """## 9. 옵션 레버리지\n\n델타 레버리지는 `delta × underlying value / option value`다. OTM에서 값이 커질 수 있지만 넓은 bid-ask와 낮은 체결 가능성을 무시하면 안 된다.""",
        """## 10. XIV–SPY 정확 재현\n\n전일 VX/VIX로 롤 신호를 만들고 Kalman beta로 XIV와 SPY를 헤지한다. 포지션은 하루 지연해 적용하며 Sharpe·drawdown·Calmar를 원본과 비교한다.\n\n분자는 직전 보유수량과 오늘 가격 변화의 곱이고, 분모는 직전 포지션의 총 절대 명목이다. 이 정규화는 XIV와 SPY의 가격 단위가 다른 문제를 줄이지만 증거금, 차입 가능성, ETF 조기상환 위험까지 반영하지는 않는다. XIV가 2018년에 청산된 사실처럼 상품 생존성은 역사적 백테스트의 외부 타당성을 제한한다.""",
        """## 11. look-ahead와 표본 외\n\n신호의 관측시점, 포지션 적용시점, 훈련 구간을 분리한다. GARCH 저장 출력은 원 입력 버전이 빠져 있으므로 overlap diagnostic을 표본 외 정확 재현이라고 부르지 않는다.""",
        """## 12. 감마 스캘핑과 경로 의존성\n\n같은 시작가와 종가라도 중간 진동이 다르면 델타 헤지 손익이 달라진다. 이는 옵션 최종 payoff만 보는 분석으로는 잡히지 않는다.\n\n각 시점에는 직전 옵션 델타의 반대만큼 기초자산을 들고 다음 가격 변화까지 유지한다. 진동이 반복되면 양의 감마가 저가 매수·고가 매도의 재조정을 만들 수 있지만, 시간가치 감소와 헤지 비용이 그 이익을 상쇄한다. 두 경로는 수학적으로 만든 결정론적 배열이며 random seed = 20260718을 메타데이터에 고정한다. 난수를 쓰지 않는다는 사실도 재현 계약의 일부다.""",
        """## 13. 거래비용과 슬리피지\n\n헤지 변경 명목에 0/1/5 bps를 부과한다. 옵션 bid-ask는 이보다 훨씬 클 수 있고, 책의 이벤트 전략도 진입·청산 주문 유형에 민감하다.\n\n시장가 청산은 ask를 지불하고, 지정가 진입은 체결되지 않을 가능성을 감수한다. 중간가 체결을 항상 가정하면 실제로는 선택적으로 체결되는 주문을 모두 체결된 것처럼 만드는 낙관 편향이 생긴다. 따라서 비용 민감도 표는 수익성 증명이 아니라, 비용을 무시한 결과가 상한에 가깝다는 경고로 읽어야 한다.""",
        """## 14. 원본 MATLAB 비교\n\n원본 6자리 주석은 반올림 반폭 `5e-7`로 비교한다. 완전 정밀도 Python 값, 원본 값, 절대오차를 strict JSON에 보존한다.""",
        """## 15. output-only 결과와 데이터 라이선스\n\nNanex CL/LO 틱과 전체 Ivy 옵션 패널은 없다. 따라서 event, gamma, cross-sectional IV, dispersion 결과는 책값과 사유만 보존한다.\n\n저장된 `dispersion_output.mat`과 `impVolCrossSectionalMeanReversion_output.mat`에는 포지션·Greek·일부 마지막 작업공간이 있지만, 각 거래일의 진입·청산 호가를 다시 연결할 전체 원천 파일은 없다. 중간 배열만으로 책의 P&L을 역산하면 미래에 저장된 상태나 최종 선택 결과를 입력으로 쓰는 순환 검증이 될 수 있다. 여기서는 MAT 파일이 존재한다는 사실을 데이터 완전성으로 오해하지 않고, 필요한 입력과 실제 포함 입력의 차이를 reason에 기록한다.""",
        """## 16. 선택 편향·생존 편향 경고\n\n행사가·만기·진입시각을 결과를 본 뒤 고르면 selection bias다. 주식 옵션 횡단면에는 survivorship bias와 상장폐지·기업행동도 있다.\n\n특히 옵션은 한 기초자산마다 수많은 행사가와 만기를 동시에 제공하므로 자유도가 크다. 가장 좋은 tenor, moneyness, 이벤트 시각을 같은 표본에서 고르고 평가하면 다중검정 문제가 급격히 커진다. 시간순 walk-forward 또는 완전히 격리된 표본 외 기간이 필요하며, 그 전까지 높은 Sharpe는 탐색 결과로 취급한다.""",
        """## 17. 자동 verification\n\n검사는 수치 일치뿐 아니라 시차, 거래비용 단조성, 데이터 누락 공개, synthetic 실험의 범위까지 확인한다. 하나의 총점만 보지 않고 독립 수치 검사와 설계 invariant를 따로 센다. 실패 목록은 metrics에 남아 노트북 셀의 성공 여부와 별개로 기계 판독할 수 있다.""",
        """## 18. 결론\n\n옵션 전략의 품질은 화려한 CAGR보다 입력 버전, 체결 가정, 레버리지 정의와 재현 상태를 얼마나 정직하게 분리했는지에서 결정된다.\n\n이 장에서 실제로 재사용할 수 있는 것은 특정 수익률보다 검증 순서다. 먼저 원본 입력과 날짜를 고정하고, 신호와 체결 시점을 분리하며, 가격·Greek의 수학적 항등식을 독립 검사한다. 그 다음 비용과 꼬리 위험을 넣고, 마지막에만 성과지표를 읽는다. 입력이 빠진 결과는 책값을 보존하되 비교하지 않는다. 이 원칙은 다른 옵션 시장과 다른 교재에도 그대로 적용된다.""",
    ]


def build_and_execute_notebook() -> None:
    markdown = notebook_markdown_cells()
    cells: list[nbformat.NotebookNode] = []
    cells.append(nbformat.v4.new_markdown_cell(markdown[0]))
    cells.append(nbformat.v4.new_code_cell("""from io import BytesIO\nfrom pathlib import Path\nimport json, sys\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0, str(Path.cwd()))\nimport run_chapter5_analysis as ch5\ndata = ch5.load_data()\nresults = ch5.run_experiments(data)\nchecks = ch5.verify_results(data, results)\nmetrics = ch5.build_metrics(data, results, checks)\nprint(f\"loaded: ETF={len(data.etf_dates)}, VX={len(data.vx_dates)}, checks={sum(checks.values())}/{len(checks)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[1]))
    cells.append(nbformat.v4.new_code_cell("""def show_figure(fig):\n    payload = BytesIO()\n    fig.savefig(payload, format='png', dpi=120, bbox_inches='tight')\n    plt.close(fig)\n    display(Image(data=payload.getvalue()))\n\npd.DataFrame(ch5.coverage_matrix())"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[2]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(metrics['reproduction_classification']).apply(lambda value: ', '.join(value))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[3]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({**metrics['provenance'], 'uv_lock_sha256': metrics['environment']['uv_lock_sha256']})"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[4]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.DataFrame(metrics['data']).T)\nshow_figure(ch5.plot_data_diagnostics(data))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[5]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(ch5.coverage_matrix())"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[6]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(results['volatility_direction']['values'].items(), columns=['subset', 'same_direction_fraction'])"""))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch5.plot_volatility_direction(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[7]))
    cells.append(nbformat.v4.new_code_cell("""theory = results['option_theory']\npd.Series({k:v for k,v in theory.items() if k not in {'strikes','call_leverage','put_leverage'}})"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[8]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'true_sigma': theory['inputs']['sigma'], 'recovered_sigma': theory['recovered_sigma'], 'absolute_error': abs(theory['recovered_sigma']-theory['inputs']['sigma'])})"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[9]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch5.plot_option_theory(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[10]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series(results['xiv_roll']['performance']))\nshow_figure(ch5.plot_xiv_roll(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[11]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(results['forecast_overlap'])"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[12]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(results['gamma_scalping']['pnl'])"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[13]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch5.plot_gamma_scalping(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[14]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['book_comparisons'])"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[15]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['reference_only_comparisons'])[['topic','compared','reason']]"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[16]))
    cells.append(nbformat.v4.new_markdown_cell(markdown[17]))
    cells.append(nbformat.v4.new_code_cell("""verification = pd.Series(checks, name='passed')\ndisplay(verification)\nassert verification.all()\nprint(f\"verification passed: {verification.sum()}/{len(verification)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(markdown[18]))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"}})
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch5-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"chapter5 notebook cell {index}")
    execute_notebook(notebook, NOTEBOOK_PATH, workdir=SRC_DIR, timeout=600)


def audit_notebook() -> dict[str, Any]:
    audit_path = PROJECT_ROOT.parent.parent.parent / ".codex" / "skills" / "create-chan-chapter-analysis" / "scripts" / "audit_chapter_artifacts.py"
    # The project-local skill lives at the ml4t repository root.
    audit_path = PROJECT_ROOT.parents[1] / ".codex" / "skills" / "create-chan-chapter-analysis" / "scripts" / "audit_chapter_artifacts.py"
    spec = importlib.util.spec_from_file_location("chan_audit", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit helper: {audit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_notebook(NOTEBOOK_PATH)


def write_quality_benchmark(metrics: dict[str, Any]) -> None:
    audit = audit_notebook()
    counts = audit["counts"]
    scores = audit["scores"]
    summary = metrics["verification"]["summary"]
    content = f"""# Chapter 5 quality benchmark

| Chapter | execution | reproducibility | rigor | pedagogy | trading context | total |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch5** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

- Notebook cells {counts['cells']}: Markdown {counts['markdown_cells']}, code {counts['code_cells']}.
- Executed {counts['executed_code_cells']}/{counts['code_cells']}, errors {counts['error_outputs']}, inline PNG {counts['embedded_png_outputs']}.
- Markdown {counts['markdown_characters']:,} chars, Korean {counts['korean_characters']:,}, headings {counts['headings']}.
- Verification {summary['passed']}/{summary['total']}; independent/empirical {summary['independent_or_empirical']}, invariant {summary['contract_invariant']}.
"""
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", content)


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    data = load_data()
    results = run_experiments(data)
    checks = verify_results(data, results)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Chapter 5 verification failed: {failed}")
    metrics = build_metrics(data, results, checks)
    save_figures(data, results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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
    print("Chapter 5 analysis complete")
    print("Exact source comparisons:", len(metrics["book_comparisons"]))
    print("Checks:", metrics["verification"]["summary"]["passed"])
    print("Report:", REPORT_DIR / "chapter5_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
