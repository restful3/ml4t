#!/usr/bin/env python3
"""Reproduce and audit Machine Trading Chapter 3 time-series examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import toeplitz


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
RAW_DIR = PROJECT_ROOT / "data/raw/book3/chapter_3"
AUDUSD_PATH = RAW_DIR / "inputData_AUDUSD_20150807.mat"
SPX_CLOSE_PATH = RAW_DIR / "inputDataOHLCDaily_20120424.mat"
ETF_PATH = RAW_DIR / "inputData_ETF.mat"
BTC_DAILY_PATH = RAW_DIR / "Jonathan_BTCUSD_trades_daily.mat"
BTC_SWAPS_PATH = RAW_DIR / "lastSwapsBTC.csv"
REPORT_DIR = CHAPTER_DIR / "src/reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = CHAPTER_DIR / "src/chapter3_full_report.ipynb"
AUDIT_SCRIPT_PATH = (
    PROJECT_ROOT.parents[1]
    / ".codex/skills/create-chan-chapter-analysis/scripts/audit_chapter_artifacts.py"
)

AUD_PERIODS_PER_YEAR = 252 * (24 * 60 - 15)
HARDWARE_SYMBOLS = ("AAPL", "EMC", "HPQ", "NTAP", "SNDK")

BOOK_RESULTS = {
    "aud_ar1_phi": 0.999998,
    "btc_daily_ar1_phi": 0.989484,
    "ar10_optimal_lag": 10,
    "ar10_annual_return": 1.584085074291988,
    "arma25_annual_return": 0.602164545675666,
    "arima_log_price_order": [1, 1, 9],
    "var1_book_annual_return": 0.48,
    "var1_book_sharpe": 0.9,
    "ssm_hardware_train_cagr": 0.407758,
    "ssm_hardware_train_sharpe": 1.846934,
    "ssm_hardware_test_cagr": -0.003842,
    "ssm_hardware_test_sharpe": 0.046716,
}

AR10_BOOK_COEFFICIENTS = np.array(
    [
        1.37196e-06,
        0.993434,
        -0.00121205,
        -0.000352717,
        0.000753222,
        0.00662641,
        -0.00224118,
        -0.00305157,
        0.00351317,
        -0.00154844,
        0.00407798,
    ]
)
ARMA25_BOOK_COEFFICIENTS = {
    "constant": 2.80383e-06,
    "ar": np.array([0.649011, 0.350986]),
    "ma": np.array(
        [0.345806, -0.00906282, -0.0106082, -0.0102606, -0.00251154]
    ),
}
PAIR_B = np.array([[-0.01015, 0.02114], [0.40606, -0.32381]])
PAIR_D = -0.07687

CHAPTER_COVERAGE = (
    ("Why time-series models; currencies, bitcoin, stocks", "conceptual explanation", "chapter introduction"),
    ("AR(1), weak stationarity, random walk", "executed + source comparison", "AUD.USD and BTC conditional AR(1)"),
    ("Bid-ask bounce and midprice choice", "executed cost diagnostic", "AUD bid/ask market-order sensitivity"),
    ("AR(p), BIC, Table 3.1", "executed numerical reproduction", "p=10 and full-sample coefficients"),
    ("Figure 3.1 AR(10) strategy", "executed numerical reproduction + corrected variant", "source look-ahead and train-only split"),
    ("ARMA(p,q), Table 3.2, Figure 3.2", "published-parameter numerical replay", "ARMA(2,5) zero-innovation forecast"),
    ("ARIMA(p,d,q) and price/return equivalence", "formula verification + source-output comparison", "ARIMA(1,1,9)"),
    ("VAR(p), BIC, Tables 3.3–3.4", "executed methodological adaptation", "official close panel; CRSP/Compustat inputs absent"),
    ("Sector-neutral equation 3.4 and Figure 3.3", "executed methodological adaptation", "common close-data OOS window"),
    ("VEC(q), error-correction matrix, cointegration", "executed identity", "C=Phi-I for VAR(1)"),
    ("State-space equations 3.6–3.9", "executed published-parameter replay", "Kalman moving-average adaptation"),
    ("Tables 3.5 and Figures 3.4–3.5", "approximate adaptation", "different close period; no MATLAB MLE"),
    ("Dynamic EWA–EWC hedge, Tables 3.6, Figures 3.6–3.9", "executed published-parameter replay", "official ETF data and published B,D"),
    ("HP/wavelet alternatives and regime change", "conceptual explanation", "limitations and alternatives"),
    ("Exercises 3.1–3.8 and endnotes", "mapped; selected exercises executed", "stationarity, costs, VAR/VEC/Kalman extensions"),
    ("BTC/MXN/HYG/SPY supporting archive scripts", "source preserved; cross-chapter/supporting", "not claimed as Ch3 prose examples"),
)

VERIFICATION_CLASSES = {
    "archive_manifest_matches": "independent_or_empirical",
    "aud_ar1_phi_matches_source": "independent_or_empirical",
    "btc_ar1_phi_matches_source": "independent_or_empirical",
    "ar_bic_selects_book_lag_10": "independent_or_empirical",
    "ar10_full_fit_matches_book_output": "independent_or_empirical",
    "arma25_replay_matches_book_output": "independent_or_empirical",
    "var_close_bic_selects_lag_1": "independent_or_empirical",
    "ar10_coefficients_match_table_31": "independent_or_empirical",
    "source_ar10_lookahead_detected": "contract_invariant",
    "train_only_ar10_differs_from_source_fit": "contract_invariant",
    "market_order_costs_do_not_improve_ar10": "contract_invariant",
    "var_positions_are_sector_neutral": "contract_invariant",
    "vec_matrix_equals_var_phi_minus_identity": "contract_invariant",
    "kalman_pair_uses_published_noise_parameters": "contract_invariant",
    "splits_are_chronological": "contract_invariant",
}

@dataclass(frozen=True)
class Performance:
    annual_return: float
    sharpe: float | None
    maximum_drawdown: float
    drawdown_duration: int
    cumulative_return: float
    mean_period_return: float
    periods_per_year: int


@dataclass
class StrategyResult:
    name: str
    returns: np.ndarray
    positions: np.ndarray
    performance: Performance
    cost_performance: Performance | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChapterData:
    aud_mid: np.ndarray
    aud_bid: np.ndarray
    aud_ask: np.ndarray
    aud_tday: np.ndarray
    aud_hhmm: np.ndarray
    aud_train_end: int
    btc_close: np.ndarray
    spx_dates: pd.DatetimeIndex
    hardware_close: np.ndarray
    etf_dates: pd.DatetimeIndex
    ewa: np.ndarray
    ewc: np.ndarray


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=3)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    return materialize_chapter_archive(PROJECT_ROOT, manifest, payload, force=force)


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def decode_matlab_strings(values: np.ndarray) -> list[str]:
    return [str(value).strip() for value in np.asarray(values).squeeze()]


def load_chapter_data() -> ChapterData:
    aud = loadmat(AUDUSD_PATH, squeeze_me=True)
    aud_mid = np.asarray(aud["mid"], dtype=float)
    first = int(np.flatnonzero(np.isfinite(aud_mid))[0])
    aud_mid = aud_mid[first:]
    aud_bid = np.asarray(aud["bid"], dtype=float)[first:]
    aud_ask = np.asarray(aud["ask"], dtype=float)[first:]
    aud_tday = np.asarray(aud["tday"], dtype=int)[first:]
    aud_hhmm = np.asarray(aud["hhmm"], dtype=int)[first:]
    if len(aud_mid) <= AUD_PERIODS_PER_YEAR:
        raise ValueError("AUD.USD data is shorter than the one-year test window")
    if not np.all(np.isfinite(aud_mid)):
        raise ValueError("AUD.USD midprice contains interior missing values")

    btc = loadmat(BTC_DAILY_PATH, squeeze_me=True)
    btc_close = np.asarray(btc["cl"], dtype=float)

    spx = loadmat(SPX_CLOSE_PATH, squeeze_me=True)
    stocks = decode_matlab_strings(spx["stocks"])
    hardware_index = [stocks.index(symbol) for symbol in HARDWARE_SYMBOLS]
    hardware_close = np.asarray(spx["cl"], dtype=float)[:, hardware_index]
    if np.any(~np.isfinite(hardware_close)):
        raise ValueError("Selected official close series unexpectedly contain missing data")
    spx_dates = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(spx["tday"], dtype=int).astype(str), format="%Y%m%d")
    )

    etf = loadmat(ETF_PATH, squeeze_me=True)
    symbols = decode_matlab_strings(etf["syms"])
    closes = np.asarray(etf["cl"], dtype=float)
    etf_dates = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(etf["tday"], dtype=int).astype(str), format="%Y%m%d")
    )
    ewa = closes[:, symbols.index("EWA")]
    ewc = closes[:, symbols.index("EWC")]

    return ChapterData(
        aud_mid=aud_mid,
        aud_bid=aud_bid,
        aud_ask=aud_ask,
        aud_tday=aud_tday,
        aud_hhmm=aud_hhmm,
        aud_train_end=len(aud_mid) - AUD_PERIODS_PER_YEAR,
        btc_close=btc_close,
        spx_dates=spx_dates,
        hardware_close=hardware_close,
        etf_dates=etf_dates,
        ewa=ewa,
        ewc=ewc,
    )


def performance(returns: np.ndarray, periods_per_year: int) -> Performance:
    clean = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(wealth)
    drawdown = wealth / peaks - 1.0
    duration = 0
    longest = 0
    for value in drawdown:
        duration = duration + 1 if value < 0 else 0
        longest = max(longest, duration)
    standard_deviation = float(np.std(clean, ddof=1))
    return Performance(
        annual_return=float(wealth[-1] ** (periods_per_year / len(clean)) - 1.0),
        sharpe=(
            float(np.sqrt(periods_per_year) * np.mean(clean) / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        maximum_drawdown=float(np.min(drawdown)),
        drawdown_duration=longest,
        cumulative_return=float(wealth[-1] - 1.0),
        mean_period_return=float(np.mean(clean)),
        periods_per_year=periods_per_year,
    )


def fit_ar_conditional(prices: np.ndarray, lag: int, end: int) -> np.ndarray:
    """Fit intercept plus AR lags using sufficient statistics, without a huge design."""
    target_rows = np.arange(lag, end)
    target = prices[target_rows]
    lagged = [prices[target_rows - offset] for offset in range(1, lag + 1)]
    gram = np.empty((lag + 1, lag + 1))
    response = np.empty(lag + 1)
    gram[0, 0] = len(target_rows)
    response[0] = np.sum(target)
    for row, values in enumerate(lagged, start=1):
        gram[0, row] = gram[row, 0] = np.sum(values)
        response[row] = values @ target
    for row, left in enumerate(lagged, start=1):
        for column, right in enumerate(lagged, start=1):
            gram[row, column] = left @ right
    return np.linalg.solve(gram, response)


def ar_bic_yule_walker(prices: np.ndarray, max_lag: int = 60) -> tuple[int, np.ndarray]:
    centered = prices - np.mean(prices)
    observations = len(centered)
    autocovariance = np.array(
        [
            centered[lag:] @ centered[: observations - lag] / observations
            for lag in range(max_lag + 1)
        ]
    )
    bic = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        coefficients = np.linalg.solve(
            toeplitz(autocovariance[:lag]), autocovariance[1 : lag + 1]
        )
        variance = max(
            float(
                autocovariance[0]
                - coefficients @ autocovariance[1 : lag + 1]
            ),
            np.finfo(float).tiny,
        )
        bic[lag - 1] = observations * np.log(variance) + (lag + 1) * np.log(
            observations
        )
    return int(np.argmin(bic) + 1), bic


def forecast_ar(prices: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    lag = len(coefficients) - 1
    forecast = np.full(len(prices), np.nan)
    forecast[lag - 1 :] = coefficients[0]
    for offset, coefficient in enumerate(coefficients[1:]):
        forecast[lag - 1 :] += coefficient * prices[
            lag - 1 - offset : len(prices) - offset
        ]
    return forecast


def forecast_arma_zero_innovations(
    prices: np.ndarray, constant: float, ar: np.ndarray, ma: np.ndarray
) -> np.ndarray:
    """Replay MATLAB forecast(..., Y0=..., E0 omitted), so future MA shocks are zero."""
    lookback = max(len(ar), len(ma))
    forecast = np.full(len(prices), np.nan)
    forecast[lookback - 1 :] = constant
    for offset, coefficient in enumerate(ar):
        forecast[lookback - 1 :] += coefficient * prices[
            lookback - 1 - offset : len(prices) - offset
        ]
    return forecast


def trade_intraday_forecast(
    name: str,
    data: ChapterData,
    forecast: np.ndarray,
    metadata: dict[str, Any],
) -> StrategyResult:
    positions = np.zeros(len(data.aud_mid))
    test_rows = np.arange(data.aud_train_end, len(data.aud_mid))
    difference = forecast[test_rows] - data.aud_mid[test_rows]
    positions[test_rows[difference > 0]] = 1.0
    positions[test_rows[difference < 0]] = -1.0

    returns = np.zeros(len(data.aud_mid))
    returns[1:] = positions[:-1] * (
        data.aud_mid[1:] / data.aud_mid[:-1] - 1.0
    )
    half_spread = (data.aud_ask - data.aud_bid) / (2.0 * data.aud_mid)
    turnover = np.zeros(len(positions))
    turnover[1:] = np.abs(positions[1:] - positions[:-1])
    costs = np.zeros(len(positions))
    costs[1:] = turnover[:-1] * half_spread[:-1]
    test_returns = returns[data.aud_train_end :]
    test_net_returns = test_returns - costs[data.aud_train_end :]
    return StrategyResult(
        name=name,
        returns=test_returns,
        positions=positions,
        performance=performance(test_returns, AUD_PERIODS_PER_YEAR),
        cost_performance=performance(test_net_returns, AUD_PERIODS_PER_YEAR),
        metadata={
            **metadata,
            "one_way_execution_cost": "observed half-spread × position turnover",
            "test_bars": len(test_rows),
            "train_segment": "no positions; strategy trades only the final one-year test set",
        },
    )


def fit_var_coefficients(
    panel: np.ndarray, lag: int, end: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a VAR with an intercept and return coefficients, residuals, and BIC."""
    rows = np.arange(lag, end)
    design = np.column_stack(
        [np.ones(len(rows))]
        + [panel[rows - offset] for offset in range(1, lag + 1)]
    )
    target = panel[rows]
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    covariance = residuals.T @ residuals / len(residuals)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("VAR residual covariance is not positive definite")
    parameter_count = target.shape[1] * design.shape[1]
    bic = len(residuals) * log_determinant + parameter_count * np.log(
        len(residuals)
    )
    return coefficients, residuals, float(bic)


def forecast_var(panel: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    assets = panel.shape[1]
    lag = (coefficients.shape[0] - 1) // assets
    forecast = np.full_like(panel, np.nan)
    rows = np.arange(lag - 1, len(panel))
    design = np.column_stack(
        [np.ones(len(rows))]
        + [panel[rows - offset] for offset in range(lag)]
    )
    forecast[rows] = design @ coefficients
    return forecast


def normalized_sector_positions(forecast_returns: np.ndarray) -> np.ndarray:
    centered = forecast_returns - np.nanmean(forecast_returns, axis=1, keepdims=True)
    denominator = np.nansum(np.abs(centered), axis=1, keepdims=True)
    positions = np.divide(
        centered,
        denominator,
        out=np.zeros_like(centered),
        where=np.isfinite(denominator) & (denominator > 0),
    )
    positions[~np.isfinite(positions)] = 0.0
    return positions


def daily_panel_strategy(
    name: str,
    prices: np.ndarray,
    positions: np.ndarray,
    test_start: int,
    metadata: dict[str, Any],
) -> StrategyResult:
    returns = np.zeros(len(prices))
    returns[1:] = np.sum(
        positions[:-1] * (prices[1:] / prices[:-1] - 1.0), axis=1
    )
    turnover = np.zeros(len(prices))
    turnover[0] = np.sum(np.abs(positions[0]))
    turnover[1:] = np.sum(np.abs(positions[1:] - positions[:-1]), axis=1)
    costs = np.zeros(len(prices))
    costs[1:] = 0.0002 * turnover[:-1]
    test_returns = returns[test_start:]
    test_net_returns = test_returns - costs[test_start:]
    return StrategyResult(
        name=name,
        returns=returns,
        positions=positions,
        performance=performance(test_returns, 252),
        cost_performance=performance(test_net_returns, 252),
        metadata={
            **metadata,
            "test_start": test_start,
            "one_way_cost_sensitivity_bps": 2.0,
            "mean_daily_turnover_test": float(np.mean(turnover[test_start:])),
            "train_performance": asdict(performance(returns[:test_start], 252)),
        },
    )


def fit_var_adaptation(data: ChapterData) -> tuple[StrategyResult, dict[str, Any]]:
    train_end = len(data.hardware_close) - 252
    bic_values = []
    for lag in range(1, 6):
        _, _, bic = fit_var_coefficients(data.hardware_close, lag, train_end)
        bic_values.append(bic)
    selected_lag = int(np.argmin(bic_values) + 1)
    coefficients, _, _ = fit_var_coefficients(
        data.hardware_close, selected_lag, train_end
    )
    forecast = forecast_var(data.hardware_close, coefficients)
    forecast_returns = forecast / data.hardware_close - 1.0
    positions = normalized_sector_positions(forecast_returns)
    positions[:train_end] = 0.0
    result = daily_panel_strategy(
        "VAR close-data sector-neutral adaptation",
        data.hardware_close,
        positions,
        train_end,
        {
            "reproduction_grade": "methodological adaptation",
            "lag": selected_lag,
            "fit_end_date": data.spx_dates[train_end - 1].date().isoformat(),
            "test_start_date": data.spx_dates[train_end].date().isoformat(),
            "input_boundary": (
                "Official close-only 2006–2012 panel; the book's CRSP bid/ask and "
                "Compustat industry inputs are absent"
            ),
            "train_segment": "no positions; the original strategy trades only the test set",
        },
    )
    phi = coefficients[1 : 1 + data.hardware_close.shape[1]].T
    details = {
        "bic_lags_1_to_5": bic_values,
        "coefficients": coefficients,
        "phi": phi,
        "vec_c": phi - np.eye(phi.shape[0]),
        "forecast": forecast,
        "train_end": train_end,
    }
    return result, details


def kalman_hardware_adaptation(
    data: ChapterData,
) -> tuple[StrategyResult, dict[str, Any]]:
    """Replay Table 3.5 diagonal noise parameters on the official close panel."""
    prices = data.hardware_close
    process_loading = np.array([3.74, 0.34, 0.73, 0.67, 1.00])
    measurement_loading = np.array([4.54e-5, 0.08, 0.22, 0.19, 0.15])
    state = np.zeros(prices.shape[1])
    covariance = np.full(prices.shape[1], 1e7)
    forecast = np.full_like(prices, np.nan)
    filtered = np.full_like(prices, np.nan)
    for row, observation in enumerate(prices):
        forecast[row] = state
        predicted_covariance = covariance + process_loading**2
        gain = predicted_covariance / (
            predicted_covariance + measurement_loading**2
        )
        state = state + gain * (observation - state)
        covariance = (1.0 - gain) * predicted_covariance
        filtered[row] = state
    forecast_returns = forecast / prices - 1.0
    positions = normalized_sector_positions(forecast_returns)
    train_end = len(prices) // 2
    result = daily_panel_strategy(
        "Published-parameter Kalman moving-average adaptation",
        prices,
        positions,
        train_end,
        {
            "reproduction_grade": "published-parameter methodological adaptation",
            "process_loading_abs": process_loading.tolist(),
            "measurement_loading_abs": measurement_loading.tolist(),
            "fit_boundary": (
                "Book parameters replayed; no MATLAB maximum-likelihood re-estimation"
            ),
            "input_boundary": "Official close data differs from the book's CRSP midquotes",
        },
    )
    details = {
        "forecast": forecast,
        "filtered": filtered,
        "train_end": train_end,
        "mean_absolute_forecast_error_fraction": float(
            np.nanmean(np.abs(forecast[1:] - prices[1:]) / prices[1:])
        ),
    }
    return result, details


def kalman_pair_replay(data: ChapterData) -> tuple[StrategyResult, dict[str, Any]]:
    """Replay the published EWA–EWC random-walk hedge parameters."""
    process_covariance = PAIR_B @ PAIR_B.T
    measurement_variance = PAIR_D**2
    state = np.zeros(2)
    covariance = np.eye(2) * 1e7
    beta = np.zeros((len(data.ewa), 2))
    innovations = np.zeros(len(data.ewa))
    innovation_standard_deviation = np.zeros(len(data.ewa))
    units = np.zeros(len(data.ewa))
    long_unit = 0.0
    short_unit = 0.0
    identity = np.eye(2)

    for row, (ewa, ewc) in enumerate(zip(data.ewa, data.ewc, strict=True)):
        predicted_state = state
        predicted_covariance = covariance + process_covariance
        measurement = np.array([ewa, 1.0])
        forecast = float(measurement @ predicted_state)
        variance = float(
            measurement @ predicted_covariance @ measurement
            + measurement_variance
        )
        innovation = ewc - forecast
        gain = predicted_covariance @ measurement / variance
        state = predicted_state + gain * innovation
        covariance = (identity - np.outer(gain, measurement)) @ predicted_covariance
        beta[row] = state
        innovations[row] = innovation
        innovation_standard_deviation[row] = np.sqrt(variance)

        if innovation < -innovation_standard_deviation[row]:
            long_unit = 1.0
        elif innovation > -innovation_standard_deviation[row]:
            long_unit = 0.0
        if innovation > innovation_standard_deviation[row]:
            short_unit = -1.0
        elif innovation < innovation_standard_deviation[row]:
            short_unit = 0.0
        units[row] = long_unit + short_unit

    pair_prices = np.column_stack([data.ewa, data.ewc])
    dollar_positions = units[:, None] * np.column_stack(
        [-beta[:, 0] * data.ewa, data.ewc]
    )
    returns = np.zeros(len(pair_prices))
    pnl = np.sum(
        dollar_positions[:-1] * (pair_prices[1:] / pair_prices[:-1] - 1.0),
        axis=1,
    )
    gross = np.sum(np.abs(dollar_positions[:-1]), axis=1)
    returns[1:] = np.divide(
        pnl, gross, out=np.zeros_like(pnl), where=gross > 0
    )
    normalized_positions = np.divide(
        dollar_positions,
        np.sum(np.abs(dollar_positions), axis=1, keepdims=True),
        out=np.zeros_like(dollar_positions),
        where=np.sum(np.abs(dollar_positions), axis=1, keepdims=True) > 0,
    )
    train_end = 1250
    turnover = np.zeros(len(pair_prices))
    turnover[1:] = np.sum(
        np.abs(normalized_positions[1:] - normalized_positions[:-1]), axis=1
    )
    costs = np.zeros(len(pair_prices))
    costs[1:] = 0.0002 * turnover[:-1]
    result = StrategyResult(
        name="Published-parameter EWA–EWC dynamic hedge",
        returns=returns,
        positions=dollar_positions,
        performance=performance(returns[train_end:], 252),
        cost_performance=performance(
            returns[train_end:] - costs[train_end:], 252
        ),
        metadata={
            "reproduction_grade": "published-parameter numerical replay",
            "train_performance": asdict(performance(returns[:train_end], 252)),
            "test_start": train_end,
            "process_loading": PAIR_B.tolist(),
            "measurement_loading": PAIR_D,
            "signal_threshold": "± one-step forecast standard deviation",
            "state_changes": int(np.count_nonzero(np.diff(units))),
            "one_way_cost_sensitivity_bps": 2.0,
        },
    )
    details = {
        "beta": beta,
        "innovations": innovations,
        "innovation_standard_deviation": innovation_standard_deviation,
        "units": units,
        "train_end": train_end,
    }
    return result, details


def run_experiments(data: ChapterData) -> dict[str, Any]:
    aud_ar1 = fit_ar_conditional(data.aud_mid, 1, len(data.aud_mid))
    btc_ar1 = fit_ar_conditional(data.btc_close, 1, len(data.btc_close))
    selected_lag, ar_bic = ar_bic_yule_walker(data.aud_mid, max_lag=60)
    ar10_full_coefficients = fit_ar_conditional(
        data.aud_mid, selected_lag, len(data.aud_mid)
    )
    ar10_train_coefficients = fit_ar_conditional(
        data.aud_mid, selected_lag, data.aud_train_end
    )
    ar10_full = trade_intraday_forecast(
        "AR(10) source-faithful full-sample fit",
        data,
        forecast_ar(data.aud_mid, ar10_full_coefficients),
        {
            "reproduction_grade": "numerical reproduction with diagnosed look-ahead",
            "fit_window": "entire sample, matching original MATLAB",
            "coefficients": ar10_full_coefficients.tolist(),
        },
    )
    ar10_train = trade_intraday_forecast(
        "AR(10) chronological train-only correction",
        data,
        forecast_ar(data.aud_mid, ar10_train_coefficients),
        {
            "reproduction_grade": "corrected out-of-sample variant",
            "fit_window": f"rows 0:{data.aud_train_end}",
            "coefficients": ar10_train_coefficients.tolist(),
        },
    )
    arma25 = trade_intraday_forecast(
        "ARMA(2,5) published-parameter replay",
        data,
        forecast_arma_zero_innovations(
            data.aud_mid,
            ARMA25_BOOK_COEFFICIENTS["constant"],
            ARMA25_BOOK_COEFFICIENTS["ar"],
            ARMA25_BOOK_COEFFICIENTS["ma"],
        ),
        {
            "reproduction_grade": "published-parameter numerical replay",
            "forecast_semantics": (
                "MATLAB forecast with Y0 supplied and future MA innovations set to zero"
            ),
            "coefficients": {
                "constant": ARMA25_BOOK_COEFFICIENTS["constant"],
                "ar": ARMA25_BOOK_COEFFICIENTS["ar"].tolist(),
                "ma": ARMA25_BOOK_COEFFICIENTS["ma"].tolist(),
            },
        },
    )
    var_result, var_details = fit_var_adaptation(data)
    hardware_kalman, hardware_details = kalman_hardware_adaptation(data)
    pair_kalman, pair_details = kalman_pair_replay(data)
    return {
        "aud_ar1_coefficients": aud_ar1,
        "btc_ar1_coefficients": btc_ar1,
        "ar_bic": ar_bic,
        "selected_ar_lag": selected_lag,
        "ar10_full_coefficients": ar10_full_coefficients,
        "ar10_train_coefficients": ar10_train_coefficients,
        "ar10_full": ar10_full,
        "ar10_train": ar10_train,
        "arma25": arma25,
        "var": var_result,
        "var_details": var_details,
        "hardware_kalman": hardware_kalman,
        "hardware_details": hardware_details,
        "pair_kalman": pair_kalman,
        "pair_details": pair_details,
    }


def create_model_map_figure() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 3.6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = (
        (0.2, 1.7, "Univariate\nAR / ARMA / ARIMA", "#dbeafe"),
        (3.0, 1.7, "Multivariate\nVAR → VEC", "#dcfce7"),
        (5.8, 1.7, "Latent state\nState-space / Kalman", "#fef3c7"),
        (8.3, 1.7, "Trading rule\nlagged position + costs", "#fee2e2"),
    )
    for x, y, label, color in boxes:
        axis.text(
            x + 0.7,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "#334155"},
        )
    for left, right in ((1.6, 3.0), (4.4, 5.8), (7.2, 8.3)):
        axis.annotate(
            "",
            xy=(right, 1.7),
            xytext=(left, 1.7),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"},
        )
    axis.text(
        5,
        0.45,
        "Observed price history → forecast distribution → position decided at t → P&L realized at t+1",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    axis.set_title("Chapter 3 model-to-trade map", fontsize=14, weight="bold")
    figure.tight_layout()
    return figure


def wealth_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))


def create_ar_diagnostics_figure(
    data: ChapterData, results: dict[str, Any]
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    bic = results["ar_bic"]
    axes[0, 0].plot(np.arange(1, len(bic) + 1), bic - np.min(bic), color="#2563eb")
    axes[0, 0].axvline(results["selected_ar_lag"], color="#dc2626", ls="--")
    axes[0, 0].set(title="AUD.USD AR lag BIC (relative)", xlabel="lag p", ylabel="ΔBIC")

    index = np.arange(11)
    axes[0, 1].plot(index, results["ar10_full_coefficients"], "o-", label="full-sample source")
    axes[0, 1].plot(index, results["ar10_train_coefficients"], "s--", label="train-only")
    axes[0, 1].set(title="AR(10) coefficient audit", xlabel="intercept / lag", ylabel="coefficient")
    axes[0, 1].legend()

    for key, label, color in (
        ("ar10_full", "AR(10) full-fit (look-ahead)", "#dc2626"),
        ("ar10_train", "AR(10) train-only", "#2563eb"),
        ("arma25", "ARMA(2,5) parameter replay", "#059669"),
    ):
        curve = wealth_curve(results[key].returns)
        step = max(1, len(curve) // 1500)
        axes[1, 0].plot(curve[::step], label=label, color=color)
    axes[1, 0].set(title="One-year intraday test — midprice gross wealth", xlabel="sampled test bar", ylabel="wealth")
    axes[1, 0].legend(fontsize=8)

    gross = wealth_curve(results["ar10_train"].returns)
    net_returns = np.zeros_like(results["ar10_train"].returns)
    net_terminal = 1.0 + results["ar10_train"].cost_performance.cumulative_return
    gross_terminal = gross[-1]
    axes[1, 1].bar(
        ["midprice gross", "observed spread net"],
        [gross_terminal, net_terminal],
        color=["#2563eb", "#dc2626"],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="Execution sensitivity (AR10 train-only)", ylabel="terminal wealth (log scale)")
    axes[1, 1].text(
        0.5,
        0.95,
        "Market-order spread overwhelms the midprice edge",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.suptitle("Univariate model diagnostics and implementation risk", fontsize=14, weight="bold")
    figure.tight_layout()
    return figure


def create_multivariate_figure(
    data: ChapterData, results: dict[str, Any]
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    var = results["var"]
    var_start = results["var_details"]["train_end"]
    axes[0, 0].plot(
        data.spx_dates[var_start:],
        wealth_curve(var.returns[var_start:]),
        color="#2563eb",
    )
    axes[0, 0].axhline(1.0, color="#64748b", lw=1)
    axes[0, 0].set(title="VAR(1) close-data OOS adaptation", ylabel="wealth")

    vec_diagonal = np.diag(results["var_details"]["vec_c"])
    axes[0, 1].bar(HARDWARE_SYMBOLS, vec_diagonal, color="#059669")
    axes[0, 1].axhline(0, color="#475569", lw=1)
    axes[0, 1].set(title="VEC diagonal C = Φ − I", ylabel="own error-correction loading")

    hardware = results["hardware_kalman"]
    hardware_start = results["hardware_details"]["train_end"]
    axes[1, 0].plot(
        data.spx_dates[:hardware_start],
        wealth_curve(hardware.returns[:hardware_start]),
        label="first half",
        color="#7c3aed",
    )
    test_curve = wealth_curve(hardware.returns[hardware_start:])
    axes[1, 0].plot(
        data.spx_dates[hardware_start:],
        test_curve,
        label="second half, reset to 1",
        color="#dc2626",
    )
    axes[1, 0].set(title="Published-noise Kalman adaptation", ylabel="wealth")
    axes[1, 0].legend(fontsize=8)

    filtered = results["hardware_details"]["filtered"]
    axes[1, 1].plot(data.spx_dates, data.hardware_close[:, 0], alpha=0.45, label="AAPL close")
    axes[1, 1].plot(data.spx_dates, filtered[:, 0], lw=1.2, label="filtered state")
    axes[1, 1].set(title="Random-walk state estimate", ylabel="price")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("VAR/VEC and state-space adaptations", fontsize=14, weight="bold")
    figure.tight_layout()
    return figure


def create_pair_figure(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    details = results["pair_details"]
    pair = results["pair_kalman"]
    train_end = details["train_end"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(data.etf_dates, details["beta"][:, 0], color="#2563eb", label="slope β")
    axes[0].axvline(data.etf_dates[train_end], color="#dc2626", ls="--", label="book split")
    axes[0].set(ylabel="hedge slope", title="EWA–EWC dynamic hedge state")
    axes[0].legend(fontsize=8)

    error = details["innovations"]
    sigma = details["innovation_standard_deviation"]
    axes[1].plot(data.etf_dates, error, color="#475569", lw=0.8, label="forecast error")
    axes[1].plot(data.etf_dates, sigma, color="#dc2626", lw=0.8, label="+1σ")
    axes[1].plot(data.etf_dates, -sigma, color="#dc2626", lw=0.8, label="−1σ")
    axes[1].set_ylim(-4, 4)
    axes[1].set(ylabel="EWC dollars", title="One-step innovation and trading thresholds")
    axes[1].legend(fontsize=8, ncol=3)

    axes[2].plot(data.etf_dates[:train_end], wealth_curve(pair.returns[:train_end]), label="train replay", color="#059669")
    axes[2].plot(data.etf_dates[train_end:], wealth_curve(pair.returns[train_end:]), label="test replay, reset to 1", color="#dc2626")
    axes[2].set(ylabel="wealth", title="Lagged-dollar-position returns")
    axes[2].legend(fontsize=8)
    figure.suptitle("Published-parameter Kalman pair replay", fontsize=14, weight="bold")
    figure.tight_layout()
    return figure


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "model_map.png": create_model_map_figure(),
        "ar_diagnostics.png": create_ar_diagnostics_figure(data, results),
        "multivariate_diagnostics.png": create_multivariate_figure(data, results),
        "pair_kalman.png": create_pair_figure(data, results),
    }
    for name, figure in figures.items():
        figure.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
        plt.close(figure)


def check_close(actual: float, expected: float, tolerance: float) -> bool:
    return bool(abs(actual - expected) <= tolerance)


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    extraction_path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    original_ar10 = (
        CHAPTER_DIR / "original_matlab/buildARp_AUDUSD.m"
    ).read_text(encoding="utf-8")
    var_positions = results["var"].positions
    active_var_rows = np.sum(np.abs(var_positions), axis=1) > 0
    aud_boundary = (
        int(data.aud_tday[data.aud_train_end - 1]),
        int(data.aud_hhmm[data.aud_train_end - 1]),
    ) < (
        int(data.aud_tday[data.aud_train_end]),
        int(data.aud_hhmm[data.aud_train_end]),
    )
    checks = {
        "archive_manifest_matches": extraction["archive_sha256"] == chapter_manifest()["sha256"],
        "aud_ar1_phi_matches_source": check_close(results["aud_ar1_coefficients"][1], BOOK_RESULTS["aud_ar1_phi"], 2e-6),
        "btc_ar1_phi_matches_source": check_close(results["btc_ar1_coefficients"][1], BOOK_RESULTS["btc_daily_ar1_phi"], 0.003),
        "ar_bic_selects_book_lag_10": results["selected_ar_lag"] == BOOK_RESULTS["ar10_optimal_lag"],
        "ar10_full_fit_matches_book_output": check_close(results["ar10_full"].performance.annual_return, BOOK_RESULTS["ar10_annual_return"], 0.005),
        "arma25_replay_matches_book_output": check_close(results["arma25"].performance.annual_return, BOOK_RESULTS["arma25_annual_return"], 0.005),
        "var_close_bic_selects_lag_1": results["var"].metadata["lag"] == 1,
        "ar10_coefficients_match_table_31": bool(np.allclose(results["ar10_full_coefficients"], AR10_BOOK_COEFFICIENTS, atol=5e-6, rtol=0)),
        "source_ar10_lookahead_detected": "fit=estimate(model, mid);" in original_ar10 and "trainset" in original_ar10,
        "train_only_ar10_differs_from_source_fit": bool(not np.allclose(results["ar10_full_coefficients"], results["ar10_train_coefficients"], atol=1e-7, rtol=0)),
        "market_order_costs_do_not_improve_ar10": results["ar10_train"].cost_performance.cumulative_return <= results["ar10_train"].performance.cumulative_return,
        "var_positions_are_sector_neutral": bool(np.allclose(np.sum(var_positions[active_var_rows], axis=1), 0.0, atol=1e-12)),
        "vec_matrix_equals_var_phi_minus_identity": bool(np.allclose(results["var_details"]["vec_c"], results["var_details"]["phi"] - np.eye(len(HARDWARE_SYMBOLS)))),
        "kalman_pair_uses_published_noise_parameters": results["pair_kalman"].metadata["process_loading"] == PAIR_B.tolist() and results["pair_kalman"].metadata["measurement_loading"] == PAIR_D,
        "splits_are_chronological": bool(
            aud_boundary
            and data.spx_dates[results["var_details"]["train_end"] - 1] < data.spx_dates[results["var_details"]["train_end"]]
            and data.etf_dates[results["pair_details"]["train_end"] - 1] < data.etf_dates[results["pair_details"]["train_end"]]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Chapter 3 verification failed: {', '.join(failed)}")
    return checks


def verification_summary(checks: dict[str, bool]) -> dict[str, Any]:
    independent = [name for name in checks if VERIFICATION_CLASSES[name] == "independent_or_empirical"]
    invariants = [name for name in checks if VERIFICATION_CLASSES[name] == "contract_invariant"]
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


def strategy_metrics(result: StrategyResult) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    train = metadata.pop("train_performance", None)
    if str(metadata.get("train_segment", "")).startswith("no positions"):
        train = None
    return {
        "name": result.name,
        "train": train,
        "test": asdict(result.performance),
        "test_with_cost_sensitivity": asdict(result.cost_performance) if result.cost_performance else None,
        "metadata": metadata,
    }


def build_metrics(
    data: ChapterData, results: dict[str, Any], checks: dict[str, bool]
) -> dict[str, Any]:
    extraction_path = CHAPTER_DIR / "original_matlab/SOURCE_MANIFEST.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    return {
        "chapter": 3,
        "chapter_coverage": [
            {"topic": topic, "status": status, "evidence": evidence}
            for topic, status, evidence in CHAPTER_COVERAGE
        ],
        "random_seed": None,
        "randomness_note": "No stochastic estimator; fixed published parameters and deterministic linear algebra",
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
            "audusd": {
                "bars_after_leading_nan_trim": len(data.aud_mid),
                "train_bars": data.aud_train_end,
                "test_bars": len(data.aud_mid) - data.aud_train_end,
                "missing_mid": int(np.count_nonzero(~np.isfinite(data.aud_mid))),
                "nonpositive_mid": int(np.count_nonzero(data.aud_mid <= 0)),
                "mean_full_spread_bps": float(np.mean((data.aud_ask - data.aud_bid) / data.aud_mid) * 1e4),
            },
            "btc_daily_observations": len(data.btc_close),
            "hardware_close": {
                "shape": list(data.hardware_close.shape),
                "start_date": data.spx_dates[0].date().isoformat(),
                "end_date": data.spx_dates[-1].date().isoformat(),
                "symbols": list(HARDWARE_SYMBOLS),
                "missing": int(np.count_nonzero(~np.isfinite(data.hardware_close))),
            },
            "ewa_ewc": {
                "observations": len(data.ewa),
                "start_date": data.etf_dates[0].date().isoformat(),
                "end_date": data.etf_dates[-1].date().isoformat(),
                "missing": int(np.count_nonzero(~np.isfinite(np.column_stack([data.ewa, data.ewc])))),
            },
        },
        "univariate": {
            "aud_ar1_coefficients": results["aud_ar1_coefficients"].tolist(),
            "btc_ar1_coefficients": results["btc_ar1_coefficients"].tolist(),
            "ar_bic_lags_1_to_60": results["ar_bic"].tolist(),
            "selected_ar_lag": results["selected_ar_lag"],
            "ar10_full_coefficients": results["ar10_full_coefficients"].tolist(),
            "ar10_train_coefficients": results["ar10_train_coefficients"].tolist(),
        },
        "strategies": {
            key: strategy_metrics(results[key])
            for key in ("ar10_full", "ar10_train", "arma25", "var", "hardware_kalman", "pair_kalman")
        },
        "multivariate": {
            "var_bic_lags_1_to_5": results["var_details"]["bic_lags_1_to_5"],
            "var_coefficients": results["var_details"]["coefficients"].tolist(),
            "var_phi": results["var_details"]["phi"].tolist(),
            "vec_c": results["var_details"]["vec_c"].tolist(),
            "hardware_kalman_mean_absolute_forecast_error_fraction": results["hardware_details"]["mean_absolute_forecast_error_fraction"],
            "pair_final_beta": results["pair_details"]["beta"][-1].tolist(),
        },
        "book_results": BOOK_RESULTS,
        "verification": {"checks": checks, "summary": verification_summary(checks)},
    }


def coverage_table() -> str:
    rows = ["| Topic | Status | Evidence |", "|---|---|---|"]
    rows.extend(f"| {topic} | {status} | {evidence} |" for topic, status, evidence in CHAPTER_COVERAGE)
    return "\n".join(rows)


def comparison_table(results: dict[str, Any]) -> str:
    rows = [
        "| Experiment | 재현 등급 | Python annual return | Book annual return | 판단 |",
        "|---|---|---:|---:|---|",
        (
            f"| AR(10), full-sample fit | 수치 재현 + 누수 진단 | "
            f"{results['ar10_full'].performance.annual_return:.6f} | "
            f"{BOOK_RESULTS['ar10_annual_return']:.6f} | 0.005 이내, 그러나 look-ahead 포함 |"
        ),
        (
            f"| ARMA(2,5), published coefficients | 계수 기반 수치 재현 | "
            f"{results['arma25'].performance.annual_return:.6f} | "
            f"{BOOK_RESULTS['arma25_annual_return']:.6f} | 0.005 이내 |"
        ),
        (
            f"| Hardware VAR(1) | 방법론적 적응 | "
            f"{results['var'].performance.annual_return:.6f} | "
            f"{BOOK_RESULTS['var1_book_annual_return']:.6f} | 데이터·종목·기간이 달라 직접 비교 금지 |"
        ),
        (
            f"| Hardware Kalman, second half | published-parameter 적응 | "
            f"{results['hardware_kalman'].performance.annual_return:.6f} | "
            f"{BOOK_RESULTS['ssm_hardware_test_cagr']:.6f} | close와 CRSP midquote 차이 |"
        ),
    ]
    return "\n".join(rows)


def write_report(
    data: ChapterData, results: dict[str, Any], metrics: dict[str, Any]
) -> None:
    verification = metrics["verification"]["summary"]
    ar10_full = results["ar10_full"]
    ar10_train = results["ar10_train"]
    arma25 = results["arma25"]
    var = results["var"]
    hardware = results["hardware_kalman"]
    pair = results["pair_kalman"]
    report = fr"""# Chapter 3 Time Series Analysis — Python 재현 리포트

## 1. 문제 정의와 결론 미리보기

이 장의 질문은 단순히 “다음 가격을 맞힐 수 있는가”가 아니다. 관측 가격의
자기상관을 AR/ARMA로 표현하고, 여러 자산의 상호작용을 VAR/VEC로 연결하며,
관측되지 않는 공정가치와 동적 헤지 비율을 상태공간 모형으로 추정한 뒤 그 예측을
**한 시점 늦춘 포지션**으로 바꿀 때 무엇이 남는가를 묻는다.

가장 중요한 결론은 세 가지다. 첫째, 공식 `buildARp_AUDUSD.m`은 테스트 구간까지
포함한 전체 `mid`로 AR(10)을 적합해 책의 158% 결과에 look-ahead가 섞인다.
둘째, train-only로 고쳐도 midprice 수익은 강하지만 실제 bid/ask 반스프레드를
시장가 비용으로 차감하면 터미널 자산이 거의 0이 된다. 셋째, VAR·Kalman의 책
수치는 공식 ZIP만으로 정확 재현할 수 없으므로 공식 close 패널에서 실행한
방법론적 적응과 published-parameter replay로 등급을 낮춰 보고한다.

![모형 지도](figures/model_map.png)

## 2. Chapter coverage / 구현 범위

{coverage_table()}

지원 ZIP에는 28개 MATLAB/FIG 소스와 5개 데이터 파일이 있다. 모든 멤버를
추출·해시 검증했지만 책의 hardware VAR 및 Kalman 예제가 읽는
`C:/Projects/reversal_data` CRSP bid/ask 패널과 Compustat 산업분류 파일은 없다.
따라서 파일이 없는 예제를 현대의 임의 데이터로 바꿔 “정확 재현”이라 부르지 않는다.

## 3. Provenance, 환경, 데이터 진단

- 공식 페이지: {chapter_manifest()['source_page']}
- 공식 ZIP: `{chapter_manifest()['url']}`
- ZIP SHA-256: `{chapter_manifest()['sha256']}`
- 환경 잠금: `uv.lock` SHA-256 `{metrics['provenance']['uv_lock_sha256']}`
- random_seed = None: 난수 추정기는 없으며 고정 계수와 결정론적 선형대수만 쓴다.
- AUD.USD: 선두 NaN 제거 후 {len(data.aud_mid):,}분봉, 마지막 {AUD_PERIODS_PER_YEAR:,}개를 1년 test로 사용
- BTC/USD: 일봉 {len(data.btc_close):,}개
- hardware close: {data.hardware_close.shape}, {data.spx_dates[0].date()}–{data.spx_dates[-1].date()}, {', '.join(HARDWARE_SYMBOLS)}
- EWA/EWC: {len(data.ewa):,}일, {data.etf_dates[0].date()}–{data.etf_dates[-1].date()}

AUD 중간가격의 내부 결측과 비양수 값, 선택한 다섯 종목 및 EWA/EWC의 결측은 모두
0이다. 평균 full spread는 {metrics['data']['audusd']['mean_full_spread_bps']:.3f}bps다.
결측을 0으로 대체하거나 미래값으로 backfill하지 않는다. 원본 파일·추출 manifest·
잠금 파일의 checksum은 `metrics.json`에 모두 보존한다.

## 4. AR(1), 정상성, random walk, bid-ask bounce

AR(1)은

$$P_t = \phi_0 + \phi_1 P_{{t-1}} + \epsilon_t$$

이고 `fit_ar_conditional(prices, lag=1, end=...)`가 절편과 지연가격의 조건부 OLS를
구한다. AUD.USD의 $\phi_1$은 `{results['aud_ar1_coefficients'][1]:.9f}`로 책/소스의
`{BOOK_RESULTS['aud_ar1_phi']}`와 일치한다. BTC 일봉은
`{results['btc_ar1_coefficients'][1]:.9f}`로 소스 주석 `{BOOK_RESULTS['btc_daily_ar1_phi']}`와
0.003 이내다. 둘 다 1에 매우 가까우므로 가격수준 예측력이 곧 경제적 정상성을
뜻하지 않는다. 특히 거래가격의 음의 1차 자기상관은 bid-ask bounce일 수 있어
중간가격과 실제 체결가격을 구분해야 한다.

## 5. AR(p), BIC, Table 3.1, Figure 3.1

BIC는 후보 지연 1–60의 잔차분산과 모수 수를 함께 벌점화한다. 공식 AUD.USD에서
`p={results['selected_ar_lag']}`을 선택해 책과 같다. `forecast_ar`는 시점 $t$까지의
가격만으로 $t+1$을 예측하고 `trade_intraday_forecast`가 그 부호를 포지션으로
만든 뒤 `position[t] × return[t+1]`로 수익을 실현한다.

문제는 추정 창이다. 원본에는 `trainset`이 선언되어 있지만 실제 줄은
`fit=estimate(model, mid);`여서 전체표본을 쓴다. 이 경로의 연환산 수익은
`{ar10_full.performance.annual_return:.6f}`로 책의 `{BOOK_RESULTS['ar10_annual_return']:.6f}`와
0.005 이내이며 Table 3.1의 11개 계수도 5e-6 이내다. 이는 전략의 유효성보다
누수가 포함된 결과를 정확히 찾아낸 증거다.

`end=data.aud_train_end`로 수정한 train-only 계수는 전체표본 계수와 유의하게
다르고 진짜 OOS 연환산 수익은 `{ar10_train.performance.annual_return:.6f}`, Sharpe는
`{ar10_train.performance.sharpe:.3f}`다. 그러나 관측 반스프레드 × 포지션 회전율을
시장가 비용으로 차감한 연환산 수익은 `{ar10_train.cost_performance.annual_return:.6f}`다.
짧은 보유기간에서 midprice backtest가 체결 가능한 알파가 아님을 보여준다.

![AR 진단](figures/ar_diagnostics.png)

## 6. ARMA(2,5), ARIMA(1,1,9), 수식에서 코드로

ARMA는

$$P_t = \phi_0 + \sum_{{i=1}}^p \phi_i P_{{t-i}} +
        \sum_{{j=1}}^q \theta_j \epsilon_{{t-j}} + \epsilon_t.$$

책 Table 3.2의 계수를 직접 넣고 MATLAB `forecast(..., Y0=..., E0 omitted)`와 같이
미래 innovation을 0으로 두면 연환산 수익은 `{arma25.performance.annual_return:.6f}`로
책의 `{BOOK_RESULTS['arma25_annual_return']:.6f}`와 일치한다. 과거 잔차를 임의로
재귀시키는 다른 구현은 같은 모형이 아니므로 사용하지 않았다. 이 경로 역시 계수를
전체 자료에서 얻은 published-parameter replay이지 새로운 OOS 추정 성과가 아니다.

ARIMA(1,1,9)는 가격을 한 번 차분한 ARMA(1,9)다. $\Delta P_t$ 또는 로그수익을
모형화하면 단위근 가격수준보다 안정적일 수 있다. 공식 소스의 주문 탐색과 주석은
보존했지만, 다시 전체표본으로 계수를 고르는 선택편향을 추가하지 않기 위해 별도
성과곡선을 만들지 않았다. 이 절은 formula-to-code 대응과 source-output 비교다.

{comparison_table(results)}

## 7. VAR(1), sector neutrality, VEC

여러 가격을 벡터 $y_t$로 쓰면

$$y_t = a + \Phi_1 y_{{t-1}} + \cdots + \Phi_p y_{{t-p}} + \epsilon_t.$$

`fit_var_coefficients`가 다변량 OLS를, `normalized_sector_positions`가 종목별 예측
수익에서 같은 날 섹터 평균을 빼고 절대노출 합을 1로 만든다. 공식 close 패널의
마지막 252일을 OOS로 남겼을 때 BIC는 VAR(1)을 고른다. OOS 연환산 수익/Sharpe는
`{var.performance.annual_return:.6f}` / `{var.performance.sharpe:.3f}`이며, 책의
48%/0.9와 직접 비교하지 않는다. 책은 7개 CRSP midquote와 point-in-time 산업분류를
쓰지만 이 ZIP의 무결 close 패널에는 5개 종목만 있기 때문이다.

VAR(1)을 차분형으로 쓰면 $\Delta y_t = a + C y_{{t-1}} + \epsilon_t$이며
$C=\Phi-I$다. 코드에서 두 행렬의 동일성을 자동 검증한다. 다섯 대각 원소가 모두
음수라는 사실만으로 공적분 순위가 증명되지는 않는다. Johansen 검정은 deterministic
term, lag, 표본기간 선택에 민감하며 본 적응의 좋은 성과를 보장하지 않는다.

![다변량 진단](figures/multivariate_diagnostics.png)

## 8. 상태공간 모형과 hardware Kalman 적응

상태·관측 방정식은

$$x_t = A x_{{t-1}} + B u_t, \qquad y_t = C x_t + D e_t.$$

이동 공정가치 예제는 $A=C=I$인 random walk다. Table 3.5의 대각 $B,D$ 절대값을
고정하고 diffuse covariance에서 필터를 시작했다. 공식 close 패널 전반부의
연환산 수익/Sharpe는 `{hardware.metadata['train_performance']['annual_return']:.6f}` /
`{hardware.metadata['train_performance']['sharpe']:.3f}`, 후반부는
`{hardware.performance.annual_return:.6f}` / `{hardware.performance.sharpe:.3f}`다.
평균 1-step 절대 예측오차/가격은
`{results['hardware_details']['mean_absolute_forecast_error_fraction']:.2%}`다.

책의 train 40.8%, test −0.38%를 맞추지 못한 것은 결함을 숨긴 결과가 아니라 입력
경계다. 원본은 별도 CRSP midquote에서 MATLAB MLE로 noise를 추정했다. 여기서는
책에 인쇄된 계수를 다른 close 기간에 적용했으므로 published-parameter
methodological adaptation이다.

## 9. EWA–EWC 동적 hedge ratio replay

EWC를 $y_t$, 측정행렬을 $[EWA_t,1]$로 두면 상태 $x_t=[\beta_t,\alpha_t]$가 시간에
따라 움직이는 hedge slope와 offset이다. Table 3.6에 주석으로 남은
`B={PAIR_B.tolist()}`, `D={PAIR_D}`를 그대로 사용했다. 일보 예측오차가 $-1\sigma$보다
작으면 EWC long, $+1\sigma$보다 크면 short에 진입하고, 포지션은 다음 날 수익에만
적용한다.

1250일 train replay의 연환산 수익/Sharpe는
`{pair.metadata['train_performance']['annual_return']:.6f}` /
`{pair.metadata['train_performance']['sharpe']:.3f}`이고, 마지막 250일 test는
`{pair.performance.annual_return:.6f}` / `{pair.performance.sharpe:.3f}`다.
상태 전환은 `{pair.metadata['state_changes']}`회다. 이 결과는 계수 replay이고,
Table 3.6의 MLE를 Python에서 다시 적합했다는 주장이 아니다. 테스트가 약해지는
것은 hedge 관계의 regime change와 임계값 과적합 가능성을 보여준다.

![동적 페어](figures/pair_kalman.png)

## 10. 거래비용, 편향, 위험과 연습문제

- **Look-ahead:** AR(10) 원본 전체표본 적합을 별도 경로로 격리하고 train-only 결과를 함께 둔다.
- **거래비용/슬리피지:** AUD는 실제 관측 반스프레드, 일봉 전략은 편도 2bps 민감도만 계산한다. market impact·latency·borrow fee는 없다.
- **Survivorship/selection bias:** 공식 close 파일 종목과 책의 CRSP/Compustat 유니버스는 다르다. 현재 살아 있는 종목만 다시 고르는 행위를 정확 재현이라 부르지 않는다.
- **Out-of-sample:** AR 수정본은 마지막 1년 분봉, VAR은 마지막 252일, pair는 1250/250 순차 분할이다. 파라미터 선택 자체가 이전 연구에서 고정되었다는 조건도 명시한다.
- **다중검정:** AR/ARMA 차수, 임계값, 종목군을 결과를 본 뒤 반복 선택하면 Sharpe와 drawdown이 과장된다.
- **위험지표:** CAGR, Sharpe, maximum drawdown, drawdown duration을 strict `metrics.json`에 기록한다. 낮은 낙폭이 미래 손실 상한은 아니다.
- **백테스트 범위:** AR/VAR/페어 곡선은 백테스트지만 HP·wavelet 설명과 ARIMA 등식은 백테스트가 아니다.

연습문제 3.1–3.8 중 정상성 해석, AR 차수 BIC, bid/ask 비용, VAR/VEC 변환,
Kalman 동적 헤지를 실행 또는 진단으로 매핑했다. HYG/SPY/MXN/BTC 보조 스크립트는
원본 archive에 보존하되 장 본문의 결과인 것처럼 섞지 않는다.

## 11. 자동 검증과 결론

총 `{verification['total']}`개 자동 검증이 모두 통과했다. 그중 공식 checksum·책 수치·
계수·BIC 같은 independent/empirical 검증이 `{verification['independent_or_empirical']}`개,
시차·중립성·VEC identity·고정 파라미터 같은 contract invariant가
`{verification['contract_invariant']}`개다. strict JSON은 NaN을 허용하지 않으며
재실행 시 같은 figure·report·notebook 산출물을 만든다.

결론은 “복잡한 시계열 모델이 이긴다”가 아니다. 가격수준의 거의 단위근인 계수,
누수로 부풀려진 AR 성과, 스프레드에 사라지는 분봉 알파, 다른 데이터에서 뒤집히는
VAR/Kalman 결과를 동시에 봐야 한다. 재사용할 절차는 **추정 창 → 1-step forecast →
lagged position → 체결비용 → 표본 외 평가 → 입력 경계**를 각각 검증하는 것이다.
"""
    assert_clean_markdown_math(report, "Chapter 3 report")
    write_text_if_changed(REPORT_DIR / "chapter3_report.md", report)


def write_readme() -> None:
    content = """# Chapter 3 Time Series Analysis 실험 환경

공식 Book 3 ZIP의 MATLAB/FIG 소스 28개를 `original_matlab/`에 보존하고, 5개
연구 데이터는 Git에서 제외된 `data/raw/book3/chapter_3/`에 둔다. AR/ARMA 책
수치 재현, 원본 AR(10) look-ahead 진단과 train-only 수정, 공식 close 기반
VAR/VEC·Kalman 적응, EWA–EWC published-parameter replay를 실행한다.

```bash
uv run python chapter_3_time_series_analysis/src/run_chapter3_analysis.py
uv run python chapter_3_time_series_analysis/src/run_chapter3_analysis.py --offline
uv run pytest tests/test_chapter3_numerics.py
```

`--skip-notebook`은 계산·검증·리포트·그림만 갱신한다. 저장소 `.codex` 감사 도구가
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
            """# Machine Trading — Chapter 3 Time Series Analysis

이 노트북은 Chapter 3의 AR, ARMA/ARIMA, VAR/VEC, 상태공간·Kalman 예제를
공식 지원 ZIP 데이터로 실행한다. 목표는 높은 수익률을 전시하는 것이 아니라
책의 식이 어떤 코드와 시차로 연결되는지, 원본 소스에 누수가 있는지, 체결비용을
넣으면 결론이 유지되는지를 감사 가능한 형태로 남기는 것이다.

학습 질문은 다음과 같다.

1. 가격수준의 AR 계수가 1에 가깝다는 사실은 어떤 위험을 뜻하는가?
2. BIC가 선택한 AR(10)의 책 수치는 train-only 결과인가?
3. ARMA의 MA 항은 MATLAB의 one-step `forecast`에서 어떻게 처리되는가?
4. VAR과 VEC의 행렬 관계는 어떻게 검증하는가?
5. Kalman filter의 예측 상태와 갱신 상태 중 어느 것을 거래 신호로 써야 하는가?
6. midprice 성과가 bid/ask 체결비용 뒤에도 살아남는가?

여기서 ‘수치 재현’, ‘published-parameter replay’, ‘방법론적 적응’, ‘데이터 부재’를
구분한다. 모든 곡선이 진짜 투자 가능한 백테스트라는 뜻은 아니다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 1. Chapter coverage / 재현 상태

| 주제 | 상태 | 핵심 증거 |
|---|---|---|
| AR(1), 정상성, random walk | 수치 재현 | AUD·BTC 원본 계수 비교 |
| bid-ask bounce | 실행 진단 | midprice gross와 관측 spread net |
| AR(p), BIC, Table 3.1 | 수치 재현 | p=10, 계수 5e-6 이내 |
| Figure 3.1 AR(10) | 원본 재현 + 수정 | full-sample 누수와 train-only 분리 |
| ARMA(2,5), Table 3.2 | 계수 replay | MATLAB zero-future-innovation 의미 |
| ARIMA(1,1,9) | 수식·source-output | 차분 가격과 수익률 연결 |
| VAR(1), sector neutral | 방법론적 적응 | 공식 close, 마지막 252일 OOS |
| VEC | identity 실행 | C = Φ − I |
| hardware Kalman | 계수 적응 | 책 noise를 다른 close 기간에 적용 |
| EWA–EWC Kalman | 계수 replay | Table 3.6 B,D와 1250/250 split |
| HP/wavelet·regime change | 개념 설명 | 입력·모형 한계 |
| 연습문제 3.1–3.8 | 선택 실행·매핑 | 비용, BIC, VAR/VEC, Kalman |

책의 hardware 예제가 요구하는 CRSP bid/ask와 Compustat 산업분류는 ZIP에 없으므로
close 적응을 책 수치의 정확 재현이라고 부르지 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            r"""## 2. 수식에서 코드로 / formula-to-code

$$P_t=\phi_0+\sum_{i=1}^{p}\phi_iP_{t-i}+\epsilon_t$$

$$y_t=a+\Phi_1y_{t-1}+\cdots+\Phi_py_{t-p}+\epsilon_t$$

$$x_t=Ax_{t-1}+Bu_t,\qquad y_t=C_tx_t+De_t$$

| 수식 단계 | 구현 함수 | 시차 계약 |
|---|---|---|
| 조건부 AR OLS | `fit_ar_conditional` | `end` 이전 자료만 적합 가능 |
| AR/ARMA 예측 | `forecast_ar`, `forecast_arma_zero_innovations` | t까지 보고 t+1 예측 |
| VAR OLS·BIC | `fit_var_coefficients` | 마지막 252일 이전 적합 |
| sector neutral | `normalized_sector_positions` | 같은 날 예측 평균 제거 |
| random-walk state | `kalman_hardware_adaptation` | prediction 후 observation update |
| 동적 hedge state | `kalman_pair_replay` | forecast error ±1σ로 상태 전환 |
| P&L | `trade_intraday_forecast`, `daily_panel_strategy` | position[t]는 return[t+1]에 적용 |

random_seed = None이다. 난수 기반 최적화는 없고, published coefficients와
결정론적 NumPy 선형대수만 사용한다."""
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
SRC = PROJECT_ROOT / "chapter_3_time_series_analysis/src"
sys.path.insert(0, str(SRC))
from run_chapter3_analysis import (
    AUDIT_SCRIPT_PATH, BOOK_RESULTS, CHAPTER_COVERAGE, NOTEBOOK_PATH,
    PYPROJECT_PATH, UV_LOCK_PATH, chapter_manifest, create_ar_diagnostics_figure,
    create_model_map_figure, create_multivariate_figure, create_pair_figure,
    environment_versions, load_chapter_data, run_experiments, sha256_file,
    validate_offline_assets, verification_summary, verify_results,
)

def show_figure(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    display(Image(data=buffer.getvalue()))

validate_offline_assets()
print("offline archive and all extracted members: checksum verified")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 3. Provenance와 잠금 환경

공식 페이지 URL, 직접 ZIP URL, ZIP SHA-256, `SOURCE_MANIFEST.json`의 각 멤버
SHA-256, `uv.lock` SHA-256을 함께 기록한다. URL만 남기면 공급자가 파일을 조용히
교체했을 때 같은 이름의 다른 데이터로 실행될 수 있다. 실제로 이 장의 ZIP은
다운로드 후 계산한 해시를 manifest에 고정했으며 offline 실행은 네트워크 없이
모든 파일을 재검증한다."""
        ),
        nbformat.v4.new_code_cell(
            """manifest = chapter_manifest()
provenance = pd.Series({
    "source_page": manifest["source_page"],
    "archive_url": manifest["url"],
    "archive_sha256": manifest["sha256"],
    "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
})
display(provenance.to_frame("value"))
display(pd.Series(environment_versions()).to_frame("environment version"))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 4. 데이터 구조와 결측 진단

AUD.USD 분봉은 파일 앞부분의 NaN만 잘라내고 내부 결측은 허용하지 않는다.
마지막 359,100분봉은 252거래일 × 하루 1,425분으로 정의한 테스트다. BTC는
345일 가격, hardware는 공식 close 패널에서 결측 없는 다섯 종목, ETF는 EWA와
EWC를 선택한다. 서로 다른 빈도와 기간의 결과를 한 숫자로 합치지 않는다.

결측을 미래 관측값으로 채우면 look-ahead가 생기고, 비양수 가격은 수익률 분모를
망가뜨린다. 아래 진단이 0인지 먼저 확인한다."""
        ),
        nbformat.v4.new_code_cell(
            """data = load_chapter_data()
diagnostics = pd.Series({
    "AUD bars": len(data.aud_mid),
    "AUD train bars": data.aud_train_end,
    "AUD test bars": len(data.aud_mid) - data.aud_train_end,
    "AUD missing mid": int(np.count_nonzero(~np.isfinite(data.aud_mid))),
    "AUD nonpositive mid": int(np.count_nonzero(data.aud_mid <= 0)),
    "BTC daily observations": len(data.btc_close),
    "hardware rows": len(data.hardware_close),
    "hardware missing": int(np.count_nonzero(~np.isfinite(data.hardware_close))),
    "EWA/EWC rows": len(data.ewa),
})
display(diagnostics.to_frame("value"))
assert diagnostics[["AUD missing mid", "AUD nonpositive mid", "hardware missing"]].sum() == 0"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 5. 모형에서 거래까지의 지도

단변량·다변량·잠재상태 모형은 모양이 달라도 거래 관점에서는 같은 경계를 지난다.
관측 가능한 시점까지 적합하고, 다음 관측을 예측하며, 예측 후에 포지션을 만들고,
그 다음 수익에 포지션을 적용해야 한다. 한 칸만 어긋나도 미래정보가 들어간다."""
        ),
        nbformat.v4.new_code_cell("""show_figure(create_model_map_figure())"""),
        nbformat.v4.new_markdown_cell(
            """## 6. 모든 실험 실행

다음 셀은 296만 분봉의 AR 충분통계, BIC 60개 후보, AR(10) 두 적합 창,
published ARMA, VAR/VEC, 두 Kalman 실험을 실행한다. 전체표본 AR(10)은 원본 수치
감사용이며 올바른 OOS 전략으로 승격시키지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """results = run_experiments(data)
print("selected AUD AR lag:", results["selected_ar_lag"])
print("experiments:", [k for k in ("ar10_full", "ar10_train", "arma25", "var", "hardware_kalman", "pair_kalman")])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 7. AR(1)과 정상성 해석

AUD와 BTC의 AR(1) 계수가 1에 매우 가깝다. 이는 가격수준 충격이 오래 지속됨을
뜻하며, 단순한 평균회귀 신호로 읽어서는 안 된다. 거래가격에서 관찰한 짧은 음의
자기상관은 spread bounce일 수 있으므로 mid, bid, ask를 분리해야 한다."""
        ),
        nbformat.v4.new_code_cell(
            """ar1 = pd.DataFrame({
    "Python phi1": [results["aud_ar1_coefficients"][1], results["btc_ar1_coefficients"][1]],
    "book/source phi1": [BOOK_RESULTS["aud_ar1_phi"], BOOK_RESULTS["btc_daily_ar1_phi"]],
}, index=["AUD.USD minute", "BTC.USD daily"])
display(ar1)
assert abs(ar1.iloc[0, 0] - ar1.iloc[0, 1]) <= 2e-6
assert abs(ar1.iloc[1, 0] - ar1.iloc[1, 1]) <= 0.003"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 8. AR(p), BIC와 원본 look-ahead

BIC는 p=10을 고른다. 책 결과를 만든 MATLAB은 `trainset`을 선언했지만
`estimate(model, mid)`로 전체 sample을 적합한다. 따라서 책의 158%는 미래 테스트
관측이 계수에 들어간 결과다. train-only 계수로 바꾸면 수익은 낮아지고, 이 차이는
반드시 별도 행으로 보고해야 한다."""
        ),
        nbformat.v4.new_code_cell(
            """comparison = pd.DataFrame({
    "annual return": [results[k].performance.annual_return for k in ("ar10_full", "ar10_train", "arma25")],
    "Sharpe": [results[k].performance.sharpe for k in ("ar10_full", "ar10_train", "arma25")],
    "spread/cost annual return": [results[k].cost_performance.annual_return for k in ("ar10_full", "ar10_train", "arma25")],
}, index=["AR10 full-fit (look-ahead)", "AR10 train-only", "ARMA25 replay"])
display(comparison)
assert results["selected_ar_lag"] == 10
assert abs(results["ar10_full"].performance.annual_return - BOOK_RESULTS["ar10_annual_return"]) <= 0.005
assert not np.allclose(results["ar10_full_coefficients"], results["ar10_train_coefficients"], atol=1e-7, rtol=0)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 9. ARMA/ARIMA와 거래비용

ARMA(2,5)는 Table 3.2의 계수를 replay한다. MATLAB `forecast`에 과거 가격 Y0만
주고 E0를 생략한 one-step 예측은 미래 MA innovation을 0으로 둔다. 임의의 잔차
재귀를 더하면 책과 다른 전략이다. ARIMA(1,1,9)는 가격을 한 번 차분한 뒤
ARMA를 적용하는 표현이며, 차분 또는 로그수익이 단위근 가격수준보다 안정적이다.

midprice gross curve는 실행 가능성의 상한에 가깝다. 관측 half-spread × turnover를
차감하면 고빈도 신호가 거의 전부 사라진다. 슬리피지, latency, market impact까지
포함하지 않았으므로 net 결과도 낙관적일 수 있다."""
        ),
        nbformat.v4.new_code_cell("""show_figure(create_ar_diagnostics_figure(data, results))"""),
        nbformat.v4.new_markdown_cell(
            """## 10. VAR(1)과 sector-neutral OOS

VAR은 다섯 가격의 lag를 동시에 사용한다. 공식 close 데이터의 마지막 252일을
완전히 남기고 이전 구간에서만 계수를 적합했다. 포지션은 예측수익에서 당일 섹터
평균을 빼 순합 0, 절대합 1로 정규화한다. 하지만 종목 수 중립은 beta·달러·산업
노출의 완전한 중립과 다르다.

책의 48%/Sharpe 0.9는 7개 CRSP midquote 및 Compustat 분류에서 나온다. 현재
공식 ZIP에는 그 입력이 없으므로 아래 결과는 방법론적 적응이며 직접 수치 비교는
금지한다."""
        ),
        nbformat.v4.new_code_cell(
            """var = results["var"]
var_table = pd.Series({
    "selected lag": var.metadata["lag"],
    "OOS annual return": var.performance.annual_return,
    "OOS Sharpe": var.performance.sharpe,
    "OOS maximum drawdown": var.performance.maximum_drawdown,
})
display(var_table.to_frame("value"))
active = np.sum(np.abs(var.positions), axis=1) > 0
assert np.allclose(var.positions[active].sum(axis=1), 0, atol=1e-12)"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 11. VEC identity와 공적분 주의

VAR(1)을 차분하면 C=Φ−I인 VEC 형태가 된다. 이 행렬 등식은 코드 계약으로
검증할 수 있지만, C의 대각이 음수라는 것만으로 공적분 rank가 정해지지는 않는다.
Johansen 검정의 lag와 deterministic term을 결과를 본 뒤 선택하면 data snooping이
생긴다. 관계가 과거에 안정적이었다는 사실도 regime change 뒤의 지속성을 보장하지
않는다."""
        ),
        nbformat.v4.new_code_cell(
            """phi = results["var_details"]["phi"]
vec_c = results["var_details"]["vec_c"]
display(pd.DataFrame(vec_c, index=["Δ"+s for s in ("AAPL","EMC","HPQ","NTAP","SNDK")], columns=("AAPL","EMC","HPQ","NTAP","SNDK")))
assert np.allclose(vec_c, phi - np.eye(5))
show_figure(create_multivariate_figure(data, results))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 12. 상태공간·Kalman hardware 적응

random-walk state는 어제의 filtered state를 오늘의 prediction으로 사용하고 오늘
가격을 본 뒤 update한다. updated state를 같은 날 예측으로 쓰면 미래관측을 섞는
오류다. Table 3.5의 noise loading을 고정해 공식 close 기간에 적용했으며 MATLAB
MLE를 다시 실행하지 않았다. 그래서 책 수치와 차이가 나도 published-parameter
methodological adaptation이라는 분류가 유지된다."""
        ),
        nbformat.v4.new_code_cell(
            """hardware = results["hardware_kalman"]
display(pd.DataFrame({
    "train replay": hardware.metadata["train_performance"],
    "second-half test": hardware.performance.__dict__,
}).loc[["annual_return", "sharpe", "maximum_drawdown"]])
print("mean absolute one-step forecast error / price:", results["hardware_details"]["mean_absolute_forecast_error_fraction"])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 13. EWA–EWC 동적 hedge ratio

EWC 관측식의 기울기와 절편을 상태로 두고 Table 3.6의 B,D를 그대로 replay한다.
innovation이 ±1σ를 넘을 때만 long/short 상태를 바꾸며, 상태가 정한 dollar
positions는 하루 lag 뒤 P&L에 적용한다. 1250일 학습 replay 뒤 250일 테스트가
약해지는 현상은 공적분·hedge 관계의 regime change와 임계값 과적합 가능성을
경고한다. 이는 새로 최적화한 백테스트가 아니라 published-parameter replay다."""
        ),
        nbformat.v4.new_code_cell(
            """pair = results["pair_kalman"]
display(pd.DataFrame({
    "train replay": pair.metadata["train_performance"],
    "test": pair.performance.__dict__,
}).loc[["annual_return", "sharpe", "maximum_drawdown"]])
print("state changes:", pair.metadata["state_changes"])
assert pair.metadata["process_loading"] == [[-0.01015, 0.02114], [0.40606, -0.32381]]
show_figure(create_pair_figure(data, results))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 14. Look-ahead, survivorship, selection bias

원본 look-ahead를 그대로 재현하는 경로는 버리지 않되 빨간 라벨로 격리한다.
train-only 수정본과 나란히 둬야 왜 숫자가 달라졌는지 설명할 수 있다. hardware
close의 다섯 종목은 책의 시점별 CRSP/Compustat universe가 아니므로 survivorship와
selection bias를 독립적으로 제거했다고 주장할 수 없다. AR/VAR 차수, 종목군,
Kalman threshold를 같은 test 결과를 보며 반복 선택하면 표본 외가 다시 학습자료가
된다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 15. 위험지표와 백테스트 범위

CAGR과 Sharpe만으로 충분하지 않아 maximum drawdown과 drawdown duration도 저장한다.
거래비용은 AUD의 관측 spread와 일봉 편도 2bps 민감도이며 borrow fee와 market
impact는 빠져 있다. AR/VAR/페어 결과는 명시한 시차의 백테스트지만, HP/wavelet
설명·ARIMA 등식·state equation 설명은 백테스트가 아님을 구분한다. 서로 다른
기간의 train과 test 곡선을 이어붙여 하나의 지속적 wealth처럼 보이지 않는다."""
        ),
        nbformat.v4.new_markdown_cell(
            """## 16. 자동 검증

검증은 책과의 독립/경험 비교와 계산 계약 invariant를 분리한다. checksum,
AR 계수, BIC, 책 수익률, Table 3.1은 전자다. 원본 누수 문자열, train/full 차이,
비용 단조성, sector neutrality, VEC identity, Kalman 고정계수, 순차 split은 후자다.
assert가 실패하면 리포트와 metrics를 성공 산출물로 남기지 않는다."""
        ),
        nbformat.v4.new_code_cell(
            """checks = verify_results(data, results)
summary = verification_summary(checks)
display(pd.Series(checks).to_frame("passed"))
display(pd.Series(summary).drop("classification").to_frame("count"))
assert all(checks.values())
assert summary["total"] == 15 and summary["passed"] == 15"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 17. 결론

좋은 시계열 실험은 예측모형 이름보다 정보 경계를 먼저 고정한다. 이 장에서는
책 AR/ARMA 숫자를 재현하면서도 AR(10)의 전체표본 적합을 발견했고, train-only
수정 뒤에도 spread가 분봉 알파를 없앤다는 점을 확인했다. VAR/VEC와 Kalman은
필요 입력이 없는 부분을 정확 재현으로 포장하지 않고 공식 close 적응과 계수
replay로 제한했다.

다른 책과 프로젝트에도 적용할 체크리스트는 같다. 원본 archive와 checksum을
고정하고, 수식마다 구현 함수를 연결하며, fit/forecast/position/P&L의 시차를
assert하고, 비용 전후를 함께 표시하며, exact·approximate·adaptation·unavailable을
명시한다. 마지막으로 test 결과를 보고 모형을 다시 고르면 새 test가 필요하다."""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch3-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"Chapter 3 notebook cell {index}")
    execute_notebook(notebook, NOTEBOOK_PATH, workdir=PROJECT_ROOT, timeout=300)


def audit_current_notebook() -> dict[str, Any] | None:
    if not AUDIT_SCRIPT_PATH.exists():
        warnings.warn(
            f"Optional quality audit unavailable: {AUDIT_SCRIPT_PATH}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    spec = importlib.util.spec_from_file_location("chapter3_audit", AUDIT_SCRIPT_PATH)
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
    content = f"""# Chapter 3 품질 벤치마크

실행된 노트북과 저장소 감사 rubric에서 자동 생성한 결과다.

| Chapter | 실행 | 재현성 | 엄밀성 | 교육 | 트레이딩 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch3** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

- 셀 {counts['cells']}개: Markdown {counts['markdown_cells']}, code {counts['code_cells']}.
- code 실행 {counts['executed_code_cells']}/{counts['code_cells']}, 오류 {counts['error_outputs']}, 인라인 PNG {counts['embedded_png_outputs']}.
- Markdown {counts['markdown_characters']:,}자, 한국어 {counts['korean_characters']:,}자, 헤딩 {counts['headings']}.
- 검증 {summary['total']}개: 독립/경험 {summary['independent_or_empirical']}, invariant {summary['contract_invariant']}.
- 원본 look-ahead와 train-only 수정본, 수치 재현과 입력 제약 적응을 수동 구분했다.
"""
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", content)


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_chapter_data()
    results = run_experiments(data)
    checks = verify_results(data, results)
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
    print("Chapter 3 analysis complete")
    print("AR(10) source-faithful annual return:", metrics["strategies"]["ar10_full"]["test"]["annual_return"])
    print("AR(10) train-only annual return:", metrics["strategies"]["ar10_train"]["test"]["annual_return"])
    print("Report:", REPORT_DIR / "chapter3_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
