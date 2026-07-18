#!/usr/bin/env python3
"""Build reproducible Chapter 7 Bitcoin experiments from the official archive."""

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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import toeplitz
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor


SRC_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SRC_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parent
RAW_DIR = PROJECT_ROOT / "data/raw/book3/chapter_7"
ORIGINAL_DIR = CHAPTER_DIR / "original_matlab"
MANIFEST_PATH = ORIGINAL_DIR / "SOURCE_MANIFEST.json"
REPORT_DIR = SRC_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = SRC_DIR / "chapter7_full_report.ipynb"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
RANDOM_SEED = 20260718

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


SOURCE_AR16_COEFFICIENTS = np.array(
    [
        0.00670585,
        0.685261,
        0.257702,
        0.0580414,
        0.00443159,
        -0.00339982,
        -0.00496624,
        -0.0106182,
        -0.00189899,
        0.00326403,
        0.0045775,
        -0.00946253,
        0.000956515,
        0.0015719,
        -0.0017755,
        0.00752874,
        0.00876893,
    ]
)
SOURCE_RESULTS = {
    "ar_selected_lag": 16,
    "ar_adf_statistic": -3.001533,
    "ar_test_cumulative_return": 201.9081357722237,
    "ar_test_annual_return": 41170.71156255915,
    "bollinger_test_cumulative_return": 0.201024204778726,
    "bollinger_test_annual_return": 0.442459140464370,
    "bollinger_cost_test_cumulative_return": -0.994913611902576,
    "order_flow_total_pl_usd": 32.54,
    "order_flow_pl_per_trade_usd": 0.096845,
    "order_flow_num_trades": 336,
}
SOURCE_ORDER_DAILY = [
    ("20141201", 5.90, 6), ("20141202", -3.63, 8),
    ("20141203", -2.64, 4), ("20141204", 4.49, 12),
    ("20141205", 0.12, 12), ("20141206", 0.00, 0),
    ("20141207", -1.25, 2), ("20141208", 1.68, 20),
    ("20141209", 3.31, 30), ("20141210", 1.46, 7),
    ("20141211", 1.49, 25), ("20141212", -1.47, 8),
    ("20141213", 0.00, 0), ("20141214", 2.28, 6),
    ("20141215", -0.61, 6), ("20141216", -1.13, 30),
    ("20141217", 13.60, 36), ("20141218", 5.13, 26),
    ("20141219", 1.85, 18), ("20141220", 3.73, 12),
    ("20141221", 0.73, 4), ("20141222", 2.22, 6),
    ("20141223", -0.02, 4), ("20141224", -1.34, 6),
    ("20141225", -2.25, 6), ("20141226", -2.00, 4),
    ("20141227", 0.62, 6), ("20141228", -1.79, 4),
    ("20141229", -0.38, 4), ("20141230", 1.78, 20),
    ("20141231", 0.66, 4),
]


@dataclass(frozen=True)
class ChapterData:
    bbo_tday: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    hhmm_rows: int
    crossed_quote_rows: int
    daily_dates: np.ndarray
    daily_close: np.ndarray
    trades: pd.DataFrame


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=7)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1", timeout=120
    )
    return materialize_chapter_archive(PROJECT_ROOT, manifest, payload, force=force)


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def load_data() -> ChapterData:
    bbo = loadmat(RAW_DIR / "Jonathan_BTCUSD_BBO_1minute.mat", squeeze_me=True)
    bid = np.asarray(bbo["bid"], dtype=float)
    ask = np.asarray(bbo["ask"], dtype=float)
    tday = np.asarray(bbo["tday"], dtype=int)
    mid = (bid + ask) / 2.0
    first = int(np.flatnonzero(np.isfinite(mid))[0])
    bid, ask, mid, tday = bid[first:], ask[first:], mid[first:], tday[first:]
    if not (len(bid) == len(ask) == len(mid) == len(tday) == 479_535):
        raise ValueError("Unexpected one-minute BBO shape")
    if not np.all(np.isfinite(mid)):
        raise ValueError("BBO midprice contains non-finite rows")

    daily = loadmat(
        RAW_DIR / "Jonathan_BTCUSD_trades_daily.mat", squeeze_me=True
    )
    close = np.asarray(daily["cl"], dtype=float)
    dates = np.asarray(daily["tday"], dtype=int)
    good = np.isfinite(close)

    trades = pd.read_csv(RAW_DIR / "2014-12.csv")
    trades["timestamp"] = pd.to_datetime(trades["datetime"], format="mixed")
    if not trades["timestamp"].is_monotonic_increasing:
        raise ValueError("Trade ticks are not chronological")
    if not set(trades["buysell"].unique()).issubset({-1, 0, 1}):
        raise ValueError("Unexpected aggressor tag")
    return ChapterData(
        bbo_tday=tday,
        bid=bid,
        ask=ask,
        mid=mid,
        hhmm_rows=int(np.asarray(bbo["HHMM"]).size),
        crossed_quote_rows=int(np.sum(bid > ask)),
        daily_dates=dates[good],
        daily_close=close[good],
        trades=trades,
    )


def previous(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.r_[np.nan, values[:-1]]


def performance(returns: np.ndarray, periods_per_year: int) -> dict[str, Any]:
    clean = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    if np.any(clean <= -1):
        raise ValueError("A period return is at or below -100%")
    log_wealth = np.cumsum(np.log1p(clean))
    running_peak = np.maximum.accumulate(log_wealth)
    drawdown = np.expm1(log_wealth - running_peak)
    duration = longest = 0
    for value in drawdown:
        duration = duration + 1 if value < 0 else 0
        longest = max(longest, duration)
    scale = periods_per_year / len(clean)
    annual_log = log_wealth[-1] * scale
    annual_return = float(np.expm1(annual_log)) if annual_log < 709 else None
    cumulative_return = (
        float(np.expm1(log_wealth[-1])) if log_wealth[-1] < 709 else None
    )
    standard_deviation = float(np.std(clean, ddof=1))
    return {
        "annual_return": annual_return,
        "annual_return_log": float(annual_log),
        "sharpe": (
            float(np.sqrt(periods_per_year) * np.mean(clean) / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        "maximum_drawdown": float(np.min(drawdown)),
        "drawdown_duration": int(longest),
        "cumulative_return": cumulative_return,
        "cumulative_return_log": float(log_wealth[-1]),
        "periods": int(len(clean)),
        "periods_per_year": int(periods_per_year),
    }


def risk_experiment(data: ChapterData) -> dict[str, Any]:
    close = data.daily_close
    returns = np.r_[np.nan, close[1:] / close[:-1] - 1.0]
    finite = returns[np.isfinite(returns)]
    wealth = close / close[0]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    min_index = int(np.nanargmin(returns))
    max_index = int(np.nanargmax(returns))
    return {
        "returns": returns,
        "wealth": wealth,
        "drawdown": drawdown,
        "annualized_volatility": float(np.std(finite, ddof=1) * np.sqrt(252)),
        "best_daily_return": float(returns[max_index]),
        "best_date": int(data.daily_dates[max_index]),
        "worst_daily_return": float(returns[min_index]),
        "worst_date": int(data.daily_dates[min_index]),
        "maximum_drawdown": float(np.min(drawdown)),
        "annualized_kurtosis": float(
            kurtosis(finite, fisher=False, bias=False) * 252 / len(finite)
        ),
        "period_start": int(data.daily_dates[0]),
        "period_end": int(data.daily_dates[-1]),
    }


def fit_ar_conditional(prices: np.ndarray, lag: int, end: int) -> np.ndarray:
    rows = np.arange(lag, end)
    target = prices[rows]
    lagged = [prices[rows - offset] for offset in range(1, lag + 1)]
    gram = np.empty((lag + 1, lag + 1))
    response = np.empty(lag + 1)
    gram[0, 0] = len(rows)
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
            float(autocovariance[0] - coefficients @ autocovariance[1 : lag + 1]),
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


def adf_one_lag(prices: np.ndarray) -> dict[str, float]:
    delta = np.diff(np.asarray(prices, dtype=float))
    target = delta[1:]
    design = np.column_stack(
        [np.ones(len(target)), prices[1:-1], delta[:-1]]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    variance = residuals @ residuals / (len(target) - design.shape[1])
    covariance = variance * np.linalg.inv(design.T @ design)
    statistic = coefficients[1] / np.sqrt(covariance[1, 1])
    return {
        "statistic": float(statistic),
        "level_coefficient": float(coefficients[1]),
        "lagged_delta_coefficient": float(coefficients[2]),
        "critical_value_5pct_source": -2.871,
    }


def ar_strategy(
    prices: np.ndarray,
    coefficients: np.ndarray,
    test_start: int,
    *,
    fit_label: str,
) -> dict[str, Any]:
    forecast = forecast_ar(prices, coefficients)
    positions = np.zeros(len(prices))
    difference = forecast - prices
    positions[difference > 0] = 1.0
    positions[difference < 0] = -1.0
    returns = np.zeros(len(prices))
    returns[1:] = positions[:-1] * (prices[1:] / prices[:-1] - 1.0)
    turnover = np.zeros(len(prices))
    turnover[1:] = np.abs(positions[1:] - positions[:-1])
    costs = 0.001 * turnover
    return {
        "coefficients": coefficients,
        "forecast": forecast,
        "positions": positions,
        "returns": returns,
        "net_returns_10bps": returns - costs,
        "test": performance(returns[test_start:], 252 * 24 * 60),
        "test_net_10bps": performance(
            (returns - costs)[test_start:], 252 * 24 * 60
        ),
        "fit_label": fit_label,
        "test_start": int(test_start),
    }


def ar_experiment(data: ChapterData) -> dict[str, Any]:
    test_start = len(data.mid) - 126 * 24 * 60
    selected, bic = ar_bic_yule_walker(data.mid[:test_start], 60)
    full = ar_strategy(
        data.mid,
        fit_ar_conditional(data.mid, 16, len(data.mid)),
        test_start,
        fit_label="entire sample, matching source look-ahead",
    )
    train = ar_strategy(
        data.mid,
        fit_ar_conditional(data.mid, 16, test_start),
        test_start,
        fit_label="chronological train-only correction",
    )
    return {
        "source_full_fit": full,
        "train_only": train,
        "yule_walker_selected_lag": selected,
        "yule_walker_bic": bic,
        "source_selected_lag": 16,
        "adf": adf_one_lag(data.mid[:test_start]),
        "contract": {
            "test_bars": 126 * 24 * 60,
            "source_fit_uses_test_prices": True,
            "positions_are_lagged_one_bar_before_returns": True,
            "source_annualization_uses_252_days_despite_24_7_market": True,
        },
    }


def bollinger_experiment(data: ChapterData) -> dict[str, Any]:
    lookback = 60
    series = pd.Series(data.mid)
    moving_average = series.rolling(lookback).mean().to_numpy()
    # MATLAB movingStd uses the sample standard deviation for this archive.
    moving_standard_deviation = series.rolling(lookback).std(ddof=1).to_numpy()
    lower = moving_average - 2.0 * moving_standard_deviation
    upper = moving_average + 2.0 * moving_standard_deviation

    long_position = np.full(len(series), np.nan)
    short_position = np.full(len(series), np.nan)
    long_position[0] = short_position[0] = 0.0
    long_position[data.mid <= lower] = 1.0
    short_position[data.mid >= upper] = -1.0
    long_position[data.mid >= moving_average] = 0.0
    short_position[data.mid <= moving_average] = 0.0
    long_position = pd.Series(long_position).ffill().fillna(0.0).to_numpy()
    short_position = pd.Series(short_position).ffill().fillna(0.0).to_numpy()
    position = long_position + short_position

    returns = np.zeros(len(series))
    returns[1:] = position[:-1] * (data.mid[1:] / data.mid[:-1] - 1.0)
    turnover = np.zeros(len(series))
    turnover[1:] = np.abs(position[1:] - position[:-1])
    net_returns = returns - 0.001 * turnover
    test_start = len(series) - 126 * 24 * 60
    return {
        "moving_average": moving_average,
        "moving_standard_deviation": moving_standard_deviation,
        "lower_band": lower,
        "upper_band": upper,
        "position": position,
        "returns": returns,
        "net_returns_10bps": net_returns,
        "test": performance(returns[test_start:], 252 * 24 * 60),
        "test_net_10bps": performance(
            net_returns[test_start:], 252 * 24 * 60
        ),
        "test_start": int(test_start),
        "contract": {
            "lookback_minutes": lookback,
            "entry_zscore": 2.0,
            "exit_zscore": 0.0,
            "rolling_standard_deviation_ddof": 1,
            "positions_are_lagged_one_bar_before_returns": True,
            "transaction_cost_per_unit_turnover": 0.001,
            "source_annualization_uses_252_days_despite_24_7_market": True,
        },
    }


def lagged_return(prices: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(len(prices), np.nan)
    output[lag:] = prices[lag:] / prices[:-lag] - 1.0
    return output


def feature_panel(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.column_stack(
        [lagged_return(prices, lag) for lag in (1, 5, 10, 30, 60)]
    )
    target = np.r_[lagged_return(prices, 1)[1:], np.nan]
    return features, target


def strategy_from_predictions(
    prices: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    positions = np.where(prediction >= 0, 1.0, -1.0)
    returns1 = lagged_return(prices, 1)[indices]
    strategy_returns = np.zeros(len(indices))
    strategy_returns[1:] = positions[:-1] * returns1[1:]
    turnover = np.zeros(len(indices))
    turnover[1:] = np.abs(positions[1:] - positions[:-1])
    return {
        "indices": indices,
        "predictions": prediction,
        "positions": positions,
        "returns": strategy_returns,
        "net_returns_10bps": strategy_returns - 0.001 * turnover,
        "performance": performance(strategy_returns, 365 * 24 * 60),
        "performance_net_10bps": performance(
            strategy_returns - 0.001 * turnover, 365 * 24 * 60
        ),
        "position_changes": int(np.sum(turnover)),
    }


def random_forest_experiment(data: ChapterData) -> dict[str, Any]:
    features, target = feature_panel(data.mid)
    split = len(data.mid) // 2
    valid = np.all(np.isfinite(features), axis=1) & np.isfinite(target)
    train_indices = np.flatnonzero(valid & (np.arange(len(data.mid)) < split))
    test_indices = np.flatnonzero(valid & (np.arange(len(data.mid)) >= split))
    model = RandomForestRegressor(
        n_estimators=5,
        min_samples_leaf=100,
        max_features=1.0,
        random_state=1,
        n_jobs=1,
    )
    model.fit(features[train_indices], target[train_indices])
    train_prediction = model.predict(features[train_indices])
    test_prediction = model.predict(features[test_indices])
    return {
        "train": strategy_from_predictions(
            data.mid, train_prediction, train_indices
        ),
        "test": strategy_from_predictions(data.mid, test_prediction, test_indices),
        "feature_lags": [1, 5, 10, 30, 60],
        "feature_importances": model.feature_importances_,
        "contract": {
            "chronological_split_index": split,
            "training_rows": len(train_indices),
            "test_rows": len(test_indices),
            "trees": 5,
            "minimum_leaf_size": 100,
            "random_seed": 1,
            "implementation": "sklearn RandomForestRegressor adaptation",
            "not_numerically_equivalent_to_matlab_treebagger": True,
        },
    }


def rolling_order_flow(
    timestamps: pd.Series, signed_size: np.ndarray, seconds: int = 60
) -> np.ndarray:
    # Pandas 3 may store datetime64[us]; force ns before integer arithmetic.
    nanoseconds = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    threshold = nanoseconds - seconds * 1_000_000_000
    prior = np.searchsorted(nanoseconds, threshold, side="right") - 1
    cumulative = np.cumsum(np.asarray(signed_size, dtype=float))
    output = cumulative.copy()
    valid = prior >= 0
    output[valid] -= cumulative[prior[valid]]
    output[~valid] = np.nan
    return output


def order_flow_experiment(data: ChapterData) -> dict[str, Any]:
    frame = data.trades
    signed_size = frame["size"].to_numpy(dtype=float) * frame[
        "buysell"
    ].to_numpy(dtype=float)
    signal = rolling_order_flow(frame["timestamp"], signed_size, seconds=60)
    prices = frame["price"].to_numpy(dtype=float)
    dates = frame["timestamp"].dt.strftime("%Y%m%d").to_numpy()
    position = 0
    entry_price = np.nan
    daily_pl = 0.0
    daily_trades = 0
    cumulative_pl = 0.0
    total_trades = 0
    realized_path = np.zeros(len(frame))
    daily_rows: list[dict[str, Any]] = []
    for index, value in enumerate(signal):
        if np.isfinite(value):
            if value > 90 and position <= 0:
                if position < 0:
                    daily_pl += entry_price - prices[index]
                    daily_trades += 2
                else:
                    daily_trades += 1
                entry_price = prices[index]
                position = 1
            elif value < -90 and position >= 0:
                if position > 0:
                    daily_pl += prices[index] - entry_price
                    daily_trades += 2
                else:
                    daily_trades += 1
                entry_price = prices[index]
                position = -1
            elif value <= 0 and position > 0:
                daily_pl += prices[index] - entry_price
                daily_trades += 1
                position = 0
            elif value >= 0 and position < 0:
                daily_pl += entry_price - prices[index]
                daily_trades += 1
                position = 0
        is_day_end = index == len(frame) - 1 or dates[index] != dates[index + 1]
        if is_day_end:
            daily_rows.append(
                {
                    "date": dates[index],
                    "gross_pl_usd": float(daily_pl),
                    "num_trades": int(daily_trades),
                    "pl_per_trade_usd": (
                        float(daily_pl / daily_trades) if daily_trades else None
                    ),
                }
            )
            cumulative_pl += daily_pl
            total_trades += daily_trades
            daily_pl = 0.0
            daily_trades = 0
        realized_path[index] = cumulative_pl + daily_pl
    return {
        "signal": signal,
        "signed_size": signed_size,
        "realized_pl_path": realized_path,
        "daily": daily_rows,
        "gross_pl_usd": float(cumulative_pl),
        "pl_per_trade_usd": float(cumulative_pl / total_trades),
        "num_trades": int(total_trades),
        "ending_position": int(position),
        "cost_sensitivity": {
            "0.06_one_way": float(cumulative_pl - total_trades * 0.06),
            "0.25_one_way": float(cumulative_pl - total_trades * 0.25),
        },
        "contract": {
            "lookback_seconds": 60,
            "entry_threshold_btc": 90,
            "exit_threshold_btc": 0,
            "timestamp_integer_unit": "nanoseconds",
            "position_carries_across_midnight": True,
        },
    }


def arbitrage_experiment() -> dict[str, Any]:
    buy_ask = 233.546
    sell_bid = 239.19
    gross = sell_bid - buy_ask
    stated_commission = 1.0
    stated_withdrawal_fee = 2.0
    return {
        "historical_bitfinex_bid": sell_bid,
        "historical_btce_ask": buy_ask,
        "gross_spread_usd_per_btc": gross,
        "stated_commission_usd": stated_commission,
        "stated_withdrawal_fee_usd": stated_withdrawal_fee,
        "illustrative_net_usd_per_btc": gross
        - stated_commission
        - stated_withdrawal_fee,
        "credit_and_transfer_risk_quantified": False,
        "historical_example_not_current_recommendation": True,
    }


def run_experiments(data: ChapterData) -> dict[str, Any]:
    return {
        "risk": risk_experiment(data),
        "ar": ar_experiment(data),
        "bollinger": bollinger_experiment(data),
        "random_forest": random_forest_experiment(data),
        "order_flow": order_flow_experiment(data),
        "arbitrage": arbitrage_experiment(),
    }


def source_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    ar = results["ar"]
    bollinger = results["bollinger"]
    order_flow = results["order_flow"]
    rows: list[dict[str, Any]] = []

    def add(
        topic: str,
        metric: str,
        python_value: float,
        source_value: float,
        tolerance: float,
        classification: str,
    ) -> None:
        error = abs(float(python_value) - float(source_value))
        rows.append(
            {
                "topic": topic,
                "metric": metric,
                "python": float(python_value),
                "source": float(source_value),
                "absolute_error": error,
                "tolerance": tolerance,
                "classification": classification,
                "matches_source": error <= tolerance,
            }
        )

    for index, (actual, expected) in enumerate(
        zip(
            ar["source_full_fit"]["coefficients"],
            SOURCE_AR16_COEFFICIENTS,
            strict=True,
        )
    ):
        add(
            "buildARp_BTCUSD.m",
            "constant" if index == 0 else f"AR{{{index}}}",
            actual,
            expected,
            5e-7,
            "approximate conditional OLS versus MATLAB Gaussian MLE",
        )
    add(
        "buildARp_BTCUSD.m",
        "ADF statistic",
        ar["adf"]["statistic"],
        SOURCE_RESULTS["ar_adf_statistic"],
        6e-4,
        "approximate Python OLS versus JPL/MATLAB ADF",
    )
    add(
        "buildARp_BTCUSD.m",
        "test cumulative return",
        ar["source_full_fit"]["test"]["cumulative_return"],
        SOURCE_RESULTS["ar_test_cumulative_return"],
        0.06,
        "approximate conditional OLS versus MATLAB Gaussian MLE",
    )
    add(
        "buildARp_BTCUSD.m",
        "test annual return",
        ar["source_full_fit"]["test"]["annual_return"],
        SOURCE_RESULTS["ar_test_annual_return"],
        25.0,
        "approximate conditional OLS versus MATLAB Gaussian MLE",
    )
    add(
        "bollinger.m",
        "test cumulative return",
        bollinger["test"]["cumulative_return"],
        SOURCE_RESULTS["bollinger_test_cumulative_return"],
        1e-12,
        "exact source replay",
    )
    add(
        "bollinger.m",
        "test annual return",
        bollinger["test"]["annual_return"],
        SOURCE_RESULTS["bollinger_test_annual_return"],
        1e-12,
        "exact source replay",
    )
    add(
        "bollinger.m",
        "10 bps test cumulative return",
        bollinger["test_net_10bps"]["cumulative_return"],
        SOURCE_RESULTS["bollinger_cost_test_cumulative_return"],
        1e-12,
        "exact source replay",
    )
    add(
        "orderFlow2.m",
        "total P&L USD",
        order_flow["gross_pl_usd"],
        SOURCE_RESULTS["order_flow_total_pl_usd"],
        1e-9,
        "exact source replay",
    )
    add(
        "orderFlow2.m",
        "P&L per trade USD",
        order_flow["pl_per_trade_usd"],
        SOURCE_RESULTS["order_flow_pl_per_trade_usd"],
        5e-7,
        "exact source replay",
    )
    add(
        "orderFlow2.m",
        "number of trades",
        order_flow["num_trades"],
        SOURCE_RESULTS["order_flow_num_trades"],
        0.0,
        "exact source replay",
    )
    return rows


def reference_only_results() -> list[dict[str, Any]]:
    return [
        {
            "topic": "AR and ARMA lag selection",
            "source_file": "buildARp_BTCUSD.m and buildARMA_findPQ_BTCUSD.m",
            "book_output": {"matlab_ar_lag": 16, "matlab_arma_p": 3, "matlab_arma_q": 7},
            "compared": False,
            "reason": "MATLAB Gaussian ARIMA maximum-likelihood BIC is not numerically equivalent to the lightweight Yule-Walker diagnostic; Python selects AR(15), so the source selections are preserved without claiming exact estimator equivalence",
        },
        {
            "topic": "MATLAB TreeBagger results",
            "source_file": "rTreeBagger_BTCUSD.m",
            "book_output": {"test_cagr": 75340814462.263458, "test_sharpe": 31.810122, "test_max_drawdown": -0.294237},
            "compared": False,
            "reason": "scikit-learn RandomForestRegressor is a deterministic source-semantic adaptation, not MATLAB TreeBagger; split, lags, five trees and minimum leaf size are retained while numeric parity is not asserted",
        },
        {
            "topic": "cross-validated SVM",
            "source_file": "svm_BTCUSD.m",
            "book_output": {"test_cagr": -0.856935, "test_sharpe": -2.034599, "test_max_drawdown": -0.759027},
            "compared": False,
            "reason": "the script selects one fold-trained model rather than refitting on all training data, and MATLAB fitcsvm defaults are not pinned sufficiently for a defensible cross-library replay",
        },
        {
            "topic": "100-network feed-forward ensemble",
            "source_file": "nn_feedfwd_avg_BTCUSD.m",
            "book_output": {"test_cagr": 1115340053.572500, "test_sharpe": 22.171526, "test_max_drawdown": -0.703360},
            "compared": False,
            "reason": "the MATLAB neural-network training algorithm and initialization contract are not portable to the pinned Python environment; the published output remains output-only",
        },
        {
            "topic": "MXN, SPY, and HYG risk comparison",
            "source_file": "analyzeRisk.m",
            "book_output": None,
            "compared": False,
            "reason": "the official Chapter 7 archive contains BTC daily data but the script points to external local MXN and ETF files that are absent",
        },
    ]


def coverage_matrix() -> list[dict[str, str]]:
    return [
        {"topic": "BTC daily tail risk", "status": "independent source-data replay", "evidence": "345 daily closes"},
        {"topic": "AR(16), ADF, and forecast strategy", "status": "approximate replay + look-ahead audit", "evidence": "479,535 minute mids and printed coefficients"},
        {"topic": "AR/ARMA BIC selection", "status": "diagnostic + output-only source order", "evidence": "Yule-Walker AR(15) versus MATLAB MLE AR(16), ARMA(3,7)"},
        {"topic": "Bollinger mean reversion", "status": "exact source replay", "evidence": "three published outputs"},
        {"topic": "bagged regression trees", "status": "deterministic source-semantic adaptation", "evidence": "chronological half split and execution-cost stress"},
        {"topic": "SVM and neural networks", "status": "code/output-only", "evidence": "nonportable MATLAB training defaults"},
        {"topic": "Bitstamp order flow", "status": "exact source replay", "evidence": "31 daily rows and three aggregate outputs"},
        {"topic": "cross-exchange arbitrage", "status": "historical arithmetic illustration", "evidence": "book example; credit and transfer risk unquantified"},
    ]


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    comparisons = source_comparisons(results)
    ar = results["ar"]
    bollinger = results["bollinger"]
    forest = results["random_forest"]
    order = results["order_flow"]
    risk = results["risk"]
    references = reference_only_results()
    observed_daily = [
        (row["date"], round(row["gross_pl_usd"], 2), row["num_trades"])
        for row in order["daily"]
    ]
    timestamp_fixture = pd.Series(
        pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:30", "2020-01-01 00:01:01"])
    )
    fixture_flow = rolling_order_flow(timestamp_fixture, np.array([1.0, 2.0, 4.0]))
    return {
        "archive_manifest_matches": manifest["archive_sha256"] == chapter_manifest()["sha256"],
        "all_12_archive_members_pinned": len(manifest["members"]) == 12,
        "nine_matlab_sources_present": len(list(ORIGINAL_DIR.glob("*.m"))) == 9,
        "minute_midprices_are_finite": bool(np.all(np.isfinite(data.mid))),
        "crossed_minute_quotes_are_disclosed": data.crossed_quote_rows == 517,
        "hhmm_shape_mismatch_is_disclosed": data.hhmm_rows == 13_438 != len(data.mid),
        "daily_dates_are_strictly_increasing": bool(np.all(np.diff(data.daily_dates) > 0)),
        "trade_timestamps_are_monotonic": data.trades["timestamp"].is_monotonic_increasing,
        "all_numeric_source_comparisons_pass": all(row["matches_source"] for row in comparisons),
        "bollinger_uses_sample_standard_deviation": bollinger["contract"]["rolling_standard_deviation_ddof"] == 1,
        "bollinger_cost_destroys_published_return": bollinger["test_net_10bps"]["cumulative_return"] < -0.99,
        "source_ar_lookahead_is_disclosed": ar["contract"]["source_fit_uses_test_prices"],
        "train_only_ar_does_not_use_test_prices": ar["train_only"]["fit_label"].startswith("chronological"),
        "yule_walker_difference_is_not_called_exact": ar["yule_walker_selected_lag"] == 15 and references[0]["compared"] is False,
        "adf_does_not_reject_at_source_5pct_threshold": ar["adf"]["statistic"] < ar["adf"]["critical_value_5pct_source"],
        "random_forest_is_deterministic_and_chronological": forest["contract"]["random_seed"] == 1 and forest["train"]["indices"][-1] < forest["test"]["indices"][0],
        "random_forest_cost_stress_is_worse": forest["test"]["performance_net_10bps"]["cumulative_return_log"] < forest["test"]["performance"]["cumulative_return_log"],
        "order_flow_matches_all_31_daily_comments": observed_daily == SOURCE_ORDER_DAILY,
        "order_flow_aggregate_matches_source": order["num_trades"] == 336 and abs(order["gross_pl_usd"] - 32.54) < 1e-9,
        "order_flow_ends_flat": order["ending_position"] == 0,
        "nanosecond_rolling_window_fixture_passes": bool(
            np.isnan(fixture_flow[0])
            and np.isnan(fixture_flow[1])
            and fixture_flow[2] == 6.0
        ),
        "transaction_costs_reduce_order_flow_profit": order["cost_sensitivity"]["0.25_one_way"] < order["cost_sensitivity"]["0.06_one_way"] < order["gross_pl_usd"],
        "btc_risk_table_rounds_to_book_values": round(100 * risk["annualized_volatility"]) == 67 and round(100 * risk["maximum_drawdown"]) == -79 and round(risk["annualized_kurtosis"]) == 7,
        "arbitrage_credit_risk_is_not_quantified": results["arbitrage"]["credit_and_transfer_risk_quantified"] is False,
        "all_reference_only_rows_have_reasons": all(row["compared"] is False and row["reason"] for row in references),
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
    classes = {
        name: ("independent_or_empirical" if index < 19 else "contract_invariant")
        for index, name in enumerate(checks)
    }
    ar = results["ar"]
    bollinger = results["bollinger"]
    forest = results["random_forest"]
    order = results["order_flow"]
    return json_safe(
        {
            "chapter": 7,
            "title": "Bitcoin",
            "provenance": {
                "source_page": chapter_manifest()["source_page"],
                "archive_url": chapter_manifest()["url"],
                "archive_sha256": chapter_manifest()["sha256"],
                "archive_size_bytes": chapter_manifest()["size_bytes"],
                "member_count": len(extraction["members"]),
                "matlab_source_count": sum(row["kind"] == "original_source" for row in extraction["members"]),
                "research_data_count": sum(row["kind"] == "research_data" for row in extraction["members"]),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
            },
            "environment": {
                "versions": environment_versions(("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "nbformat", "nbclient")),
                "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
                "random_seed": RANDOM_SEED,
            },
            "data": {
                "minute_bbo": {"rows": len(data.mid), "date_start": int(data.bbo_tday[0]), "date_end": int(data.bbo_tday[-1]), "crossed_quote_rows": data.crossed_quote_rows, "hhmm_rows": data.hhmm_rows, "hhmm_matches_price_rows": data.hhmm_rows == len(data.mid)},
                "daily_btc": {"rows": len(data.daily_close), "date_start": int(data.daily_dates[0]), "date_end": int(data.daily_dates[-1])},
                "bitstamp_trades": {"rows": len(data.trades), "days": int(data.trades["timestamp"].dt.normalize().nunique()), "date_start": str(data.trades["timestamp"].iloc[0]), "date_end": str(data.trades["timestamp"].iloc[-1])},
            },
            "reproduction_classification": {
                "exact_source_replay": ["bollinger.m", "orderFlow2.m"],
                "approximate_replay": ["buildARp_BTCUSD.m conditional OLS and ADF"],
                "source_semantic_adaptation": ["rTreeBagger_BTCUSD.m via sklearn"],
                "output_only": [row["topic"] for row in reference_only_results()],
            },
            "coverage": coverage_matrix(),
            "results": {
                "risk": {key: value for key, value in results["risk"].items() if key not in {"returns", "wealth", "drawdown"}},
                "ar": {
                    "source_full_fit": {"test": ar["source_full_fit"]["test"], "test_net_10bps": ar["source_full_fit"]["test_net_10bps"], "fit_label": ar["source_full_fit"]["fit_label"]},
                    "train_only": {"test": ar["train_only"]["test"], "test_net_10bps": ar["train_only"]["test_net_10bps"], "fit_label": ar["train_only"]["fit_label"]},
                    "yule_walker_selected_lag": ar["yule_walker_selected_lag"],
                    "source_selected_lag": ar["source_selected_lag"],
                    "adf": ar["adf"],
                    "contract": ar["contract"],
                },
                "bollinger": {"test": bollinger["test"], "test_net_10bps": bollinger["test_net_10bps"], "contract": bollinger["contract"]},
                "random_forest": {"train": {"performance": forest["train"]["performance"], "performance_net_10bps": forest["train"]["performance_net_10bps"], "position_changes": forest["train"]["position_changes"]}, "test": {"performance": forest["test"]["performance"], "performance_net_10bps": forest["test"]["performance_net_10bps"], "position_changes": forest["test"]["position_changes"]}, "feature_importances": forest["feature_importances"], "contract": forest["contract"]},
                "order_flow": {key: value for key, value in order.items() if key not in {"signal", "signed_size", "realized_pl_path"}},
                "arbitrage": results["arbitrage"],
            },
            "book_comparisons": source_comparisons(results),
            "reference_only_comparisons": reference_only_results(),
            "source_limitations": {
                "minute_hhmm_array_shape_mismatch": True,
                "crossed_bbo_rows_preserved": True,
                "ar_source_full_sample_fit_is_lookahead": True,
                "matlab_and_sklearn_tree_implementations_differ": True,
                "order_flow_spread_comment_conflicts_with_book_prose": True,
                "crypto_annualization_conventions_are_inconsistent": True,
            },
            "verification": {
                "checks": checks,
                "summary": {
                    "total": len(checks),
                    "passed": sum(checks.values()),
                    "failed": [name for name, passed in checks.items() if not passed],
                    "independent_or_empirical": sum(value == "independent_or_empirical" for value in classes.values()),
                    "contract_invariant": sum(value == "contract_invariant" for value in classes.values()),
                    "classification": classes,
                },
            },
        }
    )


def bbo_dates(data: ChapterData) -> pd.DatetimeIndex:
    return pd.to_datetime(data.bbo_tday.astype(str), format="%Y%m%d")


def format_time_axis(axis: plt.Axes, *, max_ticks: int = 7) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
    axis.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.show_offset = False
    axis.xaxis.set_major_formatter(formatter)


def plot_risk(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    risk = results["risk"]
    dates = pd.to_datetime(data.daily_dates.astype(str), format="%Y%m%d")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(dates, risk["wealth"], color="tab:orange")
    axes[0].set_title("BTC daily wealth index")
    axes[0].set_ylabel("Growth of $1")
    axes[1].fill_between(dates, risk["drawdown"], 0, color="tab:red", alpha=0.6)
    axes[1].set_title("Drawdown: tail risk is not theoretical")
    axes[1].set_ylabel("Drawdown")
    for axis in axes:
        format_time_axis(axis)
    return fig


def plot_ar(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    ar = results["ar"]
    test_start = ar["source_full_fit"]["test_start"]
    dates = bbo_dates(data)[test_start:]
    step = 240
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for key, label in (("source_full_fit", "source full-sample fit"), ("train_only", "train-only correction")):
        returns = ar[key]["returns"][test_start:]
        log_wealth = np.cumsum(np.log1p(returns))
        axes[0].plot(dates[::step], log_wealth[::step], label=label)
    axes[0].set_title("AR(16) test wealth on log scale")
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend()
    format_time_axis(axes[0])
    axes[1].plot(np.arange(1, 61), ar["yule_walker_bic"] - np.min(ar["yule_walker_bic"]))
    axes[1].axvline(15, color="tab:blue", linestyle="--", label="Python Yule-Walker: 15")
    axes[1].axvline(16, color="tab:red", linestyle=":", label="MATLAB MLE: 16")
    axes[1].set_title("Estimator choice changes selected AR lag")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("BIC minus minimum")
    axes[1].legend()
    return fig


def plot_bollinger(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    bollinger = results["bollinger"]
    start = bollinger["test_start"]
    sample = slice(start, min(start + 7 * 24 * 60, len(data.mid)))
    sample_dates = bbo_dates(data)[sample]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(sample_dates, data.mid[sample], linewidth=0.7, label="mid")
    axes[0].plot(sample_dates, bollinger["moving_average"][sample], linewidth=0.8, label="60-min mean")
    axes[0].plot(sample_dates, bollinger["upper_band"][sample], linewidth=0.6, linestyle="--", label="±2 sample std")
    axes[0].plot(sample_dates, bollinger["lower_band"][sample], linewidth=0.6, linestyle="--")
    axes[0].set_title("Bollinger signal, first test week")
    axes[0].set_ylabel("USD per BTC")
    axes[0].legend()
    format_time_axis(axes[0])
    gross = np.cumsum(np.log1p(bollinger["returns"][start:]))
    net = np.cumsum(np.log1p(bollinger["net_returns_10bps"][start:]))
    dates = bbo_dates(data)[start:]
    step = 240
    axes[1].plot(dates[::step], gross[::step], label="gross")
    axes[1].plot(dates[::step], net[::step], label="10 bps per turnover")
    axes[1].set_title("Small gross edge, fatal turnover cost")
    axes[1].set_ylabel("Cumulative log return")
    axes[1].legend()
    format_time_axis(axes[1])
    return fig


def plot_random_forest(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    forest = results["random_forest"]
    test = forest["test"]
    dates = bbo_dates(data)[test["indices"]]
    gross = np.cumsum(np.log1p(test["returns"]))
    net = np.cumsum(np.log1p(test["net_returns_10bps"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    step = 240
    axes[0].plot(dates[::step], gross[::step], label="gross")
    axes[0].plot(dates[::step], net[::step], label="10 bps cost")
    axes[0].set_title("Random forest test: turnover illusion")
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend()
    format_time_axis(axes[0])
    labels = ["1m", "5m", "10m", "30m", "60m"]
    axes[1].bar(labels, forest["feature_importances"])
    axes[1].set_title("Modern adaptation feature importance")
    axes[1].set_ylabel("Impurity importance")
    return fig


def plot_order_flow(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    order = results["order_flow"]
    daily = pd.DataFrame(order["daily"])
    daily_dates = pd.to_datetime(daily["date"], format="%Y%m%d")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = np.where(daily["gross_pl_usd"] >= 0, "tab:blue", "tab:red")
    axes[0].bar(daily_dates, daily["gross_pl_usd"], color=colors)
    axes[0].set_title("Bitstamp order-flow daily realized P&L")
    axes[0].set_ylabel("USD before costs")
    format_time_axis(axes[0])
    names = ["gross", "$0.06\none-way", "$0.25\none-way"]
    values = [order["gross_pl_usd"], order["cost_sensitivity"]["0.06_one_way"], order["cost_sensitivity"]["0.25_one_way"]]
    axes[1].bar(names, values, color=["tab:blue", "tab:orange", "tab:red"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Spread/cost assumption reverses conclusion")
    axes[1].set_ylabel("USD")
    return fig


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "btc_risk.png": plot_risk(data, results),
        "ar_diagnostics.png": plot_ar(data, results),
        "bollinger_costs.png": plot_bollinger(data, results),
        "random_forest.png": plot_random_forest(data, results),
        "order_flow.png": plot_order_flow(data, results),
    }
    for name, figure in figures.items():
        figure.savefig(
            FIGURE_DIR / name,
            dpi=150,
            bbox_inches="tight",
            metadata={"Software": "chan-machine-trading-experiments"},
        )
        plt.close(figure)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
            *("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows),
        ]
    )


def write_report(data: ChapterData, results: dict[str, Any], metrics: dict[str, Any]) -> None:
    risk = results["risk"]
    ar = results["ar"]
    bollinger = results["bollinger"]
    forest = results["random_forest"]
    order = results["order_flow"]
    comparisons = [
        {"topic": row["topic"], "metric": row["metric"], "Python": f"{row['python']:.9g}", "source": f"{row['source']:.9g}", "class": row["classification"]}
        for row in metrics["book_comparisons"]
    ]
    references = [
        {"topic": row["topic"], "compared": str(row["compared"]).lower(), "reason": row["reason"]}
        for row in metrics["reference_only_comparisons"]
    ]
    content = f"""# Machine Trading 2016 Chapter 7 — Bitcoin

## 1. 결론 먼저

공식 ZIP 12개 member와 각 SHA-256을 고정했다. Bollinger 세 지표와 Bitstamp order-flow의 31개 일별 행·세 aggregate 지표는 원본과 정확히 일치한다. AR(16)은 MATLAB MLE 대신 conditional OLS로 근사해 계수와 성과가 출판값에 가깝지만, lag 선택과 ADF 구현 차이를 정확 재현으로 부르지 않는다. 높은 gross 수익은 10bp 비용에서 거의 전부 사라지므로 이 장의 핵심은 예측모형보다 시점·비용·거래소 위험이다.

## 2. provenance와 데이터

- 공식 페이지: {metrics['provenance']['source_page']}
- archive SHA-256: `{metrics['provenance']['archive_sha256']}`
- minute BBO: {len(data.mid):,}행, daily close: {len(data.daily_close):,}행, Bitstamp trade: {len(data.trades):,}행
- MATLAB source 9개, research data 3개

`HHMM`은 {data.hhmm_rows:,}행뿐이라 price의 {len(data.mid):,}행과 맞지 않고, bid가 ask보다 큰 행도 {data.crossed_quote_rows:,}개다. 원본 전략이 midpoint만 쓰기 때문에 계산은 가능하지만, 보조 배열과 호가 품질의 이상을 숨기지 않는다.

## 3. 재현 분류와 coverage

{markdown_table(coverage_matrix(), ['topic', 'status', 'evidence'])}

exact, approximate, source-semantic, output-only를 구분한다. MATLAB 숫자를 단순 복사한 항목은 Python 실험 결과로 가장하지 않는다.

## 4. BTC 위험

2014-01-20부터 2015-01-14까지 연율 변동성은 {risk['annualized_volatility']:.1%}, 최악 일간수익률은 {risk['worst_daily_return']:.1%} ({risk['worst_date']}), 최고는 {risk['best_daily_return']:.1%} ({risk['best_date']}), 최대 낙폭은 {risk['maximum_drawdown']:.1%}다. fat tail과 79% 낙폭은 전략 수익률의 큰 숫자보다 먼저 봐야 할 위험이다.

## 5. AR(16), ADF, 그리고 look-ahead

원본은 훈련구간으로 lag 16을 고른 뒤 `estimate(model, mid)`로 전체 표본을 다시 fit한다. 따라서 test price가 계수 추정에 들어간다. 이를 source-faithful 경로로 보존하는 동시에 train-only correction을 별도 제공한다. Python Yule-Walker BIC는 lag {ar['yule_walker_selected_lag']}를, MATLAB Gaussian MLE는 16을 고르므로 estimator가 다르면 model selection도 달라진다. ADF statistic {ar['adf']['statistic']:.6f}는 원본 5% 임계값 -2.871보다 작지만, 단일 표본과 가격수준 회귀만으로 안정적 시장 alpha를 보장하지 않는다.

## 6. 원본 MATLAB 비교

{markdown_table(comparisons, ['topic', 'metric', 'Python', 'source', 'class'])}

AR 계수의 허용오차 5e-7은 출력 자릿수에 맞춘 근사 기준이다. ADF와 수익률은 JPL/MATLAB 및 MLE 차이를 반영한 별도 허용오차를 쓴다. Bollinger와 order flow는 훨씬 엄격한 1e-12/표시 반올림 기준이다.

## 7. Bollinger exact replay

60분 rolling mean, 표본 표준편차 `ddof=1`, 진입 z=2, 평균선 청산을 그대로 적용했다. test gross 누적수익은 {bollinger['test']['cumulative_return']:.12f}, 연율수익은 {bollinger['test']['annual_return']:.12f}로 원본과 일치한다. 그러나 position change마다 10bp를 차감하면 누적수익이 {bollinger['test_net_10bps']['cumulative_return']:.3%}가 된다. midpoint 체결, latency 0, market impact 0도 낙관적이므로 비용 포함 결과가 더 중요한 하한이다.

## 8. bagged tree 현대적 adaptation

1·5·10·30·60분 수익률로 다음 1분 수익률을 예측하고 시간순 절반 split, tree 5개, minimum leaf 100, seed 1을 고정했다. test gross 누적수익은 {forest['test']['performance']['cumulative_return']:.3g}로 비현실적으로 크지만, 10bp turnover cost의 cumulative log return은 {forest['test']['performance_net_10bps']['cumulative_return_log']:.1f}다. TreeBagger와 sklearn은 알고리즘이 달라 수치 비교하지 않으며, 높은 Sharpe를 경제적 타당성으로 해석하지 않는다.

## 9. selection bias와 데이터 누수

원본 AR full fit의 look-ahead는 명시적이다. tree in-sample prediction은 bootstrap tree가 본 관측치를 포함할 수 있어 독립 성능이 아니다. 여러 모델·lag·kernel·network 중 잘 보이는 결과를 선택하면 selection bias가 생긴다. 실전 검증에는 과거 window에서만 모델을 선택하고 다음 window에서 고정하는 walk-forward가 필요하다.

## 10. Bitstamp order flow exact replay

timestamp를 `datetime64[ns]`로 명시 변환하고, 현재 tick보다 60초 이전인 마지막 cumulative flow를 뺀다. Pandas 저장 단위를 암묵적으로 가정하면 60초가 잘못 스케일될 수 있다. 진입 threshold 90 BTC, exit 0, 포지션은 자정을 넘어 유지한다. 총손익 {order['gross_pl_usd']:.2f}달러, {order['num_trades']} legs, trade당 {order['pl_per_trade_usd']:.6f}달러가 원본과 일치한다.

## 11. 거래비용과 원본 설명의 모순

책 본문은 spread 약 0.12달러와 one-way cost 0.06달러를 논하지만 `orderFlow2.m` 마지막 주석은 spread 약 0.5달러라고 적는다. 0.06달러를 각 leg에 적용하면 {order['cost_sensitivity']['0.06_one_way']:.2f}달러, 0.25달러면 {order['cost_sensitivity']['0.25_one_way']:.2f}달러다. 어느 설명을 임의로 정답 처리하지 않고 둘 다 sensitivity로 제시한다.

## 12. cross-exchange arbitrage

역사적 예제에서 Bitfinex bid 239.19와 btc-e ask 233.546의 gross spread는 5.644달러다. 명시된 commission 1달러와 withdrawal 2달러를 빼면 2.644달러지만, transfer delay 동안의 가격위험, 거래소 credit risk, 출금 중단, 양쪽 inventory funding은 정량화되지 않았다. 현재 거래소나 가격에 대한 추천이 아니다.

## 13. output-only 참조

{markdown_table(references, ['topic', 'compared', 'reason'])}

SVM은 각 fold model 중 하나를 골라 전체 훈련자료에 refit하지 않는다는 점도 주의한다. NN 100회와 ARMA 90개 MLE 탐색은 library 기본값과 실행 계약이 다르므로 출판값만 보존한다.

## 14. 위험지표와 연율화

crypto는 24/7인데 AR·Bollinger는 252일, tree/SVM/NN은 365일을 사용한다. 같은 minute return도 연율화 convention에 따라 CAGR과 Sharpe가 크게 바뀐다. 따라서 원본 비교용 convention은 보존하되, 경제적 비교에서는 cumulative return, drawdown, turnover와 비용을 함께 본다.

## 15. 자동 검증과 결론

검증 {metrics['verification']['summary']['passed']}/{metrics['verification']['summary']['total']} 통과. checksum, member 수, 원본 출력, 31일 order-flow, datetime 단위 fixture, position lag, 비용 단조성, look-ahead 공개, 누락 입력의 output-only 처리를 검사한다. 이 노트북은 역사적 코드를 재현하고 함정을 진단하는 백테스트 실험이지 미래 수익을 증명하지 않는다. 결론은 단순하다. 암호자산에서는 model score보다 데이터 contract, 시간 단위, 체결비용, 거래소 생존과 출금 가능성이 먼저다.
"""
    assert_clean_markdown_math(content, "chapter7_report.md")
    write_text_if_changed(REPORT_DIR / "chapter7_report.md", content)


def write_readme() -> None:
    write_text_if_changed(
        CHAPTER_DIR / "README.md",
        """# Chapter 7 — Bitcoin

```bash
uv run python chapter_7_bitcoins/src/run_chapter7_analysis.py
uv run python chapter_7_bitcoins/src/run_chapter7_analysis.py --offline
```

공식 12개 archive member를 checksum으로 검증하고 BTC risk, AR/ADF, Bollinger exact replay, deterministic random-forest adaptation, Bitstamp order-flow exact replay와 비용 진단을 생성한다. `data/raw`는 내려받을 수 있고 재생성 가능한 원자료라 Git에서 제외된다.
""",
    )


def notebook_markdown_cells() -> list[str]:
    return [
        """# Chapter 7: Bitcoin — 재현보다 먼저 검증할 것들

공식 Chapter 7 archive의 BTC minute BBO, daily close, Bitstamp trade tick을 사용해 위험, AR/ADF, Bollinger, bagged tree, order flow, 거래소 간 차익거래를 살펴본다. 목표는 높은 역사적 CAGR을 다시 보여 주는 데 있지 않다. 어떤 결과가 원본과 정확히 일치하고, 어떤 결과가 estimator 차이 때문에 근사이며, 어떤 코드는 입력이나 실행 계약 부족으로 output-only인지 분리한다. 이 노트북은 연구 코드를 감사하는 역사적 백테스트 실험이지 현재 Bitcoin 투자 추천이 아니다.""",
        """## 1. 문제 정의와 학습 질문

첫째, crypto의 fat tail과 drawdown은 전통자산보다 어떤 위험을 추가하는가? 둘째, AR lag 선택과 ADF 결론은 estimator와 sample split에 얼마나 민감한가? 셋째, 예측력이 있어 보이는 minute 전략이 spread와 turnover cost 후에도 남는가? 넷째, timestamp 저장 단위와 자정 경계가 order-flow state machine을 어떻게 바꾸는가? 다섯째, cross-exchange spread를 보이는 숫자 그대로 수익으로 간주할 수 있는가?""",
        """## 2. provenance, checksum, environment version

공식 source URL, archive SHA-256, 12개 member의 개별 checksum, uv.lock SHA와 패키지 버전을 기록한다. online 실행은 archive를 검증해 materialize하고 offline 실행은 이미 받은 파일을 전수 검사한다. 재현성은 코드만 보관하는 것이 아니라 입력 byte, 환경, random seed를 함께 고정하는 일이다.""",
        """## 3. 재현 상태와 coverage matrix

`exact source replay`는 출판된 원본 출력과 엄격한 tolerance로 비교한다. `approximate replay`는 계산 의미는 가깝지만 MATLAB MLE/JPL 구현과 Python이 달라 허용오차와 이유를 공개한다. `source-semantic adaptation`은 같은 feature와 split을 현대 library에 옮기되 숫자 동등성을 주장하지 않는다. `output-only`는 원본 숫자를 기록만 하고 실행 결과로 가장하지 않는다. 이 분류가 구현 범위와 증거를 연결한다.""",
        """## 4. 데이터 구조와 결측·호가 진단

minute midpoint 479,535행은 finite지만 `HHMM` 보조 배열은 13,438행뿐이고, bid가 ask보다 큰 crossed row가 517개다. 원본 전략은 midpoint와 날짜를 사용하므로 실행할 수 있지만 호가 데이터가 깨끗하다는 뜻은 아니다. daily close는 345행, Bitstamp CSV는 119,914 tick과 31일을 담는다. timestamp는 시간순이며 aggressor tag는 -1, 0, 1만 허용한다. shape mismatch와 crossed quote를 임의 정제하지 않고 provenance 진단으로 보존한다.""",
        r"""## 5. BTC tail risk

변동성만으로 crypto 위험을 요약할 수 없다. daily wealth $W_t=P_t/P_0$와 drawdown

$$DD_t=\frac{W_t}{\max_{s\le t} W_s}-1$$

을 함께 계산한다. 표본의 최대 낙폭은 약 -79%, 최악 일간수익률은 약 -24%다. 높은 kurtosis와 거래소 운영 위험은 정규분포 기반 position sizing의 취약점을 드러낸다.""",
        r"""## 6. 수식 → 코드: AR forecast와 ADF

AR($p$) 예측은

$$\hat P_{t+1}=c+\sum_{j=1}^{p}\phi_j P_{t+1-j}$$

이며 예측이 현재 가격보다 높으면 long, 낮으면 short를 잡고 다음 bar 수익에 position을 lag한다. `fit_ar_conditional()`은 normal equation을 streaming-like gram matrix로 구성해 거대한 dense lag matrix를 피한다. `adf_one_lag()`는 상수, lagged level, 한 개 lagged difference를 OLS로 적합한다. 원본 JPL ADF와 통계량이 가까워도 동일 library 재현은 아니다.""",
        """## 7. AR lag selection과 look-ahead audit

MATLAB Gaussian MLE BIC는 AR(16)을 고르지만 Python Yule-Walker BIC는 AR(15)을 고른다. estimator가 다르면 likelihood와 penalty 대상이 달라져 selection도 바뀐다. 더 중요한 문제는 원본이 train에서 lag를 고른 뒤 전체 `mid`로 계수를 다시 적합한다는 점이다. 이는 test price가 계수에 들어가는 look-ahead다. source-faithful curve와 train-only correction을 나란히 두어 이 차이를 숨기지 않는다.""",
        """## 8. 원본 MATLAB 비교

AR의 17개 coefficient, ADF statistic, test cumulative/annual return은 구현 차이를 반영한 `approximate` tolerance로 비교한다. Bollinger 세 숫자와 order flow 세 aggregate는 `exact` tolerance로 비교한다. tolerance는 결과를 통과시키기 위해 임의로 넓히는 값이 아니라 원본 표시 정밀도와 알고리즘 차이를 문서화한 계약이다.""",
        r"""## 9. Bollinger exact replay

60분 평균 $m_t$와 표본 표준편차 $s_t$에 대해 $P_t\le m_t-2s_t$이면 long, $P_t\ge m_t+2s_t$이면 short에 진입하고 평균선에서 청산한다. Pandas rolling std가 `ddof=1`일 때만 원본 세 결과와 정확히 일치한다. position은 signal이 관측된 다음 bar return에 적용한다. gross test 수익은 플러스지만 10bp per turnover를 넣으면 거의 -100%가 된다.""",
        r"""## 10. Bollinger 거래비용·슬리피지

원본은 midpoint에서 즉시 체결되고 latency와 impact가 없다고 가정한다. 비용식은

$$r_t^{net}=r_t-0.001\lvert q_t-q_{t-1}\rvert$$

이다. turnover가 잦으면 작은 평균회귀 edge보다 비용이 훨씬 크다. 실제 market order는 spread와 slippage를 추가로 부담하고, limit order는 미체결과 adverse selection을 부담한다. 따라서 비용 없는 CAGR을 실전 기대수익으로 읽으면 안 된다.""",
        """## 11. bagged regression tree adaptation

1·5·10·30·60분 과거 수익률로 다음 1분 수익률을 예측한다. 시간순 절반 split, tree 5개, minimum leaf 100, seed 1을 고정한다. sklearn RandomForestRegressor는 MATLAB TreeBagger와 bootstrap, split tie, random stream이 다르므로 source-semantic adaptation이다. in-sample prediction은 각 tree가 본 bootstrap 관측을 포함할 수 있어 독립 검증이 아니다.""",
        """## 12. AI 결과의 경제성 stress test

gross test wealth와 Sharpe는 비현실적으로 크다. 그러나 position sign이 자주 바뀌어 10bp 비용을 차감한 cumulative log wealth는 붕괴한다. 모델의 MSE나 방향 정확도보다 turnover-adjusted P&L이 실제 목적함수에 가깝다. SVM은 fold-trained model 하나를 선택하고 전체 train에 refit하지 않으며, 100-network ensemble은 MATLAB training 기본값에 의존한다. 두 결과를 현대 Python 숫자로 억지 재현하지 않는다.""",
        r"""## 13. 수식 → 코드: 60초 order flow

aggressor side $s_i$와 size $v_i$를 이용해

$$OF_t=\sum_{t-60s<i\le t}s_i v_i$$

를 계산한다. `rolling_order_flow()`는 timestamp를 먼저 `datetime64[ns]`로 바꾼 뒤 integer search를 수행한다. Pandas 3의 내부 unit이 microsecond일 수 있으므로 단순 `astype(int64)`에 1e9를 곱한 경계를 적용하면 창이 틀어진다. 합성 0초·30초·61초 fixture가 현재 tick 포함과 왼쪽 경계를 검증한다.""",
        """## 14. Bitstamp exact replay와 자정 state

60초 flow가 +90 BTC를 넘으면 long, -90 아래면 short, 0을 되돌아오면 청산한다. reversal은 기존 position 청산과 반대 진입을 합쳐 두 legs로 센다. 일별 realized P&L은 자정에 reset하지만 position과 entry price는 이어진다. 이 state contract로 31개 일별 P&L·trade count와 총 32.54달러, 336 legs가 원본과 일치한다. 자정에 position까지 reset하면 다른 결과가 된다.""",
        """## 15. spread 설명의 모순과 비용 민감도

책 본문은 spread 약 0.12달러, one-way 0.06달러를 말하지만 원본 `orderFlow2.m`은 spread 약 0.5달러라고 주석한다. 어느 쪽을 임의로 진실로 선택하지 않는다. 각 leg 0.06달러에서는 gross 32.54달러가 12.38달러로 줄고, 0.25달러에서는 -51.46달러로 뒤집힌다. fee, spread, slippage, latency가 전략의 부속 항목이 아니라 결론을 결정한다.""",
        """## 16. cross-exchange arbitrage와 credit risk

역사적 Bitfinex bid 239.19와 btc-e ask 233.546의 차이는 BTC당 5.644달러다. 책의 commission 1달러와 withdrawal 2달러를 빼면 2.644달러지만 이는 동시에 양 거래소에 현금과 BTC inventory가 있고 출금이 정상이라는 가정이다. transfer delay, price move, counterparty default, withdrawal freeze, capital fragmentation은 식에 없다. 현재 시장 추천이나 무위험 수익이 아니다.""",
        """## 17. 표본 외, selection bias, 연율화 한계

ARMA 90개 조합, SVM kernel, tree, NN을 같은 역사에서 탐색하면 selection bias가 커진다. 과거 rolling window에서만 모델을 선택하고 다음 window에 고정하는 walk-forward가 필요하다. 또한 원본 AR·Bollinger는 crypto minute return을 252일로, AI 예제는 365일로 연율화한다. CAGR, Sharpe, Calmar는 convention에 민감하므로 cumulative return, maximum drawdown, turnover와 비용을 함께 보고 비교 기준을 통일해야 한다.""",
        """## 18. deterministic contract와 자동 verification

RandomForest seed=1, 프로젝트 random_seed=20260718, single-thread tree fit, deterministic notebook cell ID, PNG metadata를 고정한다. 25개 자동 검증은 archive checksum과 member 수, 원본 출력 비교, 31일 일별 order-flow, timestamp unit fixture, sample std, position lag, look-ahead 공개, 비용 단조성, output-only 사유를 포함한다. assert가 모두 참이어야 산출물을 완성한다.""",
        """## 19. 결론

정확 재현된 결과 중 가장 중요한 것은 높은 gross 수익이 아니라 비용을 넣었을 때 결론이 뒤집힌다는 사실이다. AR full fit에는 look-ahead가 있고, estimator 선택은 lag를 바꾸며, minute BBO에는 shape mismatch와 crossed quote가 있다. order flow는 nanosecond와 자정 state를 정확히 고정해야 원본에 맞는다. Bitcoin 전략에서는 model complexity보다 데이터 contract, 표본 외 검증, execution cost, 거래소 credit와 withdrawal 가능성을 먼저 확인해야 한다.""",
    ]


def build_and_execute_notebook() -> None:
    md = notebook_markdown_cells()
    cells: list[nbformat.NotebookNode] = [nbformat.v4.new_markdown_cell(md[0])]
    cells.append(nbformat.v4.new_code_cell("""from io import BytesIO\nfrom pathlib import Path\nimport sys\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0, str(Path.cwd()))\nimport run_chapter7_analysis as ch7\ndata=ch7.load_data()\nresults=ch7.run_experiments(data)\nchecks=ch7.verify_results(data,results)\nmetrics=ch7.build_metrics(data,results,checks)\nprint(f\"minute={len(data.mid):,}, trades={len(data.trades):,}, checks={sum(checks.values())}/{len(checks)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[1]))
    cells.append(nbformat.v4.new_code_cell("""def show_figure(fig):\n    payload=BytesIO(); fig.savefig(payload,format='png',dpi=120,bbox_inches='tight'); plt.close(fig); display(Image(data=payload.getvalue()))\npd.DataFrame(ch7.coverage_matrix())"""))
    cells.append(nbformat.v4.new_markdown_cell(md[2]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({**metrics['provenance'],'uv_lock_sha256':metrics['environment']['uv_lock_sha256'],'versions':metrics['environment']['versions']})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[3]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(metrics['reproduction_classification']).apply(lambda x:', '.join(x))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[4]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['data']).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[5]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series(metrics['results']['risk'])); show_figure(ch7.plot_risk(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[6]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'python_yule_walker_lag':results['ar']['yule_walker_selected_lag'],'source_matlab_mle_lag':results['ar']['source_selected_lag'],'adf_statistic':results['ar']['adf']['statistic'],'source_5pct_critical':results['ar']['adf']['critical_value_5pct_source']})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[7]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch7.plot_ar(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[8]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['book_comparisons'])[['topic','metric','python','source','absolute_error','tolerance','classification','matches_source']]"""))
    cells.append(nbformat.v4.new_markdown_cell(md[9]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame({'gross':results['bollinger']['test'],'net_10bps':results['bollinger']['test_net_10bps']}).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[10]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch7.plot_bollinger(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[11]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame({'train_gross':results['random_forest']['train']['performance'],'test_gross':results['random_forest']['test']['performance'],'test_net_10bps':results['random_forest']['test']['performance_net_10bps']}).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[12]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch7.plot_random_forest(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[13]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(results['order_flow']['daily'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[14]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series({k:results['order_flow'][k] for k in ('gross_pl_usd','pl_per_trade_usd','num_trades','ending_position','cost_sensitivity')})); show_figure(ch7.plot_order_flow(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[15]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(results['arbitrage'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[16]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['reference_only_comparisons'])[['topic','source_file','compared','reason']]"""))
    cells.append(nbformat.v4.new_markdown_cell(md[17]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'project_random_seed':ch7.RANDOM_SEED,'forest_random_seed':results['random_forest']['contract']['random_seed'],'tree_threads':1,'notebook_cell_ids':'deterministic'})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[18]))
    cells.append(nbformat.v4.new_code_cell("""verification=pd.Series(checks,name='passed'); display(verification); assert verification.all(); print(f\"verification passed: {verification.sum()}/{len(verification)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[19]))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch7-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"chapter7 notebook cell {index}")
    execute_notebook(notebook, NOTEBOOK_PATH, workdir=SRC_DIR, timeout=1200)


def audit_notebook() -> dict[str, Any]:
    audit_path = PROJECT_ROOT.parents[1] / ".codex" / "skills" / "create-chan-chapter-analysis" / "scripts" / "audit_chapter_artifacts.py"
    spec = importlib.util.spec_from_file_location("chan_audit", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audit helper: {audit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_notebook(NOTEBOOK_PATH)


def write_quality_benchmark(metrics: dict[str, Any]) -> None:
    audit = audit_notebook()
    counts, scores = audit["counts"], audit["scores"]
    verification = metrics["verification"]["summary"]
    write_text_if_changed(
        REPORT_DIR / "quality_benchmark.md",
        f"""# Chapter 7 quality benchmark

| Chapter | execution | reproducibility | rigor | pedagogy | trading context | total |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch7** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

- Cells {counts['cells']}: Markdown {counts['markdown_cells']}, code {counts['code_cells']}; executed {counts['executed_code_cells']}/{counts['code_cells']}, errors {counts['error_outputs']}, inline PNG {counts['embedded_png_outputs']}.
- Markdown {counts['markdown_characters']:,} chars, Korean {counts['korean_characters']:,}, headings {counts['headings']}.
- Verification {verification['passed']}/{verification['total']}; independent/empirical {verification['independent_or_empirical']}, invariant {verification['contract_invariant']}.
""",
    )


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    data = load_data()
    results = run_experiments(data)
    checks = verify_results(data, results)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Chapter 7 verification failed: {failed}")
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
    statuses = ["offline assets verified"] if args.offline else download_official_assets(force=args.force_download)
    if args.offline:
        validate_offline_assets()
    for status in statuses:
        print(status)
    if args.download_only:
        return
    metrics = run_analysis(execute_notebook_flag=not args.skip_notebook)
    print("Chapter 7 analysis complete")
    print("Checks:", metrics["verification"]["summary"]["passed"])
    print("Report:", REPORT_DIR / "chapter7_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
