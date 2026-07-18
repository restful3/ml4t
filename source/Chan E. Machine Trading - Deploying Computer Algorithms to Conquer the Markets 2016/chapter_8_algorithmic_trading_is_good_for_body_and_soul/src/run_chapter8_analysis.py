#!/usr/bin/env python3
"""Build reproducible Chapter 8 strategy-lifecycle experiments."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import tomllib
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


SRC_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SRC_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
REPORT_DIR = SRC_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = SRC_DIR / "chapter8_full_report.ipynb"
CHAPTER_TEXT_PATH = (
    CHAPTER_DIR
    / "08_chapter_8_algorithmic_trading_is_good_for_body_and_soul.md"
)
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


SOURCE_ACTIVE_DAY_RESULTS = {
    "annual_return": 0.23860906665831516,
    "sharpe": 1.106654196870293,
    "maximum_drawdown": -0.21831770065726142,
}


@dataclass(frozen=True)
class ChapterData:
    dates: np.ndarray
    gld: np.ndarray
    uso: np.ndarray
    stocks: tuple[str, ...]
    source_sha256: str
    cross_book_source_present: bool
    cross_book_source_sha256: str | None
    cross_book_data_present: bool
    cross_book_data_sha256: str | None
    cross_book_values_match: bool | None


def chapter8_manifest() -> dict[str, Any]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["tool"]["book3"]["chapter_8"]


def chapter5_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=5)


def cross_book_source_path() -> Path:
    return WORKSPACE_ROOT / chapter8_manifest()["cross_book_source_path"]


def data_path() -> Path:
    return PROJECT_ROOT / chapter8_manifest()["data_path"]


def cross_book_data_path() -> Path:
    return WORKSPACE_ROOT / chapter8_manifest()["cross_book_data_path"]


def download_official_assets(force: bool = False) -> list[str]:
    """Chapter 8 has no bundle; reuse the verified official Chapter 5 panel."""
    manifest = chapter5_manifest()
    payload = download_verified_archive(
        manifest,
        user_agent="chan-machine-trading-experiments/0.1",
        timeout=120,
    )
    return materialize_chapter_archive(
        PROJECT_ROOT, manifest, payload, force=force
    )


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter5_manifest())
    manifest = chapter8_manifest()
    if sha256_file(data_path()) != manifest["data_sha256"]:
        raise ValueError("Chapter 8 reused data checksum mismatch")
    if data_path().stat().st_size != manifest["data_size_bytes"]:
        raise ValueError("Chapter 8 reused data size mismatch")
    source = cross_book_source_path()
    if source.exists() and sha256_file(source) != manifest["cross_book_source_sha256"]:
        raise ValueError("Cross-book Bollinger source checksum mismatch")
    cross_data = cross_book_data_path()
    if cross_data.exists() and sha256_file(cross_data) != manifest["cross_book_data_sha256"]:
        raise ValueError("Cross-book GLD-USO data checksum mismatch")


def load_data() -> ChapterData:
    validate_offline_assets()
    payload = loadmat(data_path(), squeeze_me=True)
    stocks = tuple(str(item) for item in np.asarray(payload["stocks"]).tolist())
    close = np.asarray(payload["cl"], dtype=float)
    dates = np.asarray(payload["tday"], dtype=int)
    gld = close[:, stocks.index("GLD")]
    uso = close[:, stocks.index("USO")]
    good = (dates >= 20060426) & np.isfinite(gld) & np.isfinite(uso)
    source = cross_book_source_path()
    cross_data = cross_book_data_path()
    values_match: bool | None = None
    if cross_data.exists():
        reference = pd.read_csv(cross_data)
        panel = pd.DataFrame({"Date": dates[good], "GLD": gld[good], "USO": uso[good]})
        overlap = panel.merge(reference, on="Date", suffixes=("_book3", "_chan2013"))
        values_match = bool(
            len(overlap) == len(reference) == 1_500
            and np.array_equal(overlap["GLD_book3"], overlap["GLD_chan2013"])
            and np.array_equal(overlap["USO_book3"], overlap["USO_chan2013"])
        )
    return ChapterData(
        dates=dates[good],
        gld=gld[good],
        uso=uso[good],
        stocks=stocks,
        source_sha256=sha256_file(data_path()),
        cross_book_source_present=source.exists(),
        cross_book_source_sha256=sha256_file(source) if source.exists() else None,
        cross_book_data_present=cross_data.exists(),
        cross_book_data_sha256=sha256_file(cross_data) if cross_data.exists() else None,
        cross_book_values_match=values_match,
    )


def performance(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, Any]:
    values = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    if np.any(values <= -1):
        raise ValueError("A period return is at or below -100%")
    wealth = np.cumprod(1.0 + values)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    duration = longest = 0
    for value in drawdown:
        duration = duration + 1 if value < 0 else 0
        longest = max(longest, duration)
    standard_deviation = np.std(values, ddof=1)
    return {
        "cumulative_return": float(wealth[-1] - 1.0),
        "annual_return": float(wealth[-1] ** (periods_per_year / len(values)) - 1.0),
        "sharpe": (
            float(np.sqrt(periods_per_year) * np.mean(values) / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        "maximum_drawdown": float(np.min(drawdown)),
        "drawdown_duration": int(longest),
        "periods": int(len(values)),
    }


def rolling_hedge_ratio(gld: np.ndarray, uso: np.ndarray, lookback: int = 20) -> np.ndarray:
    ratio = np.full(len(gld), np.nan)
    for end in range(lookback, len(gld)):
        x = np.column_stack(
            [np.ones(lookback), gld[end - lookback : end]]
        )
        ratio[end - 1] = np.linalg.lstsq(
            x, uso[end - lookback : end], rcond=None
        )[0][1]
    return ratio


def positions_from_zscore(
    zscore: np.ndarray,
    *,
    source_long_exit_bug: bool = False,
) -> np.ndarray:
    zscore = np.asarray(zscore, dtype=float)
    long_position = np.full(len(zscore), np.nan)
    short_position = np.full(len(zscore), np.nan)
    long_position[0] = short_position[0] = 0.0
    long_position[zscore < -1.0] = 1.0
    long_position[zscore > (-1.0 if source_long_exit_bug else 0.0)] = 0.0
    short_position[zscore > 1.0] = -1.0
    short_position[zscore < 0.0] = 0.0
    long_position = pd.Series(long_position).ffill().fillna(0.0).to_numpy()
    short_position = pd.Series(short_position).ffill().fillna(0.0).to_numpy()
    return long_position + short_position


def bollinger_strategy(
    dates: np.ndarray,
    gld: np.ndarray,
    uso: np.ndarray,
    *,
    transaction_cost: float = 0.001,
    source_long_exit_bug: bool = False,
) -> dict[str, Any]:
    lookback = 20
    hedge_ratio = rolling_hedge_ratio(gld, uso, lookback)
    spread = uso - hedge_ratio * gld
    series = pd.Series(spread)
    moving_average = series.rolling(lookback).mean().to_numpy()
    moving_std = series.rolling(lookback).std(ddof=1).to_numpy()
    zscore = (spread - moving_average) / moving_std
    units = positions_from_zscore(
        zscore, source_long_exit_bug=source_long_exit_bug
    )
    positions = np.column_stack(
        [-units * hedge_ratio * gld, units * uso]
    )
    prices = np.column_stack([gld, uso])
    asset_returns = np.zeros_like(prices)
    asset_returns[1:] = prices[1:] / prices[:-1] - 1.0
    prior_positions = np.vstack([np.zeros(2), positions[:-1]])
    pnl = np.nansum(prior_positions * asset_returns, axis=1)
    capital = np.nansum(np.abs(prior_positions), axis=1)
    gross_returns = np.divide(
        pnl, capital, out=np.zeros(len(gld)), where=capital > 0
    )
    active_returns = np.divide(
        pnl,
        capital,
        out=np.full(len(gld), np.nan),
        where=capital > 0,
    )
    turnover = np.r_[0.0, np.abs(np.diff(units))]
    net_returns = gross_returns - transaction_cost * turnover
    active = active_returns[np.isfinite(active_returns)]
    source_active_day = performance(active)
    return {
        "dates": np.asarray(dates, dtype=int),
        "gld": np.asarray(gld, dtype=float),
        "uso": np.asarray(uso, dtype=float),
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "moving_average": moving_average,
        "moving_std": moving_std,
        "zscore": zscore,
        "units": units,
        "positions": positions,
        "gross_returns": gross_returns,
        "net_returns_10bps": net_returns,
        "turnover": turnover,
        "source_active_day": source_active_day,
        "calendar_gross": performance(gross_returns),
        "calendar_net_10bps": performance(net_returns),
        "contract": {
            "lookback_days": lookback,
            "entry_zscore": 1.0,
            "exit_zscore": 0.0,
            "rolling_standard_deviation_ddof": 1,
            "positions_are_lagged_one_day": True,
            "source_active_day_clock_excludes_flat_days": True,
            "source_long_exit_bug_replayed": source_long_exit_bug,
            "transaction_cost_per_unit_change": transaction_cost,
        },
    }


def gld_uso_experiment(data: ChapterData) -> dict[str, Any]:
    original = (data.dates >= 20060426) & (data.dates <= 20120409)
    extension = (data.dates >= 20120410) & (data.dates <= 20151125)
    corrected = bollinger_strategy(
        data.dates[original], data.gld[original], data.uso[original]
    )
    source_bug = bollinger_strategy(
        data.dates[original],
        data.gld[original],
        data.uso[original],
        source_long_exit_bug=True,
    )
    later = bollinger_strategy(
        data.dates[extension], data.gld[extension], data.uso[extension]
    )
    full = bollinger_strategy(data.dates, data.gld, data.uso)
    return {
        "original_period": corrected,
        "source_code_long_exit_bug": source_bug,
        "post_book_extension": later,
        "full_period": full,
        "classification": "cross-book empirical replay using the value-identical official Book 3 Chapter 5 panel",
    }


def strategy_lifecycle_experiment() -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    days = 1_000
    count = 10
    common = rng.normal(0.0, 0.006, size=(days, 1))
    idiosyncratic = rng.normal(0.0, 0.008, size=(days, count))
    returns = 0.25 * common + np.sqrt(1.0 - 0.25**2) * idiosyncratic
    returns += 0.00025
    returns[500:, 0] -= 0.0018
    returns[700:, 1] -= 0.0012

    active = np.ones((days, count), dtype=bool)
    for strategy in range(1, count):
        active[: strategy * 15, strategy] = False

    equal_weights = active / active.sum(axis=1, keepdims=True)
    equal_returns = np.sum(equal_weights * returns, axis=1)
    concentrated_returns = returns[:, 0]

    adaptive_weights = np.zeros_like(returns)
    individual_wealth = np.ones(count)
    individual_peaks = np.ones(count)
    for day in range(days):
        drawdown = individual_wealth / individual_peaks - 1.0
        budget = np.clip(1.0 + drawdown / 0.12, 0.0, 1.0) * active[day]
        if budget.sum() == 0:
            budget = active[day].astype(float)
        adaptive_weights[day] = budget / budget.sum()
        individual_wealth *= 1.0 + returns[day]
        individual_peaks = np.maximum(individual_peaks, individual_wealth)
    adaptive_returns = np.sum(adaptive_weights * returns, axis=1)
    return {
        "strategy_returns": returns,
        "active_mask": active,
        "equal_weights": equal_weights,
        "adaptive_weights": adaptive_weights,
        "concentrated_returns": concentrated_returns,
        "equal_returns": equal_returns,
        "adaptive_returns": adaptive_returns,
        "performance": {
            "concentrated_dying_strategy": performance(concentrated_returns),
            "equal_weight_live_pool": performance(equal_returns),
            "drawdown_budget_live_pool": performance(adaptive_returns),
        },
        "contract": {
            "random_seed": RANDOM_SEED,
            "strategies": count,
            "days": days,
            "strategy_0_alpha_break_day": 500,
            "strategy_1_alpha_break_day": 700,
            "maximum_drawdown_budget": 0.12,
            "weights_use_information_through_previous_day": True,
            "conceptual_simulation_not_backtest": True,
        },
    }


def managed_account_granularity_experiment() -> dict[str, Any]:
    account_sizes = np.array([25_000.0, 100_000.0, 500_000.0, 2_000_000.0])
    margin_per_contract = np.array(
        [12_000, 9_000, 17_000, 7_500, 25_000, 14_000, 8_000, 11_000, 20_000, 6_000],
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    units_matrix = []
    for size in account_sizes:
        target = np.full(len(margin_per_contract), size / len(margin_per_contract))
        units = np.floor(target / margin_per_contract).astype(int)
        allocated = units * margin_per_contract
        realized_weights = allocated / size
        units_matrix.append(units)
        rows.append(
            {
                "account_size_usd": size,
                "strategies_with_nonzero_allocation": int(np.count_nonzero(units)),
                "capital_utilization": float(np.sum(allocated) / size),
                "l1_weight_error": float(
                    np.sum(np.abs(realized_weights - 0.1))
                ),
            }
        )
    return {
        "rows": rows,
        "margin_per_contract": margin_per_contract,
        "integer_units": np.asarray(units_matrix),
        "target_weight_per_strategy": 0.1,
        "contract": {
            "illustrative_margin_not_market_quote": True,
            "integer_contract_constraint": True,
            "conceptual_capacity_experiment_not_legal_or_investment_advice": True,
        },
    }


def run_experiments(data: ChapterData) -> dict[str, Any]:
    return {
        "gld_uso": gld_uso_experiment(data),
        "strategy_lifecycle": strategy_lifecycle_experiment(),
        "managed_account_granularity": managed_account_granularity_experiment(),
    }


def reference_only_results() -> list[dict[str, Any]]:
    return [
        {
            "topic": "GLD-GDX fixed-spread strategy death",
            "book_output": {"spread": "GLD - 2.21*GDX", "test_cumulative_return_direction": "negative"},
            "compared": False,
            "reason": "Chapter 8 publishes a figure and directional statement but no Chapter 8 archive or the May 2006-May 2013 GDX panel; the available official Book 3 ETF panel does not contain GDX",
        },
        {
            "topic": "health, autonomy, and productivity claims",
            "book_output": None,
            "compared": False,
            "reason": "these are narrative and cited lifestyle claims, not trading calculations that can be reproduced from the bundled market data",
        },
        {
            "topic": "managed-account registration thresholds and service providers",
            "book_output": None,
            "compared": False,
            "reason": "the chapter describes a 2016 legal and business environment; this notebook provides no current legal, regulatory, or provider-availability advice",
        },
    ]


def coverage_matrix() -> list[dict[str, str]]:
    return [
        {"topic": "GLD-USO 20-day Bollinger", "status": "cross-book empirical replay", "evidence": "official Book 3 Ch5 panel; values identical to Chan 2013 CSV over 1,500 days"},
        {"topic": "active-day annualization", "status": "source-metric audit", "evidence": "23.86% source clock versus 17.76% calendar clock"},
        {"topic": "post-2012 strategy shelf life", "status": "temporal extension", "evidence": "915 later observations through 2015-11-25"},
        {"topic": "source long-exit typo", "status": "code-semantic sensitivity", "evidence": "published Python uses -1 on the long exit while prose says exit z=0"},
        {"topic": "strategy births/deaths and diversification", "status": "deterministic conceptual simulation", "evidence": "10 strategies, two alpha breaks, seed pinned"},
        {"topic": "managed-account integer allocation", "status": "deterministic capacity illustration", "evidence": "four account sizes and ten margin requirements"},
        {"topic": "GLD-GDX negative test", "status": "output-only", "evidence": "required 2006-2013 GDX panel absent"},
        {"topic": "health, service, and regulation", "status": "narrative/output-only", "evidence": "not a market-data calculation"},
    ]


def source_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    active = results["gld_uso"]["original_period"]["source_active_day"]
    rows = []
    for metric, expected in SOURCE_ACTIVE_DAY_RESULTS.items():
        actual = active[metric]
        rows.append(
            {
                "topic": "Chan 2013 GLD-USO Bollinger cross-book replay",
                "metric": metric,
                "python": actual,
                "source": expected,
                "absolute_error": abs(actual - expected),
                "tolerance": 1e-12,
                "classification": "exact replay of the corrected chapter-analysis source on value-identical data",
                "matches_source": abs(actual - expected) <= 1e-12,
            }
        )
    return rows


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    manifest = chapter8_manifest()
    gld_uso = results["gld_uso"]
    original = gld_uso["original_period"]
    extension = gld_uso["post_book_extension"]
    lifecycle = results["strategy_lifecycle"]
    granularity = results["managed_account_granularity"]
    references = reference_only_results()
    return {
        "reused_official_data_checksum_matches": data.source_sha256 == manifest["data_sha256"],
        "chapter_8_archive_absence_is_explicit": manifest["official_archive_available"] is False,
        "gld_and_uso_are_in_official_panel": "GLD" in data.stocks and "USO" in data.stocks,
        "gdx_absence_is_disclosed": "GDX" not in data.stocks and references[0]["compared"] is False,
        "dates_are_strictly_increasing": bool(np.all(np.diff(data.dates) > 0)),
        "all_used_prices_are_finite_positive": bool(np.all(np.isfinite(data.gld)) and np.all(np.isfinite(data.uso)) and np.all(data.gld > 0) and np.all(data.uso > 0)),
        "original_cross_book_period_has_1500_rows": len(original["dates"]) == 1_500,
        "cross_book_data_values_match_when_available": data.cross_book_values_match in (True, None),
        "post_book_extension_has_915_rows": len(extension["dates"]) == 915,
        "source_active_day_outputs_match": all(row["matches_source"] for row in source_comparisons(results)),
        "source_active_day_clock_is_disclosed": original["contract"]["source_active_day_clock_excludes_flat_days"],
        "calendar_clock_is_more_conservative": original["calendar_gross"]["annual_return"] < original["source_active_day"]["annual_return"],
        "positions_are_lagged_one_day": original["contract"]["positions_are_lagged_one_day"],
        "transaction_costs_reduce_original_return": original["calendar_net_10bps"]["annual_return"] < original["calendar_gross"]["annual_return"],
        "extension_remains_positive_before_cost": extension["calendar_gross"]["annual_return"] > 0,
        "extension_remains_positive_after_10bps": extension["calendar_net_10bps"]["annual_return"] > 0,
        "source_long_exit_typo_changes_results": gld_uso["source_code_long_exit_bug"]["calendar_gross"]["cumulative_return"] != original["calendar_gross"]["cumulative_return"],
        "lifecycle_seed_is_pinned": lifecycle["contract"]["random_seed"] == RANDOM_SEED,
        "lifecycle_weights_are_lagged": lifecycle["contract"]["weights_use_information_through_previous_day"],
        "lifecycle_simulation_is_not_called_backtest": lifecycle["contract"]["conceptual_simulation_not_backtest"],
        "strategy_pool_beats_dying_concentration": lifecycle["performance"]["equal_weight_live_pool"]["maximum_drawdown"] > lifecycle["performance"]["concentrated_dying_strategy"]["maximum_drawdown"],
        "small_accounts_have_fewer_active_strategies": granularity["rows"][0]["strategies_with_nonzero_allocation"] < granularity["rows"][-1]["strategies_with_nonzero_allocation"],
        "granularity_is_not_presented_as_market_quote": granularity["contract"]["illustrative_margin_not_market_quote"],
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


def public_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "period_start": int(strategy["dates"][0]),
        "period_end": int(strategy["dates"][-1]),
        "rows": len(strategy["dates"]),
        "source_active_day": strategy["source_active_day"],
        "calendar_gross": strategy["calendar_gross"],
        "calendar_net_10bps": strategy["calendar_net_10bps"],
        "position_changes": int(np.count_nonzero(strategy["turnover"])),
        "contract": strategy["contract"],
    }


def build_metrics(data: ChapterData, results: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    gld_uso = results["gld_uso"]
    lifecycle = results["strategy_lifecycle"]
    classes = {
        name: ("independent_or_empirical" if index < 18 else "contract_invariant")
        for index, name in enumerate(checks)
    }
    return json_safe(
        {
            "chapter": 8,
            "title": "Algorithmic Trading Is Good for Your Body and Soul",
            "provenance": {
                "source_page": chapter8_manifest()["source_page"],
                "official_chapter_8_archive_available": False,
                "reused_official_archive_chapter": 5,
                "reused_data_path": chapter8_manifest()["data_path"],
                "reused_data_sha256": data.source_sha256,
                "reused_data_size_bytes": data_path().stat().st_size,
                "chapter_text_sha256": sha256_file(CHAPTER_TEXT_PATH),
                "cross_book_source_path": chapter8_manifest()["cross_book_source_path"],
                "cross_book_source_present": data.cross_book_source_present,
                "cross_book_source_sha256": data.cross_book_source_sha256,
                "cross_book_data_path": chapter8_manifest()["cross_book_data_path"],
                "cross_book_data_present": data.cross_book_data_present,
                "cross_book_data_sha256": data.cross_book_data_sha256,
                "cross_book_values_match": data.cross_book_values_match,
            },
            "environment": {
                "versions": environment_versions(("numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbclient")),
                "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
                "random_seed": RANDOM_SEED,
            },
            "data": {
                "rows": len(data.dates),
                "date_start": int(data.dates[0]),
                "date_end": int(data.dates[-1]),
                "official_panel_assets": len(data.stocks),
                "gld_missing_used_period": int(np.sum(~np.isfinite(data.gld))),
                "uso_missing_used_period": int(np.sum(~np.isfinite(data.uso))),
                "gdx_available": "GDX" in data.stocks,
            },
            "reproduction_classification": {
                "exact_cross_book_replay": ["corrected GLD-USO active-day metrics"],
                "empirical_extension": ["GLD-USO 2012-2015"],
                "conceptual_simulation": ["strategy lifecycle", "managed-account granularity"],
                "output_only": [row["topic"] for row in reference_only_results()],
            },
            "coverage": coverage_matrix(),
            "results": {
                "gld_uso": {
                    "original_period": public_strategy(gld_uso["original_period"]),
                    "source_code_long_exit_bug": public_strategy(gld_uso["source_code_long_exit_bug"]),
                    "post_book_extension": public_strategy(gld_uso["post_book_extension"]),
                    "full_period": public_strategy(gld_uso["full_period"]),
                    "classification": gld_uso["classification"],
                },
                "strategy_lifecycle": {"performance": lifecycle["performance"], "contract": lifecycle["contract"]},
                "managed_account_granularity": results["managed_account_granularity"],
            },
            "book_comparisons": source_comparisons(results),
            "reference_only_comparisons": reference_only_results(),
            "source_limitations": {
                "no_official_chapter_8_code_or_data_archive": True,
                "gld_gdx_required_panel_absent": True,
                "cross_book_source_active_day_clock_is_biased": True,
                "cross_book_python_long_exit_disagrees_with_exit_zscore_variable": True,
                "post_2015_performance_not_tested": True,
                "legal_and_provider_statements_are_historical": True,
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


def to_dates(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(values).astype(str), format="%Y%m%d")


def format_time_axis(axis: plt.Axes, *, max_ticks: int = 7) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
    axis.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.show_offset = False
    axis.xaxis.set_major_formatter(formatter)


def wealth(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))


def plot_data(data: ChapterData) -> plt.Figure:
    dates = to_dates(data.dates)
    normalized_gld = data.gld / data.gld[0]
    normalized_uso = data.uso / data.uso[0]
    rolling_correlation = pd.Series(data.gld).pct_change().rolling(60).corr(
        pd.Series(data.uso).pct_change()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(dates, normalized_gld, label="GLD")
    axes[0].plot(dates, normalized_uso, label="USO")
    axes[0].set_title("Official Book 3 panel: normalized prices")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[1].plot(dates, rolling_correlation, color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("60-day return correlation is not stable")
    axes[1].set_ylabel("Correlation")
    for axis in axes:
        format_time_axis(axis)
    return fig


def plot_bollinger(results: dict[str, Any]) -> plt.Figure:
    strategy = results["gld_uso"]["original_period"]
    dates = to_dates(strategy["dates"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
    axes[0].plot(dates, strategy["spread"], linewidth=0.8, label="USO - beta*GLD")
    axes[0].plot(dates, strategy["moving_average"], linewidth=0.8, label="20-day mean")
    axes[0].plot(dates, strategy["moving_average"] + strategy["moving_std"], linestyle="--", linewidth=0.6, label="±1 sample std")
    axes[0].plot(dates, strategy["moving_average"] - strategy["moving_std"], linestyle="--", linewidth=0.6)
    axes[0].set_title("Dynamic GLD-USO spread and Bollinger bands")
    axes[0].legend()
    axes[1].plot(dates, strategy["zscore"], linewidth=0.7, label="z-score")
    axes[1].step(dates, strategy["units"], where="post", linewidth=0.8, label="position units")
    axes[1].axhline(1, color="tab:red", linestyle="--", linewidth=0.7)
    axes[1].axhline(-1, color="tab:red", linestyle="--", linewidth=0.7)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_title("Signal and next-day position contract")
    axes[1].legend()
    format_time_axis(axes[1])
    return fig


def plot_strategy_shelf_life(results: dict[str, Any]) -> plt.Figure:
    gld_uso = results["gld_uso"]
    original = gld_uso["original_period"]
    extension = gld_uso["post_book_extension"]
    source_bug = gld_uso["source_code_long_exit_bug"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    dates = to_dates(original["dates"])
    axes[0].plot(dates, wealth(original["gross_returns"]), label="corrected gross")
    axes[0].plot(dates, wealth(original["net_returns_10bps"]), label="corrected net 10 bps")
    axes[0].plot(dates, wealth(source_bug["gross_returns"]), label="source long-exit typo", alpha=0.8)
    axes[0].set_title("2006-2012: metric and code semantics matter")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    format_time_axis(axes[0])
    later_dates = to_dates(extension["dates"])
    axes[1].plot(later_dates, wealth(extension["gross_returns"]), label="gross")
    axes[1].plot(later_dates, wealth(extension["net_returns_10bps"]), label="net 10 bps")
    axes[1].set_title("2012-2015 temporal extension")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()
    format_time_axis(axes[1])
    return fig


def plot_strategy_lifecycle(results: dict[str, Any]) -> plt.Figure:
    lifecycle = results["strategy_lifecycle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(wealth(lifecycle["concentrated_returns"]), label="one dying strategy")
    axes[0].plot(wealth(lifecycle["equal_returns"]), label="equal-weight live pool")
    axes[0].plot(wealth(lifecycle["adaptive_returns"]), label="lagged drawdown budget")
    axes[0].axvline(500, color="tab:red", linestyle="--", label="strategy 0 alpha break")
    axes[0].set_title("Conceptual strategy births and deaths")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_xlabel("Synthetic day")
    axes[0].legend()
    axes[1].plot(lifecycle["adaptive_weights"][:, 0], label="strategy 0")
    axes[1].plot(lifecycle["adaptive_weights"][:, 1], label="strategy 1")
    axes[1].plot(lifecycle["adaptive_weights"][:, 2], label="strategy 2")
    axes[1].set_title("Weights use only prior-day drawdown")
    axes[1].set_ylabel("Portfolio weight")
    axes[1].set_xlabel("Synthetic day")
    axes[1].legend()
    return fig


def plot_account_granularity(results: dict[str, Any]) -> plt.Figure:
    rows = pd.DataFrame(results["managed_account_granularity"]["rows"])
    labels = [f"${value/1000:.0f}k" if value < 1_000_000 else f"${value/1_000_000:.0f}m" for value in rows["account_size_usd"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(labels, rows["strategies_with_nonzero_allocation"])
    axes[0].axhline(10, color="tab:red", linestyle="--", label="target strategies")
    axes[0].set_title("Integer contracts can erase diversification")
    axes[0].set_ylabel("Strategies with nonzero allocation")
    axes[0].legend()
    axes[1].bar(labels, rows["l1_weight_error"], color="tab:orange")
    axes[1].set_title("Equal-weight target allocation error")
    axes[1].set_ylabel("L1 weight error")
    return fig


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "data_diagnostics.png": plot_data(data),
        "gld_uso_bollinger.png": plot_bollinger(results),
        "strategy_shelf_life.png": plot_strategy_shelf_life(results),
        "strategy_lifecycle.png": plot_strategy_lifecycle(results),
        "account_granularity.png": plot_account_granularity(results),
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
    gld_uso = results["gld_uso"]
    original = gld_uso["original_period"]
    typo = gld_uso["source_code_long_exit_bug"]
    extension = gld_uso["post_book_extension"]
    lifecycle = results["strategy_lifecycle"]
    granularity = results["managed_account_granularity"]
    comparisons = [
        {"metric": row["metric"], "Python": f"{row['python']:.12f}", "source": f"{row['source']:.12f}", "error": f"{row['absolute_error']:.1e}"}
        for row in metrics["book_comparisons"]
    ]
    references = [
        {"topic": row["topic"], "compared": str(row["compared"]).lower(), "reason": row["reason"]}
        for row in metrics["reference_only_comparisons"]
    ]
    accounts = [
        {"account": f"${row['account_size_usd']:,.0f}", "active strategies": row["strategies_with_nonzero_allocation"], "utilization": f"{row['capital_utilization']:.1%}", "L1 error": f"{row['l1_weight_error']:.3f}"}
        for row in granularity["rows"]
    ]
    content = rf"""# Machine Trading 2016 Chapter 8 — Algorithmic Trading Is Good for Your Body and Soul

## 1. 결론 먼저

Chapter 8에는 독립적인 공식 code/data ZIP이 없다. 따라서 숫자를 새로 발명하지 않고, 본문이 재검토하는 GLD-USO 전략을 공식 Chapter 5 ETF panel에서 재현했다. 이 panel의 2006-04-26~2012-04-09 GLD·USO 값 1,500행은 로컬 Chan 2013 CSV와 전부 동일하다. 원본 분석의 active-day APR 23.86%는 정확히 재현되지만, flat day를 포함한 calendar APR은 17.76%, 10bp 비용 후 14.43%다. 전략은 2012-2015에도 플러스였으나 더 낮아졌고, 이것은 2015 이후 생존을 의미하지 않는다.

## 2. provenance와 archive 부재

- 공식 Book 3 page: {metrics['provenance']['source_page']}
- Chapter 8 전용 archive: 없음 (`official_archive_available=false`)
- 재사용 data: Chapter 5 official archive의 `{metrics['provenance']['reused_data_path']}`
- data SHA-256: `{metrics['provenance']['reused_data_sha256']}`
- Chan 2013 source SHA-256: `{metrics['provenance']['cross_book_source_sha256']}`

전용 bundle이 없다는 사실은 누락이 아니라 장의 성격이다. lifestyle, service, business structure와 전략 lifecycle이 중심이며, 시장 데이터 계산은 GLD-GDX와 GLD-USO 예시뿐이다.

## 3. coverage와 재현 분류

{markdown_table(coverage_matrix(), ['topic', 'status', 'evidence'])}

GLD-USO는 cross-book empirical replay, 2012-2015는 temporal extension, 전략 pool과 account granularity는 conceptual simulation이다. GLD-GDX와 규제·건강 주장은 output-only다.

## 4. 데이터 진단

사용구간은 {len(data.dates):,}일({data.dates[0]}~{data.dates[-1]}), asset 26개 중 GLD와 USO를 사용한다. 사용가격은 finite·positive이고 날짜는 엄격히 증가한다. GDX는 panel에 없으므로 Chapter 8 Figure 8.1의 2006-2013 negative test를 재현했다고 주장하지 않는다. GLD-USO 60일 상관도 시간에 따라 바뀌어, 경제적 story만으로 spread의 안정성을 가정할 수 없다.

## 5. GLD-USO 수식 → 코드

최근 20일 OLS로 $USO_t=\alpha_t+\beta_t GLD_t+\epsilon_t$를 적합하고 $S_t=USO_t-\beta_t GLD_t$를 구성한다. rolling sample mean/std의 z-score가 -1 아래면 long spread, +1 위면 short spread, 0에서 청산한다. signal은 오늘 종가로 계산하고 position은 다음 날 수익에 적용한다.

## 6. cross-book exact 비교

{markdown_table(comparisons, ['metric', 'Python', 'source', 'error'])}

여기서 exact는 Machine Trading Ch8 자체 output이 아니라, Chan 2013 data와 수정된 chapter-analysis implementation의 active-day 결과에 대한 정확 일치다. 분류 범위를 좁게 쓴 이유다.

## 7. active-day clock 편향

원본 분석은 포지션이 없는 날의 return을 NaN으로 두고 삭제한 뒤 252일 연율화한다. active {original['source_active_day']['periods']}일만 쓰면 APR {original['source_active_day']['annual_return']:.2%}, Sharpe {original['source_active_day']['sharpe']:.3f}다. 전체 {original['calendar_gross']['periods']} calendar rows를 넣으면 APR {original['calendar_gross']['annual_return']:.2%}, Sharpe {original['calendar_gross']['sharpe']:.3f}로 낮아진다. 투자자 자본의 시간은 flat day에도 흐르므로 calendar clock을 경제적 기본값으로 삼는다.

## 8. source long-exit typo

Chan 2013 `bollinger.py`는 `exitZscore=0`을 선언하지만 long exit에 `zScore > -entryZscore`를 사용한다. 수정된 chapter analyzer는 long/short 모두 0에서 청산한다. typo 경로의 calendar cumulative return {typo['calendar_gross']['cumulative_return']:.2%}와 수정 경로 {original['calendar_gross']['cumulative_return']:.2%}가 달라, 변수 이름이 아니라 실제 분기식을 감사해야 한다.

## 9. 거래비용과 표본 외

unit change마다 10bp를 부과하면 2006-2012 calendar APR은 {original['calendar_net_10bps']['annual_return']:.2%}, max drawdown은 {original['calendar_net_10bps']['maximum_drawdown']:.2%}다. spread, slippage, borrow fee와 market impact는 별도이므로 여전히 낙관적이다. position lag는 지켰지만 hedge ratio·lookback·threshold가 같은 역사에서 선택되었다면 selection bias도 남는다.

## 10. shelf-life temporal extension

새로 시작한 2012-04-10~2015-11-25 915일 window에서 gross calendar APR {extension['calendar_gross']['annual_return']:.2%}, 10bp 후 {extension['calendar_net_10bps']['annual_return']:.2%}로 플러스다. 원래 기간보다 약해졌고 2015 이후 데이터는 없다. “계속 수익성”이라는 책의 당시 서술을 2026년 현재도 유효하다고 확대하지 않는다.

## 11. 전략 births/deaths 개념 실험

seed {RANDOM_SEED}로 10개 synthetic strategy를 만들고 strategy 0은 day 500, strategy 1은 day 700 이후 alpha를 악화시켰다. 한 죽어가는 전략 집중의 max drawdown은 {lifecycle['performance']['concentrated_dying_strategy']['maximum_drawdown']:.1%}, live pool equal-weight는 {lifecycle['performance']['equal_weight_live_pool']['maximum_drawdown']:.1%}다. 이는 diversification 원리를 설명하는 simulation이지 시장 backtest가 아니다.

## 12. drawdown budget와 look-ahead

각 strategy의 전일까지 wealth와 peak만으로 12% drawdown budget을 계산하고 오늘 weight를 정한다. 미래 return은 weight에 들어가지 않는다. adaptive pool의 성과가 좋은 것은 특정 seed의 교육용 결과이며 CPPI가 미래 성과를 보장한다는 증거가 아니다. 핵심은 죽은 전략의 레버리지를 자동으로 줄이는 state transition을 명시하는 것이다.

## 13. managed-account integer granularity

{markdown_table(accounts, ['account', 'active strategies', 'utilization', 'L1 error'])}

margin 요구액은 현재 quote가 아닌 illustrative fixture다. 작은 별도계좌는 목표 비중이 한 계약 margin보다 작아 0계약이 되고, 10전략 분산을 구현하지 못한다. 펀드와 managed account의 비교에는 transparency·trade-secret·principal-agent·규제가 더 있지만 이 표는 integer allocation 한 가지 메커니즘만 보여준다.

## 14. output-only와 법률 주의

{markdown_table(references, ['topic', 'compared', 'reason'])}

Chapter 8의 투자자문 등록 threshold, capital-introduction service, provider 이름은 2016년 역사적 문맥이다. 현재 법률·서비스 상태를 확인하지 않았고 법률 또는 투자 조언으로 제공하지 않는다.

## 15. 자동 검증과 결론

검증 {metrics['verification']['summary']['passed']}/{metrics['verification']['summary']['total']} 통과. checksum, asset/date/price, 1,500·915 rows, source active-day 숫자, calendar 보정, position lag, 비용 단조성, exit typo 차이, lifecycle seed/lag, account integer allocation과 output-only 사유를 검사한다. Chapter 8의 실험적 결론은 “한 전략이 영원히 산다”가 아니다. strategy가 태어나고 죽는다는 가정을 운영 규칙에 넣고, 성과 clock과 비용을 정직하게 정의하며, 충분한 independent pool과 실행 가능한 account size를 확보해야 한다.
"""
    assert_clean_markdown_math(content, "chapter8_report.md")
    write_text_if_changed(REPORT_DIR / "chapter8_report.md", content)


def write_readme() -> None:
    write_text_if_changed(
        CHAPTER_DIR / "README.md",
        """# Chapter 8 — Algorithmic Trading Is Good for Your Body and Soul

```bash
uv run python chapter_8_algorithmic_trading_is_good_for_body_and_soul/src/run_chapter8_analysis.py
uv run python chapter_8_algorithmic_trading_is_good_for_body_and_soul/src/run_chapter8_analysis.py --offline
```

Chapter 8에는 전용 공식 ZIP이 없어 Chapter 5 공식 ETF panel의 GLD-USO 값을 재사용한다. cross-book Bollinger replay, active-day annualization audit, 2012-2015 temporal extension, strategy lifecycle 및 managed-account granularity 개념 실험을 생성한다. GLD-GDX와 건강·규제 서술은 입력 부재 또는 비수치 성격 때문에 output-only다.
""",
    )


def notebook_markdown_cells() -> list[str]:
    return [
        """# Chapter 8: 알고리즘 트레이딩은 몸과 마음에 좋은가

마지막 장은 새 예측모형보다 왜 이 일을 하는지, 전략이 죽을 때 어떻게 살아남는지, 다른 사람의 돈을 어떤 구조로 운용하는지를 묻는다. 전용 Chapter 8 code/data ZIP은 없다. 따라서 입력이 없는 GLD-GDX 그림을 억지로 재현하지 않고, 본문이 함께 언급하는 GLD-USO를 공식 Chapter 5 ETF panel로 검증한다. 그 위에 active-day annualization, 거래비용, 전략 births/deaths, 별도계좌 integer allocation을 실험한다. lifestyle·사회적 효용·규제 서술은 수치 backtest가 아니다.""",
        """## 1. 문제 정의와 학습 질문

첫째, “전략이 계속 수익성인가?”라는 질문에 어떤 시간 범위와 비용으로 답해야 하는가? 둘째, 포지션이 없는 날을 연율화 분모에서 빼면 성과가 얼마나 부풀어 오르는가? 셋째, 코드의 exit condition 한 줄이 책의 변수 설명과 다르면 무엇을 재현해야 하는가? 넷째, 죽어 가는 전략의 leverage를 줄이고 새 전략을 추가하는 lifecycle 규칙은 어떻게 look-ahead 없이 구현하는가? 다섯째, 작은 managed account가 정말 10개 전략을 담을 수 있는가?""",
        """## 2. 공식 archive 부재와 provenance

EP Chan의 Book 3 page는 Chapter 1~7 bundle을 제공하지만 Chapter 8 전용 archive는 없다. 이 노트북은 Chapter 5 공식 archive에서 SHA-256이 고정된 ETF MAT을 재사용한다. 또한 workspace의 Chan 2013 `bollinger.py`와 GLD-USO CSV가 있으면 각각 checksum을 검사하고, 1,500개 GLD·USO 값이 Book 3 panel과 완전히 같은지 확인한다. offline 실행도 모든 필수 byte를 검증한다.""",
        """## 3. 재현 상태와 coverage matrix

`exact cross-book replay`는 Ch8 자체 출력이 아니라 Chan 2013 data와 수정된 chapter analyzer의 숫자에 대한 정확 일치다. `empirical extension`은 같은 계약을 2012-2015에 새로 적용한다. `conceptual simulation`은 시장 성과 주장이 아닌 운영 원리의 합성 fixture다. `output-only`는 GDX 입력 부재, 비수치 health claim, 낡을 수 있는 법률 서술을 뜻한다. 이 좁은 분류가 과대표기를 막는다.""",
        """## 4. 데이터 구조와 결측 진단

공식 ETF panel은 26개 asset을 담고 사용 구간의 GLD·USO는 finite·positive다. 날짜가 엄격히 증가하고 중복이 없는지 확인한다. 2006-04-26~2012-04-09는 cross-book CSV 1,500행과 값이 같고, 2012-04-10~2015-11-25는 915행 temporal extension이다. GDX는 panel에 없으므로 Figure 8.1의 negative out-of-sample curve를 실행 결과로 만들 수 없다.""",
        r"""## 5. 수식 → 코드: dynamic spread

최근 20일에서

$$USO_t=\alpha_t+\beta_t GLD_t+\epsilon_t, \qquad S_t=USO_t-\beta_t GLD_t$$

를 계산한다. `rolling_hedge_ratio()`의 window는 현재 종가까지 포함하지만 position은 다음 날 return에 적용된다. 따라서 close-to-close 구현은 미래 return을 사용하지 않는다. 같은 거래일 close에서 signal을 계산하고 그 close에 체결할 수 있다는 가정은 남으므로 실전에서는 다음 open 또는 latency-aware fill을 별도 검증해야 한다.""",
        r"""## 6. 수식 → 코드: Bollinger state machine

$$z_t=\frac{S_t-\overline S_{t,20}}{s_{t,20}}.$$

$z<-1$이면 long spread, $z>1$이면 short spread, 평균선 0에서 청산한다. sample standard deviation `ddof=1`을 사용한다. `positions_from_zscore()`는 NaN 상태를 forward-fill하고, `bollinger_strategy()`는 $q_{t-1}$로 $t$일 return을 계산한다. long과 short의 진입·청산 부등식 및 경계가 명시적인 contract다.""",
        """## 7. cross-book 원본 비교

APR, Sharpe, maximum drawdown 세 숫자를 1e-12 tolerance로 비교한다. 이 숫자는 수정된 mean-exit 구현과 active-day clock의 조합이다. 데이터는 exact이지만 Chapter 8 출판 output은 아니므로 `Machine Trading Ch8 exact`라고 부르지 않는다. source 범위, data 범위, metric clock을 모두 붙여야 재현 주장이 검증 가능하다.""",
        """## 8. active-day annualization 편향

원본 chapter analyzer는 포지션이 없을 때 분모가 0인 return을 NaN으로 만들고 drop한 뒤 252일 연율화한다. 1,500 calendar days 중 1,146 active days만 분모에 남아 APR이 23.86%가 된다. flat day를 0 return으로 포함하면 17.76%다. 자본이 다른 곳에 완전히 재배치되었다는 명시적 모델이 없으면 inactive day도 투자 horizon에 포함하는 편이 보수적이다.""",
        """## 9. source long-exit typo

Chan 2013 `bollinger.py`는 `exitZscore=0`을 선언해 놓고 long exit에 `zScore > -entryZscore`를 쓴다. short exit은 0을 쓴다. 이는 symmetric mean exit 설명과 다르다. typo-faithful 경로와 corrected 경로를 둘 다 실행해 cumulative return, turnover, drawdown 차이를 공개한다. 코드 주석이나 변수 이름이 실제 boolean branch보다 우선할 수 없다.""",
        r"""## 10. 거래비용과 clock-time 성과

$$r_t^{net}=r_t-0.001\lvert q_t-q_{t-1}\rvert.$$

10bp는 spread·borrow·impact를 모두 포괄하는 현실적 모델이 아니라 최소 stress다. signal이 바뀌는 날 unit turnover에 부과한다. gross와 net을 같은 calendar clock으로 비교해 비용이 성과를 개선하지 않는지 자동 검증한다. 실제 두 ETF를 dollar-neutral하게 맞추는 주문 수량, bid-ask, fractional share 가능성은 추가 실행 제약이다.""",
        """## 11. 2006-2012 결과 해석

수정된 mean-exit 전략은 calendar gross APR 17.76%, 10bp 후 14.43%, net max drawdown 약 -22.9%다. 이 수치는 당시 data에서 전략이 양수였다는 역사적 결과다. hedge ratio, lookback 20, entry 1, exit 0이 같은 연구 역사에서 선택되었다면 selection bias가 남는다. survivorship bias, vendor correction, adjusted price convention도 범위 밖이다.""",
        """## 12. 2012-2015 temporal extension

동일 계약을 2012-04-10에 상태를 reset하고 2015-11-25까지 적용하면 gross와 10bp net APR 모두 플러스지만 이전 기간보다 낮다. 상태 reset은 실제 연속 운용과 다르므로 full-period curve도 함께 제공한다. 이 extension은 책의 “continues to have positive returns”라는 당시 서술을 제한적으로 지지하지만, 2015 이후 또는 현재 수익성을 말하지 않는다.""",
        """## 13. 전략 births/deaths simulation

10개 synthetic strategy의 공통 factor와 idiosyncratic shock을 만들고 strategy 0은 day 500, strategy 1은 day 700 이후 기대수익을 악화시킨다. 다른 strategy는 순차적으로 태어난다. 단일 dying strategy, active strategy equal-weight pool, lagged drawdown-budget pool을 비교한다. seed=20260718을 고정하지만 이 결과는 시장 backtest가 아니라 diversification과 state transition을 보여 주는 개념 실험이다.""",
        """## 14. drawdown budget와 look-ahead 방지

각 strategy의 전일까지 wealth/peak로 drawdown을 계산하고 -12%에서 budget을 0으로 줄인다. 오늘 return은 weight 결정 뒤에 반영한다. 따라서 미래 성과를 보고 죽은 전략을 제거하지 않는다. adaptive curve가 equal-weight보다 좋아도 특정 seed의 산물이며, threshold를 합성 결과에 맞춰 최적화하지 않는다. 실전에서는 false stop과 재진입 기준, strategy correlation spike를 더 검증해야 한다.""",
        """## 15. managed-account integer allocation

10개 전략의 illustrative margin requirement와 네 account size를 둔다. 목표는 전략당 10%지만 target dollar가 한 계약 margin보다 작으면 0계약이다. 25k 계정은 어느 전략도 담지 못하고, 100k도 일부만 담는다. 500k 이상에서 모두 nonzero가 되지만 weight error는 남는다. margin 숫자는 시장 quote가 아니고, account structure의 capacity mechanism만 보여 준다.""",
        """## 16. principal-agent와 운영 구조

managed account는 자금 통제와 투명성이 높지만 trade secret이 노출되고 작은 계정의 계약 배분이 거칠다. fund는 작은 strategy allocation을 pool할 수 있지만 custody·audit·administration·규제 부담이 다르다. manager가 자기자본을 함께 투자하지 않으면 risk-taking incentive가 어긋날 수 있다. 이 노트북은 integer granularity만 계산하며 어느 구조가 법적으로 적합한지 판단하지 않는다.""",
        """## 17. output-only 및 법률·건강 주의

GLD-GDX 2006-2013 test negative라는 방향, 나쁜 상사·통근·앉아 있기·반려동물에 관한 건강 서술, 2016년 registration threshold와 provider 목록은 계산하지 않는다. 특히 법률과 사업자 상태는 시간이 지나며 바뀌므로 현재 advice로 재사용할 수 없다. source text를 인용하는 것과 오늘도 사실이라고 검증하는 것은 다른 작업이다.""",
        """## 18. 위험지표와 out-of-sample 범위

APR·Sharpe·maximum drawdown·drawdown duration을 gross/net과 active/calendar clock별로 기록한다. 2012-2015 extension은 시간상 뒤지만 하나의 추가 window일 뿐 반복 가능한 walk-forward 검증은 아니다. GLD-USO 자체가 두 ETF의 경제적 관계에 의존하고 USO는 futures roll의 영향을 받는다. 전략 shelf life는 한 번의 positive extension으로 확정할 수 없다.""",
        """## 19. deterministic contract와 자동 verification

PCG64 seed, fixed simulation dimensions, deterministic cell ID와 PNG metadata를 고정한다. 자동 검증은 SHA-256, Chapter 8 archive 부재 공개, GLD/USO/GDX coverage, date·price, 1,500/915 rows, exact source metrics, active-day clock, position lag, cost monotonicity, exit typo 차이, lifecycle weight timing, account granularity와 output-only 사유를 검사한다. 모든 assert가 참이어야 산출물을 완성한다.""",
        """## 20. 결론

이 장의 핵심은 전략 하나의 영속성을 증명하는 것이 아니다. 실제 재현은 metric clock과 exit typo가 같은 data의 성과를 크게 바꾼다는 점을 보여 준다. temporal extension은 플러스지만 약해졌고 2015에서 끝난다. 합성 실험은 dying strategy 집중보다 독립 pool과 사전 정의된 leverage reduction이 견고함을 보여 준다. 마지막으로 작은 account는 정수 계약 때문에 이론적 분산을 구현하지 못한다. 연구, 실행, business structure를 하나의 lifecycle로 관리해야 한다.""",
    ]


def build_and_execute_notebook() -> None:
    md = notebook_markdown_cells()
    cells: list[nbformat.NotebookNode] = [nbformat.v4.new_markdown_cell(md[0])]
    cells.append(nbformat.v4.new_code_cell("""from io import BytesIO\nfrom pathlib import Path\nimport sys\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0,str(Path.cwd()))\nimport run_chapter8_analysis as ch8\ndata=ch8.load_data()\nresults=ch8.run_experiments(data)\nchecks=ch8.verify_results(data,results)\nmetrics=ch8.build_metrics(data,results,checks)\nprint(f\"rows={len(data.dates):,}, checks={sum(checks.values())}/{len(checks)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[1]))
    cells.append(nbformat.v4.new_code_cell("""def show_figure(fig):\n    payload=BytesIO(); fig.savefig(payload,format='png',dpi=120,bbox_inches='tight'); plt.close(fig); display(Image(data=payload.getvalue()))\npd.DataFrame(ch8.coverage_matrix())"""))
    cells.append(nbformat.v4.new_markdown_cell(md[2]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({**metrics['provenance'],'uv_lock_sha256':metrics['environment']['uv_lock_sha256'],'versions':metrics['environment']['versions']})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[3]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(metrics['reproduction_classification']).apply(lambda x:', '.join(x))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[4]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series(metrics['data'])); show_figure(ch8.plot_data(data))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[5]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(results['gld_uso']['original_period']['contract'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[6]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch8.plot_bollinger(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[7]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['book_comparisons'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[8]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame({'source_active_day':results['gld_uso']['original_period']['source_active_day'],'calendar_gross':results['gld_uso']['original_period']['calendar_gross'],'calendar_net_10bps':results['gld_uso']['original_period']['calendar_net_10bps']}).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[9]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame({'corrected_mean_exit':results['gld_uso']['original_period']['calendar_gross'],'source_long_exit_typo':results['gld_uso']['source_code_long_exit_bug']['calendar_gross']}).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[10]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch8.plot_strategy_shelf_life(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[11]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame({'original_gross':results['gld_uso']['original_period']['calendar_gross'],'original_net':results['gld_uso']['original_period']['calendar_net_10bps'],'extension_gross':results['gld_uso']['post_book_extension']['calendar_gross'],'extension_net':results['gld_uso']['post_book_extension']['calendar_net_10bps']}).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[12]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(results['strategy_lifecycle']['performance']).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[13]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch8.plot_strategy_lifecycle(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[14]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(results['managed_account_granularity']['rows'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[15]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch8.plot_account_granularity(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[16]))
    cells.append(nbformat.v4.new_markdown_cell(md[17]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['reference_only_comparisons'])[['topic','compared','reason']]"""))
    cells.append(nbformat.v4.new_markdown_cell(md[18]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'random_seed':ch8.RANDOM_SEED,'generator':'PCG64','weights_lagged':results['strategy_lifecycle']['contract']['weights_use_information_through_previous_day'],'notebook_cell_ids':'deterministic'})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[19]))
    cells.append(nbformat.v4.new_code_cell("""verification=pd.Series(checks,name='passed'); display(verification); assert verification.all(); print(f\"verification passed: {verification.sum()}/{len(verification)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[20]))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch8-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"chapter8 notebook cell {index}")
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
        f"""# Chapter 8 quality benchmark

| Chapter | execution | reproducibility | rigor | pedagogy | trading context | total |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch8** | **{scores['execution']}** | **{scores['reproducibility']}** | **{scores['rigor']}** | **{scores['pedagogy']}** | **{scores['trading_context']}** | **{audit['total']}** |

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
        raise AssertionError(f"Chapter 8 verification failed: {failed}")
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
        statuses = ["offline reused Chapter 5 assets verified"]
    else:
        statuses = download_official_assets(force=args.force_download)
    for status in statuses:
        print(status)
    if args.download_only:
        return
    metrics = run_analysis(execute_notebook_flag=not args.skip_notebook)
    print("Chapter 8 analysis complete")
    print("Checks:", metrics["verification"]["summary"]["passed"])
    print("Report:", REPORT_DIR / "chapter8_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
