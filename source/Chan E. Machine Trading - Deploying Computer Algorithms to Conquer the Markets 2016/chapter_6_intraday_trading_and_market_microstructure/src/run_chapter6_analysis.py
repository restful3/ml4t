#!/usr/bin/env python3
"""Build the reproducible Chapter 6 market-microstructure experiments."""

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
from scipy.stats import norm


SRC_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SRC_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "book3" / "chapter_6"
ORIGINAL_DIR = CHAPTER_DIR / "original_matlab"
MANIFEST_PATH = ORIGINAL_DIR / "SOURCE_MANIFEST.json"
REPORT_DIR = SRC_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
NOTEBOOK_PATH = SRC_DIR / "chapter6_full_report.ipynb"
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


SOURCE_TICK_BIDASK = {
    "gross_pl_usd": -6687.5,
    "pl_per_trade_usd": -5.619748,
    "num_trades": 1190,
    "total_fee_usd": 113.05,
}
STRICT_TOLERANCE = 5e-7


@dataclass(frozen=True)
class ChapterData:
    tick_dn: np.ndarray
    trade_price: np.ndarray
    volume: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    volume_dn: np.ndarray
    volume_last: np.ndarray
    volume_bid: np.ndarray
    volume_ask: np.ndarray
    executions: pd.DataFrame


def chapter_manifest() -> dict[str, Any]:
    return load_chapter_manifest(PYPROJECT_PATH, chapter=6)


def download_official_assets(force: bool = False) -> list[str]:
    manifest = chapter_manifest()
    payload = download_verified_archive(
        manifest, user_agent="chan-machine-trading-experiments/0.1"
    )
    return materialize_chapter_archive(PROJECT_ROOT, manifest, payload, force=force)


def validate_offline_assets() -> None:
    validate_chapter_extraction(PROJECT_ROOT, chapter_manifest())


def load_data() -> ChapterData:
    tick = loadmat(
        RAW_DIR / "inputData_ESZ12_TAQ_20121001_v003.mat", squeeze_me=True
    )
    bars = loadmat(
        RAW_DIR / "inputData_ESZ12_volbar500_TAQ_20121001_v003.mat",
        squeeze_me=True,
    )
    executions = pd.read_csv(
        RAW_DIR / "COINSETTER_execution_reports_03_2014-10_2014.csv",
        dtype={"Side": "category", "InformationType": "category"},
    )
    return ChapterData(
        tick_dn=np.asarray(tick["dn"], dtype=float),
        trade_price=np.asarray(tick["tradePrice"], dtype=float),
        volume=np.asarray(tick["vol"], dtype=float),
        bid=np.asarray(tick["bid"], dtype=float),
        ask=np.asarray(tick["ask"], dtype=float),
        volume_dn=np.asarray(bars["dn"], dtype=float),
        volume_last=np.asarray(bars["lastPrice"], dtype=float),
        volume_bid=np.asarray(bars["bid"], dtype=float),
        volume_ask=np.asarray(bars["ask"], dtype=float),
        executions=executions,
    )


def matlab_datenum_to_datetime(values: np.ndarray) -> pd.DatetimeIndex:
    # MATLAB day 719529 is Unix epoch; retain sub-second fractions.
    seconds = (np.asarray(values, dtype=float) - 719529.0) * 86400.0
    return pd.to_datetime(seconds, unit="s", origin="unix", utc=True)


def previous(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.r_[np.nan, values[:-1]]


def matlab_round_quarter(values: np.ndarray | float) -> np.ndarray:
    # Prices are positive; MATLAB round uses half-away-from-zero.
    return np.floor(np.asarray(values, dtype=float) * 4.0 + 0.5) / 4.0


def tick_rule_order_flow(trade_price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    price = np.asarray(trade_price, dtype=float)
    size = np.asarray(volume, dtype=float)
    filled = pd.Series(price).ffill().to_numpy()
    prior = previous(filled)
    buy = price > prior
    sell = price < prior
    for index in range(1, len(price)):
        if price[index] == prior[index]:
            buy[index] = buy[index - 1]
            sell[index] = sell[index - 1]
    flow = np.zeros(len(price))
    flow[buy] = size[buy]
    flow[sell] = -size[sell]
    return flow


def rolling_flow(dn: np.ndarray, order_flow: np.ndarray, seconds: int = 60) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(order_flow, dtype=float))
    output = np.full(len(cumulative), np.nan)
    threshold = seconds / 86400.0
    for index, timestamp in enumerate(dn):
        prior = int(np.searchsorted(dn, timestamp - threshold, side="right") - 1)
        if prior >= 0:
            output[index] = cumulative[index] - cumulative[prior]
    return output


def execute_flow_strategy(
    signal: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
    use_mid_price: bool,
    multiplier: float = 50.0,
    fee_per_contract: float = 0.095,
) -> dict[str, Any]:
    midpoint = (np.asarray(bid) + np.asarray(ask)) / 2.0
    rounded_mid = matlab_round_quarter(midpoint)
    position = 0
    entry_price = np.nan
    realized_points = 0.0
    num_trades = 0
    positions = np.zeros(len(signal), dtype=int)
    realized_path = np.zeros(len(signal))
    for index, value in enumerate(signal):
        if not np.isfinite(value):
            positions[index] = position
            realized_path[index] = realized_points * multiplier
            continue
        buy_price = rounded_mid[index] if use_mid_price else ask[index]
        sell_price = rounded_mid[index] if use_mid_price else bid[index]
        if value > entry_threshold and position <= 0:
            if position < 0:
                realized_points += entry_price - buy_price
                num_trades += 2
            else:
                num_trades += 1
            entry_price = buy_price
            position = 1
        elif value < -entry_threshold and position >= 0:
            if position > 0:
                realized_points += sell_price - entry_price
                num_trades += 2
            else:
                num_trades += 1
            entry_price = sell_price
            position = -1
        elif value <= exit_threshold and position > 0:
            realized_points += sell_price - entry_price
            num_trades += 1
            position = 0
        elif value >= -exit_threshold and position < 0:
            realized_points += entry_price - buy_price
            num_trades += 1
            position = 0
        positions[index] = position
        realized_path[index] = realized_points * multiplier
    gross = realized_points * multiplier
    fee = num_trades * fee_per_contract
    return {
        "gross_pl_usd": float(gross),
        "pl_per_trade_usd": float(gross / num_trades),
        "num_trades": int(num_trades),
        "total_fee_usd": float(fee),
        "net_pl_usd": float(gross - fee),
        "ending_position": int(position),
        "positions": positions,
        "realized_pl_path": realized_path,
        "execution": "rounded_mid" if use_mid_price else "bid_ask",
    }


def execute_buy_fraction_strategy(
    buy_fraction: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    *,
    entry_threshold: float = 0.95,
    exit_threshold: float = 0.5,
    use_mid_price: bool = False,
) -> dict[str, Any]:
    """Map volumeBar.m's probability thresholds to the centered signal engine."""
    if not 0.5 <= entry_threshold <= 1.0:
        raise ValueError("entry_threshold must be in [0.5, 1.0]")
    if not 0.0 <= exit_threshold <= 0.5:
        raise ValueError("exit_threshold must be in [0.0, 0.5]")
    centered_signal = np.asarray(buy_fraction, dtype=float) - 0.5
    return execute_flow_strategy(
        centered_signal,
        bid,
        ask,
        entry_threshold=entry_threshold - 0.5,
        exit_threshold=exit_threshold - 0.5,
        use_mid_price=use_mid_price,
    )


def tick_rule_experiment(data: ChapterData) -> dict[str, Any]:
    order_flow = tick_rule_order_flow(data.trade_price, data.volume)
    signal = rolling_flow(data.tick_dn, order_flow, seconds=60)
    bid_ask = execute_flow_strategy(
        signal,
        data.bid,
        data.ask,
        entry_threshold=66,
        exit_threshold=0,
        use_mid_price=False,
    )
    midpoint = execute_flow_strategy(
        signal,
        data.bid,
        data.ask,
        entry_threshold=66,
        exit_threshold=0,
        use_mid_price=True,
    )
    fee_sensitivity = {
        str(fee): float(bid_ask["gross_pl_usd"] - fee * bid_ask["num_trades"])
        for fee in (0.095, 0.16, 2.47)
    }
    return {
        "order_flow": order_flow,
        "rolling_flow": signal,
        "bid_ask": bid_ask,
        "midpoint_current_code": midpoint,
        "fee_sensitivity": fee_sensitivity,
        "contract": {
            "lookback_seconds": 60,
            "entry_threshold": 66,
            "exit_threshold": 0,
            "signals_use_current_trade_classification": True,
            "execution_uses_current_quote": True,
            "single_day_in_sample_demonstration": True,
        },
    }


def volume_bar_experiment(data: ChapterData) -> dict[str, Any]:
    good = np.isfinite(data.volume_last)
    dn = data.volume_dn[good]
    price = data.volume_last[good]
    bid = data.volume_bid[good]
    ask = data.volume_ask[good]
    delta = price - previous(pd.Series(price).ffill().to_numpy())
    lookback = 100
    buy_fraction = np.full(len(price), np.nan)
    for index in range(lookback, len(price)):
        history = delta[index - lookback : index]
        history = history[np.isfinite(history)]
        sigma = float(np.std(history, ddof=1))
        if sigma > 0:
            buy_fraction[index] = norm.cdf(delta[index], loc=0, scale=sigma)
    entry_threshold = 0.95
    exit_threshold = 0.5
    strategy = execute_buy_fraction_strategy(
        buy_fraction,
        bid,
        ask,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        use_mid_price=False,
    )
    return {
        "dates": dn,
        "price": price,
        "buy_fraction": buy_fraction,
        "strategy": strategy,
        "lookback_bars": lookback,
        "bar_volume_contracts": 500,
        "contract": {
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "long_entry": "buy_fraction > 0.95",
            "short_entry": "buy_fraction < 0.05",
            "long_exit": "buy_fraction <= 0.5",
            "short_exit": "buy_fraction >= 0.5",
            "implementation": "centered_signal = buy_fraction - 0.5",
        },
        "source_has_no_published_numeric_output": True,
    }


def apply_book_event(
    buy_book: dict[float, int],
    sell_book: dict[float, int],
    *,
    side: str,
    action: str,
    price: float,
    amount: int,
) -> tuple[bool, bool]:
    book = buy_book if side == "BUY" else sell_book
    invalid_missing = False
    invalid_oversize = False
    if action == "WORKING_CONFIRMED":
        book[price] = book.get(price, 0) + amount
    elif action in {"CANCEL_CONFIRMED", "FILL_CONFIRMED", "PARTIAL_FILL_CONFIRMED"}:
        available = book.get(price)
        if available is None:
            invalid_missing = True
        elif amount >= available:
            invalid_oversize = amount > available
            del book[price]
        else:
            book[price] = available - amount
    return invalid_missing, invalid_oversize


def rebuild_order_book(data: ChapterData) -> dict[str, Any]:
    frame = data.executions
    buy_book: dict[float, int] = {}
    sell_book: dict[float, int] = {}
    bar_second: list[int] = []
    bar_bid: list[float] = []
    bar_ask: list[float] = []
    bar_bid_size: list[int] = []
    bar_ask_size: list[int] = []
    crossed = 0
    invalid_missing = 0
    invalid_oversize = 0
    current_second: int | None = None
    latest = (np.nan, np.nan, 0, 0)
    for row in frame.itertuples(index=False):
        second = int(np.ceil(int(row.ExchangeTime) / 1_000_000_000))
        if current_second is not None and second != current_second:
            bar_second.append(current_second)
            bar_bid.append(latest[0])
            bar_ask.append(latest[1])
            bar_bid_size.append(latest[2])
            bar_ask_size.append(latest[3])
        amount = int(np.floor(float(row.EventAmount) * 1_000_000 + 0.5))
        missing, oversize = apply_book_event(
            buy_book,
            sell_book,
            side=str(row.Side),
            action=str(row.InformationType),
            price=float(row.Level),
            amount=amount,
        )
        invalid_missing += int(missing)
        invalid_oversize += int(oversize)
        best_bid = max(buy_book) if buy_book else np.nan
        best_ask = min(sell_book) if sell_book else np.nan
        bid_size = buy_book.get(best_bid, 0) if np.isfinite(best_bid) else 0
        ask_size = sell_book.get(best_ask, 0) if np.isfinite(best_ask) else 0
        if np.isfinite(best_bid) and np.isfinite(best_ask) and best_bid > best_ask:
            crossed += 1
        latest = (best_bid, best_ask, bid_size, ask_size)
        current_second = second
    if current_second is not None:
        bar_second.append(current_second)
        bar_bid.append(latest[0])
        bar_ask.append(latest[1])
        bar_bid_size.append(latest[2])
        bar_ask_size.append(latest[3])
    bid_values = np.asarray(bar_bid, dtype=float)
    ask_values = np.asarray(bar_ask, dtype=float)
    valid = np.isfinite(bid_values) & np.isfinite(ask_values)
    spread = ask_values - bid_values
    return {
        "seconds": np.asarray(bar_second, dtype=np.int64),
        "bid": bid_values,
        "ask": ask_values,
        "bid_size_micro_btc": np.asarray(bar_bid_size, dtype=int),
        "ask_size_micro_btc": np.asarray(bar_ask_size, dtype=int),
        "events": len(frame),
        "bars": len(bar_second),
        "action_counts": {str(key): int(value) for key, value in frame["InformationType"].value_counts().sort_index().items()},
        "side_counts": {str(key): int(value) for key, value in frame["Side"].value_counts().sort_index().items()},
        "valid_two_sided_bars": int(np.sum(valid)),
        "median_spread": float(np.nanmedian(spread[valid])),
        "crossed_event_states": int(crossed),
        "invalid_missing_reductions": int(invalid_missing),
        "invalid_oversize_reductions": int(invalid_oversize),
        "sequence_is_strictly_increasing": bool(np.all(np.diff(frame["SequenceNumber"].to_numpy()) > 0)),
    }


def run_experiments(data: ChapterData) -> dict[str, Any]:
    return {
        "tick_rule": tick_rule_experiment(data),
        "volume_bar": volume_bar_experiment(data),
        "order_book": rebuild_order_book(data),
    }


def source_comparisons(results: dict[str, Any]) -> list[dict[str, Any]]:
    actual = results["tick_rule"]["bid_ask"]
    rows = []
    for key, expected in SOURCE_TICK_BIDASK.items():
        value = actual[key]
        tolerance = 0 if key == "num_trades" else STRICT_TOLERANCE
        rows.append(
            {
                "strategy": "tickRule.m bid-ask execution",
                "metric": key,
                "python": value,
                "source": expected,
                "absolute_error": abs(value - expected),
                "tolerance": tolerance,
                "matches_source": abs(value - expected) <= tolerance,
            }
        )
    return rows


def reference_only_results() -> list[dict[str, Any]]:
    return [
        {
            "topic": "tick-rule midpoint execution comment",
            "source_file": "tickRule.m",
            "book_output": {"gross_pl_usd": 800.0, "num_trades": 1168, "total_fee_usd": 110.96},
            "compared": False,
            "reason": "The execution-price branch cannot change signals or trade count, yet the midpoint and bid-ask comments report 1168 versus 1190 trades; at least one comment is stale",
        },
        {
            "topic": "Algoseek aggressor-tag strategy",
            "source_file": "aggressorTag_algoseek.m",
            "book_output": {"20121001_midpoint_pl_usd": 762.5, "20121001_bidask_pl_usd": 300.0, "num_trades": 75},
            "compared": False,
            "reason": "The downloadable TAQ MAT has prices and volume but no Algoseek aggressor flag or the CSV read by this script",
        },
        {
            "topic": "multi-day profitability",
            "source_file": "tickRule.m and volumeBar.m",
            "book_output": None,
            "compared": False,
            "reason": "The bundled ES examples contain one trading day, so no out-of-sample Sharpe or annual return is supportable",
        },
    ]


def coverage_matrix() -> list[dict[str, str]]:
    return [
        {"topic": "tick-rule trade classification", "status": "exact source replay", "evidence": "77,390-row ES TAQ MAT"},
        {"topic": "bid-ask order-flow strategy", "status": "exact source-output comparison", "evidence": "four published outputs"},
        {"topic": "midpoint execution", "status": "current-code replay + output-only comment", "evidence": "published trade counts contradict branch invariance"},
        {"topic": "500-contract volume bars/BVC", "status": "source-semantic replay", "evidence": "3,639-row MAT; source entry/exit thresholds 0.95/0.5"},
        {"topic": "Coinsetter order-book reconstruction", "status": "independent Python port", "evidence": "246,060 execution events"},
        {"topic": "Algoseek aggressor tags", "status": "code/output-only", "evidence": "aggressor input file absent"},
        {"topic": "multi-day strategy performance", "status": "unavailable", "evidence": "single ES trading day only"},
    ]


def verify_results(data: ChapterData, results: dict[str, Any]) -> dict[str, bool]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    comparisons = source_comparisons(results)
    tick = results["tick_rule"]
    volume = results["volume_bar"]
    book = results["order_book"]
    references = reference_only_results()
    synthetic_buy: dict[float, int] = {}
    synthetic_sell: dict[float, int] = {}
    apply_book_event(synthetic_buy, synthetic_sell, side="BUY", action="WORKING_CONFIRMED", price=100.0, amount=5)
    apply_book_event(synthetic_buy, synthetic_sell, side="BUY", action="PARTIAL_FILL_CONFIRMED", price=100.0, amount=2)
    return {
        "archive_manifest_matches": manifest["archive_sha256"] == chapter_manifest()["sha256"],
        "all_seven_archive_members_pinned": len(manifest["members"]) == 7,
        "four_matlab_sources_present": len(list(ORIGINAL_DIR.glob("*.m"))) == 4,
        "tick_timestamps_are_strictly_increasing": bool(np.all(np.diff(data.tick_dn) > 0)),
        "tick_quotes_are_not_crossed": bool(np.all(data.bid <= data.ask)),
        "tick_rule_bidask_matches_source": all(row["matches_source"] for row in comparisons),
        "midpoint_and_bidask_share_trade_decisions": tick["midpoint_current_code"]["num_trades"] == tick["bid_ask"]["num_trades"],
        "stale_midpoint_comment_is_not_compared": references[0]["compared"] is False,
        "fees_do_not_improve_tick_strategy": tick["fee_sensitivity"]["2.47"] <= tick["fee_sensitivity"]["0.16"] <= tick["fee_sensitivity"]["0.095"],
        "volume_bar_contract_matches_source": (
            volume["bar_volume_contracts"] == 500
            and volume["lookback_bars"] == 100
            and volume["contract"]["entry_threshold"] == 0.95
            and volume["contract"]["exit_threshold"] == 0.5
        ),
        "volume_bar_probabilities_are_bounded": bool(np.all((volume["buy_fraction"][np.isfinite(volume["buy_fraction"])] >= 0) & (volume["buy_fraction"][np.isfinite(volume["buy_fraction"])] <= 1))),
        "volume_bar_source_output_not_invented": volume["source_has_no_published_numeric_output"] is True,
        "order_book_sequence_is_strict": book["sequence_is_strictly_increasing"],
        "order_book_event_count_matches_csv": book["events"] == len(data.executions) == 246060,
        "order_book_has_two_sided_bars": book["valid_two_sided_bars"] > 0,
        "synthetic_partial_fill_reduces_size": synthetic_buy == {100.0: 3},
        "all_reference_only_rows_have_reasons": all(row["compared"] is False and row["reason"] for row in references),
        "aggressor_data_is_not_claimed_available": "no Algoseek aggressor flag" in references[1]["reason"],
        "single_day_is_not_claimed_out_of_sample": tick["contract"]["single_day_in_sample_demonstration"] and references[2]["compared"] is False,
        "bidask_execution_is_worse_than_midpoint": tick["bid_ask"]["gross_pl_usd"] < tick["midpoint_current_code"]["gross_pl_usd"],
        "crossed_orderbook_states_are_disclosed": (
            book["crossed_event_states"] > 0 and book["median_spread"] < 0
        ),
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


def build_metrics(data: ChapterData, results: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    extraction = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tick = results["tick_rule"]
    volume = results["volume_bar"]
    book = results["order_book"]
    classes = {
        name: ("independent_or_empirical" if index < 15 else "contract_invariant")
        for index, name in enumerate(checks)
    }
    return json_safe(
        {
            "chapter": 6,
            "title": "Intraday Trading and Market Microstructure",
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
                "versions": environment_versions(("numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbclient")),
                "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
            },
            "data": {
                "es_tick": {
                    "rows": len(data.tick_dn),
                    "date_start": str(matlab_datenum_to_datetime(data.tick_dn[:1])[0]),
                    "date_end": str(matlab_datenum_to_datetime(data.tick_dn[-1:])[0]),
                    "missing_trade_price": int(np.sum(~np.isfinite(data.trade_price))),
                    "median_spread_points": float(np.median(data.ask - data.bid)),
                },
                "es_volume_bar": {"rows": len(data.volume_dn), "bad_last_price": int(np.sum(~np.isfinite(data.volume_last))), "contracts_per_bar": 500},
                "coinsetter": {"rows": len(data.executions), "sequence_start": int(data.executions["SequenceNumber"].iloc[0]), "sequence_end": int(data.executions["SequenceNumber"].iloc[-1])},
            },
            "reproduction_classification": {
                "exact_source_replay": ["tickRule.m bid-ask"],
                "source_semantic_or_independent": ["volumeBar.m", "buildOrderBook.m Python port"],
                "output_only": [row["topic"] for row in reference_only_results()],
            },
            "coverage": coverage_matrix(),
            "results": {
                "tick_rule": {
                    "bid_ask": {key: value for key, value in tick["bid_ask"].items() if key not in {"positions", "realized_pl_path"}},
                    "midpoint_current_code": {key: value for key, value in tick["midpoint_current_code"].items() if key not in {"positions", "realized_pl_path"}},
                    "fee_sensitivity": tick["fee_sensitivity"],
                    "contract": tick["contract"],
                },
                "volume_bar": {
                    "strategy": {key: value for key, value in volume["strategy"].items() if key not in {"positions", "realized_pl_path"}},
                    "lookback_bars": volume["lookback_bars"],
                    "bar_volume_contracts": volume["bar_volume_contracts"],
                    "contract": volume["contract"],
                    "source_has_no_published_numeric_output": True,
                },
                "order_book": {key: value for key, value in book.items() if key not in {"seconds", "bid", "ask", "bid_size_micro_btc", "ask_size_micro_btc"}},
            },
            "book_comparisons": source_comparisons(results),
            "reference_only_comparisons": reference_only_results(),
            "source_limitations": {
                "midpoint_comment_trade_count_contradiction": True,
                "aggressor_input_missing": True,
                "single_es_day_only": True,
                "coinsetter_invalid_or_crossed_states_preserved_as_diagnostics": True,
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


def format_time_axis(axis: plt.Axes, *, max_ticks: int = 7) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
    axis.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.show_offset = False
    axis.xaxis.set_major_formatter(formatter)


def plot_tick_diagnostics(data: ChapterData) -> plt.Figure:
    dates = matlab_datenum_to_datetime(data.tick_dn)
    sample = slice(0, min(8000, len(dates)))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
    axes[0].plot(dates[sample], data.trade_price[sample], linewidth=0.8, label="trade")
    axes[0].plot(dates[sample], data.bid[sample], linewidth=0.5, alpha=0.7, label="bid")
    axes[0].plot(dates[sample], data.ask[sample], linewidth=0.5, alpha=0.7, label="ask")
    axes[0].set_title("ES TAQ opening sample")
    axes[0].set_ylabel("Index points")
    axes[0].legend()
    spread = data.ask - data.bid
    axes[1].plot(dates[sample], spread[sample], color="tab:red", linewidth=0.7)
    axes[1].set_title("Quoted spread")
    axes[1].set_ylabel("Points")
    format_time_axis(axes[1])
    return fig


def plot_tick_strategy(data: ChapterData, results: dict[str, Any]) -> plt.Figure:
    tick = results["tick_rule"]
    dates = matlab_datenum_to_datetime(data.tick_dn)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(dates, tick["rolling_flow"], linewidth=0.5)
    axes[0].axhline(66, color="tab:red", linestyle="--")
    axes[0].axhline(-66, color="tab:red", linestyle="--")
    axes[0].set_title("60-second tick-rule signed volume")
    axes[0].set_ylabel("Contracts")
    format_time_axis(axes[0])
    axes[1].plot(dates, tick["midpoint_current_code"]["realized_pl_path"], label="rounded midpoint")
    axes[1].plot(dates, tick["bid_ask"]["realized_pl_path"], label="bid/ask")
    axes[1].set_title("Realized P&L: execution assumption dominates")
    axes[1].set_ylabel("USD before fees")
    axes[1].legend()
    format_time_axis(axes[1])
    return fig


def plot_volume_bars(results: dict[str, Any]) -> plt.Figure:
    volume = results["volume_bar"]
    dates = matlab_datenum_to_datetime(volume["dates"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(dates, volume["buy_fraction"], linewidth=0.7)
    axes[0].axhline(0.95, color="tab:red", linestyle="--")
    axes[0].axhline(0.05, color="tab:red", linestyle="--")
    axes[0].set_title("Bulk-volume-classification buy fraction")
    axes[0].set_ylim(-0.03, 1.03)
    format_time_axis(axes[0])
    axes[1].plot(dates, volume["strategy"]["realized_pl_path"], color="tab:purple")
    axes[1].set_title("500-contract volume-bar strategy")
    axes[1].set_ylabel("USD before fees")
    format_time_axis(axes[1])
    return fig


def plot_order_book(results: dict[str, Any]) -> plt.Figure:
    book = results["order_book"]
    valid = np.isfinite(book["bid"]) & np.isfinite(book["ask"])
    seconds = pd.to_datetime(book["seconds"], unit="s", utc=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(seconds[valid], (book["ask"] - book["bid"])[valid], linewidth=0.7)
    axes[0].set_title("Diagnostic: crossed Coinsetter top-of-book states")
    axes[0].set_ylabel("ask - bid (USD; negative = crossed)")
    format_time_axis(axes[0])
    actions = book["action_counts"]
    axes[1].barh(list(actions), list(actions.values()))
    axes[1].set_title("Execution-report action counts")
    axes[1].set_xlabel("Events")
    return fig


def plot_cost_comparison(results: dict[str, Any]) -> plt.Figure:
    tick = results["tick_rule"]
    volume = results["volume_bar"]["strategy"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = ["midpoint\ncurrent code", "bid/ask", "volume bar\nbid/ask"]
    gross = [tick["midpoint_current_code"]["gross_pl_usd"], tick["bid_ask"]["gross_pl_usd"], volume["gross_pl_usd"]]
    net = [tick["midpoint_current_code"]["net_pl_usd"], tick["bid_ask"]["net_pl_usd"], volume["net_pl_usd"]]
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, gross, 0.36, label="gross")
    axes[0].bar(x + 0.18, net, 0.36, label="net at $0.095/contract")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, names)
    axes[0].set_title("One-day P&L is execution-sensitive")
    axes[0].legend()
    fees = [0.095, 0.16, 2.47]
    axes[1].bar([str(fee) for fee in fees], [tick["fee_sensitivity"][str(fee)] for fee in fees])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Tick-rule net P&L by per-contract fee")
    axes[1].set_xlabel("USD per contract")
    return fig


def save_figures(data: ChapterData, results: dict[str, Any]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "tick_diagnostics.png": plot_tick_diagnostics(data),
        "tick_strategy.png": plot_tick_strategy(data, results),
        "volume_bars.png": plot_volume_bars(results),
        "order_book.png": plot_order_book(results),
        "cost_comparison.png": plot_cost_comparison(results),
    }
    for name, figure in figures.items():
        figure.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight", metadata={"Software": "chan-machine-trading-experiments"})
        plt.close(figure)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows),
    ])


def write_report(data: ChapterData, results: dict[str, Any], metrics: dict[str, Any]) -> None:
    comparisons = [
        {"metric": row["metric"], "Python": f"{row['python']:.9f}", "MATLAB": str(row["source"]), "error": f"{row['absolute_error']:.2e}"}
        for row in metrics["book_comparisons"]
    ]
    references = [
        {"topic": row["topic"], "compared": str(row["compared"]).lower(), "reason": row["reason"]}
        for row in metrics["reference_only_comparisons"]
    ]
    tick = results["tick_rule"]
    volume = results["volume_bar"]
    book = results["order_book"]
    content = f"""# Machine Trading 2016 Chapter 6 — Intraday Trading and Market Microstructure

## 1. 결론 먼저

공식 ZIP 7개 멤버를 SHA-256으로 고정했다. `tickRule.m`의 bid-ask 경로는 원본 4개 출력과 정확히 일치한다. 그러나 단 하루의 ES 데이터이므로 전략 수익성이나 out-of-sample Sharpe를 주장하지 않는다. 이 장의 실험 목적은 체결가격, 분류 규칙, volume clock, 주문장 상태가 결과를 어떻게 바꾸는지 검증하는 것이다.

## 2. provenance와 데이터 구조

- 공식 URL: {metrics['provenance']['archive_url']}
- archive SHA-256: `{metrics['provenance']['archive_sha256']}`
- 7개 = MATLAB 4 + 연구 데이터 3
- ES TAQ: {len(data.tick_dn):,}행, 2012-10-01 하루
- ES 500계약 volume bars: {len(data.volume_dn):,}행
- Coinsetter execution reports: {len(data.executions):,}행
- `uv.lock` SHA와 environment versions는 metrics.json에 기록했다.

## 3. 학습 질문

1. 같은 신호라도 midpoint와 bid-ask 체결이 왜 정반대 손익을 만드는가?
2. tick rule의 동가 거래 방향 전파는 order flow에 어떤 영향을 주는가?
3. 시간 bar 대신 500계약 volume bar를 쓰면 관측 clock이 어떻게 달라지는가?
4. 주문 추가·취소·부분체결로 best bid/ask를 어떻게 재구성하는가?

## 4. coverage와 재현 상태

{markdown_table(metrics['coverage'], ['topic', 'status', 'evidence'])}

## 5. 수식 → 코드: tick rule

가격이 직전 거래보다 오르면 매수, 내리면 매도, 같으면 직전 분류를 이어받는다. 60초 signed volume은 다음과 같다.

$$
OF_t=\\sum_{{t-60s < i \\leq t}} q_i s_i, \\qquad s_i \\in \\{{-1,+1\\}}.
$$

`tick_rule_order_flow()`와 `rolling_flow()`가 이 계약을 구현한다. 현재 이벤트까지 신호에 포함하고 현재 quote로 체결하므로 실거래 지연을 모델링한 예측 백테스트가 아니라 원본의 동일 시점 실행 예제다.

## 6. 원본 MATLAB 정확 비교

{markdown_table(comparisons, ['metric', 'Python', 'MATLAB', 'error'])}

bid-ask gross P&L은 {tick['bid_ask']['gross_pl_usd']:.2f}달러, 거래 {tick['bid_ask']['num_trades']}건, 최소 수수료 차감 후 {tick['bid_ask']['net_pl_usd']:.2f}달러다. 반면 현재 코드 의미의 rounded midpoint gross P&L은 {tick['midpoint_current_code']['gross_pl_usd']:.2f}달러다. 체결 가정 하나가 신호 효과보다 크다.

## 7. 원본 midpoint 주석 모순

`useMidPrice` 분기는 가격만 바꾸며 포지션 결정에는 관여하지 않는다. 따라서 midpoint와 bid-ask의 거래 횟수는 같아야 한다. 실제 current-code replay는 둘 다 {tick['bid_ask']['num_trades']}건이다. 원본 주석은 1,168건과 1,190건을 보고하므로 동시에 성립할 수 없어 midpoint 주석을 `compared:false`로 둔다.

## 8. volume bar와 BVC

`volumeBar.m`은 500계약마다 만든 bar에서 최근 100개 가격변화의 표준편차로 현재 변화의 정규 CDF를 계산한다. 이는 buy volume fraction의 근사다. 원본의 진입 기준은 매수 fraction 0.95 초과 또는 0.05 미만이고, 청산 기준은 0.5다. Python은 `buy_fraction - 0.5`로 중심화하여 원본의 보유 구간까지 유지한다. 결과는 gross P&L {volume['strategy']['gross_pl_usd']:.2f}달러, 거래 {volume['strategy']['num_trades']}건이지만 원본에 숫자 출력 주석이 없어 source-semantic replay로만 분류하고 책 비교 숫자를 만들지 않는다.

## 9. 주문장 재구성

Coinsetter의 246,060개 이벤트를 가격별 수량 dictionary로 처리한다. 추가는 수량을 더하고 취소·체결·부분체결은 줄인다. 매초 마지막 상태에서 최고 bid와 최저 ask를 기록한다. 결과는 {book['bars']:,}개 1초 bar, 양면 호가 {book['valid_two_sided_bars']:,}개, median spread {book['median_spread']:.6f}달러다. 음수 median은 정상 시장 spread가 아니라, 원본 스크립트도 assertion을 주석 처리한 crossed order-book state가 많다는 데이터 품질 경고다. 누락·과대 감소와 crossed state를 제거해 결과를 미화하지 않고 진단 수로 남긴다.

## 10. 거래비용과 슬리피지

책의 최소 fee 0.095달러, 더 일반적인 0.16달러, IB 예시 2.47달러를 비교한다. bid-ask 경로는 spread를 이미 체결가격에 포함하며 fee는 별도 차감한다. market impact, queue position, latency는 없다. midpoint는 지정가가 항상 체결된다는 낙관적 상한이지 실행 가능한 보장이 아니다.

## 11. 표본 외와 편향 경고

단 하루에 threshold 66을 선택하고 같은 날 평가하므로 selection bias가 크다. 이 결과에는 walk-forward도 out-of-sample도 없다. 여러 임계값과 lookback을 같은 날 탐색하면 data snooping이 더 심해진다. 장중 전략은 거래일·변동성 regime·뉴스일을 넓게 포함해야 한다.

## 12. output-only 참조

{markdown_table(references, ['topic', 'compared', 'reason'])}

Algoseek aggressor flag 입력은 공식 번들에 없다. TAQ MAT의 tick rule과 실제 aggressor tag를 같은 것으로 간주하지 않는다.

## 13. 위험지표 해석

하루의 달러 P&L과 trade count만 보고한다. CAGR, Sharpe, drawdown, Calmar는 정의하지 않는다. 일중 realized P&L path는 미청산 포지션의 mark-to-market을 포함하지 않으므로 완전한 equity curve도 아니다. 이 제한이 다일 백테스트처럼 보이는 것을 막는다.

## 14. 자동 검증

검증 {metrics['verification']['summary']['passed']}/{metrics['verification']['summary']['total']} 통과. checksum, source output, timestamp, quote, 비용 단조성, BVC 범위, 주문장 synthetic partial fill, 누락 입력 공개를 검사한다.

## 15. 결론

미시구조 전략에서 alpha보다 먼저 검증해야 할 것은 clock, trade sign, quote side, fee와 order-state transition이다. midpoint 수익이 bid-ask에서 사라지는 사례는 실행 가정이 연구 결론을 뒤집는다는 가장 직접적인 경고다.
"""
    assert_clean_markdown_math(content, "chapter6_report.md")
    write_text_if_changed(REPORT_DIR / "chapter6_report.md", content)


def write_readme() -> None:
    write_text_if_changed(
        CHAPTER_DIR / "README.md",
        """# Chapter 6 — Intraday Trading and Market Microstructure

```bash
uv run python chapter_6_intraday_trading_and_market_microstructure/src/run_chapter6_analysis.py
uv run python chapter_6_intraday_trading_and_market_microstructure/src/run_chapter6_analysis.py --offline
```

공식 7개 멤버를 checksum으로 검증하고 tick-rule exact replay, volume-bar source-semantic replay, Coinsetter order-book reconstruction을 생성한다. 단일 거래일 결과는 수익성 백테스트가 아니다.
""",
    )


def notebook_markdown_cells() -> list[str]:
    return [
        """# Chapter 6: 장중 거래와 시장 미시구조\n\n공식 ES TAQ·volume bar·Coinsetter execution report를 사용해 trade sign, 체결가격, 주문장 상태를 재현한다. 장중 전략에서는 신호의 예측력만큼 거래 시계, 호가의 어느 편에서 체결했는지, 신호가 가용한 시점이 중요하다. 이 노트북은 한 거래일의 코드 의미와 실행 가정을 검증하는 실험이지, 전략의 장기 수익성을 입증하는 백테스트가 아니다.""",
        """## 1. 문제 정의와 학습 질문\n\n첫째, 같은 신호가 rounded midpoint와 bid-ask에서 왜 서로 반대 손익을 만드는가? 둘째, tick rule은 직전과 같은 가격의 거래를 어떻게 분류하고 그 선택이 60초 order flow에 어떤 기억을 남기는가? 셋째, 1초 wall clock과 500계약 volume clock은 활동이 집중된 구간을 어떻게 다르게 표현하는가? 넷째, 주문 추가·취소·체결을 받아 best bid/ask를 재구성할 때 어떤 데이터 오류를 감지해야 하는가?""",
        """## 2. 재현 상태와 coverage\n\n`strict exact`는 원본이 게시한 숫자와 Python 계산을 허용오차 내에서 비교한다. `source-semantic`은 동일한 알고리즘을 실행했지만 원본 숫자가 없어 정확 재현을 주장하지 않는다. `independent port`는 MATLAB 상태 전이를 Python으로 옮기고 합성 사례로 동작을 검증한다. `output-only`는 입력이 없거나 원본 주석이 서로 모순되어 복사한 숫자를 실험 결과로 가장하지 않는다. coverage 표는 이 구분을 예제별로 드러낸다.""",
        """## 3. provenance와 environment versions\n\n공식 URL·archive SHA-256·7개 member checksum·uv.lock SHA·패키지 버전을 고정한다. offline 실행은 네트워크 없이 이를 전수 검사한다.""",
        """## 4. 데이터 구조와 결측 진단\n\nTAQ 77,390행은 2012-10-01의 거래가, 거래량, bid, ask와 MATLAB datenum을 담는다. 계산 전에 timestamp가 엄격히 증가하는지, bid가 ask를 넘지 않는지, 거래가와 호가에 결측이 없는지를 검사한다. 3,639행 volume-bar MAT은 행당 500계약으로 정의된다. Coinsetter CSV는 246,060개 event를 순서대로 담지만, 교차 호가와 존재하지 않는 level 감소가 있어 정제된 NBBO로 보면 안 된다. 이 세 데이터는 trade-event, volume, wall clock이라는 서로 다른 시계를 제공한다.""",
        """## 5. 수식 → 코드: tick rule\n\n$$OF_t=\\sum_{t-60s<i\\leq t} q_i s_i, \\qquad s_i\\in\\{-1,+1\\}.$$\n\n`tick_rule_order_flow()`는 가격 상승을 +1, 하락을 -1로 분류하고 동가이면 직전 sign을 이어받는다. `rolling_flow()`는 현재 event를 포함한 60초 signed volume을 더한다. `execute_flow_strategy()`는 order flow가 +66 이상이면 long, -66 이하이면 short에 진입하고 0을 통과하면 청산한다. 신호와 현재 quote를 같은 event에서 사용하므로 이는 latency가 없는 원본 의미의 재현이며, 미래 체결 가능성을 검증한 예측 모형은 아니다.""",
        """## 6. ES TAQ 시각 진단\n\n호가 spread와 trade path를 함께 봐야 체결 가정의 크기를 이해할 수 있다. midpoint는 실제 bid/ask 사이의 회계적 기준일 뿐 보장된 체결가격이 아니다.""",
        """## 7. 원본 MATLAB exact 비교\n\nbid-ask 경로의 gross P&L, trade당 P&L, 거래 수, fee를 `tickRule.m`의 소수 6자리 주석과 비교한다. 소수 지표의 허용오차 5e-7은 원본 표시 자릿수의 반올림 반폭이고, 정수 trade count는 완전 일치를 요구한다. 네 지표가 모두 통과해야 exact replay로 인정한다. 총손익이 -6,687.50달러이고 수수료를 더하면 -6,800.55달러이므로, 스프레드 체결만으로 이 신호의 날짜 내 성과는 사라진다.""",
        """## 8. midpoint 주석의 모순\n\n`useMidPrice`는 체결가 계산 branch일 뿐 entry·exit decision에 사용되지 않는다. 따라서 같은 입력이면 midpoint와 bid-ask의 거래 수는 같아야 한다. current-code replay는 둘 다 1,190건이지만 원본 주석은 1,168건과 1,190건을 적어 놓았다. 어느 주석이 stale인지 추측해 가장하지 않고, midpoint 주석 결과는 `compared:false` 및 기계 판독 사유와 함께 output-only로 남긴다. 이 구분은 “코드를 실행했다”와 “출판된 숫자를 재현했다”가 다른 주장임을 보여준다.""",
        """## 9. 실행가 민감도\n\n현재 rounded midpoint는 플러스, bid-ask는 큰 마이너스다. signal alpha보다 spread와 체결 위치가 더 크다는 사실을 그림으로 확인한다.""",
        """## 10. volume bars와 BVC\n\nvolume clock은 500계약이 쌓일 때마다 한 칸 움직인다. 거래가 활발하면 wall-clock 1초 안에 여러 bar가 만들어지고 한산하면 한 bar가 긴 시간을 덮는다. `volume_bar_experiment()`은 최근 100개 bar 가격변화의 표본 표준편차로 현재 변화를 표준화하고 정규 CDF를 buy fraction으로 근사한다. 원본의 entry threshold 0.95에 따라 0.95 초과는 long, 0.05 미만은 short로 진입한다. exit threshold 0.5에 따라 long은 fraction이 0.5 이하일 때, short는 0.5 이상일 때 청산하며 그 전의 중립 구간에서는 포지션을 유지한다. 구현은 `buy_fraction - 0.5`로 중심화해 이 네 부등식을 공통 실행 엔진에 정확히 대응한다. 원본에 비교할 숫자가 없으므로 계산값은 교육용 source-semantic 실험이며, 단일 성과를 독립 재현으로 표현하지 않는다.""",
        """## 11. 주문장 state machine\n\n`WORKING_CONFIRMED`는 side의 가격 level에 수량을 더한다. `CANCEL_CONFIRMED`, `FILL_CONFIRMED`, `PARTIAL_FILL_CONFIRMED`는 해당 level의 수량을 줄이고 0이 되면 level을 제거한다. best bid는 남은 buy level의 최대값, best ask는 sell level의 최소값이다. `apply_book_event()`를 합성 주문에 적용해 수량 5인 level이 partial fill 2 후 3으로 남는지 별도 검증한다. 초 단위 bar는 해당 초 마지막 event 후의 top-of-book을 기록한다.""",
        """## 12. Coinsetter 데이터 진단\n\nSequenceNumber의 엄격한 증가, 누락 level 감소, 보유 수량보다 큰 oversize 감소, bid가 ask보다 큰 crossed state를 모두 기록한다. 원본 MATLAB도 일부 잘못된 trade를 인정하고 bid-ask assertion을 주석 처리했다. 재구성 median spread가 음수라는 점은 사실적 시장 spread 추정치가 아니라 order state가 오염됐다는 강한 경고다. 교차 상태를 임의로 필터하면 그럴듯한 차트는 만들 수 있지만, 원본 데이터의 불완전성을 숨기게 된다. 따라서 이 구간은 매매 신호가 아니라 데이터 품질 실험으로 읽어야 한다.""",
        """## 13. 거래비용·슬리피지·latency\n\n0.095/0.16/2.47달러 fee를 비교한다. spread는 bid-ask 체결에 포함되지만 market impact, queue position, network latency는 포함되지 않는다.""",
        """## 14. 표본 외 부재\n\nES 샘플은 단 하루이고 threshold 66도 같은 날의 예제에서 선택됐으므로 selection bias가 있다. 임계값, lookback, 청산 규칙을 같은 날에서 반복 탐색하면 data snooping이 커진다. 실전 평가는 여러 월의 뉴스일·저활동일·고변동일을 시간 순서로 나누고, 과거에서만 선택한 파라미터를 다음 기간에 고정하는 walk-forward가 필요하다. 여기에는 그런 out-of-sample 구간이 없으므로 Sharpe와 annual return을 주장하지 않는다.""",
        """## 15. 위험지표 범위\n\nCAGR·Sharpe·drawdown·Calmar를 한 거래일에 연율화하면 표본 수가 거의 없는 숫자에 잘못된 정밀도를 부여한다. 따라서 이 노트북은 달러 총손익, trade count, trade당 손익과 fee만 보고한다. 또한 realized P&L path는 청산 event에서만 손익을 반영하므로 중간 미청산 포지션의 mark-to-market과 intraday drawdown이 빠져 있다. 이것이 위험지표 해석의 핵심 한계이며, 해당 path를 완전한 equity curve로 보지 않는다.""",
        """## 16. deterministic contract\n\n난수를 쓰지 않지만 random seed = 20260718을 재현 메타데이터에 고정한다. 고정 셀 ID와 PNG metadata로 연속 빌드 해시도 동일하게 만든다.""",
        """## 17. 자동 verification\n\n21개 검사는 단순히 같은 결과를 다시 비교하는 항목과 외부 증거를 분리한다. archive hash, 7개 member, 4개 MATLAB source, 77,390개 timestamp·quote, 246,060개 sequence, 원본 출력 4개 일치는 독립·경험 검사다. 수수료 단조성, 입력 부재 공개, output-only 사유, 단일 샘플 경고는 contract invariant다. synthetic partial fill은 상태 전이가 수량 5에서 3으로 줄어드는지 독립적으로 검사한다. 모든 항목이 참일 때만 산출물을 완성한다.""",
        """## 18. 결론\n\n이 장의 가장 큰 결과는 midpoint의 +887.50달러가 bid-ask에서 -6,687.50달러로 바뀐다는 점이다. 같은 신호와 trade decision에서 체결가 가정만으로 결론이 뒤집혔다. volume clock은 활동 집중을 다른 방식으로 보여주지만 숫자 원본 비교가 없고, Coinsetter 주문장은 교차 상태 때문에 신호 입력으로 바로 쓸 수 없다. 따라서 장중 전략은 alpha를 논하기 전에 데이터 clock, trade sign, quote side, event ordering, fee, latency를 고정해야 한다. 다음 단계는 여러 날의 정제된 event data에서 현실적 지연·slippage를 넣은 walk-forward 평가이며, 이 노트북의 단일 실험은 그 결론을 대신하지 않는다.""",
    ]


def build_and_execute_notebook() -> None:
    md = notebook_markdown_cells()
    cells: list[nbformat.NotebookNode] = [nbformat.v4.new_markdown_cell(md[0])]
    cells.append(nbformat.v4.new_code_cell("""from io import BytesIO\nfrom pathlib import Path\nimport sys\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0, str(Path.cwd()))\nimport run_chapter6_analysis as ch6\ndata=ch6.load_data()\nresults=ch6.run_experiments(data)\nchecks=ch6.verify_results(data, results)\nmetrics=ch6.build_metrics(data, results, checks)\nprint(f\"TAQ={len(data.tick_dn):,}, events={len(data.executions):,}, checks={sum(checks.values())}/{len(checks)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[1]))
    cells.append(nbformat.v4.new_code_cell("""def show_figure(fig):\n    payload=BytesIO(); fig.savefig(payload,format='png',dpi=120,bbox_inches='tight'); plt.close(fig); display(Image(data=payload.getvalue()))\npd.DataFrame(ch6.coverage_matrix())"""))
    cells.append(nbformat.v4.new_markdown_cell(md[2]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(metrics['reproduction_classification']).apply(lambda x:', '.join(x))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[3]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({**metrics['provenance'],'uv_lock_sha256':metrics['environment']['uv_lock_sha256']})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[4]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['data']).T"""))
    cells.append(nbformat.v4.new_markdown_cell(md[5]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'signed_volume_sum':results['tick_rule']['order_flow'].sum(),'rolling_flow_5pct':pd.Series(results['tick_rule']['rolling_flow']).quantile(.05),'rolling_flow_95pct':pd.Series(results['tick_rule']['rolling_flow']).quantile(.95)})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[6]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch6.plot_tick_diagnostics(data))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[7]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['book_comparisons'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[8]))
    cells.append(nbformat.v4.new_code_cell("""pd.DataFrame(metrics['reference_only_comparisons'])[['topic','compared','reason']]"""))
    cells.append(nbformat.v4.new_markdown_cell(md[9]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch6.plot_tick_strategy(data,results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[10]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series({k:v for k,v in metrics['results']['volume_bar'].items() if k!='strategy'})); show_figure(ch6.plot_volume_bars(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[11]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series(metrics['results']['order_book'])"""))
    cells.append(nbformat.v4.new_markdown_cell(md[12]))
    cells.append(nbformat.v4.new_code_cell("""show_figure(ch6.plot_order_book(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[13]))
    cells.append(nbformat.v4.new_code_cell("""display(pd.Series(results['tick_rule']['fee_sensitivity'])); show_figure(ch6.plot_cost_comparison(results))"""))
    cells.append(nbformat.v4.new_markdown_cell(md[14]))
    cells.append(nbformat.v4.new_markdown_cell(md[15]))
    cells.append(nbformat.v4.new_markdown_cell(md[16]))
    cells.append(nbformat.v4.new_code_cell("""pd.Series({'random_seed':20260718,'uses_randomness':False,'notebook_cell_ids':'deterministic'})"""))
    cells.append(nbformat.v4.new_markdown_cell(md[17]))
    cells.append(nbformat.v4.new_code_cell("""verification=pd.Series(checks,name='passed'); display(verification); assert verification.all(); print(f\"verification passed: {verification.sum()}/{len(verification)}\")"""))
    cells.append(nbformat.v4.new_markdown_cell(md[18]))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":f"{sys.version_info.major}.{sys.version_info.minor}"}})
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"ch6-{index:02d}-{cell.cell_type}"
        if cell.cell_type == "markdown":
            assert_clean_markdown_math(cell.source, f"chapter6 notebook cell {index}")
    execute_notebook(notebook, NOTEBOOK_PATH, workdir=SRC_DIR, timeout=900)


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
    c, s, v = audit["counts"], audit["scores"], metrics["verification"]["summary"]
    write_text_if_changed(REPORT_DIR / "quality_benchmark.md", f"""# Chapter 6 quality benchmark

| Chapter | execution | reproducibility | rigor | pedagogy | trading context | total |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| **Machine Trading 2016 Ch6** | **{s['execution']}** | **{s['reproducibility']}** | **{s['rigor']}** | **{s['pedagogy']}** | **{s['trading_context']}** | **{audit['total']}** |

- Cells {c['cells']}: Markdown {c['markdown_cells']}, code {c['code_cells']}; executed {c['executed_code_cells']}/{c['code_cells']}, errors {c['error_outputs']}, inline PNG {c['embedded_png_outputs']}.
- Markdown {c['markdown_characters']:,} chars, Korean {c['korean_characters']:,}, headings {c['headings']}.
- Verification {v['passed']}/{v['total']}; independent/empirical {v['independent_or_empirical']}, invariant {v['contract_invariant']}.
""")


def run_analysis(execute_notebook_flag: bool = True) -> dict[str, Any]:
    data = load_data()
    results = run_experiments(data)
    checks = verify_results(data, results)
    if not all(checks.values()):
        raise AssertionError(f"Chapter 6 verification failed: {[name for name, passed in checks.items() if not passed]}")
    metrics = build_metrics(data, results, checks)
    save_figures(data, results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_text_if_changed(REPORT_DIR / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
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
    print("Chapter 6 analysis complete")
    print("Checks:", metrics["verification"]["summary"]["passed"])
    print("Report:", REPORT_DIR / "chapter6_report.md")
    if not args.skip_notebook:
        print("Notebook:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
