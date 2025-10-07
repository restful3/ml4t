from collections import defaultdict
from datetime import timedelta
import math
from typing import Dict, Optional, Tuple

from QuantConnect import DataNormalizationMode, Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Market import QuoteBar
from QuantConnect.Indicators import MovingAverageType
from QuantConnect.Orders import OrderDirection, OrderStatus


class WeeklyExtremeTracker:
    """Collects weekday occurrences of weekly highs and lows."""

    def __init__(self, expected_days: int = 5) -> None:
        self.expected_days = expected_days
        self.current_week: Optional[Tuple[int, int]] = None  # (year, iso week)
        self.daily_records = []  # list[(datetime, high, low)]
        self.high_counts = defaultdict(int)
        self.low_counts = defaultdict(int)
        self.sample_weeks = 0
        self.skipped_weeks = 0
        self.week_history = []  # list[(week_key, high_day, low_day)]

    def add_bar(self, bar_time, high, low) -> Optional[dict]:
        week_key = bar_time.isocalendar()[:2]
        info = None
        if self.current_week is None:
            self.current_week = week_key
        elif week_key != self.current_week:
            info = self._finalize_week()
            self.current_week = week_key
        self.daily_records.append((bar_time, high, low))
        return info

    def finalize(self) -> Optional[dict]:
        return self._finalize_week()

    def _finalize_week(self) -> Optional[dict]:
        if not self.daily_records:
            self.current_week = None
            return None

        if len(self.daily_records) != self.expected_days:
            self.skipped_weeks += 1
            self.daily_records = []
            self.current_week = None
            return None

        self.daily_records.sort(key=lambda item: item[0])
        high_record = max(self.daily_records, key=lambda item: item[1])
        low_record = min(self.daily_records, key=lambda item: item[2])

        high_day = high_record[0].weekday()
        low_day = low_record[0].weekday()

        self.high_counts[high_day] += 1
        self.low_counts[low_day] += 1
        self.sample_weeks += 1
        info = {
            "week_key": self.current_week,
            "high_day": high_day,
            "low_day": low_day,
            "start": self.daily_records[0][0],
            "end": self.daily_records[-1][0],
        }
        self.week_history.append((self.current_week, high_day, low_day))

        self.daily_records = []
        self.current_week = None
        return info

    def summary(self):
        return {
            "sample_weeks": self.sample_weeks,
            "skipped_weeks": self.skipped_weeks,
            "high_counts": dict(self.high_counts),
            "low_counts": dict(self.low_counts),
        }


class ExtremeTransitionModel:
    """Implements weekday-dependent Markov transitions for weekly extremes."""

    def __init__(self, weekdays: int = 5, alpha: float = 0.75) -> None:
        self.weekdays = weekdays
        self.alpha = alpha
        self.low_initial = [alpha] * weekdays
        self.low_transitions = [[alpha] * weekdays for _ in range(weekdays)]
        self.high_initial = [alpha] * weekdays
        self.high_transitions = [[alpha] * weekdays for _ in range(weekdays)]
        self.last_low_day: Optional[int] = None
        self.last_high_day: Optional[int] = None
        self.observed_weeks = 0

    def observe(self, high_day: int, low_day: int) -> None:
        self._observe_low(low_day)
        self._observe_high(high_day)
        self.observed_weeks += 1

    def _observe_low(self, day: int) -> None:
        if self.last_low_day is None:
            self.low_initial[day] += 1
        else:
            self.low_transitions[self.last_low_day][day] += 1
        self.last_low_day = day

    def _observe_high(self, day: int) -> None:
        if self.last_high_day is None:
            self.high_initial[day] += 1
        else:
            self.high_transitions[self.last_high_day][day] += 1
        self.last_high_day = day

    def predict_low(self) -> Optional[dict]:
        return self._predict(self.last_low_day, self.low_initial, self.low_transitions)

    def predict_high(self) -> Optional[dict]:
        return self._predict(self.last_high_day, self.high_initial, self.high_transitions)

    def _predict(self, last_day: Optional[int], initial_counts, transition_counts) -> Optional[dict]:
        counts = initial_counts if last_day is None else transition_counts[last_day]
        total = sum(counts)
        if total <= 0:
            return None

        probs = [count / total for count in counts]
        best_day = max(range(self.weekdays), key=lambda idx: probs[idx])
        best_prob = probs[best_day]
        other_probs = [probs[idx] for idx in range(self.weekdays) if idx != best_day]
        runner_prob = max(other_probs) if other_probs else 0.0
        uniform_prob = 1.0 / self.weekdays
        edge = best_prob - uniform_prob
        gap = best_prob - runner_prob

        return {
            "day": best_day,
            "prob": best_prob,
            "edge": edge,
            "gap": gap,
            "probs": probs,
        }

    def propose_plan(self, min_weeks: int, bias_threshold: float, gap_threshold: float) -> Optional[dict]:
        if self.observed_weeks < min_weeks:
            return None

        entry = self.predict_low()
        if entry is None or entry["edge"] < bias_threshold or entry["gap"] < gap_threshold:
            return None

        exit_bias = self.predict_high()
        exit_day = None
        if exit_bias and exit_bias["edge"] >= bias_threshold / 2.0 and exit_bias["gap"] >= gap_threshold / 2.0:
            exit_day = exit_bias["day"]

        if exit_day is None or exit_day == entry["day"]:
            exit_day = min(4, entry["day"] + 1) if entry["day"] < 4 else 4
        if exit_day <= entry["day"]:
            exit_day = min(4, entry["day"] + 1) if entry["day"] < 4 else 4

        return {
            "entry_day": entry["day"],
            "exit_day": exit_day,
            "entry_details": entry,
            "exit_details": exit_bias,
        }


class CalendarExtremeClustering(QCAlgorithm):
    """SPY trading strategy based on calendar-week extreme clustering."""

    def Initialize(self) -> None:
        self.SetStartDate(2005, 1, 3)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        self.SetWarmUp(timedelta(days=60))

        self.min_history_weeks = 26
        self.bias_threshold = 0.08
        self.gap_threshold = 0.05
        self.position_size = 0.8
        self.stop_atr_multiplier = 1.5
        self.fallback_stop_pct = 0.03
        self.atr_period = 10

        self.SetSecurityInitializer(
            lambda security: security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        )

        equity = self.AddEquity("SPY", Resolution.Daily)
        self.symbol = equity.Symbol

        self.tracker = WeeklyExtremeTracker(expected_days=5)
        self.model = ExtremeTransitionModel(weekdays=5, alpha=0.75)
        self.atr = self.ATR(self.symbol, self.atr_period, MovingAverageType.Simple, Resolution.Daily)

        self.next_plan: Optional[dict] = None
        self.current_plan: Optional[dict] = None
        self.week_state: Dict[str, Optional[object]] = {
            "week_key": None,
            "entered": False,
            "stopped": False,
        }
        self.stop_ticket = None

        self.day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

        self.Schedule.On(
            self.DateRules.WeekEnd(self.symbol),
            self.TimeRules.AfterMarketClose(self.symbol, 1),
            self._finalize_week_data,
        )
        self.Schedule.On(
            self.DateRules.WeekStart(self.symbol),
            self.TimeRules.BeforeMarketOpen(self.symbol, 5),
            self._activate_week_plan,
        )
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 1),
            self._handle_trading,
        )

    def OnData(self, slice) -> None:
        if self.symbol in slice.Bars:
            bar = slice.Bars[self.symbol]
        elif slice.QuoteBars.ContainsKey(self.symbol):
            bar = slice.QuoteBars[self.symbol]
        else:
            return

        info = self.tracker.add_bar(bar.EndTime, bar.High, bar.Low)
        if info is not None:
            self.model.observe(info["high_day"], info["low_day"])
            self.next_plan = self.model.propose_plan(
                self.min_history_weeks, self.bias_threshold, self.gap_threshold
            )

    def OnEndOfAlgorithm(self) -> None:
        info = self.tracker.finalize()
        if info is not None:
            self.model.observe(info["high_day"], info["low_day"])
        self._cancel_stop()
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)
        self._log_statistics(self.tracker.summary())

    def _finalize_week_data(self) -> None:
        self._cancel_stop()
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)

        if self.tracker.current_week is None:
            return

        info = self.tracker.finalize()
        if info is None:
            self.next_plan = None
            self.Log("Week incomplete (<5 trading days); skipping next-week plan.")
            return

        self.model.observe(info["high_day"], info["low_day"])
        self.next_plan = self.model.propose_plan(
            self.min_history_weeks, self.bias_threshold, self.gap_threshold
        )
        if self.next_plan is not None:
            entry_day = self.day_names[self.next_plan["entry_day"]]
            exit_day = self.day_names.get(self.next_plan["exit_day"], "Fri")
            entry_edge = self.next_plan["entry_details"]["edge"]
            self.Log(
                f"Next plan: entry {entry_day} (edge={entry_edge:.3f}), exit {exit_day};"
                f" model weeks={self.model.observed_weeks}"
            )
        else:
            self.Log("Next plan: insufficient signal or history; staying flat.")

    def _activate_week_plan(self) -> None:
        self._cancel_stop()
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)

        week_key = self.Time.isocalendar()[:2]
        self.week_state = {"week_key": week_key, "entered": False, "stopped": False}
        self.current_plan = self.next_plan

    def _handle_trading(self) -> None:
        if self.IsWarmingUp or self.current_plan is None:
            return

        state = self.week_state
        if state.get("week_key") != self.Time.isocalendar()[:2]:
            return

        weekday = self.Time.weekday()
        if weekday > 4:
            return

        entry_day = self.current_plan["entry_day"]
        exit_day = self.current_plan["exit_day"]
        holdings = self.Portfolio[self.symbol].Quantity

        if holdings > 0:
            if weekday == exit_day or weekday == 4:
                self.Liquidate(self.symbol)
                self._cancel_stop()
                state["entered"] = False
                state["stopped"] = False
            return

        if state.get("stopped"):
            return

        if weekday != entry_day:
            return

        if not self.atr.IsReady:
            return

        quantity = self.CalculateOrderQuantity(self.symbol, self.position_size)
        if quantity <= 0:
            return

        ticket = self.MarketOrder(self.symbol, quantity)
        if ticket is None:
            return

        state["entered"] = True

        price = self.Securities[self.symbol].Price
        atr_value = self.atr.Current.Value
        stop_distance = atr_value * self.stop_atr_multiplier
        if stop_distance <= 0:
            stop_distance = price * self.fallback_stop_pct
        stop_price = max(0.01, price - stop_distance)
        self.stop_ticket = self.StopMarketOrder(self.symbol, -quantity, stop_price)

    def _cancel_stop(self) -> None:
        if self.stop_ticket is None:
            return
        if self.stop_ticket.Status not in (OrderStatus.Filled, OrderStatus.Canceled):
            self.stop_ticket.Cancel()
        self.stop_ticket = None

    def OnOrderEvent(self, order_event) -> None:
        if order_event.Status != OrderStatus.Filled:
            return

        if self.stop_ticket is not None and order_event.OrderId == self.stop_ticket.OrderId:
            self.stop_ticket = None
            if order_event.Direction == OrderDirection.Sell:
                self.week_state["entered"] = False
                self.week_state["stopped"] = True

    def _log_statistics(self, stats: dict) -> None:
        sample_weeks = stats["sample_weeks"]
        if sample_weeks == 0:
            self.Log("SPY: insufficient 5-day trading weeks collected.")
            return

        skipped = stats["skipped_weeks"]
        high_counts = self._ordered_counts(stats["high_counts"])
        low_counts = self._ordered_counts(stats["low_counts"])

        high_kl = self._kl_divergence(high_counts, sample_weeks)
        low_kl = self._kl_divergence(low_counts, sample_weeks)
        high_g, high_p = self._g_test(high_counts, sample_weeks)
        low_g, low_p = self._g_test(low_counts, sample_weeks)

        high_msg = self._format_counts(high_counts)
        low_msg = self._format_counts(low_counts)

        self.Log(
            f"SPY | weeks={sample_weeks} (skipped={skipped}) | Highs {high_msg} | G={high_g:.2f}"
            f" | KL={high_kl:.3f} | p={high_p if high_p is not None else 'n/a'}"
        )
        self.Log(
            f"SPY | weeks={sample_weeks} (skipped={skipped}) | Lows  {low_msg} | G={low_g:.2f}"
            f" | KL={low_kl:.3f} | p={low_p if low_p is not None else 'n/a'}"
        )

    def _ordered_counts(self, counts_dict):
        return [counts_dict.get(day, 0) for day in range(5)]

    def _format_counts(self, counts_list):
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        parts = [f"{day_names[idx]}:{count}" for idx, count in enumerate(counts_list)]
        return ", ".join(parts)

    def _kl_divergence(self, counts_list, total_weeks) -> float:
        if total_weeks == 0:
            return float("nan")
        uniform_prob = 1.0 / len(counts_list)
        divergence = 0.0
        for count in counts_list:
            if count == 0:
                continue
            p_i = count / total_weeks
            divergence += p_i * math.log(p_i / uniform_prob)
        return divergence

    def _g_test(self, counts_list, total_weeks):
        if total_weeks == 0:
            return 0.0, None
        expected = total_weeks / len(counts_list)
        if expected == 0:
            return 0.0, None
        g_stat = 0.0
        for observed in counts_list:
            if observed == 0:
                continue
            g_stat += observed * math.log(observed / expected)
        g_stat *= 2.0
        p_value = self._chi_square_sf(g_stat, len(counts_list) - 1)
        return g_stat, p_value

    def _chi_square_sf(self, statistic: float, dof: int):
        if statistic <= 0 or dof <= 0:
            return None
        try:
            from scipy.stats import chi2  # type: ignore

            return float(1.0 - chi2.cdf(statistic, dof))
        except Exception:
            return None
