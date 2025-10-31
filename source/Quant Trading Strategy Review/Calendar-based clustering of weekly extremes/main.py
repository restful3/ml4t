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
    """요일별 주간 고점/저점 출현 빈도 기록 클래스"""

    def __init__(self, expected_days: int = 5) -> None:
        self.expected_days = expected_days
        self.current_week: Optional[Tuple[int, int]] = None  # (연도, ISO 주차)
        self.daily_records = []  # [(datetime, high, low)] 각 일자별 고저 기록
        self.high_counts = defaultdict(int)  # 요일별 최고가 카운트
        self.low_counts = defaultdict(int)   # 요일별 최저가 카운트
        self.sample_weeks = 0                # 전체 분석된 주간 수 (완벽한 5거래일 주)
        self.skipped_weeks = 0               # 불완전(5일이 아닌) 주간 수
        self.week_history = []  # [(week_key, high_day, low_day)] 각 주 요약 데이터

    def add_bar(self, bar_time, high, low) -> Optional[dict]:
        week_key = bar_time.isocalendar()[:2]
        info = None
        if self.current_week is None:
            self.current_week = week_key
        elif week_key != self.current_week:
            # 새 주로 넘어가면 이전 주 finalize
            info = self._finalize_week()
            self.current_week = week_key
        self.daily_records.append((bar_time, high, low))
        return info

    def finalize(self) -> Optional[dict]:
        # 알고리즘 끝 또는 데이터끝에서 주 마감
        return self._finalize_week()

    def _finalize_week(self) -> Optional[dict]:
        # 현재 주간 기록이 없으면 스킵
        if not self.daily_records:
            self.current_week = None
            return None

        if len(self.daily_records) != self.expected_days:
            # 5거래일이 아니면 카운트만 하고 무시
            self.skipped_weeks += 1
            self.daily_records = []
            self.current_week = None
            return None

        # 기록 정렬 (혹시 순서가 바뀔 수 있으므로)
        self.daily_records.sort(key=lambda item: item[0])
        high_record = max(self.daily_records, key=lambda item: item[1])
        low_record = min(self.daily_records, key=lambda item: item[2])

        high_day = high_record[0].weekday()  # 최고가 발생한 요일
        low_day = low_record[0].weekday()    # 최저가 발생한 요일

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
        # 누적 통계 반환
        return {
            "sample_weeks": self.sample_weeks,
            "skipped_weeks": self.skipped_weeks,
            "high_counts": dict(self.high_counts),
            "low_counts": dict(self.low_counts),
        }


class ExtremeTransitionModel:
    """요일별 마코프 전이 기반 주간 극단값(고/저점) 예측 모델"""

    def __init__(self, weekdays: int = 5, alpha: float = 0.75) -> None:
        self.weekdays = weekdays           # 거래 요일 수 (월~금)
        self.alpha = alpha                 # 래플라스 스무딩(초기값)
        self.low_initial = [alpha] * weekdays         # 첫주 저점 확률 분포
        self.low_transitions = [[alpha] * weekdays for _ in range(weekdays)]  # 저점 마코프 전이 행렬
        self.high_initial = [alpha] * weekdays        # 첫주 고점 확률 분포
        self.high_transitions = [[alpha] * weekdays for _ in range(weekdays)] # 고점 마코프 전이 행렬
        self.last_low_day: Optional[int] = None
        self.last_high_day: Optional[int] = None
        self.observed_weeks = 0           # 관찰 주 수

    def observe(self, high_day: int, low_day: int) -> None:
        # 한 주 완료 시 마코프 카운트 누적
        self._observe_low(low_day)
        self._observe_high(high_day)
        self.observed_weeks += 1

    def _observe_low(self, day: int) -> None:
        if self.last_low_day is None:
            self.low_initial[day] += 1    # 첫주면 초기 확률 업데이트
        else:
            self.low_transitions[self.last_low_day][day] += 1 # 전주->금주 전이 확률
        self.last_low_day = day

    def _observe_high(self, day: int) -> None:
        if self.last_high_day is None:
            self.high_initial[day] += 1
        else:
            self.high_transitions[self.last_high_day][day] += 1
        self.last_high_day = day

    def predict_low(self) -> Optional[dict]:
        # 다음 주 저점 출현 요일/확률 예측
        return self._predict(self.last_low_day, self.low_initial, self.low_transitions)

    def predict_high(self) -> Optional[dict]:
        # 다음 주 고점 출현 요일/확률 예측
        return self._predict(self.last_high_day, self.high_initial, self.high_transitions)

    def _predict(self, last_day: Optional[int], initial_counts, transition_counts) -> Optional[dict]:
        # 마지막 요일 기준(없으면 초기분포) 다음 주 요일별 확률 예측
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
        edge = best_prob - uniform_prob   # 균등확률 대비 우위
        gap = best_prob - runner_prob     # 1위-2위 확률차

        return {
            "day": best_day,        # 가장 유력한 요일
            "prob": best_prob,      # 해당 요일 확률
            "edge": edge,           # 균등대비 우위
            "gap": gap,             # 1,2위 확률차
            "probs": probs,         # 전체 요일 확률 분포
        }

    def propose_plan(self, min_weeks: int, bias_threshold: float, gap_threshold: float) -> Optional[dict]:
        # 일정 데이터/신호 강도 이상이면 매매 플랜 제안
        if self.observed_weeks < min_weeks:
            return None

        entry = self.predict_low()  # 진입(저점) 요일 추천
        if entry is None or entry["edge"] < bias_threshold or entry["gap"] < gap_threshold:
            return None

        exit_bias = self.predict_high()  # 청산(고점) 요일 추천
        exit_day = None
        # 고점 신호도 일정 수준 넘으면 고점 요일을 exit로 사용
        if exit_bias and exit_bias["edge"] >= bias_threshold / 2.0 and exit_bias["gap"] >= gap_threshold / 2.0:
            exit_day = exit_bias["day"]

        # exit가 entry와 같거나 의미 없다면 +1일(최대 금요일)
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
    """요일별 극단값(고저점) 클러스터링 기반 SPY 트레이딩 전략"""

    def Initialize(self) -> None:
        self.SetStartDate(2005, 1, 3)       # 백테스트 시작일
        self.SetEndDate(2024, 12, 31)       # 백테스트 종료일
        self.SetCash(100_000)               # 시작 자금
        self.SetWarmUp(timedelta(days=60))  # 60일 워밍업

        self.min_history_weeks = 26         # 플랜 제안 최소 관찰 주 수
        self.bias_threshold = 0.08          # 미는 요일 신호 강도(edge) 기준
        self.gap_threshold = 0.05           # 1,2위 요일 gap 기준
        self.position_size = 0.8            # 투자 비중
        self.stop_atr_multiplier = 1.5      # 스톱로스 ATR배수
        self.fallback_stop_pct = 0.03       # ATR 미적용시 백업 스톱로스(%) 
        self.atr_period = 10                # ATR 기간

        self.SetSecurityInitializer(
            lambda security: security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        )

        equity = self.AddEquity("SPY", Resolution.Daily)
        self.symbol = equity.Symbol

        self.tracker = WeeklyExtremeTracker(expected_days=5)     # 주간 극단값 트래커
        self.model = ExtremeTransitionModel(weekdays=5, alpha=0.75)
        self.atr = self.ATR(self.symbol, self.atr_period, MovingAverageType.Simple, Resolution.Daily)

        self.next_plan: Optional[dict] = None                    # 다음주 트레이딩 플랜
        self.current_plan: Optional[dict] = None                 # 이번주 플랜
        self.week_state: Dict[str, Optional[object]] = {         # 현재 주 상태(진입 여부 등)
            "week_key": None,
            "entered": False,
            "stopped": False,
        }
        self.stop_ticket = None          # 스톱로스 주문 객체

        self.day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}  # 요일 이름

        # 스케줄: 주말 데이터 마감 후 주간 데이터 정리
        self.Schedule.On(
            self.DateRules.WeekEnd(self.symbol),
            self.TimeRules.AfterMarketClose(self.symbol, 1),
            self._finalize_week_data,
        )
        # 스케줄: 새로운 주 시작 직전 다음주 플랜 활성화
        self.Schedule.On(
            self.DateRules.WeekStart(self.symbol),
            self.TimeRules.BeforeMarketOpen(self.symbol, 5),
            self._activate_week_plan,
        )
        # 매일 시장 열림 직후 매매로직 트리거
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 1),
            self._handle_trading,
        )

    def OnData(self, slice) -> None:
        # 새 바(bar) 데이터 처리
        if self.symbol in slice.Bars:
            bar = slice.Bars[self.symbol]
        elif slice.QuoteBars.ContainsKey(self.symbol):
            bar = slice.QuoteBars[self.symbol]
        else:
            return

        info = self.tracker.add_bar(bar.EndTime, bar.High, bar.Low)
        if info is not None:
            # 주간 데이터 완성 시 모델 업데이트 & 새로운 플랜 제안
            self.model.observe(info["high_day"], info["low_day"])
            self.next_plan = self.model.propose_plan(
                self.min_history_weeks, self.bias_threshold, self.gap_threshold
            )

    def OnEndOfAlgorithm(self) -> None:
        # 백테스트 종료 시 마지막 데이터 정리 및 로그 출력
        info = self.tracker.finalize()
        if info is not None:
            self.model.observe(info["high_day"], info["low_day"])
        self._cancel_stop()
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)
        self._log_statistics(self.tracker.summary())

    def _finalize_week_data(self) -> None:
        # 주간 마감 후 미체결 포지션 정리 및 다음주 플랜 계산
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
        # 새로운 한 주 진입 시 플랜 갱신
        self._cancel_stop()
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)

        week_key = self.Time.isocalendar()[:2]
        self.week_state = {"week_key": week_key, "entered": False, "stopped": False}
        self.current_plan = self.next_plan

    def _handle_trading(self) -> None:
        # 실제 매수/매도 집행 관리
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
            # 보유 중: 청산(고점 추천 요일 혹은 금요일)
            if weekday == exit_day or weekday == 4:
                self.Liquidate(self.symbol)
                self._cancel_stop()
                state["entered"] = False
                state["stopped"] = False
            return

        if state.get("stopped"):
            # 이미 스톱로스로 청산된 주는 skip
            return

        if weekday != entry_day:
            # 진입 추천 요일이 아니면 무시
            return

        if not self.atr.IsReady:
            # ATR 지표 준비 안 됨
            return

        quantity = self.CalculateOrderQuantity(self.symbol, self.position_size)
        if quantity <= 0:
            return

        ticket = self.MarketOrder(self.symbol, quantity)
        if ticket is None:
            return

        state["entered"] = True

        # 스톱로스 주문 발주(ATR 기준 손절가)
        price = self.Securities[self.symbol].Price
        atr_value = self.atr.Current.Value
        stop_distance = atr_value * self.stop_atr_multiplier
        if stop_distance <= 0:
            stop_distance = price * self.fallback_stop_pct
        stop_price = max(0.01, price - stop_distance)
        self.stop_ticket = self.StopMarketOrder(self.symbol, -quantity, stop_price)

    def _cancel_stop(self) -> None:
        # 미체결 스톱로스 주문 취소
        if self.stop_ticket is None:
            return
        if self.stop_ticket.Status not in (OrderStatus.Filled, OrderStatus.Canceled):
            self.stop_ticket.Cancel()
        self.stop_ticket = None

    def OnOrderEvent(self, order_event) -> None:
        # 주문 체결 이벤트 대응 (스톱로스가 체결되면 포지션 out 상태 기록)
        if order_event.Status != OrderStatus.Filled:
            return

        if self.stop_ticket is not None and order_event.OrderId == self.stop_ticket.OrderId:
            self.stop_ticket = None
            if order_event.Direction == OrderDirection.Sell:
                self.week_state["entered"] = False
                self.week_state["stopped"] = True

    def _log_statistics(self, stats: dict) -> None:
        # 전체 백테스트 요약 통계 출력
        sample_weeks = stats["sample_weeks"]
        if sample_weeks == 0:
            self.Log("SPY: insufficient 5-day trading weeks collected.")
            return

        skipped = stats["skipped_weeks"]
        high_counts = self._ordered_counts(stats["high_counts"])
        low_counts = self._ordered_counts(stats["low_counts"])

        high_kl = self._kl_divergence(high_counts, sample_weeks)  # 고점의 정보이득
        low_kl = self._kl_divergence(low_counts, sample_weeks)
        high_g, high_p = self._g_test(high_counts, sample_weeks) # 고점의 G-test(카이제곱 근사)
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
        # 요일별 카운트 리스트화(0:월~4:금)
        return [counts_dict.get(day, 0) for day in range(5)]

    def _format_counts(self, counts_list):
        # 보기 좋게 요일별 카운트 포맷팅
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        parts = [f"{day_names[idx]}:{count}" for idx, count in enumerate(counts_list)]
        return ", ".join(parts)

    def _kl_divergence(self, counts_list, total_weeks) -> float:
        # KL-발산(요일별 분포 vs. 균등 분포)
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
        # G-테스트(로그형 카이제곱 테스트)
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
        # 자유도 기반 chi2 p-value 계산
        if statistic <= 0 or dof <= 0:
            return None
        try:
            from scipy.stats import chi2  # type: ignore

            return float(1.0 - chi2.cdf(statistic, dof))
        except Exception:
            return None
