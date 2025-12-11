# https://quantpedia.com/strategies/time-series-momentum-effect/
#
# Time Series Momentum Strategy adapted for US Stock Market
#
# 전략:
# 1. 미국 거래대금 상위 500개 주식 선택 (Universe)
# 2. 모멘텀 상위 10% 선택 (~50개)
# 3. 역변동성 가중치 적용 (Inverse Volatility Weighting)
#    - 변동성이 낮을수록 높은 비중 할당
#    - weight = (1/volatility) / sum(1/volatilities)

from math import sqrt
from AlgorithmImports import *
import numpy as np
import pandas as pd

class TimeSeriesMomentum(QCAlgorithm):
    def Initialize(self) -> None:
        self.SetStartDate(2022, 1, 1)
        self.SetCash(10_000_000)

        # 벤치마크 설정: SPY (S&P 500 ETF)
        self.SetBenchmark("SPY")

        # Universe Selection: 미국 시가총액 상위 500개 주식
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        self.period: int = 12 * 21     # 모멘텀 산출 기간 (12개월 * 21거래일 = 약 252일)
        self.SetWarmUp(self.period, Resolution.Daily) # 데이터가 충분히 쌓일 때까지

        self.vol_target_period: int = 60      # 변동성 계산 기간 (최근 60일, 약 3개월)
        self.num_stocks: int = 10             # 최종 선택할 종목 수

        # Daily rolled data.
        self.data: Dict[Symbol, RollingWindow] = {}

        self.recent_month: int = -1  # 리밸런싱 시점을 체크하기 위한 변수
        # 포트폴리오 비중이 작아도 주문이 나가도록 최소 마진율 0으로 설정
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.
        self.Settings.daily_precise_end_time = False


    def CoarseSelectionFunction(self, coarse):
        """미국 시가총액 상위 500개 주식 선택"""
        # 펀더멘탈 데이터와 거래량이 있는 주식만 필터링
        filtered = [x for x in coarse if x.HasFundamentalData and x.DollarVolume > 0 and x.Price > 5]

        # 거래대금(DollarVolume) 기준 내림차순 정렬 후 상위 500개 선택
        # DollarVolume = 시가총액의 좋은 프록시
        sorted_by_dollar_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        top_500 = sorted_by_dollar_volume[:500]

        # 선택된 심볼들
        selected_symbols = [x.Symbol for x in top_500]

        # 새로 선택된 심볼에 대해 RollingWindow 초기화
        for symbol in selected_symbols:
            if symbol not in self.data:
                self.data[symbol] = RollingWindow[float](self.period)

        # 제거된 심볼의 데이터 정리
        symbols_to_remove = [s for s in self.data.keys() if s not in selected_symbols]
        for symbol in symbols_to_remove:
            if symbol in self.data:
                del self.data[symbol]

        return selected_symbols


    def OnData(self, slice: Slice) -> None:
        # 데이터 수집
        # 1. 매일 들어오는 가격 데이터를 RollingWindow에 저장
        for symbol in self.data.keys():
            if slice.Bars.ContainsKey(symbol):
                price = slice.Bars[symbol].Close
                self.data[symbol].Add(price)  # 최신 가격이 인덱스 0에 저장됨

        # 2. 월간 리밸런싱 체크: 달(Month)이 바뀌지 않았으면 리턴(아무것도 안 함)
        if self.recent_month == self.Time.month:
            return
        self.recent_month = self.Time.month  # 현재 달 업데이트

        # Performance and volatility data.
        performance_volatility: Dict[Symbol, Tuple[float, float]] = {}
        daily_returns: Dict[Symbol, np.ndarray] = {}


        for symbol in self.data.keys():
            if self.data[symbol].IsReady:
                # RollingWindow를 numpy 배열로 변환 (e.g.0번이 최신, -1번이 1년 전)
                back_adjusted_prices: np.ndarray = np.array([x for x in self.data[symbol]])
                # [모멘텀 신호] 12개월 수익률 계산: (현재가 / 12개월전 가격) - 1
                performance: float = back_adjusted_prices[0] / back_adjusted_prices[-1] - 1
                # 전체 기간 일일 수익률 계산
                daily_rets: np.ndarray = back_adjusted_prices[:-1] / back_adjusted_prices[1:] - 1

                # [변동성 계산용 데이터] 최근 60일치(3개월) 데이터만 슬라이싱
                back_adjusted_prices: np.ndarray = back_adjusted_prices[:self.vol_target_period]
                daily_rets: np.ndarray = back_adjusted_prices[:-1] / back_adjusted_prices[1:] - 1
                # 연환산 변동성 계산 (일일 수익률 표준편차 * sqrt(252))
                volatility_3M: float = np.std(daily_rets) * sqrt(252)

                # 변동성이 0이거나 너무 작으면 건너뛰기
                if volatility_3M < 0.001:
                    continue

                # 나중에 공분산 계산을 위해 일일 수익률 저장 (순서를 시간순으로 뒤집음 [::-1])
                daily_returns[symbol] = daily_rets[::-1][:self.vol_target_period]
                # (모멘텀 수익률, 3개월 변동성) 저장
                performance_volatility[symbol] = (performance, volatility_3M)

        if len(performance_volatility) == 0:
            return

        # Step 1: 모멘텀 기준으로 상위 10% 선택 (~50개) - 1차 필터링
        sorted_by_momentum = sorted(performance_volatility.items(), key=lambda x: x[1][0], reverse=True)
        top_momentum_count = max(1, int(len(sorted_by_momentum) * 0.1))  # 상위 10%
        top_momentum_stocks = sorted_by_momentum[:top_momentum_count]

        # Step 2: 모멘텀 상위 10% 중에서 변동성 기준 하위 정렬 - 2차 필터링
        sorted_by_volatility = sorted(top_momentum_stocks, key=lambda x: x[1][1])

        # Step 3: 변동성 낮은 하위 10개 선택
        low_volatility_stocks = sorted_by_volatility[:self.num_stocks]
        selected_symbols = [x[0] for x in low_volatility_stocks]

        # Step 4: 역변동성 가중치 계산 (Inverse Volatility Weighting)
        # 변동성이 낮을수록 높은 비중을 받음
        inv_vol_sum = sum([1/performance_volatility[s][1] for s in selected_symbols])

        # Trade execution.
        # 기존 포트폴리오 중 선택되지 않은 종목은 청산
        invested: List[Symbol] = [x.Key for x in self.Portfolio if x.Value.Invested]
        for symbol in invested:
            if symbol not in selected_symbols:
                self.Liquidate(symbol)

        # 역변동성 비중으로 매수
        for symbol in selected_symbols:
            if slice.Bars.ContainsKey(symbol):
                vol = performance_volatility[symbol][1]
                weight = (1 / vol) / inv_vol_sum  # 변동성 낮을수록 높은 비중
                self.SetHoldings(symbol, weight)
