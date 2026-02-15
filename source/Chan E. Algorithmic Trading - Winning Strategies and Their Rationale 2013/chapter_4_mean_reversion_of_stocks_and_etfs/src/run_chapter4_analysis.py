#!/usr/bin/env python3
"""
Chapter 4: 주식과 ETF의 평균 회귀 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 4의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. Buy-on-Gap 모델 (예제 4.1)
2. SPY와 구성 주식 간 인덱스 차익거래 (예제 4.2)
3. 횡단면 선형 롱-숏 모델 (예제 4.3)
4. 일중 선형 롱-숏 모델 (예제 4.4)
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 리포트 출력 설정
REPORT_DIR = Path(__file__).parent / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


def calculateMaxDD(cumret):
    """최대 낙폭(Maximum Drawdown) 계산

    Args:
        cumret: 누적 복리 수익률 배열

    Returns:
        maxDD: 최대 낙폭
        maxDDD: 최대 낙폭 기간 (일수)
        i: 최대 낙폭 발생 인덱스
    """
    highwatermark = np.zeros(cumret.shape)
    drawdown = np.zeros(cumret.shape)
    drawdownduration = np.zeros(cumret.shape)

    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t] = np.maximum(highwatermark[t-1], cumret.iloc[t] if hasattr(cumret, 'iloc') else cumret[t])
        val = cumret.iloc[t] if hasattr(cumret, 'iloc') else cumret[t]
        drawdown[t] = (1 + val) / (1 + highwatermark[t]) - 1
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1

    maxDD = np.min(drawdown)
    i = np.argmin(drawdown)
    maxDDD = np.max(drawdownduration)
    return maxDD, maxDDD, i


class Chapter4Analyzer:
    """Chapter 4 주식과 ETF의 평균 회귀 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.figures = []

        # 디렉토리 생성
        REPORT_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)

    def load_data(self):
        """데이터 로드 및 전처리"""
        print("=" * 60)
        print("📊 데이터 로드 중...")
        print("=" * 60)

        data_dir = Path(__file__).parent

        # 주식 티커 이름 로드
        stocks_path = data_dir / "inputDataOHLCDaily_20120424_stocks.csv"
        stocks_df = pd.read_csv(stocks_path)
        self.stock_names = stocks_df.iloc[0].values.tolist()
        print(f"  ✓ 주식 티커: {len(self.stock_names)}개 로드")

        # OHLC 데이터 로드 함수
        def load_price_file(filename):
            path = data_dir / filename
            df = pd.read_csv(path)
            # 첫 번째 컬럼은 Date
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
            df.set_index(date_col, inplace=True)
            df.index.name = 'Date'
            # 컬럼 이름을 실제 티커로 변환
            df.columns = self.stock_names
            return df

        # 종가, 시가, 고가, 저가 로드
        self.cl = load_price_file("inputDataOHLCDaily_20120424_cl.csv")
        self.op = load_price_file("inputDataOHLCDaily_20120424_op.csv")
        self.hi = load_price_file("inputDataOHLCDaily_20120424_hi.csv")
        self.lo = load_price_file("inputDataOHLCDaily_20120424_lo.csv")
        print(f"  ✓ OHLC 데이터: {len(self.cl)} 거래일 x {len(self.cl.columns)} 주식")
        print(f"    기간: {self.cl.index[0].strftime('%Y-%m-%d')} ~ {self.cl.index[-1].strftime('%Y-%m-%d')}")

        # ETF 데이터 로드
        etf_stocks_path = data_dir / "inputData_ETF_stocks.csv"
        etf_stocks_df = pd.read_csv(etf_stocks_path)
        self.etf_names = etf_stocks_df.iloc[0].values.tolist()

        etf_cl_path = data_dir / "inputData_ETF_cl.csv"
        etf_df = pd.read_csv(etf_cl_path)
        date_col = etf_df.columns[0]
        etf_df[date_col] = pd.to_datetime(etf_df[date_col], format='%Y%m%d')
        etf_df.set_index(date_col, inplace=True)
        etf_df.index.name = 'Date'
        etf_df.columns = self.etf_names
        self.etf_cl = etf_df
        print(f"  ✓ ETF 데이터: {len(self.etf_cl)} 거래일 x {len(self.etf_cl.columns)} ETF")

        # 어닝 발표 데이터 로드
        earnann_path = data_dir / "earnannFile.csv"
        if earnann_path.exists():
            self.earnann = pd.read_csv(earnann_path)
            date_col = self.earnann.columns[0]
            self.earnann[date_col] = pd.to_datetime(self.earnann[date_col], format='%Y%m%d')
            self.earnann.set_index(date_col, inplace=True)
            self.earnann.index.name = 'Date'
            print(f"  ✓ 어닝 발표 데이터: {len(self.earnann)} 거래일")
        else:
            self.earnann = None
            print(f"  ✗ 어닝 발표 데이터 없음")

        print()

    def analyze_buy_on_gap(self):
        """예제 4.1: Buy-on-Gap 모델 분석

        전일 저가 대비 갭 다운한 주식을 매수하고 당일 종가에 청산하는 일중 전략.
        모멘텀 필터(20일 MA)를 적용하여 추세 하락 주식을 제외.
        """
        print("=" * 60)
        print("📈 분석 1: Buy-on-Gap 모델 (예제 4.1)")
        print("=" * 60)

        topN = 10
        entryZscore = 1
        lookback = 20  # MA용

        # 종가 대 종가 수익률의 90일 롤링 표준편차 (전일 기준)
        retC2C = self.cl.pct_change()
        stdretC2C90d = retC2C.rolling(90).std().shift(1)

        # 매수 가격: 전일 저가 * (1 - entryZscore * std)
        buyPrice = self.lo.shift(1) * (1 - entryZscore * stdretC2C90d)

        # 갭 수익률: (시가 - 전일 저가) / 전일 저가
        retGap = (self.op - self.lo.shift(1)) / self.lo.shift(1)

        # 20일 이동평균 (전일 기준)
        ma = self.cl.rolling(lookback).mean().shift(1)

        pnl_list = []
        trade_counts = []

        for t in range(1, len(self.cl)):
            # 조건: 데이터 존재 + 시가 < 매수가격 + 시가 > 20일 MA
            hasData = (retGap.iloc[t].notna() &
                      (self.op.iloc[t] < buyPrice.iloc[t]) &
                      (self.op.iloc[t] > ma.iloc[t]))

            valid_stocks = retGap.iloc[t][hasData].dropna()

            if len(valid_stocks) > 0:
                # 갭이 가장 큰 (가장 음의) 주식 topN개 선택
                sorted_stocks = valid_stocks.sort_values(ascending=True)
                selected = sorted_stocks.head(min(topN, len(sorted_stocks)))

                # 시가-종가 수익률
                retO2C = (self.cl.iloc[t] - self.op.iloc[t]) / self.op.iloc[t]
                daily_pnl = retO2C[selected.index].sum()
                pnl_list.append(daily_pnl / topN)
                trade_counts.append(len(selected))
            else:
                pnl_list.append(0.0)
                trade_counts.append(0)

        ret = pd.Series(pnl_list, index=self.cl.index[1:])
        ret = ret.fillna(0)

        cumret = (1 + ret).cumprod() - 1

        # 성과 지표 계산
        apr = (1 + ret).prod() ** (252 / len(ret)) - 1
        sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0
        maxDD, maxDDD, _ = calculateMaxDD(cumret)

        avg_trades = np.mean([c for c in trade_counts if c > 0]) if any(c > 0 for c in trade_counts) else 0
        active_days = sum(1 for c in trade_counts if c > 0)

        self.results['bog'] = {
            'apr': apr,
            'sharpe': sharpe,
            'maxDD': maxDD,
            'maxDDD': maxDDD,
            'avg_trades_per_day': avg_trades,
            'active_trading_days': active_days,
            'total_days': len(ret),
        }

        print(f"  APR = {apr:.4f} ({apr*100:.2f}%)")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD:.4f} ({maxDD*100:.2f}%)")
        print(f"  Max DD Duration = {maxDDD:.0f} 일")
        print(f"  활성 거래일 = {active_days}/{len(ret)}, 평균 종목수 = {avg_trades:.1f}")

        # 차트 생성
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('Buy-on-Gap Model - Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linewidth=0.5)

        # 일별 거래 수
        trade_series = pd.Series(trade_counts, index=self.cl.index[1:])
        axes[1].bar(trade_series.index, trade_series.values, color='steelblue', alpha=0.5, width=3)
        axes[1].set_title('Daily Number of Trades', fontsize=14)
        axes[1].set_ylabel('Number of Stocks')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch4_buy_on_gap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch4_buy_on_gap.png', 'Buy-on-Gap 모델 누적 수익률'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_index_arbitrage(self):
        """예제 4.2: SPY와 구성 주식 간 인덱스 차익거래

        요한센 공적분 검정으로 SPY와 개별적으로 공적분하는 주식을 찾고,
        이들의 롱온리 포트폴리오와 SPY 사이에 선형 평균 회귀 전략을 적용.
        """
        print("=" * 60)
        print("📈 분석 2: SPY 인덱스 차익거래 (예제 4.2)")
        print("=" * 60)

        # SPY 데이터 추출
        if 'SPY' not in self.etf_cl.columns:
            print("  ✗ SPY 데이터가 ETF 파일에 없습니다")
            return

        spy = self.etf_cl[['SPY']].copy()

        # 공통 날짜로 병합
        common_dates = self.cl.index.intersection(spy.index)
        cl_common = self.cl.loc[common_dates].copy()
        spy_common = spy.loc[common_dates].copy()

        # 훈련/테스트 기간 설정
        train_mask = (common_dates >= '2007-01-01') & (common_dates <= '2007-12-31')
        test_mask = common_dates > '2007-12-31'

        train_dates = common_dates[train_mask]
        test_dates = common_dates[test_mask]

        print(f"  훈련 기간: {train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)}일)")
        print(f"  테스트 기간: {test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)}일)")

        # 각 주식과 SPY의 공적분 검정 (훈련 세트)
        isCoint = {}
        spy_train = spy_common.loc[train_dates, 'SPY'].values

        tested = 0
        coint_count = 0
        for stock in cl_common.columns:
            stock_train = cl_common.loc[train_dates, stock].values
            y2 = np.column_stack([stock_train, spy_train])

            # NaN 제거
            bad = np.any(np.isnan(y2), axis=1)
            y2_clean = y2[~bad]

            if y2_clean.shape[0] > 250:
                tested += 1
                try:
                    result = vm.coint_johansen(y2_clean, det_order=0, k_ar_diff=1)
                    # 90% 신뢰도에서 공적분 확인 (lr1 첫 번째 > cvt 첫 번째 행 첫 번째 열)
                    if result.lr1[0] > result.cvt[0, 0]:
                        isCoint[stock] = True
                        coint_count += 1
                except Exception:
                    pass

        coint_stocks = [s for s in isCoint.keys()]
        print(f"  검정 완료: {tested}개 주식 중 {coint_count}개가 SPY와 공적분")

        if coint_count < 5:
            print("  ✗ 공적분 주식이 너무 적어 분석을 건너뜁니다")
            self.results['indexArb'] = {'apr': 0, 'sharpe': 0, 'maxDD': 0, 'maxDDD': 0, 'coint_count': coint_count}
            return

        # 동일 자본 배분 롱온리 포트폴리오 구성 (로그 가격)
        coint_prices_train = cl_common.loc[train_dates, coint_stocks]
        logMktVal_train = np.log(coint_prices_train).sum(axis=1)

        # 포트폴리오와 SPY의 공적분 확인
        ytest = np.column_stack([logMktVal_train.values, np.log(spy_common.loc[train_dates, 'SPY'].values)])
        bad = np.any(np.isnan(ytest), axis=1)
        ytest_clean = ytest[~bad]

        try:
            result_port = vm.coint_johansen(ytest_clean, det_order=0, k_ar_diff=1)
            port_coint = result_port.lr1[0] > result_port.cvt[0, 1]  # 95% 신뢰도
            evec = result_port.evec[:, 0]  # 첫 번째 고유벡터
            print(f"  포트폴리오-SPY 공적분: {'Yes (95%)' if port_coint else 'No'}")
            print(f"  고유벡터: [{evec[0]:.4f}, {evec[1]:.4f}]")
        except Exception as e:
            print(f"  ✗ 포트폴리오 공적분 검정 실패: {e}")
            self.results['indexArb'] = {'apr': 0, 'sharpe': 0, 'maxDD': 0, 'maxDDD': 0, 'coint_count': coint_count}
            return

        # 테스트 기간에 선형 평균 회귀 전략 적용
        coint_prices_test = cl_common.loc[test_dates, coint_stocks]
        spy_test = spy_common.loc[test_dates, 'SPY']

        # 결합 데이터
        yNplus = pd.concat([coint_prices_test, spy_test], axis=1)

        # 가중치: 고유벡터 적용
        weights = pd.DataFrame(index=test_dates, columns=yNplus.columns)
        for col in coint_stocks:
            weights[col] = evec[0]
        weights['SPY'] = evec[1]
        weights = weights.astype(float)

        # 로그 시장 가치
        logMktVal = (weights * np.log(yNplus)).sum(axis=1)

        lookback = 5
        ma = logMktVal.rolling(lookback).mean()
        mstd = logMktVal.rolling(lookback).std()
        numUnits = -(logMktVal - ma) / mstd
        numUnits = numUnits.fillna(0)

        # 포지션 = numUnits * weights
        positions = weights.multiply(numUnits, axis=0)

        # PnL 계산 (로그 수익률 기반)
        log_prices = np.log(yNplus)
        log_ret = log_prices - log_prices.shift(1)

        pnl = (positions.shift(1) * log_ret).sum(axis=1)
        capital = positions.shift(1).abs().sum(axis=1)
        capital = capital.replace(0, np.nan)
        ret = pnl / capital
        ret = ret.fillna(0)

        # 초기 NaN 기간 제거
        ret = ret.iloc[lookback:]
        cumret = (1 + ret).cumprod() - 1

        apr = (1 + ret).prod() ** (252 / len(ret)) - 1
        sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0
        maxDD, maxDDD, _ = calculateMaxDD(cumret)

        self.results['indexArb'] = {
            'apr': apr,
            'sharpe': sharpe,
            'maxDD': maxDD,
            'maxDDD': maxDDD,
            'coint_count': coint_count,
            'port_coint': port_coint if 'port_coint' in dir() else False,
            'evec': evec.tolist(),
        }

        print(f"  APR = {apr:.4f} ({apr*100:.2f}%)")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD:.4f} ({maxDD*100:.2f}%)")
        print(f"  Max DD Duration = {maxDDD:.0f} 일")

        # 차트 생성
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('SPY Index Arbitrage - Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linewidth=0.5)

        # z-score 추적
        zScore = numUnits.iloc[lookback:]
        axes[1].plot(zScore.index, zScore.values, 'r-', linewidth=0.5, alpha=0.7)
        axes[1].set_title('Portfolio Z-Score', fontsize=14)
        axes[1].set_ylabel('Z-Score')
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].axhline(y=1, color='gray', linewidth=0.5, linestyle='--')
        axes[1].axhline(y=-1, color='gray', linewidth=0.5, linestyle='--')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch4_index_arbitrage.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch4_index_arbitrage.png', 'SPY 인덱스 차익거래 누적 수익률'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_cross_sectional_mean_reversion(self):
        """예제 4.3: 횡단면 선형 롱-숏 모델

        Khandani & Lo의 선형 롱-숏 모델. 각 주식의 일일 수익률에서
        시장 평균 수익률을 빼고, 이를 역방향으로 투자.
        """
        print("=" * 60)
        print("📈 분석 3: 횡단면 선형 롱-숏 모델 (예제 4.3)")
        print("=" * 60)

        # 2007-01-02 ~ 2011-12-30 필터링
        mask = (self.cl.index >= '2007-01-03') & (self.cl.index <= '2011-12-30')
        cl = self.cl.loc[mask].copy()
        op = self.op.loc[mask].copy()

        # === Close-to-Close 전략 (예제 4.3) ===
        print("\n  [Close-to-Close 전략]")
        ret = cl.pct_change()

        # 시장 수익률 (동일 가중 평균)
        marketRet = ret.mean(axis=1)

        # 가중치: -(개별수익률 - 시장수익률), 정규화
        weights = -(ret.subtract(marketRet, axis=0))
        abs_sum = weights.abs().sum(axis=1)
        abs_sum = abs_sum.replace(0, np.nan)
        weights = weights.div(abs_sum, axis=0)
        weights = weights.fillna(0)

        # 일일 수익률
        dailyret = (weights.shift(1) * ret).sum(axis=1)
        dailyret = dailyret.iloc[1:]  # 첫 날 제거
        dailyret = dailyret.fillna(0)

        cumret_c2c = (1 + dailyret).cumprod() - 1
        apr_c2c = (1 + dailyret).prod() ** (252 / len(dailyret)) - 1
        sharpe_c2c = np.sqrt(252) * dailyret.mean() / dailyret.std() if dailyret.std() > 0 else 0
        maxDD_c2c, maxDDD_c2c, _ = calculateMaxDD(cumret_c2c)

        print(f"    APR = {apr_c2c:.4f} ({apr_c2c*100:.2f}%)")
        print(f"    Sharpe = {sharpe_c2c:.4f}")
        print(f"    Max DD = {maxDD_c2c:.4f} ({maxDD_c2c*100:.2f}%)")

        # === Intraday 전략 (예제 4.4) - 오버나이트 수익률로 가중치 결정 ===
        print("\n  [Intraday 전략 (Close-to-Open -> Open-to-Close)]")
        retC2O = (op - cl.shift(1)) / cl.shift(1)
        marketRetC2O = retC2O.mean(axis=1)

        weights_intra = -(retC2O.subtract(marketRetC2O, axis=0))
        abs_sum_intra = weights_intra.abs().sum(axis=1)
        abs_sum_intra = abs_sum_intra.replace(0, np.nan)
        weights_intra = weights_intra.div(abs_sum_intra, axis=0)
        weights_intra = weights_intra.fillna(0)

        # Open-to-Close 수익률
        retO2C = (cl - op) / op
        capital_intra = weights_intra.abs().sum(axis=1).replace(0, np.nan)
        dailyret_intra = (weights_intra * retO2C).sum(axis=1) / capital_intra
        dailyret_intra = dailyret_intra.iloc[1:].fillna(0)

        cumret_intra = (1 + dailyret_intra).cumprod() - 1
        apr_intra = (1 + dailyret_intra).prod() ** (252 / len(dailyret_intra)) - 1
        sharpe_intra = np.sqrt(252) * dailyret_intra.mean() / dailyret_intra.std() if dailyret_intra.std() > 0 else 0
        maxDD_intra, maxDDD_intra, _ = calculateMaxDD(cumret_intra)

        print(f"    APR = {apr_intra:.4f} ({apr_intra*100:.2f}%)")
        print(f"    Sharpe = {sharpe_intra:.4f}")
        print(f"    Max DD = {maxDD_intra:.4f} ({maxDD_intra*100:.2f}%)")

        self.results['crossSectional'] = {
            'c2c': {
                'apr': apr_c2c,
                'sharpe': sharpe_c2c,
                'maxDD': maxDD_c2c,
                'maxDDD': maxDDD_c2c,
            },
            'intraday': {
                'apr': apr_intra,
                'sharpe': sharpe_intra,
                'maxDD': maxDD_intra,
                'maxDDD': maxDDD_intra,
            }
        }

        # 연도별 성과 분석
        print("\n  [연도별 성과 비교]")
        print(f"    {'연도':<8} {'C2C APR':>10} {'C2C Sharpe':>12} {'Intra APR':>12} {'Intra Sharpe':>14}")
        print(f"    {'-'*56}")

        yearly_results = {}
        for year in sorted(dailyret.index.year.unique()):
            yr_mask = dailyret.index.year == year
            yr_ret = dailyret[yr_mask]
            yr_apr = (1 + yr_ret).prod() ** (252/len(yr_ret)) - 1
            yr_sharpe = np.sqrt(252) * yr_ret.mean() / yr_ret.std() if yr_ret.std() > 0 else 0

            yr_mask_i = dailyret_intra.index.year == year
            yr_ret_i = dailyret_intra[yr_mask_i]
            yr_apr_i = (1 + yr_ret_i).prod() ** (252/len(yr_ret_i)) - 1 if len(yr_ret_i) > 0 else 0
            yr_sharpe_i = np.sqrt(252) * yr_ret_i.mean() / yr_ret_i.std() if len(yr_ret_i) > 0 and yr_ret_i.std() > 0 else 0

            yearly_results[year] = {
                'c2c_apr': yr_apr, 'c2c_sharpe': yr_sharpe,
                'intra_apr': yr_apr_i, 'intra_sharpe': yr_sharpe_i
            }
            print(f"    {year:<8} {yr_apr*100:>9.2f}% {yr_sharpe:>11.2f} {yr_apr_i*100:>11.2f}% {yr_sharpe_i:>13.2f}")

        self.results['crossSectional']['yearly'] = yearly_results

        # 차트 생성
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret_c2c.index, cumret_c2c.values * 100, 'b-', linewidth=1, label='Close-to-Close')
        axes[0].set_title('Cross-Sectional Mean Reversion: Close-to-Close (Example 4.3)', fontsize=14)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(cumret_intra.index, cumret_intra.values * 100, 'r-', linewidth=1, label='Intraday (C2O -> O2C)')
        axes[1].set_title('Cross-Sectional Mean Reversion: Intraday (Example 4.4)', fontsize=14)
        axes[1].set_ylabel('Cumulative Return (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch4_cross_sectional.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch4_cross_sectional.png', '횡단면 평균 회귀 전략 비교'))
        print(f"\n  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_pead(self):
        """PEAD (Post-Earnings Announcement Drift) 분석

        어닝 발표일에 갭이 큰 주식에 대해 일중 모멘텀 포지션.
        """
        print("=" * 60)
        print("📈 분석 4: 실적 발표 후 표류 (PEAD)")
        print("=" * 60)

        if self.earnann is None:
            print("  ✗ 어닝 발표 데이터가 없어 분석을 건너뜁니다")
            self.results['pead'] = None
            print()
            return

        # 공통 날짜와 주식
        common_dates = self.cl.index.intersection(self.earnann.index)
        common_stocks = [s for s in self.cl.columns if s in self.earnann.columns]

        cl = self.cl.loc[common_dates, common_stocks]
        op = self.op.loc[common_dates, common_stocks]
        earnann = self.earnann.loc[common_dates, common_stocks]

        print(f"  공통 데이터: {len(common_dates)} 거래일 x {len(common_stocks)} 주식")

        lookback = 90
        maxPositions = 30

        # Close-to-Open 수익률
        retC2O = (op - cl.shift(1)) / cl.shift(1)

        # 90일 롤링 표준편차
        stdC2O = retC2O.rolling(lookback).std()

        # 어닝 발표일에 갭이 큰 주식
        longs = (retC2O >= 0.5 * stdC2O) & (earnann == 1)
        shorts = (retC2O <= -0.5 * stdC2O) & (earnann == 1)

        # Open-to-Close 수익률
        retO2C = (cl - op) / op

        # 포지션 및 PnL
        positions = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
        positions[longs] = 1
        positions[shorts] = -1

        pnl = (positions * retO2C).sum(axis=1) / maxPositions
        pnl = pnl.iloc[lookback:]  # 롤링 윈도우 이후

        cumret = (1 + pnl).cumprod() - 1

        apr = (1 + pnl).prod() ** (252 / len(pnl)) - 1
        sharpe = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else 0
        maxDD, maxDDD, _ = calculateMaxDD(cumret)

        # 일별 포지션 수
        daily_positions = (positions.abs().sum(axis=1)).iloc[lookback:]
        avg_positions = daily_positions[daily_positions > 0].mean() if (daily_positions > 0).any() else 0

        self.results['pead'] = {
            'apr': apr,
            'sharpe': sharpe,
            'maxDD': maxDD,
            'maxDDD': maxDDD,
            'avg_positions': avg_positions,
        }

        print(f"  APR = {apr:.4f} ({apr*100:.2f}%)")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD:.4f} ({maxDD*100:.2f}%)")
        print(f"  Max DD Duration = {maxDDD:.0f} 일")
        print(f"  평균 동시 포지션 수 = {avg_positions:.1f}")

        # 차트 생성
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(cumret.index, cumret.values * 100, 'g-', linewidth=1)
        ax.set_title('Post-Earnings Announcement Drift (PEAD) - Cumulative Returns', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch4_pead.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch4_pead.png', 'PEAD 전략 누적 수익률'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 60)
        print("📝 리포트 생성 중...")
        print("=" * 60)

        report = []
        report.append("# Chapter 4: 주식과 ETF의 평균 회귀 (Mean Reversion of Stocks and ETFs)")
        report.append(f"\n> 분석 실행일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 개요
        report.append("## 1. 개요 및 문제 정의\n")
        report.append("Chapter 4는 주식과 ETF 시장에서의 평균 회귀 전략을 다룬다. 핵심 질문:")
        report.append("- 개별 주식의 페어 트레이딩은 왜 어려운가?")
        report.append("- ETF와 구성 주식 간 차익거래는 어떻게 구현하는가?")
        report.append("- 횡단면(cross-sectional) 평균 회귀 전략의 원리와 성과는?\n")
        report.append("### 핵심 수학적 개념\n")
        report.append("**횡단면 선형 롱-숏 가중치 (식 4.1):**\n")
        report.append("$$w_i = -\\frac{r_i - \\langle r_j \\rangle}{\\sum_k |r_k - \\langle r_j \\rangle|}$$\n")
        report.append("여기서 $r_i$는 $i$번째 주식의 일일 수익률, $\\langle r_j \\rangle$는 유니버스 내 모든 주식의 평균 일일 수익률이다.")
        report.append("분모의 정규화 계수로 인해 매일 동일한 총 자본($1)을 투자한다.\n")
        report.append("**시계열 vs 횡단면 평균 회귀의 차이:**")
        report.append("- **시계열 평균 회귀**: 가격이 자기 과거 평균으로 회귀")
        report.append("- **횡단면 평균 회귀**: 상대 수익률의 직렬 역상관에 의존. 유니버스 대비 상대적 성과가 반전\n")

        # 2. 사용 데이터
        report.append("## 2. 사용 데이터\n")
        report.append("| 파일명 | 내용 | 컬럼 수 | 기간 | 용도 |")
        report.append("|--------|------|---------|------|------|")
        report.append("| `inputDataOHLCDaily_20120424_cl.csv` | S&P 500 일일 종가 | 498 | 2006-05~2012-04 | 모든 전략 |")
        report.append("| `inputDataOHLCDaily_20120424_op.csv` | S&P 500 일일 시가 | 498 | 2006-05~2012-04 | BoG, 일중 전략 |")
        report.append("| `inputDataOHLCDaily_20120424_hi.csv` | S&P 500 일일 고가 | 498 | 2006-05~2012-04 | BoG |")
        report.append("| `inputDataOHLCDaily_20120424_lo.csv` | S&P 500 일일 저가 | 498 | 2006-05~2012-04 | BoG |")
        report.append("| `inputDataOHLCDaily_20120424_stocks.csv` | 주식 티커명 | 497 | - | 컬럼명 매핑 |")
        report.append("| `inputData_ETF_cl.csv` | 67개 ETF 일일 종가 | 68 | 2006-04~2012-04 | 인덱스 차익거래 |")
        report.append("| `inputData_ETF_stocks.csv` | ETF 티커명 | 67 | - | 컬럼명 매핑 |")
        report.append("| `earnannFile.csv` | 어닝 발표 플래그 | 498 | 2011~ | PEAD 전략 |\n")
        report.append("**데이터 특성**: S&P 500 구성 주식 약 497개의 일일 OHLC 데이터. 생존자 편향(survivorship bias)이 있음에 유의.\n")

        # 3. 분석 1: Buy-on-Gap
        report.append("## 3. 분석 1: Buy-on-Gap 모델 (예제 4.1)\n")
        report.append("### 전략 원리\n")
        report.append("주가 지수 선물이 개장 전 하락하는 날, 특정 주식이 패닉 셀링으로 과도하게 하락한다.")
        report.append("이 패닉이 끝나면 주식은 하루 동안 점차 상승하는 일중 평균 회귀 현상을 이용한다.\n")
        report.append("**전략 규칙:**")
        report.append("1. 전일 저가에서 1 표준편차(90일) 이상 갭다운한 주식을 선택")
        report.append("2. 시가가 20일 이동평균보다 높은 주식만 필터링 (모멘텀 필터)")
        report.append("3. 갭이 가장 큰 10개 주식 매수")
        report.append("4. 장 마감 시 청산\n")
        report.append("**모멘텀 필터의 중요성**: 시가 > 20일 MA 조건은 장기 하락 추세에 있는 주식(부정적 뉴스)을")
        report.append("걸러내고, 일시적 유동성 수요로 인한 하락만 포착한다.\n")

        if 'bog' in self.results:
            r = self.results['bog']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 8.7% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 1.5 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | - |")
            report.append(f"| Max DD Duration | {r['maxDDD']:.0f}일 | - |")
            report.append(f"| 활성 거래일 | {r['active_trading_days']}/{r['total_days']} | - |")
            report.append(f"| 평균 종목수/일 | {r['avg_trades_per_day']:.1f} | - |\n")
            report.append("![Buy-on-Gap 모델](figures/ch4_buy_on_gap.png)\n")

        # 4. 분석 2: 인덱스 차익거래
        report.append("## 4. 분석 2: SPY 인덱스 차익거래 (예제 4.2)\n")
        report.append("### 방법론\n")
        report.append("1. **훈련 단계** (2007): 각 SPX 주식과 SPY에 대해 요한센 공적분 검정 수행")
        report.append("2. 공적분하는 주식으로 동일 자본 배분 롱온리 포트폴리오 구성")
        report.append("3. 포트폴리오와 SPY의 공적분 재확인")
        report.append("4. **테스트 단계** (2008~): 5일 룩백의 선형 평균 회귀 전략 적용\n")
        report.append("**공적분 검정 수식** (요한센 검정):")
        report.append("$$\\Delta Y_t = \\Pi Y_{t-1} + \\epsilon_t$$")
        report.append("여기서 $\\Pi = \\alpha \\beta'$, $\\beta$는 공적분 벡터(고유벡터), $\\alpha$는 조정 속도\n")

        if 'indexArb' in self.results:
            r = self.results['indexArb']
            report.append("### 결과\n")
            report.append(f"- SPY와 공적분하는 주식: **{r.get('coint_count', 'N/A')}**개")
            if 'evec' in r:
                report.append(f"- 고유벡터: [{r['evec'][0]:.4f}, {r['evec'][1]:.4f}]\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 4.5% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 1.3 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | - |")
            report.append(f"| Max DD Duration | {r['maxDDD']:.0f}일 | - |\n")
            report.append("![SPY 인덱스 차익거래](figures/ch4_index_arbitrage.png)\n")

        # 5. 분석 3: 횡단면 평균 회귀
        report.append("## 5. 분석 3: 횡단면 선형 롱-숏 모델 (예제 4.3, 4.4)\n")
        report.append("### 전략 원리\n")
        report.append("Khandani & Lo (2007)가 제안한 전략. 매일 각 주식의 수익률에서 시장 평균을 빼고,")
        report.append("이 상대 수익률을 역방향으로 투자한다.\n")
        report.append("$$w_i = -\\frac{r_i - \\langle r_j \\rangle}{\\sum_k |r_k - \\langle r_j \\rangle|}$$\n")
        report.append("**특징**: 완전히 선형, 매개변수 없음, 달러 중립. 2008년 리먼 위기에서도 양의 수익.\n")
        report.append("**두 가지 변형:**")
        report.append("- **Close-to-Close (예제 4.3)**: 전일 종가→당일 종가 수익률로 가중치 결정")
        report.append("- **Intraday (예제 4.4)**: 전일 종가→당일 시가 수익률로 가중치, 시가→종가로 수익 실현\n")

        if 'crossSectional' in self.results:
            cs = self.results['crossSectional']
            c = cs['c2c']
            i = cs['intraday']

            report.append("### 전체 기간 성과 비교\n")
            report.append("| 지표 | Close-to-Close | Intraday | 책 기대값 (C2C) | 책 기대값 (Intra) |")
            report.append("|------|---------------|----------|---------------|-----------------|")
            report.append(f"| APR | {c['apr']*100:.2f}% | {i['apr']*100:.2f}% | 13.7% | 73% |")
            report.append(f"| Sharpe | {c['sharpe']:.4f} | {i['sharpe']:.4f} | 1.3 | 4.7 |")
            report.append(f"| Max DD | {c['maxDD']*100:.2f}% | {i['maxDD']*100:.2f}% | - | - |")
            report.append(f"| Max DDD | {c['maxDDD']:.0f}일 | {i['maxDDD']:.0f}일 | - | - |\n")

            if 'yearly' in cs:
                report.append("### 연도별 성과\n")
                report.append("| 연도 | C2C APR | C2C Sharpe | Intraday APR | Intraday Sharpe |")
                report.append("|------|---------|-----------|-------------|----------------|")
                for year, yr in sorted(cs['yearly'].items()):
                    report.append(f"| {year} | {yr['c2c_apr']*100:.2f}% | {yr['c2c_sharpe']:.2f} | {yr['intra_apr']*100:.2f}% | {yr['intra_sharpe']:.2f} |")
                report.append("")

            report.append("![횡단면 평균 회귀](figures/ch4_cross_sectional.png)\n")

        # 6. 분석 4: PEAD
        if self.results.get('pead') is not None:
            report.append("## 6. 분석 4: 실적 발표 후 표류 (PEAD)\n")
            report.append("### 전략 원리\n")
            report.append("어닝 발표일에 종가→시가 갭이 90일 표준편차의 0.5배를 초과하면 롱,")
            report.append("-0.5배 미만이면 숏. 당일 종가에 청산하는 일중 전략.\n")

            r = self.results['pead']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 6.8% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 1.49 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | -2.6% |")
            report.append(f"| Max DD Duration | {r['maxDDD']:.0f}일 | 109 |")
            report.append(f"| 평균 동시 포지션 | {r['avg_positions']:.1f} | - |\n")
            report.append("![PEAD 전략](figures/ch4_pead.png)\n")

        # 7. 전략 종합 비교
        report.append("## 7. 전략 종합 비교\n")
        report.append("| 전략 | APR | Sharpe | Max DD | 특성 |")
        report.append("|------|-----|--------|--------|------|")

        if 'bog' in self.results:
            r = self.results['bog']
            report.append(f"| Buy-on-Gap | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | {r['maxDD']*100:.2f}% | 일중, 롱온리 |")

        if 'indexArb' in self.results:
            r = self.results['indexArb']
            report.append(f"| SPY Index Arb | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | {r['maxDD']*100:.2f}% | 일간, 롱-숏 |")

        if 'crossSectional' in self.results:
            c = self.results['crossSectional']['c2c']
            i = self.results['crossSectional']['intraday']
            report.append(f"| Linear L/S (C2C) | {c['apr']*100:.2f}% | {c['sharpe']:.2f} | {c['maxDD']*100:.2f}% | 일간, 달러 중립 |")
            report.append(f"| Linear L/S (Intra) | {i['apr']*100:.2f}% | {i['sharpe']:.2f} | {i['maxDD']*100:.2f}% | 일중, 달러 중립 |")

        if self.results.get('pead') is not None:
            r = self.results['pead']
            report.append(f"| PEAD | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | {r['maxDD']*100:.2f}% | 일중, 이벤트 기반 |")
        report.append("")

        # 8. 결론
        report.append("## 8. 결론 및 권고사항\n")
        report.append("### 핵심 발견\n")
        report.append("1. **개별 주식 페어 트레이딩의 한계**: 기업 펀더멘털 변화로 공적분 관계가 무너질 위험이 높다")
        report.append("2. **ETF 기반 전략의 안정성**: ETF는 바스켓 경제 변화가 느려 공적분 관계가 더 안정적")
        report.append("3. **횡단면 전략의 강건성**: Khandani-Lo 모델은 매개변수 없이도 안정적 수익 달성")
        report.append("4. **일중 전략의 높은 성과**: 시가-종가 전략이 종가-종가보다 월등히 높은 수익률\n")
        report.append("### 트레이딩 권고\n")
        report.append("- 평균 회귀 전략에 **모멘텀 필터**를 중첩하면 일관성 향상")
        report.append("- 횡단면 전략에서 **소형주 유니버스**를 사용하면 더 높은 수익률 기대 가능")
        report.append("- 인덱스 차익거래는 **주기적 재훈련**이 필수적\n")
        report.append("### 주의사항\n")
        report.append("- **생존자 편향**: 사용된 S&P 500 데이터에 생존자 편향 존재")
        report.append("- **거래비용**: 모든 백테스트에 거래비용 미포함. 특히 일중 전략은 거래비용 2배")
        report.append("- **시그널 노이즈**: 시가 기반 진입 시 사전개장 가격과 실제 시가의 차이")
        report.append("- **공매도 제약**: 숏 포지션의 Alternative Uptick Rule, NBBO 규모 제한")
        report.append("- **슬리피지**: 통합 가격 vs 기본 거래소 가격 차이로 인한 백테스트 과대평가 가능성\n")

        # 리포트 저장
        report_path = REPORT_DIR / "chapter4_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"  ✓ 리포트 저장: {report_path}")
        print()

    def run(self):
        """전체 분석 오케스트레이션"""
        print("\n" + "🔬" * 30)
        print("  Chapter 4: 주식과 ETF의 평균 회귀 - 종합 분석")
        print("🔬" * 30 + "\n")

        self.load_data()
        self.analyze_buy_on_gap()
        self.analyze_index_arbitrage()
        self.analyze_cross_sectional_mean_reversion()
        self.analyze_pead()
        self.generate_report()

        print("=" * 60)
        print("✅ Chapter 4 분석 완료!")
        print(f"   리포트: reports/chapter4_report.md")
        print(f"   차트: {len(self.figures)}개 생성")
        for fig_name, fig_desc in self.figures:
            print(f"     - {fig_name}: {fig_desc}")
        print("=" * 60)


if __name__ == "__main__":
    analyzer = Chapter4Analyzer()
    analyzer.run()
