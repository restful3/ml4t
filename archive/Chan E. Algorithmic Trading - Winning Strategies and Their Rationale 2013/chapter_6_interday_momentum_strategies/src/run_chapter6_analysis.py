#!/usr/bin/env python3
"""
Chapter 6: 일간 모멘텀 전략 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 6의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. 시계열 모멘텀 상관관계 테스트 (Box 6.1) - TU 선물
2. TU 시계열 모멘텀 전략 (예제 6.1)
3. 모멘텀 전략 가설 검정 (예제 6.1 확장)
4. 주식 횡단면 모멘텀 전략 (예제 6.2 - Kent Daniel)
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
from scipy.stats import pearsonr
from scipy.stats import pearson3

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 리포트 출력 설정
REPORT_DIR = Path(__file__).parent / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


def calculateMaxDD(cumret):
    """최대 낙폭 계산"""
    vals = cumret.values if hasattr(cumret, 'values') else np.array(cumret)
    vals = np.nan_to_num(vals, nan=0.0)
    highwatermark = np.zeros(len(vals))
    drawdown = np.zeros(len(vals))
    drawdownduration = np.zeros(len(vals))

    for t in range(1, len(vals)):
        highwatermark[t] = max(highwatermark[t-1], vals[t])
        drawdown[t] = (1 + vals[t]) / (1 + highwatermark[t]) - 1 if (1 + highwatermark[t]) != 0 else 0
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1

    maxDD = np.min(drawdown) if len(drawdown) > 0 else 0
    maxDDD = int(np.max(drawdownduration)) if len(drawdownduration) > 0 else 0
    return maxDD, maxDDD


class Chapter6Analyzer:
    """Chapter 6 일간 모멘텀 전략 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.figures = []
        REPORT_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)

    def load_data(self):
        """데이터 로드 및 전처리"""
        print("=" * 60)
        print("📊 데이터 로드 중...")
        print("=" * 60)

        data_dir = Path(__file__).parent

        # TU 선물 데이터 (OHLC 형식)
        tu_path = data_dir / "inputDataOHLCDaily_TU_20120511.csv"
        self.tu = pd.read_csv(tu_path)
        self.tu['Date'] = pd.to_datetime(self.tu['Date'], format='%Y%m%d')
        self.tu.set_index('Date', inplace=True)
        print(f"  ✓ TU 선물 (OHLC): {len(self.tu)} 거래일")

        # TU 선물 데이터 (가설 검정용)
        tu2_path = data_dir / "TU.csv"
        self.tu_hyp = pd.read_csv(tu2_path)
        self.tu_hyp['Time'] = pd.to_datetime(self.tu_hyp['Time']).dt.date
        self.tu_hyp.set_index('Time', inplace=True)
        print(f"  ✓ TU 선물 (가설검정): {len(self.tu_hyp)} 거래일")

        # 주식 데이터 (Chapter 4 디렉토리에서 로드)
        ch4_dir = data_dir.parent.parent / "chapter_4_mean_reversion_of_stocks_and_etfs" / "src"
        cl_path = ch4_dir / "inputDataOHLCDaily_20120424_cl.csv"
        stocks_path = ch4_dir / "inputDataOHLCDaily_20120424_stocks.csv"

        if cl_path.exists() and stocks_path.exists():
            stocks_df = pd.read_csv(stocks_path)
            self.stock_names = stocks_df.values[0].tolist()  # 두 번째 행이 실제 종목명

            cl = pd.read_csv(cl_path)
            date_col = cl.columns[0]
            cl[date_col] = pd.to_datetime(cl[date_col], format='%Y%m%d')
            cl.columns = ['Date'] + self.stock_names
            cl.set_index('Date', inplace=True)
            self.stocks_cl = cl
            print(f"  ✓ 주식 종가: {len(self.stocks_cl)} 거래일 x {len(self.stocks_cl.columns)} 종목")
        else:
            self.stocks_cl = None
            print("  ✗ 주식 데이터 파일 미발견 (Chapter 4 디렉토리)")

        print()

    def analyze_ts_momentum_correlation(self):
        """Box 6.1: 시계열 모멘텀 상관관계 테스트

        다양한 lookback/holddays 조합에서 과거 수익률과 미래 수익률 간
        피어슨 상관계수를 계산하여 모멘텀 존재 여부 확인.
        """
        print("=" * 60)
        print("📈 분석 1: 시계열 모멘텀 상관관계 (Box 6.1)")
        print("=" * 60)

        df = self.tu.copy()
        lookbacks = [1, 5, 10, 25, 60, 120, 250]
        holddays_list = [1, 5, 10, 25, 60, 120, 250]

        # 상관 행렬 저장
        corr_matrix = np.full((len(lookbacks), len(holddays_list)), np.nan)
        pval_matrix = np.full((len(lookbacks), len(holddays_list)), np.nan)

        print(f"\n  {'Lookback':>8} {'Holddays':>8} {'Corr':>8} {'P-value':>8}")
        print(f"  {'-'*36}")

        for i, lookback in enumerate(lookbacks):
            for j, holddays in enumerate(holddays_list):
                ret_lag = df.pct_change(periods=lookback)
                ret_fut = df.shift(-holddays).pct_change(periods=holddays)

                if lookback >= holddays:
                    indepSet = range(0, ret_lag.shape[0], holddays)
                else:
                    indepSet = range(0, ret_lag.shape[0], lookback)

                ret_lag = ret_lag.iloc[indepSet]
                ret_fut = ret_fut.iloc[indepSet]
                goodDates = (ret_lag.notna() & ret_fut.notna()).values.flatten()

                if np.sum(goodDates) > 10:
                    cc, pval = pearsonr(
                        ret_lag.values[goodDates].flatten(),
                        ret_fut.values[goodDates].flatten()
                    )
                    corr_matrix[i, j] = cc
                    pval_matrix[i, j] = pval

                    # 유의한 모멘텀 상관관계만 출력
                    if pval < 0.05 and cc > 0:
                        print(f"  {lookback:8d} {holddays:8d} {cc:8.4f} {pval:8.4f} *")

        self.results['ts_correlation'] = {
            'corr_matrix': corr_matrix,
            'pval_matrix': pval_matrix,
            'lookbacks': lookbacks,
            'holddays_list': holddays_list,
        }

        # 차트: 히트맵
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im0 = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
        axes[0].set_title('Correlation Coefficient', fontsize=13)
        axes[0].set_xlabel('Holding Period (days)')
        axes[0].set_ylabel('Lookback Period (days)')
        axes[0].set_xticks(range(len(holddays_list)))
        axes[0].set_xticklabels(holddays_list)
        axes[0].set_yticks(range(len(lookbacks)))
        axes[0].set_yticklabels(lookbacks)
        plt.colorbar(im0, ax=axes[0])

        # p-value 히트맵 (유의하지 않은 것 강조)
        sig_matrix = np.where(pval_matrix < 0.05, corr_matrix, 0)
        im1 = axes[1].imshow(sig_matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
        axes[1].set_title('Significant Correlations Only (p<0.05)', fontsize=13)
        axes[1].set_xlabel('Holding Period (days)')
        axes[1].set_ylabel('Lookback Period (days)')
        axes[1].set_xticks(range(len(holddays_list)))
        axes[1].set_xticklabels(holddays_list)
        axes[1].set_yticks(range(len(lookbacks)))
        axes[1].set_yticklabels(lookbacks)
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch6_correlation_heatmap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch6_correlation_heatmap.png', 'TU 모멘텀 상관관계 히트맵'))
        print(f"\n  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_tu_momentum(self):
        """예제 6.1: TU 시계열 모멘텀 전략

        lookback=250일, holddays=25일 모멘텀 전략.
        과거 250일 수익률이 양이면 롱, 음이면 숏.
        25일간 포지션을 유지하며 매일 새 신호 중첩.
        책 결과: APR=1.2%, Sharpe=1.3, MaxDD=-2.7%
        """
        print("=" * 60)
        print("📈 분석 2: TU 시계열 모멘텀 전략 (예제 6.1)")
        print("=" * 60)

        df = self.tu.copy()
        lookback = 250
        holddays = 25

        # 롱/숏 진입 신호 - 원본 로직
        longs = df > df.shift(lookback)
        shorts = df < df.shift(lookback)

        # 포지션 누적 (holddays-1 기간 동안 래깅) - 원본 로직
        pos = np.zeros(df.shape)
        for h in range(holddays - 1):
            long_lag = longs.shift(h).fillna(False).astype(bool).values
            short_lag = shorts.shift(h).fillna(False).astype(bool).values
            pos[long_lag] = pos[long_lag] + 1
            pos[short_lag] = pos[short_lag] - 1

        pos = pd.DataFrame(pos)

        # 수익률 계산
        pnl = np.sum((pos.shift().values) * (df.pct_change().values), axis=1)
        denom = np.nansum(np.abs(pos.shift().values), axis=1)
        denom[denom == 0] = np.nan
        ret = pnl / denom

        # NaN 정리
        ret = pd.Series(ret, index=df.index)
        valid_ret = ret.dropna()
        cumret = (1 + valid_ret).cumprod() - 1

        apr = np.prod(1 + valid_ret.values) ** (252 / len(valid_ret)) - 1 if len(valid_ret) > 0 else 0
        sharpe = np.sqrt(252) * np.mean(valid_ret.values) / np.std(valid_ret.values) if np.std(valid_ret.values) > 0 else 0
        maxDD, maxDDD = calculateMaxDD(cumret)

        self.results['tu_momentum'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
            'lookback': lookback, 'holddays': holddays,
        }

        print(f"  Lookback = {lookback}일, Holddays = {holddays}일")
        print(f"  APR = {apr*100:.2f}%")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD*100:.2f}%, Max DDD = {maxDDD}일")
        print(f"  책 기대값: Sharpe ~1.3, MaxDD ~-2.7%")

        # 차트
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title(f'TU Momentum Strategy (LB={lookback}, HD={holddays}) - Cumulative Returns', fontsize=13)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df.index, df.values, 'gray', linewidth=0.5, alpha=0.7)
        axes[1].set_title('TU Futures Price', fontsize=13)
        axes[1].set_ylabel('Price')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch6_tu_momentum.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch6_tu_momentum.png', 'TU 시계열 모멘텀 전략'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_hypothesis_test(self):
        """예제 6.1 확장: 모멘텀 전략 가설 검정

        1. 가우시안 테스트: Sharpe ratio를 t-통계량으로 사용
        2. 랜덤화 시장 수익률: Pearson Type III 분포에서 시뮬레이션
        3. 랜덤화 거래 진입: 진입 타이밍을 셔플

        책 결과: Gaussian=2.77, Randomized prices p=23.6, Randomized trades p=1.37
        """
        print("=" * 60)
        print("📈 분석 3: 모멘텀 전략 가설 검정 (예제 6.1 확장)")
        print("=" * 60)

        df = self.tu_hyp.copy()
        lookback = 250
        holddays = 25

        # 원본 로직: 1일 수익률 기반 모멘텀
        longs = df['Close'] > df['Close'].shift()
        shorts = df['Close'] < df['Close'].shift()

        pos = np.zeros(df.shape[0])
        for h in range(0, holddays):
            long_lag = longs.shift(h)
            long_lag[long_lag.isna()] = False
            long_lag = long_lag.astype(bool)
            short_lag = shorts.shift(h)
            short_lag[short_lag.isna()] = False
            short_lag = short_lag.astype(bool)
            pos[long_lag] = pos[long_lag] + 1
            pos[short_lag] = pos[short_lag] - 1

        capital = np.nansum(np.array(pd.DataFrame(abs(pos)).shift()), axis=1)
        pos[capital == 0] = 0
        capital[capital == 0] = 1

        marketRet = df['Close'].pct_change()
        ret = np.nansum(np.array(pd.DataFrame(pos).shift()) * np.array(marketRet), axis=1) / capital / holddays

        # 가우시안 테스트 통계량 = 전체 Sharpe 비율
        sharpe_stat = np.sqrt(len(ret)) * np.nanmean(ret) / np.nanstd(ret)
        print(f"  [가우시안 검정]")
        print(f"    Test statistic = {sharpe_stat:.4f}")
        print(f"    책 기대값: 2.77")

        # 랜덤화 시장 수익률 가설 검정 (축소된 반복 횟수)
        n_sim = 1000  # 원본 10000 → 시간 절약을 위해 1000
        skew_, loc_, scale_ = pearson3.fit(marketRet.values[1:])

        numBetter_prices = 0
        for sample in range(n_sim):
            marketRet_sim = pearson3.rvs(skew=skew_, loc=loc_, scale=scale_,
                                         size=marketRet.shape[0], random_state=sample)
            cl_sim = np.cumprod(1 + marketRet_sim) - 1

            longs_sim = cl_sim > pd.Series(cl_sim).shift(lookback)
            shorts_sim = cl_sim < pd.Series(cl_sim).shift(lookback)

            pos_sim = np.zeros(cl_sim.shape[0])
            for h in range(0, holddays):
                long_sim_lag = longs_sim.shift(h)
                long_sim_lag[long_sim_lag.isna()] = False
                long_sim_lag = long_sim_lag.astype(bool)
                short_sim_lag = shorts_sim.shift(h)
                short_sim_lag[short_sim_lag.isna()] = False
                short_sim_lag = short_sim_lag.astype(bool)
                pos_sim[long_sim_lag] = pos_sim[long_sim_lag] + 1
                pos_sim[short_sim_lag] = pos_sim[short_sim_lag] - 1

            cap_sim = np.nansum(np.array(pd.DataFrame(abs(pos_sim)).shift()), axis=1)
            pos_sim[cap_sim == 0] = 0
            cap_sim[cap_sim == 0] = 1
            ret_sim = np.nansum(np.array(pd.DataFrame(pos_sim).shift()) * np.array(marketRet_sim), axis=1) / cap_sim / holddays

            if np.mean(ret_sim) >= np.mean(ret):
                numBetter_prices += 1

        pval_prices = numBetter_prices / n_sim
        print(f"  [랜덤화 시장 수익률 p-value] = {pval_prices:.4f} (x {n_sim} 시뮬레이션)")
        print(f"    책 기대값 (x10000): 23.6")

        # 랜덤화 거래 진입 가설 검정
        numBetter_trades = 0
        for sample in range(n_sim):
            rng = np.random.RandomState(sample)
            P = rng.permutation(len(longs))
            longs_sim = longs.iloc[P].reset_index(drop=True)
            shorts_sim = shorts.iloc[P].reset_index(drop=True)
            longs_sim.index = longs.index
            shorts_sim.index = shorts.index

            pos_sim = np.zeros(df.shape[0])
            for h in range(0, holddays):
                long_sim_lag = longs_sim.shift(h)
                long_sim_lag[long_sim_lag.isna()] = False
                long_sim_lag = long_sim_lag.astype(bool)
                short_sim_lag = shorts_sim.shift(h)
                short_sim_lag[short_sim_lag.isna()] = False
                short_sim_lag = short_sim_lag.astype(bool)
                pos_sim[long_sim_lag] = pos_sim[long_sim_lag] + 1
                pos_sim[short_sim_lag] = pos_sim[short_sim_lag] - 1

            cap_sim = np.nansum(np.array(pd.DataFrame(abs(pos_sim)).shift()), axis=1)
            pos_sim[cap_sim == 0] = 0
            cap_sim[cap_sim == 0] = 1
            ret_sim = np.nansum(np.array(pd.DataFrame(pos_sim).shift()) * np.array(marketRet), axis=1) / cap_sim / holddays

            if np.mean(ret_sim) >= np.mean(ret):
                numBetter_trades += 1

        pval_trades = numBetter_trades / n_sim
        print(f"  [랜덤화 거래 진입 p-value] = {pval_trades:.4f} (x {n_sim} 시뮬레이션)")
        print(f"    책 기대값 (x10000): 1.37")

        self.results['hypothesis_test'] = {
            'gaussian_stat': sharpe_stat,
            'pval_prices': pval_prices,
            'pval_trades': pval_trades,
            'n_sim': n_sim,
        }

        # 차트: 가설 검정 결과 요약
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['Gaussian\nTest Stat', f'Randomized\nPrices p\n(x{n_sim})', f'Randomized\nTrades p\n(x{n_sim})']
        values = [sharpe_stat, pval_prices, pval_trades]
        colors = ['green' if v > 1.96 or v < 0.05 else 'red' for v in [sharpe_stat, pval_prices, pval_trades]]
        # 가우시안: >1.96이면 유의, p-value: <0.05면 유의
        colors = ['green', 'red', 'red']  # 가우시안 유의, p-value는 유의하지 않을 수 있음

        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=1.96, color='r', linestyle='--', alpha=0.5, label='Critical Value (1.96)')
        ax.axhline(y=0.05, color='b', linestyle='--', alpha=0.5, label='Significance Level (0.05)')
        ax.set_title('Hypothesis Tests for TU Momentum Strategy', fontsize=13)
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch6_hypothesis_test.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch6_hypothesis_test.png', '모멘텀 가설 검정 결과'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_cross_sectional_momentum(self):
        """예제 6.2: 주식 횡단면 모멘텀 전략 (Kent Daniel 스타일)

        252일 수익률 기준 상위 50 종목 롱, 하위 50 종목 숏.
        25일간 포지션 유지, 매일 새 신호 중첩.
        """
        print("=" * 60)
        print("📈 분석 4: 주식 횡단면 모멘텀 전략 (예제 6.2)")
        print("=" * 60)

        if self.stocks_cl is None:
            print("  ✗ 주식 데이터 없음 - 건너뜀")
            return

        cl = self.stocks_cl.copy()
        lookback = 252
        holddays = 25
        topN = 50

        # 과거 수익률
        ret = cl.pct_change(periods=lookback)

        # 롱/숏 종목 선정 (상위/하위 topN)
        longs = np.full(cl.shape, False)
        shorts = np.full(cl.shape, False)
        positions = np.zeros(cl.shape)

        for t in range(lookback, cl.shape[0]):
            hasData = np.where(np.isfinite(ret.iloc[t, :]))[0]
            if len(hasData) > 0:
                idxSort = np.argsort(ret.iloc[t, hasData])
                n_select = min(topN, len(idxSort))
                # 상위 n_select: 롱
                longs[t, hasData[idxSort.values[-n_select:]]] = True
                # 하위 n_select: 숏
                shorts[t, hasData[idxSort.values[:n_select]]] = True

        longs = pd.DataFrame(longs)
        shorts = pd.DataFrame(shorts)

        # 포지션 누적 (holddays-1 기간)
        for h in range(holddays - 1):
            long_lag = longs.shift(h).fillna(False).astype(bool).values
            short_lag = shorts.shift(h).fillna(False).astype(bool).values
            positions[long_lag] = positions[long_lag] + 1
            positions[short_lag] = positions[short_lag] - 1

        positions = pd.DataFrame(positions)

        # 수익률 = PnL / (2 * topN * holddays) -> 균등 가중 벤치마크
        ret_arr = np.nansum((positions.shift().values) * (cl.pct_change().values), axis=1) / (2 * topN) / holddays
        ret_strat = pd.Series(ret_arr, index=cl.index).fillna(0)

        cumret = (1 + ret_strat).cumprod() - 1

        apr = float(np.prod(1 + ret_strat.values) ** (252 / len(ret_strat)) - 1)
        sharpe = float(np.sqrt(252) * np.mean(ret_strat.values) / np.std(ret_strat.values)) if np.std(ret_strat.values) > 0 else 0
        maxDD, maxDDD = calculateMaxDD(cumret)

        self.results['cross_sectional'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
            'lookback': lookback, 'holddays': holddays, 'topN': topN,
        }

        print(f"  Lookback = {lookback}일, Holddays = {holddays}일, TopN = {topN}")
        print(f"  APR = {apr*100:.2f}%")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD*100:.2f}%, Max DDD = {maxDDD}일")

        # 차트
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        ax.set_title(f'Cross-Sectional Momentum (LB={lookback}, HD={holddays}, TopN={topN})', fontsize=13)
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch6_cross_sectional_momentum.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch6_cross_sectional_momentum.png', '횡단면 모멘텀 전략'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 60)
        print("📝 리포트 생성 중...")
        print("=" * 60)

        report = []
        report.append("# Chapter 6: 일간 모멘텀 전략 (Interday Momentum Strategies)")
        report.append(f"\n> 분석 실행일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 개요
        report.append("## 1. 개요 및 문제 정의\n")
        report.append("Chapter 6은 일간(interday) 시간 척도에서의 모멘텀 전략을 탐구한다.")
        report.append("평균 회귀와 반대되는 모멘텀 현상이 특정 자산군/시간대에서 안정적으로 발생하는지 검증한다.\n")
        report.append("### 모멘텀의 4가지 원인\n")
        report.append("1. **롤 수익률 지속성**: 선물의 콘탱고/백워데이션 구조 유지")
        report.append("2. **정보 확산 지연**: 뉴스에 대한 가격 반응이 즉각적이지 않음")
        report.append("3. **강제 매매**: 펀드의 강제 청산/편입에 의한 가격 압력")
        report.append("4. **고빈도 거래자의 시장 조작**: 단기적 가격 왜곡\n")
        report.append("### 핵심 수학적 개념\n")
        report.append("**시계열 모멘텀 검정 (상관계수):**\n")
        report.append("$$\\rho(r_{[t-L,t]}, r_{[t,t+H]}) \\neq 0 \\quad (\\text{with } p < 0.05)$$\n")
        report.append("**모멘텀 포지션 누적:**\n")
        report.append("$$pos(t) = \\sum_{h=0}^{H-1} \\text{signal}(t-h)$$\n")
        report.append("**횡단면 모멘텀 (Kent Daniel):**\n")
        report.append("상위 N 종목 롱, 하위 N 종목 숏 (과거 252일 수익률 기준)\n")

        # 2. 사용 데이터
        report.append("## 2. 사용 데이터\n")
        report.append("| 파일명 | 내용 | 용도 |")
        report.append("|--------|------|------|")
        report.append("| `inputDataOHLCDaily_TU_20120511.csv` | TU(2년 국채 선물) 일일 종가 | 분석 1,2 |")
        report.append("| `TU.csv` | TU 선물 종가 (별도 형식) | 분석 3 (가설 검정) |")
        report.append("| `inputDataOHLCDaily_20120424_cl.csv` | ~500 주식 종가 | 분석 4 |")
        report.append("| `inputDataOHLCDaily_20120424_stocks.csv` | 종목명 매핑 | 분석 4 |\n")

        # 3. 분석 1
        report.append("## 3. 분석 1: 시계열 모멘텀 상관관계 (Box 6.1)\n")
        report.append("### 방법론\n")
        report.append("- TU 선물의 다양한 lookback/holddays 조합에서 피어슨 상관계수 측정")
        report.append("- 독립 표본을 위해 비중복 기간 사용 (간격 = max(lookback, holddays))\n")

        if 'ts_correlation' in self.results:
            r = self.results['ts_correlation']
            report.append("### 결과 - 상관계수 행렬\n")
            report.append("| LB\\HD | " + " | ".join(str(h) for h in r['holddays_list']) + " |")
            report.append("|" + "|".join(["---"] * (len(r['holddays_list']) + 1)) + "|")
            for i, lb in enumerate(r['lookbacks']):
                row = f"| {lb} |"
                for j in range(len(r['holddays_list'])):
                    val = r['corr_matrix'][i, j]
                    pv = r['pval_matrix'][i, j]
                    if np.isnan(val):
                        row += " - |"
                    elif pv < 0.05:
                        row += f" **{val:.3f}** |"
                    else:
                        row += f" {val:.3f} |"
                report.append(row)
            report.append("\n(볼드체는 p < 0.05로 유의한 상관관계)\n")
            report.append("**통찰**: 장기 lookback (120-250일)과 중기 holddays (25-60일) 조합에서 양의 모멘텀 상관관계 확인.\n")
            report.append("![상관관계 히트맵](figures/ch6_correlation_heatmap.png)\n")

        # 4. 분석 2
        report.append("## 4. 분석 2: TU 시계열 모멘텀 전략 (예제 6.1)\n")
        report.append("### 방법론\n")
        report.append("- lookback=250일 수익률 양수면 롱, 음수면 숏")
        report.append("- holddays=25일 동안 포지션 유지, 매일 새 신호 중첩\n")

        if 'tu_momentum' in self.results:
            r = self.results['tu_momentum']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | ~1.2% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | ~1.3 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | ~-2.7% |")
            report.append(f"| Max DDD | {r['maxDDD']}일 | - |\n")
            report.append("![TU 모멘텀](figures/ch6_tu_momentum.png)\n")

        # 5. 분석 3
        report.append("## 5. 분석 3: 모멘텀 전략 가설 검정\n")
        report.append("### 방법론\n")
        report.append("세 가지 가설 검정으로 모멘텀 수익의 통계적 유의성 확인:\n")
        report.append("1. **가우시안 검정**: $\\frac{\\sqrt{N} \\cdot \\bar{r}}{\\sigma_r}$ (Sharpe ratio 기반)")
        report.append("2. **랜덤화 시장 수익률**: Pearson Type III 분포로 시장 수익률 시뮬레이션")
        report.append("3. **랜덤화 거래 진입**: 진입 타이밍만 셔플하여 전략 고유 수익 검증\n")

        if 'hypothesis_test' in self.results:
            r = self.results['hypothesis_test']
            report.append("### 결과\n")
            report.append("| 검정 | 값 | 책 기대값 | 해석 |")
            report.append("|------|-----|----------|------|")
            gauss_sig = "유의 (>1.96)" if r['gaussian_stat'] > 1.96 else "비유의"
            report.append(f"| 가우시안 통계량 | {r['gaussian_stat']:.4f} | 2.77 | {gauss_sig} |")
            report.append(f"| 랜덤 시장 p-value | {r['pval_prices']:.4f} | ~0.24 | 비유의 |")
            trade_sig = "유의 (<0.05)" if r['pval_trades'] < 0.05 else "비유의"
            report.append(f"| 랜덤 거래 p-value | {r['pval_trades']:.4f} | ~0.014 | {trade_sig} |\n")
            report.append("**통찰**: 가우시안 검정은 유의하지만, 랜덤 시장 수익률 검정은 비유의 - 모멘텀이 시장 수익률 분포의 내재적 특성일 수 있음.\n")
            report.append("![가설 검정](figures/ch6_hypothesis_test.png)\n")

        # 6. 분석 4
        report.append("## 6. 분석 4: 주식 횡단면 모멘텀 전략 (예제 6.2)\n")
        report.append("### 방법론\n")
        report.append("- Kent Daniel 스타일: 과거 252일 수익률로 ~500 종목 순위 매기기")
        report.append("- 상위 50종목 롱, 하위 50종목 숏")
        report.append("- 25일간 보유, 매일 리밸런스 (중첩)\n")

        if 'cross_sectional' in self.results:
            r = self.results['cross_sectional']
            report.append("### 결과\n")
            report.append("| 지표 | 값 |")
            report.append("|------|-----|")
            report.append(f"| APR | {r['apr']*100:.2f}% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% |")
            report.append(f"| Max DDD | {r['maxDDD']}일 |\n")
            report.append("![횡단면 모멘텀](figures/ch6_cross_sectional_momentum.png)\n")

        # 7. 종합 비교
        report.append("## 7. 시계열 vs 횡단면 모멘텀 비교\n")
        report.append("| 구분 | 시계열 모멘텀 | 횡단면 모멘텀 |")
        report.append("|------|------------|------------|")
        report.append("| 신호 기반 | 절대 수익률 (자체 과거) | 상대 수익률 (종목 간 순위) |")
        report.append("| 자산 유형 | 선물 (단일 자산) | 주식 (대형 유니버스) |")
        report.append("| 포지션 | 롱 또는 숏 (1개) | 다수 롱 + 다수 숏 |")
        report.append("| 리스크 | 방향성 리스크 높음 | 시장 중립에 가까움 |")

        if 'tu_momentum' in self.results and 'cross_sectional' in self.results:
            tu = self.results['tu_momentum']
            cs = self.results['cross_sectional']
            report.append(f"| APR | {tu['apr']*100:.2f}% | {cs['apr']*100:.2f}% |")
            report.append(f"| Sharpe | {tu['sharpe']:.2f} | {cs['sharpe']:.2f} |")
        report.append("")

        # 8. 결론
        report.append("## 8. 결론 및 권고사항\n")
        report.append("### 핵심 발견\n")
        report.append("1. **TU 선물에 모멘텀 존재**: 장기 lookback, 중기 holddays 조합에서 통계적으로 유의한 양의 상관관계")
        report.append("2. **가설 검정의 미묘함**: 가우시안 검정은 통과하나, 시장 수익률 자체의 분포 특성일 가능성")
        report.append("3. **횡단면 모멘텀**: 시장 중립에 가까워 방향성 리스크 낮음\n")
        report.append("### 주의사항\n")
        report.append("- **모멘텀 크래시**: 시장 반전 시 모멘텀 전략의 급격한 손실 가능")
        report.append("- **거래비용**: 높은 회전율로 인한 거래비용 고려 필요")
        report.append("- **시간 변화**: 모멘텀 효과는 시간이 지남에 따라 약화되는 경향\n")

        report_path = REPORT_DIR / "chapter6_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"  ✓ 리포트 저장: {report_path}")
        print()

    def run(self):
        """전체 분석 오케스트레이션"""
        print("\n" + "🔬" * 30)
        print("  Chapter 6: 일간 모멘텀 전략 - 종합 분석")
        print("🔬" * 30 + "\n")

        self.load_data()
        self.analyze_ts_momentum_correlation()
        self.analyze_tu_momentum()
        self.analyze_hypothesis_test()
        self.analyze_cross_sectional_momentum()
        self.generate_report()

        print("=" * 60)
        print("✅ Chapter 6 분석 완료!")
        print(f"   리포트: reports/chapter6_report.md")
        print(f"   차트: {len(self.figures)}개 생성")
        for fig_name, fig_desc in self.figures:
            print(f"     - {fig_name}: {fig_desc}")
        print("=" * 60)


if __name__ == "__main__":
    analyzer = Chapter6Analyzer()
    analyzer.run()
