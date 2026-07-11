#!/usr/bin/env python3
"""
Chapter 3: 백테스팅 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Quantitative Trading" (2nd Ed., 2021) Chapter 3의
핵심 백테스팅 개념들을 실행하고 분석 결과를 종합 리포트로 생성합니다.

분석 내용:
1. 예제 3.4 - 샤프 비율 계산 (롱온리 vs 시장중립)
2. 예제 3.5 - 최대 낙폭(MDD) 및 최대 낙폭 지속기간 계산
3. 예제 3.6 - GLD-GDX 페어 트레이딩 (z-score 진입/청산)
4. 예제 3.7 - 횡단면 평균회귀 전략 (거래비용 포함/미포함)
5. 예제 3.8 - 시가 기반 평균회귀 전략

참고: 예제 3.1은 Yahoo Finance 다운로드 데모, 예제 3.2/3.3은 MATLAB/Excel 전용이므로 생략
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UTIL_DIR = PROJECT_ROOT / "src"
REPORT_DIR = SCRIPT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

# 공유 유틸리티 임포트
sys.path.insert(0, str(UTIL_DIR))
from calculateMaxDD import calculateMaxDD

# matplotlib 스타일 설정
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

# 한글 폰트 설정 시도 (없으면 기본 폰트 사용)
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False


def read_excel_safe(path, **kwargs):
    """XLS 파일을 읽되, calamine 엔진 우선 사용 후 폴백"""
    try:
        return pd.read_excel(path, engine='calamine', **kwargs)
    except (ImportError, Exception):
        return pd.read_excel(path, **kwargs)


class Chapter3Analyzer:
    """Chapter 3 백테스팅 종합 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.figures = []

        # 디렉토리 생성
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 데이터 로드
    # ============================================================
    def load_data(self):
        """모든 분석에 필요한 데이터를 로드"""
        print("=" * 70)
        print("📊 데이터 로드 중...")
        print("=" * 70)

        # --- IGE 데이터 ---
        ige_path = DATA_DIR / "IGE.xls"
        if ige_path.exists():
            self.df_ige = read_excel_safe(ige_path)
            self.df_ige.sort_values(by="Date", inplace=True)
            self.df_ige.reset_index(drop=True, inplace=True)
            print(f"  ✅ IGE: {len(self.df_ige)} 거래일 "
                  f"({self.df_ige['Date'].iloc[0].date()} ~ "
                  f"{self.df_ige['Date'].iloc[-1].date()})")
        else:
            self.df_ige = None
            print(f"  ❌ IGE 데이터 없음: {ige_path}")

        # --- SPY 데이터 ---
        spy_path = DATA_DIR / "SPY.xls"
        if spy_path.exists():
            self.df_spy = read_excel_safe(spy_path)
            self.df_spy.sort_values(by="Date", inplace=True)
            self.df_spy.reset_index(drop=True, inplace=True)
            print(f"  ✅ SPY: {len(self.df_spy)} 거래일 "
                  f"({self.df_spy['Date'].iloc[0].date()} ~ "
                  f"{self.df_spy['Date'].iloc[-1].date()})")
        else:
            self.df_spy = None
            print(f"  ❌ SPY 데이터 없음: {spy_path}")

        # --- GLD 데이터 ---
        gld_path = DATA_DIR / "GLD.xls"
        if gld_path.exists():
            self.df_gld = read_excel_safe(gld_path)
            self.df_gld.sort_values(by="Date", inplace=True)
            self.df_gld.reset_index(drop=True, inplace=True)
            print(f"  ✅ GLD: {len(self.df_gld)} 거래일 "
                  f"({self.df_gld['Date'].iloc[0].date()} ~ "
                  f"{self.df_gld['Date'].iloc[-1].date()})")
        else:
            self.df_gld = None
            print(f"  ❌ GLD 데이터 없음: {gld_path}")

        # --- GDX 데이터 ---
        gdx_path = DATA_DIR / "GDX.xls"
        if gdx_path.exists():
            self.df_gdx = read_excel_safe(gdx_path)
            self.df_gdx.sort_values(by="Date", inplace=True)
            self.df_gdx.reset_index(drop=True, inplace=True)
            print(f"  ✅ GDX: {len(self.df_gdx)} 거래일 "
                  f"({self.df_gdx['Date'].iloc[0].date()} ~ "
                  f"{self.df_gdx['Date'].iloc[-1].date()})")
        else:
            self.df_gdx = None
            print(f"  ❌ GDX 데이터 없음: {gdx_path}")

        # --- SPX 종가 데이터 (횡단면 전략) ---
        spx_cl_path = DATA_DIR / "SPX_20071123.txt"
        if spx_cl_path.exists():
            self.df_spx_cl = pd.read_table(spx_cl_path)
            self.df_spx_cl["Date"] = self.df_spx_cl["Date"].astype("int")
            self.df_spx_cl.set_index("Date", inplace=True)
            self.df_spx_cl.sort_index(inplace=True)
            print(f"  ✅ SPX 종가: {self.df_spx_cl.shape[0]} 거래일 x "
                  f"{self.df_spx_cl.shape[1]} 종목")
        else:
            self.df_spx_cl = None
            print(f"  ❌ SPX 종가 데이터 없음: {spx_cl_path}")

        # --- SPX 시가 데이터 ---
        spx_op_path = DATA_DIR / "SPX_op_20071123.txt"
        if spx_op_path.exists():
            self.df_spx_op = pd.read_table(spx_op_path)
            self.df_spx_op["Date"] = self.df_spx_op["Date"].astype("int")
            self.df_spx_op.set_index("Date", inplace=True)
            self.df_spx_op.sort_index(inplace=True)
            print(f"  ✅ SPX 시가: {self.df_spx_op.shape[0]} 거래일 x "
                  f"{self.df_spx_op.shape[1]} 종목")
        else:
            self.df_spx_op = None
            print(f"  ❌ SPX 시가 데이터 없음: {spx_op_path}")

        print()

    # ============================================================
    # 예제 3.4: 샤프 비율 계산
    # ============================================================
    def analyze_sharpe_ratio(self):
        """롱온리 vs 시장중립 전략의 샤프 비율 비교 (예제 3.4)"""
        print("=" * 70)
        print("📈 예제 3.4: 샤프 비율 계산 (롱온리 vs 시장중립)")
        print("=" * 70)

        if self.df_ige is None or self.df_spy is None:
            print("  ⚠️ IGE 또는 SPY 데이터가 없어 분석을 건너뜁니다.")
            self.results['sharpe'] = None
            return

        # --- 1) IGE 롱온리 전략 ---
        ige_daily_ret = self.df_ige["Adj Close"].pct_change()
        rf_daily = 0.04 / 252  # 연 4% 무위험이자율
        excess_ret = ige_daily_ret - rf_daily
        sharpe_long = float(np.sqrt(252) * np.mean(excess_ret.dropna())
                           / np.std(excess_ret.dropna()))

        print(f"\n  [1] IGE 롱온리 전략")
        print(f"      일간 평균 수익률:    {ige_daily_ret.mean():.6f}")
        print(f"      일간 수익률 표준편차: {ige_daily_ret.std():.6f}")
        print(f"      연율화 샤프 비율:    {sharpe_long:.4f}")

        # --- 2) IGE-SPY 시장중립 전략 ---
        df_merged = pd.merge(self.df_ige, self.df_spy, on="Date",
                             suffixes=("_IGE", "_SPY"))
        df_merged["Date"] = pd.to_datetime(df_merged["Date"])
        df_merged.set_index("Date", inplace=True)
        df_merged.sort_index(inplace=True)

        daily_ret = df_merged[["Adj Close_IGE", "Adj Close_SPY"]].pct_change()
        daily_ret.rename(columns={"Adj Close_IGE": "IGE", "Adj Close_SPY": "SPY"},
                         inplace=True)
        # 시장중립: (IGE - SPY) / 2
        net_ret = (daily_ret["IGE"] - daily_ret["SPY"]) / 2
        sharpe_neutral = float(np.sqrt(252) * np.mean(net_ret.dropna())
                               / np.std(net_ret.dropna()))

        print(f"\n  [2] IGE-SPY 시장중립 전략")
        print(f"      일간 평균 순수익률:   {net_ret.mean():.6f}")
        print(f"      일간 순수익률 표준편차: {net_ret.std():.6f}")
        print(f"      연율화 샤프 비율:     {sharpe_neutral:.4f}")

        # --- 결과 저장 ---
        self.results['sharpe'] = {
            'long_only': {
                'sharpe_ratio': sharpe_long,
                'mean_daily_ret': float(ige_daily_ret.mean()),
                'std_daily_ret': float(ige_daily_ret.std()),
            },
            'market_neutral': {
                'sharpe_ratio': sharpe_neutral,
                'mean_daily_ret': float(net_ret.mean()),
                'std_daily_ret': float(net_ret.std()),
            },
        }

        # --- 시각화: 일간 수익률 비교 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(ige_daily_ret.values, linewidth=0.5, alpha=0.7, color='steelblue')
        axes[0].axhline(y=ige_daily_ret.mean(), color='red', linestyle='--',
                        linewidth=1, label=f'Mean={ige_daily_ret.mean():.4f}')
        axes[0].set_title('IGE Long-Only Daily Returns', fontsize=12)
        axes[0].set_ylabel('Daily Return')
        axes[0].legend(fontsize=9)

        axes[1].plot(net_ret.values, linewidth=0.5, alpha=0.7, color='darkorange')
        axes[1].axhline(y=net_ret.mean(), color='red', linestyle='--',
                        linewidth=1, label=f'Mean={net_ret.mean():.4f}')
        axes[1].set_title('IGE-SPY Market Neutral Daily Returns', fontsize=12)
        axes[1].set_ylabel('Daily Return')
        axes[1].legend(fontsize=9)

        fig.suptitle('Example 3.4: Sharpe Ratio Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        fig_path = FIGURES_DIR / "ex3_4_sharpe_ratio_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(('ex3_4_sharpe_ratio_comparison.png',
                             '예제 3.4: 롱온리 vs 시장중립 일간 수익률 비교'))

        # 누적 수익률 비교 차트
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        cum_long = (1 + ige_daily_ret.fillna(0)).cumprod() - 1
        cum_neutral = (1 + net_ret.fillna(0)).cumprod() - 1
        ax2.plot(cum_long.values, label=f'IGE Long-Only (SR={sharpe_long:.2f})',
                 linewidth=1.2, color='steelblue')
        ax2.plot(cum_neutral.values,
                 label=f'IGE-SPY Market Neutral (SR={sharpe_neutral:.2f})',
                 linewidth=1.2, color='darkorange')
        ax2.set_title('Cumulative Returns: Long-Only vs Market Neutral', fontsize=13)
        ax2.set_ylabel('Cumulative Return')
        ax2.set_xlabel('Trading Days')
        ax2.legend(fontsize=10)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        fig2_path = FIGURES_DIR / "ex3_4_cumulative_returns.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        self.figures.append(('ex3_4_cumulative_returns.png',
                             '예제 3.4: 누적 수익률 비교'))

        # 내부 변수 저장 (이후 MDD 분석에서 사용)
        self._net_ret = net_ret
        print()

    # ============================================================
    # 예제 3.5: 최대 낙폭 계산
    # ============================================================
    def analyze_max_drawdown(self):
        """최대 낙폭(MDD) 및 최대 낙폭 지속기간 계산 (예제 3.5)"""
        print("=" * 70)
        print("📉 예제 3.5: 최대 낙폭 (Maximum Drawdown) 계산")
        print("=" * 70)

        if not hasattr(self, '_net_ret') or self._net_ret is None:
            print("  ⚠️ 선행 분석(예제 3.4)의 결과가 없어 건너뜁니다.")
            self.results['max_drawdown'] = None
            return

        net_ret = self._net_ret.dropna()
        # 누적 복리 수익률
        cum_ret = np.cumprod(1 + net_ret.values) - 1

        # calculateMaxDD 호출
        max_dd, max_ddd, dd_start_idx = calculateMaxDD(cum_ret)

        print(f"\n  시장중립(IGE-SPY) 전략 최대 낙폭 분석:")
        print(f"  -------------------------------------------")
        print(f"  최대 낙폭 (Max Drawdown):        {max_dd:.4f} ({max_dd*100:.2f}%)")
        print(f"  최대 낙폭 지속기간 (Max DD Dur):  {int(max_ddd)} 거래일")
        print(f"  최대 낙폭 발생일 인덱스:          {int(dd_start_idx)}")

        self.results['max_drawdown'] = {
            'max_dd': float(max_dd),
            'max_dd_pct': float(max_dd * 100),
            'max_dd_duration': int(max_ddd),
            'dd_start_idx': int(dd_start_idx),
        }

        # --- 시각화: 누적 수익률 + 낙폭 ---
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})

        # 누적 수익률
        axes[0].plot(cum_ret, linewidth=1.0, color='steelblue',
                     label='Cumulative Return')
        # 고수위선 (High Water Mark)
        hwm = np.maximum.accumulate(cum_ret)
        axes[0].plot(hwm, linewidth=0.8, color='green', linestyle='--',
                     alpha=0.7, label='High Water Mark')
        axes[0].axvline(x=dd_start_idx, color='red', linestyle=':', alpha=0.6,
                        label=f'Max DD at idx={dd_start_idx}')
        axes[0].set_title('IGE-SPY Market Neutral: Cumulative Return & HWM',
                          fontsize=13)
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend(fontsize=9)

        # 낙폭 시계열
        drawdown = (1 + cum_ret) / (1 + hwm) - 1
        axes[1].fill_between(range(len(drawdown)), drawdown, 0,
                             color='salmon', alpha=0.5)
        axes[1].plot(drawdown, linewidth=0.8, color='red')
        axes[1].set_title('Drawdown', fontsize=12)
        axes[1].set_ylabel('Drawdown')
        axes[1].set_xlabel('Trading Days')

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ex3_5_max_drawdown.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(('ex3_5_max_drawdown.png',
                             '예제 3.5: 최대 낙폭 분석'))
        print()

    # ============================================================
    # 예제 3.6: GLD-GDX 페어 트레이딩
    # ============================================================
    def analyze_pair_trading(self):
        """GLD-GDX 페어 트레이딩: z-score 기반 진입/청산 (예제 3.6)"""
        print("=" * 70)
        print("🔄 예제 3.6: GLD-GDX 페어 트레이딩")
        print("=" * 70)

        if self.df_gld is None or self.df_gdx is None:
            print("  ⚠️ GLD 또는 GDX 데이터가 없어 건너뜁니다.")
            self.results['pair_trading'] = None
            return

        # 데이터 병합
        df = pd.merge(self.df_gld, self.df_gdx, on="Date",
                       suffixes=("_GLD", "_GDX"))
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # 학습/테스트 분할 (첫 252일 = 학습, 나머지 = 테스트)
        trainset = np.arange(0, 252)
        testset = np.arange(trainset.shape[0], df.shape[0])

        print(f"\n  학습 기간: {df.index[trainset[0]].date()} ~ "
              f"{df.index[trainset[-1]].date()} ({len(trainset)}일)")
        print(f"  테스트 기간: {df.index[testset[0]].date()} ~ "
              f"{df.index[testset[-1]].date()} ({len(testset)}일)")

        # --- 헤지 비율 추정 (OLS, 학습 기간) ---
        model = sm.OLS(
            df["Adj Close_GLD"].iloc[trainset],
            df["Adj Close_GDX"].iloc[trainset]
        )
        results_ols = model.fit()
        hedge_ratio = results_ols.params.iloc[0]
        print(f"\n  헤지 비율 (Hedge Ratio): {hedge_ratio:.4f}")
        print(f"  OLS R-squared: {results_ols.rsquared:.4f}")

        # --- 스프레드 및 z-score ---
        spread = df["Adj Close_GLD"] - hedge_ratio * df["Adj Close_GDX"]
        spread_mean = np.mean(spread.iloc[trainset])
        spread_std = np.std(spread.iloc[trainset])
        df["zscore"] = (spread - spread_mean) / spread_std

        print(f"  스프레드 평균 (학습): {spread_mean:.4f}")
        print(f"  스프레드 표준편차 (학습): {spread_std:.4f}")

        # --- 포지션 생성 (z-score 기반) ---
        # 진입: |z| >= 2, 청산: |z| <= 1
        df["pos_GLD_Long"] = 0.0
        df["pos_GDX_Long"] = 0.0
        df["pos_GLD_Short"] = 0.0
        df["pos_GDX_Short"] = 0.0

        # 숏 스프레드: z >= 2 → GLD 숏, GDX 롱
        df.loc[df.zscore >= 2, ("pos_GLD_Short", "pos_GDX_Short")] = [-1, 1]
        # 롱 스프레드: z <= -2 → GLD 롱, GDX 숏
        df.loc[df.zscore <= -2, ("pos_GLD_Long", "pos_GDX_Long")] = [1, -1]
        # 숏 스프레드 청산: z <= 1
        df.loc[df.zscore <= 1, ("pos_GLD_Short", "pos_GDX_Short")] = 0
        # 롱 스프레드 청산: z >= -1
        df.loc[df.zscore >= -1, ("pos_GLD_Long", "pos_GDX_Long")] = 0

        # ffill로 포지션 유지 (최신 pandas API)
        df[["pos_GLD_Long", "pos_GDX_Long",
            "pos_GLD_Short", "pos_GDX_Short"]] = (
            df[["pos_GLD_Long", "pos_GDX_Long",
                "pos_GLD_Short", "pos_GDX_Short"]].ffill()
        )

        positions_long = df[["pos_GLD_Long", "pos_GDX_Long"]]
        positions_short = df[["pos_GLD_Short", "pos_GDX_Short"]]
        positions = np.array(positions_long) + np.array(positions_short)
        positions_df = pd.DataFrame(positions,
                                    index=df.index,
                                    columns=["GLD", "GDX"])

        # --- PnL 계산 ---
        daily_ret = df[["Adj Close_GLD", "Adj Close_GDX"]].pct_change()
        pnl = (np.array(positions_df.shift()) * np.array(daily_ret)).sum(axis=1)

        # 학습/테스트 샤프 비율
        sharpe_train = float(
            np.sqrt(252) * np.mean(pnl[trainset[1:]]) / np.std(pnl[trainset[1:]])
        )
        sharpe_test = float(
            np.sqrt(252) * np.mean(pnl[testset]) / np.std(pnl[testset])
        )

        # 누적 PnL
        cum_pnl_train = np.cumsum(pnl[trainset[1:]])
        cum_pnl_test = np.cumsum(pnl[testset])

        # MDD 계산 (테스트 기간)
        cum_ret_test = np.cumprod(1 + pnl[testset]) - 1
        if len(cum_ret_test) > 1:
            mdd_test, mdd_dur_test, _ = calculateMaxDD(cum_ret_test)
        else:
            mdd_test, mdd_dur_test = 0.0, 0

        print(f"\n  --- 성과 요약 ---")
        print(f"  학습 샤프 비율:  {sharpe_train:.4f}")
        print(f"  테스트 샤프 비율: {sharpe_test:.4f}")
        print(f"  테스트 최대 낙폭: {mdd_test:.4f} ({mdd_test*100:.2f}%)")
        print(f"  테스트 MDD 지속:  {int(mdd_dur_test)} 거래일")

        self.results['pair_trading'] = {
            'hedge_ratio': float(hedge_ratio),
            'r_squared': float(results_ols.rsquared),
            'spread_mean': float(spread_mean),
            'spread_std': float(spread_std),
            'sharpe_train': sharpe_train,
            'sharpe_test': sharpe_test,
            'mdd_test': float(mdd_test),
            'mdd_dur_test': int(mdd_dur_test),
            'n_train': len(trainset),
            'n_test': len(testset),
        }

        # --- 시각화 1: 스프레드 및 z-score ---
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        axes[0].plot(spread.iloc[trainset].values, label='Train', color='steelblue',
                     linewidth=0.9)
        axes[0].plot(range(len(trainset), len(trainset) + len(testset)),
                     spread.iloc[testset].values, label='Test', color='darkorange',
                     linewidth=0.9)
        axes[0].axhline(y=spread_mean, color='green', linestyle='--', linewidth=0.8,
                        label=f'Mean={spread_mean:.2f}')
        axes[0].set_title('GLD-GDX Spread (GLD - HR*GDX)', fontsize=13)
        axes[0].set_ylabel('Spread')
        axes[0].legend(fontsize=9)

        zscore_vals = df["zscore"].values
        axes[1].plot(zscore_vals, linewidth=0.7, color='purple')
        axes[1].axhline(y=2, color='red', linestyle='--', linewidth=0.8,
                        label='Entry: z=+/-2')
        axes[1].axhline(y=-2, color='red', linestyle='--', linewidth=0.8)
        axes[1].axhline(y=1, color='green', linestyle=':', linewidth=0.8,
                        label='Exit: z=+/-1')
        axes[1].axhline(y=-1, color='green', linestyle=':', linewidth=0.8)
        axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axes[1].set_title('Z-Score', fontsize=12)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_xlabel('Trading Days')
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ex3_6_pair_spread_zscore.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(('ex3_6_pair_spread_zscore.png',
                             '예제 3.6: GLD-GDX 스프레드 및 Z-Score'))

        # --- 시각화 2: 누적 PnL ---
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(cum_pnl_test, linewidth=1.2, color='darkorange',
                 label=f'Test Cumulative PnL (SR={sharpe_test:.2f})')
        ax2.set_title('GLD-GDX Pair Trading: Cumulative PnL (Test Period)',
                      fontsize=13)
        ax2.set_ylabel('Cumulative PnL')
        ax2.set_xlabel('Trading Days (Test Period)')
        ax2.legend(fontsize=10)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        fig2_path = FIGURES_DIR / "ex3_6_pair_cumulative_pnl.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        self.figures.append(('ex3_6_pair_cumulative_pnl.png',
                             '예제 3.6: GLD-GDX 페어 트레이딩 누적 PnL (테스트)'))
        print()

    # ============================================================
    # 예제 3.7: 횡단면 평균회귀 (종가 기반)
    # ============================================================
    def analyze_cross_sectional_mean_reversion(self):
        """SPX 횡단면 평균회귀 전략 - 거래비용 유무 비교 (예제 3.7)"""
        print("=" * 70)
        print("📊 예제 3.7: 횡단면 평균회귀 전략 (종가 기반)")
        print("=" * 70)

        if self.df_spx_cl is None:
            print("  ⚠️ SPX 종가 데이터가 없어 건너뜁니다.")
            self.results['cross_sectional'] = None
            return

        start_date = 20060101
        end_date = 20061231

        df = self.df_spx_cl
        daily_ret = df.pct_change()

        # 시장 일간 수익률 (전 종목 평균)
        market_daily_ret = daily_ret.mean(axis=1)

        # 가중치: -(개별 수익률 - 시장 수익률) → 전일 하락 종목 매수, 상승 종목 매도
        weights = -(
            np.array(daily_ret)
            - np.array(market_daily_ret).reshape((daily_ret.shape[0], 1))
        )
        wtsum = np.nansum(np.abs(weights), axis=1)
        weights[wtsum == 0, :] = 0
        wtsum[wtsum == 0] = 1
        weights = weights / wtsum.reshape((daily_ret.shape[0], 1))

        # PnL 계산
        daily_pnl = np.nansum(
            np.array(pd.DataFrame(weights).shift()) * np.array(daily_ret),
            axis=1
        )
        # 분석 기간 필터
        mask = np.logical_and(df.index >= start_date, df.index <= end_date)
        daily_pnl_period = daily_pnl[mask]

        sharpe_no_tcost = float(
            np.sqrt(252) * np.mean(daily_pnl_period)
            / np.std(daily_pnl_period)
        )

        print(f"\n  분석 기간: {start_date} ~ {end_date}")
        print(f"  종목 수: {df.shape[1]}")
        print(f"  [1] 거래비용 미포함")
        print(f"      일간 평균 PnL:   {np.mean(daily_pnl_period):.6f}")
        print(f"      일간 PnL 표준편차: {np.std(daily_pnl_period):.6f}")
        print(f"      연율화 샤프 비율:  {sharpe_no_tcost:.4f}")

        # --- 거래비용 포함 ---
        one_way_tcost = 0.0005  # 5 bps 편도
        weights_period = weights[mask]
        turnover = np.nansum(
            np.abs(weights_period - np.array(pd.DataFrame(weights_period).shift())),
            axis=1
        )
        daily_pnl_minus_tcost = daily_pnl_period - turnover * one_way_tcost
        sharpe_with_tcost = float(
            np.sqrt(252) * np.mean(daily_pnl_minus_tcost)
            / np.std(daily_pnl_minus_tcost)
        )

        print(f"\n  [2] 거래비용 포함 (편도 {one_way_tcost*10000:.0f} bps)")
        print(f"      일간 평균 PnL:   {np.mean(daily_pnl_minus_tcost):.6f}")
        print(f"      연율화 샤프 비율:  {sharpe_with_tcost:.4f}")
        print(f"      샤프 비율 감소:    {sharpe_no_tcost - sharpe_with_tcost:.4f}")

        # MDD 계산
        cum_ret_no_tc = np.cumprod(1 + daily_pnl_period) - 1
        if len(cum_ret_no_tc) > 1:
            mdd_no_tc, mdd_dur_no_tc, _ = calculateMaxDD(cum_ret_no_tc)
        else:
            mdd_no_tc, mdd_dur_no_tc = 0.0, 0

        cum_ret_tc = np.cumprod(1 + daily_pnl_minus_tcost) - 1
        if len(cum_ret_tc) > 1:
            mdd_tc, mdd_dur_tc, _ = calculateMaxDD(cum_ret_tc)
        else:
            mdd_tc, mdd_dur_tc = 0.0, 0

        self.results['cross_sectional'] = {
            'start_date': start_date,
            'end_date': end_date,
            'n_stocks': int(df.shape[1]),
            'sharpe_no_tcost': sharpe_no_tcost,
            'sharpe_with_tcost': sharpe_with_tcost,
            'tcost_bps': one_way_tcost * 10000,
            'mean_pnl_no_tc': float(np.mean(daily_pnl_period)),
            'mean_pnl_tc': float(np.mean(daily_pnl_minus_tcost)),
            'mdd_no_tc': float(mdd_no_tc),
            'mdd_dur_no_tc': int(mdd_dur_no_tc),
            'mdd_tc': float(mdd_tc),
            'mdd_dur_tc': int(mdd_dur_tc),
            'mean_turnover': float(np.mean(turnover[1:])),
        }

        # --- 시각화 ---
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))

        # 누적 PnL 비교
        cum_pnl_no_tc = np.cumsum(daily_pnl_period)
        cum_pnl_tc = np.cumsum(daily_pnl_minus_tcost)
        axes[0].plot(cum_pnl_no_tc, linewidth=1.2, color='steelblue',
                     label=f'Without TC (SR={sharpe_no_tcost:.2f})')
        axes[0].plot(cum_pnl_tc, linewidth=1.2, color='darkorange',
                     label=f'With TC @ {one_way_tcost*10000:.0f}bps '
                           f'(SR={sharpe_with_tcost:.2f})')
        axes[0].set_title('Cross-Sectional Mean Reversion (Close Prices): '
                          'Cumulative PnL', fontsize=13)
        axes[0].set_ylabel('Cumulative PnL')
        axes[0].legend(fontsize=10)
        axes[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # 일간 PnL 분포
        axes[1].hist(daily_pnl_period, bins=50, alpha=0.6, color='steelblue',
                     edgecolor='white', label='Without TC')
        axes[1].hist(daily_pnl_minus_tcost, bins=50, alpha=0.5, color='darkorange',
                     edgecolor='white', label='With TC')
        axes[1].set_title('Daily PnL Distribution', fontsize=12)
        axes[1].set_xlabel('Daily PnL')
        axes[1].set_ylabel('Frequency')
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ex3_7_cross_sectional_close.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(('ex3_7_cross_sectional_close.png',
                             '예제 3.7: 횡단면 평균회귀 (종가 기반) 성과'))
        print()

    # ============================================================
    # 예제 3.8: 시가 기반 평균회귀
    # ============================================================
    def analyze_open_price_strategy(self):
        """시가 기반 횡단면 평균회귀 전략 (예제 3.8)"""
        print("=" * 70)
        print("📊 예제 3.8: 횡단면 평균회귀 전략 (시가 기반)")
        print("=" * 70)

        if self.df_spx_op is None:
            print("  ⚠️ SPX 시가 데이터가 없어 건너뜁니다.")
            self.results['open_price'] = None
            return

        start_date = 20060101
        end_date = 20061231

        df = self.df_spx_op
        daily_ret = df.pct_change()
        market_daily_ret = daily_ret.mean(axis=1)

        weights = -(
            np.array(daily_ret)
            - np.array(market_daily_ret).reshape((daily_ret.shape[0], 1))
        )
        wtsum = np.nansum(np.abs(weights), axis=1)
        weights[wtsum == 0, :] = 0
        wtsum[wtsum == 0] = 1
        weights = weights / wtsum.reshape((daily_ret.shape[0], 1))

        daily_pnl = np.nansum(
            np.array(pd.DataFrame(weights).shift()) * np.array(daily_ret),
            axis=1
        )
        mask = np.logical_and(df.index >= start_date, df.index <= end_date)
        daily_pnl_period = daily_pnl[mask]

        sharpe_no_tcost = float(
            np.sqrt(252) * np.mean(daily_pnl_period)
            / np.std(daily_pnl_period)
        )

        print(f"\n  분석 기간: {start_date} ~ {end_date}")
        print(f"  종목 수: {df.shape[1]}")
        print(f"  [1] 거래비용 미포함")
        print(f"      일간 평균 PnL:   {np.mean(daily_pnl_period):.6f}")
        print(f"      연율화 샤프 비율:  {sharpe_no_tcost:.4f}")

        # --- 거래비용 포함 ---
        one_way_tcost = 0.0005
        weights_period = weights[mask]
        turnover = np.nansum(
            np.abs(weights_period - np.array(pd.DataFrame(weights_period).shift())),
            axis=1
        )
        daily_pnl_minus_tcost = daily_pnl_period - turnover * one_way_tcost
        sharpe_with_tcost = float(
            np.sqrt(252) * np.mean(daily_pnl_minus_tcost)
            / np.std(daily_pnl_minus_tcost)
        )

        print(f"\n  [2] 거래비용 포함 (편도 {one_way_tcost*10000:.0f} bps)")
        print(f"      일간 평균 PnL:   {np.mean(daily_pnl_minus_tcost):.6f}")
        print(f"      연율화 샤프 비율:  {sharpe_with_tcost:.4f}")

        # MDD 계산
        cum_ret_no_tc = np.cumprod(1 + daily_pnl_period) - 1
        if len(cum_ret_no_tc) > 1:
            mdd_no_tc, mdd_dur_no_tc, _ = calculateMaxDD(cum_ret_no_tc)
        else:
            mdd_no_tc, mdd_dur_no_tc = 0.0, 0

        cum_ret_tc = np.cumprod(1 + daily_pnl_minus_tcost) - 1
        if len(cum_ret_tc) > 1:
            mdd_tc, mdd_dur_tc, _ = calculateMaxDD(cum_ret_tc)
        else:
            mdd_tc, mdd_dur_tc = 0.0, 0

        self.results['open_price'] = {
            'start_date': start_date,
            'end_date': end_date,
            'n_stocks': int(df.shape[1]),
            'sharpe_no_tcost': sharpe_no_tcost,
            'sharpe_with_tcost': sharpe_with_tcost,
            'tcost_bps': one_way_tcost * 10000,
            'mean_pnl_no_tc': float(np.mean(daily_pnl_period)),
            'mean_pnl_tc': float(np.mean(daily_pnl_minus_tcost)),
            'mdd_no_tc': float(mdd_no_tc),
            'mdd_dur_no_tc': int(mdd_dur_no_tc),
            'mdd_tc': float(mdd_tc),
            'mdd_dur_tc': int(mdd_dur_tc),
            'mean_turnover': float(np.mean(turnover[1:])),
        }

        # --- 시각화: 종가 vs 시가 기반 전략 비교 ---
        fig, ax = plt.subplots(figsize=(12, 5))
        cum_pnl_op = np.cumsum(daily_pnl_period)
        ax.plot(cum_pnl_op, linewidth=1.2, color='forestgreen',
                label=f'Open Price (SR={sharpe_no_tcost:.2f})')

        # 종가 전략이 이미 계산되었다면 비교
        if self.results.get('cross_sectional'):
            cs = self.results['cross_sectional']
            # 종가 전략의 누적 PnL은 재계산 필요
            df_cl = self.df_spx_cl
            dr_cl = df_cl.pct_change()
            mdr_cl = dr_cl.mean(axis=1)
            w_cl = -(np.array(dr_cl)
                     - np.array(mdr_cl).reshape((dr_cl.shape[0], 1)))
            ws_cl = np.nansum(np.abs(w_cl), axis=1)
            w_cl[ws_cl == 0, :] = 0
            ws_cl[ws_cl == 0] = 1
            w_cl = w_cl / ws_cl.reshape((dr_cl.shape[0], 1))
            dpnl_cl = np.nansum(
                np.array(pd.DataFrame(w_cl).shift()) * np.array(dr_cl), axis=1
            )
            mask_cl = np.logical_and(
                df_cl.index >= start_date, df_cl.index <= end_date
            )
            cum_pnl_cl = np.cumsum(dpnl_cl[mask_cl])
            ax.plot(cum_pnl_cl, linewidth=1.2, color='steelblue',
                    label=f'Close Price (SR={cs["sharpe_no_tcost"]:.2f})')

        ax.set_title('Cross-Sectional Mean Reversion: Open vs Close Price',
                     fontsize=13)
        ax.set_ylabel('Cumulative PnL')
        ax.set_xlabel('Trading Days (2006)')
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        fig_path = FIGURES_DIR / "ex3_8_open_vs_close.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(('ex3_8_open_vs_close.png',
                             '예제 3.8: 시가 vs 종가 기반 횡단면 전략 비교'))
        print()

    # ============================================================
    # 리포트 생성
    # ============================================================
    def generate_report(self):
        """마크다운 형식의 종합 리포트 생성"""
        print("=" * 70)
        print("📝 종합 리포트 생성 중...")
        print("=" * 70)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        # ─── 헤더 ───
        lines.append(f"# 제3장 백테스팅 - 종합 분석 리포트")
        lines.append("")
        lines.append(f"> **생성 일시:** {now}")
        lines.append(f"> **출처:** Ernest Chan, *Quantitative Trading* (2nd Ed., 2021), Chapter 3")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ─── 1. 개요 ───
        lines.append("## 1. 개요 및 문제 정의")
        lines.append("")
        lines.append("백테스팅(Backtesting)은 과거 데이터를 사용하여 트레이딩 전략의 성과를 검증하는 과정이다.")
        lines.append("이 챕터에서는 다음 핵심 성과 지표와 전략을 다룬다:")
        lines.append("")
        lines.append("### 핵심 수식")
        lines.append("")
        lines.append("**샤프 비율(Sharpe Ratio)** 의 연율화:")
        lines.append("")
        lines.append("$$")
        lines.append(r"\text{Annualized Sharpe Ratio} = \sqrt{N_T} \times "
                     r"\frac{\bar{R} - r_f}{\sigma_R}")
        lines.append("$$")
        lines.append("")
        lines.append(r"여기서 $N_T$ 는 연간 거래 기간 수(일별이면 252), "
                     r"$\bar{R}$ 은 기간 평균 수익률, $r_f$ 는 무위험이자율, "
                     r"$\sigma_R$ 은 수익률 표준편차이다.")
        lines.append("")
        lines.append("**최대 낙폭(Maximum Drawdown):**")
        lines.append("")
        lines.append("$$")
        lines.append(r"\text{MDD} = \min_t \left( "
                     r"\frac{1 + \text{CumRet}(t)}{1 + \text{HWM}(t)} - 1 \right)")
        lines.append("$$")
        lines.append("")
        lines.append(r"여기서 $\text{HWM}(t) = \max_{s \le t} \text{CumRet}(s)$ 는 "
                     r"고수위선(High Water Mark)이다.")
        lines.append("")
        lines.append("**달러-중립 포트폴리오** 의 경우, 자기금융(self-financing) "
                     "특성상 무위험이자율 차감이 불필요하다.")
        lines.append("")
        lines.append("### 예제 구성")
        lines.append("")
        lines.append("| 예제 | 내용 | 핵심 개념 |")
        lines.append("|------|------|-----------|")
        lines.append("| 3.1 | Yahoo Finance 데이터 다운로드 | 데이터 수집 (MATLAB/Python/R) |")
        lines.append("| 3.2 | 주식분할/배당 조정 | 데이터 전처리 (MATLAB/Excel) |")
        lines.append("| 3.3 | 생존자 편향 영향 | 데이터 품질 (MATLAB/Excel) |")
        lines.append("| 3.4 | 샤프 비율 계산 | 롱온리 vs 시장중립 |")
        lines.append("| 3.5 | 최대 낙폭 계산 | MDD, MDD Duration |")
        lines.append("| 3.6 | GLD-GDX 페어 트레이딩 | 공적분, z-score 진입/청산 |")
        lines.append("| 3.7 | 횡단면 평균회귀 (종가) | 거래비용 영향 분석 |")
        lines.append("| 3.8 | 횡단면 평균회귀 (시가) | 시가 vs 종가 비교 |")
        lines.append("")

        # ─── 2. 사용 데이터 ───
        lines.append("---")
        lines.append("")
        lines.append("## 2. 사용 데이터")
        lines.append("")
        lines.append("| 파일명 | 형식 | 설명 | 사용 예제 |")
        lines.append("|--------|------|------|-----------|")
        lines.append("| `IGE.xls` | XLS | iShares North American Natural Resources ETF | 3.4, 3.5 |")
        lines.append("| `SPY.xls` | XLS | SPDR S&P 500 ETF Trust | 3.4 |")
        lines.append("| `GLD.xls` | XLS | SPDR Gold Trust | 3.6 |")
        lines.append("| `GDX.xls` | XLS | VanEck Gold Miners ETF | 3.6 |")
        lines.append("| `SPX_20071123.txt` | TSV | S&P 500 구성 종목 종가 | 3.7 |")
        lines.append("| `SPX_op_20071123.txt` | TSV | S&P 500 구성 종목 시가 | 3.8 |")
        lines.append("")

        # ─── 3. 예제 3.4 ───
        lines.append("---")
        lines.append("")
        lines.append("## 3. 예제 3.4: 샤프 비율 계산")
        lines.append("")
        sr = self.results.get('sharpe')
        if sr:
            lo = sr['long_only']
            mn = sr['market_neutral']
            lines.append("IGE 롱온리 전략과 IGE-SPY 시장중립 전략의 샤프 비율을 비교한다.")
            lines.append("")
            lines.append("| 지표 | 롱온리 (IGE) | 시장중립 (IGE-SPY) |")
            lines.append("|------|:------------:|:-----------------:|")
            lines.append(f"| 일간 평균 수익률 | {lo['mean_daily_ret']:.6f} | "
                         f"{mn['mean_daily_ret']:.6f} |")
            lines.append(f"| 일간 수익률 표준편차 | {lo['std_daily_ret']:.6f} | "
                         f"{mn['std_daily_ret']:.6f} |")
            lines.append(f"| **연율화 샤프 비율** | **{lo['sharpe_ratio']:.4f}** | "
                         f"**{mn['sharpe_ratio']:.4f}** |")
            lines.append("")
            # 해석
            if lo['sharpe_ratio'] > mn['sharpe_ratio']:
                lines.append("✅ IGE 롱온리 전략의 샤프 비율이 더 높다. 이 기간 동안 에너지 섹터의 "
                             "강세장이 시장중립 전략보다 유리했다.")
            else:
                lines.append("✅ 시장중립 전략의 샤프 비율이 더 높다. 시장 리스크를 제거함으로써 "
                             "위험 조정 성과가 개선되었다.")
            lines.append("")
            lines.append("⚠️ **주의:** 시장중립 전략의 경우 달러-중립이므로 무위험이자율을 차감하지 않았다.")
            lines.append("")
        else:
            lines.append("*데이터 부재로 분석 생략*")
            lines.append("")

        for fname, caption in self.figures:
            if fname.startswith('ex3_4_'):
                lines.append(f"![{caption}](figures/{fname})")
                lines.append("")

        # ─── 4. 예제 3.5 ───
        lines.append("---")
        lines.append("")
        lines.append("## 4. 예제 3.5: 최대 낙폭 분석")
        lines.append("")
        mdd = self.results.get('max_drawdown')
        if mdd:
            lines.append("IGE-SPY 시장중립 전략의 누적 복리 수익률에서 최대 낙폭을 계산한다.")
            lines.append("")
            lines.append("| 지표 | 값 |")
            lines.append("|------|---:|")
            lines.append(f"| 최대 낙폭 (MDD) | {mdd['max_dd_pct']:.2f}% |")
            lines.append(f"| 최대 낙폭 지속기간 | {mdd['max_dd_duration']} 거래일 |")
            lines.append(f"| MDD 발생 인덱스 | {mdd['dd_start_idx']} |")
            lines.append("")
            if abs(mdd['max_dd_pct']) > 20:
                lines.append(f"❌ 최대 낙폭 {mdd['max_dd_pct']:.1f}% 는 상당히 크며, "
                             "실전 운용 시 심리적 압박이 될 수 있다.")
            elif abs(mdd['max_dd_pct']) > 10:
                lines.append(f"⚠️ 최대 낙폭 {mdd['max_dd_pct']:.1f}% 는 보통 수준이나, "
                             "리스크 관리에 주의가 필요하다.")
            else:
                lines.append(f"✅ 최대 낙폭 {mdd['max_dd_pct']:.1f}% 는 양호한 수준이다.")
            lines.append("")
        else:
            lines.append("*선행 분석 결과 부재로 생략*")
            lines.append("")

        for fname, caption in self.figures:
            if fname.startswith('ex3_5_'):
                lines.append(f"![{caption}](figures/{fname})")
                lines.append("")

        # ─── 5. 예제 3.6 ───
        lines.append("---")
        lines.append("")
        lines.append("## 5. 예제 3.6: GLD-GDX 페어 트레이딩")
        lines.append("")
        pt = self.results.get('pair_trading')
        if pt:
            lines.append("GLD(금 ETF)와 GDX(금광 ETF) 간의 페어 트레이딩 전략이다.")
            lines.append("학습 기간에서 OLS로 헤지 비율을 추정하고, z-score 기반으로 진입/청산한다.")
            lines.append("")
            lines.append("### 전략 파라미터")
            lines.append("")
            lines.append("| 파라미터 | 값 |")
            lines.append("|----------|---:|")
            lines.append(f"| 헤지 비율 | {pt['hedge_ratio']:.4f} |")
            lines.append(f"| OLS $R^2$ | {pt['r_squared']:.4f} |")
            lines.append(f"| 스프레드 평균 | {pt['spread_mean']:.4f} |")
            lines.append(f"| 스프레드 표준편차 | {pt['spread_std']:.4f} |")
            lines.append(f"| 진입 임계치 | $\\|z\\| \\ge 2$ |")
            lines.append(f"| 청산 임계치 | $\\|z\\| \\le 1$ |")
            lines.append(f"| 학습 기간 | {pt['n_train']} 거래일 |")
            lines.append(f"| 테스트 기간 | {pt['n_test']} 거래일 |")
            lines.append("")
            lines.append("### 성과 요약")
            lines.append("")
            lines.append("| 지표 | 학습(In-Sample) | 테스트(Out-of-Sample) |")
            lines.append("|------|:---------------:|:--------------------:|")
            lines.append(f"| **샤프 비율** | **{pt['sharpe_train']:.4f}** | "
                         f"**{pt['sharpe_test']:.4f}** |")
            lines.append(f"| MDD | - | {pt['mdd_test']*100:.2f}% |")
            lines.append(f"| MDD 지속기간 | - | {pt['mdd_dur_test']} 거래일 |")
            lines.append("")
            # 해석
            if pt['sharpe_test'] > 0.5:
                lines.append("✅ 테스트 기간 샤프 비율이 양호하며, 전략의 유효성이 확인된다.")
            elif pt['sharpe_test'] > 0:
                lines.append("⚠️ 테스트 기간 샤프 비율이 양수이나, 학습 기간 대비 "
                             "하락이 관찰된다. 과적합 가능성을 점검해야 한다.")
            else:
                lines.append("❌ 테스트 기간 샤프 비율이 음수이다. "
                             "학습 기간의 성과가 표본 외에서 재현되지 않았다.")
            lines.append("")
        else:
            lines.append("*데이터 부재로 분석 생략*")
            lines.append("")

        for fname, caption in self.figures:
            if fname.startswith('ex3_6_'):
                lines.append(f"![{caption}](figures/{fname})")
                lines.append("")

        # ─── 6. 예제 3.7 ───
        lines.append("---")
        lines.append("")
        lines.append("## 6. 예제 3.7: 횡단면 평균회귀 (종가)")
        lines.append("")
        cs = self.results.get('cross_sectional')
        if cs:
            lines.append("S&P 500 구성 종목의 종가 기반 횡단면 평균회귀 전략이다.")
            lines.append("전일 시장 대비 하락(상승)한 종목을 매수(매도)하여 다음 날 청산한다.")
            lines.append("")
            lines.append("**가중치 산출:**")
            lines.append("")
            lines.append("$$")
            lines.append(r"w_i = -\left(r_i - \bar{r}_{\text{market}}\right)")
            lines.append("$$")
            lines.append("")
            lines.append("| 지표 | 거래비용 미포함 | 거래비용 포함 |")
            lines.append("|------|:--------------:|:------------:|")
            lines.append(f"| 일간 평균 PnL | {cs['mean_pnl_no_tc']:.6f} | "
                         f"{cs['mean_pnl_tc']:.6f} |")
            lines.append(f"| **연율화 샤프 비율** | **{cs['sharpe_no_tcost']:.4f}** | "
                         f"**{cs['sharpe_with_tcost']:.4f}** |")
            lines.append(f"| MDD | {cs['mdd_no_tc']*100:.2f}% | {cs['mdd_tc']*100:.2f}% |")
            lines.append(f"| MDD 지속기간 | {cs['mdd_dur_no_tc']} 거래일 | "
                         f"{cs['mdd_dur_tc']} 거래일 |")
            lines.append(f"| 평균 일간 턴오버 | {cs['mean_turnover']:.4f} | - |")
            lines.append("")
            sr_diff = cs['sharpe_no_tcost'] - cs['sharpe_with_tcost']
            lines.append(f"⚠️ 편도 {cs['tcost_bps']:.0f}bps의 거래비용만으로 "
                         f"샤프 비율이 {sr_diff:.4f} 하락했다. "
                         "고빈도 전략에서 거래비용의 영향이 매우 크다.")
            lines.append("")
        else:
            lines.append("*데이터 부재로 분석 생략*")
            lines.append("")

        for fname, caption in self.figures:
            if fname.startswith('ex3_7_'):
                lines.append(f"![{caption}](figures/{fname})")
                lines.append("")

        # ─── 7. 예제 3.8 ───
        lines.append("---")
        lines.append("")
        lines.append("## 7. 예제 3.8: 횡단면 평균회귀 (시가)")
        lines.append("")
        op = self.results.get('open_price')
        if op:
            lines.append("동일한 횡단면 평균회귀 전략을 시가(Open Price) 기반으로 실행한 결과이다.")
            lines.append("시가에서 포지션을 진입/청산하면 종가 대비 실행 가능성(executability) 이 높아지지만,")
            lines.append("시가 데이터의 잡음(noise) 이 더 클 수 있다.")
            lines.append("")
            lines.append("| 지표 | 거래비용 미포함 | 거래비용 포함 |")
            lines.append("|------|:--------------:|:------------:|")
            lines.append(f"| 일간 평균 PnL | {op['mean_pnl_no_tc']:.6f} | "
                         f"{op['mean_pnl_tc']:.6f} |")
            lines.append(f"| **연율화 샤프 비율** | **{op['sharpe_no_tcost']:.4f}** | "
                         f"**{op['sharpe_with_tcost']:.4f}** |")
            lines.append(f"| MDD | {op['mdd_no_tc']*100:.2f}% | {op['mdd_tc']*100:.2f}% |")
            lines.append(f"| MDD 지속기간 | {op['mdd_dur_no_tc']} 거래일 | "
                         f"{op['mdd_dur_tc']} 거래일 |")
            lines.append("")
            # 종가 대비 비교
            if cs:
                lines.append("### 종가 vs 시가 비교")
                lines.append("")
                lines.append("| 가격 유형 | 샤프 (TC 미포함) | 샤프 (TC 포함) |")
                lines.append("|----------|:---------------:|:------------:|")
                lines.append(f"| 종가 (Close) | {cs['sharpe_no_tcost']:.4f} | "
                             f"{cs['sharpe_with_tcost']:.4f} |")
                lines.append(f"| 시가 (Open) | {op['sharpe_no_tcost']:.4f} | "
                             f"{op['sharpe_with_tcost']:.4f} |")
                lines.append("")
                if op['sharpe_no_tcost'] > cs['sharpe_no_tcost']:
                    lines.append("✅ 시가 기반 전략이 종가 기반보다 높은 샤프 비율을 보인다.")
                else:
                    lines.append("⚠️ 종가 기반 전략이 시가 기반보다 높은 샤프 비율을 보인다. "
                                 "단, 종가에서의 실행 가능성은 시가 대비 낮을 수 있다.")
                lines.append("")
        else:
            lines.append("*데이터 부재로 분석 생략*")
            lines.append("")

        for fname, caption in self.figures:
            if fname.startswith('ex3_8_'):
                lines.append(f"![{caption}](figures/{fname})")
                lines.append("")

        # ─── 8. 전략 비교 종합 ───
        lines.append("---")
        lines.append("")
        lines.append("## 8. 전략 백테스트 종합 비교")
        lines.append("")
        lines.append("| 전략 | 유형 | 샤프 비율 | MDD | 비고 |")
        lines.append("|------|------|:---------:|:---:|------|")

        if sr:
            lines.append(f"| IGE 롱온리 | 단일자산 롱 | "
                         f"{sr['long_only']['sharpe_ratio']:.4f} | - | "
                         f"무위험이자율 4% 차감 |")
            lines.append(f"| IGE-SPY 시장중립 | 롱숏 페어 | "
                         f"{sr['market_neutral']['sharpe_ratio']:.4f} | "
                         f"{mdd['max_dd_pct']:.1f}% | 달러 중립 |")
        if pt:
            lines.append(f"| GLD-GDX 페어 (학습) | 통계 차익 | "
                         f"{pt['sharpe_train']:.4f} | - | z-score 기반 |")
            lines.append(f"| GLD-GDX 페어 (테스트) | 통계 차익 | "
                         f"{pt['sharpe_test']:.4f} | {pt['mdd_test']*100:.1f}% | "
                         f"Out-of-sample |")
        if cs:
            lines.append(f"| 횡단면 평균회귀 (종가, TC 없음) | 크로스섹션 | "
                         f"{cs['sharpe_no_tcost']:.4f} | {cs['mdd_no_tc']*100:.1f}% | "
                         f"SPX {cs['n_stocks']} 종목 |")
            lines.append(f"| 횡단면 평균회귀 (종가, TC 포함) | 크로스섹션 | "
                         f"{cs['sharpe_with_tcost']:.4f} | {cs['mdd_tc']*100:.1f}% | "
                         f"편도 {cs['tcost_bps']:.0f}bps |")
        if op:
            lines.append(f"| 횡단면 평균회귀 (시가, TC 없음) | 크로스섹션 | "
                         f"{op['sharpe_no_tcost']:.4f} | {op['mdd_no_tc']*100:.1f}% | "
                         f"시가 기반 |")
            lines.append(f"| 횡단면 평균회귀 (시가, TC 포함) | 크로스섹션 | "
                         f"{op['sharpe_with_tcost']:.4f} | {op['mdd_tc']*100:.1f}% | "
                         f"편도 {op['tcost_bps']:.0f}bps |")
        lines.append("")

        # ─── 9. 결론 ───
        lines.append("---")
        lines.append("")
        lines.append("## 9. 결론 및 권고사항")
        lines.append("")
        lines.append("### 핵심 발견")
        lines.append("")
        lines.append("1. **샤프 비율의 중요성:** 단순 수익률이 아닌 위험 조정 성과(샤프 비율) 로 "
                     "전략을 평가해야 한다.")
        lines.append("   롱온리와 시장중립 전략의 절대 수익률은 크게 다르지만, "
                     "샤프 비율로 비교하면 보다 공정한 평가가 가능하다.")
        lines.append("")
        lines.append("2. **최대 낙폭의 실무적 의미:** 샤프 비율이 양호하더라도 "
                     "MDD가 크면 실전 운용이 어렵다.")
        lines.append("   투자자의 심리적 한계와 마진콜 리스크를 고려하면, "
                     "MDD는 20%를 넘지 않도록 관리하는 것이 바람직하다.")
        lines.append("")
        lines.append("3. **거래비용의 파괴적 영향:** 횡단면 평균회귀 전략에서 "
                     "편도 5bps의 거래비용만으로 샤프 비율이 크게 하락한다.")
        lines.append("   고빈도/고회전 전략일수록 거래비용 모델링이 백테스트의 신뢰성을 좌우한다.")
        lines.append("")
        lines.append("4. **학습/테스트 분할의 필요성:** GLD-GDX 페어 트레이딩에서 "
                     "학습 기간과 테스트 기간의 샤프 비율 차이가 과적합의 정도를 나타낸다.")
        lines.append("")
        lines.append("5. **시가 vs 종가:** 동일 전략이라도 어떤 가격을 사용하느냐에 따라 "
                     "성과가 달라진다. 시가 기반 전략은 실행 가능성이 높으나, "
                     "데이터 잡음에 더 민감할 수 있다.")
        lines.append("")
        lines.append("### 권고사항")
        lines.append("")
        lines.append("- 백테스트 결과를 맹신하지 말고, 반드시 **표본 외(out-of-sample) 검증** 을 수행하라.")
        lines.append("- 거래비용, 슬리피지, 시장 충격 등 **실행 비용** 을 현실적으로 모델링하라.")
        lines.append("- 생존자 편향, 선행편향 등 **데이터 편향** 을 사전에 점검하라.")
        lines.append("- 샤프 비율, MDD, MAR 비율 등 **복수의 성과 지표** 로 전략을 종합 평가하라.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*이 리포트는 `run_chapter3_analysis.py`에 의해 자동 생성되었습니다.*")
        lines.append("")

        # 파일 저장
        report_path = REPORT_DIR / "chapter3_report.md"
        report_content = "\n".join(lines)
        report_path.write_text(report_content, encoding='utf-8')
        print(f"\n  ✅ 리포트 저장: {report_path}")
        print(f"  📊 생성된 차트: {len(self.figures)}개")
        for fname, caption in self.figures:
            print(f"      - {fname}: {caption}")
        print()

    # ============================================================
    # 실행
    # ============================================================
    def run(self):
        """전체 분석 파이프라인 실행"""
        start_time = datetime.now()
        print()
        print("╔" + "═" * 68 + "╗")
        print("║  Chapter 3: 백테스팅 - 종합 분석 리포트 생성기".ljust(67) + " ║")
        print("║  Ernest Chan, Quantitative Trading (2nd Ed., 2021)".ljust(67) + "  ║")
        print("╚" + "═" * 68 + "╝")
        print()

        # 단계별 실행
        self.load_data()
        self.analyze_sharpe_ratio()       # 예제 3.4
        self.analyze_max_drawdown()       # 예제 3.5
        self.analyze_pair_trading()       # 예제 3.6
        self.analyze_cross_sectional_mean_reversion()  # 예제 3.7
        self.analyze_open_price_strategy()             # 예제 3.8
        self.generate_report()

        elapsed = (datetime.now() - start_time).total_seconds()
        print("=" * 70)
        print(f"🏁 전체 분석 완료! (소요 시간: {elapsed:.1f}초)")
        print("=" * 70)


if __name__ == "__main__":
    Chapter3Analyzer().run()
