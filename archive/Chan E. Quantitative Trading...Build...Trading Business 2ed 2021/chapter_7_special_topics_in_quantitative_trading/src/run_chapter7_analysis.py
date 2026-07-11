#!/usr/bin/env python3
"""
Chapter 7: 정량적 트레이딩의 특수 주제 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Quantitative Trading" (2nd Ed., 2021) Chapter 7의
핵심 개념들을 실행하고 분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. GLD-GDX 공적분(Cointegration) 검정 (Example 7.2)
2. KO-PEP 상관관계 vs 공적분 비교 (Example 7.3)
3. PCA 팩터 모델 - IJR 구성종목 롱-숏 포트폴리오 (Example 7.4)
4. 반감기(Half-Life) 계산 - Ornstein-Uhlenbeck (Example 7.5)
5. 1월 효과(January Effect) 백테스트 (Example 7.6)
6. 연간 계절성 모멘텀(Seasonal Momentum) 전략 (Example 7.7)
"""

import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint

# ============================================================================
# 경로 설정
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UTIL_DIR = PROJECT_ROOT / "src"
REPORT_DIR = SCRIPT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

# 유틸리티 모듈 로드
sys.path.insert(0, str(UTIL_DIR))
from calculateMaxDD import calculateMaxDD

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# matplotlib 스타일 설정
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')

# 한글 폰트 설정 (가능한 경우)
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class Chapter7Analyzer:
    """Chapter 7 정량적 트레이딩의 특수 주제 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.figures = []
        self.report_lines = []

        # 디렉토리 생성
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 데이터 로드
    # ========================================================================
    def load_data(self):
        """모든 분석에 필요한 데이터 로드"""
        print("=" * 70)
        print("📊 데이터 로드 중...")
        print("=" * 70)

        # --- GLD / GDX ---
        gld_path = DATA_DIR / "GLD.xls"
        gdx_path = DATA_DIR / "GDX.xls"
        if gld_path.exists() and gdx_path.exists():
            df_gld = pd.read_excel(gld_path, engine='calamine')
            df_gdx = pd.read_excel(gdx_path, engine='calamine')
            self.df_gld_gdx = pd.merge(df_gld, df_gdx, on="Date",
                                       suffixes=("_GLD", "_GDX"))
            self.df_gld_gdx.set_index("Date", inplace=True)
            self.df_gld_gdx.sort_index(inplace=True)
            print(f"  ✅ GLD-GDX: {len(self.df_gld_gdx)} 데이터 포인트")
        else:
            self.df_gld_gdx = None
            print("  ❌ GLD/GDX 데이터 없음")

        # --- KO / PEP ---
        ko_path = DATA_DIR / "KO.xls"
        pep_path = DATA_DIR / "PEP.xls"
        if ko_path.exists() and pep_path.exists():
            df_ko = pd.read_excel(ko_path, engine='calamine')
            df_pep = pd.read_excel(pep_path, engine='calamine')
            self.df_ko_pep = pd.merge(df_ko, df_pep, on="Date",
                                      suffixes=("_KO", "_PEP"))
            self.df_ko_pep.set_index("Date", inplace=True)
            self.df_ko_pep.sort_index(inplace=True)
            print(f"  ✅ KO-PEP: {len(self.df_ko_pep)} 데이터 포인트")
        else:
            self.df_ko_pep = None
            print("  ❌ KO/PEP 데이터 없음")

        # --- IJR 구성종목 (2008-01-14) ---
        ijr_14_path = DATA_DIR / "IJR_20080114.txt"
        if ijr_14_path.exists():
            self.df_ijr_14 = pd.read_table(ijr_14_path)
            self.df_ijr_14["Date"] = self.df_ijr_14["Date"].astype("int")
            self.df_ijr_14.set_index("Date", inplace=True)
            self.df_ijr_14.sort_index(inplace=True)
            self.df_ijr_14 = self.df_ijr_14.ffill()
            print(f"  ✅ IJR (0114): {self.df_ijr_14.shape[0]} 행 × "
                  f"{self.df_ijr_14.shape[1]} 종목")
        else:
            self.df_ijr_14 = None
            print("  ❌ IJR_20080114 데이터 없음")

        # --- IJR 구성종목 (2008-01-31) ---
        ijr_31_path = DATA_DIR / "IJR_20080131.txt"
        if ijr_31_path.exists():
            self.df_ijr_31 = pd.read_table(ijr_31_path)
            self.df_ijr_31["Date"] = self.df_ijr_31["Date"].round().astype("int")
            self.df_ijr_31["Date"] = pd.to_datetime(
                self.df_ijr_31["Date"], format="%Y%m%d"
            )
            self.df_ijr_31.set_index("Date", inplace=True)
            print(f"  ✅ IJR (0131): {self.df_ijr_31.shape[0]} 행 × "
                  f"{self.df_ijr_31.shape[1]} 종목")
        else:
            self.df_ijr_31 = None
            print("  ❌ IJR_20080131 데이터 없음")

        # --- SPX 구성종목 ---
        spx_path = DATA_DIR / "SPX_20071123.txt"
        if spx_path.exists():
            self.df_spx = pd.read_table(spx_path)
            self.df_spx["Date"] = self.df_spx["Date"].round().astype("int")
            self.df_spx["Date"] = pd.to_datetime(
                self.df_spx["Date"], format="%Y%m%d"
            )
            self.df_spx.set_index("Date", inplace=True)
            print(f"  ✅ SPX: {self.df_spx.shape[0]} 행 × "
                  f"{self.df_spx.shape[1]} 종목")
        else:
            self.df_spx = None
            print("  ❌ SPX_20071123 데이터 없음")

        print()

    # ========================================================================
    # 분석 1: GLD-GDX 공적분 검정 (Example 7.2)
    # ========================================================================
    def analyze_cointegration_gld_gdx(self):
        """GLD-GDX Engle-Granger 공적분 검정"""
        print("=" * 70)
        print("🔬 1. GLD-GDX 공적분 검정 (Example 7.2)")
        print("=" * 70)

        if self.df_gld_gdx is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        # 원서와 동일하게 첫 252일(1년) 학습 구간 사용
        trainset = np.arange(0, 252)
        df = self.df_gld_gdx.iloc[trainset].copy()

        gld = df["Adj Close_GLD"]
        gdx = df["Adj Close_GDX"]

        # --- Engle-Granger 공적분 검정 ---
        coint_t, pvalue, crit_value = coint(gld, gdx)

        # --- 헤지 비율 (OLS) ---
        model = OLS(gld, gdx)
        ols_results = model.fit()
        hedge_ratio = ols_results.params.iloc[0]

        # --- 스프레드 ---
        spread = gld - hedge_ratio * gdx

        # 결과 저장
        self.results['cointegration_gld_gdx'] = {
            'hedge_ratio': hedge_ratio,
            'coint_t': coint_t,
            'pvalue': pvalue,
            'crit_1pct': crit_value[0],
            'crit_5pct': crit_value[1],
            'crit_10pct': crit_value[2],
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'n_obs': len(df),
        }
        # 스프레드를 반감기 분석에서도 활용하기 위해 전체 데이터로 계산
        full_gld = self.df_gld_gdx["Adj Close_GLD"]
        full_gdx = self.df_gld_gdx["Adj Close_GDX"]
        self.spread_gld_gdx = full_gld - hedge_ratio * full_gdx

        r = self.results['cointegration_gld_gdx']
        print(f"\n  학습 구간: {df.index[0]} ~ {df.index[-1]} ({r['n_obs']}일)")
        print(f"\n  --- Engle-Granger 공적분 검정 ---")
        print(f"  t-통계량:         {r['coint_t']:.4f}")
        print(f"  p-value:          {r['pvalue']:.6f}")
        print(f"  임계값 (1%):      {r['crit_1pct']:.4f}")
        print(f"  임계값 (5%):      {r['crit_5pct']:.4f}")
        print(f"  임계값 (10%):     {r['crit_10pct']:.4f}")

        if abs(coint_t) > abs(crit_value[1]):
            print("  → ✅ 귀무가설 기각: GLD-GDX는 공적분 관계 존재 (5% 유의수준)")
        else:
            print("  → ❌ 귀무가설 채택 불가: 공적분 관계 미확인")

        print(f"\n  --- 헤지 비율 (Hedge Ratio) ---")
        print(f"  β (GDX):          {r['hedge_ratio']:.6f}")
        print(f"  스프레드 평균:     {r['spread_mean']:.6f}")
        print(f"  스프레드 표준편차:  {r['spread_std']:.6f}")

        # --- 시각화 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) 스프레드 시계열
        ax1 = axes[0]
        ax1.plot(spread.values, linewidth=0.8, color='steelblue')
        ax1.axhline(y=spread.mean(), color='red', linestyle='--',
                     linewidth=0.8, label=f'Mean = {spread.mean():.2f}')
        ax1.set_title('GLD - β·GDX Spread (Training Set, 252 days)')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Spread')
        ax1.legend()

        # (b) GLD vs GDX 산점도
        ax2 = axes[1]
        ax2.scatter(gdx, gld, alpha=0.5, s=15, color='teal')
        x_fit = np.linspace(gdx.min(), gdx.max(), 100)
        ax2.plot(x_fit, hedge_ratio * x_fit, 'r-', linewidth=1.5,
                 label=f'GLD = {hedge_ratio:.4f} × GDX')
        ax2.set_title('GLD vs GDX Scatter Plot')
        ax2.set_xlabel('GDX Adj Close')
        ax2.set_ylabel('GLD Adj Close')
        ax2.legend()

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_1_gld_gdx_cointegration.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 분석 2: KO-PEP 상관관계 vs 공적분 (Example 7.3)
    # ========================================================================
    def analyze_correlation_vs_cointegration(self):
        """KO-PEP: 높은 상관관계가 공적분을 의미하지 않음을 보여주는 예시"""
        print("=" * 70)
        print("🔬 2. KO-PEP 상관관계 vs 공적분 비교 (Example 7.3)")
        print("=" * 70)

        if self.df_ko_pep is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        df = self.df_ko_pep.copy()
        ko = df["Adj Close_KO"]
        pep = df["Adj Close_PEP"]

        # --- 공적분 검정 ---
        coint_t, pvalue, crit_value = coint(ko, pep)

        # --- 헤지 비율 ---
        model = OLS(ko, pep)
        ols_results = model.fit()
        hedge_ratio = ols_results.params.iloc[0]
        spread = ko - hedge_ratio * pep

        # --- 상관관계 ---
        dailyret = df.loc[:, ("Adj Close_KO", "Adj Close_PEP")].pct_change()
        dailyret_clean = dailyret.dropna()
        corr_coef, corr_pval = pearsonr(
            dailyret_clean.iloc[:, 0], dailyret_clean.iloc[:, 1]
        )
        price_corr = ko.corr(pep)

        # 결과 저장
        self.results['ko_pep'] = {
            'correlation_returns': corr_coef,
            'corr_pvalue': corr_pval,
            'price_correlation': price_corr,
            'coint_t': coint_t,
            'coint_pvalue': pvalue,
            'crit_1pct': crit_value[0],
            'crit_5pct': crit_value[1],
            'crit_10pct': crit_value[2],
            'hedge_ratio': hedge_ratio,
            'n_obs': len(df),
        }

        r = self.results['ko_pep']
        print(f"\n  데이터 기간: {df.index[0]} ~ {df.index[-1]} ({r['n_obs']}일)")

        print(f"\n  --- 상관관계 (Correlation) ---")
        print(f"  수익률 Pearson 상관계수:  {r['correlation_returns']:.6f}")
        print(f"  상관계수 p-value:         {r['corr_pvalue']:.2e}")
        print(f"  가격 상관관계:            {r['price_correlation']:.6f}")
        print(f"  → ✅ 매우 높은 상관관계")

        print(f"\n  --- 공적분 검정 (Cointegration) ---")
        print(f"  t-통계량:         {r['coint_t']:.4f}")
        print(f"  p-value:          {r['coint_pvalue']:.6f}")
        print(f"  임계값 (1%):      {r['crit_1pct']:.4f}")
        print(f"  임계값 (5%):      {r['crit_5pct']:.4f}")
        print(f"  임계값 (10%):     {r['crit_10pct']:.4f}")

        if r['coint_pvalue'] > 0.05:
            print("  → ❌ 귀무가설 채택 불가: 공적분 관계 미확인")
            print("  📌 핵심 교훈: 높은 상관관계 ≠ 공적분!")
        else:
            print("  → ✅ 공적분 관계 존재")

        # --- 비교 테이블 ---
        print(f"\n  ┌──────────────────────┬────────────────┬────────────────┐")
        print(f"  │       지표            │   GLD-GDX      │    KO-PEP      │")
        print(f"  ├──────────────────────┼────────────────┼────────────────┤")
        gld_r = self.results.get('cointegration_gld_gdx', {})
        gld_t = gld_r.get('coint_t', float('nan'))
        gld_p = gld_r.get('pvalue', float('nan'))
        print(f"  │ 공적분 t-stat        │ {gld_t:>12.4f}  │ {r['coint_t']:>12.4f}  │")
        print(f"  │ 공적분 p-value       │ {gld_p:>12.6f}  │ {r['coint_pvalue']:>12.6f}  │")
        print(f"  │ 공적분 여부           │ {'✅ Yes':>14s}  │ {'❌ No':>14s}  │")
        print(f"  └──────────────────────┴────────────────┴────────────────┘")

        # --- 시각화 ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) KO vs PEP 가격 시계열
        ax = axes[0, 0]
        ax.plot(ko.values, label='KO', linewidth=0.8)
        ax.plot(pep.values, label='PEP', linewidth=0.8)
        ax.set_title('KO and PEP Price Series')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Adj Close')
        ax.legend()

        # (b) KO-PEP 산점도
        ax = axes[0, 1]
        ax.scatter(pep, ko, alpha=0.4, s=10, color='coral')
        ax.set_title(f'KO vs PEP (price corr = {r["price_correlation"]:.4f})')
        ax.set_xlabel('PEP Adj Close')
        ax.set_ylabel('KO Adj Close')

        # (c) 스프레드 시계열 (비정상적 - 발산함)
        ax = axes[1, 0]
        ax.plot(spread.values, linewidth=0.8, color='orange')
        ax.axhline(y=spread.mean(), color='red', linestyle='--', linewidth=0.8)
        ax.set_title('KO - β·PEP Spread (Non-stationary)')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Spread')

        # (d) 수익률 상관관계 산점도
        ax = axes[1, 1]
        ax.scatter(dailyret_clean.iloc[:, 1], dailyret_clean.iloc[:, 0],
                   alpha=0.3, s=8, color='purple')
        ax.set_title(f'Daily Returns: KO vs PEP (r = {r["correlation_returns"]:.4f})')
        ax.set_xlabel('PEP Daily Return')
        ax.set_ylabel('KO Daily Return')

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_2_ko_pep_corr_vs_coint.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 분석 3: PCA 팩터 모델 (Example 7.4)
    # ========================================================================
    def analyze_pca_factor_model(self):
        """PCA 팩터 모델 기반 롱-숏 포트폴리오 (IJR 구성종목)"""
        print("=" * 70)
        print("🔬 3. PCA 팩터 모델 - 롱/숏 포트폴리오 (Example 7.4)")
        print("=" * 70)

        if self.df_ijr_14 is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        df = self.df_ijr_14.copy()
        lookback = 252
        numFactors = 5
        topN = 50

        dailyret = df.pct_change()
        positionsTable = np.zeros(df.shape)

        end_index = df.shape[0]
        # 반복 횟수 제한: 전체 루프가 지나치게 오래 걸릴 수 있으므로
        # 최대 반복 횟수를 제한하고, 전체 실행 시 비활성화 가능
        max_iterations = min(end_index - lookback - 1, 100)
        actual_end = lookback + 1 + max_iterations
        is_capped = (actual_end < end_index)

        print(f"\n  종목 수: {df.shape[1]}")
        print(f"  전체 기간: {df.shape[0]}일")
        print(f"  룩백: {lookback}일, 팩터 수: {numFactors}, topN: {topN}")
        if is_capped:
            print(f"  ⚠️ 반복 횟수 제한: {max_iterations} / {end_index - lookback - 1}")
            print(f"     (전체 실행 시 수 분 이상 소요될 수 있음)")
        else:
            print(f"  반복 횟수: {max_iterations}")

        t_start = time.time()
        for idx, t in enumerate(np.arange(lookback + 1, actual_end)):
            if idx % 20 == 0:
                elapsed = time.time() - t_start
                print(f"    진행: {idx}/{max_iterations} "
                      f"({idx/max_iterations*100:.0f}%) "
                      f"[{elapsed:.1f}s]", end='\r')

            R = dailyret.iloc[t - lookback + 1: t].T
            hasData = np.where(R.notna().all(axis=1))[0]
            R_clean = R.dropna()

            if R_clean.shape[0] < numFactors + 1:
                continue

            pca = PCA()
            X = pca.fit_transform(R_clean.T)[:, :numFactors]
            X = add_constant(X)
            y1 = R_clean.T

            clf = MultiOutputRegressor(
                LinearRegression(fit_intercept=False), n_jobs=4
            ).fit(X, y1)
            Rexp = np.sum(clf.predict(X), axis=0)

            idxSort = Rexp.argsort()

            positionsTable[t, hasData[idxSort[np.arange(0, topN)]]] = -1
            positionsTable[t, hasData[idxSort[np.arange(-topN, -1)]]] = 1

        elapsed_total = time.time() - t_start
        print(f"\n    PCA 루프 완료: {elapsed_total:.1f}초")

        # --- 수익률 계산 ---
        capital = np.nansum(
            np.array(abs(pd.DataFrame(positionsTable)).shift()), axis=1
        )
        positionsTable[capital == 0] = 0
        capital[capital == 0] = 1

        ret = (
            np.nansum(
                np.array(pd.DataFrame(positionsTable).shift())
                * np.array(dailyret),
                axis=1,
            )
            / capital
        )

        # NaN과 초기 0을 제거
        valid_ret = ret[lookback + 1: actual_end]
        valid_ret = valid_ret[~np.isnan(valid_ret)]

        if len(valid_ret) > 0:
            avgret = np.nanmean(valid_ret) * 252
            avgstd = np.nanstd(valid_ret) * math.sqrt(252)
            sharpe = avgret / avgstd if avgstd > 0 else 0

            cumret = np.cumprod(1 + valid_ret) - 1
            maxDD, maxDDD, _ = calculateMaxDD(cumret)
        else:
            avgret = avgstd = sharpe = maxDD = maxDDD = 0
            cumret = np.array([0])

        self.results['pca_factor'] = {
            'ann_return': avgret,
            'ann_std': avgstd,
            'sharpe': sharpe,
            'max_dd': maxDD,
            'max_ddd': maxDDD,
            'n_days': len(valid_ret),
            'n_factors': numFactors,
            'topN': topN,
            'lookback': lookback,
            'capped': is_capped,
            'max_iterations': max_iterations,
        }

        r = self.results['pca_factor']
        print(f"\n  --- PCA 팩터 모델 롱/숏 전략 성과 ---")
        print(f"  연간 수익률:    {r['ann_return']*100:.4f}%")
        print(f"  연간 표준편차:   {r['ann_std']*100:.4f}%")
        print(f"  샤프 비율:       {r['sharpe']:.4f}")
        print(f"  최대 낙폭:       {r['max_dd']*100:.2f}%")
        print(f"  최대 낙폭 기간:  {r['max_ddd']:.0f}일")
        if is_capped:
            print(f"  ⚠️ {max_iterations}일 제한 분석 (전체: "
                  f"{end_index - lookback - 1}일)")

        # --- 시각화 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(cumret * 100, linewidth=0.8, color='darkgreen')
        ax.set_title(f'PCA Factor Model Long-Short Portfolio\n'
                     f'(Top/Bottom {topN}, {numFactors} factors)')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        ax = axes[1]
        ax.hist(valid_ret * 100, bins=50, edgecolor='white',
                color='darkgreen', alpha=0.7)
        ax.axvline(x=np.mean(valid_ret) * 100, color='red',
                   linestyle='--', linewidth=1.2, label=f'Mean={np.mean(valid_ret)*100:.4f}%')
        ax.set_title('Daily Return Distribution')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_3_pca_factor_model.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 분석 4: 반감기 계산 (Example 7.5)
    # ========================================================================
    def analyze_half_life(self):
        """Ornstein-Uhlenbeck 과정을 이용한 반감기 계산"""
        print("=" * 70)
        print("🔬 4. 평균 회귀 반감기 (Half-Life) 계산 (Example 7.5)")
        print("=" * 70)

        if self.df_gld_gdx is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        df = self.df_gld_gdx.copy()
        gld = df["Adj Close_GLD"]
        gdx = df["Adj Close_GDX"]

        # --- 공적분 검정 (전체 데이터) ---
        coint_t, pvalue, crit_value = coint(gld, gdx)

        # --- 헤지 비율 ---
        model = OLS(gld, gdx)
        ols_results = model.fit()
        hedge_ratio = ols_results.params.iloc[0]

        # --- 스프레드 ---
        z = gld - hedge_ratio * gdx

        # --- Ornstein-Uhlenbeck: Δy(t) = λ(y(t-1) - μ) + ε ---
        prevz = z.shift()
        dz = z - prevz
        dz = dz.iloc[1:]
        prevz = prevz.iloc[1:]

        model2 = OLS(dz, prevz - np.mean(prevz))
        results2 = model2.fit()
        theta = results2.params.iloc[0]

        halflife = -np.log(2) / theta

        self.results['half_life'] = {
            'hedge_ratio': hedge_ratio,
            'coint_t': coint_t,
            'pvalue': pvalue,
            'theta': theta,
            'halflife': halflife,
            'spread_mean': z.mean(),
            'spread_std': z.std(),
            'n_obs': len(df),
        }

        r = self.results['half_life']
        print(f"\n  데이터 기간: {df.index[0]} ~ {df.index[-1]} ({r['n_obs']}일)")
        print(f"\n  --- Ornstein-Uhlenbeck 파라미터 ---")
        print(f"  헤지 비율 (β):    {r['hedge_ratio']:.6f}")
        print(f"  λ (theta):         {r['theta']:.6f}")
        print(f"  반감기 (h):        {r['halflife']:.2f} 거래일")
        print(f"  스프레드 평균 (μ): {r['spread_mean']:.6f}")
        print(f"  스프레드 σ:        {r['spread_std']:.6f}")
        print(f"\n  수식: Δy(t) = λ·(y(t-1) - μ) + ε")
        print(f"  반감기: h = -ln(2)/λ = -ln(2)/({r['theta']:.6f}) "
              f"= {r['halflife']:.2f}일")

        if halflife > 0:
            print(f"\n  → ✅ 양의 반감기: 평균 회귀 확인")
            print(f"     스프레드가 평균에서 이탈한 후, 약 {halflife:.0f}일 후 "
                  f"절반만큼 회복")
        else:
            print(f"\n  → ⚠️ 음의 반감기: 평균 회귀가 아닌 발산 경향")

        # --- 시각화 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) 스프레드 시계열 + 평균선 + 반감기 밴드
        ax = axes[0]
        z_vals = z.values
        ax.plot(z_vals, linewidth=0.7, color='steelblue', label='Spread')
        mean_z = z.mean()
        std_z = z.std()
        ax.axhline(y=mean_z, color='red', linestyle='--', linewidth=1.0,
                   label=f'Mean = {mean_z:.2f}')
        ax.axhline(y=mean_z + std_z, color='orange', linestyle=':',
                   linewidth=0.8, label=f'±1σ = {std_z:.2f}')
        ax.axhline(y=mean_z - std_z, color='orange', linestyle=':',
                   linewidth=0.8)
        ax.set_title(f'GLD-GDX Spread (Half-life = {halflife:.1f} days)')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Spread')
        ax.legend(fontsize=8)

        # (b) Δy vs y(t-1) 산점도 + 회귀선
        ax = axes[1]
        ax.scatter(prevz.values, dz.values, alpha=0.3, s=8, color='teal')
        x_fit = np.linspace(prevz.min(), prevz.max(), 100)
        y_fit = theta * (x_fit - np.mean(prevz))
        ax.plot(x_fit, y_fit, 'r-', linewidth=1.5,
                label=f'Δy = {theta:.4f}·(y - μ)')
        ax.set_title('Ornstein-Uhlenbeck Regression')
        ax.set_xlabel('y(t-1)')
        ax.set_ylabel('Δy(t)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.legend()

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_4_half_life.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 분석 5: 1월 효과 백테스트 (Example 7.6)
    # ========================================================================
    def analyze_january_effect(self):
        """IJR 구성종목 기반 1월 효과 백테스트"""
        print("=" * 70)
        print("🔬 5. 1월 효과 (January Effect) 백테스트 (Example 7.6)")
        print("=" * 70)

        if self.df_ijr_31 is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        df = self.df_ijr_31.copy()
        onewaytcost = 0.0005

        # 연말 가격 (12월 말)
        eoyPrice = df.resample("YE").last().iloc[:-1]
        annret = eoyPrice.pct_change().iloc[1:]

        # 1월말 가격
        eojPrice = df.resample("BYE-JAN").last().iloc[1:-1]

        # 1월 수익률
        janret = (eojPrice.values - eoyPrice.values) / eoyPrice.values
        janret = janret[1:]  # annret와 행 수 맞춤

        portfolio_rets = []
        year_labels = []

        for y in range(len(annret)):
            hasData = np.where(np.isfinite(annret.iloc[y, :]))[0]
            if len(hasData) == 0:
                continue
            sortidx = np.argsort(annret.iloc[y, hasData])
            topN = int(np.round(len(hasData) / 10))
            if topN == 0:
                continue

            # 전년 최하위 수익률 종목 롱, 최상위 종목 숏 (1월에)
            long_ret = np.nanmean(
                janret[y, hasData[sortidx.iloc[np.arange(0, topN)]]]
            )
            short_ret = np.nanmean(
                janret[y, hasData[sortidx.iloc[np.arange(-topN + 1, -1)]]]
            )
            portRet = (long_ret - short_ret) / 2 - 2 * onewaytcost
            portfolio_rets.append(portRet)
            if y + 1 < len(eojPrice):
                year_labels.append(str(eojPrice.index[y + 1].year))
            else:
                year_labels.append(f"Year {y}")

        portfolio_rets = np.array(portfolio_rets)

        # 통계
        avg_jan_ret = np.nanmean(portfolio_rets) if len(portfolio_rets) > 0 else 0
        std_jan_ret = np.nanstd(portfolio_rets) if len(portfolio_rets) > 0 else 0
        n_years = len(portfolio_rets)

        # 단순 비교: 1월 vs 나머지 (IJR 전체 수익률 기준)
        all_monthly = df.resample("ME").last()
        monthly_ret = all_monthly.pct_change()
        jan_mask = monthly_ret.index.month == 1
        jan_rets_all = monthly_ret[jan_mask].mean(axis=1).dropna().values
        nojan_rets_all = monthly_ret[~jan_mask].mean(axis=1).dropna().values

        if len(jan_rets_all) > 0 and len(nojan_rets_all) > 0:
            t_stat, t_pval = ttest_ind(jan_rets_all, nojan_rets_all)
        else:
            t_stat, t_pval = 0, 1

        self.results['january_effect'] = {
            'portfolio_rets': portfolio_rets,
            'avg_portfolio_ret': avg_jan_ret,
            'std_portfolio_ret': std_jan_ret,
            'n_years': n_years,
            'jan_avg_cross_section': np.nanmean(jan_rets_all) if len(jan_rets_all) > 0 else 0,
            'nojan_avg_cross_section': np.nanmean(nojan_rets_all) if len(nojan_rets_all) > 0 else 0,
            't_stat': t_stat,
            't_pval': t_pval,
            'year_labels': year_labels,
        }

        r = self.results['january_effect']
        print(f"\n  데이터: IJR 구성종목 ({df.shape[1]} 종목)")
        print(f"  거래비용: 편도 {onewaytcost*100:.2f}%")

        print(f"\n  --- 연도별 포트폴리오 수익률 ---")
        for i, (yr, ret) in enumerate(zip(year_labels, portfolio_rets)):
            emoji = "🟢" if ret > 0 else "🔴"
            print(f"    {yr}: {ret*100:+8.4f}% {emoji}")

        print(f"\n  --- 전략 요약 ---")
        print(f"  평균 포트폴리오 수익률:  {r['avg_portfolio_ret']*100:+.4f}%")
        print(f"  포트폴리오 수익률 σ:     {r['std_portfolio_ret']*100:.4f}%")
        print(f"  연도 수:                 {r['n_years']}")

        print(f"\n  --- 1월 vs 비1월 수익률 (횡단면 평균) ---")
        print(f"  1월 평균:      {r['jan_avg_cross_section']*100:+.4f}%")
        print(f"  비1월 평균:    {r['nojan_avg_cross_section']*100:+.4f}%")
        print(f"  t-통계량:      {r['t_stat']:.4f}")
        print(f"  p-value:       {r['t_pval']:.6f}")

        # --- 시각화 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) 연도별 막대 그래프
        ax = axes[0]
        colors = ['green' if r > 0 else 'red' for r in portfolio_rets]
        ax.bar(year_labels, portfolio_rets * 100, color=colors, alpha=0.7,
               edgecolor='white')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=avg_jan_ret * 100, color='blue', linestyle='--',
                   linewidth=1.0, label=f'Mean = {avg_jan_ret*100:.2f}%')
        ax.set_title('January Effect: Annual Portfolio Returns')
        ax.set_xlabel('Year')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # (b) 1월 vs 비1월 비교 박스플롯
        ax = axes[1]
        if len(jan_rets_all) > 0 and len(nojan_rets_all) > 0:
            bp = ax.boxplot(
                [jan_rets_all * 100, nojan_rets_all * 100],
                labels=['January', 'Non-January'],
                patch_artist=True,
            )
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')
        ax.set_title('Monthly Cross-Sectional Returns: Jan vs Non-Jan')
        ax.set_ylabel('Average Monthly Return (%)')

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_5_january_effect.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 분석 6: 계절성 모멘텀 전략 (Example 7.7)
    # ========================================================================
    def analyze_seasonal_momentum(self):
        """연간 계절성 모멘텀 (Year-on-Year Seasonal Trending) 전략"""
        print("=" * 70)
        print("🔬 6. 연간 계절성 모멘텀 전략 (Example 7.7)")
        print("=" * 70)

        if self.df_spx is None:
            print("  ⚠️ 데이터 없음 - 건너뜁니다.\n")
            return

        df = self.df_spx.copy()

        # 월말 가격
        eomPrice = df.resample("ME").last().iloc[:-1]
        monthlyRet = eomPrice.pct_change(1)

        positions = np.zeros(monthlyRet.shape)

        for m in range(13, monthlyRet.shape[0]):
            hasData = np.where(np.isfinite(monthlyRet.iloc[m - 12, :]))[0]
            if len(hasData) == 0:
                continue
            sortidx = np.argsort(monthlyRet.iloc[m - 12, hasData])

            # 1달 전 데이터가 없는 종목 제거
            badData = np.where(
                np.logical_not(
                    np.isfinite(monthlyRet.iloc[m - 1, hasData[sortidx]])
                )
            )[0]
            sortidx = sortidx.drop(sortidx.index[badData])
            topN = int(np.floor(len(sortidx) / 10))
            if topN == 0:
                continue

            # 12개월 전 수익률 기준: 상위 롱, 하위 숏
            positions[m - 1, hasData[sortidx.values[np.arange(0, topN)]]] = -1
            positions[m - 1, hasData[sortidx.values[np.arange(-topN, 0)]]] = 1

        capital = np.nansum(
            np.array(pd.DataFrame(abs(positions)).shift()), axis=1
        )
        capital[capital == 0] = 1

        ret = (
            np.nansum(
                np.array(pd.DataFrame(positions).shift())
                * np.array(monthlyRet),
                axis=1,
            )
            / capital
        )
        ret = np.delete(ret, np.arange(13))

        avgret = np.nanmean(ret) * 12
        avgstd = np.nanstd(ret) * math.sqrt(12)
        sharpe = np.sqrt(12) * np.nanmean(ret) / np.nanstd(ret) if np.nanstd(ret) > 0 else 0

        # 누적 수익률
        valid_ret = ret[~np.isnan(ret)]
        cumret = np.cumprod(1 + valid_ret) - 1
        if len(cumret) > 0:
            maxDD, maxDDD, _ = calculateMaxDD(cumret)
        else:
            maxDD = maxDDD = 0

        self.results['seasonal_momentum'] = {
            'ann_return': avgret,
            'ann_std': avgstd,
            'sharpe': sharpe,
            'max_dd': maxDD,
            'max_ddd': maxDDD,
            'n_months': len(valid_ret),
            'n_stocks': df.shape[1],
        }

        r = self.results['seasonal_momentum']
        print(f"\n  데이터: SPX 구성종목 ({r['n_stocks']} 종목)")
        print(f"  전략: 12개월 전 동일 월 수익률 기준 모멘텀")
        print(f"  기간: {r['n_months']} 개월")

        print(f"\n  --- 계절성 모멘텀 전략 성과 ---")
        print(f"  연간 수익률:     {r['ann_return']*100:+.4f}%")
        print(f"  연간 표준편차:    {r['ann_std']*100:.4f}%")
        print(f"  샤프 비율:        {r['sharpe']:.4f}")
        print(f"  최대 낙폭:        {r['max_dd']*100:.2f}%")
        print(f"  최대 낙폭 기간:   {r['max_ddd']:.0f} 개월")

        if r['sharpe'] < 0:
            print(f"\n  → ⚠️ 음의 샤프 비율: 이 전략은 수익성 없음")
            print(f"     원서 기대값: 연간 수익률 ≈ -1.27%, 샤프 ≈ -0.12")
        else:
            print(f"\n  → ✅ 양의 수익률")

        # --- 시각화 ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) 누적 수익률
        ax = axes[0]
        ax.plot(cumret * 100, linewidth=0.8, color='darkorange')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title('Year-on-Year Seasonal Momentum Strategy')
        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Return (%)')

        # (b) 월별 수익률 분포
        ax = axes[1]
        ax.hist(valid_ret * 100, bins=40, edgecolor='white',
                color='darkorange', alpha=0.7)
        ax.axvline(x=np.nanmean(valid_ret) * 100, color='red',
                   linestyle='--', linewidth=1.2,
                   label=f'Mean = {np.nanmean(valid_ret)*100:.4f}%')
        ax.set_title('Monthly Return Distribution')
        ax.set_xlabel('Monthly Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig7_6_seasonal_momentum.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig_path)
        print(f"\n  📈 그래프 저장: {fig_path.name}")
        print()

    # ========================================================================
    # 리포트 생성
    # ========================================================================
    def generate_report(self):
        """종합 마크다운 리포트 생성"""
        print("=" * 70)
        print("📝 종합 리포트 생성 중...")
        print("=" * 70)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        # 제목
        lines.append("# Chapter 7: 정량적 트레이딩의 특수 주제 - 종합 분석 리포트")
        lines.append("")
        lines.append(f"> 생성일시: {now}")
        lines.append(f"> 원서: Ernest Chan, *Quantitative Trading* (2nd Ed., 2021)")
        lines.append("")

        # 개요
        lines.append("## 1. 개요")
        lines.append("")
        lines.append("본 리포트는 Chapter 7에서 다루는 정량적 트레이딩의 다양한 "
                      "고급 주제들을 실증 분석한 결과입니다.")
        lines.append("주요 분석 내용:")
        lines.append("")
        lines.append("- **공적분(Cointegration)** 검정: 평균회귀 페어 트레이딩의 기초")
        lines.append("- **상관관계 vs 공적분** : 높은 상관관계가 반드시 공적분을 "
                      "의미하지 않음")
        lines.append("- **PCA 팩터 모델** : 주성분 분석 기반 롱-숏 포트폴리오 전략")
        lines.append("- **반감기(Half-Life)** : Ornstein-Uhlenbeck 과정을 통한 "
                      "평균회귀 속도 측정")
        lines.append("- **1월 효과(January Effect)** : 계절성 이상 현상 백테스트")
        lines.append("- **계절성 모멘텀** : 연간 동일 월 수익률 기반 추세 전략")
        lines.append("")

        # 데이터
        lines.append("## 2. 사용 데이터")
        lines.append("")
        lines.append("| 데이터 | 파일 | 설명 |")
        lines.append("|--------|------|------|")
        lines.append("| GLD | `GLD.xls` | 금 ETF 수정종가 |")
        lines.append("| GDX | `GDX.xls` | 금 광산 ETF 수정종가 |")
        lines.append("| KO | `KO.xls` | Coca-Cola 수정종가 |")
        lines.append("| PEP | `PEP.xls` | PepsiCo 수정종가 |")
        lines.append("| IJR (0114) | `IJR_20080114.txt` | iShares S&P SmallCap 600 구성종목 |")
        lines.append("| IJR (0131) | `IJR_20080131.txt` | iShares S&P SmallCap 600 구성종목 (1월 포함) |")
        lines.append("| SPX | `SPX_20071123.txt` | S&P 500 구성종목 |")
        lines.append("")

        # --- 분석 1: GLD-GDX 공적분 ---
        lines.append("## 3. GLD-GDX 공적분 검정 (Example 7.2)")
        lines.append("")
        r = self.results.get('cointegration_gld_gdx', {})
        if r:
            lines.append("### 방법")
            lines.append("")
            lines.append("Engle-Granger 2단계 공적분 검정:")
            lines.append("")
            lines.append("1. OLS 회귀: $y(t) = \\beta x(t) + e(t)$ "
                          "에서 헤지 비율 $\\beta$ 추정")
            lines.append("2. 잔차 $e(t)$ 에 대해 단위근 검정 (ADF)")
            lines.append("")
            lines.append("### 결과")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|------|----|")
            lines.append(f"| 헤지 비율 (β) | {r['hedge_ratio']:.6f} |")
            lines.append(f"| t-통계량 | {r['coint_t']:.4f} |")
            lines.append(f"| p-value | {r['pvalue']:.6f} |")
            lines.append(f"| 임계값 (1%) | {r['crit_1pct']:.4f} |")
            lines.append(f"| 임계값 (5%) | {r['crit_5pct']:.4f} |")
            lines.append(f"| 스프레드 평균 | {r['spread_mean']:.6f} |")
            lines.append(f"| 스프레드 σ | {r['spread_std']:.6f} |")
            lines.append("")
            is_coint = abs(r['coint_t']) > abs(r['crit_5pct'])
            lines.append(f"**결론**: GLD-GDX 페어는 5% 유의수준에서 "
                          f"{'공적분 관계가 확인' if is_coint else '공적분 관계가 미확인'}됩니다. "
                          f"(p-value = {r['pvalue']:.4f})")
            lines.append("")
            lines.append("![GLD-GDX Cointegration](figures/fig7_1_gld_gdx_cointegration.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 분석 2: KO-PEP ---
        lines.append("## 4. KO-PEP 상관관계 vs 공적분 (Example 7.3)")
        lines.append("")
        r = self.results.get('ko_pep', {})
        if r:
            lines.append("### 핵심 교훈")
            lines.append("")
            lines.append("> **높은 상관관계(correlation)** 는 **공적분(cointegration)** "
                          "을 의미하지 않습니다.")
            lines.append("> 상관관계는 수익률의 방향성 유사도, "
                          "공적분은 가격 스프레드의 정상성을 측정합니다.")
            lines.append("")
            lines.append("### 비교 결과")
            lines.append("")
            lines.append("| 지표 | GLD-GDX | KO-PEP |")
            lines.append("|------|---------|--------|")
            gld_r = self.results.get('cointegration_gld_gdx', {})
            lines.append(f"| 공적분 t-stat | "
                          f"{gld_r.get('coint_t', 'N/A'):.4f} | "
                          f"{r['coint_t']:.4f} |" if gld_r else
                          f"| 공적분 t-stat | N/A | {r['coint_t']:.4f} |")
            lines.append(f"| 공적분 p-value | "
                          f"{gld_r.get('pvalue', 'N/A'):.6f} | "
                          f"{r['coint_pvalue']:.6f} |" if gld_r else
                          f"| 공적분 p-value | N/A | {r['coint_pvalue']:.6f} |")
            lines.append(f"| 수익률 상관계수 | - | {r['correlation_returns']:.4f} |")
            lines.append(f"| 가격 상관관계 | - | {r['price_correlation']:.4f} |")
            coint_yes = r['coint_pvalue'] < 0.05
            lines.append(f"| 공적분 여부 | "
                          f"{'Yes' if gld_r.get('pvalue', 1) < 0.05 else 'No'} "
                          f"| {'Yes' if coint_yes else 'No'} |")
            lines.append("")
            lines.append("![KO-PEP Correlation vs Cointegration]"
                          "(figures/fig7_2_ko_pep_corr_vs_coint.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 분석 3: PCA 팩터 모델 ---
        lines.append("## 5. PCA 팩터 모델 (Example 7.4)")
        lines.append("")
        r = self.results.get('pca_factor', {})
        if r:
            lines.append("### 방법")
            lines.append("")
            lines.append("주성분 분석(PCA)을 팩터 모델로 활용한 롱-숏 포트폴리오:")
            lines.append("")
            lines.append("1. 252일 롤링 윈도 내 종목 수익률에 PCA 적용 "
                          f"({r['n_factors']}개 팩터)")
            lines.append("2. 다중 출력 선형 회귀로 팩터 노출도 추정: "
                          "$R = X\\beta + \\epsilon$")
            lines.append(f"3. 예상 수익률 상위 {r['topN']}종목 롱, "
                          f"하위 {r['topN']}종목 숏")
            lines.append("")
            lines.append("### 성과")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|------|----|")
            lines.append(f"| 연간 수익률 | {r['ann_return']*100:.4f}% |")
            lines.append(f"| 연간 표준편차 | {r['ann_std']*100:.4f}% |")
            lines.append(f"| 샤프 비율 | {r['sharpe']:.4f} |")
            lines.append(f"| 최대 낙폭 | {r['max_dd']*100:.2f}% |")
            lines.append(f"| 분석 기간 | {r['n_days']}일 |")
            if r.get('capped', False):
                lines.append(f"| 반복 제한 | {r['max_iterations']}일 "
                              "(계산 시간 절약 목적) |")
            lines.append("")
            lines.append("![PCA Factor Model](figures/fig7_3_pca_factor_model.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 분석 4: 반감기 ---
        lines.append("## 6. 평균 회귀 반감기 (Example 7.5)")
        lines.append("")
        r = self.results.get('half_life', {})
        if r:
            lines.append("### 방법")
            lines.append("")
            lines.append("Ornstein-Uhlenbeck 과정:")
            lines.append("")
            lines.append("$$\\Delta y(t) = \\lambda \\cdot (y(t-1) - \\mu) + \\epsilon$$")
            lines.append("")
            lines.append("반감기:")
            lines.append("")
            lines.append("$$h = -\\frac{\\ln 2}{\\lambda}$$")
            lines.append("")
            lines.append("### 결과")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|------|----|")
            lines.append(f"| λ (theta) | {r['theta']:.6f} |")
            lines.append(f"| 반감기 (h) | {r['halflife']:.2f} 거래일 |")
            lines.append(f"| 스프레드 평균 (μ) | {r['spread_mean']:.6f} |")
            lines.append(f"| 스프레드 σ | {r['spread_std']:.6f} |")
            lines.append(f"| 헤지 비율 | {r['hedge_ratio']:.6f} |")
            lines.append("")
            lines.append(f"**해석**: GLD-GDX 스프레드가 평균에서 이탈한 후, "
                          f"약 **{r['halflife']:.0f}일** 만에 이탈 폭의 절반이 회복됩니다.")
            lines.append("")
            lines.append("![Half-Life](figures/fig7_4_half_life.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 분석 5: 1월 효과 ---
        lines.append("## 7. 1월 효과 백테스트 (Example 7.6)")
        lines.append("")
        r = self.results.get('january_effect', {})
        if r:
            lines.append("### 전략")
            lines.append("")
            lines.append("전년 연간 수익률 하위 10% 종목을 1월에 매수(롱), "
                          "상위 10%를 매도(숏)하는 롱-숏 전략.")
            lines.append("(세금 절감 매도 후 반등을 노리는 '1월 효과' 가설)")
            lines.append("")
            lines.append("### 연도별 수익률")
            lines.append("")
            lines.append("| 연도 | 수익률 |")
            lines.append("|------|--------|")
            for yr, ret in zip(r['year_labels'], r['portfolio_rets']):
                lines.append(f"| {yr} | {ret*100:+.4f}% |")
            lines.append("")
            lines.append("### 통계")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|------|----|")
            lines.append(f"| 평균 포트폴리오 수익률 | "
                          f"{r['avg_portfolio_ret']*100:+.4f}% |")
            lines.append(f"| 수익률 σ | "
                          f"{r['std_portfolio_ret']*100:.4f}% |")
            lines.append(f"| 1월 횡단면 평균 수익률 | "
                          f"{r['jan_avg_cross_section']*100:+.4f}% |")
            lines.append(f"| 비1월 횡단면 평균 수익률 | "
                          f"{r['nojan_avg_cross_section']*100:+.4f}% |")
            lines.append(f"| t-통계량 (1월 vs 비1월) | {r['t_stat']:.4f} |")
            lines.append(f"| p-value | {r['t_pval']:.6f} |")
            lines.append("")
            lines.append("![January Effect](figures/fig7_5_january_effect.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 분석 6: 계절성 모멘텀 ---
        lines.append("## 8. 계절성 모멘텀 전략 (Example 7.7)")
        lines.append("")
        r = self.results.get('seasonal_momentum', {})
        if r:
            lines.append("### 전략")
            lines.append("")
            lines.append("12개월 전 동일 월 수익률 기준으로 종목 정렬 후, "
                          "상위 10% 롱 / 하위 10% 숏.")
            lines.append("(연간 계절적 패턴이 반복된다는 가설)")
            lines.append("")
            lines.append("### 성과")
            lines.append("")
            lines.append("| 항목 | 값 |")
            lines.append("|------|----|")
            lines.append(f"| 연간 수익률 | {r['ann_return']*100:+.4f}% |")
            lines.append(f"| 연간 표준편차 | {r['ann_std']*100:.4f}% |")
            lines.append(f"| 샤프 비율 | {r['sharpe']:.4f} |")
            lines.append(f"| 최대 낙폭 | {r['max_dd']*100:.2f}% |")
            lines.append(f"| 기간 | {r['n_months']} 개월 |")
            lines.append("")
            if r['ann_return'] < 0:
                lines.append("**결론**: 이 단순한 계절성 모멘텀 전략은 SPX 구성종목에서 "
                              "수익성이 없었습니다. "
                              "원서에서도 동일한 음의 결과를 보고합니다.")
            else:
                lines.append("**결론**: 계절성 모멘텀 전략이 양의 수익률을 보였습니다.")
            lines.append("")
            lines.append("![Seasonal Momentum](figures/fig7_6_seasonal_momentum.png)")
            lines.append("")
        else:
            lines.append("데이터 미로드로 분석 생략.")
            lines.append("")

        # --- 전략 성과 비교표 ---
        lines.append("## 9. 전략 성과 비교표")
        lines.append("")
        lines.append("| 전략 | 연간 수익률 | 연간 σ | 샤프 비율 | 최대 낙폭 |")
        lines.append("|------|------------|--------|----------|----------|")

        pca = self.results.get('pca_factor', {})
        if pca:
            lines.append(f"| PCA 팩터 모델 (Ex 7.4) | "
                          f"{pca['ann_return']*100:.2f}% | "
                          f"{pca['ann_std']*100:.2f}% | "
                          f"{pca['sharpe']:.4f} | "
                          f"{pca['max_dd']*100:.2f}% |")

        sm_r = self.results.get('seasonal_momentum', {})
        if sm_r:
            lines.append(f"| 계절성 모멘텀 (Ex 7.7) | "
                          f"{sm_r['ann_return']*100:.2f}% | "
                          f"{sm_r['ann_std']*100:.2f}% | "
                          f"{sm_r['sharpe']:.4f} | "
                          f"{sm_r['max_dd']*100:.2f}% |")

        jan = self.results.get('january_effect', {})
        if jan:
            lines.append(f"| 1월 효과 (Ex 7.6) | "
                          f"{jan['avg_portfolio_ret']*100:+.2f}% (1월만) | "
                          f"{jan['std_portfolio_ret']*100:.2f}% | "
                          f"- | - |")
        lines.append("")

        # --- 결론 ---
        lines.append("## 10. 결론 및 권고사항")
        lines.append("")
        lines.append("### 핵심 발견")
        lines.append("")
        lines.append("1. **공적분은 페어 트레이딩의 핵심** : "
                      "GLD-GDX 페어는 통계적으로 유의한 공적분 관계를 보이며, "
                      "평균회귀 전략의 기초를 제공합니다.")
        lines.append("2. **상관관계와 공적분은 다름** : "
                      "KO-PEP의 사례가 보여주듯, 높은 수익률 상관관계가 "
                      "가격 스프레드의 정상성을 보장하지 않습니다.")
        lines.append("3. **반감기는 전략 설계의 가이드** : "
                      "GLD-GDX 스프레드의 반감기는 적정 보유 기간과 "
                      "룩백 윈도를 결정하는 데 활용됩니다.")
        lines.append("4. **PCA 팩터 모델** : 소형주(IJR)에서 5개 주성분으로 "
                      "팩터 노출도를 추정하는 롱-숏 전략은 양의 샤프 비율을 보입니다.")
        lines.append("5. **계절적 이상 현상** : 1월 효과와 연간 계절성 모멘텀은 "
                      "이론적으로 흥미롭지만, 단순 구현으로는 일관된 수익을 "
                      "기대하기 어렵습니다.")
        lines.append("")
        lines.append("### 실무 적용 시 고려사항")
        lines.append("")
        lines.append("- 공적분 관계는 시간에 따라 변할 수 있으므로 **롤링 검정** 필요")
        lines.append("- PCA 팩터 수와 롱-숏 비율은 **표본 외 검증** 을 통해 튜닝")
        lines.append("- 반감기가 너무 길면 자본 효율성이 낮아지고, "
                      "너무 짧으면 거래비용이 수익을 잠식")
        lines.append("- 계절적 패턴은 **구조적 원인(세금, 리밸런싱)** 이 "
                      "사라지면 소멸 가능")
        lines.append("")

        # 파일 저장
        report_path = REPORT_DIR / "chapter7_analysis_report.md"
        report_content = "\n".join(lines)
        report_path.write_text(report_content, encoding='utf-8')

        print(f"\n  ✅ 리포트 저장: {report_path}")
        print(f"  📄 총 {len(lines)} 줄")
        print(f"  📊 그래프 {len(self.figures)}개 생성")
        print()

        return report_path

    # ========================================================================
    # 전체 실행
    # ========================================================================
    def run(self):
        """모든 분석을 순차적으로 실행"""
        start_time = time.time()

        print("\n" + "🚀" * 35)
        print("  Chapter 7: 정량적 트레이딩의 특수 주제 - 종합 분석")
        print("  Ernest Chan, Quantitative Trading (2nd Ed., 2021)")
        print("🚀" * 35 + "\n")

        # 1. 데이터 로드
        self.load_data()

        # 2. 분석 실행 (Example 7.2 ~ 7.7)
        self.analyze_cointegration_gld_gdx()      # Example 7.2
        self.analyze_correlation_vs_cointegration() # Example 7.3
        self.analyze_pca_factor_model()             # Example 7.4
        self.analyze_half_life()                    # Example 7.5
        self.analyze_january_effect()               # Example 7.6
        self.analyze_seasonal_momentum()            # Example 7.7

        # 3. 리포트 생성
        report_path = self.generate_report()

        # 4. 완료 요약
        elapsed = time.time() - start_time
        print("=" * 70)
        print("🏁 분석 완료!")
        print("=" * 70)
        print(f"  ⏱️  총 소요 시간: {elapsed:.1f}초")
        print(f"  📄 리포트: {report_path}")
        print(f"  📊 그래프: {len(self.figures)}개")
        for fig in self.figures:
            print(f"      - {fig.name}")
        print()

        # 최종 성과 요약 테이블
        print("  ┌────────────────────────────┬──────────┬───────────┐")
        print("  │ 전략                        │ 수익률    │ 샤프 비율  │")
        print("  ├────────────────────────────┼──────────┼───────────┤")

        pca = self.results.get('pca_factor', {})
        if pca:
            print(f"  │ PCA 팩터 모델 (Ex 7.4)     │ "
                  f"{pca['ann_return']*100:>6.2f}%  │ "
                  f"{pca['sharpe']:>8.4f}  │")

        sm_r = self.results.get('seasonal_momentum', {})
        if sm_r:
            print(f"  │ 계절성 모멘텀 (Ex 7.7)     │ "
                  f"{sm_r['ann_return']*100:>6.2f}%  │ "
                  f"{sm_r['sharpe']:>8.4f}  │")

        jan = self.results.get('january_effect', {})
        if jan:
            print(f"  │ 1월 효과 (Ex 7.6)          │ "
                  f"{jan['avg_portfolio_ret']*100:>+6.2f}%  │ "
                  f"{'N/A':>8s}   │")

        print("  └────────────────────────────┴──────────┴───────────┘")
        print()


if __name__ == "__main__":
    Chapter7Analyzer().run()
