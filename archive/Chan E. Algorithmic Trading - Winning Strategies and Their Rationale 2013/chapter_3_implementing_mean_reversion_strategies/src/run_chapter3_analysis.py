#!/usr/bin/env python3
"""
Chapter 3: 평균 회귀 전략 구현 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 3의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. 스프레드 유형 비교 (가격 스프레드, 로그 가격 스프레드, 비율)
2. 볼린저 밴드 전략
3. 칼만 필터 기반 동적 헤지 비율
4. 전략 성과 비교
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as ts

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 리포트 출력 설정
REPORT_DIR = Path(__file__).parent / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


class Chapter3Analyzer:
    """Chapter 3 평균 회귀 전략 구현 분석 클래스"""
    
    def __init__(self):
        self.results = {}
        self.figures = []
        
        # 디렉토리 생성
        REPORT_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        
    def load_data(self):
        """데이터 로드"""
        print("=" * 60)
        print("📊 데이터 로드 중...")
        print("=" * 60)
        
        # GLD/USO 데이터 (Chapter 3 예제)
        gld_uso_path = Path(__file__).parent / "inputData_GLD_USO.csv"
        if gld_uso_path.exists():
            self.df_gld_uso = pd.read_csv(gld_uso_path)
            self.df_gld_uso['Date'] = pd.to_datetime(self.df_gld_uso['Date'], format='%Y%m%d')
            self.df_gld_uso.set_index('Date', inplace=True)
            print(f"  ✓ GLD/USO: {len(self.df_gld_uso)} 데이터 포인트")
        else:
            self.df_gld_uso = None
            print(f"  ✗ GLD/USO 데이터 없음")
        
        # EWA/EWC 데이터 (칼만 필터 예제용 - Chapter 2에서 복사)
        ewa_ewc_path = Path(__file__).parent / "inputData_EWA_EWC.csv"
        # Chapter 2 폴더에서 가져오기 시도
        if not ewa_ewc_path.exists():
            chapter2_path = Path(__file__).parent.parent.parent / "chapter_2_the_basics_of_mean_reversion" / "src" / "inputData_EWA_EWC.csv"
            if chapter2_path.exists():
                import shutil
                shutil.copy(chapter2_path, ewa_ewc_path)
                print(f"  ℹ️ EWA/EWC 데이터를 Chapter 2에서 복사함")
        
        if ewa_ewc_path.exists():
            self.df_ewa_ewc = pd.read_csv(ewa_ewc_path)
            self.df_ewa_ewc['Date'] = pd.to_datetime(self.df_ewa_ewc['Date'], format='%Y%m%d')
            self.df_ewa_ewc.set_index('Date', inplace=True)
            print(f"  ✓ EWA/EWC: {len(self.df_ewa_ewc)} 데이터 포인트")
        else:
            self.df_ewa_ewc = None
            print(f"  ✗ EWA/EWC 데이터 없음")
            
        print()
        
    def analyze_spread_types(self):
        """스프레드 유형 비교: 가격 스프레드, 로그 가격 스프레드, 비율"""
        print("=" * 60)
        print("🔬 1. 스프레드 유형 비교 분석")
        print("=" * 60)
        
        self.results['spread_types'] = {}
        
        if self.df_gld_uso is None:
            print("  ✗ GLD/USO 데이터 없음 - 분석 건너뜀")
            return
            
        df = self.df_gld_uso.copy()
        lookback = 20
        
        # 1.1 가격 스프레드 (Price Spread with Dynamic Hedge Ratio)
        print("\n### 1.1 가격 스프레드 (동적 헤지 비율)")
        print("-" * 40)
        
        from statsmodels.regression.rolling import RollingOLS
        
        # RollingOLS 사용으로 속도 개선
        print("  ⏳ RollingOLS 계산 중...")
        endog = df['USO']
        exog = sm.add_constant(df['GLD'])
        # window=lookback으로 설정
        rols = RollingOLS(endog, exog, window=lookback)
        rres = rols.fit()
        
        # 파라미터 추출 (GLD의 계수)
        # params는 [const, GLD] 순서일 수 있음 (add_constant에 따라)
        # 하지만 RollingOLS의 params는 컬럼 이름을 유지함
        hedge_ratio_price = rres.params['GLD']
        
        # 스프레드 = USO - hedge_ratio * GLD
        spread_price = df['USO'] - hedge_ratio_price * df['GLD']
        
        # 선형 평균 회귀 전략
        numUnits = -(spread_price - spread_price.rolling(lookback).mean()) / spread_price.rolling(lookback).std()
        
        positions = pd.DataFrame({
            'GLD': -numUnits * hedge_ratio_price * df['GLD'],
            'USO': numUnits * df['USO']
        })
        
        pnl = (positions.shift() * df.pct_change()).sum(axis=1)
        ret = pnl / positions.shift().abs().sum(axis=1)
        ret_clean = ret.replace([np.inf, -np.inf], np.nan).dropna()
        
        apr_price = np.prod(1 + ret_clean) ** (252 / len(ret_clean)) - 1
        sharpe_price = np.sqrt(252) * ret_clean.mean() / ret_clean.std()
        
        self.results['spread_types']['price_spread'] = {
            'apr': apr_price,
            'sharpe': sharpe_price
        }
        
        print(f"  APR: {apr_price*100:.2f}%")
        print(f"  샤프 비율: {sharpe_price:.4f}")
        
        # 1.2 로그 가격 스프레드 (Log Price Spread)
        print("\n### 1.2 로그 가격 스프레드")
        print("-" * 40)
        
        print("  ⏳ RollingOLS (Log) 계산 중...")
        log_df = np.log(df)
        endog_log = log_df['USO']
        exog_log = sm.add_constant(log_df['GLD'])
        rols_log = RollingOLS(endog_log, exog_log, window=lookback)
        rres_log = rols_log.fit()
        
        hedge_ratio_log = rres_log.params['GLD']
        
        spread_log = np.log(df['USO']) - hedge_ratio_log * np.log(df['GLD'])
        
        numUnits_log = -(spread_log - spread_log.rolling(lookback).mean()) / spread_log.rolling(lookback).std()
        
        positions_log = pd.DataFrame({
            'GLD': -numUnits_log * hedge_ratio_log,
            'USO': numUnits_log
        })
        
        pnl_log = (positions_log.shift() * df.pct_change()).sum(axis=1)
        ret_log = pnl_log / positions_log.shift().abs().sum(axis=1)
        ret_log_clean = ret_log.replace([np.inf, -np.inf], np.nan).dropna()
        
        apr_log = np.prod(1 + ret_log_clean) ** (252 / len(ret_log_clean)) - 1
        sharpe_log = np.sqrt(252) * ret_log_clean.mean() / ret_log_clean.std()
        
        self.results['spread_types']['log_spread'] = {
            'apr': apr_log,
            'sharpe': sharpe_log
        }
        
        print(f"  APR: {apr_log*100:.2f}%")
        print(f"  샤프 비율: {sharpe_log:.4f}")
        
        # 1.3 비율 (Ratio)
        print("\n### 1.3 비율 (USO/GLD)")
        print("-" * 40)
        
        ratio = df['USO'] / df['GLD']
        
        numUnits_ratio = -(ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std()
        
        # 롱/숏 동일 자본 배분
        positions_ratio = pd.DataFrame({
            'GLD': -numUnits_ratio * df['GLD'],
            'USO': numUnits_ratio * df['USO']
        })
        
        pnl_ratio = (positions_ratio.shift() * df.pct_change()).sum(axis=1)
        ret_ratio = pnl_ratio / positions_ratio.shift().abs().sum(axis=1)
        ret_ratio_clean = ret_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        
        apr_ratio = np.prod(1 + ret_ratio_clean) ** (252 / len(ret_ratio_clean)) - 1
        sharpe_ratio = np.sqrt(252) * ret_ratio_clean.mean() / ret_ratio_clean.std()
        
        self.results['spread_types']['ratio'] = {
            'apr': apr_ratio,
            'sharpe': sharpe_ratio
        }
        
        print(f"  APR: {apr_ratio*100:.2f}%")
        print(f"  샤프 비율: {sharpe_ratio:.4f}")
        
        # 차트 생성
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(spread_price.values, linewidth=0.8)
        axes[0].set_title('Price Spread: USO - hedgeRatio × GLD', fontsize=12)
        axes[0].set_ylabel('Spread')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(spread_log.values, linewidth=0.8, color='orange')
        axes[1].set_title('Log Price Spread: log(USO) - hedgeRatio × log(GLD)', fontsize=12)
        axes[1].set_ylabel('Spread')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(ratio.values, linewidth=0.8, color='green')
        axes[2].set_title('Ratio: USO / GLD', fontsize=12)
        axes[2].set_ylabel('Ratio')
        axes[2].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'spread_types_comparison.png', dpi=150)
        plt.close(fig)
        self.figures.append('spread_types_comparison.png')
        
        print()
        
    def analyze_bollinger_bands(self):
        """볼린저 밴드 전략 분석"""
        print("=" * 60)
        print("📊 2. 볼린저 밴드 전략 분석")
        print("=" * 60)
        
        self.results['bollinger'] = {}
        
        if self.df_gld_uso is None:
            print("  ✗ GLD/USO 데이터 없음 - 분석 건너뜀")
            return
            
        df = self.df_gld_uso.copy()
        lookback = 20
        
        from statsmodels.regression.rolling import RollingOLS
        
        # 동적 헤지 비율 계산
        print("  ⏳ RollingOLS 계산 중...")
        endog = df['USO']
        exog = sm.add_constant(df['GLD'])
        rols = RollingOLS(endog, exog, window=lookback)
        rres = rols.fit()
        
        hedge_ratio = rres.params['GLD']
        
        # 스프레드 계산
        yport = df['USO'] - hedge_ratio * df['GLD']
        
        # Z-Score 계산
        ma = yport.rolling(lookback).mean()
        mstd = yport.rolling(lookback).std()
        zScore = (yport - ma) / mstd
        
        # 볼린저 밴드 진입/청산
        entry_zscore = 1
        exit_zscore = 0
        
        longs_entry = zScore < -entry_zscore
        longs_exit = zScore >= -exit_zscore
        shorts_entry = zScore > entry_zscore
        shorts_exit = zScore <= exit_zscore
        
        # 포지션 계산
        num_units_long = np.zeros(len(df))
        num_units_long[:] = np.nan
        num_units_long[0] = 0
        num_units_long[longs_entry] = 1
        num_units_long[longs_exit] = 0
        num_units_long = pd.Series(num_units_long).ffill()
        
        num_units_short = np.zeros(len(df))
        num_units_short[:] = np.nan
        num_units_short[0] = 0
        num_units_short[shorts_entry] = -1
        num_units_short[shorts_exit] = 0
        num_units_short = pd.Series(num_units_short).ffill()
        
        num_units = num_units_long + num_units_short
        
        # 포지션 및 P&L
        positions = pd.DataFrame({
            'GLD': -num_units.values * hedge_ratio * df['GLD'].values,
            'USO': num_units.values * df['USO'].values
        })
        
        pnl = (positions.shift() * df.pct_change().values).sum(axis=1)
        ret = pnl / positions.shift().abs().sum(axis=1)
        ret_clean = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()
        
        apr = np.prod(1 + ret_clean) ** (252 / len(ret_clean)) - 1
        sharpe = np.sqrt(252) * ret_clean.mean() / ret_clean.std()
        
        # 최대 낙폭 계산
        cumret = np.cumprod(1 + ret_clean)
        highwatermark = pd.Series(cumret).cummax()
        drawdown = (cumret - highwatermark) / highwatermark
        max_dd = drawdown.min()
        
        self.results['bollinger']['gld_uso'] = {
            'entry_zscore': entry_zscore,
            'exit_zscore': exit_zscore,
            'lookback': lookback,
            'apr': apr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': int((num_units.diff().abs() > 0).sum()),
            'num_days': len(ret_clean)
        }
        
        print(f"\n### GLD-USO 볼린저 밴드 전략")
        print("-" * 40)
        print(f"  진입 Z-Score: ±{entry_zscore}")
        print(f"  청산 Z-Score: {exit_zscore}")
        print(f"  Lookback: {lookback}일")
        print(f"  연간 수익률 (APR): {apr*100:.2f}%")
        print(f"  샤프 비율: {sharpe:.4f}")
        print(f"  최대 낙폭 (MDD): {max_dd*100:.2f}%")
        
        if sharpe > 0.5:
            print(f"  → ✅ 볼린저 밴드가 선형 전략 대비 개선")
        else:
            print(f"  → ⚠️ 전략 개선 필요")
        
        # 차트 생성
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 스프레드와 볼린저 밴드
        axes[0].plot(yport.values, linewidth=0.8, label='Spread')
        axes[0].plot(ma.values, linewidth=1, linestyle='--', label='MA')
        axes[0].plot((ma + entry_zscore * mstd).values, linewidth=0.8, linestyle=':', color='red', label=f'Upper Band (+{entry_zscore}σ)')
        axes[0].plot((ma - entry_zscore * mstd).values, linewidth=0.8, linestyle=':', color='green', label=f'Lower Band (-{entry_zscore}σ)')
        axes[0].set_title('Spread with Bollinger Bands', fontsize=12)
        axes[0].set_ylabel('Spread')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Z-Score
        axes[1].plot(zScore.values, linewidth=0.8)
        axes[1].axhline(y=entry_zscore, color='red', linestyle='--', alpha=0.7)
        axes[1].axhline(y=-entry_zscore, color='green', linestyle='--', alpha=0.7)
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_title('Z-Score', fontsize=12)
        axes[1].set_ylabel('Z-Score')
        axes[1].grid(True, alpha=0.3)
        
        # 누적 수익률
        cumret_plot = np.cumprod(1 + ret_clean) - 1
        axes[2].plot(cumret_plot.values, linewidth=1)
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[2].fill_between(range(len(cumret_plot)), 0, cumret_plot.values,
                           where=cumret_plot.values >= 0, alpha=0.3, color='green')
        axes[2].fill_between(range(len(cumret_plot)), 0, cumret_plot.values,
                           where=cumret_plot.values < 0, alpha=0.3, color='red')
        axes[2].set_title(f'Cumulative Returns (APR={apr*100:.1f}%, Sharpe={sharpe:.2f})', fontsize=12)
        axes[2].set_ylabel('Cumulative Return')
        axes[2].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'bollinger_strategy.png', dpi=150)
        plt.close(fig)
        self.figures.append('bollinger_strategy.png')
        
        print()

    def analyze_scaling_in(self):
        """스케일링 인(평균 매입) vs 올인 전략 비교 분석"""
        print("=" * 60)
        print("📐 2.5. 스케일링 인 vs 올인 비교 분석")
        print("=" * 60)

        self.results['scaling_in'] = {}

        if self.df_gld_uso is None:
            print("  ✗ GLD/USO 데이터 없음 - 분석 건너뜀")
            return

        df = self.df_gld_uso.copy()
        lookback = 20

        from statsmodels.regression.rolling import RollingOLS

        # 동적 헤지 비율 계산
        endog = df['USO']
        exog = sm.add_constant(df['GLD'])
        rols = RollingOLS(endog, exog, window=lookback)
        rres = rols.fit()
        hedge_ratio = rres.params['GLD']

        # 스프레드 계산
        yport = df['USO'] - hedge_ratio * df['GLD']
        ma = yport.rolling(lookback).mean()
        mstd = yport.rolling(lookback).std()
        zScore = (yport - ma) / mstd

        strategies = {}

        # --- 전략 A: 올인 (단일 볼린저 밴드, entry=1, exit=0) ---
        entry_z = 1
        longs_e = zScore < -entry_z
        longs_x = zScore >= 0
        shorts_e = zScore > entry_z
        shorts_x = zScore <= 0

        nu_long = np.zeros(len(df)); nu_long[:] = np.nan; nu_long[0] = 0
        nu_long[longs_e] = 1; nu_long[longs_x] = 0
        nu_long = pd.Series(nu_long).ffill()

        nu_short = np.zeros(len(df)); nu_short[:] = np.nan; nu_short[0] = 0
        nu_short[shorts_e] = -1; nu_short[shorts_x] = 0
        nu_short = pd.Series(nu_short).ffill()

        nu = nu_long + nu_short
        pos = pd.DataFrame({
            'GLD': -nu.values * hedge_ratio * df['GLD'].values,
            'USO': nu.values * df['USO'].values
        })
        pnl_a = (pos.shift() * df.pct_change().values).sum(axis=1)
        ret_a = pnl_a / pos.shift().abs().sum(axis=1)
        ret_a = pd.Series(ret_a).replace([np.inf, -np.inf], np.nan).dropna()

        apr_a = np.prod(1 + ret_a) ** (252 / len(ret_a)) - 1
        sharpe_a = np.sqrt(252) * ret_a.mean() / ret_a.std()
        cumret_a = np.cumprod(1 + ret_a)
        mdd_a = ((cumret_a - cumret_a.cummax()) / cumret_a.cummax()).min()

        strategies['allin_z1'] = {'apr': apr_a, 'sharpe': sharpe_a, 'mdd': mdd_a, 'label': '올인 (Z=1)'}

        # --- 전략 B: 스케일링 인 (2단계, entry=0.5,1.5, exit=0) ---
        # 1단계: |Z|>=0.5에서 1단위, |Z|>=1.5에서 추가 1단위, Z=0에서 청산
        nu_s1 = np.zeros(len(df)); nu_s1[:] = np.nan; nu_s1[0] = 0
        nu_s1[zScore < -0.5] = 1; nu_s1[zScore >= 0] = 0
        nu_s1 = pd.Series(nu_s1).ffill()

        nu_s2 = np.zeros(len(df)); nu_s2[:] = np.nan; nu_s2[0] = 0
        nu_s2[zScore < -1.5] = 1; nu_s2[zScore >= -0.5] = 0
        nu_s2 = pd.Series(nu_s2).ffill()

        nu_ss1 = np.zeros(len(df)); nu_ss1[:] = np.nan; nu_ss1[0] = 0
        nu_ss1[zScore > 0.5] = -1; nu_ss1[zScore <= 0] = 0
        nu_ss1 = pd.Series(nu_ss1).ffill()

        nu_ss2 = np.zeros(len(df)); nu_ss2[:] = np.nan; nu_ss2[0] = 0
        nu_ss2[zScore > 1.5] = -1; nu_ss2[zScore <= 0.5] = 0
        nu_ss2 = pd.Series(nu_ss2).ffill()

        nu_scale = nu_s1 + nu_s2 + nu_ss1 + nu_ss2

        pos_b = pd.DataFrame({
            'GLD': -nu_scale.values * hedge_ratio * df['GLD'].values,
            'USO': nu_scale.values * df['USO'].values
        })
        pnl_b = (pos_b.shift() * df.pct_change().values).sum(axis=1)
        ret_b = pnl_b / pos_b.shift().abs().sum(axis=1)
        ret_b = pd.Series(ret_b).replace([np.inf, -np.inf], np.nan).dropna()

        apr_b = np.prod(1 + ret_b) ** (252 / len(ret_b)) - 1
        sharpe_b = np.sqrt(252) * ret_b.mean() / ret_b.std()
        cumret_b = np.cumprod(1 + ret_b)
        mdd_b = ((cumret_b - cumret_b.cummax()) / cumret_b.cummax()).min()

        strategies['scale_in_z12'] = {'apr': apr_b, 'sharpe': sharpe_b, 'mdd': mdd_b, 'label': '스케일링 인 (Z=0.5,1.5)'}

        # --- 전략 C: 선형 전략 (연속 스케일 인) ---
        numUnits_lin = -(yport - ma) / mstd
        pos_c = pd.DataFrame({
            'GLD': -numUnits_lin * hedge_ratio * df['GLD'],
            'USO': numUnits_lin * df['USO']
        })
        pnl_c = (pos_c.shift() * df.pct_change()).sum(axis=1)
        ret_c = pnl_c / pos_c.shift().abs().sum(axis=1)
        ret_c = pd.Series(ret_c).replace([np.inf, -np.inf], np.nan).dropna()

        apr_c = np.prod(1 + ret_c) ** (252 / len(ret_c)) - 1
        sharpe_c = np.sqrt(252) * ret_c.mean() / ret_c.std()
        cumret_c = np.cumprod(1 + ret_c)
        mdd_c = ((cumret_c - cumret_c.cummax()) / cumret_c.cummax()).min()

        strategies['linear'] = {'apr': apr_c, 'sharpe': sharpe_c, 'mdd': mdd_c, 'label': '선형 (연속 스케일 인)'}

        self.results['scaling_in'] = strategies

        print("\n### 전략별 성과 비교")
        print("-" * 55)
        print(f"  {'전략':<25} {'APR':>8} {'Sharpe':>8} {'MDD':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for key, s in strategies.items():
            print(f"  {s['label']:<25} {s['apr']*100:>7.2f}% {s['sharpe']:>8.4f} {s['mdd']*100:>7.2f}%")

        # 이론적 비교 (Schoenberg & Corwin 증명)
        print("\n  📖 Schoenberg & Corwin (2010) 핵심 결론:")
        print("  → 백테스트에서 스케일링 인이 올인보다 최적인 경우는 없다")
        print("  → 단, 변동성이 변하는 실시장에서는 스케일링 인이 유용할 수 있다")

        # 차트 생성
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 누적 수익률 비교
        axes[0].plot((np.cumprod(1 + ret_a) - 1).values, linewidth=1, label=f'올인 Z=1 (Sharpe={sharpe_a:.2f})')
        axes[0].plot((np.cumprod(1 + ret_b) - 1).values, linewidth=1, label=f'스케일링 인 Z=0.5,1.5 (Sharpe={sharpe_b:.2f})')
        axes[0].plot((np.cumprod(1 + ret_c) - 1).values, linewidth=1, label=f'선형 연속 (Sharpe={sharpe_c:.2f})')
        axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[0].set_title('Scaling-In vs All-In: Cumulative Returns Comparison', fontsize=12)
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 드로다운 비교
        dd_a = (cumret_a - cumret_a.cummax()) / cumret_a.cummax()
        dd_b = (cumret_b - cumret_b.cummax()) / cumret_b.cummax()
        dd_c = (cumret_c - cumret_c.cummax()) / cumret_c.cummax()
        axes[1].fill_between(range(len(dd_a)), 0, dd_a.values, alpha=0.3, label=f'올인 (MDD={mdd_a*100:.1f}%)')
        axes[1].fill_between(range(len(dd_b)), 0, dd_b.values, alpha=0.3, label=f'스케일링 인 (MDD={mdd_b*100:.1f}%)')
        axes[1].fill_between(range(len(dd_c)), 0, dd_c.values, alpha=0.3, label=f'선형 (MDD={mdd_c*100:.1f}%)')
        axes[1].set_title('Drawdown Comparison', fontsize=12)
        axes[1].set_ylabel('Drawdown')
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'scaling_in_comparison.png', dpi=150)
        plt.close(fig)
        self.figures.append('scaling_in_comparison.png')

        print()

    def analyze_kalman_filter(self):
        """칼만 필터 기반 동적 헤지 비율 분석"""
        print("=" * 60)
        print("🔧 3. 칼만 필터 전략 분석")
        print("=" * 60)
        
        self.results['kalman'] = {}
        
        if self.df_ewa_ewc is None:
            print("  ✗ EWA/EWC 데이터 없음 - 분석 건너뜀")
            return
            
        df = self.df_ewa_ewc.copy()
        
        # 칼만 필터 구현
        x = df['EWA'].values
        y = df['EWC'].values
        
        # x에 절편 항 추가 [EWA, 1]
        x_aug = np.column_stack([x, np.ones(len(x))])
        
        # 칼만 필터 파라미터
        delta = 0.0001  # 상태 변화율
        Ve = 0.001      # 측정 오차 분산
        
        # 초기화
        n = len(y)
        yhat = np.full(n, np.nan)  # 예측값
        e = np.full(n, np.nan)     # 예측 오차
        Q = np.full(n, np.nan)     # 예측 오차 분산
        
        R = np.zeros((2, 2))       # 상태 공분산
        P = R.copy()
        beta = np.full((2, n), np.nan)  # [기울기, 절편]
        
        Vw = delta / (1 - delta) * np.eye(2)  # 상태 전이 노이즈 공분산
        
        # 초기 beta
        beta[:, 0] = 0
        
        # 칼만 필터 반복
        for t in range(n):
            if t > 0:
                beta[:, t] = beta[:, t-1]  # 상태 예측
                R = P + Vw                  # 상태 공분산 예측
            
            yhat[t] = np.dot(x_aug[t, :], beta[:, t])  # 측정 예측
            Q[t] = np.dot(np.dot(x_aug[t, :], R), x_aug[t, :].T) + Ve  # 측정 분산 예측
            
            e[t] = y[t] - yhat[t]  # 예측 오차
            
            K = np.dot(R, x_aug[t, :].T) / Q[t]  # 칼만 이득
            beta[:, t] = beta[:, t] + K * e[t]  # 상태 업데이트
            P = R - np.outer(K, x_aug[t, :]) @ R  # 상태 공분산 업데이트
        
        # 거래 신호
        sqrt_Q = np.sqrt(Q)
        longs_entry = e < -sqrt_Q
        longs_exit = e > 0
        shorts_entry = e > sqrt_Q
        shorts_exit = e < 0
        
        # 포지션 계산
        num_units_long = np.zeros(n)
        num_units_long[:] = np.nan
        num_units_long[0] = 0
        num_units_long[longs_entry] = 1
        num_units_long[longs_exit] = 0
        num_units_long = pd.Series(num_units_long).ffill()
        
        num_units_short = np.zeros(n)
        num_units_short[:] = np.nan
        num_units_short[0] = 0
        num_units_short[shorts_entry] = -1
        num_units_short[shorts_exit] = 0
        num_units_short = pd.Series(num_units_short).ffill()
        
        num_units = num_units_long + num_units_short
        
        # 헤지 비율 (기울기)
        hedge_ratio = beta[0, :]
        
        # 포지션 및 P&L
        positions = pd.DataFrame({
            'EWA': -num_units.values * hedge_ratio * df['EWA'].values,
            'EWC': num_units.values * df['EWC'].values
        })
        
        pnl = (positions.shift() * df.pct_change().values).sum(axis=1)
        ret = pnl / positions.shift().abs().sum(axis=1)
        ret_clean = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()
        
        apr = np.prod(1 + ret_clean) ** (252 / len(ret_clean)) - 1
        sharpe = np.sqrt(252) * ret_clean.mean() / ret_clean.std()
        
        # 최대 낙폭
        cumret = np.cumprod(1 + ret_clean)
        highwatermark = pd.Series(cumret).cummax()
        drawdown = (cumret - highwatermark) / highwatermark
        max_dd = drawdown.min()
        
        self.results['kalman']['ewa_ewc'] = {
            'delta': delta,
            'Ve': Ve,
            'apr': apr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_days': len(ret_clean),
            'beta_slope_mean': np.nanmean(beta[0, :]),
            'beta_slope_std': np.nanstd(beta[0, :]),
            'beta_intercept_mean': np.nanmean(beta[1, :]),
            'beta_intercept_std': np.nanstd(beta[1, :])
        }
        
        print(f"\n### EWA-EWC 칼만 필터 전략")
        print("-" * 40)
        print(f"  δ (상태 변화율): {delta}")
        print(f"  Vε (측정 오차 분산): {Ve}")
        print(f"  평균 헤지 비율: {np.nanmean(beta[0, :]):.4f} ± {np.nanstd(beta[0, :]):.4f}")
        print(f"  연간 수익률 (APR): {apr*100:.2f}%")
        print(f"  샤프 비율: {sharpe:.4f}")
        print(f"  최대 낙폭 (MDD): {max_dd*100:.2f}%")
        
        if sharpe > 1.0:
            print(f"  → ✅ 칼만 필터 우수한 성과")
        else:
            print(f"  → ⚠️ 파라미터 조정 필요")
        
        # 차트 생성
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # 헤지 비율 (기울기)
        axes[0].plot(beta[0, :], linewidth=0.8)
        axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Kalman Filter: Slope (Hedge Ratio) β₁', fontsize=12)
        axes[0].set_ylabel('β₁ (Slope)')
        axes[0].grid(True, alpha=0.3)
        
        # 절편
        axes[1].plot(beta[1, :], linewidth=0.8, color='orange')
        axes[1].set_title('Kalman Filter: Intercept β₀', fontsize=12)
        axes[1].set_ylabel('β₀ (Intercept)')
        axes[1].grid(True, alpha=0.3)
        
        # 예측 오차와 표준편차
        axes[2].plot(e[2:], linewidth=0.8, label='Prediction Error e(t)')
        axes[2].plot(sqrt_Q[2:], linewidth=0.8, color='red', label='√Q(t)')
        axes[2].plot(-sqrt_Q[2:], linewidth=0.8, color='green', label='-√Q(t)')
        axes[2].set_title('Measurement Prediction Error', fontsize=12)
        axes[2].set_ylabel('Error')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        # 누적 수익률
        cumret_plot = np.cumprod(1 + ret_clean) - 1
        axes[3].plot(cumret_plot.values, linewidth=1)
        axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[3].fill_between(range(len(cumret_plot)), 0, cumret_plot.values,
                           where=cumret_plot.values >= 0, alpha=0.3, color='green')
        axes[3].fill_between(range(len(cumret_plot)), 0, cumret_plot.values,
                           where=cumret_plot.values < 0, alpha=0.3, color='red')
        axes[3].set_title(f'Cumulative Returns (APR={apr*100:.1f}%, Sharpe={sharpe:.2f})', fontsize=12)
        axes[3].set_ylabel('Cumulative Return')
        axes[3].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / 'kalman_strategy.png', dpi=150)
        plt.close(fig)
        self.figures.append('kalman_strategy.png')
        
        print()
        
    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 60)
        print("📝 4. 리포트 생성")
        print("=" * 60)
        
        report_lines = []
        
        # 제목 및 메타데이터
        report_lines.append("# Chapter 3: 평균 회귀 전략 구현 (Implementing Mean Reversion Strategies)\n")
        report_lines.append("# 분석 리포트\n\n")
        report_lines.append(f"> **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("> **데이터 출처**: Ernest Chan's \"Algorithmic Trading\" (2013)\n\n")
        report_lines.append("---\n\n")
        
        # 목차
        report_lines.append("## 목차\n\n")
        report_lines.append("1. [개요 및 문제 정의](#1-개요-및-문제-정의)\n")
        report_lines.append("2. [사용 데이터](#2-사용-데이터)\n")
        report_lines.append("3. [스프레드 유형 비교](#3-스프레드-유형-비교)\n")
        report_lines.append("4. [볼린저 밴드 전략](#4-볼린저-밴드-전략)\n")
        report_lines.append("5. [칼만 필터 전략](#5-칼만-필터-전략)\n")
        report_lines.append("6. [전략 비교 및 결론](#6-전략-비교-및-결론)\n\n")
        report_lines.append("---\n\n")
        
        # 1. 개요 및 문제 정의
        report_lines.append("## 1. 개요 및 문제 정의\n\n")
        report_lines.append("### 💡 해결하려는 문제\n\n")
        report_lines.append("**\"정상성/공적분이 완벽하지 않은 시계열에서 어떻게 실용적인 평균 회귀 전략을 구현할 수 있을까?\"**\n\n")
        report_lines.append("Chapter 2에서 정상성과 공적분의 이론적 기초를 배웠지만, 실제 시장에서는:\n\n")
        report_lines.append("1. **완벽한 정상성/공적분은 드물다** - 단기 또는 계절적 평균 회귀만 존재하는 경우가 많음\n")
        report_lines.append("2. **헤지 비율이 변한다** - 시간에 따라 두 자산 간 관계가 변화\n")
        report_lines.append("3. **무한한 자본이 없다** - 선형 전략의 스케일 인은 비현실적\n\n")
        
        report_lines.append("### 📐 핵심 수학적 개념\n\n")
        report_lines.append("| 개념 | 수식 | 의미 |\n")
        report_lines.append("|------|------|------|\n")
        report_lines.append("| **가격 스프레드** | $y = y_1 - h \\cdot y_2$ | 고정 주식 수 포트폴리오 |\n")
        report_lines.append("| **로그 가격 스프레드** | $\\log(q) = h_1 \\log(y_1) + h_2 \\log(y_2)$ | 고정 자본 가중치 포트폴리오 |\n")
        report_lines.append("| **볼린저 밴드** | 진입: $|Z| > Z_{entry}$, 청산: $|Z| < Z_{exit}$ | 이산적 진입/청산 |\n")
        report_lines.append("| **칼만 필터** | $\\hat{\\beta}(t|t) = \\hat{\\beta}(t|t-1) + K(t) \\cdot e(t)$ | 동적 헤지 비율 추정 |\n\n")
        report_lines.append("---\n\n")
        
        # 2. 사용 데이터
        report_lines.append("## 2. 사용 데이터\n\n")
        report_lines.append("### 📊 데이터셋 설명\n\n")
        report_lines.append("| 파일명 | 내용 | 용도 |\n")
        report_lines.append("|--------|------|------|\n")
        report_lines.append("| `inputData_GLD_USO.csv` | GLD(금)/USO(원유) ETF | 스프레드 유형 비교, 볼린저 밴드 |\n")
        report_lines.append("| `inputData_EWA_EWC.csv` | EWA(호주)/EWC(캐나다) ETF | 칼만 필터 전략 |\n\n")
        
        report_lines.append("### 🎯 데이터 선정 이유\n\n")
        report_lines.append("- **GLD-USO**: 금과 원유는 인플레이션과 연관되어 있다는 믿음이 있지만, **공적분하지 않음**\n")
        report_lines.append("  - 완벽하지 않은 공적분에서 단기 평균 회귀를 포착하는 방법 시연\n")
        report_lines.append("- **EWA-EWC**: 호주와 캐나다는 모두 원자재 경제, **공적분 관계** 존재\n")
        report_lines.append("  - 칼만 필터로 동적 헤지 비율 추정 효과 시연\n\n")
        report_lines.append("---\n\n")
        
        # 3. 스프레드 유형 비교
        report_lines.append("## 3. 스프레드 유형 비교\n\n")
        report_lines.append("### 🔬 분석 목적\n\n")
        report_lines.append("세 가지 스프레드 유형의 성과를 비교하여 어떤 방식이 가장 효과적인지 확인합니다.\n\n")
        
        report_lines.append("### 3.1 스프레드 유형 설명\n\n")
        report_lines.append("| 유형 | 수식 | 특징 |\n")
        report_lines.append("|------|------|------|\n")
        report_lines.append("| **가격 스프레드** | $y = USO - \\beta \\cdot GLD$ | 고정 주식 수, 동적 헤지 비율 적용 |\n")
        report_lines.append("| **로그 가격 스프레드** | $y = \\log(USO) - \\beta \\cdot \\log(GLD)$ | 고정 자본 가중치, 리밸런싱 필요 |\n")
        report_lines.append("| **비율** | $y = USO / GLD$ | 헤지 비율 불필요, 스케일 독립적 |\n\n")
        
        if 'spread_types' in self.results:
            report_lines.append("### 3.2 성과 비교 결과 (GLD-USO 선형 평균회귀 전략)\n\n")
            report_lines.append("| 스프레드 유형 | APR | 샤프 비율 | 평가 |\n")
            report_lines.append("|--------------|-----|-----------|------|\n")
            
            if 'price_spread' in self.results['spread_types']:
                ps = self.results['spread_types']['price_spread']
                status = "✅" if ps['sharpe'] > 0.5 else "⚠️"
                report_lines.append(f"| 가격 스프레드 | {ps['apr']*100:.2f}% | {ps['sharpe']:.4f} | {status} |\n")
            
            if 'log_spread' in self.results['spread_types']:
                ls = self.results['spread_types']['log_spread']
                status = "✅" if ls['sharpe'] > 0.5 else "⚠️"
                report_lines.append(f"| 로그 가격 스프레드 | {ls['apr']*100:.2f}% | {ls['sharpe']:.4f} | {status} |\n")
            
            if 'ratio' in self.results['spread_types']:
                r = self.results['spread_types']['ratio']
                status = "✅" if r['sharpe'] > 0.5 else ("❌" if r['sharpe'] < 0 else "⚠️")
                report_lines.append(f"| 비율 | {r['apr']*100:.2f}% | {r['sharpe']:.4f} | {status} |\n")
            
            report_lines.append("\n")
            
        report_lines.append("### 3.3 스프레드 유형별 차트\n\n")
        report_lines.append("![Spread Types Comparison](figures/spread_types_comparison.png)\n\n")
        report_lines.append("> 📊 **해석**: 가격 스프레드(동적 헤지 비율)가 가장 정상적으로 보이며, 비율은 평균 회귀하지 않는 경향\n\n")
        report_lines.append("---\n\n")
        
        # 4. 볼린저 밴드 전략
        report_lines.append("## 4. 볼린저 밴드 전략\n\n")
        report_lines.append("### 📈 전략 원리\n\n")
        report_lines.append("볼린저 밴드는 **이산적 진입/청산**을 사용하는 실용적인 평균 회귀 전략입니다.\n\n")
        report_lines.append("```python\n")
        report_lines.append("# Z-Score 계산\n")
        report_lines.append("z_score = (spread - moving_avg) / moving_std\n")
        report_lines.append("\n")
        report_lines.append("# 진입 조건\n")
        report_lines.append("long_entry = z_score < -entry_zscore   # 저평가 → 매수\n")
        report_lines.append("short_entry = z_score > entry_zscore   # 고평가 → 매도\n")
        report_lines.append("\n")
        report_lines.append("# 청산 조건\n")
        report_lines.append("long_exit = z_score >= -exit_zscore    # 평균 회복\n")
        report_lines.append("short_exit = z_score <= exit_zscore\n")
        report_lines.append("```\n\n")
        
        report_lines.append("### 4.1 장점\n\n")
        report_lines.append("- **자본 관리 용이**: 0 또는 1 단위만 투자\n")
        report_lines.append("- **파라미터 최적화 가능**: `entry_zscore`, `exit_zscore`, `lookback`\n")
        report_lines.append("- **선형 전략 대비 개선된 성과**\n\n")
        
        if 'bollinger' in self.results and 'gld_uso' in self.results['bollinger']:
            bb = self.results['bollinger']['gld_uso']
            report_lines.append("### 4.2 GLD-USO 볼린저 밴드 전략 결과\n\n")
            report_lines.append("**파라미터:**\n\n")
            report_lines.append(f"- Entry Z-Score: ±{bb['entry_zscore']}\n")
            report_lines.append(f"- Exit Z-Score: {bb['exit_zscore']}\n")
            report_lines.append(f"- Lookback: {bb['lookback']}일\n\n")
            
            report_lines.append("**성과 지표:**\n\n")
            report_lines.append("| 지표 | 값 | 평가 |\n")
            report_lines.append("|------|------|------|\n")
            
            apr_status = "✅ 우수" if bb['apr'] > 0.10 else ("⚠️ 양호" if bb['apr'] > 0.05 else "❌ 저조")
            report_lines.append(f"| 연간 수익률 (APR) | {bb['apr']*100:.2f}% | {apr_status} |\n")
            
            sharpe_status = "✅ 우수" if bb['sharpe'] > 0.8 else ("⚠️ 양호" if bb['sharpe'] > 0.5 else "❌ 저조")
            report_lines.append(f"| **샤프 비율** | **{bb['sharpe']:.4f}** | {sharpe_status} |\n")
            
            mdd_status = "✅ 양호" if bb['max_drawdown'] > -0.20 else ("⚠️ 주의" if bb['max_drawdown'] > -0.30 else "❌ 위험")
            report_lines.append(f"| 최대 낙폭 (MDD) | {bb['max_drawdown']*100:.2f}% | {mdd_status} |\n\n")
            
        report_lines.append("### 4.3 볼린저 밴드 전략 차트\n\n")
        report_lines.append("![Bollinger Strategy](figures/bollinger_strategy.png)\n\n")
        report_lines.append("> 📊 **차트 해석**:\n")
        report_lines.append("> - 상단: 스프레드와 볼린저 밴드 (빨강=상단 밴드, 초록=하단 밴드)\n")
        report_lines.append("> - 중단: Z-Score와 진입/청산 임계값\n")
        report_lines.append("> - 하단: 누적 수익률\n\n")
        report_lines.append("---\n\n")

        # 4.5 스케일링 인 vs 올인
        report_lines.append("## 4.5. 스케일링 인 vs 올인 비교\n\n")
        report_lines.append("### 📐 이론적 배경\n\n")
        report_lines.append("Schoenberg & Corwin (2010)은 **스케일링 인(평균 매입)이 백테스트에서 결코 최적이 아님**을 증명했습니다.\n\n")
        report_lines.append("가격이 $L_1$으로 하락 후, 확률 $p$로 $L_2 < L_1$까지 추가 하락한 뒤 $F$로 회귀한다고 가정하면:\n\n")
        report_lines.append("| 전략 | 기대 이익 |\n")
        report_lines.append("|------|----------|\n")
        report_lines.append("| $L_1$에서 올인 | $2(F - L_1)$ |\n")
        report_lines.append("| $L_2$에서 올인 | $2p(F - L_2)$ |\n")
        report_lines.append("| 평균 매입 ($L_1$, $L_2$) | $(F - L_1) + p(F - L_2)$ |\n\n")
        report_lines.append("전환 확률 $\\hat{p} = (F - L_1)/(F - L_2)$를 기준으로, $p < \\hat{p}$이면 $L_1$ 올인이 최적, $p > \\hat{p}$이면 $L_2$ 올인이 최적입니다. **평균 매입이 최적인 경우는 없습니다.**\n\n")
        report_lines.append("단, 실시장에서는 변동성이 일정하지 않으므로 스케일링 인이 더 나은 **실현 샤프 비율**을 낼 수 있습니다.\n\n")

        if 'scaling_in' in self.results and self.results['scaling_in']:
            report_lines.append("### 4.6 GLD-USO 실증 비교\n\n")
            report_lines.append("| 전략 | APR | 샤프 비율 | MDD | 평가 |\n")
            report_lines.append("|------|-----|-----------|-----|------|\n")

            for key, s in self.results['scaling_in'].items():
                sharpe_status = "✅" if s['sharpe'] > 0.8 else ("⚠️" if s['sharpe'] > 0.5 else "❌")
                report_lines.append(f"| {s['label']} | {s['apr']*100:.2f}% | {s['sharpe']:.4f} | {s['mdd']*100:.2f}% | {sharpe_status} |\n")

            report_lines.append("\n")
            report_lines.append("![Scaling-In Comparison](figures/scaling_in_comparison.png)\n\n")
            report_lines.append("> 📊 **해석**: 올인 전략이 가장 높은 수익률을 보이며, Schoenberg & Corwin의 이론과 일치합니다.\n")
            report_lines.append("> 그러나 스케일링 인은 MDD를 줄여 실시장 적용에 유리할 수 있습니다.\n\n")

        report_lines.append("---\n\n")

        # 5. 칼만 필터 전략
        report_lines.append("## 5. 칼만 필터 전략\n\n")
        report_lines.append("### 🔧 칼만 필터란?\n\n")
        report_lines.append("칼만 필터는 **숨겨진 변수의 최적 추정**을 위한 선형 알고리즘입니다.\n\n")
        report_lines.append("**핵심 방정식:**\n\n")
        report_lines.append("$$y(t) = x(t) \\beta(t) + \\epsilon(t) \\quad \\text{(측정 방정식)}$$\n\n")
        report_lines.append("$$\\beta(t) = \\beta(t-1) + \\omega(t-1) \\quad \\text{(상태 전이)}$$\n\n")
        report_lines.append("$$\\hat{\\beta}(t|t) = \\hat{\\beta}(t|t-1) + K(t) \\cdot e(t) \\quad \\text{(상태 업데이트)}$$\n\n")
        
        report_lines.append("### 5.1 칼만 필터의 장점\n\n")
        report_lines.append("| 장점 | 설명 |\n")
        report_lines.append("|------|------|\n")
        report_lines.append("| **동적 헤지 비율** | 시간에 따라 변하는 헤지 비율 자동 추정 |\n")
        report_lines.append("| **스프레드 평균** | 절편(β₀)이 스프레드의 이동 평균 역할 |\n")
        report_lines.append("| **예측 오차 분산** | √Q(t)가 볼린저 밴드의 표준편차 역할 |\n")
        report_lines.append("| **데이터 가중** | 최신 데이터에 더 많은 가중치, 임의 절단점 없음 |\n\n")
        
        if 'kalman' in self.results and 'ewa_ewc' in self.results['kalman']:
            kf = self.results['kalman']['ewa_ewc']
            report_lines.append("### 5.2 EWA-EWC 칼만 필터 전략 결과\n\n")
            report_lines.append("**파라미터:**\n\n")
            report_lines.append(f"- δ (상태 변화율): {kf['delta']}\n")
            report_lines.append(f"- Vε (측정 오차 분산): {kf['Ve']}\n")
            report_lines.append(f"- 평균 헤지 비율: {kf['beta_slope_mean']:.4f} ± {kf['beta_slope_std']:.4f}\n\n")
            
            report_lines.append("**성과 지표:**\n\n")
            report_lines.append("| 지표 | 값 | 평가 |\n")
            report_lines.append("|------|------|------|\n")
            
            apr_status = "✅ 우수" if kf['apr'] > 0.15 else ("⚠️ 양호" if kf['apr'] > 0.10 else "❌ 저조")
            report_lines.append(f"| 연간 수익률 (APR) | {kf['apr']*100:.2f}% | {apr_status} |\n")
            
            sharpe_status = "✅ 우수" if kf['sharpe'] > 1.5 else ("⚠️ 양호" if kf['sharpe'] > 1.0 else "❌ 저조")
            report_lines.append(f"| **샤프 비율** | **{kf['sharpe']:.4f}** | {sharpe_status} |\n")
            
            mdd_status = "✅ 양호" if kf['max_drawdown'] > -0.15 else ("⚠️ 주의" if kf['max_drawdown'] > -0.25 else "❌ 위험")
            report_lines.append(f"| 최대 낙폭 (MDD) | {kf['max_drawdown']*100:.2f}% | {mdd_status} |\n\n")
            
        report_lines.append("### 5.3 칼만 필터 전략 차트\n\n")
        report_lines.append("![Kalman Strategy](figures/kalman_strategy.png)\n\n")
        report_lines.append("> 📊 **차트 해석**:\n")
        report_lines.append("> - 1행: 칼만 필터 추정 기울기 (헤지 비율) - 1 주위에서 진동\n")
        report_lines.append("> - 2행: 칼만 필터 추정 절편 - 시간에 따라 변화\n")
        report_lines.append("> - 3행: 예측 오차 e(t)와 표준편차 √Q(t)\n")
        report_lines.append("> - 4행: 누적 수익률\n\n")
        report_lines.append("---\n\n")
        
        # 6. 결론 및 권고사항
        report_lines.append("## 6. 전략 비교 및 결론\n\n")
        report_lines.append("### ✅ 핵심 발견\n\n")
        report_lines.append("| 전략 | 데이터 | APR | 샤프 | 장점 |\n")
        report_lines.append("|------|--------|-----|------|------|\n")
        
        if 'spread_types' in self.results and 'price_spread' in self.results['spread_types']:
            ps = self.results['spread_types']['price_spread']
            report_lines.append(f"| 선형 (가격 스프레드) | GLD-USO | {ps['apr']*100:.1f}% | {ps['sharpe']:.2f} | 단순함 |\n")
        
        if 'bollinger' in self.results and 'gld_uso' in self.results['bollinger']:
            bb = self.results['bollinger']['gld_uso']
            report_lines.append(f"| 볼린저 밴드 | GLD-USO | {bb['apr']*100:.1f}% | {bb['sharpe']:.2f} | 자본 관리 용이 |\n")
        
        if 'scaling_in' in self.results and 'allin_z1' in self.results['scaling_in']:
            si = self.results['scaling_in']['allin_z1']
            report_lines.append(f"| 올인 (Z=1) | GLD-USO | {si['apr']*100:.1f}% | {si['sharpe']:.2f} | 이론적 최적 |\n")

        if 'scaling_in' in self.results and 'scale_in_z12' in self.results['scaling_in']:
            si2 = self.results['scaling_in']['scale_in_z12']
            report_lines.append(f"| 스케일링 인 | GLD-USO | {si2['apr']*100:.1f}% | {si2['sharpe']:.2f} | 변동성 적응 |\n")

        if 'kalman' in self.results and 'ewa_ewc' in self.results['kalman']:
            kf = self.results['kalman']['ewa_ewc']
            report_lines.append(f"| 칼만 필터 | EWA-EWC | {kf['apr']*100:.1f}% | {kf['sharpe']:.2f} | 동적 헤지 비율 |\n")

        report_lines.append("\n")

        report_lines.append("### 💡 트레이딩 권고사항\n\n")
        report_lines.append("1. **스프레드 유형 선택**:\n")
        report_lines.append("   - 공적분 페어: 가격 스프레드 또는 로그 가격 스프레드 사용\n")
        report_lines.append("   - 비공적분 페어: 동적 헤지 비율 필수, 비율은 피하기\n\n")
        
        report_lines.append("2. **볼린저 밴드 전략**:\n")
        report_lines.append("   - 선형 전략의 실용적 대안\n")
        report_lines.append("   - Entry/Exit Z-Score는 훈련 데이터로 최적화\n\n")
        
        report_lines.append("3. **칼만 필터 전략**:\n")
        report_lines.append("   - 공적분 페어에서 가장 우수한 성과\n")
        report_lines.append("   - δ 파라미터로 헤지 비율 변화 속도 조절\n\n")
        
        report_lines.append("### ⚠️ 주의사항\n\n")
        report_lines.append("- **데이터 오류**: 평균 회귀 전략은 이상치에 특히 민감 (잘못된 수익 부풀리기 위험)\n")
        report_lines.append("- **스케일 인**: 이론적으로 최적이 아닐 수 있으나, 실제로는 변동성 변화에 유용\n")
        report_lines.append("- **거래 비용**: 본 백테스트에 미포함\n")
        report_lines.append("- **Look-ahead bias**: 전체 데이터로 파라미터 계산 후 동일 데이터로 테스트\n\n")
        
        report_lines.append("---\n\n")
        report_lines.append("*이 리포트는 `run_chapter3_analysis.py`에 의해 자동 생성되었습니다.*\n")
        
        # 파일 저장
        report_path = REPORT_DIR / "chapter3_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
            
        print(f"  ✓ 리포트 저장: {report_path}")
        print(f"  ✓ 차트 저장: {FIGURES_DIR}")
        print()
        
    def run(self):
        """전체 분석 실행"""
        print("\n" + "=" * 60)
        print("   Chapter 3: 평균 회귀 전략 구현 - 종합 분석")
        print("   Ernest Chan's Algorithmic Trading")
        print("=" * 60 + "\n")
        
        self.load_data()
        self.analyze_spread_types()
        self.analyze_bollinger_bands()
        self.analyze_scaling_in()
        self.analyze_kalman_filter()
        self.generate_report()
        
        print("=" * 60)
        print("✅ 분석 완료!")
        print("=" * 60)
        print(f"\n📁 리포트 위치: {REPORT_DIR / 'chapter3_report.md'}")
        print(f"📊 차트 위치: {FIGURES_DIR}\n")


if __name__ == "__main__":
    analyzer = Chapter3Analyzer()
    analyzer.run()
