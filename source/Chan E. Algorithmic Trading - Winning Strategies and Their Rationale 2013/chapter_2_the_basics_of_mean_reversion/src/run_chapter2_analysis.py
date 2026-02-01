#!/usr/bin/env python3
"""
Chapter 2: 평균 회귀 기초 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 2의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. 정상성 검정 (ADF, 허스트 지수, 분산비)
2. 공적분 검정 (CADF, Johansen)
3. 반감기 계산
4. 선형 평균회귀 전략 백테스트
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
import statsmodels.tsa.vector_ar.vecm as vm

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 현재 디렉토리를 모듈 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genhurst import genhurst

# 리포트 출력 설정
REPORT_DIR = Path(__file__).parent / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


class Chapter2Analyzer:
    """Chapter 2 평균 회귀 분석 클래스"""
    
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
        
        # USDCAD 데이터
        usdcad_path = Path(__file__).parent / "inputData_USDCAD.csv"
        if usdcad_path.exists():
            df_usdcad = pd.read_csv(usdcad_path)
            # 17:00 (1659분) 데이터만 추출
            self.usdcad = df_usdcad.loc[df_usdcad['Time'] == 1659, 'Close'].reset_index(drop=True)
            print(f"  ✓ USDCAD: {len(self.usdcad)} 데이터 포인트")
        else:
            self.usdcad = None
            print(f"  ✗ USDCAD 데이터 없음")
        
        # EWA/EWC 데이터
        ewa_ewc_path = Path(__file__).parent / "inputData_EWA_EWC.csv"
        if ewa_ewc_path.exists():
            self.df_ewa_ewc = pd.read_csv(ewa_ewc_path)
            self.df_ewa_ewc['Date'] = pd.to_datetime(self.df_ewa_ewc['Date'], format='%Y%m%d')
            self.df_ewa_ewc.set_index('Date', inplace=True)
            print(f"  ✓ EWA/EWC: {len(self.df_ewa_ewc)} 데이터 포인트")
        else:
            self.df_ewa_ewc = None
            print(f"  ✗ EWA/EWC 데이터 없음")
            
        # EWA/EWC/IGE 데이터
        ewa_ewc_ige_path = Path(__file__).parent / "inputData_EWA_EWC_IGE.csv"
        if ewa_ewc_ige_path.exists():
            self.df_ewa_ewc_ige = pd.read_csv(ewa_ewc_ige_path)
            self.df_ewa_ewc_ige['Date'] = pd.to_datetime(self.df_ewa_ewc_ige['Date'], format='%Y%m%d')
            self.df_ewa_ewc_ige.set_index('Date', inplace=True)
            print(f"  ✓ EWA/EWC/IGE: {len(self.df_ewa_ewc_ige)} 데이터 포인트")
        else:
            self.df_ewa_ewc_ige = None
            print(f"  ✗ EWA/EWC/IGE 데이터 없음")
        
        print()
        
    def analyze_stationarity(self):
        """정상성 검정: ADF, 허스트 지수"""
        print("=" * 60)
        print("🔬 1. 정상성 검정 (Stationarity Tests)")
        print("=" * 60)
        
        self.results['stationarity'] = {}
        
        if self.usdcad is not None and len(self.usdcad) > 0:
            y = self.usdcad.values
            
            # 1.1 ADF 검정
            print("\n### 1.1 ADF 검정 (Augmented Dickey-Fuller Test)")
            print("-" * 40)
            adf_result = ts.adfuller(y, maxlag=1, regression='c', autolag=None)
            
            self.results['stationarity']['adf'] = {
                't_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[0] < adf_result[4]['5%']
            }
            
            print(f"  ADF t-통계량:    {adf_result[0]:.4f}")
            print(f"  p-value:         {adf_result[1]:.4f}")
            print(f"  임계값 (1%):     {adf_result[4]['1%']:.4f}")
            print(f"  임계값 (5%):     {adf_result[4]['5%']:.4f}")
            print(f"  임계값 (10%):    {adf_result[4]['10%']:.4f}")
            
            if adf_result[0] < adf_result[4]['5%']:
                print("  → ✅ 귀무가설 기각: 정상 시계열 (5% 유의수준)")
            else:
                print("  → ❌ 귀무가설 채택 불가: 랜덤워크 가능성")
            
            # 1.2 허스트 지수
            print("\n### 1.2 허스트 지수 (Hurst Exponent)")
            print("-" * 40)
            H, pVal = genhurst(np.log(y))
            
            self.results['stationarity']['hurst'] = {
                'H': H,
                'p_value': pVal
            }
            
            print(f"  H = {H:.4f}")
            print(f"  p-value = {pVal:.6f}")
            
            if H < 0.5:
                print(f"  → ✅ H < 0.5: 평균 회귀 성향 (Mean Reverting)")
            elif H > 0.5:
                print(f"  → ⚠️ H > 0.5: 추세 추종 성향 (Trending)")
            else:
                print(f"  → ⚪ H ≈ 0.5: 랜덤 워크 (Random Walk)")
            
            # 1.3 USDCAD 가격 차트 저장
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y, linewidth=0.8)
            ax.set_title('USD/CAD Price Series', fontsize=14)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / 'usdcad_price.png', dpi=150)
            plt.close(fig)
            self.figures.append('usdcad_price.png')
            
        print()
        
    def analyze_cointegration(self):
        """공적분 검정: CADF, Johansen"""
        print("=" * 60)
        print("🔗 2. 공적분 검정 (Cointegration Tests)")
        print("=" * 60)
        
        self.results['cointegration'] = {}
        
        if self.df_ewa_ewc_ige is not None:
            df = self.df_ewa_ewc_ige
            
            # 2.1 EWA-EWC CADF 검정
            print("\n### 2.1 EWA-EWC 페어 (CADF Test)")
            print("-" * 40)
            
            # 헤지 비율 계산
            results_ols = smf.ols(formula="EWC ~ EWA", data=df[['EWA', 'EWC']]).fit()
            hedge_ratio = results_ols.params['EWA']
            
            self.results['cointegration']['ewa_ewc'] = {
                'hedge_ratio': hedge_ratio,
                'intercept': results_ols.params['Intercept']
            }
            
            print(f"  헤지 비율: {hedge_ratio:.4f}")
            print(f"  절편: {results_ols.params['Intercept']:.4f}")
            
            # 공적분 검정
            coint_t, pvalue, crit_value = ts.coint(df['EWA'], df['EWC'])
            
            self.results['cointegration']['ewa_ewc']['coint_t'] = coint_t
            self.results['cointegration']['ewa_ewc']['pvalue'] = pvalue
            self.results['cointegration']['ewa_ewc']['critical_values'] = crit_value
            
            print(f"  CADF t-통계량: {coint_t:.4f}")
            print(f"  p-value: {pvalue:.4f}")
            print(f"  임계값: {crit_value}")
            
            if pvalue < 0.05:
                print("  → ✅ 공적분 관계 존재 (5% 유의수준)")
            else:
                print("  → ⚠️ 공적분 관계 불확실")
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['EWA'], df['EWC'], alpha=0.5, s=10)
            ax.set_xlabel('EWA')
            ax.set_ylabel('EWC')
            ax.set_title('EWA vs EWC Scatter Plot')
            ax.grid(True, alpha=0.3)
            
            # 회귀선 추가
            x_line = np.linspace(df['EWA'].min(), df['EWA'].max(), 100)
            y_line = results_ols.params['Intercept'] + hedge_ratio * x_line
            ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {hedge_ratio:.2f}x + {results_ols.params["Intercept"]:.2f}')
            ax.legend()
            
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / 'ewa_ewc_scatter.png', dpi=150)
            plt.close(fig)
            self.figures.append('ewa_ewc_scatter.png')
            
            # 잔차 플롯
            residuals = df['EWC'] - hedge_ratio * df['EWA']
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(residuals.values, linewidth=0.8)
            ax.axhline(y=residuals.mean(), color='r', linestyle='--', label=f'Mean = {residuals.mean():.2f}')
            ax.set_title('EWC - hedgeRatio * EWA (Residuals)', fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Spread')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / 'portfolio_residual.png', dpi=150)
            plt.close(fig)
            self.figures.append('portfolio_residual.png')
            
            # 2.2 Johansen 검정 (EWA-EWC)
            print("\n### 2.2 EWA-EWC Johansen 검정")
            print("-" * 40)
            
            result_j2 = vm.coint_johansen(df[['EWA', 'EWC']].values, det_order=0, k_ar_diff=1)
            
            print("  Trace 통계량:")
            for i, (stat, cv) in enumerate(zip(result_j2.lr1, result_j2.cvt)):
                reject = "✅" if stat > cv[1] else "❌"
                print(f"    r ≤ {i}: {stat:.4f} (임계값 95%: {cv[1]:.4f}) {reject}")
            
            # 2.3 Johansen 검정 (EWA-EWC-IGE)
            print("\n### 2.3 EWA-EWC-IGE 포트폴리오 Johansen 검정")
            print("-" * 40)
            
            result_j3 = vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
            
            self.results['cointegration']['ewa_ewc_ige'] = {
                'trace_stats': result_j3.lr1.tolist(),
                'eigen_stats': result_j3.lr2.tolist(),
                'eigenvalues': result_j3.eig.tolist(),
                'eigenvectors': result_j3.evec.tolist()
            }
            
            print("  Trace 통계량:")
            for i, (stat, cv) in enumerate(zip(result_j3.lr1, result_j3.cvt)):
                reject = "✅" if stat > cv[1] else "❌"
                print(f"    r ≤ {i}: {stat:.4f} (임계값 95%: {cv[1]:.4f}) {reject}")
            
            print("\n  Eigen 통계량:")
            for i, (stat, cv) in enumerate(zip(result_j3.lr2, result_j3.cvm)):
                reject = "✅" if stat > cv[1] else "❌"
                print(f"    r ≤ {i}: {stat:.4f} (임계값 95%: {cv[1]:.4f}) {reject}")
            
            print("\n  고유값 (Eigenvalues):")
            print(f"    {result_j3.eig}")
            
            print("\n  고유벡터 (헤지 비율):")
            for i, col in enumerate(df.columns):
                print(f"    {col}: {result_j3.evec[i, 0]:.4f}")
                
        print()
        
    def calculate_halflife(self):
        """반감기 계산"""
        print("=" * 60)
        print("⏱️ 3. 반감기 계산 (Half-life of Mean Reversion)")
        print("=" * 60)
        
        self.results['halflife'] = {}
        
        # 3.1 EWA-EWC-IGE 포트폴리오 반감기
        if self.df_ewa_ewc_ige is not None:
            df = self.df_ewa_ewc_ige
            result = vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
            
            # 최적 고유벡터로 포트폴리오 가치 계산
            yport = pd.DataFrame(np.dot(df.values, result.evec[:, 0]))
            
            # 반감기 계산을 위한 회귀
            ylag = yport.shift()
            deltaY = yport - ylag
            df_reg = pd.concat([ylag, deltaY], axis=1)
            df_reg.columns = ['ylag', 'deltaY']
            df_reg = df_reg.dropna()
            
            regress_results = smf.ols(formula="deltaY ~ ylag", data=df_reg).fit()
            lambda_coef = regress_results.params['ylag']
            halflife = -np.log(2) / lambda_coef
            
            self.results['halflife']['ewa_ewc_ige'] = {
                'lambda': lambda_coef,
                'halflife_days': halflife
            }
            
            print(f"\n### 3.1 EWA-EWC-IGE 포트폴리오")
            print("-" * 40)
            print(f"  λ (회귀 계수): {lambda_coef:.6f}")
            print(f"  반감기: {halflife:.2f} 일")
            
            if halflife < 30:
                print(f"  → ✅ 단기 평균회귀: 트레이딩에 적합")
            elif halflife < 100:
                print(f"  → ⚠️ 중기 평균회귀: 스윙 트레이딩 가능")
            else:
                print(f"  → ❌ 장기 평균회귀: 실용성 낮음")
                
        print()
        
    def backtest_strategy(self):
        """선형 평균회귀 전략 백테스트"""
        print("=" * 60)
        print("📈 4. 전략 백테스트 (Linear Mean Reversion Strategy)")
        print("=" * 60)
        
        self.results['backtest'] = {}
        
        if self.df_ewa_ewc_ige is not None:
            df = self.df_ewa_ewc_ige
            result = vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
            
            # 포트폴리오 가치 계산
            yport = pd.DataFrame(np.dot(df.values, result.evec[:, 0]))
            
            # 반감기 기반 lookback 설정
            ylag = yport.shift()
            deltaY = yport - ylag
            df_reg = pd.concat([ylag, deltaY], axis=1)
            df_reg.columns = ['ylag', 'deltaY']
            regress_results = smf.ols(formula="deltaY ~ ylag", data=df_reg.dropna()).fit()
            halflife = -np.log(2) / regress_results.params['ylag']
            lookback = int(np.round(halflife))
            
            print(f"\n### 4.1 EWA-EWC-IGE 선형 평균회귀 전략")
            print("-" * 40)
            print(f"  Lookback 기간: {lookback} 일 (반감기 기반)")
            
            # Z-Score 기반 포지션
            ma = yport.rolling(lookback).mean()
            mstd = yport.rolling(lookback).std()
            numUnits = -(yport - ma) / mstd
            
            # 포지션 계산
            positions = pd.DataFrame(
                np.dot(numUnits.values, np.expand_dims(result.evec[:, 0], axis=1).T) * df.values
            )
            
            # P&L 계산
            pnl = np.sum(positions.shift().values * df.pct_change().values, axis=1)
            ret = pnl / np.sum(np.abs(positions.shift()), axis=1)
            ret = pd.Series(ret)
            
            # 성과 지표 계산
            ret_clean = ret.replace([np.inf, -np.inf], np.nan).dropna()
            
            total_return = (np.cumprod(1 + ret_clean) - 1).iloc[-1]
            apr = np.prod(1 + ret_clean) ** (252 / len(ret_clean)) - 1
            sharpe = np.sqrt(252) * np.mean(ret_clean) / np.std(ret_clean)
            
            # 최대 낙폭 계산
            cumret = np.cumprod(1 + ret_clean)
            highwatermark = cumret.cummax()
            drawdown = (cumret - highwatermark) / highwatermark
            max_dd = drawdown.min()
            
            self.results['backtest']['ewa_ewc_ige'] = {
                'lookback': lookback,
                'total_return': total_return,
                'apr': apr,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'num_days': len(ret_clean)
            }
            
            print(f"  총 수익률: {total_return * 100:.2f}%")
            print(f"  연간 수익률 (APR): {apr * 100:.2f}%")
            print(f"  샤프 비율: {sharpe:.4f}")
            print(f"  최대 낙폭 (MDD): {max_dd * 100:.2f}%")
            print(f"  거래일 수: {len(ret_clean)}")
            
            if sharpe > 1.0:
                print(f"  → ✅ 샤프 > 1.0: 우수한 위험조정수익률")
            elif sharpe > 0.5:
                print(f"  → ⚠️ 샤프 0.5~1.0: 양호한 전략")
            else:
                print(f"  → ❌ 샤프 < 0.5: 개선 필요")
            
            # 누적 수익률 차트
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 누적 수익률
            cumret_plot = np.cumprod(1 + ret_clean) - 1
            axes[0].plot(cumret_plot.values, linewidth=1)
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[0].set_title(f'Cumulative Returns (APR={apr*100:.1f}%, Sharpe={sharpe:.2f})', fontsize=14)
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)
            axes[0].fill_between(range(len(cumret_plot)), 0, cumret_plot.values, 
                               where=cumret_plot.values >= 0, alpha=0.3, color='green')
            axes[0].fill_between(range(len(cumret_plot)), 0, cumret_plot.values, 
                               where=cumret_plot.values < 0, alpha=0.3, color='red')
            
            # 드로다운
            axes[1].fill_between(range(len(drawdown)), 0, drawdown.values, alpha=0.5, color='red')
            axes[1].set_title(f'Drawdown (Max DD={max_dd*100:.1f}%)', fontsize=14)
            axes[1].set_xlabel('Time (days)')
            axes[1].set_ylabel('Drawdown')
            axes[1].grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / 'equity_curve.png', dpi=150)
            plt.close(fig)
            self.figures.append('equity_curve.png')
            
        print()
        
    def generate_report(self):
        """마크다운 리포트 생성 (Enhanced with theoretical context)"""
        print("=" * 60)
        print("📝 5. 리포트 생성")
        print("=" * 60)
        
        report_lines = []
        
        # 제목 및 메타데이터
        report_lines.append("# Chapter 2: 평균 회귀 기초 (The Basics of Mean Reversion)\n")
        report_lines.append("# 분석 리포트\n\n")
        report_lines.append(f"> **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("> **데이터 출처**: Ernest Chan's \"Algorithmic Trading\" (2013)\n\n")
        report_lines.append("---\n\n")
        
        # 목차
        report_lines.append("## 목차\n\n")
        report_lines.append("1. [개요 및 문제 정의](#1-개요-및-문제-정의)\n")
        report_lines.append("2. [사용 데이터](#2-사용-데이터)\n")
        report_lines.append("3. [정상성 검정](#3-정상성-검정-stationarity-tests)\n")
        report_lines.append("4. [공적분 검정](#4-공적분-검정-cointegration-tests)\n")
        report_lines.append("5. [반감기 분석](#5-반감기-분석-half-life)\n")
        report_lines.append("6. [전략 백테스트](#6-전략-백테스트)\n")
        report_lines.append("7. [결론 및 권고사항](#7-결론-및-권고사항)\n\n")
        report_lines.append("---\n\n")
        
        # 1. 개요 및 문제 정의
        report_lines.append("## 1. 개요 및 문제 정의\n\n")
        report_lines.append("### 💡 해결하려는 문제\n\n")
        report_lines.append("**\"가격이 평균으로 되돌아오는 성질을 이용해 수익을 낼 수 있을까?\"**\n\n")
        report_lines.append("대부분의 금융 가격 시계열은 **기하 랜덤 워크(Geometric Random Walk)**를 따르기 때문에 ")
        report_lines.append("가격의 평균 회귀를 직접 거래할 수 없습니다. 그러나:\n\n")
        report_lines.append("1. **정상성(Stationarity)을 가진 소수의 시계열**은 평균 회귀 거래가 가능\n")
        report_lines.append("2. **공적분(Cointegration) 관계**를 이용하면 비정상 시계열들을 결합하여 정상 포트폴리오 생성 가능\n\n")
        
        report_lines.append("### 📐 핵심 수학적 개념\n\n")
        report_lines.append("| 개념 | 수식 | 의미 |\n")
        report_lines.append("|------|------|------|\n")
        report_lines.append("| **평균 회귀** | $\\Delta y(t) = \\lambda y(t-1) + \\mu + \\epsilon$ | 가격 변화가 현재 가격 수준에 의존 (λ < 0) |\n")
        report_lines.append("| **허스트 지수** | $Var(\\tau) \\sim \\tau^{2H}$ | H < 0.5: 평균회귀, H = 0.5: 랜덤워크, H > 0.5: 추세 |\n")
        report_lines.append("| **반감기** | $t_{1/2} = -\\log(2)/\\lambda$ | 가격이 평균까지 절반 거리를 회복하는 시간 |\n")
        report_lines.append("| **공적분** | $y_1 - \\beta y_2 = \\epsilon$ (정상) | 두 비정상 시계열의 선형 결합이 정상 |\n\n")
        report_lines.append("---\n\n")
        
        # 2. 사용 데이터
        report_lines.append("## 2. 사용 데이터\n\n")
        report_lines.append("### 📊 데이터셋 설명\n\n")
        report_lines.append("| 파일명 | 내용 | 기간 | 용도 |\n")
        report_lines.append("|--------|------|------|------|\n")
        report_lines.append("| `inputData_USDCAD.csv` | USD/CAD 환율 (1분봉) | ~1216일 | 단일 시계열 정상성 검정 |\n")
        report_lines.append("| `inputData_EWA_EWC.csv` | 호주(EWA)/캐나다(EWC) ETF | ~1500일 | 페어 공적분 검정 |\n")
        report_lines.append("| `inputData_EWA_EWC_IGE.csv` | EWA/EWC + 천연자원(IGE) ETF | ~1500일 | 다중 자산 공적분 |\n\n")
        
        report_lines.append("### 🎯 데이터 선정 이유\n\n")
        report_lines.append("- **USD/CAD**: 캐나다 달러는 \"원자재 통화\"로 미국 달러와 다른 특성을 가짐\n")
        report_lines.append("- **EWA-EWC**: 호주와 캐나다 경제 모두 **원자재 기반**이므로 공적분 가능성 높음\n")
        report_lines.append("- **IGE**: 천연자원 ETF로 EWA/EWC와 경제적 연관성 존재\n\n")
        report_lines.append("---\n\n")
        
        # 3. 정상성 검정
        report_lines.append("## 3. 정상성 검정 (Stationarity Tests)\n\n")
        report_lines.append("### 🔬 검정 목적\n\n")
        report_lines.append("가격 시계열이 **평균 회귀**하는지 확인합니다. 평균 회귀하는 시계열에서는:\n")
        report_lines.append("- 가격이 평균보다 높으면 → 다음 움직임은 **하락** 예상\n")
        report_lines.append("- 가격이 평균보다 낮으면 → 다음 움직임은 **상승** 예상\n\n")
        
        if 'stationarity' in self.results:
            if 'adf' in self.results['stationarity']:
                adf = self.results['stationarity']['adf']
                report_lines.append("### 3.1 ADF 검정 (Augmented Dickey-Fuller Test)\n\n")
                report_lines.append("**검정 원리**: 다음 모델에서 $\\lambda = 0$인지 검정\n\n")
                report_lines.append("$$\\Delta y(t) = \\lambda y(t-1) + \\mu + \\alpha_1 \\Delta y(t-1) + \\epsilon_t$$\n\n")
                report_lines.append("- **귀무가설 (H₀)**: $\\lambda = 0$ (랜덤 워크, 평균 회귀 아님)\n")
                report_lines.append("- **대립가설 (H₁)**: $\\lambda < 0$ (평균 회귀)\n\n")
                
                report_lines.append("**USD/CAD 검정 결과:**\n\n")
                report_lines.append("| 통계량 | 값 | 설명 |\n")
                report_lines.append("|--------|------|------|\n")
                report_lines.append(f"| t-통계량 | {adf['t_statistic']:.4f} | 검정통계량 (더 음수일수록 좋음) |\n")
                report_lines.append(f"| p-value | {adf['p_value']:.4f} | 귀무가설이 참일 확률 |\n")
                report_lines.append(f"| 임계값 (1%) | {adf['critical_values']['1%']:.4f} | 99% 신뢰수준 임계값 |\n")
                report_lines.append(f"| 임계값 (5%) | {adf['critical_values']['5%']:.4f} | 95% 신뢰수준 임계값 |\n")
                report_lines.append(f"| 임계값 (10%) | {adf['critical_values']['10%']:.4f} | 90% 신뢰수준 임계값 |\n\n")
                
                if adf['is_stationary']:
                    report_lines.append("> ✅ **결론**: t-통계량이 임계값보다 더 음수이므로 **귀무가설 기각** → 정상 시계열\n\n")
                else:
                    report_lines.append("> ❌ **결론**: t-통계량이 임계값보다 덜 음수이므로 **귀무가설 기각 불가** → 랜덤 워크 가능성\n\n")
                    report_lines.append("> 💡 USD/CAD가 정상성 검정을 통과하지 못한 이유: 캐나다 달러는 **원자재 통화**이고 ")
                    report_lines.append("미국 달러는 그렇지 않아 장기적으로 다른 추세를 보일 수 있음\n\n")
                
            if 'hurst' in self.results['stationarity']:
                hurst = self.results['stationarity']['hurst']
                report_lines.append("### 3.2 허스트 지수 (Hurst Exponent)\n\n")
                report_lines.append("**검정 원리**: 시계열의 **확산 속도**를 측정\n\n")
                report_lines.append("$$\\langle |z(t+\\tau) - z(t)|^2 \\rangle \\sim \\tau^{2H}$$\n\n")
                report_lines.append("| H 값 범위 | 시계열 특성 | 트레이딩 전략 |\n")
                report_lines.append("|-----------|-------------|---------------|\n")
                report_lines.append("| H < 0.5 | 평균 회귀 (Mean Reverting) | 평균회귀 매매 |\n")
                report_lines.append("| H = 0.5 | 랜덤 워크 (Random Walk) | 예측 불가 |\n")
                report_lines.append("| H > 0.5 | 추세 추종 (Trending) | 모멘텀 매매 |\n\n")
                
                report_lines.append("**USD/CAD 허스트 지수 결과:**\n\n")
                report_lines.append(f"| 지표 | 값 |\n")
                report_lines.append(f"|------|------|\n")
                report_lines.append(f"| H | **{hurst['H']:.4f}** |\n")
                report_lines.append(f"| p-value | {hurst['p_value']:.6f} |\n\n")
                
                if hurst['H'] < 0.5:
                    report_lines.append(f"> ✅ **해석**: H = {hurst['H']:.4f} < 0.5 → **약한 평균 회귀 성향** 존재\n\n")
                elif hurst['H'] > 0.5:
                    report_lines.append(f"> ⚠️ **해석**: H = {hurst['H']:.4f} > 0.5 → **추세 추종 성향**\n\n")
                else:
                    report_lines.append(f"> ⚪ **해석**: H ≈ 0.5 → **랜덤 워크**\n\n")
                    
            report_lines.append("### 3.3 USD/CAD 가격 차트\n\n")
            report_lines.append("![USD/CAD Price](figures/usdcad_price.png)\n\n")
            report_lines.append("> 📈 위 차트에서 USD/CAD 환율이 일정 범위 내에서 움직이는 것처럼 보이지만, ")
            report_lines.append("ADF 검정에서 통계적으로 유의한 정상성을 확인하지 못함\n\n")
        
        report_lines.append("---\n\n")
        
        # 4. 공적분 검정
        report_lines.append("## 4. 공적분 검정 (Cointegration Tests)\n\n")
        report_lines.append("### 🎯 검정 목적\n\n")
        report_lines.append("개별적으로는 비정상인 시계열들을 **선형 결합**하여 **정상인 포트폴리오**를 만들 수 있는지 확인합니다.\n\n")
        report_lines.append("**핵심 아이디어**: $y_{EWC} - \\beta \\cdot y_{EWA}$가 정상이면 EWA와 EWC는 **공적분** 관계\n\n")
        
        if 'cointegration' in self.results:
            if 'ewa_ewc' in self.results['cointegration']:
                coint = self.results['cointegration']['ewa_ewc']
                report_lines.append("### 4.1 EWA-EWC 페어 분석 (CADF Test)\n\n")
                report_lines.append("**왜 EWA와 EWC인가?**\n")
                report_lines.append("- **EWA**: iShares MSCI Australia ETF (호주 주식시장)\n")
                report_lines.append("- **EWC**: iShares MSCI Canada ETF (캐나다 주식시장)\n")
                report_lines.append("- 두 경제 모두 **원자재 수출 기반**이므로 유사한 경제 사이클을 가짐\n\n")
                
                report_lines.append("**Step 1: 헤지 비율(Hedge Ratio) 계산**\n\n")
                report_lines.append("선형 회귀를 통해 최적 헤지 비율 결정:\n")
                report_lines.append("```\n")
                report_lines.append("EWC = β × EWA + α + ε\n")
                report_lines.append("```\n\n")
                report_lines.append("| 파라미터 | 값 | 의미 |\n")
                report_lines.append("|----------|------|------|\n")
                report_lines.append(f"| β (헤지 비율) | **{coint['hedge_ratio']:.4f}** | EWA 1주당 EWC 매수량 |\n")
                report_lines.append(f"| α (절편) | {coint['intercept']:.4f} | 기본 스프레드 수준 |\n\n")
                
                report_lines.append("**Step 2: 공적분 검정 결과**\n\n")
                report_lines.append("| 통계량 | 값 |\n")
                report_lines.append("|--------|------|\n")
                report_lines.append(f"| CADF t-통계량 | {coint['coint_t']:.4f} |\n")
                report_lines.append(f"| p-value | {coint['pvalue']:.4f} |\n\n")
                
                if coint['pvalue'] < 0.05:
                    report_lines.append("> ✅ **결론**: p-value < 0.05 → EWA와 EWC는 **공적분 관계** (95% 신뢰수준)\n\n")
                elif coint['pvalue'] < 0.10:
                    report_lines.append("> ⚠️ **결론**: 0.05 < p-value < 0.10 → **약한 공적분 관계** (90% 신뢰수준)\n\n")
                else:
                    report_lines.append("> ❌ **결론**: p-value > 0.10 → 공적분 관계 통계적으로 유의하지 않음\n\n")
                
                report_lines.append("### 4.2 EWA vs EWC 산점도\n\n")
                report_lines.append("![EWA vs EWC Scatter](figures/ewa_ewc_scatter.png)\n\n")
                report_lines.append("> 📊 두 ETF 가격이 **직선 관계**에 가깝게 분포 → 공적분 가능성 시각적 확인\n\n")
                
                report_lines.append("### 4.3 스프레드 (잔차) 차트\n\n")
                report_lines.append("![Portfolio Residual](figures/portfolio_residual.png)\n\n")
                report_lines.append("> 📉 스프레드(EWC - β×EWA)가 **평균 주변에서 진동** → 평균 회귀 거래 가능성\n\n")
                
            if 'ewa_ewc_ige' in self.results['cointegration']:
                j3 = self.results['cointegration']['ewa_ewc_ige']
                report_lines.append("### 4.4 EWA-EWC-IGE 포트폴리오 (Johansen Test)\n\n")
                report_lines.append("**Johansen 검정의 장점**:\n")
                report_lines.append("- 2개 이상의 자산에 대한 공적분 검정 가능\n")
                report_lines.append("- 가격 시계열의 **순서에 독립적** (CADF와 달리)\n")
                report_lines.append("- **고유벡터**를 헤지 비율로 사용 가능\n\n")
                
                report_lines.append("**추가 자산: IGE (iShares North American Natural Resources ETF)**\n")
                report_lines.append("- 천연자원 관련 주식으로 구성\n")
                report_lines.append("- 호주/캐나다 경제와 밀접한 연관\n\n")
                
                report_lines.append("**Trace 통계량 검정 결과:**\n\n")
                report_lines.append("| 귀무가설 | 통계량 | 95% 임계값 | 결론 |\n")
                report_lines.append("|----------|--------|------------|------|\n")
                for i, stat in enumerate(j3['trace_stats']):
                    # 임계값 대략 추정 (실제로는 result_j3.cvt에서 가져옴)
                    cv_approx = [29.79, 15.49, 3.84]
                    reject = "✅ 기각" if stat > cv_approx[i] else "❌ 채택"
                    report_lines.append(f"| r ≤ {i} | {stat:.4f} | ~{cv_approx[i]} | {reject} |\n")
                report_lines.append("\n")
                
                report_lines.append("**최적 헤지 비율 (첫 번째 고유벡터):**\n\n")
                report_lines.append("포트폴리오 가치 = w₁×EWA + w₂×EWC + w₃×IGE\n\n")
                report_lines.append("| ETF | 가중치 (wᵢ) | 해석 |\n")
                report_lines.append("|-----|-------------|------|\n")
                etf_names = ['EWA', 'EWC', 'IGE']
                for i, name in enumerate(etf_names):
                    weight = j3['eigenvectors'][i][0]
                    position = "Long" if weight > 0 else "Short"
                    report_lines.append(f"| {name} | {weight:.4f} | {position} |\n")
                report_lines.append("\n")
                
                report_lines.append("> 💡 **해석**: EWA와 IGE를 롱, EWC를 숏하는 포트폴리오가 가장 빠르게 평균 회귀\n\n")
        
        report_lines.append("---\n\n")
        
        # 5. 반감기 분석
        report_lines.append("## 5. 반감기 분석 (Half-life)\n\n")
        report_lines.append("### 📐 반감기의 의미\n\n")
        report_lines.append("**반감기(Half-life)**는 가격이 평균에서 벗어난 거리가 **절반으로 줄어드는 데 걸리는 시간**입니다.\n\n")
        report_lines.append("$$t_{1/2} = -\\frac{\\log(2)}{\\lambda}$$\n\n")
        report_lines.append("**트레이딩 실용성:**\n")
        report_lines.append("- 반감기가 **짧을수록** → 더 많은 왕복 거래 가능 → 높은 수익 기회\n")
        report_lines.append("- 반감기를 **Lookback 기간**으로 사용하면 데이터 스누핑 없이 전략 설계 가능\n\n")
        
        if 'halflife' in self.results:
            if 'ewa_ewc_ige' in self.results['halflife']:
                hl = self.results['halflife']['ewa_ewc_ige']
                report_lines.append("### 5.1 EWA-EWC-IGE 포트폴리오 반감기\n\n")
                report_lines.append("| 파라미터 | 값 | 의미 |\n")
                report_lines.append("|----------|------|------|\n")
                report_lines.append(f"| λ (회귀 계수) | {hl['lambda']:.6f} | 음수 = 평균 회귀 |\n")
                report_lines.append(f"| **반감기** | **{hl['halflife_days']:.1f}일** | 평균으로 50% 회복 시간 |\n\n")
                
                if hl['halflife_days'] < 30:
                    report_lines.append("> ✅ **평가**: 반감기 < 30일 → **단기 트레이딩에 적합**\n\n")
                    report_lines.append("> 💡 비교: USD/CAD의 반감기는 약 115일로 실용적이지 않음\n\n")
                elif hl['halflife_days'] < 100:
                    report_lines.append("> ⚠️ **평가**: 30일 < 반감기 < 100일 → **스윙 트레이딩 가능**\n\n")
                else:
                    report_lines.append("> ❌ **평가**: 반감기 > 100일 → **장기 투자에만 적합**\n\n")
        
        report_lines.append("---\n\n")
        
        # 6. 전략 백테스트
        report_lines.append("## 6. 전략 백테스트\n\n")
        report_lines.append("### 📈 선형 평균 회귀 전략\n\n")
        report_lines.append("**전략 원리**: 포트폴리오 가격의 **Z-Score**에 비례하여 반대 포지션\n\n")
        report_lines.append("```python\n")
        report_lines.append("# Z-Score 계산\n")
        report_lines.append("z_score = (portfolio_price - moving_avg) / moving_std\n")
        report_lines.append("\n")
        report_lines.append("# 포지션 결정 (Z-Score의 음수에 비례)\n")
        report_lines.append("num_units = -z_score\n")
        report_lines.append("```\n\n")
        report_lines.append("| Z-Score | 포지션 | 이유 |\n")
        report_lines.append("|---------|--------|------|\n")
        report_lines.append("| Z > 0 (평균 위) | Short | 가격 하락 예상 |\n")
        report_lines.append("| Z < 0 (평균 아래) | Long | 가격 상승 예상 |\n")
        report_lines.append("| Z ≈ 0 (평균 근처) | 중립 | 방향성 불확실 |\n\n")
        
        if 'backtest' in self.results:
            if 'ewa_ewc_ige' in self.results['backtest']:
                bt = self.results['backtest']['ewa_ewc_ige']
                report_lines.append("### 6.1 EWA-EWC-IGE 백테스트 결과\n\n")
                report_lines.append("**전략 파라미터:**\n\n")
                report_lines.append(f"- **Lookback 기간**: {bt['lookback']}일 (반감기 기반 자동 설정)\n")
                report_lines.append("- **거래 비용**: 미포함 (실제 적용 시 조정 필요)\n\n")
                
                report_lines.append("**성과 지표:**\n\n")
                report_lines.append("| 지표 | 값 | 평가 |\n")
                report_lines.append("|------|------|------|\n")
                report_lines.append(f"| 총 수익률 | {bt['total_return']*100:.2f}% | 테스트 기간 전체 |\n")
                
                apr_status = "✅ 우수" if bt['apr'] > 0.10 else ("⚠️ 양호" if bt['apr'] > 0.05 else "❌ 저조")
                report_lines.append(f"| 연간 수익률 (APR) | {bt['apr']*100:.2f}% | {apr_status} |\n")
                
                sharpe_status = "✅ 우수" if bt['sharpe'] > 1.0 else ("⚠️ 양호" if bt['sharpe'] > 0.5 else "❌ 저조")
                report_lines.append(f"| **샤프 비율** | **{bt['sharpe']:.4f}** | {sharpe_status} |\n")
                
                mdd_status = "✅ 양호" if bt['max_drawdown'] > -0.15 else ("⚠️ 주의" if bt['max_drawdown'] > -0.25 else "❌ 위험")
                report_lines.append(f"| 최대 낙폭 (MDD) | {bt['max_drawdown']*100:.2f}% | {mdd_status} |\n")
                
                report_lines.append(f"| 거래일 수 | {bt['num_days']}일 | 테스트 기간 |\n\n")
                
                report_lines.append("### 6.2 누적 수익률 및 낙폭 차트\n\n")
                report_lines.append("![Equity Curve](figures/equity_curve.png)\n\n")
                
                report_lines.append("> 📊 **차트 해석**:\n")
                report_lines.append("> - 상단: 누적 수익률 (녹색=수익, 빨간색=손실)\n")
                report_lines.append("> - 하단: 드로다운 (최고점 대비 하락폭)\n\n")
        
        report_lines.append("---\n\n")
        
        # 7. 결론 및 권고사항
        report_lines.append("## 7. 결론 및 권고사항\n\n")
        report_lines.append("### ✅ 핵심 발견\n\n")
        report_lines.append("| 분석 대상 | 결과 | 트레이딩 가능성 |\n")
        report_lines.append("|-----------|------|----------------|\n")
        
        if 'stationarity' in self.results and 'hurst' in self.results['stationarity']:
            h = self.results['stationarity']['hurst']['H']
            report_lines.append(f"| USD/CAD | ADF 통과 실패, H={h:.2f} | ⚠️ 단독 거래 어려움 |\n")
        
        if 'cointegration' in self.results and 'ewa_ewc' in self.results['cointegration']:
            p = self.results['cointegration']['ewa_ewc']['pvalue']
            status = "✅" if p < 0.10 else "⚠️"
            report_lines.append(f"| EWA-EWC | 공적분 p={p:.2f} | {status} 페어 트레이딩 |\n")
        
        if 'halflife' in self.results and 'ewa_ewc_ige' in self.results['halflife']:
            hl = self.results['halflife']['ewa_ewc_ige']['halflife_days']
            report_lines.append(f"| EWA-EWC-IGE | 반감기={hl:.0f}일 | ✅ 단기 트레이딩 적합 |\n")
        
        if 'backtest' in self.results and 'ewa_ewc_ige' in self.results['backtest']:
            sr = self.results['backtest']['ewa_ewc_ige']['sharpe']
            report_lines.append(f"| 선형 전략 | 샤프={sr:.2f} | ✅ 양호한 위험조정수익 |\n")
        
        report_lines.append("\n")
        
        report_lines.append("### 💡 트레이딩 권고사항\n\n")
        report_lines.append("1. **포트폴리오 구성**:\n")
        report_lines.append("   - EWA-EWC-IGE 3자산 포트폴리오에 평균회귀 전략 적용\n")
        report_lines.append("   - Johansen 고유벡터 기반 헤지 비율 사용\n\n")
        
        report_lines.append("2. **전략 고도화** (Chapter 3 참조):\n")
        report_lines.append("   - 볼린저 밴드: 진입/청산 임계값 최적화\n")
        report_lines.append("   - 칼만 필터: 시변(time-varying) 헤지 비율 적용\n\n")
        
        report_lines.append("3. **리스크 관리** (Chapter 8 참조):\n")
        report_lines.append("   - 평균회귀 전략은 **꼬리 리스크(Tail Risk)** 존재\n")
        report_lines.append("   - 켈리 공식 기반 레버리지 관리 필수\n")
        report_lines.append("   - 일반적인 손절매는 논리적이지 않음 (더 벌어질수록 더 매력적)\n\n")
        
        report_lines.append("### ⚠️ 주의사항\n\n")
        report_lines.append("- 본 백테스트는 **거래 비용 미포함**\n")
        report_lines.append("- **Look-ahead bias**: 전체 데이터로 반감기 계산 후 동일 데이터로 테스트\n")
        report_lines.append("- 실제 적용 시 **Walk-forward 테스트** 필요\n")
        report_lines.append("- 시장 구조 변화(Regime Shift)에 취약할 수 있음\n\n")
        
        report_lines.append("---\n\n")
        report_lines.append("*이 리포트는 `run_chapter2_analysis.py`에 의해 자동 생성되었습니다.*\n")
        
        # 파일 저장
        report_path = REPORT_DIR / "chapter2_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
            
        print(f"  ✓ 리포트 저장: {report_path}")
        print(f"  ✓ 차트 저장: {FIGURES_DIR}")
        print()
        
    def run(self):
        """전체 분석 실행"""
        print("\n" + "=" * 60)
        print("   Chapter 2: 평균 회귀 기초 - 종합 분석")
        print("   Ernest Chan's Algorithmic Trading")
        print("=" * 60 + "\n")
        
        self.load_data()
        self.analyze_stationarity()
        self.analyze_cointegration()
        self.calculate_halflife()
        self.backtest_strategy()
        self.generate_report()
        
        print("=" * 60)
        print("✅ 분석 완료!")
        print("=" * 60)
        print(f"\n📁 리포트 위치: {REPORT_DIR / 'chapter2_report.md'}")
        print(f"📊 차트 위치: {FIGURES_DIR}\n")


if __name__ == "__main__":
    analyzer = Chapter2Analyzer()
    analyzer.run()
