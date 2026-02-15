#!/usr/bin/env python3
"""
Chapter 5: 통화와 선물의 평균 회귀 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 5의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. AUD/CAD 페어 트레이딩 (요한센 고유벡터) - 예제 5.1
2. AUD/CAD 롤오버 이자 포함 전략 - 예제 5.2
3. 선물 스팟/롤 수익률 추정 - 예제 5.3
4. CL 캘린더 스프레드 평균 회귀 - 예제 5.4
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
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts_stats
import statsmodels.tsa.vector_ar.vecm as vm

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# 리포트 출력 설정
REPORT_DIR = Path(__file__).parent / "reports"
FIGURES_DIR = REPORT_DIR / "figures"


def calculateMaxDD(cumret):
    """최대 낙폭 계산"""
    vals = cumret.values if hasattr(cumret, 'values') else cumret
    highwatermark = np.zeros(len(vals))
    drawdown = np.zeros(len(vals))
    drawdownduration = np.zeros(len(vals))

    for t in range(1, len(vals)):
        highwatermark[t] = max(highwatermark[t-1], vals[t])
        drawdown[t] = (1 + vals[t]) / (1 + highwatermark[t]) - 1
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1

    maxDD = np.min(drawdown)
    i = np.argmin(drawdown)
    maxDDD = np.max(drawdownduration)
    return maxDD, maxDDD, i


class Chapter5Analyzer:
    """Chapter 5 통화와 선물의 평균 회귀 분석 클래스"""

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

        # AUD/USD 데이터
        audusd_path = data_dir / "inputData_AUDUSD_20120426.csv"
        self.audusd = pd.read_csv(audusd_path)
        self.audusd['Date'] = pd.to_datetime(self.audusd['Date'], format='%Y%m%d')
        self.audusd.set_index('Date', inplace=True)
        print(f"  ✓ AUD/USD: {len(self.audusd)} 데이터 포인트")

        # USD/CAD 데이터
        usdcad_path = data_dir / "inputData_USDCAD_20120426.csv"
        self.usdcad = pd.read_csv(usdcad_path)
        self.usdcad['Date'] = pd.to_datetime(self.usdcad['Date'], format='%Y%m%d')
        self.usdcad.set_index('Date', inplace=True)
        print(f"  ✓ USD/CAD: {len(self.usdcad)} 데이터 포인트")

        # AUD/CAD 데이터
        audcad_path = data_dir / "inputData_AUDCAD_20120426.csv"
        self.audcad = pd.read_csv(audcad_path)
        self.audcad['Date'] = pd.to_datetime(self.audcad['Date'], format='%Y%m%d')
        self.audcad.set_index('Date', inplace=True)
        print(f"  ✓ AUD/CAD: {len(self.audcad)} 데이터 포인트")

        # 금리 데이터 로드
        aud_rate_path = data_dir / "AUD_interestRate.csv"
        cad_rate_path = data_dir / "CAD_interestRate.csv"

        self.aud_rate = pd.read_csv(aud_rate_path)
        self.cad_rate = pd.read_csv(cad_rate_path)
        print(f"  ✓ AUD 금리: {len(self.aud_rate)} 월")
        print(f"  ✓ CAD 금리: {len(self.cad_rate)} 월")

        # 옥수수 선물 데이터
        corn_path = data_dir / "inputDataDaily_C2_20120813.csv"
        self.corn = pd.read_csv(corn_path)
        self.corn['Date'] = pd.to_datetime(self.corn['Date'], format='%Y%m%d')
        self.corn.set_index('Date', inplace=True)
        print(f"  ✓ 옥수수 선물: {len(self.corn)} 거래일 x {len(self.corn.columns)} 컬럼")

        # WTI 원유 선물 데이터
        cl_path = data_dir / "inputDataDaily_CL_20120502.csv"
        self.cl = pd.read_csv(cl_path)
        self.cl['Date'] = pd.to_datetime(self.cl['Date'], format='%Y%m%d')
        self.cl.set_index('Date', inplace=True)
        print(f"  ✓ WTI 원유 선물: {len(self.cl)} 거래일 x {len(self.cl.columns)} 컬럼")
        print()

    def analyze_audcad_unequal(self):
        """예제 5.1: AUD/USD vs CAD/USD 페어 트레이딩 (요한센 고유벡터)

        두 통화 모두 USD를 공통 호가 통화로 사용하여 요한센 검정 수행.
        동적 헤지 비율로 선형 평균 회귀 전략 적용.
        책 결과: APR=6.45%, Sharpe=1.36
        """
        print("=" * 60)
        print("📈 분석 1: AUD/USD vs CAD/USD 페어 트레이딩 (예제 5.1)")
        print("=" * 60)

        # CAD/USD로 변환 (USD/CAD의 역수) - 원본과 동일
        df1 = self.usdcad.copy()
        df1.columns = ['CAD']
        df1['CAD'] = 1.0 / df1['CAD']

        df2 = self.audusd.copy()
        df2.columns = ['AUD']

        df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
        print(f"  공통 거래일: {len(df)}")

        trainlen = 250
        lookback = 20

        hedgeRatio = np.full(df.shape, np.nan)
        numUnits = np.full(df.shape[0], np.nan)

        # 원본 로직: 루프 내에서 요한센 + z-score 계산
        for t in range(trainlen + 1, df.shape[0]):
            result = vm.coint_johansen(df.values[(t - trainlen - 1):t - 1], det_order=0, k_ar_diff=1)
            hedgeRatio[t, :] = result.evec[:, 0]
            yport = np.dot(df.values[(t - lookback):t], result.evec[:, 0])
            ma = np.mean(yport)
            mstd = np.std(yport, ddof=1)
            if mstd > 0:
                numUnits[t] = -(yport[-1] - ma) / mstd

        # 포지션 계산 - 원본 로직 (순수 numpy)
        pos_arr = np.expand_dims(numUnits, axis=1) * hedgeRatio * df.values
        pos_shift = np.full_like(pos_arr, np.nan)
        pos_shift[1:] = pos_arr[:-1]
        pnl = np.nansum(pos_shift * df.pct_change().values, axis=1)
        denom = np.nansum(np.abs(pos_shift), axis=1)
        denom[denom == 0] = np.nan
        ret_arr = pnl / denom

        # NaN/inf 정리
        valid_mask = np.isfinite(ret_arr)
        ret = pd.Series(ret_arr[valid_mask], index=df.index[valid_mask])
        cumret = (1 + ret).cumprod() - 1

        apr = np.prod(1 + ret.values) ** (252 / len(ret)) - 1 if len(ret) > 0 else 0
        sharpe = np.sqrt(252) * np.mean(ret.values) / np.std(ret.values) if len(ret) > 0 and np.std(ret.values) > 0 else 0
        maxDD, maxDDD, _ = calculateMaxDD(cumret)

        self.results['audcad_unequal'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
        }

        print(f"  APR = {apr:.4f} ({apr*100:.2f}%)")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD:.4f} ({maxDD*100:.2f}%)")
        print(f"  책 기대값: APR=6.45%, Sharpe=1.36")

        # 차트 생성
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('AUD/USD vs CAD/USD Pair Trading (Johansen) - Cumulative Returns', fontsize=13)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df.index, df['AUD'].values, label='AUD/USD', linewidth=0.8)
        axes[1].plot(df.index, df['CAD'].values, label='CAD/USD', linewidth=0.8)
        axes[1].set_title('AUD/USD and CAD/USD Prices', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch5_audcad_unequal.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch5_audcad_unequal.png', 'AUD-CAD 요한센 페어 트레이딩'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_audcad_daily(self):
        """예제 5.2: AUD/CAD 롤오버 이자 포함 전략

        AUD/CAD 크로스레이트에 단순 선형 평균 회귀 적용.
        롤오버 이자 포함/미포함 성과 비교.
        """
        print("=" * 60)
        print("📈 분석 2: AUD/CAD 롤오버 이자 전략 (예제 5.2)")
        print("=" * 60)

        df = self.audcad.copy()
        df.columns = ['Close']

        # 금리 데이터를 날짜 인덱스로 변환
        def rate_to_daily(rate_df, currency):
            rate_df = rate_df.copy()
            rate_df['Date'] = pd.to_datetime(
                rate_df['Year'].astype(str) + rate_df['Month'].astype(str).str.zfill(2) + '01',
                format='%Y%m%d'
            )
            rate_df.set_index('Date', inplace=True)
            rate_df = rate_df[['Rates']]
            rate_df.columns = [f'{currency}_rate']
            return rate_df

        aud_rate = rate_to_daily(self.aud_rate, 'AUD')
        cad_rate = rate_to_daily(self.cad_rate, 'CAD')

        # 일일 데이터로 forward-fill
        df = df.join(aud_rate, how='left')
        df = df.join(cad_rate, how='left')
        df['AUD_rate'] = df['AUD_rate'].ffill()
        df['CAD_rate'] = df['CAD_rate'].ffill()

        # 연율 → 일율 변환 (% → 비율, / 365)
        df['AUD_daily'] = df['AUD_rate'] / 100 / 365
        df['CAD_daily'] = df['CAD_rate'] / 100 / 365

        # 요일 정보
        df['DayOfWeek'] = df.index.dayofweek  # 0=Monday, 2=Wednesday, 3=Thursday

        # 3x 롤오버: AUD는 수요일(T+2), CAD는 목요일(T+1)
        df['AUD_rollover'] = df['AUD_daily']
        df.loc[df['DayOfWeek'] == 2, 'AUD_rollover'] = df['AUD_daily'] * 3  # 수요일
        df['CAD_rollover'] = df['CAD_daily']
        df.loc[df['DayOfWeek'] == 3, 'CAD_rollover'] = df['CAD_daily'] * 3  # 목요일

        lookback = 20

        # z-score
        ma = df['Close'].rolling(lookback).mean()
        mstd = df['Close'].rolling(lookback).std()
        zScore = (df['Close'] - ma) / mstd

        # 포지션: -sign(zScore)
        numUnits = -np.sign(zScore)
        numUnits = numUnits.fillna(0)

        # === 롤오버 이자 포함 수익률 ===
        log_close = np.log(df['Close'])
        ret_with_rollover = numUnits.shift(1) * (
            log_close.diff() +
            np.log(1 + df['AUD_rollover']) - np.log(1 + df['CAD_rollover'])
        )
        ret_with_rollover = ret_with_rollover.iloc[lookback:].fillna(0)

        cumret_with = (1 + ret_with_rollover).cumprod() - 1
        apr_with = (1 + ret_with_rollover).prod() ** (252/len(ret_with_rollover)) - 1
        sharpe_with = np.sqrt(252) * ret_with_rollover.mean() / ret_with_rollover.std() if ret_with_rollover.std() > 0 else 0
        maxDD_with, maxDDD_with, _ = calculateMaxDD(cumret_with)

        # === 롤오버 이자 미포함 수익률 ===
        ret_without = numUnits.shift(1) * log_close.diff()
        ret_without = ret_without.iloc[lookback:].fillna(0)

        cumret_without = (1 + ret_without).cumprod() - 1
        apr_without = (1 + ret_without).prod() ** (252/len(ret_without)) - 1
        sharpe_without = np.sqrt(252) * ret_without.mean() / ret_without.std() if ret_without.std() > 0 else 0

        # 평균 롤오버 이자율 차이
        avg_rate_diff = (df['AUD_rate'] - df['CAD_rate']).mean()

        self.results['audcad_daily'] = {
            'with_rollover': {'apr': apr_with, 'sharpe': sharpe_with, 'maxDD': maxDD_with, 'maxDDD': maxDDD_with},
            'without_rollover': {'apr': apr_without, 'sharpe': sharpe_without},
            'avg_rate_diff': avg_rate_diff,
        }

        print(f"  [롤오버 이자 포함]")
        print(f"    APR = {apr_with*100:.2f}%, Sharpe = {sharpe_with:.4f}")
        print(f"  [롤오버 이자 미포함]")
        print(f"    APR = {apr_without*100:.2f}%, Sharpe = {sharpe_without:.4f}")
        print(f"  평균 금리차 (AUD - CAD) = {avg_rate_diff:.2f}% 연율")

        # 차트
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(cumret_with.index, cumret_with.values * 100, 'b-', linewidth=1, label='With Rollover')
        ax.plot(cumret_without.index, cumret_without.values * 100, 'r--', linewidth=1, label='Without Rollover')
        ax.set_title('AUD/CAD Mean Reversion with/without Rollover Interest', fontsize=13)
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch5_audcad_rollover.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch5_audcad_rollover.png', 'AUD/CAD 롤오버 이자 비교'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_futures_returns(self):
        """예제 5.3: 옥수수 선물의 스팟/롤 수익률 추정

        상수 수익률 모델로 스팟 수익률(alpha)과 롤 수익률(gamma) 추정.
        원본: 가장 가까운 5개 연속 컬럼의 순차 인덱스로 회귀 (실제 만기월 아님).
        책 결과: alpha=+2.8%, gamma=-12.8%
        """
        print("=" * 60)
        print("📈 분석 3: 선물 스팟/롤 수익률 추정 (예제 5.3)")
        print("=" * 60)

        # 옥수수 선물 분석 - 원본 로직 그대로
        corn = self.corn.copy()

        # 스팟 컬럼 분리
        spot = corn['C_Spot']
        df_futures = corn.drop('C_Spot', axis=1)

        # === 스팟 수익률 (alpha) ===
        T = sm.add_constant(range(spot.shape[0]))
        model = sm.OLS(np.log(spot), T)
        res = model.fit()
        alpha_annual = 252 * res.params.iloc[1]

        print(f"  [스팟 수익률]")
        print(f"    일일 알파 = {res.params.iloc[1]:.6f}")
        print(f"    연율 알파 = {alpha_annual:.4f} ({alpha_annual*100:.2f}%)")

        # === 롤 수익률 (gamma) ===
        # 원본 로직: 각 행에서 유한한 값의 인덱스를 찾고
        # 처음 5개가 연속인지 확인 후 순차 인덱스 0-4로 회귀
        gamma = np.full(df_futures.shape[0], np.nan)
        for t in range(df_futures.shape[0]):
            idx = np.where(np.isfinite(df_futures.iloc[t, :]))[0]
            idxDiff = np.roll(idx, -1) - idx
            all_ones = all(idxDiff[0:4] == 1)

            if (len(idx) >= 5) and all_ones:
                FT = df_futures.iloc[t, idx[:5]]
                T = sm.add_constant(np.arange(FT.shape[0]))
                model = sm.OLS(np.log(FT.values), T)
                res = model.fit()
                gamma[t] = -12 * res.params[1]  # params is ndarray here

        mean_gamma = np.nanmean(gamma)

        self.results['futures_returns'] = {
            'alpha_annual': alpha_annual,
            'gamma_annual': mean_gamma,
            'gamma_series': gamma,
            'gamma_index': corn.index,
        }

        print(f"  [롤 수익률]")
        print(f"    평균 연율 감마 = {mean_gamma:.4f} ({mean_gamma*100:.2f}%)")
        n_valid = np.sum(np.isfinite(gamma))
        print(f"    감마 추정 일수 = {n_valid}")
        print(f"    책 기대값: 알파=+2.8%, 감마=-12.8%")

        # 차트
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(corn.index, spot.values, 'b-', linewidth=0.8)
        axes[0].set_title('Corn Spot Price', fontsize=13)
        axes[0].set_ylabel('Price (cents/bushel)')
        axes[0].grid(True, alpha=0.3)

        valid_idx = np.where(np.isfinite(gamma))[0]
        axes[1].plot(corn.index[valid_idx], gamma[valid_idx] * 100, 'r-', linewidth=0.5, alpha=0.7)
        axes[1].axhline(y=mean_gamma*100, color='k', linewidth=1, linestyle='--', label=f'Mean: {mean_gamma*100:.1f}%')
        axes[1].set_title('Corn Roll Return (Gamma, annualized)', fontsize=13)
        axes[1].set_ylabel('Roll Return (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch5_futures_returns.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch5_futures_returns.png', '옥수수 선물 스팟/롤 수익률'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_calendar_spread(self):
        """예제 5.4: CL 캘린더 스프레드 평균 회귀

        WTI 원유 12개월 캘린더 스프레드의 감마(롤 수익률)를 이용한 평균 회귀 전략.
        원본 로직: 컬럼 기반 포지션 매트릭스, 만기일 추적, holddays 제약.
        책 결과: APR=2.4%, Sharpe=1.28
        """
        print("=" * 60)
        print("📈 분석 4: CL 캘린더 스프레드 평균 회귀 (예제 5.4)")
        print("=" * 60)

        df = self.cl.copy()

        # 감마 계산 - 원본 로직 그대로 (포워드 커브의 처음 5개 연속 컬럼)
        gamma = np.full(df.shape[0], np.nan)
        for t in range(df.shape[0]):
            idx = np.where(np.isfinite(df.iloc[t, :]))[0]
            idxDiff = np.array(list(set(idx[1:]) - set(idx)))
            if (len(idx) >= 5) and (all(idxDiff[0:4] == 1)):
                FT = df.iloc[t, idx[:5]]
                T = sm.add_constant(np.arange(FT.shape[0]))
                model = sm.OLS(np.log(FT.values), T)
                res = model.fit()
                gamma[t] = -12 * res.params[1]  # params is ndarray here

        # ADF 검정
        gamma_finite = gamma[np.where(np.isfinite(gamma))]
        adf_result = ts_stats.adfuller(gamma_finite, maxlag=1, regression='c', autolag=None)
        print(f"  ADF 검정 on gamma: t-stat={adf_result[0]:.4f}, p-value={adf_result[1]:.6f}")

        gamma = pd.DataFrame(gamma)
        gamma.ffill(inplace=True)

        # 반감기 계산
        gammaGood = gamma[gamma.notna().values]
        gammalag = gammaGood.shift()
        deltaGamma = gammaGood - gammalag
        deltaGamma = deltaGamma.iloc[1:]
        gammalag = gammalag.iloc[1:]

        X = sm.add_constant(gammalag)
        model = sm.OLS(deltaGamma, X)
        res = model.fit()
        halflife = -np.log(2) / res.params.iloc[1] if res.params.iloc[1] < 0 else 100
        halflife_int = int(round(halflife))
        print(f"  반감기 = {halflife_int} 일")

        lookback = halflife_int

        # z-score
        MA = gamma.rolling(lookback).mean()
        MSTD = gamma.rolling(lookback).std()
        zScore = (gamma - MA) / MSTD

        # 원본 로직: 컬럼 기반 포지션 매트릭스
        positions = np.zeros(df.shape)
        isExpireDate = np.isfinite(df) & ~np.isfinite(df.shift(-1))
        holddays = 3 * 21
        numDaysStart = holddays + 10
        numDaysEnd = 10
        spreadMonth = 12

        for c in range(0, df.shape[1] - spreadMonth):
            expireIdx = np.where(isExpireDate.iloc[:, c])[-1]
            if c == 0:
                startIdx = max(0, expireIdx - numDaysStart)
                endIdx = expireIdx - numDaysEnd
            else:
                myStartIdx = endIdx + 1
                myEndIdx = expireIdx - numDaysEnd
                if len(myEndIdx) > 0 and len(myStartIdx) > 0:
                    if (myEndIdx[0] - myStartIdx[0] >= holddays):
                        startIdx = myStartIdx
                        endIdx = myEndIdx
                    else:
                        startIdx = np.array([np.inf])
                else:
                    startIdx = np.array([np.inf])

            if (len(expireIdx) > 0) and (len(startIdx) > 0) and (len(endIdx) > 0):
                s = int(startIdx[0]) if not np.isinf(startIdx[0]) else None
                e = int(endIdx[0]) if not np.isinf(endIdx[0]) else None
                if s is not None and e is not None and e > s:
                    positions[s:e, c] = -1
                    positions[s:e, c + spreadMonth] = 1

        positions[zScore.isna().values.flatten(), :] = 0
        zScore.fillna(-np.inf, inplace=True)

        positions[zScore.values.flatten() > 0, :] = -positions[zScore.values.flatten() > 0, :]
        # 수익률 계산 (순수 numpy)
        pos_shift = np.zeros_like(positions)
        pos_shift[1:] = positions[:-1]
        pnl = np.nansum(pos_shift * df.pct_change().values, axis=1)
        denom = np.nansum(np.abs(pos_shift), axis=1)
        denom[denom == 0] = np.nan
        ret_arr = pnl / denom

        valid_mask = np.isfinite(ret_arr)
        valid_ret = pd.Series(ret_arr[valid_mask], index=df.index[valid_mask])
        cumret = (1 + valid_ret).cumprod() - 1

        apr = np.prod(1 + valid_ret.values) ** (252 / len(valid_ret)) - 1 if len(valid_ret) > 0 else 0
        sharpe = np.sqrt(252) * np.mean(valid_ret.values) / np.std(valid_ret.values) if len(valid_ret) > 0 and np.std(valid_ret.values) > 0 else 0
        maxDD, maxDDD, _ = calculateMaxDD(cumret)

        self.results['calendar_spread'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
            'halflife': halflife_int, 'adf_pvalue': adf_result[1],
        }

        print(f"  APR = {apr:.4f} ({apr*100:.2f}%)")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD:.4f} ({maxDD*100:.2f}%)")
        print(f"  책 기대값: APR=2.4%, Sharpe=1.28")

        # 차트
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('CL Calendar Spread Mean Reversion - Cumulative Returns', fontsize=13)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)

        gamma_plot = gamma.values.flatten()
        valid_g = np.where(np.isfinite(gamma_plot))[0]
        axes[1].plot(df.index[valid_g], gamma_plot[valid_g] * 100, 'r-', linewidth=0.5, alpha=0.7)
        axes[1].set_title('CL Roll Return (Gamma, annualized)', fontsize=13)
        axes[1].set_ylabel('Roll Return (%)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch5_calendar_spread.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch5_calendar_spread.png', 'CL 캘린더 스프레드 평균 회귀'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 60)
        print("📝 리포트 생성 중...")
        print("=" * 60)

        report = []
        report.append("# Chapter 5: 통화와 선물의 평균 회귀 (Mean Reversion of Currencies and Futures)")
        report.append(f"\n> 분석 실행일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 개요
        report.append("## 1. 개요 및 문제 정의\n")
        report.append("Chapter 5는 통화와 선물 시장에서의 평균 회귀 전략을 탐구한다. 전통적으로 모멘텀과 연관되는 시장에서도")
        report.append("특정 니치에서 활용 가능한 평균 회귀 기회를 발견한다.\n")
        report.append("### 핵심 수학적 개념\n")
        report.append("**포트폴리오 수익률 (식 5.1):**\n")
        report.append("$$r(t+1) = \\frac{n_1 \\cdot y_{1,U}(t) \\cdot r_1(t+1) + n_2 \\cdot y_{2,U}(t) \\cdot r_2(t+1)}{|n_1| \\cdot y_{1,U}(t) + |n_2| \\cdot y_{2,U}(t)}$$\n")
        report.append("**롤오버 이자 보정 초과 수익률 (식 5.6):**\n")
        report.append("$$r(t+1) = \\log(y_{B,Q}(t+1)) - \\log(y_{B,Q}(t)) + \\log(1 + i_B(t)) - \\log(1 + i_Q(t))$$\n")
        report.append("**선물 가격 모델 (식 5.7-5.10):**\n")
        report.append("$$F(t, T) = S(t) \\cdot e^{\\gamma(t-T)}$$")
        report.append("$$\\frac{d(\\log F)}{dt} = \\alpha + \\gamma \\quad (\\text{총 수익률 = 스팟 수익률 + 롤 수익률})$$\n")

        # 2. 사용 데이터
        report.append("## 2. 사용 데이터\n")
        report.append("| 파일명 | 내용 | 용도 |")
        report.append("|--------|------|------|")
        report.append("| `inputData_AUDUSD_20120426.csv` | AUD/USD 일일 종가 | 예제 5.1 |")
        report.append("| `inputData_USDCAD_20120426.csv` | USD/CAD 일일 종가 | 예제 5.1 (역수→CAD/USD) |")
        report.append("| `inputData_AUDCAD_20120426.csv` | AUD/CAD 일일 종가 | 예제 5.2 |")
        report.append("| `AUD_interestRate.csv` | AUD 월별 금리 | 예제 5.2 롤오버 |")
        report.append("| `CAD_interestRate.csv` | CAD 월별 금리 | 예제 5.2 롤오버 |")
        report.append("| `inputDataDaily_C2_20120813.csv` | 옥수수 선물 30계약 + 스팟 | 예제 5.3 |")
        report.append("| `inputDataDaily_CL_20120502.csv` | WTI 원유 선물 88계약 | 예제 5.4 |\n")

        # 3. 예제 5.1
        report.append("## 3. 분석 1: AUD/USD vs CAD/USD 페어 트레이딩 (예제 5.1)\n")
        report.append("### 방법론\n")
        report.append("- AUD/USD와 CAD/USD(= 1/USD.CAD)를 공통 호가 통화(USD)로 맞춤")
        report.append("- 250일 롤링 요한센 공적분 검정으로 동적 헤지 비율 산출")
        report.append("- 20일 롤링 z-score 기반 선형 평균 회귀\n")
        report.append("**핵심**: 공적분 검정 시 두 통화가 동일한 호가 통화를 공유해야 포인트 가치가 동일해진다.\n")

        if 'audcad_unequal' in self.results:
            r = self.results['audcad_unequal']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 6.45% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 1.36 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | - |\n")
            report.append("![AUD-CAD 페어 트레이딩](figures/ch5_audcad_unequal.png)\n")

        # 4. 예제 5.2
        report.append("## 4. 분석 2: AUD/CAD 롤오버 이자 전략 (예제 5.2)\n")
        report.append("### 방법론\n")
        report.append("- AUD/CAD 직접 크로스레이트에 단순 선형 평균 회귀")
        report.append("- 롤오버 이자 반영: AUD(T+2, 수요일 3x) / CAD(T+1, 목요일 3x)\n")

        if 'audcad_daily' in self.results:
            r = self.results['audcad_daily']
            report.append("### 결과\n")
            report.append("| 지표 | 롤오버 포함 | 롤오버 미포함 | 책 기대값 |")
            report.append("|------|-----------|------------|----------|")
            report.append(f"| APR | {r['with_rollover']['apr']*100:.2f}% | {r['without_rollover']['apr']*100:.2f}% | 6.2% / 6.7% |")
            report.append(f"| Sharpe | {r['with_rollover']['sharpe']:.4f} | {r['without_rollover']['sharpe']:.4f} | 0.54 / 0.58 |")
            report.append(f"\n평균 금리차 (AUD - CAD): {r['avg_rate_diff']:.2f}% 연율\n")
            report.append("**통찰**: 연간 ~5%의 롤오버 이자에도 불구하고 단기 전략에서는 영향이 미미하다.\n")
            report.append("![AUD/CAD 롤오버](figures/ch5_audcad_rollover.png)\n")

        # 5. 예제 5.3
        report.append("## 5. 분석 3: 선물 스팟/롤 수익률 추정 (예제 5.3)\n")
        report.append("### 방법론\n")
        report.append("- **스팟 수익률(alpha)**: log(스팟 가격) ~ 시간 선형 회귀의 기울기 x 252")
        report.append("- **롤 수익률(gamma)**: 매일 가장 가까운 5개 연속 계약의 log(가격) ~ 만기까지 월수 회귀, gamma = -12 x 기울기\n")

        if 'futures_returns' in self.results:
            r = self.results['futures_returns']
            report.append("### 결과 (옥수수 선물)\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| 스팟 수익률 (alpha) | {r['alpha_annual']*100:.2f}% | +2.8% |")
            report.append(f"| 롤 수익률 (gamma) | {r['gamma_annual']*100:.2f}% | -12.8% |\n")
            report.append("**핵심 통찰**: BR, C, TU 등에서 롤 수익률의 크기가 스팟 수익률을 압도한다.")
            report.append("스팟 가격의 평균 회귀가 선물 가격의 평균 회귀를 의미하지 않는다.\n")
            report.append("![선물 수익률](figures/ch5_futures_returns.png)\n")

        # 6. 예제 5.4
        report.append("## 6. 분석 4: CL 캘린더 스프레드 평균 회귀 (예제 5.4)\n")
        report.append("### 방법론\n")
        report.append("1. CL 선물 포워드 커브에서 매일 감마(롤 수익률) 계산")
        report.append("2. 감마의 ADF 검정으로 정상성 확인")
        report.append("3. 반감기 계산 → z-score 룩백으로 사용")
        report.append("4. 근월-원월(12개월) 스프레드 포지션, z-score로 방향 결정\n")

        if 'calendar_spread' in self.results:
            r = self.results['calendar_spread']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 2.4% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 1.28 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | - |")
            report.append(f"| 반감기 | {r['halflife']}일 | 41일 |")
            report.append(f"| ADF p-value | {r['adf_pvalue']:.6f} | <0.01 |\n")
            report.append("![캘린더 스프레드](figures/ch5_calendar_spread.png)\n")

        # 7. 종합 비교
        report.append("## 7. 전략 종합 비교\n")
        report.append("| 전략 | APR | Sharpe | 시장 | 특성 |")
        report.append("|------|-----|--------|------|------|")
        if 'audcad_unequal' in self.results:
            r = self.results['audcad_unequal']
            report.append(f"| AUD/USD-CAD/USD Johansen | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | FX | 동적 헤지 |")
        if 'audcad_daily' in self.results:
            r = self.results['audcad_daily']['with_rollover']
            report.append(f"| AUD/CAD + Rollover | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | FX | 단순 헤지 |")
        if 'calendar_spread' in self.results:
            r = self.results['calendar_spread']
            report.append(f"| CL Calendar Spread | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | Futures | 감마 기반 |")
        report.append("")

        # 8. 결론
        report.append("## 8. 결론 및 권고사항\n")
        report.append("### 핵심 발견\n")
        report.append("1. **통화 페어 메커니즘**: 공적분 검정 시 동일 호가 통화를 사용해야 의미 있는 결과")
        report.append("2. **롤오버 이자의 미미한 영향**: 단기 전략에서 연 5% 금리차도 전략 성과에 작은 영향")
        report.append("3. **롤 수익률의 지배력**: 많은 선물에서 롤 수익률이 스팟 수익률을 압도")
        report.append("4. **캘린더 스프레드 신호**: 스팟 가격이 아닌 롤 수익률(감마)이 거래 신호\n")
        report.append("### 주의사항\n")
        report.append("- **레짐 변화**: VX-ES 관계는 2008년 전후로 레짐이 다름")
        report.append("- **선물 가격 동기화**: 서로 다른 거래소 선물 간 종가 시간 불일치 주의")
        report.append("- **생존자 편향**: 현존하는 계약만으로 백테스트하면 편향 발생 가능\n")

        report_path = REPORT_DIR / "chapter5_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"  ✓ 리포트 저장: {report_path}")
        print()

    def run(self):
        """전체 분석 오케스트레이션"""
        print("\n" + "🔬" * 30)
        print("  Chapter 5: 통화와 선물의 평균 회귀 - 종합 분석")
        print("🔬" * 30 + "\n")

        self.load_data()
        self.analyze_audcad_unequal()
        self.analyze_audcad_daily()
        self.analyze_futures_returns()
        self.analyze_calendar_spread()
        self.generate_report()

        print("=" * 60)
        print("✅ Chapter 5 분석 완료!")
        print(f"   리포트: reports/chapter5_report.md")
        print(f"   차트: {len(self.figures)}개 생성")
        for fig_name, fig_desc in self.figures:
            print(f"     - {fig_name}: {fig_desc}")
        print("=" * 60)


if __name__ == "__main__":
    analyzer = Chapter5Analyzer()
    analyzer.run()
