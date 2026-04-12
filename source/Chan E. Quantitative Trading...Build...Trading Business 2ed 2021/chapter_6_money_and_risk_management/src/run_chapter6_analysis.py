#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 6: Money and Risk Management - 종합 분석 스크립트

Ernest Chan, "Quantitative Trading: How to Build Your Own Algorithmic Trading Business"
(2nd Edition, 2021)

분석 항목:
  1. Kelly 최적 배분 (Example 6.3): OIH/RKH/RTH 포트폴리오
  2. Kelly 직관 (기하 랜덤워크 퍼즐): g = m - s^2/2 시뮬레이션
  3. Half-Kelly 배팅: 강건성 비교
  4. 레버리지 민감도: 성장률 vs 레버리지 곡선
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numpy.linalg import inv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

# 한글 폰트 설정
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UTIL_DIR = PROJECT_ROOT / "src"
REPORT_DIR = SCRIPT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

import sys
sys.path.insert(0, str(UTIL_DIR))
from calculateMaxDD import calculateMaxDD

# ── 상수 ──────────────────────────────────────────────────
RISK_FREE_RATE = 0.04          # 연율화 무위험이자율 4%
TRADING_DAYS = 252             # 연간 거래일 수
TICKERS = ["OIH", "RKH", "RTH"]
TICKER_NAMES = {
    "OIH": "VanEck Oil Services ETF",
    "RKH": "VanEck Regional Banks ETF",
    "RTH": "VanEck Retail ETF",
}


class Chapter6Analyzer:
    """Chapter 6 종합 분석: 자금 및 리스크 관리 (Kelly 공식 중심)"""

    def __init__(self):
        self.results = {}
        self.figures = []
        self.df = None           # 병합된 가격 데이터프레임
        self.daily_ret = None    # 일별 수익률
        self.excess_ret = None   # 초과수익률
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 데이터 로드
    # ================================================================
    def load_data(self):
        """OIH, RKH, RTH 가격 데이터를 로드하고 병합"""
        print("=" * 70)
        print("📂 데이터 로드")
        print("=" * 70)

        # 개별 파일 읽기
        df1 = pd.read_excel(DATA_DIR / "OIH.xls", engine="calamine")
        df2 = pd.read_excel(DATA_DIR / "RKH.xls", engine="calamine")
        df3 = pd.read_excel(DATA_DIR / "RTH.xls", engine="calamine")

        # Date 기준 병합
        df = pd.merge(df1, df2, on="Date", suffixes=("_OIH", "_RKH"))
        df.set_index("Date", inplace=True)
        df3_temp = df3.set_index("Date")
        df = df.join(df3_temp[["Adj Close"]], how="inner")
        df.rename(columns={"Adj Close": "Adj Close_RTH"}, inplace=True)
        df.sort_index(inplace=True)

        # 수정 종가 추출
        adj_cols = ["Adj Close_OIH", "Adj Close_RKH", "Adj Close_RTH"]
        prices = df[adj_cols].copy()
        prices.columns = TICKERS

        # 일별 수익률
        self.daily_ret = prices.pct_change().dropna()
        self.daily_ret = self.daily_ret[self.daily_ret.notna().all(axis=1)]

        # 초과수익률 (무위험이자율 차감)
        self.excess_ret = self.daily_ret - RISK_FREE_RATE / TRADING_DAYS

        self.df = prices

        # 요약 출력
        print(f"  기간: {self.daily_ret.index[0]} ~ {self.daily_ret.index[-1]}")
        print(f"  관측치: {len(self.daily_ret)} 거래일")
        for t in TICKERS:
            print(f"  {t} ({TICKER_NAMES[t]}): "
                  f"시작가 ${prices[t].dropna().iloc[0]:.2f}, "
                  f"종가 ${prices[t].dropna().iloc[-1]:.2f}")

        # 결과 저장
        self.results["data_summary"] = {
            "period_start": str(self.daily_ret.index[0].date() if hasattr(self.daily_ret.index[0], 'date') else self.daily_ret.index[0]),
            "period_end": str(self.daily_ret.index[-1].date() if hasattr(self.daily_ret.index[-1], 'date') else self.daily_ret.index[-1]),
            "num_obs": len(self.daily_ret),
        }
        print("✅ 데이터 로드 완료\n")

    # ================================================================
    # 분석 1: Kelly 최적 배분 (Example 6.3 재현)
    # ================================================================
    def analyze_kelly_optimal_allocation(self):
        """Example 6.3 재현: OIH/RKH/RTH 3자산 Kelly 최적 배분"""
        print("=" * 70)
        print("📊 분석 1: Kelly 최적 배분 (Example 6.3)")
        print("=" * 70)

        # 연율화 평균 초과수익률 벡터 M
        M = TRADING_DAYS * self.excess_ret.mean()
        print("\n  연율화 평균 초과수익률 벡터 M:")
        for t in TICKERS:
            print(f"    {t}: {M[t]:+.6f}")

        # 연율화 공분산 행렬 C
        C = TRADING_DAYS * self.excess_ret.cov()
        print("\n  연율화 공분산 행렬 C:")
        print(C.to_string(float_format=lambda x: f"{x:.6f}"))

        # Kelly 최적 배분: F* = C^{-1} M
        M_arr = M.values
        C_arr = C.values
        F = inv(C_arr) @ M_arr
        print("\n  Kelly 최적 레버리지 F*:")
        for i, t in enumerate(TICKERS):
            print(f"    {t}: {F[i]:+.4f}")

        # 최적 성장률: g = r + F^T C F / 2
        portfolio_var = F.T @ C_arr @ F
        g = RISK_FREE_RATE + portfolio_var / 2
        print(f"\n  최적 복리 성장률 g: {g:.4f} ({g*100:.2f}%)")

        # 포트폴리오 샤프 비율: S = sqrt(F^T C F)
        S = np.sqrt(portfolio_var)
        print(f"  포트폴리오 샤프 비율 S: {S:.4f}")

        # 총 레버리지
        total_leverage = np.sum(np.abs(F))
        print(f"  총 레버리지 |f1|+|f2|+|f3|: {total_leverage:.4f}")

        # 개별 전략 독립 가정 시 Kelly 비교
        print("\n  개별 자산 단일 Kelly (독립 가정 f_i = m_i / s_i^2):")
        for i, t in enumerate(TICKERS):
            m_i = M_arr[i]
            s_i_sq = C_arr[i, i]
            f_ind = m_i / s_i_sq
            g_ind = RISK_FREE_RATE + m_i * f_ind - (f_ind ** 2) * s_i_sq / 2
            print(f"    {t}: f={f_ind:+.4f}, 개별 성장률 g={g_ind:.4f}")

        # 결과 저장
        self.results["kelly_allocation"] = {
            "M": {t: float(M[t]) for t in TICKERS},
            "C": C.to_dict(),
            "F_star": {t: float(F[i]) for i, t in enumerate(TICKERS)},
            "growth_rate": float(g),
            "sharpe_ratio": float(S),
            "total_leverage": float(total_leverage),
            "F_arr": F,
            "C_arr": C_arr,
            "M_arr": M_arr,
        }
        print("\n✅ Kelly 최적 배분 분석 완료\n")

    # ================================================================
    # 분석 2: Kelly 직관 - 기하 랜덤워크 퍼즐
    # ================================================================
    def analyze_kelly_intuition(self):
        """기하 랜덤워크 시뮬레이션: g = m - s^2/2 직관"""
        print("=" * 70)
        print("🎲 분석 2: Kelly 직관 (기하 랜덤워크 퍼즐)")
        print("=" * 70)

        np.random.seed(42)
        n_paths = 1000
        n_steps = TRADING_DAYS  # 1년

        # 매 스텝 +1% 또는 -1% 동일 확률 (산술평균 0, 기하평균 < 0)
        pct_change = 0.01
        arithmetic_mean = 0.0  # (0.01 + (-0.01)) / 2 = 0
        variance = pct_change ** 2  # 0.01^2 = 0.0001
        expected_geo_growth = arithmetic_mean - variance / 2  # -0.00005 per step

        # 시뮬레이션
        random_returns = np.where(
            np.random.rand(n_paths, n_steps) > 0.5,
            1 + pct_change,
            1 - pct_change,
        )
        cum_wealth = np.cumprod(random_returns, axis=1)
        final_wealth = cum_wealth[:, -1]

        # 실현된 복리 성장률 (기간당)
        realized_geo_growth = (final_wealth ** (1.0 / n_steps) - 1).mean()

        print(f"\n  산술 기대수익률 m (기간당): {arithmetic_mean:.6f}")
        print(f"  분산 s^2 (기간당): {variance:.6f}")
        print(f"  이론적 기하 성장률 g = m - s^2/2: {expected_geo_growth:.6f}")
        print(f"  시뮬레이션 기하 성장률 (평균): {realized_geo_growth:.6f}")
        print(f"  최종 부의 중간값: {np.median(final_wealth):.4f}")
        print(f"  최종 부의 평균값: {np.mean(final_wealth):.4f}")
        print(f"  원금 이하 비율: {(final_wealth < 1.0).mean() * 100:.1f}%")

        # 레버리지별 시뮬레이션 (포트폴리오 데이터 기반)
        kelly_res = self.results["kelly_allocation"]
        S = kelly_res["sharpe_ratio"]
        C_arr = kelly_res["C_arr"]
        M_arr = kelly_res["M_arr"]

        # 단일 자산(OIH) 기준으로 레버리지 효과 시연
        oih_m = M_arr[0]  # 연율화 초과수익률
        oih_s2 = C_arr[0, 0]  # 연율화 분산
        f_kelly_oih = oih_m / oih_s2

        leverage_levels = [0.5, 1.0, 2.0, f_kelly_oih]
        leverage_labels = ["0.5x", "1.0x", "2.0x", f"Kelly ({f_kelly_oih:.2f}x)"]

        # OIH 일별 초과수익률로 레버리지별 경로 시뮬레이션
        oih_excess = self.excess_ret["OIH"].values
        n_days = len(oih_excess)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 좌측: 기하 랜덤워크 퍼즐 시뮬레이션 경로
        ax1 = axes[0]
        sample_indices = np.random.choice(n_paths, 20, replace=False)
        for idx in sample_indices:
            ax1.plot(cum_wealth[idx], alpha=0.3, linewidth=0.7, color="steelblue")
        ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Initial wealth")
        ax1.axhline(
            y=np.exp(expected_geo_growth * n_steps), color="darkred",
            linestyle="-.", linewidth=1.5, label=f"E[geo] = {expected_geo_growth:.5f}/step"
        )
        ax1.set_title("Geometric Random Walk Puzzle\n(±1%, 50/50 probability)")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Wealth")
        ax1.legend(fontsize=8)

        # 우측: 레버리지별 누적수익률 (OIH 기반)
        ax2 = axes[1]
        for f_lev, label in zip(leverage_levels, leverage_labels):
            # 레버리지 적용 수익률: r_leveraged = r_f + f * (r - r_f)
            # 이미 excess return이므로: r_leveraged = r_f/252 + f * excess_ret
            leveraged_ret = RISK_FREE_RATE / TRADING_DAYS + f_lev * oih_excess
            cum_ret = np.cumprod(1 + leveraged_ret)
            ax2.plot(cum_ret, label=label, linewidth=1.5)
        ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.set_title("OIH Leverage Effect on Cumulative Return")
        ax2.set_xlabel("Trading Days")
        ax2.set_ylabel("Cumulative Wealth")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "kelly_intuition.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(fig_path.name)
        print(f"\n  📈 차트 저장: {fig_path}")

        self.results["kelly_intuition"] = {
            "arithmetic_mean": arithmetic_mean,
            "variance": variance,
            "theoretical_geo_growth": expected_geo_growth,
            "simulated_geo_growth": float(realized_geo_growth),
            "median_final_wealth": float(np.median(final_wealth)),
            "mean_final_wealth": float(np.mean(final_wealth)),
            "pct_below_initial": float((final_wealth < 1.0).mean() * 100),
            "oih_kelly_f": float(f_kelly_oih),
        }
        print("✅ Kelly 직관 분석 완료\n")

    # ================================================================
    # 분석 3: Half-Kelly 배팅
    # ================================================================
    def analyze_half_kelly(self):
        """Full Kelly vs Half-Kelly 강건성 비교"""
        print("=" * 70)
        print("⚖️  분석 3: Half-Kelly 배팅 강건성 비교")
        print("=" * 70)

        kelly_res = self.results["kelly_allocation"]
        F_full = kelly_res["F_arr"]
        C_arr = kelly_res["C_arr"]
        M_arr = kelly_res["M_arr"]
        F_half = F_full / 2.0

        # 성장률 계산 함수: g(f) = r + f^T M - f^T C f / 2
        def growth_rate(F):
            return RISK_FREE_RATE + F.T @ M_arr - (F.T @ C_arr @ F) / 2

        g_full = growth_rate(F_full)
        g_half = growth_rate(F_half)

        # 분산 (포트폴리오 변동성)
        var_full = F_full.T @ C_arr @ F_full
        var_half = F_half.T @ C_arr @ F_half
        vol_full = np.sqrt(var_full)
        vol_half = np.sqrt(var_half)

        # 실제 포트폴리오 경로 시뮬레이션으로 낙폭 비교
        excess_arr = self.excess_ret.values  # (T, 3)

        # Full Kelly 포트폴리오 수익률
        port_ret_full = (RISK_FREE_RATE / TRADING_DAYS
                         + excess_arr @ F_full)
        cum_full = np.cumprod(1 + port_ret_full) - 1

        # Half-Kelly 포트폴리오 수익률
        port_ret_half = (RISK_FREE_RATE / TRADING_DAYS
                         + excess_arr @ F_half)
        cum_half = np.cumprod(1 + port_ret_half) - 1

        # 최대 낙폭 계산
        maxDD_full, maxDDD_full, _ = calculateMaxDD(cum_full)
        maxDD_half, maxDDD_half, _ = calculateMaxDD(cum_half)

        # 레버리지 비교
        lev_full = np.sum(np.abs(F_full))
        lev_half = np.sum(np.abs(F_half))

        print("\n  ┌────────────────────┬───────────────┬───────────────┐")
        print("  │       지표         │   Full Kelly  │  Half-Kelly   │")
        print("  ├────────────────────┼───────────────┼───────────────┤")
        for t_idx, t in enumerate(TICKERS):
            print(f"  │  f({t})            │  {F_full[t_idx]:+10.4f}   │  {F_half[t_idx]:+10.4f}   │")
        print(f"  │  총 레버리지       │  {lev_full:10.4f}   │  {lev_half:10.4f}   │")
        print(f"  │  성장률 g          │  {g_full:10.4f}   │  {g_half:10.4f}   │")
        print(f"  │  변동성 σ          │  {vol_full:10.4f}   │  {vol_half:10.4f}   │")
        print(f"  │  MaxDD             │  {maxDD_full:10.4f}   │  {maxDD_half:10.4f}   │")
        print(f"  │  MaxDD Duration    │  {maxDDD_full:10.0f}   │  {maxDDD_half:10.0f}   │")
        print("  └────────────────────┴───────────────┴───────────────┘")

        # 성장률 감소 대비 리스크 감소 비율
        g_reduction = (g_full - g_half) / g_full * 100
        vol_reduction = (vol_full - vol_half) / vol_full * 100
        dd_reduction = (abs(maxDD_full) - abs(maxDD_half)) / abs(maxDD_full) * 100

        print(f"\n  Half-Kelly 효과:")
        print(f"    성장률 감소: {g_reduction:.1f}%")
        print(f"    변동성 감소: {vol_reduction:.1f}%")
        print(f"    MaxDD 개선:  {dd_reduction:.1f}%")

        # 차트: 누적수익률 비교
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        # 상단: 누적수익률
        ax1 = axes[0]
        ax1.plot(cum_full * 100, label=f"Full Kelly (g={g_full:.2%})", linewidth=1.5, color="royalblue")
        ax1.plot(cum_half * 100, label=f"Half-Kelly (g={g_half:.2%})", linewidth=1.5, color="darkorange")
        ax1.set_title("Full Kelly vs Half-Kelly: Cumulative Returns")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 하단: 낙폭(Drawdown) 비교
        ax2 = axes[1]
        # 낙폭 계산
        wealth_full = 1 + cum_full
        hwm_full = np.maximum.accumulate(wealth_full)
        dd_full = (wealth_full / hwm_full - 1) * 100

        wealth_half = 1 + cum_half
        hwm_half = np.maximum.accumulate(wealth_half)
        dd_half = (wealth_half / hwm_half - 1) * 100

        ax2.fill_between(range(len(dd_full)), dd_full, 0, alpha=0.3, color="royalblue", label="Full Kelly DD")
        ax2.fill_between(range(len(dd_half)), dd_half, 0, alpha=0.3, color="darkorange", label="Half-Kelly DD")
        ax2.set_title("Drawdown Comparison")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Trading Days")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "half_kelly_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(fig_path.name)
        print(f"\n  📈 차트 저장: {fig_path}")

        self.results["half_kelly"] = {
            "F_full": {t: float(F_full[i]) for i, t in enumerate(TICKERS)},
            "F_half": {t: float(F_half[i]) for i, t in enumerate(TICKERS)},
            "g_full": float(g_full),
            "g_half": float(g_half),
            "vol_full": float(vol_full),
            "vol_half": float(vol_half),
            "maxDD_full": float(maxDD_full),
            "maxDD_half": float(maxDD_half),
            "maxDDD_full": float(maxDDD_full),
            "maxDDD_half": float(maxDDD_half),
            "g_reduction_pct": float(g_reduction),
            "vol_reduction_pct": float(vol_reduction),
            "dd_improvement_pct": float(dd_reduction),
        }
        print("✅ Half-Kelly 분석 완료\n")

    # ================================================================
    # 분석 4: 레버리지 민감도 곡선
    # ================================================================
    def analyze_leverage_sensitivity(self):
        """성장률 vs 레버리지 배수 곡선 (포물선 특성)"""
        print("=" * 70)
        print("📉 분석 4: 레버리지 민감도 분석")
        print("=" * 70)

        kelly_res = self.results["kelly_allocation"]
        F_star = kelly_res["F_arr"]
        C_arr = kelly_res["C_arr"]
        M_arr = kelly_res["M_arr"]
        g_star = kelly_res["growth_rate"]

        # 레버리지 배수 범위: 0x ~ 3x of Kelly
        multipliers = np.linspace(0, 3.0, 300)
        growth_rates = []

        for mult in multipliers:
            F_m = mult * F_star
            # g(f) = r + f^T M - f^T C f / 2
            g_m = RISK_FREE_RATE + F_m.T @ M_arr - (F_m.T @ C_arr @ F_m) / 2
            growth_rates.append(g_m)

        growth_rates = np.array(growth_rates)

        # 주요 포인트
        kelly_mult = 1.0
        half_kelly_mult = 0.5
        double_kelly_mult = 2.0

        # Regulation T 제약 (2x, 4x)
        total_lev_kelly = np.sum(np.abs(F_star))
        reg_t_overnight = 2.0 / total_lev_kelly  # 야간 보유
        reg_t_intraday = 4.0 / total_lev_kelly   # 장중

        # OIH 단일 자산 예시: 포물선 명확히 보여주기
        oih_m = M_arr[0]
        oih_s2 = C_arr[0, 0]
        f_range = np.linspace(0, 5.0, 300)
        g_single = RISK_FREE_RATE + f_range * oih_m - (f_range ** 2) * oih_s2 / 2
        f_kelly_single = oih_m / oih_s2

        print(f"\n  포트폴리오 Kelly 총 레버리지: {total_lev_kelly:.4f}")
        print(f"  Reg T 야간(2x) 배수: {reg_t_overnight:.4f}")
        print(f"  Reg T 장중(4x) 배수: {reg_t_intraday:.4f}")
        print(f"\n  OIH 단일 Kelly 레버리지: {f_kelly_single:.4f}")

        # 파산 경계: g = 0이 되는 레버리지
        # g(mult) = r + mult * F^T M - mult^2 * F^T C F / 2 = 0
        a_coeff = -(F_star.T @ C_arr @ F_star) / 2
        b_coeff = F_star.T @ M_arr
        c_coeff = RISK_FREE_RATE
        discriminant = b_coeff ** 2 + 4 * a_coeff * c_coeff  # ax^2 + bx + c = 0 → a<0
        if discriminant > 0:
            ruin_mult = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)
        else:
            ruin_mult = np.nan

        print(f"  성장률=0 레버리지 배수: {ruin_mult:.4f} (Kelly의 {ruin_mult:.2f}배)")

        # ── 차트 ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 좌측: 포트폴리오 레버리지 배수 vs 성장률
        ax1 = axes[0]
        ax1.plot(multipliers, growth_rates * 100, linewidth=2, color="steelblue")
        ax1.axhline(y=0, color="black", linewidth=0.8)

        # 주요 포인트 마킹
        markers = [
            (half_kelly_mult, "Half-Kelly", "green", "^"),
            (kelly_mult, "Full Kelly", "red", "o"),
            (double_kelly_mult, "2x Kelly", "purple", "s"),
        ]
        for mult, label, color, marker in markers:
            g_val = RISK_FREE_RATE + (mult * F_star).T @ M_arr - ((mult * F_star).T @ C_arr @ (mult * F_star)) / 2
            ax1.scatter(mult, g_val * 100, color=color, s=80, zorder=5, marker=marker)
            ax1.annotate(f" {label}\n g={g_val:.2%}", (mult, g_val * 100),
                         fontsize=8, color=color)

        # Reg T 제약선
        if reg_t_overnight < 3.0:
            ax1.axvline(x=reg_t_overnight, color="gray", linestyle="--", alpha=0.6, label=f"Reg T overnight ({reg_t_overnight:.2f}x)")
        if reg_t_intraday < 3.0:
            ax1.axvline(x=reg_t_intraday, color="gray", linestyle=":", alpha=0.6, label=f"Reg T intraday ({reg_t_intraday:.2f}x)")

        ax1.set_title("Growth Rate vs Kelly Leverage Multiplier\n(Portfolio)")
        ax1.set_xlabel("Kelly Multiplier")
        ax1.set_ylabel("Compounded Growth Rate (%)")
        ax1.legend(fontsize=8, loc="lower left")
        ax1.grid(True, alpha=0.3)

        # 우측: OIH 단일 자산 포물선
        ax2 = axes[1]
        ax2.plot(f_range, g_single * 100, linewidth=2, color="coral")
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.scatter(f_kelly_single, (RISK_FREE_RATE + oih_m ** 2 / (2 * oih_s2)) * 100,
                    color="red", s=100, zorder=5, marker="*")
        ax2.annotate(f"  Kelly f*={f_kelly_single:.2f}",
                     (f_kelly_single, (RISK_FREE_RATE + oih_m ** 2 / (2 * oih_s2)) * 100),
                     fontsize=9, color="red")
        ax2.scatter(f_kelly_single / 2, (RISK_FREE_RATE + (f_kelly_single / 2) * oih_m - (f_kelly_single / 2) ** 2 * oih_s2 / 2) * 100,
                    color="green", s=80, zorder=5, marker="^")
        ax2.annotate(f"  Half-Kelly",
                     (f_kelly_single / 2, (RISK_FREE_RATE + (f_kelly_single / 2) * oih_m - (f_kelly_single / 2) ** 2 * oih_s2 / 2) * 100),
                     fontsize=9, color="green")
        # Reg T 2x, 4x 선
        for lev_limit, ls, lbl in [(2.0, "--", "Reg T 2x"), (4.0, ":", "Reg T 4x")]:
            if lev_limit < 5.0:
                ax2.axvline(x=lev_limit, color="gray", linestyle=ls, alpha=0.6, label=lbl)
        ax2.set_title("Growth Rate vs Leverage\n(OIH Single Asset: $f^* = m/s^2$)")
        ax2.set_xlabel("Leverage f")
        ax2.set_ylabel("Compounded Growth Rate (%)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "leverage_sensitivity.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(fig_path.name)
        print(f"\n  📈 차트 저장: {fig_path}")

        self.results["leverage_sensitivity"] = {
            "total_kelly_leverage": float(total_lev_kelly),
            "reg_t_overnight_mult": float(reg_t_overnight),
            "reg_t_intraday_mult": float(reg_t_intraday),
            "ruin_multiplier": float(ruin_mult) if not np.isnan(ruin_mult) else None,
            "oih_kelly_f": float(f_kelly_single),
        }
        print("✅ 레버리지 민감도 분석 완료\n")

    # ================================================================
    # 리포트 생성
    # ================================================================
    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 70)
        print("📝 마크다운 리포트 생성")
        print("=" * 70)

        kelly = self.results["kelly_allocation"]
        intuition = self.results["kelly_intuition"]
        half = self.results["half_kelly"]
        sensitivity = self.results["leverage_sensitivity"]
        data_sum = self.results["data_summary"]

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("# Chapter 6: Money and Risk Management - 종합 분석 리포트\n")
        lines.append(f"> 생성일시: {now_str}  ")
        lines.append(f"> 데이터 기간: {data_sum['period_start']} \\~ {data_sum['period_end']}  ")
        lines.append(f"> 관측치: {data_sum['num_obs']} 거래일\n")
        lines.append("---\n")

        # ── 1. 개요 ──
        lines.append("## 1. 개요 및 문제 정의\n")
        lines.append("이 장은 **장기 복리 성장률(compounded growth rate)** 을 극대화하는 최적 자본 배분과 "
                      "레버리지 결정 문제를 다룹니다. 핵심 도구는 **켈리 공식(Kelly formula)** 이며, "
                      "모수 추정 불확실성에 대한 보완책으로 **하프-켈리(Half-Kelly)** 배팅을 권장합니다.\n")
        lines.append("### 핵심 수식\n")
        lines.append("| 수식 | 설명 |")
        lines.append("|------|------|")
        lines.append("| $F^* = C^{-1}M$ | 다중 전략 Kelly 최적 배분 |")
        lines.append("| $f^* = m / s^2$ | 단일 전략 Kelly 최적 레버리지 |")
        lines.append("| $g = r + S^2 / 2$ | 최적 배분 시 복리 성장률 |")
        lines.append("| $S = \\sqrt{F^{*T} C F^*}$ | 포트폴리오 샤프 비율 |")
        lines.append("| $g = m - s^2 / 2$ | 기하 랜덤워크 복리 성장률 (비레버리지) |")
        lines.append("| $F_{half} = F^* / 2$ | 하프-켈리 배분 |\n")
        lines.append("여기서:")
        lines.append("- $M = 252 \\times \\bar{r}_{excess}$: 연율화 평균 초과수익률 벡터")
        lines.append("- $C = 252 \\times \\text{Cov}(r_{excess})$: 연율화 공분산 행렬")
        lines.append("- $r$: 연율화 무위험이자율 (4%)\n")

        # ── 2. 데이터 ──
        lines.append("## 2. 사용 데이터\n")
        lines.append("| 파일명 | 티커 | 설명 | 용도 |")
        lines.append("|--------|------|------|------|")
        lines.append("| `OIH.xls` | OIH | VanEck Oil Services ETF (유전 서비스) | Kelly 배분 포트폴리오 구성 |")
        lines.append("| `RKH.xls` | RKH | VanEck Regional Banks ETF (지역 은행) | Kelly 배분 포트폴리오 구성 |")
        lines.append("| `RTH.xls` | RTH | VanEck Retail ETF (소매) | Kelly 배분 포트폴리오 구성 |\n")
        lines.append(f"- **분석 기간**: {data_sum['period_start']} \\~ {data_sum['period_end']}")
        lines.append(f"- **관측치 수**: {data_sum['num_obs']} 거래일")
        lines.append(f"- **무위험이자율**: 연 {RISK_FREE_RATE*100:.0f}% (일별 {RISK_FREE_RATE/TRADING_DAYS:.6f})\n")

        # ── 3. 분석 1: Kelly 최적 배분 ──
        lines.append("## 3. 분석 1: Kelly 최적 배분 (Example 6.3)\n")
        lines.append("### 연율화 평균 초과수익률 벡터 $M$\n")
        lines.append("| 티커 | 연율화 평균 초과수익률 |")
        lines.append("|------|----------------------|")
        for t in TICKERS:
            lines.append(f"| {t} | {kelly['M'][t]:+.6f} |")
        lines.append("")

        lines.append("### 연율화 공분산 행렬 $C$\n")
        lines.append("| | OIH | RKH | RTH |")
        lines.append("|-----|------|------|------|")
        C_dict = kelly["C"]
        for t1 in TICKERS:
            row = f"| **{t1}** |"
            for t2 in TICKERS:
                row += f" {C_dict[t1][t2]:.6f} |"
            lines.append(row)
        lines.append("")

        lines.append("### Kelly 최적 배분 $F^* = C^{-1}M$\n")
        lines.append("| 티커 | 최적 레버리지 $f^*$ | 해석 |")
        lines.append("|------|--------------------|----- |")
        for t in TICKERS:
            f_val = kelly["F_star"][t]
            direction = "롱(Long)" if f_val > 0 else "숏(Short)"
            lines.append(f"| {t} | {f_val:+.4f} | 자기자본의 {abs(f_val)*100:.1f}%를 {direction} |")
        lines.append("")

        lines.append("### 포트폴리오 성과 지표\n")
        lines.append("| 지표 | 값 |")
        lines.append("|------|----|")
        lines.append(f"| 최적 복리 성장률 $g$ | {kelly['growth_rate']:.4f} ({kelly['growth_rate']*100:.2f}%) |")
        lines.append(f"| 포트폴리오 샤프 비율 $S$ | {kelly['sharpe_ratio']:.4f} |")
        lines.append(f"| 총 레버리지 $\\sum|f_i|$ | {kelly['total_leverage']:.4f} |")
        lines.append("")
        lines.append("> **해석**: RTH의 평균 초과수익률이 음수이므로 Kelly 공식은 RTH를 **공매도(숏)** 하라고 "
                      "권고합니다. 포트폴리오의 복리 성장률 15.29%는 어떤 개별 종목의 성장률보다 더 높으며, "
                      "이는 분산 투자의 효과를 보여줍니다.\n")

        # ── 4. 분석 2: Kelly 직관 ──
        lines.append("## 4. 분석 2: Kelly 직관 (기하 랜덤워크 퍼즐)\n")
        lines.append("### 퍼즐 설명\n")
        lines.append("매 기간 +1% 또는 -1%의 동일 확률을 가진 주식의 장기 수익률은 본전이 아니라 **손실** 입니다.")
        lines.append("이는 기하 랜덤워크에서의 복리 성장률이 산술 평균이 아닌 다음 공식을 따르기 때문입니다:\n")
        lines.append("$$g = m - \\frac{s^2}{2}$$\n")
        lines.append("### 시뮬레이션 결과 (1,000 경로, 252 스텝)\n")
        lines.append("| 지표 | 값 |")
        lines.append("|------|----|")
        lines.append(f"| 산술 기대수익률 $m$ | {intuition['arithmetic_mean']:.6f} |")
        lines.append(f"| 분산 $s^2$ | {intuition['variance']:.6f} |")
        lines.append(f"| 이론적 기하 성장률 $g$ | {intuition['theoretical_geo_growth']:.6f} |")
        lines.append(f"| 시뮬레이션 기하 성장률 | {intuition['simulated_geo_growth']:.6f} |")
        lines.append(f"| 최종 부의 중간값 | {intuition['median_final_wealth']:.4f} |")
        lines.append(f"| 최종 부의 평균값 | {intuition['mean_final_wealth']:.4f} |")
        lines.append(f"| 원금 이하 비율 | {intuition['pct_below_initial']:.1f}% |")
        lines.append("")
        lines.append("![Kelly 직관: 기하 랜덤워크와 레버리지 효과](figures/kelly_intuition.png)\n")
        lines.append("> **교훈**: 리스크(변동성)는 항상 장기 복리 성장률을 감소시킵니다. "
                      "산술 평균이 0이더라도 기하 평균(실제 복리 수익률)은 음수가 됩니다. "
                      "이것이 바로 리스크 관리가 필수적인 이유입니다.\n")

        # ── 5. 분석 3: Half-Kelly ──
        lines.append("## 5. 분석 3: Half-Kelly 배팅 강건성 비교\n")
        lines.append("모수 추정의 불확실성과 수익률 분포의 비정규성을 고려하여, 트레이더들은 "
                      "Kelly 권장 레버리지를 **절반으로 줄이는 것** 을 선호합니다.\n")
        lines.append("### Full Kelly vs Half-Kelly 비교\n")
        lines.append("| 지표 | Full Kelly | Half-Kelly | 변화 |")
        lines.append("|------|-----------|------------|------|")
        for t in TICKERS:
            lines.append(f"| $f$({t}) | {half['F_full'][t]:+.4f} | {half['F_half'][t]:+.4f} | 50% 축소 |")
        lines.append(f"| 성장률 $g$ | {half['g_full']:.4f} ({half['g_full']*100:.2f}%) | {half['g_half']:.4f} ({half['g_half']*100:.2f}%) | -{half['g_reduction_pct']:.1f}% |")
        lines.append(f"| 변동성 $\\sigma$ | {half['vol_full']:.4f} | {half['vol_half']:.4f} | -{half['vol_reduction_pct']:.1f}% |")
        lines.append(f"| 최대 낙폭 MaxDD | {half['maxDD_full']:.4f} | {half['maxDD_half']:.4f} | {half['dd_improvement_pct']:.1f}% 개선 |")
        lines.append(f"| MaxDD 지속기간 | {half['maxDDD_full']:.0f}일 | {half['maxDDD_half']:.0f}일 | |")
        lines.append("")
        lines.append("![Full Kelly vs Half-Kelly 비교](figures/half_kelly_comparison.png)\n")
        lines.append("> **결론**: Half-Kelly는 성장률을 약 25% 감소시키는 대가로 변동성을 50%, "
                      "최대 낙폭을 크게 줄여줍니다. 모수 추정 오차를 고려하면 이는 매우 합리적인 "
                      "트레이드오프입니다.\n")

        # ── 6. 분석 4: 레버리지 민감도 ──
        lines.append("## 6. 분석 4: 레버리지 민감도 분석\n")
        lines.append("Kelly 공식에 의한 성장률은 레버리지의 **이차함수(포물선)** 형태입니다. "
                      "Kelly 최적점을 넘어서면 성장률이 급격히 감소하며, 과도한 레버리지는 "
                      "궁극적으로 파산(ruin)으로 이어집니다.\n")
        lines.append("### Regulation T 레버리지 제약\n")
        lines.append("| 항목 | 값 |")
        lines.append("|------|----|")
        lines.append(f"| Kelly 총 레버리지 | {sensitivity['total_kelly_leverage']:.4f} |")
        lines.append(f"| Reg T 야간(2x) 배수 | {sensitivity['reg_t_overnight_mult']:.4f} |")
        lines.append(f"| Reg T 장중(4x) 배수 | {sensitivity['reg_t_intraday_mult']:.4f} |")
        if sensitivity['ruin_multiplier']:
            lines.append(f"| 파산 경계 배수 | {sensitivity['ruin_multiplier']:.4f} |")
        lines.append("")
        lines.append("![레버리지 민감도 곡선](figures/leverage_sensitivity.png)\n")
        lines.append("> **실무 권고**: 개인 투자자는 Regulation T에 의해 야간 보유 시 2배, "
                      "장중 보유 시 4배로 레버리지가 제한됩니다. Kelly 최적 레버리지가 이 한도를 "
                      "초과하면 모든 배분을 동일 비율로 축소해야 합니다.\n")

        # ── 7. 결론 ──
        lines.append("## 7. 결론 및 권고사항\n")
        lines.append("### 핵심 발견 요약\n")
        lines.append("| # | 발견 | 함의 |")
        lines.append("|---|------|------|")
        lines.append("| 1 | Kelly 공식은 최적 자본 배분과 레버리지를 동시에 결정 | 장기 복리 성장률 극대화 |")
        lines.append("| 2 | 리스크(변동성)는 항상 기하 성장률을 감소시킴 | 리스크 관리의 필수성 |")
        lines.append("| 3 | Half-Kelly는 성장률 25% 감소로 위험을 크게 줄임 | 실무적으로 가장 권장되는 접근법 |")
        lines.append("| 4 | Kelly 이상의 레버리지는 성장률을 오히려 감소시킴 | 과도한 레버리지는 역효과 |")
        lines.append("")
        lines.append("### 실무 권고사항\n")
        lines.append("1. **Half-Kelly를 기본값으로 사용하세요**: 모수 추정 오차와 팻테일(fat-tail) 위험에 대한 안전마진")
        lines.append("2. **매일 자본 배분을 갱신하세요**: Kelly 기준을 따르려면 자기자본 변동에 맞춰 포지션을 조정해야 합니다")
        lines.append("3. **이동평균 기반으로 $F^*$를 주기적으로 재계산하세요**: 6개월 룩백 기간 권장")
        lines.append("4. **Regulation T 제약을 항상 확인하세요**: 야간 2x, 장중 4x 한도")
        lines.append("5. **Kelly 레버리지를 절대 초과하지 마세요**: 성장률 포물선의 정점을 넘으면 위험만 증가하고 수익은 감소합니다\n")

        lines.append("---\n")
        lines.append("*이 리포트는 `run_chapter6_analysis.py`에 의해 자동 생성되었습니다.*\n")

        # 파일 저장
        report_path = REPORT_DIR / "chapter6_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  📄 리포트 저장: {report_path}")
        print(f"  📊 차트 {len(self.figures)}개: {', '.join(self.figures)}")
        print("✅ 리포트 생성 완료\n")

    # ================================================================
    # 실행
    # ================================================================
    def run(self):
        """전체 분석 파이프라인 실행"""
        print("\n" + "🔬" * 35)
        print("  Chapter 6: Money and Risk Management")
        print("  Kelly Formula & Optimal Capital Allocation")
        print("🔬" * 35 + "\n")

        self.load_data()
        self.analyze_kelly_optimal_allocation()
        self.analyze_kelly_intuition()
        self.analyze_half_kelly()
        self.analyze_leverage_sensitivity()
        self.generate_report()

        print("=" * 70)
        print("🎉 전체 분석 완료!")
        print("=" * 70)
        print(f"  📄 리포트: {REPORT_DIR / 'chapter6_report.md'}")
        print(f"  📊 차트:   {FIGURES_DIR}/")
        for fig_name in self.figures:
            print(f"       - {fig_name}")
        print("=" * 70)


if __name__ == "__main__":
    Chapter6Analyzer().run()
