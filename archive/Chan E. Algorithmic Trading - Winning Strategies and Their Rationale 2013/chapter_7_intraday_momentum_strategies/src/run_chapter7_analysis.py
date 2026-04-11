#!/usr/bin/env python3
"""
Chapter 7: 일중 모멘텀 전략 - 종합 분석 리포트 생성기

이 스크립트는 Ernest Chan의 "Algorithmic Trading" Chapter 7의 핵심 개념들을 실행하고
분석 결과를 종합 리포트 형태로 출력합니다.

분석 내용:
1. FSTX 오프닝 갭 전략 (예제 7.1)
2. VX-ES 롤 수익률 모멘텀 전략
"""

import os
import sys
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


class Chapter7Analyzer:
    """Chapter 7 일중 모멘텀 전략 분석 클래스"""

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

        # FSTX 선물 데이터 (OHLC)
        fstx_path = data_dir / "inputDataDaily_FSTX_20120517.csv"
        self.fstx = pd.read_csv(fstx_path)
        self.fstx['Date'] = pd.to_datetime(self.fstx['Date'], format='%Y%m%d')
        self.fstx.set_index('Date', inplace=True)
        print(f"  ✓ FSTX: {len(self.fstx)} 거래일 (OHLC)")

        # VX 선물 데이터
        vx_path = data_dir / "inputDataDaily_VX_20120507.csv"
        self.vx = pd.read_csv(vx_path)
        self.vx['Date'] = pd.to_datetime(self.vx['Date'], format='%Y%m%d')
        self.vx.set_index('Date', inplace=True)
        print(f"  ✓ VX 선물: {len(self.vx)} 거래일 x {len(self.vx.columns)} 계약")

        # VIX 지수
        vix_path = data_dir / "VIX.csv"
        self.vix = pd.read_csv(vix_path)
        self.vix['Date'] = pd.to_datetime(self.vix['Date'], format='%Y-%m-%d')
        self.vix.set_index('Date', inplace=True)
        self.vix = self.vix[['Close']]
        self.vix.rename(columns={'Close': 'VIX'}, inplace=True)
        print(f"  ✓ VIX: {len(self.vix)} 거래일")

        # ES 선물 (백조정 연속 계약)
        es_path = data_dir / "inputDataDaily_ES_20120507.csv"
        self.es = pd.read_csv(es_path)
        self.es['Date'] = pd.to_datetime(self.es['Date'], format='%Y%m%d')
        self.es.set_index('Date', inplace=True)
        self.es.rename(columns={'Close': 'ES'}, inplace=True)
        print(f"  ✓ ES: {len(self.es)} 거래일")
        print()

    def analyze_opening_gap(self):
        """예제 7.1: FSTX 오프닝 갭 전략

        오프닝 가격이 전일 고가 이상이면 롱, 전일 저가 이하면 숏.
        당일 종가에 청산 (일중 전략).
        책 결과: APR=7.5%, Sharpe=0.49, MaxDD=-23.4%
        """
        print("=" * 60)
        print("📈 분석 1: FSTX 오프닝 갭 전략 (예제 7.1)")
        print("=" * 60)

        df = self.fstx.copy()
        entryZscore = 0.1

        # 90일 롤링 종가-종가 수익률의 표준편차
        stdretC2C90d = df['Close'].pct_change().rolling(90).std().shift()

        # 진입 조건 - 원본 로직 그대로
        longs = df['Open'] >= df['High'].shift() * (1 + entryZscore * stdretC2C90d)
        shorts = df['Open'] >= df['Low'].shift() * (1 - entryZscore * stdretC2C90d)

        positions = np.zeros(longs.shape)
        positions[longs] = 1
        positions[shorts] = -1

        # 일중 수익률: (종가 - 시가) / 시가
        ret = positions * (df['Close'] - df['Open']).values / df['Open'].values

        cumret = np.cumprod(1 + ret) - 1
        cumret = pd.Series(cumret, index=df.index)

        apr = np.prod(1 + ret) ** (252 / len(ret)) - 1
        sharpe = np.sqrt(252) * np.mean(ret) / np.std(ret) if np.std(ret) > 0 else 0
        maxDD, maxDDD = calculateMaxDD(cumret)

        # 거래 통계
        n_longs = int(np.sum(positions > 0))
        n_shorts = int(np.sum(positions < 0))
        n_total = n_longs + n_shorts

        self.results['opening_gap'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
            'n_trades': n_total, 'n_longs': n_longs, 'n_shorts': n_shorts,
            'entryZscore': entryZscore,
        }

        print(f"  Entry Z-score = {entryZscore}")
        print(f"  거래 횟수: {n_total} (롱={n_longs}, 숏={n_shorts})")
        print(f"  APR = {apr*100:.2f}%")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD*100:.2f}%, Max DDD = {maxDDD}일")
        print(f"  책 기대값: APR=7.5%, Sharpe=0.49, MaxDD=-23.4%")

        # 차트
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('FSTX Opening Gap Strategy - Cumulative Returns', fontsize=13)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df.index, df['Close'].values, 'gray', linewidth=0.5, alpha=0.7)
        axes[1].set_title('FSTX Futures Price', fontsize=13)
        axes[1].set_ylabel('Price')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch7_opening_gap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch7_opening_gap.png', 'FSTX 오프닝 갭 전략'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def analyze_vx_es_rollreturn(self):
        """VX-ES 롤 수익률 모멘텀 전략

        VX 선물과 VIX 지수 간 롤 수익률 기반 거래.
        콘탱고 시 VX 숏 + ES 숏, 백워데이션 시 반대.
        책 결과: APR=37.8%, Sharpe=2.12, MaxDD=-43.4%
        """
        print("=" * 60)
        print("📈 분석 2: VX-ES 롤 수익률 전략")
        print("=" * 60)

        entryThreshold = 0.1
        onewaytcost = 1 / 10000

        # 공통 날짜로 병합
        df = pd.merge(self.vx, self.vix, left_index=True, right_index=True, how='inner')
        df = pd.merge(df, self.es, left_index=True, right_index=True, how='inner')

        # 분리
        vx_cols = [c for c in df.columns if c.startswith('VX_')]
        vx = df[vx_cols]
        vix = df[['VIX']]
        es = df[['ES']]

        print(f"  공통 거래일: {len(df)}")
        print(f"  VX 계약: {len(vx_cols)}개")

        # 만기일 감지: 현재 값이 있고 다음 날 값이 없는 날
        isExpireDate = vx.notnull() & vx.shift(-1).isnull()

        numDaysStart = 40
        numDaysEnd = 10

        # 포지션: VX 각 계약 + ES (마지막 컬럼)
        positions = np.zeros((vx.shape[0], vx.shape[1] + 1))

        for c in range(vx.shape[1] - 1):
            expireIdx = np.where(isExpireDate.iloc[:, c])[0]
            if len(expireIdx) == 0:
                continue

            exp = expireIdx[0]
            if c == 0:
                startIdx = max(0, exp - numDaysStart)
                endIdx = exp - numDaysEnd
            else:
                startIdx = max(endIdx + 1, exp - numDaysStart)
                endIdx = exp - numDaysEnd

            if exp >= 0 and endIdx > startIdx:
                idx = np.arange(startIdx, endIdx + 1)
                idx = idx[idx < len(vx)]  # 범위 제한

                # 일일 롤: (VX - VIX) / (만기까지 남은 일수)
                days_to_exp = np.arange(exp - startIdx + 1, exp - endIdx, -1)
                days_to_exp = days_to_exp[:len(idx)]

                vx_vals = vx.iloc[idx, c].values
                vix_vals = vix.iloc[idx, 0].values

                valid = np.isfinite(vx_vals) & np.isfinite(vix_vals) & (days_to_exp > 0)
                daily_roll = np.full(len(idx), np.nan)
                daily_roll[valid] = (vx_vals[valid] - vix_vals[valid]) / days_to_exp[valid]

                # 콘탱고 (롤 > threshold): VX 숏, ES 숏
                long_cond = np.where(valid & (daily_roll > entryThreshold))[0]
                short_cond = np.where(valid & (daily_roll < -entryThreshold))[0]

                positions[idx[long_cond], c] = -1
                positions[idx[long_cond], -1] = -1

                positions[idx[short_cond], c] = 1
                positions[idx[short_cond], -1] = 1

        # 포인트 가치: VX x 1000, ES x 50
        y = pd.merge(vx * 1000, es * 50, left_index=True, right_index=True, how='inner')
        positions_df = pd.DataFrame(positions, index=y.index)

        # PnL: 가격 변화 * 포지션 - 거래비용
        y_diff = y - y.shift()
        pos_y = positions_df.values * y.values
        pos_y_shift = pd.DataFrame(pos_y).shift().values

        pnl = np.nansum(positions_df.shift().values * y_diff.values, axis=1) \
              - onewaytcost * np.nansum(np.abs(pos_y - np.nan_to_num(pos_y_shift, nan=0)), axis=1)

        denom = np.nansum(np.abs(pos_y_shift), axis=1)
        denom[denom == 0] = np.nan
        ret = pnl / denom

        # 2008-08-04 이후 500일 지점부터 사용 (원본 로직)
        ret = pd.Series(ret, index=y.index)
        start_date = pd.Timestamp('2008-08-04')
        idx = ret.index[ret.index >= start_date]
        if len(idx) > 500:
            ret_subset = ret[idx[500:]]
        else:
            ret_subset = ret[idx]

        ret_clean = ret_subset.replace([np.inf, -np.inf], 0).fillna(0)
        cumret = (1 + ret_clean).cumprod() - 1

        apr = np.prod(1 + ret_clean.values) ** (252 / len(ret_clean)) - 1 if len(ret_clean) > 0 else 0
        sharpe = np.sqrt(252) * np.mean(ret_clean.values) / np.std(ret_clean.values) if np.std(ret_clean.values) > 0 else 0
        maxDD, maxDDD = calculateMaxDD(cumret)

        self.results['vx_es'] = {
            'apr': apr, 'sharpe': sharpe, 'maxDD': maxDD, 'maxDDD': maxDDD,
            'entryThreshold': entryThreshold,
        }

        print(f"  Entry Threshold = {entryThreshold}")
        print(f"  APR = {apr*100:.2f}%")
        print(f"  Sharpe = {sharpe:.4f}")
        print(f"  Max DD = {maxDD*100:.2f}%, Max DDD = {maxDDD}일")
        print(f"  책 기대값: APR=37.8%, Sharpe=2.12, MaxDD=-43.4%")

        # VX-VIX 롤 구조 분석
        # 근월 VX와 VIX 차이 시계열
        front_vx = pd.Series(np.nan, index=vx.index)
        for i in range(len(vx)):
            row = vx.iloc[i]
            valid_vals = row.dropna()
            if len(valid_vals) > 0:
                front_vx.iloc[i] = valid_vals.iloc[0]

        roll_diff = front_vx - vix['VIX']

        # 차트
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(cumret.index, cumret.values * 100, 'b-', linewidth=1)
        axes[0].set_title('VX-ES Roll Return Strategy - Cumulative Returns', fontsize=13)
        axes[0].set_ylabel('Cumulative Return (%)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(vix.index, vix.values, 'r-', linewidth=0.8, label='VIX')
        axes[1].plot(front_vx.index, front_vx.values, 'b-', linewidth=0.8, alpha=0.6, label='Front VX')
        axes[1].set_title('VIX vs Front Month VX Futures', fontsize=13)
        axes[1].set_ylabel('Level')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(roll_diff.index, roll_diff.values, 'g-', linewidth=0.5, alpha=0.7)
        axes[2].axhline(y=0, color='k', linewidth=0.5)
        axes[2].set_title('VX - VIX (Contango/Backwardation)', fontsize=13)
        axes[2].set_ylabel('Difference')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "ch7_vx_es_rollreturn.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.figures.append(('ch7_vx_es_rollreturn.png', 'VX-ES 롤 수익률 전략'))
        print(f"  ✓ 차트 저장: {fig_path.name}")
        print()

    def generate_report(self):
        """마크다운 리포트 생성"""
        print("=" * 60)
        print("📝 리포트 생성 중...")
        print("=" * 60)

        report = []
        report.append("# Chapter 7: 일중 모멘텀 전략 (Intraday Momentum Strategies)")
        report.append(f"\n> 분석 실행일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 개요
        report.append("## 1. 개요 및 문제 정의\n")
        report.append("Chapter 7은 일중(intraday) 시간 척도에서의 모멘텀 전략을 탐구한다.")
        report.append("주로 오프닝 갭(opening gap)과 변동성 선물의 롤 수익률을 활용한다.\n")
        report.append("### 핵심 개념\n")
        report.append("1. **오프닝 갭**: 전일 종가 대비 당일 시가의 비정상적 변동을 모멘텀 신호로 활용")
        report.append("2. **VX 롤 수익률**: VX 선물과 VIX 지수 간 갭(콘탱고/백워데이션)을 수확")
        report.append("3. **ETF-선물 차익거래**: 레버리지 ETF의 일일 리밸런싱을 활용한 거래\n")
        report.append("### 핵심 수학적 개념\n")
        report.append("**오프닝 갭 진입 조건:**\n")
        report.append("$$\\text{Long if: } O(t) \\geq H(t-1) \\cdot (1 + z \\cdot \\sigma_{90d})$$")
        report.append("$$\\text{Short if: } O(t) \\leq L(t-1) \\cdot (1 - z \\cdot \\sigma_{90d})$$\n")
        report.append("**일일 롤 수익률:**\n")
        report.append("$$\\text{dailyRoll} = \\frac{F_{VX}(t) - VIX(t)}{T - t}$$\n")

        # 2. 사용 데이터
        report.append("## 2. 사용 데이터\n")
        report.append("| 파일명 | 내용 | 용도 |")
        report.append("|--------|------|------|")
        report.append("| `inputDataDaily_FSTX_20120517.csv` | FSTX(일본 선물) 일일 OHLC | 예제 7.1 |")
        report.append("| `inputDataDaily_VX_20120507.csv` | VX(변동성 선물) 72개 계약 | VX-ES 전략 |")
        report.append("| `VIX.csv` | CBOE VIX 지수 | VX-ES 전략 |")
        report.append("| `inputDataDaily_ES_20120507.csv` | ES(S&P 500 선물) 연속 계약 | VX-ES 전략 |\n")

        # 3. 분석 1
        report.append("## 3. 분석 1: FSTX 오프닝 갭 전략 (예제 7.1)\n")
        report.append("### 방법론\n")
        report.append("- 90일 롤링 종가-종가 수익률 표준편차 산출")
        report.append("- 시가가 전일 고가 x (1 + 0.1 x sigma) 이상이면 롱")
        report.append("- 시가가 전일 저가 x (1 - 0.1 x sigma) 이하면 숏")
        report.append("- 당일 종가에 청산 (일중 전략)\n")

        if 'opening_gap' in self.results:
            r = self.results['opening_gap']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 7.5% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 0.49 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | -23.4% |")
            report.append(f"| Max DDD | {r['maxDDD']}일 | 789일 |")
            report.append(f"| 총 거래 횟수 | {r['n_trades']} | - |")
            report.append(f"| 롱/숏 | {r['n_longs']}/{r['n_shorts']} | - |\n")
            report.append("![FSTX 갭](figures/ch7_opening_gap.png)\n")

        # 4. 분석 2
        report.append("## 4. 분석 2: VX-ES 롤 수익률 전략\n")
        report.append("### 방법론\n")
        report.append("- VX 근월 선물 - VIX 지수 차이를 만기까지 남은 일수로 나누어 일일 롤 수익률 산출")
        report.append("- 만기 40-10일 전 구간에서만 거래")
        report.append("- 콘탱고 (dailyRoll > 0.1): VX 숏 + ES 숏")
        report.append("- 백워데이션 (dailyRoll < -0.1): VX 롱 + ES 롱")
        report.append("- 포인트 가치: VX x $1,000, ES x $50\n")

        if 'vx_es' in self.results:
            r = self.results['vx_es']
            report.append("### 결과\n")
            report.append("| 지표 | 값 | 책 기대값 |")
            report.append("|------|-----|----------|")
            report.append(f"| APR | {r['apr']*100:.2f}% | 37.8% |")
            report.append(f"| Sharpe Ratio | {r['sharpe']:.4f} | 2.12 |")
            report.append(f"| Max Drawdown | {r['maxDD']*100:.2f}% | -43.4% |")
            report.append(f"| Max DDD | {r['maxDDD']}일 | 73일 |\n")
            report.append("**핵심 통찰**: VX 콘탱고 구조가 지속적이므로 롤 수익률 수확이 가능하지만,")
            report.append("시장 급변 시 극심한 낙폭 리스크 존재.\n")
            report.append("![VX-ES 전략](figures/ch7_vx_es_rollreturn.png)\n")

        # 5. 종합 비교
        report.append("## 5. 전략 종합 비교\n")
        report.append("| 전략 | APR | Sharpe | MaxDD | 특성 |")
        report.append("|------|-----|--------|-------|------|")
        if 'opening_gap' in self.results:
            r = self.results['opening_gap']
            report.append(f"| FSTX Opening Gap | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | {r['maxDD']*100:.1f}% | 일중, 단순 |")
        if 'vx_es' in self.results:
            r = self.results['vx_es']
            report.append(f"| VX-ES Roll Return | {r['apr']*100:.2f}% | {r['sharpe']:.2f} | {r['maxDD']*100:.1f}% | 다일, 복잡 |")
        report.append("")

        # 6. 결론
        report.append("## 6. 결론 및 권고사항\n")
        report.append("### 핵심 발견\n")
        report.append("1. **오프닝 갭**: 모멘텀 신호로서 유효하나, 단독 전략으로는 낮은 샤프 비율")
        report.append("2. **VX 롤 수익률**: 구조적 콘탱고에서 높은 수익률 가능하나 테일 리스크 극심")
        report.append("3. **일중 전략 한계**: 거래비용과 슬리피지가 수익의 상당 부분을 잠식할 수 있음\n")
        report.append("### 주의사항\n")
        report.append("- **VX 스파이크 리스크**: 시장 급락 시 VX가 급등하여 숏 포지션에 큰 손실")
        report.append("- **실행 리스크**: 시가 주문의 슬리피지가 갭 전략 수익을 감소시킬 수 있음")
        report.append("- **데이터 주파수**: 진정한 일중 전략은 틱/분 단위 데이터 필요\n")

        report_path = REPORT_DIR / "chapter7_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"  ✓ 리포트 저장: {report_path}")
        print()

    def run(self):
        """전체 분석 오케스트레이션"""
        print("\n" + "🔬" * 30)
        print("  Chapter 7: 일중 모멘텀 전략 - 종합 분석")
        print("🔬" * 30 + "\n")

        self.load_data()
        self.analyze_opening_gap()
        self.analyze_vx_es_rollreturn()
        self.generate_report()

        print("=" * 60)
        print("✅ Chapter 7 분석 완료!")
        print(f"   리포트: reports/chapter7_report.md")
        print(f"   차트: {len(self.figures)}개 생성")
        for fig_name, fig_desc in self.figures:
            print(f"     - {fig_name}: {fig_desc}")
        print("=" * 60)


if __name__ == "__main__":
    analyzer = Chapter7Analyzer()
    analyzer.run()
