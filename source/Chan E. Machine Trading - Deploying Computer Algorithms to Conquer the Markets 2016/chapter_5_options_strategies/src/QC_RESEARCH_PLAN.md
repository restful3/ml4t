# QuantConnect Cloud Research Platform: Chapter 5 Replication & 2015–2026 Out-of-Sample (OOS) Plan

## 1. 개요 및 배경 (Executive Summary)

본 문서는 Ernest P. Chan의 저서 *Machine Trading* (2017) 제5장 **「Options Strategies」**에 수록된 5가지 핵심 옵션 및 변동성 퀀트 전략을 **QuantConnect Cloud Research Platform (QuantBook)** 상에서 완전히 재현(In-Sample: 2004–2015)하고, 교재 출간 이후의 최신 시장 데이터인 **2015년부터 2026년까지의 진정한 표본 외(Out-of-Sample, OOS) 기간**에 걸쳐 스트레스 테스트 및 확장 분석을 수행하기 위한 체계적 실행 계획서입니다.

---

## 2. 분석 대상 전략 및 데이터 맵핑

| 번호 | 분석 전략 (Strategy) | 교재 원본 데이터 (In-Sample: 2004~2015) | QC Cloud Research 데이터 (OOS: 2015~2026) | 핵심 분석 목표 |
|---|---|---|---|---|
| **Strategy 1** | **VX 숏 vs SPY 롱 & 동적 헤징** | CBOE VX 선물, SPY, XIV ETN (2010~2015) | `Futures.Indices.VIX` (VX), `SPY`, `SVXY` (2018 Volmageddon 이후 0.5x 인버스) 및 연속 선물 | Kelly 레버리지 비교, 15분 정산 시차 편향(Look-ahead) 통제, 칼만 필터 동적 베타 헤지 성과 및 2018 Volmageddon 생존성 검증 |
| **Strategy 2** | **GARCH(1,2) 변동성 예측 & VXX 역추종** | SPY 일별 로그수익률, VXX ETN (2011~2015) | `SPY` Equity, `VXX` / `VIXY` ETN, `arch` 패키지 롤링 GARCH(1,2) | 조건부 분산 예측치와 VXX 가격 방향 일치율(In-sample 35.07%)의 OOS 지속성 검증 및 RV(t+1)-RV(t) 역추종 매매 성과 분석 |
| **Strategy 3** | **이벤트 기반 원유 옵션 전략 (EIA)** | EIA 원유 재고 발표일정, CL 선물, LO 옵션 틱 | `Futures.Energies.CrudeOilWTI` (CL) 및 `OptionUniverse` / `FutureOption` (LO) | 매주 수요일 10:30 ET 발표 전후 롱 스트래들(Vol Crush 손실) vs 목~수 OTM 숏 스트랭글 프리미엄 수취 성과 및 2020년 마이너스 유가 쇼크 스트레스 테스트 |
| **Strategy 4** | **감마 스캘핑 (CL / LO)** | 5% OTM LO 스트랭글 + CL 선물 델타 헤징 | CL 선물 + CL Future Options 체인, 이산적(1%) 델타 리밸런싱 시뮬레이션 | 롱 스트랭글 헤지 하에서 델타 중립을 유지하기 위한 선물 고가매도/저가매수 스캘핑 수익과 이산 리밸런싱 마찰비용(0, 1, 5 bps) 민감도 분석 |
| **Strategy 5** | **내재변동성 횡단면 & 분산 거래** | S&P 500 개별 주식 옵션 vs SPX/SPY 지수 옵션 | S&P 500 상위 20개 대형주 `OptionUniverse` vs `SPY` `OptionUniverse` | 개별 IV 간의 횡단면 Z-score 평균회귀 페어 매매 및 개별 IV 합 vs 지수 IV 스프레드 분산 거래(Dispersion Trading), 2020/2022 시장 위기 시 상관관계 점프(Correlation Jump) 리스크 감사 |

---

## 3. Jupyter Notebook 모듈 아키텍처

생성되는 단일 주피터 노트북(`qc_cloud_research_ch5_replication_and_oos.ipynb`)은 QuantConnect Cloud Research 환경의 `QuantBook` 커널에서 즉시 실행될 수 있도록 아래 8개 모듈로 설계됩니다.

```text
qc_cloud_research_ch5_replication_and_oos.ipynb
├── Module 0: [Setup] QuantBook 초기화, 라이브러리 임포트, 공통 분석 & 시각화 함수 정의
├── Module 1: [Data Ingestion] In-Sample (2004-2015) vs OOS (2015-2026) 데이터 파이프라인
├── Module 2: [Strategy 1] Short VX vs Long SPY & 칼만 필터 동적 헤지 (Volmageddon 검증)
├── Module 3: [Strategy 2] GARCH(1,2) 조건부 분산 예측 및 VXX 방향성 역설 (RV vs IV)
├── Module 4: [Strategy 3] EIA 주간 원유 재고 이벤트 기반 스트래들 vs 숏 스트랭글
├── Module 5: [Strategy 4] CL/LO 감마 스캘핑 및 이산적 델타 리밸런싱·마찰 비용 분석
├── Module 6: [Strategy 5] 개별 옵션 횡단면 IV 평균회귀 및 분산 거래 (Dispersion)
└── Module 7: [Summary & Audit] 2004-2015 vs 2015-2026 종합 성과 매트릭스 및 레짐별 결론
```

---

## 4. 모듈별 세부 알고리즘 및 구현 사양

### Module 0: 초기화 및 환경 설정 (Setup & Utilities)
- **QC 클라우드 네이티브 임포트**:
  ```python
  from AlgorithmImports import *
  import numpy as np
  import pandas as pd
  import scipy.stats as stats
  from scipy.optimize import brentq
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  from arch import arch_model
  
  # QuantBook 인스턴스화
  qb = QuantBook()
  ```
- **시간대 원칙**: 모든 타임스탬프는 `America/New_York` (ET) 기준으로 통일.
- **공통 헬퍼 함수**:
  - `calc_performance_metrics(returns_series)`: CAGR, Annualized Volatility, Sharpe Ratio, Max Drawdown, Calmar Ratio, Win Rate 자동 계산.
  - `black_scholes(S, K, T, r, sigma, option_type)`: 가격 및 Greeks(Delta, Gamma, Vega, Theta) 벡터 연산.
  - `implied_volatility(price, S, K, T, r, option_type)`: Brent's Method 기반 안정적 IV 역산.

---

### Module 1: 데이터 인제스천 파이프라인 (Data Pipeline)
- **지수 & 주식 ETF**: `SPY`, `VXX`, `SVXY`, `XIV` (과거 이력 접근)
- **선물 계약**:
  - VIX 선물: `Futures.Indices.VIX` (일별 바, `DataNormalizationMode.BACKWARDS_RATIO`)
  - S&P 500 E-mini: `Futures.Indices.SP500EMini`
  - WTI 원유: `Futures.Energies.CrudeOilWTI` (분봉 및 일봉)
- **옵션 체인**:
  - `SPY` 주식 옵션 체인 (`qb.add_option("SPY")` + `set_filter(-5, 5, 0, 60)`)
  - `CL` 선물 옵션 체인 (`qb.add_future_option(...)`)
- **데이터 캐싱 메커니즘**: `qb.object_store`를 활용하여 추출된 대용량 시계열을 Parquet으로 영속화하여 중복 쿼리 시간 단축.

---

### Module 2: Strategy 1 — Short VX vs Long SPY & 칼만 필터 동적 헤지
1. **표본 분할**:
   - In-Sample: 2004-04-05 ~ 2015-08-19
   - Out-of-Sample: 2015-08-20 ~ 2026-08-01 (2018년 2월 5일 Volmageddon 및 2020년 팬데믹 포함)
2. **Kelly 레버리지 최적화**:
   - $f^* = \frac{\mu - r}{\sigma^2}$ 공식을 산출하여 SPY와 VX에 각각 최적 비중 부과 후 일별 복리 수익 곡선 산출.
3. **타임라인 시차 엄격 통제**:
   - 16:15 ET에 확정되는 VX 롤 신호를 $t+1$일 16:00 ET ES 거래에 반영하여 미래 정보 편향(Look-ahead bias) 원천 차단.
4. **칼만 필터(Kalman Filter) 상태 공간 모형**:
   - 측정 방정식: $y_t = \beta_t x_t + \epsilon_t, \quad \epsilon_t \sim N(0, R)$
   - 상태 전이 방정식: $\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$
   - 매일 실시간으로 동적 헤지 비율 $\beta_t$를 업데이트하여 XIV(또는 SVXY)와 SPY 포트폴리오의 Calmar Ratio 개선 효과 검증.
5. **OOS Volmageddon 분석**:
   - 2018년 2월 5일 VIX 일일 100%+ 폭등 당시 고정 헤지 vs 칼만 필터 vs 숏 선물 전략의 Drawdown 충격 비교.

---

### Module 3: Strategy 2 — GARCH(1,2) 변동성 예측 및 VXX 역추종 전략
1. **GARCH(1,2) 조건부 분산 모델링**:
   $$\sigma_t^2 = \omega + \alpha_1 \sigma_{t-1}^2 + \beta_1 r_{t-1}^2 + \beta_2 r_{t-2}^2$$
   - `arch_model` 라이브러리를 통해 SPY 로그수익률의 시계열 적합.
   - 고정 파라미터(In-sample 피팅) vs 252일 롤링 윈도우 파라미터 추정 비교.
2. **실현변동성(RV) vs 내재변동성(VXX) 방향성 일치도 감사**:
   - $\text{sign}(\sigma_{t+1} - \sigma_t)$ 와 $\text{sign}(r_{VXX, t+1})$ 간의 일별 부호 일치율 산출.
   - In-Sample (35.07%) 재현 및 OOS (2015–2026) 기간 일치율 추적.
3. **RV(t+1) - RV(t) 역추종 백테스트**:
   - 신호: $S_t = -\text{sign}(\sigma_{t+1} - \sigma_t)$ (변동성 증가 예측 시 Short VXX, 감소 예측 시 Long VXX).
   - In-sample (CAGR 81%, Calmar 1.9) 대비 OOS 기간의 실제 누적 성과 및 슬리피지 감안 후 샤프 비율 비교.

---

### Module 4: Strategy 3 — EIA 주간 원유 재고 이벤트 기반 옵션 분석
1. **이벤트 타임라인 매핑**:
   - 매주 수요일 10:30 ET (EIA Weekly Petroleum Status Report) 캘린더 생성.
2. **이벤트 직전 롱 스트래들(Long Straddle)의 참패 원인 분석**:
   - 수요일 10:29 ET ATM 콜/풋 매수 $\rightarrow$ 10:35 ET 청산 시의 내재변동성 급락(Vol Crush) 및 호가 스프레드(Bid-Ask) 잠식 정량화.
3. **대안 전략: 평일 숏 스트랭글 (Short 5% OTM Strangle)**:
   - 목요일 09:00 ET 진입 $\rightarrow$ 차주 수요일 10:29 ET 전량 청산.
   - 2015–2026 OOS 백테스트 및 2020년 4월 WTI 원유 선물 마이너스 유가 사태($-37.63/bbl) 시의 극단적 마진콜 위험 분석.

---

### Module 5: Strategy 4 — CL/LO 감마 스캘핑 및 미시구조 시뮬레이션
1. **전략 구조**:
   - 5% OTM 원유 옵션 스트랭글 매수(양의 감마 확보) + 기초 원유 선물(CL) 델타 헤징.
2. **이산적 리밸런싱 알고리즘**:
   - 선물 가격이 진입가 대비 $\pm 0.5\%$, $\pm 1.0\%$, $\pm 2.0\%$ 임계값(Threshold)을 돌파할 때마다 포트폴리오 델타를 0으로 리밸런싱.
3. **경로 의존성 및 거래비용(Friction) 분석**:
   - 주중(목요일 오전 ~ 금요일 오후) 매매 사이클 검증.
   - 거래 수수료 및 슬리피지(0 bps, 1 bps, 5 bps) 부과 시 스캘핑 차익이 옵션 시간감소(Theta) 비용을 방어할 수 있는지의 손익 분기점(BEP) 도출.

---

### Module 6: Strategy 5 — 내재변동성 횡단면 평균회귀 & 분산 거래 (Dispersion)
1. **횡단면 IV 페어 트레이딩**:
   - S&P 500 상위 20개 대형주(AAPL, MSFT, NVDA, AMZN, GOOGL 등)의 30일 ATM IV Z-score 계산:
     $$Z_{i,t} = \frac{IV_{i,t} - \mu_{IV,i}}{\sigma_{IV,i}}$$
   - 고평가 IV 종목 매도 + 저평가 IV 종목 매수 델타 중립 페어 백테스트.
2. **분산 거래(Dispersion Trading) 아키텍처**:
   - 개별 주식 옵션 매수(Long Basket) + SPY 지수 옵션 매도(Short Index).
   - 평상시 상관관계 프리미엄 수취 성과와 2020년 팬데믹 / 2022년 금리 인상기 **상관관계 1 수렴(Correlation Jump)** 시의 최대 낙폭 분석.

---

### Module 7: 종합 성과 비교 및 레짐별 결론 (Summary & Audit)
1. **종합 성과 지표 비교표 (In-Sample vs Out-of-Sample)**:
   - 각 5개 전략별 CAGR, Annualized Volatility, Sharpe Ratio, Max Drawdown, Calmar Ratio 매트릭스 도출.
2. **레짐별 스트레스 테스트 감사**:
   - 2018 Volmageddon (VIX ETN 붕괴)
   - 2020 COVID-19 Crash & Negative Oil
   - 2022 Fed Rate Hike (고금리 환경의 옵션 이자율 효과)
   - 2024–2026 0DTE 옵션 확산 및 현대 변동성 시장 구조
3. **퀀트 트레이더를 위한 핵심 결론 및 교재 가정의 비판적 총평**.

---

## 5. 실행 및 클라우드 배포 절차

1. **저장소 파일 위치**:
   - `source/Chan E. Machine Trading - Deploying Computer Algorithms to Conquer the Markets 2016/chapter_5_options_strategies/src/qc_cloud_research_ch5_replication_and_oos.ipynb`
   - `source/Chan E. Machine Trading - Deploying Computer Algorithms to Conquer the Markets 2016/chapter_5_options_strategies/src/QC_RESEARCH_GUIDE.md`
2. **QuantConnect Cloud 업로드 방법**:
   - QuantConnect 웹 콘솔(https://www.quantconnect.com) 접속 $\rightarrow$ **Research** 탭 클릭 $\rightarrow$ **New Project** 생성 $\rightarrow$ `qc_cloud_research_ch5_replication_and_oos.ipynb` 업로드 $\rightarrow$ Run All Cells.
3. **로컬 무결성 사전 검증**:
   - `nbformat` 및 Python AST를 통한 문법 에러 0건 확인 후 배포.
