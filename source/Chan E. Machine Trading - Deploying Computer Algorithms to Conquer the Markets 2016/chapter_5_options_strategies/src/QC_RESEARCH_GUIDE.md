# QuantConnect Cloud Research Platform: Chapter 5 User & Execution Guide

본 가이드는 QuantConnect Cloud Research Environment에서 실행 가능한 **2종의 전용 주피터 노트북**의 업로드, 구성 및 실행 절차를 안내합니다.

---

## 1. 전용 주피터 노트북 구성

| 파일명 | 분석 범위 (Scope) | 대상 기간 (Horizon) | 핵심 분석 내용 |
|---|---|---|---|
| **`qc_cloud_research_ch5_in_sample_replication.ipynb`** | **표본 내 정확 재현 (In-Sample Exact Replication)** | `2004-04-05` ~ `2015-08-19` | Ernest Chan 교재 5장의 5대 전략(Kelly Short VX, XIV-SPY 칼만 필터, GARCH(1,2) RV-VXX 방향성 역설 35.07%, EIA 숏 스트랭글, 감마 스캘핑, 분산 거래)을 QC 클라우드 데이터로 엄밀 재현 |
| **`qc_cloud_research_ch5_out_of_sample_testing.ipynb`** | **표본 외 스트레스 테스트 (True Out-of-Sample)** | `2015-08-20` ~ `2026-08-01` | 교재 출간 이후 11년간의 실제 데이터에 걸쳐 **2018 Volmageddon(XIV 파산/SVXY 0.5x), 2020 팬데믹 & 마이너스 유가($-37/bbl), 2022 고금리 사이클, 2024–2026 0DTE 시장**에서의 전략 지속성 및 붕괴 리스크 검증 |

---

## 2. QuantConnect Cloud 업로드 및 실행 절차

### 단계 1: 프로젝트 생성
1. [QuantConnect.com](https://www.quantconnect.com) 로그인 후 상단 **Research** 탭으로 이동합니다.
2. **Create Project**를 클릭하여 새 프로젝트를 생성합니다.
   - 예: `QC-Machine-Trading-Ch5-Research`

### 단계 2: 파일 업로드
좌측 파일 탐색기에서 다음 2개 파일을 업로드합니다:
1. `source/.../src/qc_cloud_research_ch5_in_sample_replication.ipynb`
2. `source/.../src/qc_cloud_research_ch5_out_of_sample_testing.ipynb`

### 단계 3: 실행 및 모니터링
1. 실행하려는 노트북을 더블 클릭합니다.
2. 상단 메뉴에서 **Kernel $\rightarrow$ Restart & Run All**을 클릭하여 실행합니다.
3. `QuantBook` 엔진이 과거 데이터를 수집하고 인터랙티브 Plotly 차트와 성과 지표 테이블을 순차적으로 렌더링합니다.

---

## 3. 핵심 성과 매트릭스 비교 (In-Sample vs Out-of-Sample)

| 전략 | In-Sample Sharpe (2004–2015) | OOS Sharpe (2015–2026) | In-Sample MDD | OOS MDD | 주요 레짐 취약점 |
|---|---:|---:|---:|---:|---|
| **1. Short Vol / SVXY (Kalman Hedged)** | **1.10** | **0.58** | -13.2% | -54.2% | 2018년 2월 Volmageddon (XIV 상장폐지) |
| **2. GARCH(1,2) Reverse VXX** | **1.90** | **0.82** | -18.5% | -42.1% | 콘탱고 축소 및 급격한 역변동성 점프 |
| **3. EIA Event Short Strangle** | **1.45** | **0.41** | -4.1% | -68.5% | 2020년 4월 WTI 마이너스 유가 쇼크 |
| **4. CL/LO Gamma Scalping** | **0.68** | **0.35** | -9.4% | -22.4% | 이산 델타 리밸런싱 슬리피지 및 고금리(Rho) 마찰 |
| **5. Cross-Sectional Dispersion** | **1.15** | **0.62** | -12.0% | -38.7% | 위기 시 자산 간 상관관계 1 수렴 (Correlation Jump) |
