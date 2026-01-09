# 프로젝트 코드 및 제출 파일 분석 리포트

`/home/restful3/workspace/ml4t/source/kaggle/stocks-return-prediction-v-2` 폴더 내에 생성된 주요 제출 파일들과, 해당 파일을 생성하는 데 사용된 코드를 상세히 설명합니다. (Phase 4는 제외되었습니다)

---

## 1. submission_baseline.csv (Baseline)
**관련 코드**: `baseline.py`

### 개요
프로젝트의 시작점으로, 가장 기본적인 LightGBM 회귀 모델을 구현한 코드입니다.
- **목표**: 파이프라인(데이터 로드 -> 전처리 -> 학습 -> 추론 -> 제출) 정상 작동 확인
- **결과**: Score 0.05707

### 주요 로직
1.  **데이터 전처리**:
    *   `f_3`: 범주형 변수 처리.
    *   `f_5`: 상하위 1% Outlier Clipping.
2.  **피처**:
    *   이동평균(Rolling Mean), 시차 변수(Lag), 종목별/섹터별 통계량 등 기본적인 시계열 피처.
3.  **모델**:
    *   `LGBMRegressor` 사용.
    *   **Objective**: RMSE (평균 제곱근 오차) - 절대적인 수익률 값을 맞추도록 학습.

---

## 2. submission_rank_ic.csv (Rank IC Optimization)
**관련 코드**: `baseline_rank_ic.py`

### 개요
Baseline 모델에 대회 평가지표인 **Rank IC**를 고려한 피처를 추가하고 모델 사이즈를 키운 버전입니다.
- **목표**: 팩터의 설명력을 높이고 평가 지표와의 연관성 강화
- **결과**: Score 0.06081 (Baseline 대비 소폭 상승)

### 주요 변경점
1.  **피처 엔지니어링 강화**:
    *   **Cross-Sectional Rank**: "오늘 날짜 기준 이 종목의 거래량 순위는?" (`rank_f_4` 등)과 같은 상대적 순위 변수를 추가. 이는 Rank IC 지표에 직접적인 도움을 줍니다.
    *   **Volatility**: 변동성 지표 추가.
2.  **모델 용량 증대**:
    *   `max_depth`, `num_leaves` 등을 키워 더 복잡한 패턴을 학습하도록 설정.

---

## 3. submission_regressor.csv (Improved Regressor)
**관련 코드**: `solution_regressor.py`

### 개요
교차 검증 전략을 개선하여 과적합을 방지하려 한 시도입니다.
- **목표**: **Purged CV** 도입으로 미래 정보 누수(Look-ahead Bias) 차단
- **결과**: Score 0.06246

### 주요 변경점
1.  **Purged Group Time Series Split**:
    *   Train 기간과 Validation 기간 사이에 `gap=20`일의 공백을 두어, 학습 데이터 정보가 검증 데이터에 유출되는 것을 엄격히 차단했습니다.
2.  **모델 안정성**:
    *   엄격한 검증을 통해 얻은 점수이므로, 리더보드 점수와의 괴리가 줄어듭니다. 절대 점수는 크게 오르지 않았지만 모델의 신뢰도는 상승했습니다.

---

## 4. submission_phase2.csv (Target Rank Transformation)
**관련 코드**: `solution_phase2.py`

### 개요
**가장 큰 성능 향상**을 이뤄낸 핵심 코드입니다. "값을 맞추는 것"에서 "**순위를 맞추는 것**"으로 패러다임을 전환했습니다.
- **목표**: 평가 지표(Spearman Correlation)와 손실 함수의 불일치 해결
- **결과**: Score **0.08720** (대폭 상승)

### 주요 로직 (핵심)
1.  **Target Transformation**:
    ```python
    # 수익률(y)을 그대로 학습하지 않고, 날짜별 백분위 순위(0~1)로 변환
    train_data['y_rank'] = train_data.groupby('date')['y'].transform(lambda x: x.rank(pct=True))
    ```
    *   모델이 -5% 수익률을 예측하는 대신, "하위 20%"임을 예측하도록 합니다.
2.  **5-Seed Ensemble**:
    *   Random Seed를 5개로 달리하여 학습한 뒤 평균을 내어 분산을 줄였습니다.

---

## 5. submission_phase3.csv (Interaction & Bagging)
**관련 코드**: `solution_phase3.py`

### 개요
Phase 2의 성공 방식을 유지하면서, 변수 간 상호작용을 추가하여 디테일을 잡은 현재 **Best 모델**입니다.
- **목표**: 파생 변수를 통한 정보량 증대 및 앙상블 강화
- **결과**: Score **0.08957** (최고 기록)

### 주요 변경점
1.  **Interaction Features**:
    *   단순 변수가 아닌 변수 간의 결합을 피처로 사용했습니다.
    *   예: `f_0 * f_5` (가격 * 기술지표), `f_0 / f_4` (가격 대비 거래량) 등 경제적 의미를 부여한 변수 추가.
2.  **10-Seed Bagging**:
    *   앙상블 개수를 5개에서 10개로 늘려 예측의 안정성을 극대화했습니다.

---

## 요약

| 제출 파일 | 관련 코드 | 주요 전략 | Score | 비고 |
|---|---|---|---|---|
| `submission_baseline.csv` | `baseline.py` | Basic RMSE Regressor | 0.057 | 시작점 |
| `submission_rank_ic.csv` | `baseline_rank_ic.py` | + Rank Features | 0.060 | 피처 개선 |
| `submission_regressor.csv` | `solution_regressor.py` | + Purged CV | 0.062 | 검증 개선 |
| `submission_phase2.csv` | `solution_phase2.py` | **Target Rank Transfrom** | **0.087** | **핵심 도약** |
| `submission_phase3.csv` | `solution_phase3.py` | **Interaction + 10 Seeds** | **0.089** | **Best** |
