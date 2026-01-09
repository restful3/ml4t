# Rank IC Optimization Model Report

**생성일시:** 2026-01-05 09:01:55

## 1. 코드 개요

이 스크립트는 **Rank IC (순위 상관계수)** 최적화를 목표로 개선된 모델입니다. 
주요 특징으로는 **순위 변수(Rank Features)** 도입, **변동성(Volatility)** 변수 추가, 그리고 **Rank IC 기반 검증**이 있습니다.

## 2. 데이터 준비



### 2.1 데이터 로드

- Train data shape: (4670279, 10)

- Test data shape: (4879631, 9)

### 2.2 데이터 전처리

- f_3: 결측치 -1 처리 및 Category 타입 변환 (LGBM 전용)

- f_5: 이상치 Clipping 적용 (1% ~ 99% 구간, 0.9547 ~ 1.0455)

### 2.3 Feature Engineering (변수 생성)

총 25개의 Feature를 사용하였습니다. 주요 파생 변수는 다음과 같습니다.

- **Cross-Sectional Rank Features (핵심):** 날짜별 각 종목의 상대적 순위 (rank_f_0, rank_f_4 등)

- **Volatility Features:** 주가 변동성 (volatility_5, volatility_10 등)

- **Trends:** 이동평균 (f_0_ma_5, f_0_ma_20 등)

- **Stock Stats:** 종목별 평균 및 표준편차

- **Date Cyclic:** 날짜 주기성 (date_mod_5)

## 3. 모델 학습



### 3.1 모델 설정

Rank IC 향상을 위해 모델 복잡도를 높였습니다.

```python

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

```

### 3.2 교차 검증 (Cross-Validation) 결과

TimeSeriesSplit (n_splits=5)을 사용하여 검증하였습니다.

| Fold | RMSE | Rank IC (Spearman) |
|---|---|---|

| 1 | 0.8580 | **0.1108** |

| 2 | 0.8408 | **0.1212** |

| 3 | 0.8490 | **0.1328** |

| 4 | 0.8395 | **0.1308** |

| 5 | 0.8381 | **0.1375** |


**Mean CV Rank IC: 0.1266** (+/- 0.0095)

### 3.3 변수 중요도 (Top 15)

Rank 변수와 변동성 변수가 상위권을 차지하는지 확인합니다.

| Rank | Feature | Importance |
|---|---|---|
| 1 | f_3 | 11908 |
| 2 | stock_std_f_0 | 8644 |
| 3 | stock_mean_f_0 | 8367 |
| 4 | f_0_ma_20 | 7333 |
| 5 | f_4 | 7220 |
| 6 | volatility_20 | 6788 |
| 7 | rank_f_4 | 6667 |
| 8 | volatility_10 | 5696 |
| 9 | volatility_5 | 5498 |
| 10 | f_0_ma_5 | 5402 |
| 11 | f_0_ma_10 | 5157 |
| 12 | rank_f_5 | 4897 |
| 13 | rank_price_ratio | 4854 |
| 14 | rank_f_2 | 4368 |
| 15 | rank_f_0 | 4272 |


## 4. 최종 제출



### 4.1 예측 결과 통계

- Mean: -0.0104

- Std: 0.1643

- Min: -4.3929

- Max: 9.7440

### 4.2 저장 파일

제출 파일이 'submissions/submission_rank_ic.csv'에 저장되었습니다.