# Baseline Model Report

Generated on: 2026-01-05 08:56:43

## 1. 코드 설명

이 코드는 주가 수익률 예측을 위한 Baseline 모델 학습 및 추론 스크립트입니다. 
LightGBM Regressor를 사용하며, 시계열 교차 검증(Time-Series Cross Validation)을 수행합니다.

## 2. 작업 내용



### 2.1 데이터 로드

- Train data shape: (4670279, 10)

- Test data shape: (4879631, 9)

### 2.2 데이터 전처리

- f_3 변수: Numeric 변환 및 결측치 -1로 대체 후 Category 타입으로 변환

- f_5 변수: 이상치 Clipping (1% ~ 99% quantile 적용, range: 0.9547 ~ 1.0455)

### 2.3 Feature Engineering

기본 변수 외에 다음과 같은 파생 변수를 생성하였습니다.

- price_ratio

- volume_normalized

- date_mod_5

- f_0_ma_5

- f_5_ma_5

- f_0_ma_10

- f_5_ma_10

- stock_mean_f_0

- stock_std_f_0

- sector_mean_f_0


총 Feature 개수: 17개

## 3. 모델 설명



LightGBM Regressor를 사용하였습니다. 주요 파라미터는 다음과 같습니다.

```python

n_estimators=500
learning_rate=0.05
max_depth=8
num_leaves=63
colsample_bytree=0.8
subsample=0.8
random_state=42

```

## 4. 학습 결과



### 4.1 교차 검증 (Cross-Validation) 결과

TimeSeriesSplit (n_splits=5)을 사용하여 검증하였습니다.

| Fold | RMSE | MAE |
|---|---|---|

| 1 | 0.8577 | 0.5736 |

| 2 | 0.8409 | 0.5624 |

| 3 | 0.8505 | 0.5663 |

| 4 | 0.8413 | 0.5627 |

| 5 | 0.8401 | 0.5629 |


**Mean CV RMSE: 0.8461 (+/- 0.0069)**

### 4.2 변수 중요도 (Top 10)

전체 데이터로 학습한 모델 기준 상위 10개 중요 변수입니다.

| Rank | Feature | Importance |
|---|---|---|
| 1 | f_3 | 4602.0000 |
| 2 | sector_mean_f_0 | 3655.0000 |
| 3 | stock_std_f_0 | 2516.0000 |
| 4 | f_4 | 2499.0000 |
| 5 | stock_mean_f_0 | 2257.0000 |
| 6 | f_0_ma_10 | 2033.0000 |
| 7 | f_0_ma_5 | 1649.0000 |
| 8 | f_5_ma_10 | 1645.0000 |
| 9 | f_5_ma_5 | 1545.0000 |
| 10 | f_2 | 1505.0000 |


## 5. Test 결과 및 제출



### 5.1 Prediction Statistics

- Mean: -0.0119

- Std: 0.1462

- Min: -4.5436

- Max: 8.4927

### 5.2 제출 파일

제출 파일이 'submissions/submission_baseline.csv'에 저장되었습니다.