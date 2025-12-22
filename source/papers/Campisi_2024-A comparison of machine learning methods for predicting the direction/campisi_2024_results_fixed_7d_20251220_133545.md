# 머신러닝 기반 S&P 500 방향성 예측 성능 분석 (FIXED VERSION)

**생성일시**: 2025-12-20 14:51:32

**예측 기간**: 7일 후 방향성

**사용 방법론**: Volatility Indices 기반 머신러닝 모델 (Campisi et al., 2024)

**주요 개선사항**:
- ✅ Data Leakage 문제 해결 (Pipeline 사용)
- ✅ Diebold-Mariano 통계 검정 추가
- ✅ Performance 최적화 (Refit Frequency)

---

## 1. 데이터 요약

- **수집 기간**: 2011-01-03 ~ 2022-07-29
- **총 관측치 수**: 2,913
- **변수 수**: 10
- **Walk-Forward CV 이터레이션**: 755개
- **Refit Frequency**: 매 1일
- **데이터 소스**: Yahoo Finance

## 2. 기술 통계량

| Variable   |    N |    Mean |   St. Dev. |   Skewness |   Kurtosis |   rho1 |   rho2 |   rho3 |    ADF |         JB |
|:-----------|-----:|--------:|-----------:|-----------:|-----------:|-------:|-------:|-------:|-------:|-----------:|
| VIX        | 2913 |  18.145 |      7.353 |      2.532 |     11.379 |  0.967 |  0.945 |  0.921 | -5.33  |  18767     |
| VIX9D      | 2913 |  17.56  |      8.801 |      3.294 |     19.505 |  0.939 |  0.908 |  0.87  | -6.658 |  51273.3   |
| VIX3M      | 2913 |  20.037 |      6.572 |      1.979 |      7.007 |  0.981 |  0.968 |  0.951 | -3.729 |   7834.16  |
| VIX6M      | 2913 |  21.419 |      5.879 |      1.395 |      2.607 |  0.987 |  0.977 |  0.965 | -2.987 |   1764.74  |
| VVIX       | 2913 |  96.772 |     16.883 |      1.153 |      2.859 |  0.946 |  0.899 |  0.856 | -5.954 |   1631.77  |
| SKEW       | 2913 | 128.957 |      9.666 |      0.763 |      0.295 |  0.928 |  0.896 |  0.866 | -4.248 |    292.503 |
| VXN        | 2913 |  20.863 |      7.504 |      1.838 |      5.776 |  0.972 |  0.953 |  0.932 | -4.768 |   5670.55  |
| GVZ        | 2913 |  17.144 |      4.955 |      1.284 |      3.126 |  0.976 |  0.954 |  0.934 | -4.12  |   1980.05  |
| OVX        | 2913 |  37.335 |     18.956 |      5.082 |     44.54  |  0.966 |  0.93  |  0.907 | -4.896 | 252462     |
| RVOL       | 2913 |  14.769 |      9.661 |      3.675 |     20.19  |  0.996 |  0.99  |  0.981 | -6.667 |  55847     |
| Returns7   | 2913 |   0.003 |      0.026 |     -1.436 |      8.696 |  0.844 |  0.704 |  0.545 | -9.758 |  10142     |

## 3. VIF (Variance Inflation Factor)

| Variable   |      VIF |
|:-----------|---------:|
| VIX3M      | 2619.89  |
| VIX6M      | 1277.7   |
| VIX        | 1143.52  |
| VIX9D      |  211.145 |
| VXN        |  158.932 |
| VVIX       |  116.066 |
| SKEW       |   82.123 |
| GVZ        |   30.637 |
| RVOL       |   16.598 |
| OVX        |   14.972 |

## 4. 모델 성능 비교 (Feature Selection 전)

### 4.1 분류 모델

| Model                   |   Accuracy |    AUC |   F-measure |
|:------------------------|-----------:|-------:|------------:|
| Logistic Regression     |     0.604  | 0.5525 |      0.7506 |
| LDA                     |     0.6026 | 0.5491 |      0.7496 |
| Random Forest (Clf)     |     0.547  | 0.6045 |      0.693  |
| Bagging (Clf)           |     0.5232 | 0.6172 |      0.6673 |
| Gradient Boosting (Clf) |     0.5841 | 0.5467 |      0.7265 |

### 4.2 회귀 모델

| Model                   |   Accuracy |    AUC |   F-measure |
|:------------------------|-----------:|-------:|------------:|
| Linear Regression       |     0.5974 | 0.5497 |      0.7415 |
| Ridge Regression        |     0.6    | 0.5488 |      0.7445 |
| Lasso Regression        |     0.6238 | 0.542  |      0.7676 |
| Random Forest (Reg)     |     0.5073 | 0.5599 |      0.6367 |
| Bagging (Reg)           |     0.4993 | 0.5557 |      0.6287 |
| Gradient Boosting (Reg) |     0.5351 | 0.5361 |      0.671  |

## 5. 모델 성능 비교 (Feature Selection 후)

### 5.1 분류 모델

| Model                   |   Accuracy |    AUC |   F-measure |
|:------------------------|-----------:|-------:|------------:|
| Logistic Regression     |     0.6066 | 0.5593 |      0.7527 |
| LDA                     |     0.6066 | 0.5585 |      0.7527 |
| Random Forest (Clf)     |     0.5404 | 0.613  |      0.6882 |
| Bagging (Clf)           |     0.5205 | 0.6255 |      0.6629 |
| Gradient Boosting (Clf) |     0.5762 | 0.5642 |      0.7217 |

### 5.2 회귀 모델

| Model                   |   Accuracy |    AUC |   F-measure |
|:------------------------|-----------:|-------:|------------:|
| Linear Regression       |     0.6172 | 0.5309 |      0.7602 |
| Ridge Regression        |     0.6172 | 0.5309 |      0.7602 |
| Lasso Regression        |     0.6238 | 0.542  |      0.7676 |
| Random Forest (Reg)     |     0.5483 | 0.5054 |      0.6573 |
| Bagging (Reg)           |     0.551  | 0.5083 |      0.6593 |
| Gradient Boosting (Reg) |     0.5616 | 0.5023 |      0.6983 |

## 6. Diebold-Mariano 통계 검정

### 6.1 분류 모델 쌍별 비교

**통계적으로 유의한 차이 (p < 0.05):**

| Model 1             | Model 2                 |   DM Statistic |   p-value | Significant (5%)   |
|:--------------------|:------------------------|---------------:|----------:|:-------------------|
| Logistic Regression | Random Forest (Clf)     |        -4.8375 |    0      | True               |
| Logistic Regression | Bagging (Clf)           |        -5.4625 |    0      | True               |
| Logistic Regression | Gradient Boosting (Clf) |        -2.7413 |    0.0061 | True               |
| LDA                 | Random Forest (Clf)     |        -4.8375 |    0      | True               |
| LDA                 | Bagging (Clf)           |        -5.4625 |    0      | True               |
| LDA                 | Gradient Boosting (Clf) |        -2.7413 |    0.0061 | True               |
| Random Forest (Clf) | Bagging (Clf)           |        -2.242  |    0.025  | True               |
| Random Forest (Clf) | Gradient Boosting (Clf) |         2.8126 |    0.0049 | True               |
| Bagging (Clf)       | Gradient Boosting (Clf) |         3.9369 |    0.0001 | True               |

### 6.2 회귀 모델 쌍별 비교

**통계적으로 유의한 차이 (p < 0.05):**

| Model 1           | Model 2                 |   DM Statistic |   p-value | Significant (5%)   |
|:------------------|:------------------------|---------------:|----------:|:-------------------|
| Linear Regression | Random Forest (Reg)     |        -3.5002 |    0.0005 | True               |
| Linear Regression | Bagging (Reg)           |        -3.3334 |    0.0009 | True               |
| Linear Regression | Gradient Boosting (Reg) |        -3.5264 |    0.0004 | True               |
| Ridge Regression  | Random Forest (Reg)     |        -3.5002 |    0.0005 | True               |
| Ridge Regression  | Bagging (Reg)           |        -3.3334 |    0.0009 | True               |
| Ridge Regression  | Gradient Boosting (Reg) |        -3.5264 |    0.0004 | True               |
| Lasso Regression  | Random Forest (Reg)     |        -3.8171 |    0.0001 | True               |
| Lasso Regression  | Bagging (Reg)           |        -3.6643 |    0.0002 | True               |
| Lasso Regression  | Gradient Boosting (Reg) |        -4.1503 |    0      | True               |

## 7. 모델 성능 요약

### 7.1 Feature Selection 전후 비교

**분류 모델:**
- Feature Selection 전: Logistic Regression (Accuracy: 0.6040)
- Feature Selection 후: Logistic Regression (Accuracy: 0.6066)

**회귀 모델:**
- Feature Selection 전: Lasso Regression (Accuracy: 0.6238)
- Feature Selection 후: Lasso Regression (Accuracy: 0.6238)

### 7.2 전체 최고 성능 모델

**최고 성능**: Lasso Regression
- Accuracy: 0.6238
- AUC: 0.5420
- F-measure: 0.7676

## 8. 주요 인사이트

**1. Feature Selection 효과**
- 전체 평균 Accuracy: 0.5658 → 0.5790
- Feature Selection을 통해 성능 향상 (+0.0132)

**2. 분류 vs 회귀 모델**
- 분류 모델 평균 Accuracy: 0.5701
- 회귀 모델 평균 Accuracy: 0.5865
- 회귀 모델이 방향 예측에 더 효과적

**3. Data Leakage 문제 해결**
- ✅ Standardization을 CV loop 내부로 이동 (Pipeline 사용)
- ✅ Feature Selection을 CV loop 내부로 이동 (SelectFromModel)
- ✅ 이로 인해 원본 코드 대비 성능이 낮아질 수 있으나, 이것이 정확한 out-of-sample 성능임

---

## 9. 시각화

- `figures/correlation_heatmap.png`: 상관관계 히트맵
- `figures/roc_curves.png`: ROC 곡선
- `figures/returns_timeseries.png`: 수익률 시계열
