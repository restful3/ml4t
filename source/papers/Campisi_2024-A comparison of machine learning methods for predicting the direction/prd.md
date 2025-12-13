# PRD: Campisi et al. (2024) 논문 실증 구현

## 프로젝트 개요

**목표**: "A comparison of machine learning methods for predicting the direction of the US stock market on the basis of volatility indices" 논문의 실증 재현

**출력 위치**: `/home/restful3/workspace/ml4t/source/papers/Campisi_2024-A comparison of machine learning methods for predicting the direction/`

**참고 문헌**: Campisi, G., Muzzioli, S., & De Baets, B. (2023). International Journal of Forecasting.

---

## 1. 환경 설정

### 1.1 가상환경 생성
```bash
cd "/home/restful3/workspace/ml4t/source/papers/Campisi_2024-A comparison of machine learning methods for predicting the direction"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.40
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
scipy>=1.11.0
tabulate>=0.9.0
joblib>=1.3.0
tqdm>=4.66.0
```

---

## 2. 데이터 수집

### 2.1 데이터 소스
- **Yahoo Finance (yfinance)**: 모든 변동성 지수 수집 가능 (검증됨)
- **CBOE 직접 다운로드**: Put/Call Ratio (CSV)

### 2.2 수집 대상 변수 (Yahoo Finance 티커 검증됨)

| 변수 | Yahoo Finance Ticker | 설명 | 데이터 가용성 |
|------|---------------------|------|--------------|
| S&P 500 | ^GSPC | 기초 지수 | ✅ 확인됨 |
| VIX | ^VIX | 30일 변동성 지수 | ✅ 확인됨 |
| VIX9D | ^VIX9D | 9일 변동성 지수 | ✅ 확인됨 |
| VIX3M | ^VIX3M | 3개월 변동성 지수 | ✅ 확인됨 |
| VIX6M | ^VIX6M | 6개월 변동성 지수 | ✅ 확인됨 |
| VVIX | ^VVIX | VIX의 변동성 | ✅ 확인됨 |
| VXN | ^VXN | NASDAQ-100 변동성 | ✅ 확인됨 |
| GVZ | ^GVZ | 금 변동성 지수 | ✅ 확인됨 |
| OVX | ^OVX | 원유 변동성 지수 | ✅ 확인됨 |
| SKEW | ^SKEW | S&P 500 왜도 지수 (100-150 범위) | ✅ 확인됨 |

### 2.3 계산/수집 필요 변수
- **PUTCALL**: CBOE CSV 직접 다운로드
  - URL: `https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv`
  - 또는 제외하고 10개 변수로 진행 (논문 재현에 큰 영향 없음)
- **RVOL**: 실현 변동성 (30일 롤링 표준편차 × √252 연율화)
- **Returns30**: 30일 후 로그 수익률 (타겟 변수)

### 2.4 데이터 기간
- **논문 기간**: 2011년 1월 ~ 2022년 7월 (3,040일)
- **실증 기간**: 2011-01-01 ~ 2022-07-31 (논문과 동일)

### 2.5 데이터 수집 코드 예시
```python
import yfinance as yf

tickers = {
    'SP500': '^GSPC',
    'VIX': '^VIX',
    'VIX9D': '^VIX9D',
    'VIX3M': '^VIX3M',
    'VIX6M': '^VIX6M',
    'VVIX': '^VVIX',
    'VXN': '^VXN',
    'GVZ': '^GVZ',
    'OVX': '^OVX',
    'SKEW': '^SKEW'
}

data = yf.download(list(tickers.values()), start='2011-01-01', end='2022-07-31')['Close']
```

---

## 3. 데이터 전처리

### 3.1 결측치 처리
- Forward fill → Backward fill
- 여전히 결측인 행 제거

### 3.2 표준화
- StandardScaler 적용 (평균 0, 표준편차 1)

### 3.3 타겟 변수 생성
```python
# 30일 후 로그 수익률
returns30 = np.log(sp500.shift(-30) / sp500)

# 이진 분류용 타겟 (1: 양수, 0: 음수)
target_binary = (returns30 > 0).astype(int)
```

---

## 4. Feature Selection

### 4.1 Lasso 회귀 기반 변수 선택
- 논문과 동일하게 Lasso 회귀로 중요도 0인 변수 제거
- 예상 제거 변수: VIX9D, VIX3M, VIX6M (다중공선성)

### 4.2 VIF (Variance Inflation Factor) 계산
- VIF > 10인 변수 식별

---

## 5. 모델 구현

### 5.1 분류 모델 (Classification)
| 모델 | scikit-learn 클래스 |
|------|---------------------|
| Logistic Regression | `LogisticRegression` |
| LDA | `LinearDiscriminantAnalysis` |
| Random Forest | `RandomForestClassifier` |
| Bagging | `BaggingClassifier` |
| Gradient Boosting | `GradientBoostingClassifier` |

### 5.2 회귀 모델 (Regression)
| 모델 | scikit-learn 클래스 |
|------|---------------------|
| Linear Regression | `LinearRegression` |
| Ridge Regression | `Ridge` |
| Lasso Regression | `Lasso` |
| Random Forest | `RandomForestRegressor` |
| Bagging | `BaggingRegressor` |
| Gradient Boosting | `GradientBoostingRegressor` |

### 5.3 하이퍼파라미터 (논문 기준)
```python
# Random Forest / Bagging
n_estimators = 500
max_features = 'sqrt'  # mtry = sqrt(p)

# Gradient Boosting
n_estimators = [50, 150]
learning_rate = [0.01, 0.1]
max_depth = [1, 2, 3]
min_samples_split = [5, 10]

# Ridge / Lasso
alpha = np.logspace(-4, 2, 100)
```

---

## 6. 검증 방식

### 6.1 Walk-Forward Validation (Rolling Window)

논문의 검증 방식을 정확히 재현:
- **학습 세트 크기**: 2,128개 (전체의 70%)
- **테스트 예측 수**: 883개
- **Forward-looking 방지**: t 시점 예측 시 t-30까지의 데이터만 사용

```python
from sklearn.model_selection import TimeSeriesSplit

# 커스텀 Walk-Forward Validation
class WalkForwardCV:
    def __init__(self, train_size=2128, gap=30):
        self.train_size = train_size
        self.gap = gap  # forward-looking 방지용 갭

    def split(self, X):
        n_samples = len(X)
        for test_idx in range(self.train_size + self.gap, n_samples):
            train_end = test_idx - self.gap
            train_start = train_end - self.train_size
            if train_start >= 0:
                yield (
                    list(range(train_start, train_end)),
                    [test_idx]
                )

# 사용 예시
cv = WalkForwardCV(train_size=2128, gap=30)
predictions = []
actuals = []

for train_idx, test_idx in cv.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions.append(pred[0])
    actuals.append(y_test[0])
```

### 6.2 scikit-learn TimeSeriesSplit 참고
- 기본 제공되는 `TimeSeriesSplit`은 expanding window 방식
- 논문은 **rolling window** 방식이므로 커스텀 구현 필요

---

## 7. 평가 지표

### 7.1 주요 지표
```python
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
f1 = f1_score(y_true, y_pred)
```

### 7.2 Diebold-Mariano 검정
- 모델 간 예측 정확도 차이의 통계적 유의성 검정

---

## 8. 출력 파일

### 8.1 메인 스크립트
**파일명**: `campisi_2024_replication.py`

```
구조:
├── 데이터 수집 함수
├── 전처리 함수
├── Feature Selection 함수
├── 모델 클래스/함수
├── Walk-Forward Validation 함수
├── 평가 함수
├── 시각화 함수
└── main() 실행
```

### 8.2 결과 리포트
**파일명**: `campisi_2024_results.md`

```markdown
# 실증 결과 리포트

## 1. 데이터 요약
- 수집 기간, 관측치 수, 결측치 처리

## 2. 기술 통계량
- 논문 Table 1과 비교

## 3. Feature Selection 결과
- Lasso 중요도, VIF 값

## 4. 모델 성능 비교
- 분류 모델 (Table 9 재현)
- 회귀 모델 (Table 10 재현)

## 5. 논문 결과와 비교
- 일치/불일치 분석

## 6. 시각화
- ROC 곡선
- 변수 중요도 플롯
```

---

## 9. 구현 순서

1. **환경 설정**: .venv 생성, requirements.txt 설치
2. **데이터 수집**: yfinance로 변동성 지수 다운로드
3. **전처리**: 결측치 처리, 표준화, 타겟 생성
4. **Feature Selection**: Lasso 기반 변수 선택
5. **모델 학습**: Walk-Forward Validation으로 7개 모델 학습
6. **평가**: Accuracy, AUC, F1 계산
7. **결과 저장**: .md 리포트 자동 생성

---

## 10. 예상 이슈 및 대응

| 이슈 | 대응 방안 |
|------|-----------|
| 일부 변동성 지수 데이터 없음 | Yahoo Finance에서 모든 티커 확인됨 ✅ |
| PUTCALL 데이터 수집 어려움 | CBOE CSV 다운로드 또는 제외 (영향 미미) |
| 계산 시간 (883회 × 11모델) | joblib 병렬처리 적용 |
| 논문과 결과 차이 | 데이터 소스 차이 명시, Bloomberg vs Yahoo Finance |

---

## 11. 논문 주요 결과 (재현 목표)

### 11.1 Feature Selection 후 분류 모델 성능 (Table 9)
| 모델 | Accuracy | AUC | F-measure |
|------|----------|-----|-----------|
| Logistic Regression | 0.6776 | 0.6365 | 0.8076 |
| LDA | 0.6776 | 0.6365 | 0.8076 |
| Random Forest | 0.8239 | **0.8495** | 0.8828 |
| **Bagging** | **0.8275** | 0.8493 | **0.8845** |
| Gradient Boosting | 0.7113 | 0.7187 | 0.8215 |

### 11.2 Feature Selection 후 회귀 모델 성능 (Table 10)
| 모델 | Accuracy | AUC | F-measure |
|------|----------|-----|-----------|
| Linear Regression | 0.6373 | 0.6042 | 0.7614 |
| **Random Forest** | **0.8003** | **0.8412** | **0.8592** |
| Bagging | 0.7958 | 0.8370 | 0.8500 |
| Gradient Boosting | 0.6854 | 0.6999 | 0.7694 |
| Ridge Regression | 0.6080 | 0.5595 | 0.7212 |
| Lasso Regression | 0.6178 | 0.5800 | 0.7465 |

---

## 12. 성공 기준

- [ ] 11개 ML 모델 모두 구현 및 학습 완료 (분류 5개 + 회귀 6개)
- [ ] 논문 Table 9, 10과 유사한 성능 재현 (±10% 오차 허용)
- [ ] Random Forest/Bagging이 최고 성능 확인
- [ ] 결과 리포트 (`campisi_2024_results.md`) 자동 생성
- [ ] Diebold-Mariano 검정으로 통계적 유의성 확인

---

## 13. 참고 자료

- [Yahoo Finance VIX](https://finance.yahoo.com/quote/%5EVIX/)
- [Yahoo Finance SKEW](https://finance.yahoo.com/quote/%5ESKEW/)
- [CBOE Put/Call Ratio Archive](https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv)
- [Kaggle: Downloading VIX Data using Yfinance](https://www.kaggle.com/code/guillemservera/downloading-vix-data-using-yfinance)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Walk Forward Validation Guide](https://medium.com/eatpredlove/time-series-cross-validation-a-walk-forward-approach-in-python-8534dd1db51a)
