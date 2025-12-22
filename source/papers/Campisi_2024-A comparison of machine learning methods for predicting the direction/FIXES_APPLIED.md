# Code Fixes Applied - Campisi et al. (2024) Replication

**생성일**: 2025-12-20
**원본 파일**: `campisi_2024_replication.py`
**수정 파일**: `campisi_2024_replication_fixed.py`

## Executive Summary

Gemini의 코드 리뷰 결과를 바탕으로 **Data Leakage** 문제를 해결하고, 통계적 검정 및 성능 최적화를 추가한 개선 버전을 작성했습니다.

---

## 주요 수정 사항

### 🚨 Critical Issues (반드시 수정)

#### 1. Data Leakage: Standardization 문제 해결

**원본 코드 (WRONG)**:
```python
# Line 982: CV loop 밖에서 전체 데이터 표준화
X_scaled, scaler = standardize_features(X)  # ❌ 미래 데이터 유출

# Inside CV loop
for train_idx, test_idx in cv.split(X_scaled):
    model.fit(X_train, y_train)  # 이미 미래 정보로 스케일링된 데이터 사용
```

**수정 코드 (CORRECT)**:
```python
# Pipeline 사용으로 CV loop 내부에서 자동으로 처리
from sklearn.pipeline import Pipeline

def get_classification_models(use_feature_selection=True):
    models = {}
    for name, base_model in base_models.items():
        steps = [
            ('scaler', StandardScaler()),  # ✅ CV 내부에서 fit
            ('classifier', base_model)
        ]
        models[name] = Pipeline(steps)
    return models

# Train 함수에서
for train_idx, test_idx in cv.split(X):  # X는 raw data
    X_train, X_test = X[train_idx], X[test_idx]
    pipeline.fit(X_train, y_train)  # ✅ train에서만 scaler fit
    pred = pipeline.predict(X_test)  # ✅ test는 transform만
```

**효과**:
- Train set의 평균/분산만 사용하여 스케일링
- Test set 정보 유출 방지
- 정확한 out-of-sample 성능 측정

---

#### 2. Data Leakage: Feature Selection 문제 해결

**원본 코드 (WRONG)**:
```python
# Line 990: CV loop 밖에서 전체 데이터로 feature selection
selected_features, importance = lasso_feature_selection(
    X_scaled, y_continuous, feature_cols
)  # ❌ 2012년 모델이 2021년 정보를 알고 있음

X_selected = X_scaled[:, selected_idx]
```

**수정 코드 (CORRECT)**:
```python
# Pipeline에 SelectFromModel 추가
from sklearn.feature_selection import SelectFromModel

steps = [
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(  # ✅ CV 내부에서 자동 feature selection
        Lasso(alpha=lasso_alpha, random_state=RANDOM_STATE),
        threshold=1e-5
    )),
    ('classifier', base_model)
]
pipeline = Pipeline(steps)

# CV loop에서 자동으로 train data로만 feature selection 수행
pipeline.fit(X_train, y_train)
```

**효과**:
- 각 CV iteration마다 해당 시점의 train data로만 feature 선택
- 미래 정보 활용 방지
- 시간에 따른 feature 중요도 변화 반영

---

#### 3. Diebold-Mariano Test 구현 및 호출

**원본 코드**:
```python
# Line 562: 함수만 정의되고 호출 안 됨
def diebold_mariano_test(y_true, pred1, pred2):
    # ... 구현은 되어 있음
    pass

# main()에서 호출되지 않음 ❌
```

**수정 코드**:
```python
def perform_dm_tests(results: Dict, y_true: np.ndarray) -> pd.DataFrame:
    """Perform pairwise Diebold-Mariano tests between models."""
    model_names = list(results.keys())
    dm_results = []

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                pred1 = results[name1]['predictions']
                pred2 = results[name2]['predictions']
                dm_stat, p_val = diebold_mariano_test(y_true, pred1, pred2)
                dm_results.append({
                    'Model 1': name1,
                    'Model 2': name2,
                    'DM Statistic': dm_stat,
                    'p-value': p_val,
                    'Significant (5%)': p_val < 0.05
                })

    return pd.DataFrame(dm_results)

# main()에서 호출 ✅
dm_tests_clf = perform_dm_tests(clf_results_after, y_true_binary)
dm_tests_reg = perform_dm_tests(reg_results_after, y_true_reg_binary)
```

**효과**:
- 모델 간 성능 차이가 통계적으로 유의한지 검정
- 논문의 핵심("A comparison of ML methods") 증명 가능

---

### ⚡ Performance & Optimization

#### 1. Refit Frequency 옵션 추가

**문제**: 매일 모든 모델을 재학습하면 계산량이 과도함
- 11 models × ~750 iterations = ~8,250 training runs

**수정**:
```python
class WalkForwardCV:
    def __init__(self, train_size=TRAIN_SIZE, gap=GAP,
                 max_iterations=None,
                 refit_frequency=1):  # ✅ 새로 추가
        self.refit_frequency = refit_frequency

    def split(self, X):
        for iteration_count, test_idx in enumerate(...):
            should_refit = (
                last_train_idx is None or
                iteration_count % self.refit_frequency == 0
            )
            yield train_idx, test_idx, should_refit

# 사용 예
cv = WalkForwardCV(refit_frequency=30)  # 30일마다 재학습
```

**효과**:
- 계산 시간 대폭 감소 (refit_frequency=30 시 ~96% 감소)
- 실무적으로 합리적 (매일 재학습은 현실적이지 않음)

---

#### 2. Command Line Arguments 추가

```python
parser.add_argument('--refit-frequency', '-r', type=int, default=1,
                    help='모델 재학습 주기 (일 단위)')
parser.add_argument('--no-feature-selection', action='store_true',
                    help='Feature selection 비활성화')
```

**사용 예**:
```bash
# 빠른 테스트 (10 iterations, 30일마다 재학습)
python campisi_2024_replication_fixed.py -m 10 -r 30

# Feature selection 없이 실행
python campisi_2024_replication_fixed.py --no-feature-selection

# 전체 실행 (시간이 오래 걸림)
python campisi_2024_replication_fixed.py
```

---

## 코드 구조 개선

### Before (Original)
```
Phase 1-2: Data Collection
Phase 3: Preprocessing
Phase 4: Feature Selection (❌ 여기서 leakage 발생)
  - standardize_features(X)  # 전체 데이터
  - lasso_feature_selection(X_scaled, y)  # 전체 데이터
Phase 5-7: Model Training
  - CV loop에서 이미 스케일링/선택된 데이터 사용
```

### After (Fixed)
```
Phase 1-2: Data Collection
Phase 3: Preprocessing
Phase 4: Prepare Data (raw data만 준비)
  - X = data[feature_cols].values  # ✅ raw data
Phase 5-7: Model Training with Pipeline
  - Pipeline이 CV loop 내부에서 자동으로 처리
    1. StandardScaler().fit(X_train)
    2. SelectFromModel().fit(X_train, y_train)
    3. Classifier.fit(X_train_transformed, y_train)
Phase 8: Diebold-Mariano Tests (✅ 새로 추가)
Phase 9-10: Visualization & Report
```

---

## 예상 성능 변화

### 원본 코드 (Data Leakage 있음)
- 높은 Accuracy (예: ~0.82)
- 과도하게 낙관적인 결과
- 실제 배포 시 성능 하락 가능

### 수정 코드 (Data Leakage 없음)
- 상대적으로 낮은 Accuracy (예: ~0.55-0.65)
- **정확한** out-of-sample 성능
- 실제 배포 시 예상 성능과 일치

> **중요**: 성능이 낮아지는 것이 정상입니다. 이것이 실제 시장에서 기대할 수 있는 성능입니다.

---

## 검증 방법

### 1. Pipeline 동작 확인
```python
# Pipeline의 각 단계를 확인
fitted_pipeline = pipeline.fit(X_train, y_train)

# Scaler 파라미터 확인
print("Scaler mean:", fitted_pipeline.named_steps['scaler'].mean_)
print("Scaler std:", fitted_pipeline.named_steps['scaler'].scale_)

# 선택된 features 확인 (feature selection 사용 시)
selector = fitted_pipeline.named_steps['selector']
print("Selected features:", np.where(selector.get_support())[0])
```

### 2. CV Split 확인
```python
cv = WalkForwardCV(train_size=100, gap=10)
for train_idx, test_idx, should_refit in cv.split(X):
    print(f"Train: {train_idx[0]}~{train_idx[-1]}, "
          f"Test: {test_idx[0]}, Refit: {should_refit}")

    # Gap 확인
    assert train_idx[-1] + gap < test_idx[0]
```

### 3. Diebold-Mariano Test 해석
```python
# DM Statistic > 0: Model 1이 Model 2보다 나쁨
# DM Statistic < 0: Model 1이 Model 2보다 좋음
# p-value < 0.05: 통계적으로 유의한 차이

dm_tests = perform_dm_tests(clf_results_after, y_true)
significant = dm_tests[dm_tests['Significant (5%)'] == True]
print(significant)
```

---

## 추가 개선 가능 사항

### 1. 병렬 처리
```python
from joblib import Parallel, delayed

# CV loop를 병렬로 실행
results = Parallel(n_jobs=-1)(
    delayed(train_one_fold)(pipeline, X, y, train_idx, test_idx)
    for train_idx, test_idx, _ in cv.split(X)
)
```

### 2. 모델 캐싱
```python
import joblib

# 학습된 모델 저장
joblib.dump(pipeline, f'models/model_{iteration}.pkl')

# 나중에 로드
pipeline = joblib.load(f'models/model_{iteration}.pkl')
```

### 3. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 500, 1000],
    'classifier__max_depth': [3, 5, 10],
    'selector__estimator__alpha': [0.001, 0.01, 0.1]
}

# Nested CV로 하이퍼파라미터 튜닝
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## 참고 자료

### Data Leakage 관련
- [Sklearn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Common Pitfalls in Time Series Cross-Validation](https://robjhyndman.com/hyndsight/tscv/)

### Walk-Forward Validation
- [Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Campisi et al. (2024) - Section 3.2: Validation Strategy

### Statistical Testing
- Diebold & Mariano (1995): "Comparing Predictive Accuracy"
- Harvey et al. (1997): "Testing the equality of prediction mean squared errors"

---

## 결론

이 수정 버전은:
1. ✅ Data Leakage 문제를 완전히 해결
2. ✅ 통계적 검정을 통해 모델 비교 가능
3. ✅ 성능 최적화로 실용성 향상
4. ✅ 정확한 out-of-sample 성능 측정

**추천**: 실제 논문 재현 및 전략 개발에는 **반드시 수정 버전을 사용**하세요.
