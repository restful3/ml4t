# 구현 계획서: Campisi et al. (2024) 논문 실증

## 프로젝트 정보
- **논문**: A comparison of machine learning methods for predicting the direction of the US stock market on the basis of volatility indices
- **목표**: 논문의 실험 결과 재현 (Table 9, 10)
- **출력 파일**: `campisi_2024_replication.py`, `campisi_2024_results.md`

---

## Phase 1: 환경 설정

### 1.1 가상환경 생성
- [ ] 프로젝트 디렉토리로 이동
  ```bash
  cd "/home/restful3/workspace/ml4t/source/papers/Campisi_2024-A comparison of machine learning methods for predicting the direction"
  ```
- [ ] Python 가상환경 생성
  ```bash
  python3 -m venv .venv
  ```
- [ ] 가상환경 활성화
  ```bash
  source .venv/bin/activate
  ```

### 1.2 requirements.txt 생성
- [ ] `requirements.txt` 파일 생성
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
- [ ] 패키지 설치
  ```bash
  pip install -r requirements.txt
  ```

### 1.3 환경 검증
- [ ] Python 버전 확인 (3.8+)
- [ ] 주요 패키지 import 테스트
  ```python
  import pandas, numpy, sklearn, yfinance, matplotlib, seaborn, statsmodels, scipy
  print("All packages imported successfully!")
  ```

---

## Phase 2: 데이터 수집 모듈 구현

### 2.1 Yahoo Finance 데이터 수집 함수
- [ ] `download_volatility_indices()` 함수 구현
  - [ ] 티커 목록 정의
    ```python
    TICKERS = {
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
    ```
  - [ ] yfinance로 데이터 다운로드 (2011-01-01 ~ 2022-07-31)
  - [ ] Close 가격만 추출
  - [ ] 컬럼명 변경 (티커 → 변수명)

### 2.2 PUTCALL 데이터 수집 (선택사항)
- [ ] `download_putcall_ratio()` 함수 구현
  - [ ] CBOE CSV URL에서 다운로드 시도
    ```python
    url = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv"
    ```
  - [ ] 실패 시 PUTCALL 변수 제외 처리
  - [ ] 날짜 인덱스 맞추기

### 2.3 파생 변수 계산
- [ ] `calculate_derived_features()` 함수 구현
  - [ ] **RVOL (실현 변동성)** 계산
    ```python
    # 30일 롤링 표준편차 × √252 (연율화)
    daily_returns = np.log(sp500 / sp500.shift(1))
    rvol = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
    ```
  - [ ] **Returns30 (30일 후 로그 수익률)** 계산
    ```python
    returns30 = np.log(sp500.shift(-30) / sp500)
    ```
  - [ ] **Target (이진 분류용)** 계산
    ```python
    target_binary = (returns30 > 0).astype(int)
    ```

### 2.4 데이터 병합 및 저장
- [ ] `merge_all_data()` 함수 구현
  - [ ] 모든 데이터프레임 날짜 기준 병합
  - [ ] 최종 데이터프레임 구조 확인
  - [ ] CSV로 저장 (캐싱용): `data/raw_data.csv`

### 2.5 데이터 수집 검증
- [ ] 전체 관측치 수 확인 (목표: ~3,040개)
- [ ] 각 변수별 결측치 비율 확인
- [ ] 날짜 범위 확인 (2011-01 ~ 2022-07)

---

## Phase 3: 데이터 전처리 모듈 구현

### 3.1 결측치 처리
- [ ] `handle_missing_values()` 함수 구현
  - [ ] Forward fill 적용
    ```python
    df = df.fillna(method='ffill')
    ```
  - [ ] Backward fill 적용
    ```python
    df = df.fillna(method='bfill')
    ```
  - [ ] 여전히 결측인 행 제거
    ```python
    df = df.dropna()
    ```
  - [ ] 처리 전/후 행 수 로깅

### 3.2 기술 통계량 계산
- [ ] `calculate_summary_statistics()` 함수 구현
  - [ ] 평균 (Mean)
  - [ ] 표준편차 (Std. Dev.)
  - [ ] 왜도 (Skewness)
  - [ ] 첨도 (Kurtosis)
  - [ ] 자기상관계수 (ρ₁, ρ₂, ρ₃)
  - [ ] ADF 검정 통계량
  - [ ] Jarque-Bera 검정 통계량
- [ ] 논문 Table 1과 비교 가능한 형태로 출력

### 3.3 상관관계 분석
- [ ] `calculate_correlation_matrix()` 함수 구현
  - [ ] 피어슨 상관계수 계산
  - [ ] 논문 Table 2와 비교
  - [ ] 히트맵 시각화 저장

### 3.4 표준화
- [ ] `standardize_features()` 함수 구현
  - [ ] StandardScaler 사용
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
  - [ ] 표준화 후 평균=0, 표준편차=1 확인
  - [ ] 타겟 변수(Returns30, Target)는 표준화 제외

---

## Phase 4: Feature Selection 모듈 구현

### 4.1 VIF (Variance Inflation Factor) 계산
- [ ] `calculate_vif()` 함수 구현
  ```python
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  def calculate_vif(X):
      vif_data = pd.DataFrame()
      vif_data["Variable"] = X.columns
      vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
      return vif_data
  ```
- [ ] VIF > 10인 변수 식별
- [ ] 논문 Table 5와 비교

### 4.2 Lasso 기반 변수 선택
- [ ] `lasso_feature_selection()` 함수 구현
  - [ ] LassoCV로 최적 alpha 탐색
    ```python
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5)
    lasso.fit(X_train, y_train)
    ```
  - [ ] 계수가 0인 변수 식별
  - [ ] 변수 중요도 플롯 생성 (논문 Figure 2 재현)

### 4.3 Feature Selection 적용
- [ ] 제거 대상 변수 확인
  - [ ] 예상: VIX9D, VIX3M, VIX6M (다중공선성)
- [ ] 두 가지 변수 세트 준비
  - [ ] `features_all`: 전체 11개 변수
  - [ ] `features_selected`: Lasso 선택 후 8개 변수

---

## Phase 5: Walk-Forward Validation 구현

### 5.1 커스텀 CV 클래스 구현
- [ ] `WalkForwardCV` 클래스 구현
  ```python
  class WalkForwardCV:
      def __init__(self, train_size=2128, gap=30):
          self.train_size = train_size
          self.gap = gap

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

      def get_n_splits(self, X):
          return len(X) - self.train_size - self.gap
  ```

### 5.2 검증 로직 확인
- [ ] 학습 세트 크기: 2,128개 (70%)
- [ ] 테스트 예측 수: ~883개
- [ ] Gap: 30일 (forward-looking 방지)
- [ ] Rolling window 방식 확인 (expanding 아님)

### 5.3 검증 테스트
- [ ] 더미 데이터로 split 동작 확인
- [ ] 학습/테스트 인덱스 겹침 없음 확인
- [ ] 시간 순서 유지 확인

---

## Phase 6: 모델 구현

### 6.1 분류 모델 정의
- [ ] `get_classification_models()` 함수 구현
  ```python
  def get_classification_models():
      return {
          'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
          'LDA': LinearDiscriminantAnalysis(),
          'Random Forest (Clf)': RandomForestClassifier(
              n_estimators=500,
              max_features='sqrt',
              random_state=42,
              n_jobs=-1
          ),
          'Bagging (Clf)': BaggingClassifier(
              n_estimators=500,
              max_features=1.0,
              random_state=42,
              n_jobs=-1
          ),
          'Gradient Boosting (Clf)': GradientBoostingClassifier(
              n_estimators=100,
              learning_rate=0.1,
              max_depth=3,
              random_state=42
          )
      }
  ```

### 6.2 회귀 모델 정의
- [ ] `get_regression_models()` 함수 구현
  ```python
  def get_regression_models():
      return {
          'Linear Regression': LinearRegression(),
          'Ridge Regression': Ridge(alpha=1.0),
          'Lasso Regression': Lasso(alpha=0.001),
          'Random Forest (Reg)': RandomForestRegressor(
              n_estimators=500,
              max_features='sqrt',
              random_state=42,
              n_jobs=-1
          ),
          'Bagging (Reg)': BaggingRegressor(
              n_estimators=500,
              max_features=1.0,
              random_state=42,
              n_jobs=-1
          ),
          'Gradient Boosting (Reg)': GradientBoostingRegressor(
              n_estimators=100,
              learning_rate=0.1,
              max_depth=3,
              random_state=42
          )
      }
  ```

### 6.3 하이퍼파라미터 튜닝 (선택사항)
- [ ] Ridge/Lasso alpha 탐색
  ```python
  alphas = np.logspace(-4, 2, 100)
  ```
- [ ] Gradient Boosting 파라미터 그리드
  ```python
  param_grid = {
      'n_estimators': [50, 100, 150],
      'learning_rate': [0.01, 0.05, 0.1],
      'max_depth': [1, 2, 3]
  }
  ```

---

## Phase 7: 모델 학습 및 예측

### 7.1 학습 파이프라인 구현
- [ ] `train_and_predict()` 함수 구현
  ```python
  def train_and_predict(model, X, y, cv):
      predictions = []
      probabilities = []  # 분류 모델용
      actuals = []

      for train_idx, test_idx in tqdm(cv.split(X)):
          X_train, y_train = X[train_idx], y[train_idx]
          X_test, y_test = X[test_idx], y[test_idx]

          model_clone = clone(model)
          model_clone.fit(X_train, y_train)

          pred = model_clone.predict(X_test)
          predictions.append(pred[0])
          actuals.append(y_test[0])

          # 분류 모델의 경우 확률도 저장
          if hasattr(model_clone, 'predict_proba'):
              prob = model_clone.predict_proba(X_test)[:, 1]
              probabilities.append(prob[0])

      return np.array(predictions), np.array(probabilities), np.array(actuals)
  ```

### 7.2 분류 모델 학습 (Feature Selection 전)
- [ ] 5개 분류 모델 학습
  - [ ] Logistic Regression
  - [ ] LDA
  - [ ] Random Forest
  - [ ] Bagging
  - [ ] Gradient Boosting
- [ ] 예측 결과 저장

### 7.3 분류 모델 학습 (Feature Selection 후)
- [ ] 5개 분류 모델 학습 (선택된 변수만 사용)
- [ ] 예측 결과 저장

### 7.4 회귀 모델 학습 (Feature Selection 전)
- [ ] 6개 회귀 모델 학습
  - [ ] Linear Regression
  - [ ] Ridge Regression
  - [ ] Lasso Regression
  - [ ] Random Forest
  - [ ] Bagging
  - [ ] Gradient Boosting
- [ ] 연속 예측값 → 이진 변환 (양수=1, 음수=0)

### 7.5 회귀 모델 학습 (Feature Selection 후)
- [ ] 6개 회귀 모델 학습 (선택된 변수만 사용)
- [ ] 예측 결과 저장

---

## Phase 8: 평가 지표 계산

### 8.1 성능 지표 함수 구현
- [ ] `calculate_metrics()` 함수 구현
  ```python
  def calculate_metrics(y_true, y_pred, y_prob=None):
      metrics = {
          'Accuracy': accuracy_score(y_true, y_pred),
          'F-measure': f1_score(y_true, y_pred)
      }
      if y_prob is not None:
          metrics['AUC'] = roc_auc_score(y_true, y_prob)
      return metrics
  ```

### 8.2 분류 모델 평가
- [ ] Feature Selection 전 성능 계산 (Table 7 재현)
  | 모델 | ACC | AUC | F |
  |------|-----|-----|---|
  | Logistic Regression | ? | ? | ? |
  | LDA | ? | ? | ? |
  | Random Forest | ? | ? | ? |
  | Bagging | ? | ? | ? |
  | Gradient Boosting | ? | ? | ? |

- [ ] Feature Selection 후 성능 계산 (Table 9 재현)

### 8.3 회귀 모델 평가
- [ ] Feature Selection 전 성능 계산 (Table 8 재현)
- [ ] Feature Selection 후 성능 계산 (Table 10 재현)

### 8.4 Diebold-Mariano 검정
- [ ] `diebold_mariano_test()` 함수 구현
  ```python
  def diebold_mariano_test(y_true, pred1, pred2):
      e1 = (y_true - pred1) ** 2  # MSE loss
      e2 = (y_true - pred2) ** 2
      d = e1 - e2

      mean_d = np.mean(d)
      var_d = np.var(d, ddof=1)

      dm_stat = mean_d / np.sqrt(var_d / len(d))
      p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

      return dm_stat, p_value
  ```
- [ ] 모델 쌍별 DM 검정 수행 (Table 11, 12 재현)

---

## Phase 9: 시각화

### 9.1 변수 중요도 플롯
- [ ] Lasso 계수 기반 중요도 바 차트 (Figure 2 재현)
- [ ] Random Forest feature importance 플롯

### 9.2 ROC 곡선
- [ ] 각 분류 모델별 ROC 곡선 그리기
- [ ] 모든 모델 비교 ROC 곡선

### 9.3 시계열 플롯
- [ ] Returns30 시계열 그래프 (Figure 1 재현)
- [ ] 예측 vs 실제 비교 그래프

### 9.4 상관관계 히트맵
- [ ] 변수 간 상관관계 히트맵 (Table 2 시각화)

### 9.5 저장
- [ ] 모든 그래프 `figures/` 디렉토리에 PNG로 저장

---

## Phase 10: 결과 리포트 생성

### 10.1 리포트 템플릿 구현
- [ ] `generate_report()` 함수 구현
- [ ] Markdown 형식으로 자동 생성

### 10.2 리포트 섹션 작성
- [ ] **1. 데이터 요약**
  - [ ] 수집 기간
  - [ ] 총 관측치 수
  - [ ] 결측치 처리 내역

- [ ] **2. 기술 통계량**
  - [ ] 논문 Table 1과 비교 테이블

- [ ] **3. 상관관계 분석**
  - [ ] 논문 Table 2와 비교

- [ ] **4. Feature Selection 결과**
  - [ ] VIF 값 테이블
  - [ ] Lasso 중요도
  - [ ] 제거된 변수 목록

- [ ] **5. 모델 성능 비교**
  - [ ] Feature Selection 전 (Table 7, 8)
  - [ ] Feature Selection 후 (Table 9, 10)

- [ ] **6. Diebold-Mariano 검정**
  - [ ] Table 11, 12 재현

- [ ] **7. 논문 결과와 비교**
  - [ ] 일치/불일치 분석
  - [ ] 차이 원인 분석

- [ ] **8. 결론**
  - [ ] 주요 발견사항
  - [ ] 한계점

### 10.3 리포트 저장
- [ ] `campisi_2024_results.md` 파일로 저장

---

## Phase 11: 메인 스크립트 통합

### 11.1 스크립트 구조
```python
# campisi_2024_replication.py

"""
Campisi et al. (2024) 논문 실증 재현
"""

# Imports
import pandas as pd
import numpy as np
# ...

# Constants
TICKERS = {...}
START_DATE = '2011-01-01'
END_DATE = '2022-07-31'
TRAIN_SIZE = 2128
GAP = 30

# Data Collection
def download_volatility_indices(): ...
def download_putcall_ratio(): ...
def calculate_derived_features(): ...

# Preprocessing
def handle_missing_values(): ...
def standardize_features(): ...

# Feature Selection
def calculate_vif(): ...
def lasso_feature_selection(): ...

# Models
def get_classification_models(): ...
def get_regression_models(): ...

# Validation
class WalkForwardCV: ...
def train_and_predict(): ...

# Evaluation
def calculate_metrics(): ...
def diebold_mariano_test(): ...

# Visualization
def plot_feature_importance(): ...
def plot_roc_curves(): ...

# Report Generation
def generate_report(): ...

# Main
def main():
    print("=" * 60)
    print("Campisi et al. (2024) 논문 실증 재현")
    print("=" * 60)

    # 1. 데이터 수집
    # 2. 전처리
    # 3. Feature Selection
    # 4. 모델 학습 (Feature Selection 전)
    # 5. 모델 학습 (Feature Selection 후)
    # 6. 평가
    # 7. 시각화
    # 8. 리포트 생성

if __name__ == "__main__":
    main()
```

### 11.2 체크포인트
- [ ] 각 Phase 완료 시 중간 결과 저장
- [ ] 에러 발생 시 복구 가능하도록 구현

---

## Phase 12: 테스트 및 검증

### 12.1 단위 테스트
- [ ] 데이터 수집 함수 테스트
- [ ] 전처리 함수 테스트
- [ ] CV split 테스트
- [ ] 평가 지표 계산 테스트

### 12.2 통합 테스트
- [ ] 전체 파이프라인 실행
- [ ] 메모리 사용량 확인
- [ ] 실행 시간 측정 (예상: 30분~1시간)

### 12.3 결과 검증
- [ ] 논문 Table 9와 비교 (분류 모델)
  - [ ] Accuracy 오차 ±10% 이내
  - [ ] AUC 오차 ±10% 이내
  - [ ] F-measure 오차 ±10% 이내

- [ ] 논문 Table 10과 비교 (회귀 모델)
  - [ ] Accuracy 오차 ±10% 이내
  - [ ] AUC 오차 ±10% 이내
  - [ ] F-measure 오차 ±10% 이내

- [ ] 주요 결론 확인
  - [ ] Random Forest/Bagging이 최고 성능
  - [ ] 분류 모델 > 회귀 모델
  - [ ] Feature Selection 후 성능 향상

---

## Phase 13: 문서화 및 정리

### 13.1 코드 문서화
- [ ] 모든 함수에 docstring 추가
- [ ] 타입 힌트 추가
- [ ] 인라인 주석 추가

### 13.2 README 작성 (선택사항)
- [ ] 프로젝트 설명
- [ ] 실행 방법
- [ ] 결과 요약

### 13.3 파일 정리
- [ ] 불필요한 파일 삭제
- [ ] 디렉토리 구조 정리
  ```
  Campisi_2024-.../
  ├── .venv/
  ├── data/
  │   └── raw_data.csv
  ├── figures/
  │   ├── feature_importance.png
  │   ├── roc_curves.png
  │   └── correlation_heatmap.png
  ├── requirements.txt
  ├── prd.md
  ├── plan.md
  ├── campisi_2024_replication.py
  └── campisi_2024_results.md
  ```

---

## 완료 체크리스트

### 필수 산출물
- [ ] `requirements.txt` 생성
- [ ] `.venv/` 가상환경 생성
- [ ] `campisi_2024_replication.py` 구현
- [ ] `campisi_2024_results.md` 생성

### 성공 기준
- [ ] 11개 ML 모델 모두 구현 및 학습 완료
- [ ] 논문 Table 9, 10과 유사한 성능 재현 (±10%)
- [ ] Random Forest/Bagging이 최고 성능 확인
- [ ] 결과 리포트 자동 생성

### 보너스
- [ ] Diebold-Mariano 검정 구현
- [ ] 시각화 그래프 생성
- [ ] 하이퍼파라미터 튜닝

---

## 예상 소요 시간

| Phase | 예상 시간 |
|-------|----------|
| Phase 1: 환경 설정 | 10분 |
| Phase 2: 데이터 수집 | 20분 |
| Phase 3: 전처리 | 15분 |
| Phase 4: Feature Selection | 15분 |
| Phase 5: Walk-Forward CV | 15분 |
| Phase 6: 모델 정의 | 20분 |
| Phase 7: 학습 및 예측 | 30분 (실행 시간 별도) |
| Phase 8: 평가 | 15분 |
| Phase 9: 시각화 | 20분 |
| Phase 10: 리포트 | 15분 |
| Phase 11: 통합 | 15분 |
| Phase 12: 테스트 | 20분 |
| Phase 13: 문서화 | 10분 |
| **총계** | **약 3-4시간** |

---

## 참고 사항

1. **데이터 소스 차이**: 논문은 Bloomberg, 우리는 Yahoo Finance 사용
2. **PUTCALL 변수**: 수집 실패 시 제외하고 진행 (영향 미미)
3. **병렬 처리**: `n_jobs=-1`로 CPU 코어 최대 활용
4. **재현성**: `random_state=42` 고정
