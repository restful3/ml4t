# Stocks Return Prediction V2

Kaggle 대회: [Stocks Return Prediction V2](https://www.kaggle.com/competitions/stocks-return-prediction-v-2)

[![Status](https://img.shields.io/badge/Status-Active-green)]()
[![Deadline](https://img.shields.io/badge/Deadline-2026--07--15-blue)]()
[![Teams](https://img.shields.io/badge/Teams-44-orange)]()

## 프로젝트 요약

- **문제 유형**: 퀀트 트레이딩 - 시계열 순위 예측 (Ranking Prediction)
- **목표**: 중국 A주 수익률의 **상대적 순위** 예측
- **평가 지표**: Rank IC (Spearman Correlation) 추정
- **데이터**: 470만 개의 학습 샘플, 488만 개의 테스트 샘플, 익명화된 알파 팩터
- **베이스라인 성능**: RMSE 0.8460 (Rank IC는 미측정)
- **현재 모델**: LightGBM (MSE objective)
- **개선 방향**: Rank IC 최적화, Learning to Rank, Purged CV

## 대회 개요

중국 A주(A-Shares) 주식의 미래 수익률을 예측하는 **퀀트 트레이딩 시계열 회귀** 문제입니다. 주어진 익명화된 팩터(Factors) 데이터를 사용하여 주식 수익률을 예측합니다.

### 주요 정보

- **대회 유형**: Community Competition (교육/퀀트 연구용)
- **주최자**: KaiQQQ
- **대상 시장**: 중국 A주(A-Shares)
- **보상**: Kudos
- **마감일**: 2026년 7월 15일 16:00:00 UTC
- **평가 지표**: **Rank IC (Information Coefficient)** (추정)
- **참가 팀**: 44팀
- **상태**: 참가 중

### 평가 지표: Rank IC

이 대회는 일반적인 RMSE/MAE와 다르게 **Rank IC (Spearman Correlation)**로 평가될 가능성이 높습니다:

```
Rank IC(t) = Spearman_Correlation(predicted_return(t), actual_return(t))
Final Score = Mean(Rank IC across all time periods)
```

**중요한 차이점:**
- 절대적인 수익률 값을 정확히 맞추는 것이 아님
- **어떤 주식이 다른 주식보다 더 오를 것인가(상대적 순위)**를 맞추는 것이 핵심
- 예측값의 순위와 실제 수익률의 순위 간 상관관계를 측정

## 데이터셋 설명

### 데이터 크기

- **Train Data**: 4,670,279 rows × 10 columns (329 MB)
- **Test Data**: 4,879,631 rows × 9 columns (321 MB)
- **Sample Submission**: 4,879,631 rows × 4 columns (186 MB)

### 시간 범위

- **Train**: date 0 ~ 1701 (1702 time periods)
- **Test**: date 1702 ~ 2803 (1102 time periods)
- 시계열 데이터이며 train과 test 기간이 명확히 구분됨

### 주식 정보

- **Train 고유 주식 수**: 3,761개
- **Test 고유 주식 수**: 5,214개
- **Train과 Test 공통 주식**: 3,688개
- **Train에만 있는 주식**: 73개
- **Test에만 있는 주식**: 1,526개 (새로운 주식 포함)

### 피처 설명 (익명화된 Alpha Factors)

모든 피처는 퀀트 투자에서 사용하는 **익명화된 알파 팩터(Alpha Factors)**입니다. 실제 의미는 공개되지 않았으나, 통계적 특성으로 추정 가능합니다.

#### 식별자
- **code** (object): 주식 코드 (예: s_0, s_4394, s_4451)
- **date** (int64): 시간 인덱스 (0부터 시작)

#### 수치형 팩터 (Numeric Factors)
- **f_0** (float64): 평균 1.020, 범위 [0.834, 10.859] - 가격 관련 비율 추정
- **f_1** (float64): 평균 0.980, 범위 [0.711, 8.227] - 가격 관련 비율 추정
- **f_2** (float64): 평균 1.001, 범위 [0.711, 10.859] - 가격 관련 비율 추정
- **f_4** (float64): 평균 175M, 범위 [7K, 67.9B] - **거래량(Volume) 관련**
- **f_5** (float64): 평균 0.999, 범위 [0.764, 9.000] - 기술적 지표 추정
- **f_6** (float64): 평균 1.000, 범위 [0.778, 9.926] - 기술적 지표 추정

#### 범주형 팩터
- **f_3** (object/int64): 30개 고유 값 - **산업 섹터 또는 업종 분류**
  - Train에서는 object 타입 ('11', '22', '6', ...)
  - Test에서는 int64 타입 - 전처리 시 타입 통일 필요

#### 타겟 변수 (Train만)
- **y** (float64): **주식 수익률 (Return)**
  - 평균: -0.014 (약 -1.4%)
  - 표준편차: 0.858
  - 범위: [-9.481, 15.424]
  - **Skewness**: 1.647 (오른쪽 꼬리가 긴 분포, 큰 수익 발생)
  - **Kurtosis**: 11.558 (극단값이 많음, Fat-tail 분포)

### 타겟 변수 특성

타겟 변수 y는 다음과 같은 특징을 보입니다:

- 중앙값: -0.093 (음의 수익률로 치우침)
- 25% 분위수: -0.452
- 75% 분위수: 0.300
- 극단값 존재: 최소 -9.48, 최대 +15.42
- 우편향된 분포 (positive skew)
- 높은 첨도 (extreme values가 많음)

### 피처-타겟 상관관계

```
f_5:  0.042  (가장 높은 양의 상관관계)
f_1:  0.028
f_6:  0.013
f_0: -0.000  (거의 무상관)
f_2: -0.003
f_4: -0.031  (가장 높은 음의 상관관계)
```

### 시계열 특성

- 날짜당 평균 레코드 수: 2,744개
- 날짜당 최소 레코드 수: 1,304개
- 날짜당 최대 레코드 수: 3,592개
- 각 주식당 평균 1,241개 시계열 포인트 (train 기준)

### 데이터 품질

- **결측값**: 없음 (모든 컬럼 0개)
- **중복 행**: 없음
- **무한값**: 없음
- **타입 불일치**: f_3가 train(object)과 test(int64)에서 다름

## 평가 방법

Sample submission 파일 구조:
- **id**: 예측 인덱스
- **code**: 주식 코드
- **date**: 예측 날짜
- **y_pred**: 예측 수익률

제출 파일은 4,879,631개의 예측값을 포함해야 합니다.

## 퀀트 트레이딩 공략 전략

### 1. 핵심 원칙: Rank IC 최적화

일반적인 회귀 문제와 다르게, **상대적 순위(Relative Ranking)**를 맞추는 것이 핵심입니다.

**전략 차이점:**
```python
# ❌ 잘못된 접근: MSE/MAE 최소화만 집중
# → 절대값은 맞지만 순위는 틀릴 수 있음

# ✅ 올바른 접근: Rank IC (Spearman Correlation) 최적화
# → 어떤 주식이 더 오를지 순위를 정확하게 예측
```

### 2. 손실 함수 최적화

**추천 접근법:**

1. **Pearson/Spearman Correlation Objective**
   ```python
   # LightGBM 커스텀 objective
   def correlation_objective(preds, train_data):
       labels = train_data.get_label()
       corr = np.corrcoef(preds, labels)[0, 1]
       return -corr, None  # Maximize correlation
   ```

2. **Learning to Rank (LambdaRank/RankNet)**
   - XGBoost: `objective='rank:pairwise'`
   - LightGBM: `objective='lambdarank'`

3. **Hybrid Approach**
   - MSE/MAE로 먼저 학습 → Correlation objective로 fine-tuning

### 3. 교차 검증 전략 (매우 중요!)

**⚠️ Look-ahead Bias 방지가 필수**

```python
# ❌ 잘못된 방법: Random K-Fold
from sklearn.model_selection import KFold
# → 미래 정보가 학습에 포함되어 과적합 발생!

# ✅ 올바른 방법 1: TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# ✅ 올바른 방법 2: Purged Group Time Series Split
# 학습/검증 사이에 gap을 두어 정보 누수 방지
def purged_group_time_series_split(df, n_splits=5, gap=5):
    """
    gap: 학습 데이터의 마지막 날짜와 검증 데이터의 첫 날짜 사이 간격
    """
    dates = df['date'].unique()
    fold_size = len(dates) // (n_splits + 1)

    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        val_start = train_end + gap
        val_end = val_start + fold_size

        train_idx = df[df['date'] < dates[train_end]].index
        val_idx = df[(df['date'] >= dates[val_start]) &
                     (df['date'] < dates[val_end])].index

        yield train_idx, val_idx
```

### 4. 피처 엔지니어링 전략

**4.1 전처리: 이상치 및 스케일링**

```python
# Winsorization (극단값 클리핑)
from scipy.stats.mstats import winsorize
df['f_0_winsor'] = winsorize(df['f_0'], limits=[0.01, 0.01])

# Rank Transformation (순위 기반 정규화)
df['f_0_rank'] = df.groupby('date')['f_0'].rank(pct=True)

# Z-score Normalization (시점별)
df['f_0_zscore'] = df.groupby('date')['f_0'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**4.2 Feature Interaction (팩터 조합)**

```python
# 팩터 간 상호작용
df['f_0_f_1_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
df['f_0_f_1_product'] = df['f_0'] * df['f_1']
df['f_0_f_5_diff'] = df['f_0'] - df['f_5']

# 거래량 관련
df['volume_rank'] = df.groupby('date')['f_4'].rank(pct=True)
df['volume_zscore'] = df.groupby('date')['f_4'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**4.3 시계열 피처 (주의: Test 추론 가능한 것만)**

```python
# 과거 수익률 (학습 시에만 사용 가능)
for lag in [1, 3, 5]:
    df[f'return_lag_{lag}'] = df.groupby('code')['y'].shift(lag)

# 팩터의 변화율 (Test에서도 사용 가능)
df['f_0_change'] = df.groupby('code')['f_0'].diff()
df['f_0_pct_change'] = df.groupby('code')['f_0'].pct_change()

# 이동평균
for window in [5, 10, 20]:
    df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].rolling(window).mean().reset_index(0, drop=True)
```

**4.4 섹터/크로스 섹셔널 피처**

```python
# 섹터별 평균 대비 차이
df['sector_mean_f_0'] = df.groupby(['date', 'f_3'])['f_0'].transform('mean')
df['f_0_vs_sector'] = df['f_0'] - df['sector_mean_f_0']

# 시장 전체 대비
df['market_mean_f_0'] = df.groupby('date')['f_0'].transform('mean')
df['f_0_vs_market'] = df['f_0'] - df['market_mean_f_0']
```

### 5. 모델 선택 및 앙상블

**추천 모델 조합:**

1. **GBDT Models** (주력)
   - LightGBM (빠르고 안정적)
   - XGBoost (성능 우수)
   - CatBoost (범주형 변수 처리 우수)

2. **Linear Models** (보조)
   - Ridge Regression (정규화)
   - ElasticNet

3. **Ensemble Strategy**
   ```python
   # 다양한 시드/하이퍼파라미터 조합
   predictions = []
   for seed in [42, 123, 456, 789, 2024]:
       model = LGBMRegressor(random_state=seed, ...)
       model.fit(X_train, y_train)
       predictions.append(model.predict(X_test))

   # 평균 앙상블
   final_pred = np.mean(predictions, axis=0)
   ```

### 6. 평가 메트릭 구현

```python
from scipy.stats import spearmanr

def calculate_rank_ic(y_true, y_pred, dates):
    """
    날짜별 Rank IC 계산
    """
    ics = []
    for date in dates.unique():
        mask = dates == date
        if mask.sum() > 1:  # 최소 2개 이상의 샘플 필요
            ic = spearmanr(y_true[mask], y_pred[mask])[0]
            ics.append(ic)

    return np.mean(ics)

# 교차 검증 시 사용
rank_ic = calculate_rank_ic(y_val, y_pred, val_dates)
print(f"Validation Rank IC: {rank_ic:.4f}")
```

## 파일 구조

```
stocks-return-prediction-v-2/
├── README.md                    # 이 파일
├── .gitignore                   # Git 무시 파일 목록
├── data/                        # 데이터 폴더 (git에서 제외)
│   ├── train_data.pkl          # 학습 데이터 (329 MB)
│   ├── test_data.pkl           # 테스트 데이터 (321 MB)
│   └── sample_submission.csv   # 제출 형식 샘플 (186 MB)
├── submissions/                 # 제출 파일 폴더 (git에서 제외)
│   └── submission_baseline.csv # 베이스라인 제출 파일 (189 MB)
├── eda.py                      # 통합 EDA 스크립트 (탐색 + 분석)
├── baseline.py                 # 베이스라인 모델 (LightGBM)
├── validate_submission.py      # 제출 파일 검증 스크립트
└── .venv/                      # Python 가상환경 (git에서 제외)
```

## 시작하기

### 데이터 다운로드

먼저 Kaggle API를 사용하여 대회 데이터를 다운로드하세요:

```bash
# Kaggle API 설정 (처음 한 번만)
# ~/.kaggle/kaggle.json 파일에 API 키 설정 필요

# 데이터 다운로드
cd source/kaggle/stocks-return-prediction-v-2
kaggle competitions download -c stocks-return-prediction-v-2

# 압축 해제
unzip stocks-return-prediction-v-2.zip -d data/
rm stocks-return-prediction-v-2.zip
```

### 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 필요한 패키지 설치
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost kaggle
```

### 데이터 탐색 및 분석

```bash
# 통합 EDA 실행 (구조 확인 + 통계 분석 + 품질 체크)
python eda.py
```

### 베이스라인 모델 실행

```bash
# 베이스라인 모델 학습 및 예측 (작성 예정)
python baseline.py
```

## 주요 도전 과제

### 퀀트 트레이딩 특화 과제

1. **평가 지표 차이**: Rank IC 최적화 vs 일반 RMSE/MAE
   - 절대값 정확도보다 상대적 순위가 중요
   - 손실 함수와 평가 지표 불일치 문제

2. **Look-ahead Bias 방지**
   - 시계열 데이터에서 미래 정보 누수 주의
   - 교차 검증 전략이 성능에 직접적 영향

3. **약한 신호 (Weak Signal)**
   - 피처-타겟 상관계수가 매우 낮음 (최대 0.042)
   - 노이즈 대비 신호가 약해 과적합 위험 높음

4. **Fat-tail 분포**
   - Skewness 1.647, Kurtosis 11.558
   - 극단값이 많아 일반 회귀 모델로는 예측 어려움

### 일반적인 도전 과제

5. **데이터 크기**: 약 500만 행의 대용량 데이터 처리
6. **신규 주식**: Test에만 있는 1,526개 주식 처리 (Cold Start 문제)
7. **타입 불일치**: f_3의 train(object)/test(int64) 타입 차이
8. **시장 환경 변화**: 중국 A주 시장의 특수성 (정책, 규제 등)

## 베이스라인 결과

### 모델 성능

베이스라인 모델 (LightGBM)로 다음과 같은 결과를 얻었습니다:

- **Cross-Validation RMSE**: 0.8460 (±0.0067)
- **Cross-Validation MAE**: 0.5632
- **사용된 피처 수**: 16개

### 주요 피처 중요도

1. sector_mean_f_0 (3,662) - 섹터별 평균
2. stock_std_f_0 (2,920) - 주식별 표준편차
3. stock_mean_f_0 (2,901) - 주식별 평균
4. f_4 (2,669) - 거래량 관련
5. f_3 (2,057) - 섹터/산업 분류

### 예측 분포

- **예측 평균**: -0.0111 (실제: -0.0144)
- **예측 표준편차**: 0.1453 (실제: 0.8583)
- **예측 범위**: [-6.09, 8.54] (실제: [-9.48, 15.42])

예측값의 표준편차가 실제값보다 훨씬 작습니다. 이는 모델이 극단값을 충분히 예측하지 못함을 의미합니다.

### 제출 파일

- `submissions/submission_baseline.csv` - 4,879,631개 예측값

### 베이스라인의 한계점

현재 베이스라인은 **RMSE 최적화**에 초점을 맞췄지만, 실제 평가는 **Rank IC**로 이루어질 가능성이 높습니다.

**주요 문제점:**
1. ❌ MSE/MAE 손실 함수 사용 → Rank IC와 목표 불일치
2. ❌ 예측 분산이 실제보다 작음 (0.145 vs 0.858)
3. ❌ Random TimeSeriesSplit 사용 → Purged Split 필요
4. ❌ 피처 엔지니어링 부족 (Rank Transform, Cross-sectional 등)

### 개선 방향 (우선순위)

**1단계: 평가 지표 정렬 (최우선)**
- [ ] Rank IC 계산 함수 구현
- [ ] Spearman Correlation objective 적용
- [ ] Learning to Rank 모델 실험 (LambdaRank)

**2단계: 교차 검증 개선**
- [ ] Purged Group Time Series Split 구현
- [ ] Gap period 설정 (정보 누수 방지)
- [ ] 날짜별 Rank IC 모니터링

**3단계: 피처 엔지니어링**
- [ ] Rank Transformation (날짜별 순위 변환)
- [ ] Winsorization (극단값 처리)
- [ ] Cross-sectional Features (섹터/시장 대비)
- [ ] Feature Interaction (팩터 조합)

**4단계: 모델 최적화**
- [ ] 하이퍼파라미터 튜닝 (Optuna)
- [ ] 다중 시드 앙상블
- [ ] GBDT 모델 조합 (LightGBM + XGBoost + CatBoost)

**5단계: 고급 전략**
- [ ] 주식 그룹별 모델 (섹터별, 거래량별)
- [ ] Stacking/Blending
- [ ] Neural Network 실험

## 다음 단계

- [x] 베이스라인 모델 구현 (RMSE 기준)
- [ ] **Rank IC 기준 모델 재구현** (최우선!)
- [ ] Purged Time Series Split 적용
- [ ] 퀀트 피처 엔지니어링 (Rank, Winsorize 등)
- [ ] Learning to Rank 실험
- [ ] 앙상블 전략 개발

## 참고 자료

### 대회 및 문서
- [Kaggle Competition](https://www.kaggle.com/competitions/stocks-return-prediction-v-2)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

### 퀀트 트레이딩 참고 자료
- [Quantopian Lectures: Alpha Factors](https://www.quantopian.com/lectures)
- [101 Formulaic Alphas (WorldQuant)](https://arxiv.org/abs/1601.00991)
- [Machine Learning for Trading (Georgia Tech)](https://omscs.gatech.edu/cs-7646-machine-learning-trading)

### 논문 및 기법
- **Information Coefficient (IC)**: [Fundamental Law of Active Management](https://www.investopedia.com/terms/f/fundamentallawofactivemanagement.asp)
- **Learning to Rank**: Burges et al., "Learning to Rank using Gradient Descent" (2005)
- **Purged Cross-Validation**: López de Prado, "Advances in Financial Machine Learning" (2018)
- **Alpha Factor Construction**: Kakushadze, "101 Formulaic Alphas" (2016)

### 유용한 라이브러리
```bash
# 퀀트 분석 도구
pip install alphalens-reloaded  # Alpha factor analysis
pip install pyfolio-reloaded    # Portfolio analysis
pip install ta-lib              # Technical indicators
pip install optuna              # Hyperparameter tuning
```

### 추천 Kaggle Notebooks
대회 페이지의 "Code" 섹션에서 다음을 검색하세요:
- "Starter" 또는 "Baseline" - 기본 접근법
- "Rank IC" - 평가 지표 구현
- "Feature Engineering" - 피처 엔지니어링 예제
