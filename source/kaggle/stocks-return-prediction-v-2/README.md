# Stocks Return Prediction V2

Kaggle 대회: [Stocks Return Prediction V2](https://www.kaggle.com/competitions/stocks-return-prediction-v-2)

[![Status](https://img.shields.io/badge/Status-Active-green)]()
[![Score](https://img.shields.io/badge/Rank_IC-0.08957-red)]()

---

## Executive Summary

본 문서는 Kaggle의 "Stocks Return Prediction V2" 대회를 위해 수행된 머신러닝 프로젝트의 핵심 전략, 개발 과정, 그리고 주요 성과를 종합적으로 브리핑합니다. 프로젝트의 목표는 중국 A주 시장의 개별 주식 수익률에 대한 절대값이 아닌 상대적 순위를 예측하는 것입니다.

핵심적인 도전 과제는 일반적인 회귀 모델의 손실 함수(RMSE 등)와 대회의 공식 평가 지표인 Rank IC(순위 정보 계수, Spearman Correlation) 간의 불일치였습니다. 초기 모델은 이 문제로 인해 0.05707이라는 저조한 성능을 기록했습니다.

프로젝트의 결정적인 돌파구는 **"타겟 순위 변환(Target Rank Transformation)"** 전략의 도입이었습니다. 이 접근법은 실제 수익률 값을 예측하는 대신, 각 시점(date)별로 수익률의 백분위 순위(0~1)를 예측하도록 모델의 목표를 재설정했습니다. 이 패러다임 전환을 통해 손실 함수와 평가 지표를 정렬시킴으로써, 공개 리더보드 점수는 **0.08720으로 폭발적으로 상승했습니다(+52.8%)**.

이후 상호작용 피처(Interaction Features) 추가 및 10개 시드(Seed)를 활용한 배깅(Bagging) 앙상블 기법을 통해 모델을 더욱 안정화하고 미세 조정하여, 최종적으로 **0.08957의 최고 점수를 달성**했습니다. 본 프로젝트는 퀀트 트레이딩의 순위 예측 문제에서 평가 지표에 최적화된 전략 설계가 얼마나 중요한지를 명확하게 보여주는 성공 사례입니다.

![Project Infographic](infograph.png)

---

## 1. 프로젝트 개요

### 1.1. 대회 목표 및 문제 정의

본 프로젝트는 중국 A주(A-Shares) 시장의 미래 주식 수익률을 예측하는 퀀트 트레이딩 시계열 순위 예측 문제입니다. 익명화된 알파 팩터(Alpha Factors)를 기반으로, 어떤 주식이 다른 주식보다 더 높은 수익률을 기록할 것인지, 즉 수익률의 상대적 순위를 예측하는 것이 핵심 목표입니다.

* **문제 유형**: 퀀트 트레이딩 - 시계열 순위 예측 (Ranking Prediction)
* **대상 시장**: 중국 A주 (A-Shares)
* **주최자**: KaiQQQ (Community Competition)
* **참가 팀**: 44팀

### 1.2. 평가 지표: Rank IC

대회의 주 평가 지표는 **Rank IC(Information Coefficient)**로, 이는 예측된 수익률의 순위와 실제 수익률의 순위 간의 **스피어만 상관계수(Spearman Correlation)**로 추정됩니다.

* **핵심**: 절대적인 수익률 값(예: 5% 상승)의 정확도보다, 주식 간 수익률 순서(예: A주식 > B주식)를 맞추는 것이 중요합니다.
* **의미**: 일반적인 회귀 평가지표인 RMSE나 MAE와 근본적으로 다른 접근이 필요함을 시사합니다.

### 1.3. 데이터셋 요약

데이터는 익명화된 7개의 알파 팩터(`f_0`~`f_6`)와 타겟 변수인 수익률(`y`)로 구성되어 있습니다.

| 항목 | 학습 데이터 (Train) | 테스트 데이터 (Test) |
|---|---|---|
| **샘플 수** | 4,670,279 개 | 4,879,631 개 |
| **기간 (date)** | 0 ~ 1701 (1702 기간) | 1702 ~ 2803 (1102 기간) |
| **고유 주식 수** | 3,761 개 | 5,214 개 |
| **공통 주식 수** | \multicolumn{2}{c}{3,688 개} |
| **Test에만 존재하는 주식** | - | 1,526 개 (Cold Start 문제) |
| **피처 수** | 9개 (code, date, 7 팩터) | 8개 (타겟 y 제외) |
| **결측값** | 없음 | 없음 |

**주요 피처 추정:**
* `f_0`, `f_1`, `f_2`: 가격 관련 비율
* `f_3`: 산업 섹터 또는 업종 분류 (30개 고유 값)
* `f_4`: 거래량 관련
* `f_5`, `f_6`: 기술적 지표

---

## 2. 핵심 도전 과제

### 2.1. 평가 지표와 손실 함수의 불일치

가장 근본적인 문제로, 모델이 최적화하는 목표(RMSE)와 대회가 평가하는 목표(Rank IC)가 다릅니다. RMSE는 예측 오차의 크기를 줄이는 데 집중하지만, Rank IC는 순서의 일치도를 측정하기 때문에 모델 학습 방향과 평가 방향이 어긋날 수 있습니다.

### 2.2. 금융 시계열 데이터의 특성

* **약한 신호(Weak Signal)**: 피처와 타겟(y) 간의 상관계수 절댓값이 최대 0.042에 불과하여, 노이즈 대비 신호가 매우 약해 과적합 위험이 높습니다.
* **Fat-tail 분포**: 타겟 변수 y는 왜도(Skewness) 1.647, 첨도(Kurtosis) 11.558로 극단적인 값(큰 수익 또는 큰 손실)이 빈번하게 나타나는 분포를 보입니다. 이는 일반적인 회귀 모델이 예측하기 어려운 특성입니다.
* **미래 정보 누수(Look-ahead Bias)**: 시계열 데이터를 다룰 때 미래의 정보를 사용하여 현재를 예측하는 실수를 방지해야 합니다. 이를 위해 Purged Cross-Validation과 같은 엄격한 검증 전략이 필수적입니다.

### 2.3. 데이터 관련 문제

* **Cold Start**: 테스트 데이터에만 존재하는 1,526개의 신규 주식에 대한 예측이 필요합니다. 이 주식들은 과거 데이터가 없어 모델이 어려움을 겪을 수 있습니다.
* **타입 불일치**: 범주형 피처 `f_3`가 학습 데이터에서는 object 타입, 테스트 데이터에서는 int64 타입으로 달라 전처리 시 통일이 필요합니다.

---

## 3. 모델 개발 및 성능 진화 과정

프로젝트는 총 4단계(베이스라인 포함)에 걸쳐 점진적으로 성능을 개선했으며, 각 단계는 명확한 전략적 목표를 가집니다.

### 3.1. 베이스라인: RMSE 기반 회귀

* **관련 코드**: `baseline.py`
* **전략**: 기본적인 데이터 전처리 후, LightGBM 회귀 모델을 사용하여 타겟 y의 절대값을 직접 예측. 손실 함수로 RMSE를 사용.
* **주요 피처**: 이동평균, 종목/섹터별 통계량 등
* **공개 점수 (Rank IC)**: 0.05707
* **분석**: 평가 지표와 손실 함수의 불일치로 인해 잠재 성능을 발휘하지 못하는 시작점.

### 3.2. Phase 1: Rank IC 최적화 및 검증 강화

* **관련 코드**: `baseline_rank_ic.py`, `solution_regressor.py`
* **전략**:
  1. **Cross-Sectional Rank 피처 추가**: "특정 날짜 기준, 이 종목의 거래량(f_4) 순위는?"과 같이 각 팩터의 상대적 순위를 새로운 피처로 생성하여 Rank IC 지표에 직접적인 정보를 제공.
  2. **Purged CV 도입**: 학습-검증 데이터 간 시간적 간격(gap)을 두어 미래 정보 누수를 엄격하게 차단하고 모델의 신뢰도를 높임.
* **공개 점수 (Rank IC)**: 0.06246
* **분석**: 피처 엔지니어링과 검증 전략 개선을 통해 소폭의 성능 향상을 이룸. 모델의 안정성이 강화됨.

### 3.3. Phase 2: 핵심 전략 - 타겟 순위 변환 (가장 큰 성능 향상)

* **관련 코드**: `solution_phase2.py`
* **전략**:
  1. **Target Rank Transformation (핵심)**: 기존 타겟 y(수익률)를 각 date별 백분위 순위(0~1 사이의 값)로 변환하여 새로운 타겟으로 사용. 모델은 이제 수익률 값 대신 **"상위 몇 %에 속하는가?"**를 학습.
  2. **5-Seed 앙상블**: 5개의 서로 다른 랜덤 시드로 모델을 학습한 후 예측 결과를 평균하여 모델의 분산을 줄이고 일반화 성능을 향상.
* **공개 점수 (Rank IC)**: 0.08720
* **분석**: "값을 맞추는 문제"에서 **"순위를 맞추는 문제"**로 패러다임을 전환하여 손실 함수(순위에 대한 MSE)와 평가 지표(Rank IC)를 일치시킨 것이 성능 폭등의 핵심 원인.

### 3.4. Phase 3: 최종 모델 - 상호작용 피처 및 앙상블 강화

* **관련 코드**: `solution_phase3.py`
* **전략**:
  1. **Interaction Features 추가**: `f_0 / f_4` (가격 대비 거래량), `f_0 * f_5` (가격 * 기술지표) 등 경제적 의미를 부여한 비선형 조합 피처를 추가하여 모델의 표현력을 높임.
  2. **10-Seed Bagging 앙상블**: 앙상블 규모를 10개 시드로 확장하고, 학습 시 서브샘플링(Subsample Bagging)을 적용하여 과적합을 더욱 강력하게 억제하고 예측 안정성을 극대화.
* **공개 점수 (Rank IC)**: 0.08957
* **분석**: Phase 2의 성공 전략을 기반으로 피처와 앙상블을 정교화하여 추가적인 점수 향상을 이끌어낸 최종 최적화 단계.

---

## 4. 성능 향상 요약 및 분석

### 4.1. 단계별 성능 비교

모델 개발 단계에 따른 성능 향상 추이는 아래와 같습니다.

| 단계 | 제출 파일 | 관련 코드 | 주요 전략 | Leaderboard Score | 개선폭 (vs Baseline) |
|---|---|---|---|---|---|
| Baseline | `submission_baseline.csv` | `baseline.py` | 기본 RMSE 회귀 | 0.05707 | - |
| Phase 1 | `submission_regressor.csv` | `solution_regressor.py` | Rank 피처 + Purged CV | 0.06246 | +9.4% |
| Phase 2 | `submission_phase2.csv` | `solution_phase2.py` | 타겟 순위 변환 + 5-Seed 앙상블 | 0.08720 | +52.8% |
| **Phase 3** | `submission_phase3.csv` | `solution_phase3.py` | **상호작용 피처 + 10-Seed 배깅** | **0.08957** | **+56.9%** |

### 4.2. 주요 성공 요인 분석

* **핵심 동인(Key Driver)**: 가장 큰 성능 향상은 Phase 2의 **타겟 순위 변환**에서 비롯되었습니다. 이는 평가 지표의 본질(순위 비교)을 이해하고 모델의 학습 목표를 그에 맞게 재설계하는 것이 얼마나 중요한지를 입증합니다.
* **안정성 및 추가 이득**: Phase 1의 Purged CV는 모델의 신뢰도를 확보하는 데 기여했으며, Phase 3의 상호작용 피처와 앙상블 강화는 이미 높은 성능을 기록한 모델에서 추가 점수를 확보하는 데 유효한 전략임을 보여주었습니다.

---

## 5. 결론 및 향후 방향

### 5.1. 최종 결론

본 프로젝트는 **"평가 지표에 대한 깊은 이해를 바탕으로 한 전략 설계"**가 머신러닝 문제 해결의 핵심임을 명확히 보여줍니다. 특히 순위 기반 평가 지표가 사용되는 경우, 전통적인 회귀 접근법에서 벗어나 순위 자체를 예측 대상으로 삼는 패러다임 전환이 결정적인 성능 향상을 가져올 수 있습니다. 단계별로 검증된 전략(피처 엔지니어링, 검증, 앙상블)을 체계적으로 적용하여 안정적으로 최고 성능을 달성할 수 있었습니다.

한편, 리더보드 최상위권의 1.0에 가까운 점수는 금융 데이터의 특성상 일반적인 방법으로는 도달이 불가능하며, **데이터 누수(Data Leakage)**나 **비식별화 해제(De-anonymization)**를 활용했을 가능성이 매우 높다고 판단됩니다.

### 5.2. 제안된 개선 방향

현재 모델을 넘어서는 추가적인 성능 향상을 위해 다음 방향을 고려할 수 있습니다.

* **유전 프로그래밍(Genetic Programming)**: gplearn과 같은 라이브러리를 활용하여 인간의 직관을 넘어선 복잡한 수식 기반의 피처를 자동으로 탐색하고 생성합니다.
* **AutoML 활용**: TabNet, AutoGluon과 같은 AutoML 프레임워크를 도입하여 다양한 모델 구조를 탐색하고, 현재의 GBDT 계열 모델과 앙상블하여 모델 다양성을 확보합니다.

---

## 6. 실행 방법 및 환경 설정

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
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost kaggle scipy
```

### 모델 실행

**Phase 3 (Best Model) 실행:**
```bash
python solution_phase3.py
```
실행 후 `submissions/submission_phase3.csv` 파일이 생성됩니다.

---

## 7. 참고 자료

### 대회 및 문서
- [Kaggle Competition](https://www.kaggle.com/competitions/stocks-return-prediction-v-2)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

### 퀀트 트레이딩 참고 자료
- [Quantopian Lectures: Alpha Factors](https://www.quantopian.com/lectures)
- [101 Formulaic Alphas (WorldQuant)](https://arxiv.org/abs/1601.00991)
- [Machine Learning for Trading (Georgia Tech)](https://omscs.gatech.edu/cs-7646-machine-learning-trading)
