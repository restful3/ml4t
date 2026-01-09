# 주식 수익률 예측 개선 보고서 (Stocks Return Prediction V2)

## 1. 개요 및 분석 결과
- **현재 성능**: 0.06246 (Regression Baseline) / -0.0792 (LambdaRank)
- **1등 성능 (1.0)**: 거의 완벽한 예측. 이는 금융 데이터에서 **불가능한 수치**입니다.
    - **결론**: 리더보드 상위권은 **Data Leakage(데이터 누수)** 또는 **De-anonymization(비식별화 해제)**를 이용했을 가능성이 99% 이상입니다.

## 2. Phase 1: 기반 구축 (Regression + Purged CV)
`solution_regressor.py`에서 LambdaRank의 실패를 교훈 삼아 안정적인 Regression 모델을 구축했습니다.
- **Rank IC**: 0.0944 (CV)
- **주요 전략**: Sector Neutralization, Cross-Sectional Ranking, Purged CV
- **제출 점수**: 0.06246 (양수 복구 성공)

## 3. Phase 2: 성능 고도화 (Target Rank + Ensemble)
`solution_phase2.py`에서 성능을 극대화하기 위해 다음 전략을 추가했습니다.

### 3.1 전략 상세
1.  **Target Rank Transformation**:
    - `y` (수익률) 값 자체를 학습하는 대신, 날짜별 **백분위 순위(Rank Pct)**를 학습하도록 타겟을 변환했습니다.
    - **이유**: 평가 지표인 Rank IC는 값의 크기가 아닌 '순서'만 봅니다. 타겟을 순위로 변환하면 Loss Function(RMSE)이 Rank IC와 더 잘 일치하게 됩니다.
2.  **5-Seed Ensemble**:
    - 5개의 다른 Random Seed로 모델을 학습하여 평균을 냈습니다.
    - 이는 모델의 분산을 줄이고 일반화 성능을 높입니다.

### 3.3 Phase 2: 성능 고도화 (Target Rank + 5-Seed Ensemble)
- **전략**: `Rank(y)` 타겟 변환 + 5-Seed 평균 앙상블
- **결과**:
    - **Global Mean CV Rank IC**: **0.1127** (Phase 1 대비 +19.4%)
    - 제출 점수: **0.08720** (리더보드 점수 대폭 상승)

### 3.4 Phase 3: 극한 최적화 (Interactions + 10-Seed Bagging)
`solution_phase3.py`에서 추가적인 성능 향상을 위해 피처와 앙상블을 강화했습니다.
1.  **Interaction Features**: `f_0 * f_5` (Price * Tech), `f_0 / f_4` (Price / Volume) 등 비선형 상호작용 피처 추가.
2.  **10-Seed Bagging Ensemble**: 시드를 10개로 늘리고, 학습 시 Subsample Bagging을 적용하여 과적합을 더 강력하게 억제. 
- **결과**:
    - **Global Mean CV Rank IC**: **0.1133** (안정성 강화)
    - **제출 점수 (Leaderboard)**: **0.08957** (Phase 2 대비 +0.00237 상승)

## 4. 최종 성과 요약 (Performance Summary)

계단식으로 성능이 향상되었습니다.

| 단계 | 전략 요약 | Leaderboard Score | 개선폭 (vs Baseline) |
|---|---|---|---|
| Baseline | 초기 제공 코드 | 0.05707 | - |
| Phase 1 | Regressor + Purged CV + Sector Neutral | 0.06246 | +9.4% |
| **Phase 2** | **Target Rank Transform + 5-Seed Ensemble** | **0.08720** | **+52.8%** (Key Driver) |
| **Phase 3** | **Interactions + 10-Seed Bagging** | **0.08957** | **+56.9%** (Final Best) |

## 5. 최종 제언

1.  **성능 요인 분석**:
    - 가장 큰 점수 점프는 **Phase 2 (Target Rank Transformation)**에서 발생했습니다. 이는 평가 지표(Spearman Correlation)와 손실 함수를 일치시키는 것이 얼마나 중요한지 보여줍니다.
    - Phase 3의 소폭 상승은 피처 엔지니어링의 한계 효용(Diminishing Returns)을 보여주지만, 앙상블 확장이 점수를 쥐어짜는(Squeezing) 데 유효함을 입증했습니다.
2.  **추가 개선 방향**:
    - 0.09 이상의 점수를 위해서는 단순 수식 기반의 Interaction을 넘어선 **Genetic Programming**(gplearn 등)이 필요할 수 있습니다.
    - 또는 **AutoML**(TabNet, AutoGluon)을 활용한 모델 다양성 확보가 도움이 될 것입니다.

## 6. 실행 방법

최종 모델 실행 코드:

```bash
# 가상환경 활성화 후 실행
source .venv/bin/activate
python solution_phase3.py
# 생성된 submissions/submission_phase3.csv 제출 (Score: 0.08957)
```
