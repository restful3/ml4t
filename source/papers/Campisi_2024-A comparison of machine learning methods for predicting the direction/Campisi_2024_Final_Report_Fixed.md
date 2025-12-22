# Campisi et al. (2024) Fixed Replication – Final Report

## 1. 목적 및 실험 개요
- `run_experiments_fixed.sh`는 campisi_2024_replication_fixed.py를 7·15·30·60일 예측 구간별로 실행하여 각 모델의 성능 리포트를 생성했다. 결과물은 `campisi_2024_results_fixed_*d_20251220_*.md`에 저장되어 있으며 모든 리포트는 동일한 데이터 구간(2011-01-03~2022-07-29)과 755개의 walk-forward 반복을 사용한다.
- 고정 버전 스크립트에서는 **표준화·Feature Selection을 CV 루프 내부로 이동**시키고 Diebold-Mariano(DM) 검정을 통합하여 데이터 누수와 통계 검증 공백을 해소했다 (`FIXES_APPLIED.md:17-138`).
- 비교 대상인 Campisi et al. (2024)은 30일 예측 1개 구간만 분석했으며, Feature Selection 이후 Bagging/Random Forest가 최고 성능(정확도 0.828/0.824)을 보인다고 보고했다 (`Campisi_2024-A comparison of machine learning methods for predicting the direction.md:403-425`).

## 2. 예측 구간별 재현 결과 요약 (Feature Selection 적용 후)
| 예측 기간 | 최고 모델 | Accuracy | AUC | F-measure | 근거 |
|-----------|-----------|---------:|----:|----------:|------|
| 7일 | Lasso Regression | 0.6238 | 0.5420 | 0.7676 | `campisi_2024_results_fixed_7d_20251220_133545.md:93-100` |
| 15일 | Linear/Ridge/Lasso Regression | 0.6980 | 0.5506 | 0.8219 | `campisi_2024_results_fixed_15d_20251220_145134.md:93-101` |
| 30일 | Lasso Regression | 0.7245 | 0.5369 | 0.8402 | `campisi_2024_results_fixed_30d_20251220_160519.md:93-100` |
| 60일 | Random Forest (Classification) | 0.6675 | 0.5476 | 0.7971 | `campisi_2024_results_fixed_60d_20251220_171822.md:83-100` |

**관찰**
- 모든 구간에서 회귀 프레임(이진 방향으로 변환) 혹은 선형 모델이 가장 안정적이며, 장기 구간(60일)에서만 분류용 Random Forest가 근소하게 우위다.
- AUC는 0.53~0.62 범위로, 정확도 대비 확률 보정이 미흡함을 시사한다.

## 3. Campisi et al. (2024) 30일 결과와의 비교
| 항목 | 논문 보고값 | Fixed 재현 | 차이 |
|------|------------|-----------|------|
| 최고 Accuracy (Classification) | Bagging 0.8275 (`…direction.md:407-414`) | Logistic/LDA 0.6675 (`campisi_2024_results_fixed_30d_20251220_160519.md:83-89`) | -16.0%p |
| 최고 Accuracy (Regression) | Random Forest 0.8003 (`…direction.md:421-425`) | Lasso 0.7245 (`campisi_2024_results_fixed_30d_20251220_160519.md:93-100`) | -7.6%p |
| 최고 AUC (Classification) | Random Forest 0.8495 | Random Forest 0.6603 | -0.1892 |
| 최고 AUC (Regression) | Random Forest 0.8412 | Bagging 0.6598 | -0.1814 |

### 차이 해석
1. **데이터 누수 제거의 영향**: 원본 구현은 전체 데이터로 스케일링·Feature Selection을 수행하여 각 fold가 미래 정보를 포함하고 있었다. 파이프라인 기반 수정(`FIXES_APPLIED.md:17-93`) 이후 정확도는 현실적인 0.66~0.72 수준으로 하락했다.
2. **통계 검증 결과**: DM 검정은 Logistic/LDA가 모든 앙상블 분류 모델보다 유의하게 우수함을 보여주지만(`campisi_2024_results_fixed_30d_20251220_160519.md:104-118`), 이는 정확도 1~2%p 수준의 미세한 우위이며 논문에서 보고된 “앙상블 절대 우위” 결론과 상충한다.
3. **평가지표 일관성**: 논문은 Accuracy와 AUC가 함께 상승한다고 보고했으나, 재현 결과에서는 Accuracy 개선 시 AUC가 오히려 낮아진다. 이는 확률 추정치가 과신된다는 뜻으로, 논문에서 사용된 확률 임계값 조정 혹은 class imbalance 보정이 코드베이스에 반영되지 않았을 가능성이 있다.

## 4. Horizon별 인사이트
- **7일**: 모든 모델이 0.62 이하의 Accuracy를 기록하여 단기 변동성을 설명하지 못했다. Logistic/LDA 대비 Lasso가 통계적으로 우수하며(모든 모델 쌍 DM 유의, `campisi_2024_results_fixed_7d_20251220_133545.md:108-135`), 단기 예측에는 규제가 강한 회귀가 필요함을 시사한다.
- **15일**: 선형 모델이 0.70 근처 정확도를 달성하는 첫 구간이다. Random Forest/Bagging과의 성능 차이가 DM 검정에서 4~6σ 수준으로 확인되어(`campisi_2024_results_fixed_15d_20251220_145134.md:108-134`), 단순 모델로도 충분히 설명 가능한 영역임을 보여준다.
- **30일**: 회귀 모델 평균 정확도가 분류 모델보다 높아(`campisi_2024_results_fixed_30d_20251220_160519.md:164-167`), 논문과 반대로 **분류 모델을 고집할 이유가 없다**. Feature Selection은 오히려 평균 정확도를 1.4%p 낮춘다(`campisi_2024_results_fixed_30d_20251220_160-163`).
- **60일**: Random Forest 분류기가 0.6675로 최상위를 차지하지만, AUC 0.5476으로 신뢰도는 낮다(`campisi_2024_results_fixed_60d_20251220_171822.md:83-100`). 회귀·분류 모두 DM 검정에서 다수의 유의 차이를 보이며(`campisi_2024_results_fixed_60d_20251220_171822.md:104-129`), 예측기 간 일관성이 떨어진다.

## 5. Feature Selection과 통계 검증
- Feature Selection 효과는 구간마다 상이하다. 7·15일에서는 전체 평균 정확도가 1%p 내외 상승하지만, 30·60일에서는 각각 1.36%p, 0.16%p 감소한다(`campisi_2024_results_fixed_7d_20251220_133545.md:157-164`, `…15d…:156-163`, `…30d…:160-167`, `…60d…:151-159`).
- DM 검정 도입으로 모든 구간에서 “선형 vs 앙상블” 간 성능 격차가 3~10σ 수준임을 확인할 수 있었고(`campisi_2024_results_fixed_*d_20251220_*.md:104-135`), 논문의 핵심 타이틀인 “모델 비교”가 통계적으로 타당하게 수행되었다.

## 6. 결론 및 추천 사항
1. **논문 결과는 데이터 누수에 기인한 과대평가 가능성이 높다.** 고정 버전에서는 30일 구간 정확도가 0.7245에 그쳐, 보고치(0.828) 대비 약 10%p 낮다.
2. **중·장기(≥15일) 예측에는 선형/라쏘 회귀가 충분히 경쟁력 있다.** 복잡한 앙상블은 계산 비용 대비 추가 성능을 제공하지 못했다.
3. **확률 캘리브레이션 개선이 필요**: 낮은 AUC는 포트폴리오 크기 조정 같은 응용에서 문제이므로 Platt scaling이나 isotonic calibration을 차기 작업으로 권장한다.
4. **Feature Selection은 구간별로 조건부 적용**: 장기 구간에서는 전체 피처를 사용하는 것이 안전하며, 단기 구간만 규제 기반 선택을 유지할 것을 권장한다.

---
*작성일: 2025-12-20*
